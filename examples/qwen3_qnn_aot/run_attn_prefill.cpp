// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device driver + accuracy validator for the QUANTIZED full-attention BLOCK-PREFILL
// graph (compile_attn_prefill). Sets the block inputs from the reference .mllm
// (export_attn_prefill.py), builds the additive causal mask, dispatches "model.0.s2048",
// and validates y[B,hidden] vs the fp32 reference. Reports latency.
//
//   ./mllm-qwen3-aot-attn-prefill-run -m qwen3-attn-prefill-s128.bin --seq 128 \
//        --ref attn-prefill-l3-ref.mllm --nreal 35
//
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::aot::QnnAOTModule;

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-attn-prefill-s128.bin");
  auto& seq_arg = Argparse::add<int>("--seq").def(128);
  auto& H_arg = Argparse::add<int>("--heads").def(8);
  auto& kv_arg = Argparse::add<int>("--kv").def(2);
  auto& d_arg = Argparse::add<int>("--dim").def(256);
  auto& rot_arg = Argparse::add<int>("--rot").def(64);
  auto& reps_arg = Argparse::add<int>("--reps").def(20);
  auto& nreal_arg = Argparse::add<int>("--nreal").help("validate only the first N (real) tokens").def(0);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("reference .mllm -> validate").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = H_arg.get(), KV = kv_arg.get(), D = d_arg.get(), rot = rot_arg.get(), B = seq_arg.get();
  const int hidden = 2048, reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());
  auto f16 = [&](std::vector<int> s) { return Tensor::empty(s, mllm::kFloat16, mllm::kQNN).alloc(); };
  auto f32 = [&](std::vector<int> s) { return Tensor::empty(s, mllm::kFloat32, mllm::kQNN).alloc(); };

  std::vector<Tensor> ins;
  auto x = f16({B, hidden}); ins.push_back(x);
  auto sin = f16({1, B, rot}); ins.push_back(sin);
  auto cos = f16({1, B, rot}); ins.push_back(cos);
  auto qn = f32({1, 1, D}); ins.push_back(qn);
  auto kn = f32({1, 1, D}); ins.push_back(kn);
  auto eps = f32({1, 1, 1}); ins.push_back(eps);
  auto cmask = f16({1, 1, B, B}); ins.push_back(cmask);

  // additive causal mask: 0 if j<=i else -50000
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < B; ++j) cmask.ptr<__fp16>()[i * B + j] = (j <= i) ? (__fp16)0.f : (__fp16)(-50000.f);

  std::vector<Tensor> outs;
  auto y = f16({B, hidden}); outs.push_back(y);
  outs.push_back(f16({1, KV, D, B}));   // k_all
  outs.push_back(f16({1, KV, B, D}));   // v_all

  const bool validate = !ref_arg.get().empty();
  Tensor exp_y;
  if (validate) {
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto P = [&](const std::string& n) { return ref->pull(n); };
    auto cp16 = [&](Tensor& d, Tensor s) { for (int64_t i = 0; i < d.numel(); ++i) d.ptr<__fp16>()[i] = (__fp16)s.ptr<float>()[i]; };
    auto cp32 = [&](Tensor& d, Tensor s) { for (int64_t i = 0; i < d.numel(); ++i) d.ptr<float>()[i] = s.ptr<float>()[i]; };
    cp16(x, P("x")); cp16(sin, P("sin")); cp16(cos, P("cos"));
    cp32(qn, P("q_norm_w")); cp32(kn, P("k_norm_w")); cp32(eps, P("eps"));
    exp_y = P("exp_y");
  } else {
    for (int64_t i = 0; i < x.numel(); ++i) x.ptr<__fp16>()[i] = (__fp16)0.05f;
    eps.ptr<float>()[0] = 1e-6f;
  }

  QnnAOTModule g("model.0.s" + std::to_string(hidden));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();

  if (validate) {
    const int nrows = (nreal_arg.get() > 0 && nreal_arg.get() < B) ? nreal_arg.get() : B;
    double ey = 0, yr = 0;
    for (int i = 0; i < nrows * hidden; ++i) {
      ey = std::max(ey, (double)std::fabs((float)y.ptr<__fp16>()[i] - exp_y.ptr<float>()[i]));
      yr = std::max(yr, (double)std::fabs(exp_y.ptr<float>()[i]));
    }
    fmt::print("[attn prefill VALIDATE] B={} (first {} real tokens):\n", B, nrows);
    fmt::print("  y: max|err|={:.5f}  |y|max={:.4f}  rel={:.2f}pct  -> {}\n",
               ey, yr, 100.0 * ey / (yr + 1e-9), (100.0 * ey / (yr + 1e-9) < 5.0) ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[attn prefill LPBQ] B={} H={} avg = {:.4f} ms ({} reps)\n", B, H,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
});
