// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device validator/latency driver for the LFM2 GQA attention decode graph
// (compile_lfm2_attn_decode). Loads the ref (x, sin, cos, q/k_norm_w, eps, past_k/v, mask,
// exp_y/exp_k_new/exp_v_new from export_attn_decode.py), dispatches, validates vs golden.
//
//   ./mllm-lfm2-aot-attn-decode-run -m lfm2-attn.bin --heads 32 --kv 8 --dim 64 --hidden 2048 \
//       --ctx 256 --ref lfm2_attn-ref.mllm
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("lfm2-attn-decode.bin");
  auto& H_arg = Argparse::add<int>("--heads").def(32);
  auto& kv_arg = Argparse::add<int>("--kv").def(8);
  auto& d_arg = Argparse::add<int>("--dim").def(64);
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& ctx_arg = Argparse::add<int>("--ctx").def(256);
  auto& reps_arg = Argparse::add<int>("--reps").def(50);
  auto& ref_arg = Argparse::add<std::string>("--ref").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = H_arg.get(), KV = kv_arg.get(), D = d_arg.get(), hidden = hidden_arg.get();
  const int ctx = ctx_arg.get(), rot = D, P = ctx - 1, reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());
  auto mk = [&](std::vector<int> shp) { return Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc(); };
  auto x = mk({1, hidden}), sin = mk({1, 1, rot}), cos = mk({1, 1, rot});
  auto q_norm_w = mk({1, 1, D}), k_norm_w = mk({1, 1, D}), eps = mk({1, 1, 1});
  auto past_k = mk({1, KV, D, P}), past_v = mk({1, KV, P, D}), mask = mk({1, 1, 1, ctx});
  auto y = mk({1, hidden}), k_new = mk({1, KV, D, 1}), v_new = mk({1, KV, 1, D});
  std::vector<Tensor> ins{x, sin, cos, q_norm_w, k_norm_w, eps, past_k, past_v, mask}, outs{y, k_new, v_new};

  const bool validate = !ref_arg.get().empty();
  Tensor exp_y;
  if (validate) {
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto setf16 = [&](Tensor& dst, const Tensor& src) {
      for (int64_t i = 0; i < dst.numel(); ++i) dst.ptr<__fp16>()[i] = (__fp16)src.ptr<float>()[i];
    };
    setf16(x, ref->pull("x"));
    setf16(sin, ref->pull("sin"));
    setf16(cos, ref->pull("cos"));
    setf16(q_norm_w, ref->pull("q_norm_w"));
    setf16(k_norm_w, ref->pull("k_norm_w"));
    setf16(eps, ref->pull("eps"));
    setf16(past_k, ref->pull("past_k"));
    setf16(past_v, ref->pull("past_v"));
    setf16(mask, ref->pull("mask"));
    exp_y = ref->pull("exp_y");
  }

  QnnAOTModule g("model.0.s" + std::to_string(hidden));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();

  if (validate) {
    double e = 0, range = 0;
    for (int i = 0; i < hidden; ++i) {
      float yi = (float)y.ptr<__fp16>()[i], ei = exp_y.ptr<float>()[i];
      e = std::max(e, (double)std::fabs(yi - ei));
      range = std::max(range, (double)std::fabs(ei));
    }
    double rel = 100.0 * e / (range + 1e-9);
    fmt::print("[attn VALIDATE] max|err|={:.5f} |y|max={:.4f} rel={:.2f}pct -> {}\n", e, range, rel,
               rel < 5.0 ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[attn LPBQ] H={} KV={} D={} ctx={} avg = {:.4f} ms ({} reps)\n", H, KV, D, ctx,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
  mllm::shutdownContext();
});
