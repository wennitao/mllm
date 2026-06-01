// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device latency + accuracy driver for the QUANTIZED (w4a16-LPBQ) full-attention
// decode-step graph (compile_attn_decode). Allocates the per-step inputs, dispatches
// the single "model.0.s2048" graph, reports avg latency, and (with --ref) validates
// y / k_new / v_new against the int4 reference bundle from export_attn_decode.py.
//
//   ./mllm-qwen3-aot-attn-decode-run -m qwen3-attn-decode.bin --ctx 256 --ref attn-l3-ref.mllm
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-attn-decode.bin");
  auto& H_arg = Argparse::add<int>("--heads").def(8);
  auto& kv_arg = Argparse::add<int>("--kv").def(2);
  auto& d_arg = Argparse::add<int>("--dim").def(256);
  auto& rot_arg = Argparse::add<int>("--rot").def(64);
  auto& ctx_arg = Argparse::add<int>("--ctx").def(256);
  auto& reps_arg = Argparse::add<int>("--reps").def(50);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("reference .mllm -> validate accuracy").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = H_arg.get(), KV = kv_arg.get(), D = d_arg.get(), rot = rot_arg.get(), ctx = ctx_arg.get();
  const int hidden = 2048, P = ctx - 1;
  const int reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());

  auto mkf16 = [&](std::vector<int> shp, float v) {
    auto t = Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc();
    for (int64_t i = 0; i < t.numel(); ++i) t.ptr<__fp16>()[i] = (__fp16)v;
    return t;
  };

  // Inputs in the exact order the compiled graph expects.
  std::vector<Tensor> ins;
  ins.push_back(mkf16({1, hidden}, 0.1f));        // x
  ins.push_back(mkf16({1, 1, rot}, 1.0f));        // sin
  ins.push_back(mkf16({1, 1, rot}, 1.0f));        // cos
  ins.push_back(mkf16({1, 1, D}, 1.0f));          // q_norm_w
  ins.push_back(mkf16({1, 1, D}, 1.0f));          // k_norm_w
  ins.push_back(mkf16({1, 1, 1}, 1e-6f));         // eps
  ins.push_back(mkf16({1, KV, D, P}, 0.0f));      // past_k
  ins.push_back(mkf16({1, KV, P, D}, 0.0f));      // past_v
  ins.push_back(mkf16({1, 1, 1, ctx}, 0.0f));     // mask

  // Outputs: y[1,hidden], k_new[1,KV,D,1], v_new[1,KV,1,D].
  std::vector<Tensor> outs;
  outs.push_back(Tensor::empty({1, hidden}, mllm::kFloat16, mllm::kQNN).alloc());
  outs.push_back(Tensor::empty({1, KV, D, 1}, mllm::kFloat16, mllm::kQNN).alloc());
  outs.push_back(Tensor::empty({1, KV, 1, D}, mllm::kFloat16, mllm::kQNN).alloc());

  // ---- accuracy validation: overwrite inputs from the reference .mllm ----
  const bool validate = !ref_arg.get().empty();
  Tensor exp_y, exp_k, exp_v;
  if (validate) {
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto f = [&](const std::string& n) { return ref->pull(n); };
    auto setf16 = [&](Tensor& dst, const float* src, int64_t n) {
      for (int64_t i = 0; i < n; ++i) dst.ptr<__fp16>()[i] = (__fp16)src[i];
    };
    setf16(ins[0], f("x").ptr<float>(), hidden);
    setf16(ins[1], f("sin").ptr<float>(), rot);
    setf16(ins[2], f("cos").ptr<float>(), rot);
    setf16(ins[3], f("q_norm_w").ptr<float>(), D);
    setf16(ins[4], f("k_norm_w").ptr<float>(), D);
    setf16(ins[5], f("eps").ptr<float>(), 1);
    setf16(ins[6], f("past_k").ptr<float>(), (int64_t)KV * D * P);
    setf16(ins[7], f("past_v").ptr<float>(), (int64_t)KV * P * D);
    setf16(ins[8], f("mask").ptr<float>(), ctx);
    exp_y = f("exp_y");
    exp_k = f("exp_k_new");
    exp_v = f("exp_v_new");
  }

  QnnAOTModule g("model.0.s" + std::to_string(hidden));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();  // warm

  if (validate) {
    auto relerr = [&](const Tensor& got, const Tensor& exp, int64_t n) {
      double e = 0, r = 0;
      for (int64_t i = 0; i < n; ++i) {
        double gv = (double)got.ptr<__fp16>()[i], ev = exp.ptr<float>()[i];
        e = std::max(e, std::fabs(gv - ev));
        r = std::max(r, std::fabs(ev));
      }
      return std::pair<double, double>{e, r};
    };
    auto [ey, ry] = relerr(outs[0], exp_y, hidden);
    auto [ek, rk] = relerr(outs[1], exp_k, (int64_t)KV * D);
    auto [ev, rv] = relerr(outs[2], exp_v, (int64_t)KV * D);
    double yr = 100.0 * ey / (ry + 1e-9), kr = 100.0 * ek / (rk + 1e-9), vr = 100.0 * ev / (rv + 1e-9);
    fmt::print("[attn decode VALIDATE] graph vs int4 ref:\n");
    fmt::print("  y    : max|err|={:.5f}  |y|max={:.4f}  rel={:.2f}pct\n", ey, ry, yr);
    fmt::print("  k_new: max|err|={:.5f}  |k|max={:.4f}  rel={:.2f}pct\n", ek, rk, kr);
    fmt::print("  v_new: max|err|={:.5f}  |v|max={:.4f}  rel={:.2f}pct\n", ev, rv, vr);
    fmt::print("  -> {} (fp16 + a16 + int4-weight precision; rel<5pct)\n",
               (yr < 5.0 && kr < 5.0 && vr < 5.0) ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[attn decode LPBQ] H={} KV={} D={} ctx={} avg = {:.4f} ms ({} reps)\n", H, KV, D, ctx,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
});
