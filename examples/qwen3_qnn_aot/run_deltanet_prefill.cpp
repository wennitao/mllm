// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device driver + accuracy validator for the QUANTIZED (w4a16-LPBQ) GatedDeltaNet
// BLOCK-PREFILL graph (compile_deltanet_prefill). Sets the block inputs from the
// reference .mllm (export_deltanet_prefill.py), builds the constant chunk masks
// (Ltri/strict/eye), dispatches "model.0.s<H*Dv>", and validates y[B,hidden] + the
// final state Sp[H,Dk,Dv] against the sequential fp32 reference. Also reports latency.
//
//   ./mllm-qwen3-aot-deltanet-prefill-run -m qwen3-deltanet-prefill-s128.bin \
//        --seq 128 --chunk 64 --heads 16 --ref deltanet-prefill-l0-ref.mllm
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-deltanet-prefill-s128.bin");
  auto& seq_arg = Argparse::add<int>("--seq").def(128);
  auto& chunk_arg = Argparse::add<int>("--chunk").def(64);
  auto& H_arg = Argparse::add<int>("--heads").def(16);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  auto& reps_arg = Argparse::add<int>("--reps").def(20);
  auto& nreal_arg = Argparse::add<int>("--nreal").help("validate only the first N (real, non-padded) tokens").def(0);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("reference .mllm -> validate accuracy").def("");
  auto& dbg_arg = Argparse::add<bool>("--dbg").help("bin compiled with --dbg: bind 3 extra head-0 outputs").def(false);
  auto& emit_cat_arg = Argparse::add<bool>("--emit_cat").help("bin compiled with --emit_cat: out[0]=pre-out_proj gated; validate vs exp_gated_full").def(false);
  auto& emit_core_arg = Argparse::add<bool>("--emit_core").help("bin compiled with --emit_core: out[0]=pre-gated-norm core; validate vs exp_core_full").def(false);
  auto& probe_dv_arg = Argparse::add<bool>("--probe_dv").help("out[0]=head-0 gated [B,Dv]; validate vs exp_gated_h0").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get(), hidden = H * Dv;
  const int B = seq_arg.get(), C = chunk_arg.get();
  const int kd = H * Dk, vd = H * Dv, reps = reps_arg.get();
  const bool probe_dv = probe_dv_arg.get();

  mllm::initQnnBackend(model_path.get());

  auto f16 = [&](std::vector<int> s) { return Tensor::empty(s, mllm::kFloat16, mllm::kQNN).alloc(); };
  auto f32 = [&](std::vector<int> s) { return Tensor::empty(s, mllm::kFloat32, mllm::kQNN).alloc(); };
  auto fill32 = [&](Tensor& t, float v) { for (int64_t i = 0; i < t.numel(); ++i) t.ptr<float>()[i] = v; };

  // ---- inputs in graph order ----
  std::vector<Tensor> ins;
  auto x = f16({B, hidden}); ins.push_back(x);
  auto eps = f32({1, 1, 1}); ins.push_back(eps);
  auto qscale = f32({1, 1, 1}); ins.push_back(qscale);
  auto norm_w = f32({1, 1, Dv}); ins.push_back(norm_w);
  auto A_log = f32({1, 1, H}); ins.push_back(A_log);
  auto dt_bias = f32({1, 1, H}); ins.push_back(dt_bias);
  auto Wa = f32({1, hidden, H}); ins.push_back(Wa);
  auto Wb = f32({1, hidden, H}); ins.push_back(Wb);
  auto cw_q = f32({1, 4, kd}); ins.push_back(cw_q);
  auto cw_k = f32({1, 4, kd}); ins.push_back(cw_k);
  auto cw_v = f32({1, 4, vd}); ins.push_back(cw_v);
  auto cs_q = f32({1, 3, kd}); ins.push_back(cs_q);
  auto cs_k = f32({1, 3, kd}); ins.push_back(cs_k);
  auto cs_v = f32({1, 3, vd}); ins.push_back(cs_v);
  auto S0 = f32({H, Dk, Dv}); ins.push_back(S0);
  auto Ltri = f32({1, 1, C, C}); ins.push_back(Ltri);
  auto strict = f32({1, 1, C, C}); ins.push_back(strict);
  auto eye = f32({1, 1, C, C}); ins.push_back(eye);

  // constant chunk masks
  for (int i = 0; i < C; ++i)
    for (int j = 0; j < C; ++j) {
      Ltri.ptr<float>()[i * C + j] = (j <= i) ? 1.f : 0.f;
      strict.ptr<float>()[i * C + j] = (j < i) ? 1.f : 0.f;
      eye.ptr<float>()[i * C + j] = (i == j) ? 1.f : 0.f;
    }

  // ---- outputs: y[B,hidden] (0), Sp_h[1,Dk,Dv] (1..H) ----
  std::vector<Tensor> outs;
  auto y = probe_dv ? f16({B, Dv}) : f16({B, hidden}); outs.push_back(y);
  for (int h = 0; h < H; ++h) outs.push_back(f32({1, Dk, Dv}));
  if (dbg_arg.get()) {  // qc[1,C,Dk], attn[1,C,C], v_new[1,C,Dv], U[1,C,Dv]
    outs.push_back(f32({1, C, Dk})); outs.push_back(f32({1, C, C}));
    outs.push_back(f32({1, C, Dv})); outs.push_back(f32({1, C, Dv}));
  }

  // ---- fill from reference, or defaults ----
  const bool validate = !ref_arg.get().empty();
  Tensor exp_y, exp_Sp;
  if (validate) {
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto P = [&](const std::string& n) { return ref->pull(n); };
    auto cpy32 = [&](Tensor& dst, Tensor src) { for (int64_t i = 0; i < dst.numel(); ++i) dst.ptr<float>()[i] = src.ptr<float>()[i]; };
    auto cpy16 = [&](Tensor& dst, Tensor src) { for (int64_t i = 0; i < dst.numel(); ++i) dst.ptr<__fp16>()[i] = (__fp16)src.ptr<float>()[i]; };
    cpy16(x, P("x"));
    cpy32(eps, P("eps")); cpy32(qscale, P("qscale")); cpy32(norm_w, P("norm_w"));
    cpy32(A_log, P("A_log")); cpy32(dt_bias, P("dt_bias")); cpy32(Wa, P("Wa")); cpy32(Wb, P("Wb"));
    cpy32(cw_q, P("cw_q")); cpy32(cw_k, P("cw_k")); cpy32(cw_v, P("cw_v"));
    cpy32(cs_q, P("cs_q")); cpy32(cs_k, P("cs_k")); cpy32(cs_v, P("cs_v"));
    cpy32(S0, P("S0"));
    exp_y = (probe_dv && std::getenv("PROBE_QN")) ? P("exp_qn_h0")
            : probe_dv ? P("exp_gated_h0")
            : emit_core_arg.get() ? P("exp_core_full")
            : (emit_cat_arg.get() ? P("exp_gated_full") : P("exp_y"));
    exp_Sp = P("exp_Sp");
  } else {
    fill32(eps, 1e-6f); fill32(qscale, 1.f / std::sqrt((float)Dk)); fill32(norm_w, 1.f);
    for (int64_t i = 0; i < x.numel(); ++i) x.ptr<__fp16>()[i] = (__fp16)0.1f;
  }

  QnnAOTModule g("model.0.s" + std::to_string(hidden));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();  // warm

  if (std::getenv("DBG_OUTS")) {
    const char* nm[] = {"x","eps","qscale","norm_w","A_log","dt_bias","Wa","Wb","cw_q","cw_k","cw_v","cs_q","cs_k","cs_v","S0","Ltri","strict","eye"};
    for (size_t o = 0; o < ins.size(); ++o) {
      double mx = 0;
      for (int64_t i = 0; i < ins[o].numel(); ++i) {
        float v = (ins[o].dtype() == mllm::kFloat16) ? (float)ins[o].ptr<__fp16>()[i] : ins[o].ptr<float>()[i];
        mx = std::max(mx, (double)std::fabs(v));
      }
      fmt::print("  in[{}] {} |max|={:.5f}\n", o, nm[o], mx);
    }
  }
  // debug: magnitude of every output slot (locate where data lands)
  if (std::getenv("DBG_OUTS")) {
    for (size_t o = 0; o < outs.size(); ++o) {
      double mx = 0, lo = 1e30, hi = -1e30; bool bad = false;
      for (int64_t i = 0; i < outs[o].numel(); ++i) {
        float v = (outs[o].dtype() == mllm::kFloat16) ? (float)outs[o].ptr<__fp16>()[i] : outs[o].ptr<float>()[i];
        if (!std::isfinite(v)) { bad = true; continue; }
        mx = std::max(mx, (double)std::fabs(v)); lo = std::min(lo, (double)v); hi = std::max(hi, (double)v);
      }
      fmt::print("  out[{}] dtype={} numel={} {} min={:.4f} max={:.4f} |max|={:.5f}\n",
                 o, (int)outs[o].dtype(), outs[o].numel(), bad ? "*NONFINITE*" : "", lo, hi, mx);
    }
  }

  if (validate) {
    double ey = 0, yr0 = 0;
    const int row = probe_dv ? Dv : hidden;
    const int nrows = (nreal_arg.get() > 0 && nreal_arg.get() < B) ? nreal_arg.get() : B;
    const int ylen = nrows * row;
    for (int i = 0; i < ylen; ++i) {
      ey = std::max(ey, (double)std::fabs((float)y.ptr<__fp16>()[i] - exp_y.ptr<float>()[i]));
      yr0 = std::max(yr0, (double)std::fabs(exp_y.ptr<float>()[i]));
    }
    double es = 0, sr0 = 0;
    const int shd = Dk * Dv;
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < shd; ++i)
        es = std::max(es, (double)std::fabs(outs[1 + h].ptr<float>()[i] - exp_Sp.ptr<float>()[(size_t)h * shd + i]));
    for (int64_t i = 0; i < exp_Sp.numel(); ++i) sr0 = std::max(sr0, (double)std::fabs(exp_Sp.ptr<float>()[i]));
    double yrel = 100.0 * ey / (yr0 + 1e-9), srel = 100.0 * es / (sr0 + 1e-9);
    fmt::print("[deltanet prefill VALIDATE] B={} C={} graph vs int4 ref:\n", B, C);
    fmt::print("  y    : max|err|={:.5f}  |y|max={:.4f}  rel={:.2f}pct\n", ey, yr0, yrel);
    fmt::print("  state: max|err|={:.5f}  |S|max={:.4f}  rel={:.2f}pct\n", es, sr0, srel);
    fmt::print("  -> {} (rel<5pct)\n", (yrel < 5.0 && srel < 5.0) ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[deltanet prefill LPBQ] B={} C={} H={} avg = {:.4f} ms ({} reps)\n", B, C, H,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
});
