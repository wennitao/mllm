// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device latency driver for the QUANTIZED (w4a16-LPBQ) GatedDeltaNet decode-step
// graph (compile_deltanet_decode). Allocates the per-head inputs, dispatches the
// single "model.0.s<H*Dv>" graph, and reports avg latency — the faithful per-token
// deltanet-layer cost on the HTP (LPBQ projections + fp16 recurrence [+ gated norm
// + LPBQ out_proj in full mode]).
//
// Weights are synthetic-but-non-degenerate, so the number is representative of the
// real LPBQ memory/compute cost; accuracy needs a real PTQ checkpoint (TODO).
//
//   ./mllm-qwen3-aot-deltanet-decode-run -m qwen3-deltanet-decode-recur.bin --mode recur --heads 16
//   ./mllm-qwen3-aot-deltanet-decode-run -m qwen3-deltanet-decode-full.bin  --mode full  --heads 16
//
#include <chrono>
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-deltanet-decode-recur.bin");
  auto& mode_arg = Argparse::add<std::string>("--mode").help("recur | full").def("recur");
  auto& H_arg = Argparse::add<int>("--heads").def(16);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  auto& reps_arg = Argparse::add<int>("--reps").def(50);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("reference .mllm (from export_deltanet_decode.py) -> validate accuracy").def("");
  auto& conv_arg = Argparse::add<bool>("--conv1d").help("bin includes depthwise causal conv1d (match the compiled bin)").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get(), hidden = H * Dv;
  const bool full = (mode_arg.get() == "full");
  const bool conv1d = full && conv_arg.get();
  const int kd = H * Dk, vd = H * Dv;
  const int reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());

  auto mkf16 = [&](std::vector<int> shp, float v) {
    auto t = Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc();
    for (int64_t i = 0; i < t.numel(); ++i) t.ptr<__fp16>()[i] = (__fp16)v;
    return t;
  };

  auto mkf32 = [&](std::vector<int> shp, float v) {
    auto t = Tensor::empty(shp, mllm::kFloat32, mllm::kQNN).alloc();
    for (int64_t i = 0; i < t.numel(); ++i) t.ptr<float>()[i] = v;
    return t;
  };

  // Inputs in the exact order the compiled graph expects.
  std::vector<Tensor> ins;
  ins.push_back(mkf16({1, hidden}, 0.1f));                            // x
  for (int h = 0; h < H; ++h) ins.push_back(mkf32({1, Dk, Dv}, 0.01f));  // S_h (fp32 state)
  for (int h = 0; h < H; ++h) ins.push_back(mkf16({1, 1, 1}, 0.9f));     // gt_h
  for (int h = 0; h < H; ++h) ins.push_back(mkf16({1, 1, 1}, 0.5f));     // beta_h
  if (full) {
    ins.push_back(mkf16({1, 1, 1}, 1e-6f));   // eps
    ins.push_back(mkf16({1, 1, 1}, 1.f / 11.3137f));  // qscale ~ 1/sqrt(128)
    ins.push_back(mkf16({1, 1, Dv}, 1.0f));   // norm_w
  }
  if (conv1d) {
    ins.push_back(mkf16({1, 4, kd}, 0.1f));   // cw_q
    ins.push_back(mkf16({1, 4, kd}, 0.1f));   // cw_k
    ins.push_back(mkf16({1, 4, vd}, 0.1f));   // cw_v
    ins.push_back(mkf16({1, 3, kd}, 0.0f));   // cs_q
    ins.push_back(mkf16({1, 3, kd}, 0.0f));   // cs_k
    ins.push_back(mkf16({1, 3, vd}, 0.0f));   // cs_v
  }

  // Outputs: recur -> {Sp_h, out_h}*H ; full -> {Sp_h}*H then y (index H).
  std::vector<Tensor> outs;
  for (int h = 0; h < H; ++h) {
    outs.push_back(Tensor::empty({1, Dk, Dv}, mllm::kFloat32, mllm::kQNN).alloc());  // Sp_h (fp32 state)
    if (!full) outs.push_back(Tensor::empty({1, 1, Dv}, mllm::kFloat16, mllm::kQNN).alloc());  // out_h
  }
  if (full) outs.push_back(Tensor::empty({1, hidden}, mllm::kFloat16, mllm::kQNN).alloc());  // y (index H)
  if (conv1d) {  // updated conv state for the next decode step (indices H+1,H+2,H+3)
    outs.push_back(Tensor::empty({1, 3, kd}, mllm::kFloat16, mllm::kQNN).alloc());  // new_cs_q
    outs.push_back(Tensor::empty({1, 3, kd}, mllm::kFloat16, mllm::kQNN).alloc());  // new_cs_k
    outs.push_back(Tensor::empty({1, 3, vd}, mllm::kFloat16, mllm::kQNN).alloc());  // new_cs_v
  }

  // ---- accuracy validation: overwrite inputs from the reference .mllm ----
  const bool validate = !ref_arg.get().empty();
  Tensor exp_y, exp_Sp;
  if (validate) {
    if (!full) { fmt::print("[deltanet decode] --ref only supported for --mode full\n"); return 1; }
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto fp32 = [&](const std::string& n) { return ref->pull(n); };
    auto setf16 = [&](Tensor& dst, const float* src, int64_t n) {
      for (int64_t i = 0; i < n; ++i) dst.ptr<__fp16>()[i] = (__fp16)src[i];
    };
    auto setf32 = [&](Tensor& dst, const float* src, int64_t n) {
      for (int64_t i = 0; i < n; ++i) dst.ptr<float>()[i] = src[i];
    };
    auto x = fp32("x"); setf16(ins[0], x.ptr<float>(), hidden);
    auto S0 = fp32("S0");      // [H,Dk,Dv]
    auto gt = fp32("gt");      // [H,1,1]
    auto beta = fp32("beta");  // [H,1,1]
    const int shd = Dk * Dv;
    for (int h = 0; h < H; ++h) {
      setf32(ins[1 + h], S0.ptr<float>() + (size_t)h * shd, shd);  // S_h is fp32
      setf16(ins[1 + H + h], gt.ptr<float>() + h, 1);
      setf16(ins[1 + 2 * H + h], beta.ptr<float>() + h, 1);
    }
    auto eps = fp32("eps"); setf16(ins[1 + 3 * H], eps.ptr<float>(), 1);
    auto qs = fp32("qscale"); setf16(ins[2 + 3 * H], qs.ptr<float>(), 1);
    auto nw = fp32("norm_w"); setf16(ins[3 + 3 * H], nw.ptr<float>(), Dv);
    if (conv1d) {
      auto cwq = fp32("cw_q"); setf16(ins[4 + 3 * H], cwq.ptr<float>(), 4 * kd);
      auto cwk = fp32("cw_k"); setf16(ins[5 + 3 * H], cwk.ptr<float>(), 4 * kd);
      auto cwv = fp32("cw_v"); setf16(ins[6 + 3 * H], cwv.ptr<float>(), 4 * vd);
      auto csq = fp32("cs_q"); setf16(ins[7 + 3 * H], csq.ptr<float>(), 3 * kd);
      auto csk = fp32("cs_k"); setf16(ins[8 + 3 * H], csk.ptr<float>(), 3 * kd);
      auto csv = fp32("cs_v"); setf16(ins[9 + 3 * H], csv.ptr<float>(), 3 * vd);
    }
    exp_y = fp32("exp_y");
    exp_Sp = fp32("exp_Sp");
  }

  QnnAOTModule g("model.0.s" + std::to_string(hidden));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();  // warm

  if (validate) {
    // outs (full): {Sp_h}*H then y[1,hidden]. Sp at h, y at H.
    double ey = 0, es = 0;
    const Tensor& y = outs[H];
    for (int i = 0; i < hidden; ++i) ey = std::max(ey, (double)std::fabs((float)y.ptr<__fp16>()[i] - exp_y.ptr<float>()[i]));
    const int shd = Dk * Dv;
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < shd; ++i)
        es = std::max(es, (double)std::fabs(outs[h].ptr<float>()[i] - exp_Sp.ptr<float>()[(size_t)h * shd + i]));  // Sp fp32
    double yrange = 0, srange = 0;
    for (int i = 0; i < hidden; ++i) yrange = std::max(yrange, (double)std::fabs(exp_y.ptr<float>()[i]));
    for (int64_t i = 0; i < exp_Sp.numel(); ++i) srange = std::max(srange, (double)std::fabs(exp_Sp.ptr<float>()[i]));
    double yr = 100.0 * ey / (yrange + 1e-9), sr = 100.0 * es / (srange + 1e-9);
    fmt::print("[deltanet decode VALIDATE] graph vs int4 ref ({}):\n", conv1d ? "with conv1d" : "conv1d-free");
    fmt::print(
               "  y    : max|err|={:.5f}  |y|max={:.4f}  rel={:.2f}pct\n"
               "  state: max|err|={:.5f}  |S|max={:.4f}  rel={:.2f}pct\n"
               "  -> {} (fp16 + a16 + int4-weight precision; rel<5pct)\n",
               ey, yrange, yr, es, srange, sr, (yr < 5.0 && sr < 5.0) ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[deltanet decode LPBQ] mode={} H={} Dk={} Dv={} avg = {:.4f} ms ({} reps)\n", mode_arg.get(), H, Dk, Dv,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
});
