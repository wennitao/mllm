// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device validation for the GatedDeltaNet decode-step graph. Feeds a known
// state S_past + per-head q/k/v/gt/beta, runs the recurrence on the HTP, and
// compares S_present + out against a host fp32 reference. Then ping-pongs
// S_present -> S_past and runs a 2nd step (the recurrent-state-cache mechanism),
// checking it matches two reference steps.
//
//   ./mllm-qwen3-aot-deltanet-step-run -m qwen3-deltanet-step.bin
//
#include <cmath>
#include <cstdio>
#include <vector>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::aot::QnnAOTModule;

// One host (fp32) reference step matching the on-device decode-core graph exactly:
// gate math, q/k L2-norm + q-scale, recurrence, then gated RMSNorm * silu(z).
static void refStep(std::vector<float>& S, const std::vector<float>& q_raw, const std::vector<float>& k_raw,
                    const std::vector<float>& v, const std::vector<float>& a, const std::vector<float>& bb,
                    const std::vector<float>& A_log, const std::vector<float>& dt_bias, const std::vector<float>& z,
                    const std::vector<float>& norm_w, float eps, float qscale, std::vector<float>& gated, int H, int Dk,
                    int Dv) {
  auto softplus = [](float x) { return x > 0.f ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); };
  auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };
  for (int h = 0; h < H; ++h) {
    float* Sh = S.data() + (size_t)h * Dk * Dv;
    const float* qrh = q_raw.data() + (size_t)h * Dk;
    const float* krh = k_raw.data() + (size_t)h * Dk;
    const float* vh = v.data() + (size_t)h * Dv;
    const float* zh = z.data() + (size_t)h * Dv;
    const float* wh = norm_w.data() + (size_t)h * Dv;
    float g = std::exp(-std::exp(A_log[h]) * softplus(a[h] + dt_bias[h]));  // gt
    float b = sigmoid(bb[h]);                                              // beta
    // L2-norm q,k over Dk; scale q.
    std::vector<float> qh(Dk), kh(Dk);
    float qs = 0, ks = 0;
    for (int d = 0; d < Dk; ++d) { qs += qrh[d] * qrh[d]; ks += krh[d] * krh[d]; }
    float qinv = 1.f / std::sqrt(qs + eps), kinv = 1.f / std::sqrt(ks + eps);
    for (int d = 0; d < Dk; ++d) { qh[d] = qrh[d] * qinv * qscale; kh[d] = krh[d] * kinv; }
    // Recurrence.
    for (int i = 0; i < Dk * Dv; ++i) Sh[i] *= g;
    std::vector<float> kv(Dv, 0.f), out(Dv);
    for (int d = 0; d < Dk; ++d)
      for (int e = 0; e < Dv; ++e) kv[e] += kh[d] * Sh[d * Dv + e];
    std::vector<float> delta(Dv);
    for (int e = 0; e < Dv; ++e) delta[e] = (vh[e] - kv[e]) * b;
    for (int d = 0; d < Dk; ++d)
      for (int e = 0; e < Dv; ++e) Sh[d * Dv + e] += kh[d] * delta[e];
    for (int e = 0; e < Dv; ++e) {
      float acc = 0.f;
      for (int d = 0; d < Dk; ++d) acc += qh[d] * Sh[d * Dv + e];
      out[e] = acc;
    }
    // Gated RMSNorm(out) over Dv * silu(z).
    float ms = 0;
    for (int e = 0; e < Dv; ++e) ms += out[e] * out[e];
    float minv = 1.f / std::sqrt(ms / Dv + eps);
    float* gh = gated.data() + (size_t)h * Dv;
    for (int e = 0; e < Dv; ++e) gh[e] = (out[e] * minv * wh[e]) * (zh[e] * sigmoid(zh[e]));
  }
}

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-deltanet-step.bin");
  auto& H_arg = Argparse::add<int>("--heads").def(16);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get();

  mllm::initQnnBackend(model_path.get());

  auto mkf16 = [&](std::vector<int> shp) { return Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc(); };
  const float eps = 1e-6f, qscale = 1.f / std::sqrt((float)Dk);
  auto S = mkf16({H, Dk, Dv}), q = mkf16({H, 1, Dk}), k = mkf16({H, 1, Dk}), v = mkf16({H, 1, Dv});
  auto a = mkf16({H, 1, 1}), b = mkf16({H, 1, 1}), A_log = mkf16({H, 1, 1}), dt_bias = mkf16({H, 1, 1});
  auto z = mkf16({H, 1, Dv}), eps_t = mkf16({H, 1, 1}), qscale_t = mkf16({H, 1, 1}), norm_w = mkf16({H, 1, Dv});
  auto Sp = mkf16({H, Dk, Dv}), out = mkf16({H, 1, Dv});

  // Deterministic small inputs (fp16-friendly).
  std::vector<float> hS(H * Dk * Dv), hq(H * Dk), hk(H * Dk), hv(H * Dv), ha(H), hb(H), hAlog(H), hdt(H);
  std::vector<float> hz(H * Dv), hnw(H * Dv);
  auto frac = [](int j, int m) { return (float)((j * 7 + 3) % m) / m; };
  for (int i = 0; i < H * Dk * Dv; ++i) hS[i] = 0.02f * frac(i, 11) - 0.01f;
  for (int i = 0; i < H * Dk; ++i) { hq[i] = 0.1f * frac(i, 13); hk[i] = 0.1f * frac(i + 5, 13); }
  for (int i = 0; i < H * Dv; ++i) { hv[i] = 0.2f * frac(i + 2, 17); hz[i] = 0.3f * frac(i + 1, 19) - 0.1f; hnw[i] = 0.8f + 0.4f * frac(i, 23); }
  for (int h = 0; h < H; ++h) {
    ha[h] = 0.5f * frac(h, 5);            // in_proj_a output
    hb[h] = 0.6f * frac(h + 1, 7) - 0.3f; // in_proj_b output
    hAlog[h] = 0.4f * frac(h + 2, 5) - 0.2f;  // log decay param
    hdt[h] = 0.1f * frac(h + 3, 5);       // dt_bias param
  }
  auto fill = [](Tensor& t, const std::vector<float>& src) {
    for (size_t i = 0; i < src.size(); ++i) t.ptr<__fp16>()[i] = (__fp16)src[i];
  };
  auto fillv = [](Tensor& t, float val) {
    for (int64_t i = 0; i < t.numel(); ++i) t.ptr<__fp16>()[i] = (__fp16)val;
  };
  fill(S, hS); fill(q, hq); fill(k, hk); fill(v, hv);
  fill(a, ha); fill(b, hb); fill(A_log, hAlog); fill(dt_bias, hdt);
  fill(z, hz); fill(norm_w, hnw); fillv(eps_t, eps); fillv(qscale_t, qscale);

  QnnAOTModule g("model.0.s128");
  g.to(mllm::kQNN);
  std::vector<Tensor> ins = {S, q, k, v, a, b, A_log, dt_bias, z, eps_t, qscale_t, norm_w}, outs = {Sp, out};
  g.setOutputTensors(outs);

  auto run_and_check = [&](const char* tag, std::vector<float>& refS) -> bool {
    (void)g(ins);
    std::vector<float> refOut(H * Dv);
    refStep(refS, hq, hk, hv, ha, hb, hAlog, hdt, hz, hnw, eps, qscale, refOut, H, Dk, Dv);  // refS -> S_present
    double mo = 0, ms = 0;
    for (int i = 0; i < H * Dv; ++i) mo = std::max(mo, (double)std::fabs((float)out.ptr<__fp16>()[i] - refOut[i]));
    for (int i = 0; i < H * Dk * Dv; ++i) ms = std::max(ms, (double)std::fabs((float)Sp.ptr<__fp16>()[i] - refS[i]));
    fmt::print("[deltanet {}] out[0..2]={:.4f} {:.4f} {:.4f}  ref={:.4f} {:.4f} {:.4f}\n", tag,
               (float)out.ptr<__fp16>()[0], (float)out.ptr<__fp16>()[1], (float)out.ptr<__fp16>()[2], refOut[0],
               refOut[1], refOut[2]);
    fmt::print("[deltanet {}] max|err| out={:.4f}  state={:.4f}  -> {}\n", tag, mo, ms,
               (mo < 0.02 && ms < 0.02) ? "PASS" : "CHECK");
    return mo < 0.02 && ms < 0.02;
  };

  // Step 1.
  std::vector<float> refS = hS;
  bool ok1 = run_and_check("step1", refS);

  // Ping-pong: feed S_present back as S_past (the recurrent-state cache), run step 2.
  for (int i = 0; i < H * Dk * Dv; ++i) S.ptr<__fp16>()[i] = Sp.ptr<__fp16>()[i];
  bool ok2 = run_and_check("step2", refS);  // refS already holds S after step1

  fmt::print("[deltanet] overall -> {}\n", (ok1 && ok2) ? "PASS" : "CHECK");
});
