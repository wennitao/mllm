// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Host driver for the MONOLITHIC Sq=B BLOCK-PREFILL graph (compile_monolithic_prefill
// --layers 24 --head -> qwen3-mono-prefill-full.bin). Loads the single context + consts/
// embed, sets the per-layer constant inputs (norms/gates/conv-weights from consts.mllm,
// same mapping as run_monolithic.cpp), embeds the WHOLE prompt as x[B,hidden] (padded to
// B with the last real token), builds per-token RoPE + the causal mask + chunk masks, and
// dispatches ONCE. With --head, argmax of the LAST REAL token's logits = the first
// generated token (should match the int4 golden, e.g. ' Tokyo' 25358 for the Japan prompt).
//
//   ./mllm-qwen3-aot-mono-prefill-run -m qwen3-mono-prefill-full.bin --dir wm --layers 24 \
//        --seq 128 --chunk 32 --head
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

namespace {
constexpr int kHidden = 2048, kLH = 16, kDk = 128, kDv = 128;
constexpr int kH = 8, kKV = 2, kAttD = 256, kRot = 64, kVocab = 248320;
bool isDeltanet(int i) { return ((i + 1) % 4) != 0; }
}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-mono-prefill-full.bin");
  auto& dir_arg = Argparse::add<std::string>("--dir").def("wm");
  auto& layers_arg = Argparse::add<int>("--layers").def(24);
  auto& seq_arg = Argparse::add<int>("--seq").def(128);
  auto& chunk_arg = Argparse::add<int>("--chunk").def(32);
  auto& head_arg = Argparse::add<bool>("--head").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int N = layers_arg.get(), B = seq_arg.get(), C = chunk_arg.get();
  const bool head = head_arg.get();
  const std::string dir = dir_arg.get();
  const int kd = kLH * kDk, vd = kLH * kDv;

  mllm::initQnnBackend(model_path.get());
  auto consts = mllm::load(dir + "/consts.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  auto seed = mllm::load(dir + "/seed.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  auto embedf = mllm::load(dir + "/embed.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  auto C_ = [&](const std::string& n) { return consts->pull(n); };
  auto embed = embedf->pull("embed");
  const int T = (int)seed->pull("prompt_ids").numel();
  std::vector<int> prompt(T);
  for (int t = 0; t < T; ++t) prompt[t] = seed->pull("prompt_ids").ptr<int32_t>()[t];
  const int nreal = std::min(T, B);
  fmt::print("[mono-prefill] loaded; prompt T={} -> block B={} C={} head={}\n", T, B, C, head);

  std::vector<Tensor> ins;
  auto add16 = [&](std::vector<int> s) { auto t = Tensor::empty(s, mllm::kFloat16, mllm::kQNN).alloc(); for (int64_t i=0;i<t.numel();++i) t.ptr<__fp16>()[i]=(__fp16)0.f; ins.push_back(t); return (int)ins.size()-1; };
  auto add32 = [&](std::vector<int> s) { auto t = Tensor::empty(s, mllm::kFloat32, mllm::kQNN).alloc(); for (int64_t i=0;i<t.numel();++i) t.ptr<float>()[i]=0.f; ins.push_back(t); return (int)ins.size()-1; };
  auto setHW = [&](int idx, Tensor t) { for (int64_t i=0;i<t.numel();++i) ins[idx].ptr<float>()[i]=t.ptr<float>()[i]; };  // fp32 const -> fp32 input

  // shared
  int ix   = add16({B, kHidden});
  int isin = add16({1, B, kRot});
  int icos = add16({1, B, kRot});
  int icm  = add16({1, 1, B, B});
  int ieps = add32({1, 1, 1}); ins[ieps].ptr<float>()[0] = 1e-6f;
  int iLt  = add32({1, 1, C, C});
  int iSt  = add32({1, 1, C, C});
  int iEy  = add32({1, 1, C, C});
  add32({kLH, kDk, kDv});            // S0z (zeros)
  add32({1, 3, kd}); add32({1, 3, kd}); add32({1, 3, vd});  // cs zeros

  // x = embed gather (pad to B with last real token), per-token RoPE, causal mask, chunk masks
  for (int b = 0; b < B; ++b) {
    int tok = (b < nreal) ? prompt[b] : prompt[nreal - 1];
    for (int d = 0; d < kHidden; ++d) ins[ix].ptr<__fp16>()[(size_t)b * kHidden + d] = embed.ptr<__fp16>()[(size_t)tok * kHidden + d];
  }
  auto invf = consts->pull("inv_freq");
  float ascale = 1.0f; try { ascale = consts->pull("rope_attn_scaling").ptr<float>()[0]; } catch (...) {}
  const int half = kRot / 2;
  for (int b = 0; b < B; ++b) for (int d = 0; d < half; ++d) {
    float f = (float)b * invf.ptr<float>()[d];
    float cv = std::cos(f) * ascale, sv = std::sin(f) * ascale;
    ins[icos].ptr<__fp16>()[(size_t)b*kRot+d]=(__fp16)cv; ins[icos].ptr<__fp16>()[(size_t)b*kRot+d+half]=(__fp16)cv;
    ins[isin].ptr<__fp16>()[(size_t)b*kRot+d]=(__fp16)sv; ins[isin].ptr<__fp16>()[(size_t)b*kRot+d+half]=(__fp16)sv;
  }
  for (int i = 0; i < B; ++i) for (int j = 0; j < B; ++j) ins[icm].ptr<__fp16>()[i*B+j] = (j<=i)?(__fp16)0.f:(__fp16)(-50000.f);
  for (int i = 0; i < C; ++i) for (int j = 0; j < C; ++j) {
    ins[iLt].ptr<float>()[i*C+j]=(j<=i)?1.f:0.f; ins[iSt].ptr<float>()[i*C+j]=(j<i)?1.f:0.f; ins[iEy].ptr<float>()[i*C+j]=(i==j)?1.f:0.f;
  }

  // per-layer consts (same mapping as run_monolithic.cpp)
  for (int i = 0; i < N; ++i) {
    std::string s = "l" + std::to_string(i) + ".";
    int in_w = add32({1,1,kHidden}); setHW(in_w, C_(s+"input_norm_w"));
    int post_w = add32({1,1,kHidden}); setHW(post_w, C_(s+"post_norm_w"));
    if (isDeltanet(i)) {
      int gn = add32({1,1,kDv}); setHW(gn, C_(s+"gated_norm_w"));
      int qs = add32({1,1,1}); setHW(qs, C_(s+"qscale"));
      int wa = add32({1,kHidden,kLH}), wb = add32({1,kHidden,kLH});
      auto a = C_(s+"in_proj_a"); auto b = C_(s+"in_proj_b");   // [LH,hidden]
      for (int r=0;r<kHidden;++r) for (int c=0;c<kLH;++c) {
        ins[wa].ptr<float>()[r*kLH+c]=a.ptr<float>()[c*kHidden+r];
        ins[wb].ptr<float>()[r*kLH+c]=b.ptr<float>()[c*kHidden+r];
      }
      int al = add32({1,1,kLH}); setHW(al, C_(s+"A_log"));
      int db = add32({1,1,kLH}); setHW(db, C_(s+"dt_bias"));
      int cwq=add32({1,4,kd}), cwk=add32({1,4,kd}), cwv=add32({1,4,vd});
      auto cw = C_(s+"conv1d_w");  // [6144,4]
      auto fill=[&](int idx,int base){ for (int tap=0;tap<4;++tap) for (int ch=0;ch<kd;++ch) ins[idx].ptr<float>()[tap*kd+ch]=cw.ptr<float>()[(base+ch)*4+tap]; };
      fill(cwq,0); fill(cwk,kd); fill(cwv,2*kd);
    } else {
      int qn = add32({1,1,kAttD}); setHW(qn, C_(s+"q_norm_w"));
      int kn = add32({1,1,kAttD}); setHW(kn, C_(s+"k_norm_w"));
    }
  }
  if (head) { int fn = add32({1,1,kHidden}); setHW(fn, C_("final_norm_w")); }
  fmt::print("[mono-prefill] built {} input tensors\n", (int)ins.size());

  // outputs: [logits|hidden, then per layer: deltanet Sp / attn k_all,v_all]
  std::vector<Tensor> outs;
  if (head) outs.push_back(Tensor::empty({B, kVocab}, mllm::kFloat16, mllm::kQNN).alloc());
  else outs.push_back(Tensor::empty({B, kHidden}, mllm::kFloat16, mllm::kQNN).alloc());
  for (int i = 0; i < N; ++i) {
    if (isDeltanet(i)) outs.push_back(Tensor::empty({1, kLH, kDk, kDv}, mllm::kFloat32, mllm::kQNN).alloc());
    else { outs.push_back(Tensor::empty({1, kKV, kAttD, B}, mllm::kFloat16, mllm::kQNN).alloc());
           outs.push_back(Tensor::empty({1, kKV, B, kAttD}, mllm::kFloat16, mllm::kQNN).alloc()); }
  }
  fmt::print("[mono-prefill] built {} output tensors\n", (int)outs.size());

  QnnAOTModule g("model.0.s" + std::to_string(kHidden));
  g.to(mllm::kQNN);
  g.setOutputTensors(outs);
  auto t0 = std::chrono::high_resolution_clock::now();
  (void)g(ins);
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[mono-prefill] dispatch {:.1f} ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

  if (head) {
    // argmax of the LAST REAL token's logits -> first generated token
    const int last = nreal - 1;
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < kVocab; ++v) { float lv = (float)outs[0].ptr<__fp16>()[(size_t)last * kVocab + v]; if (lv > bv) { bv = lv; best = v; } }
    fmt::print("[mono-prefill] last-real-token (pos {}) argmax = {} (logit {:.3f})\n", last, best, bv);
  } else {
    double mx = 0; for (int i = 0; i < nreal * kHidden; ++i) mx = std::max(mx, (double)std::fabs((float)outs[0].ptr<__fp16>()[i]));
    fmt::print("[mono-prefill] final hidden (real tokens) |max|={:.4f}\n", mx);
  }
});
