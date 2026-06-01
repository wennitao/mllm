// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Host decode loop for the COMPLETE one-graph Qwen3.5-2B model (compile_monolithic
// --layers 24 --head -> qwen3-mono-full.bin). Loads the single 1 GB context, seeds the
// recurrent/conv/KV states from the prefill (seed.mllm), sets the per-layer constant
// inputs ONCE (norm weights / gates / conv weights from consts.mllm), then per token:
// embed gather (embed.mllm) -> RoPE -> dispatch -> argmax logits -> carry the 354 state
// outputs back into the next token's state inputs. Validates against whole_model_golden.py.
//
// Input order MUST match compile_monolithic.cpp buildInputs(); output order matches the
// forward()'s states-append order (logits, then per layer: deltanet Sp_h0..15 + new_cs_q/k/v
// OR attn k_new,v_new — in layer order).
//
//   ./mllm-qwen3-aot-mono-run -m qwen3-mono-full.bin --dir wm --first 25358 --max_new 24
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
constexpr int kH = 8, kKV = 2, kAttD = 256, kRot = 64;
constexpr int kVocab = 248320, kCtx = 256, kP = 255, kN = 24;
bool isDeltanet(int i) { return ((i + 1) % 4) != 0; }

// per-layer index bookkeeping into the flat ins/outs vectors
struct LayerIO {
  bool deltanet;
  // input indices
  std::vector<int> S_in;          // 16 deltanet S
  int csq_in, csk_in, csv_in;     // deltanet conv state
  int pk_in, pv_in;               // attn past_k/v
  // output indices
  std::vector<int> Sp_out;        // 16 deltanet Sp
  int ncsq_out, ncsk_out, ncsv_out;
  int knew_out, vnew_out;
};
}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-mono-full.bin");
  auto& dir_arg = Argparse::add<std::string>("--dir").help("dir with consts.mllm/seed.mllm/embed.mllm").def("wm");
  auto& first_arg = Argparse::add<int>("--first").help("first decode token id (CPU-prefill mode only)").def(25358);
  auto& maxnew_arg = Argparse::add<int>("--max_new").def(24);
  auto& npu_prefill = Argparse::add<bool>("--npu_prefill").help("prefill ON NPU: zero state, run prompt tokens through the decode graph (no HF seed)").def(true);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const std::string dir = dir_arg.get();
  const bool prefillNpu = npu_prefill.get();

  mllm::initQnnBackend(model_path.get());
  auto consts = mllm::load(dir + "/consts.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  auto seed = mllm::load(dir + "/seed.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  auto embedf = mllm::load(dir + "/embed.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  auto C = [&](const std::string& n) { return consts->pull(n); };
  auto S = [&](const std::string& n) { return seed->pull(n); };
  auto embed = embedf->pull("embed");  // [vocab, hidden] fp16
  const int T = (int)seed->pull("prompt_ids").numel();
  fmt::print("[mono] loaded; prompt T={} first_token={} max_new={}\n", T, first_arg.get(), maxnew_arg.get());

  // ---------- build the flat input vector in buildInputs() order ----------
  std::vector<Tensor> ins;
  auto add = [&](std::vector<int> shp) {
    auto t = Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc();
    for (int64_t i = 0; i < t.numel(); ++i) t.ptr<__fp16>()[i] = (__fp16)0.f;
    ins.push_back(t);
    return (int)ins.size() - 1;
  };
  auto setf16 = [&](int idx, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) ins[idx].ptr<__fp16>()[i] = (__fp16)src[i];
  };
  // shared
  int ix = add({1, kHidden});       // x
  int isin = add({1, 1, kRot});     // sin
  int icos = add({1, 1, kRot});     // cos
  int imask = add({1, 1, 1, kCtx}); // mask
  int ieps = add({1, 1, 1});        // eps
  { float e = 1e-6f; setf16(ieps, &e, 1); }

  std::vector<LayerIO> L(kN);
  auto setConstHW = [&](int idx, Tensor t) {  // t fp32 -> fp16 input
    setf16(idx, t.ptr<float>(), t.numel());
  };
  for (int i = 0; i < kN; ++i) {
    std::string s = "l" + std::to_string(i) + ".";
    L[i].deltanet = isDeltanet(i);
    int in_w = add({1, 1, kHidden}); setConstHW(in_w, C(s + "input_norm_w"));
    int post_w = add({1, 1, kHidden}); setConstHW(post_w, C(s + "post_norm_w"));
    if (isDeltanet(i)) {
      int gn = add({1, 1, kDv}); setConstHW(gn, C(s + "gated_norm_w"));
      int qs = add({1, 1, 1}); setConstHW(qs, C(s + "qscale"));
      // Wa/Wb pre-transposed [hidden,LH] from in_proj_a/b [LH,hidden]
      int wa = add({1, kHidden, kLH});
      int wb = add({1, kHidden, kLH});
      {
        auto a = C(s + "in_proj_a"); auto b = C(s + "in_proj_b");  // [LH,hidden]
        for (int r = 0; r < kHidden; ++r) for (int c = 0; c < kLH; ++c) {
          ins[wa].ptr<__fp16>()[r * kLH + c] = (__fp16)a.ptr<float>()[c * kHidden + r];
          ins[wb].ptr<__fp16>()[r * kLH + c] = (__fp16)b.ptr<float>()[c * kHidden + r];
        }
      }
      int al = add({1, 1, kLH}); setConstHW(al, C(s + "A_log"));
      int db = add({1, 1, kLH}); setConstHW(db, C(s + "dt_bias"));
      // cw_{q,k,v}: conv1d_w [6144,4] (channel,tap) -> per-proj [4, 2048] (tap,channel)
      int cwq = add({1, 4, kLH * kDk}), cwk = add({1, 4, kLH * kDk}), cwv = add({1, 4, kLH * kDv});
      {
        auto cw = C(s + "conv1d_w");  // [6144,4]
        const int CD = kLH * kDk;     // 2048
        auto fill = [&](int idx, int base) {
          for (int tap = 0; tap < 4; ++tap) for (int ch = 0; ch < CD; ++ch)
            ins[idx].ptr<__fp16>()[tap * CD + ch] = (__fp16)cw.ptr<float>()[(base + ch) * 4 + tap];
        };
        fill(cwq, 0); fill(cwk, CD); fill(cwv, 2 * CD);
      }
      for (int h = 0; h < kLH; ++h) L[i].S_in.push_back(add({1, kDk, kDv}));
      L[i].csq_in = add({1, 3, kLH * kDk});
      L[i].csk_in = add({1, 3, kLH * kDk});
      L[i].csv_in = add({1, 3, kLH * kDv});
      // CPU-prefill: seed S from recurrent_states [1,LH,Dk,Dv]; cs from conv_states taps 1..3.
      // NPU-prefill: states stay ZERO (the prompt is run through the decode graph).
      if (!prefillNpu) {
        auto rec = S(s + "recurrent");   // [1,LH,Dk,Dv]
        for (int h = 0; h < kLH; ++h)
          setf16(L[i].S_in[h], rec.ptr<float>() + (size_t)h * kDk * kDv, kDk * kDv);
        auto cs = S(s + "conv");         // [1,6144,4]
        const int CD = kLH * kDk;
        auto seedcs = [&](int idx, int base) {
          for (int tap = 0; tap < 3; ++tap) for (int ch = 0; ch < CD; ++ch)
            ins[idx].ptr<__fp16>()[tap * CD + ch] = (__fp16)cs.ptr<float>()[(base + ch) * 4 + (tap + 1)];
        };
        seedcs(L[i].csq_in, 0); seedcs(L[i].csk_in, CD); seedcs(L[i].csv_in, 2 * CD);
      }
    } else {
      int qn = add({1, 1, kAttD}); setConstHW(qn, C(s + "q_norm_w"));
      int kn = add({1, 1, kAttD}); setConstHW(kn, C(s + "k_norm_w"));
      L[i].pk_in = add({1, kKV, kAttD, kP});  // past_k [1,KV,D,P]
      L[i].pv_in = add({1, kKV, kP, kAttD});  // past_v [1,KV,P,D]
      // CPU-prefill: seed KV from keys/values [1,KV,T,D]. NPU-prefill: KV stays zero.
      if (!prefillNpu) {
        auto keys = S(s + "keys"); auto vals = S(s + "values");  // [1,KV,T,D]
        for (int kv = 0; kv < kKV; ++kv) for (int t = 0; t < T && t < kP; ++t) for (int d = 0; d < kAttD; ++d) {
          ins[L[i].pk_in].ptr<__fp16>()[((size_t)kv * kAttD + d) * kP + t] = (__fp16)keys.ptr<float>()[((size_t)kv * T + t) * kAttD + d];
          ins[L[i].pv_in].ptr<__fp16>()[((size_t)kv * kP + t) * kAttD + d] = (__fp16)vals.ptr<float>()[((size_t)kv * T + t) * kAttD + d];
        }
      }
    }
  }
  int ifinal = add({1, 1, kHidden}); setConstHW(ifinal, C("final_norm_w"));
  fmt::print("[mono] built {} input tensors\n", (int)ins.size());

  // ---------- outputs (logits + 354 states), record indices ----------
  std::vector<Tensor> outs;
  int ologits = (int)outs.size();
  outs.push_back(Tensor::empty({1, kVocab}, mllm::kFloat16, mllm::kQNN).alloc());
  for (int i = 0; i < kN; ++i) {
    if (L[i].deltanet) {
      for (int h = 0; h < kLH; ++h) { L[i].Sp_out.push_back((int)outs.size()); outs.push_back(Tensor::empty({1, kDk, kDv}, mllm::kFloat32, mllm::kQNN).alloc()); }
      L[i].ncsq_out = (int)outs.size(); outs.push_back(Tensor::empty({1, 3, kLH * kDk}, mllm::kFloat16, mllm::kQNN).alloc());
      L[i].ncsk_out = (int)outs.size(); outs.push_back(Tensor::empty({1, 3, kLH * kDk}, mllm::kFloat16, mllm::kQNN).alloc());
      L[i].ncsv_out = (int)outs.size(); outs.push_back(Tensor::empty({1, 3, kLH * kDv}, mllm::kFloat16, mllm::kQNN).alloc());
    } else {
      L[i].knew_out = (int)outs.size(); outs.push_back(Tensor::empty({1, kKV, kAttD, 1}, mllm::kFloat16, mllm::kQNN).alloc());
      L[i].vnew_out = (int)outs.size(); outs.push_back(Tensor::empty({1, kKV, 1, kAttD}, mllm::kFloat16, mllm::kQNN).alloc());
    }
  }
  fmt::print("[mono] built {} output tensors\n", (int)outs.size());

  // ---------- RoPE: cos/sin for a position from inv_freq ----------
  auto invf = consts->pull("inv_freq");  // [rot/2]
  float ascale = 1.0f;
  try { ascale = consts->pull("rope_attn_scaling").ptr<float>()[0]; } catch (...) {}
  const int half = kRot / 2;
  auto setRoPE = [&](int pos) {
    for (int d = 0; d < half; ++d) {
      float f = (float)pos * invf.ptr<float>()[d];
      float cv = std::cos(f) * ascale, sv = std::sin(f) * ascale;
      ins[icos].ptr<__fp16>()[d] = (__fp16)cv; ins[icos].ptr<__fp16>()[d + half] = (__fp16)cv;
      ins[isin].ptr<__fp16>()[d] = (__fp16)sv; ins[isin].ptr<__fp16>()[d + half] = (__fp16)sv;
    }
  };
  auto setMask = [&](int pos) {  // valid slots 0..pos-1 (past) + slot ctx-1 (new token)
    for (int j = 0; j < kCtx; ++j) ins[imask].ptr<__fp16>()[j] = (__fp16)(-50000.f);
    for (int j = 0; j < pos && j < kP; ++j) ins[imask].ptr<__fp16>()[j] = (__fp16)0.f;
    ins[imask].ptr<__fp16>()[kCtx - 1] = (__fp16)0.f;
  };

  QnnAOTModule g("model.0.s" + std::to_string(kHidden));
  g.to(mllm::kQNN);

  // ---------- decode loop ----------
  // NPU prefill: feed prompt tokens 0..T-1 through the decode graph from ZERO state
  // (building S/conv/KV on the NPU), then generate. The argmax after the LAST prompt
  // token is the first generated token (should be the golden's first token if correct).
  std::vector<int> prompt(T);
  for (int t = 0; t < T; ++t) prompt[t] = seed->pull("prompt_ids").ptr<int32_t>()[t];
  const int total = prefillNpu ? (T + maxnew_arg.get()) : maxnew_arg.get();
  int pos = prefillNpu ? 0 : T;
  int prevBest = -1;
  std::vector<int> generated;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < total; ++step) {
    int cur = prefillNpu ? (step < T ? prompt[step] : prevBest)
                         : (step == 0 ? first_arg.get() : prevBest);
    // x = embed[cur]
    for (int d = 0; d < kHidden; ++d) ins[ix].ptr<__fp16>()[d] = embed.ptr<__fp16>()[(size_t)cur * kHidden + d];
    setRoPE(pos);
    setMask(pos);
    g.setOutputTensors(outs);
    (void)g(ins);
    // argmax logits
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < kVocab; ++v) { float lv = (float)outs[ologits].ptr<__fp16>()[v]; if (lv > bv) { bv = lv; best = v; } }
    bool is_gen = prefillNpu ? (step >= T - 1) : true;
    if (is_gen) generated.push_back(best);
    // carry states: deltanet Sp->S, new_cs->cs ; attn k_new/v_new -> write ring buffer at slot pos
    for (int i = 0; i < kN; ++i) {
      if (L[i].deltanet) {
        for (int h = 0; h < kLH; ++h)
          for (int e = 0; e < kDk * kDv; ++e) ins[L[i].S_in[h]].ptr<__fp16>()[e] = (__fp16)outs[L[i].Sp_out[h]].ptr<float>()[e];
        auto cp = [&](int dst, int src) { for (int64_t e = 0; e < ins[dst].numel(); ++e) ins[dst].ptr<__fp16>()[e] = outs[src].ptr<__fp16>()[e]; };
        cp(L[i].csq_in, L[i].ncsq_out); cp(L[i].csk_in, L[i].ncsk_out); cp(L[i].csv_in, L[i].ncsv_out);
      } else if (pos < kP) {
        // k_new [1,KV,D,1] -> past_k[:,:,:,pos] ; v_new [1,KV,1,D] -> past_v[:,:,pos,:]
        for (int kv = 0; kv < kKV; ++kv) {
          for (int d = 0; d < kAttD; ++d)
            ins[L[i].pk_in].ptr<__fp16>()[((size_t)kv * kAttD + d) * kP + pos] = outs[L[i].knew_out].ptr<__fp16>()[(size_t)kv * kAttD + d];
          for (int d = 0; d < kAttD; ++d)
            ins[L[i].pv_in].ptr<__fp16>()[((size_t)kv * kP + pos) * kAttD + d] = outs[L[i].vnew_out].ptr<__fp16>()[(size_t)kv * kAttD + d];
        }
      }
    }
    prevBest = best;
    fmt::print("  step {:2d} pos {:3d} {} -> token {}\n", step, pos,
               (prefillNpu && step < T) ? "[prefill]" : "[ gen   ]", best);
    pos++;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[mono] generated tokens:");
  for (int t : generated) fmt::print(" {}", t);
  fmt::print("\n[mono] avg {:.2f} ms/token over {} tokens\n",
             std::chrono::duration<double, std::milli>(t1 - t0).count() / maxnew_arg.get(), maxnew_arg.get());
});
