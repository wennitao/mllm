// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Shared BLOCK-PREFILL (Sq=B) layer modules for the Qwen3.5 whole-model NPU prefill
// graph — the prefill counterpart of qwen3_5_decode_layers.hpp. These are the validated
// per-layer forwards extracted from compile_deltanet_prefill.cpp (chunked gated-delta-rule,
// y 0.29%/state 0.34% at chunk<=32) and compile_attn_prefill.cpp (dense causal, y 0.16%),
// so the monolithic Qwen3_5PrefillStack can chain them into one Sq=B graph.
//
// HTP gotchas baked in (see the two compile_*_prefill.cpp for the debug story):
//  - deltanet: decay-exp args clamped to [-50,0] (HTP exp garbage on large-neg); decay [C,C]
//    diff via outer-product matmuls (no both-dims broadcast); doubling UT inverse with a P*1
//    distinct copy (no self-aliased matmul); K=200 scale-invariant gated RMSNorm (fp16
//    underflow); chunk C<=32 (chunk [C,C] matmuls run fp16). Reference calibrates q/k/v_out
//    QDQ on the PRE-conv projection.
//  - attn: per-token RoPE; rmsnorm denom floored (fp16 eps underflow); pad blocks with the
//    last real token (host), NOT zeros, else rsqrt(0)=nan poisons real tokens.
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include <mllm/mllm.hpp>
#include "modeling_qwen_qnn_aot_sha.hpp"  // CONV2D_PROPERTY + ptq::QDQ

namespace mllm::models::qwen3::sha {

using mllm::Tensor;

// ===========================================================================
// GatedDeltaNet block prefill (chunked gated-delta-rule, Sq=B).
// ===========================================================================
class DeltaNetPrefillLPBQ final : public nn::Module {
  int H_ = 16, Dk_ = 128, Dv_ = 128, hidden_ = 2048, B_ = 128, C_ = 32;
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_, z_proj_;
  nn::Conv2D out_proj_;

 public:
  DeltaNetPrefillLPBQ() = default;
  DeltaNetPrefillLPBQ(const std::string& name, int H, int Dk, int Dv, int hidden, int B, int C)
      : nn::Module(name), H_(H), Dk_(Dk), Dv_(Dv), hidden_(hidden), B_(B), C_(C) {
    for (int h = 0; h < H; ++h) {
      auto hs = std::to_string(h);
      q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
      z_proj_.emplace_back(reg<nn::Conv2D>("z_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
    }
    out_proj_ = reg<nn::Conv2D>("out_proj", H * Dv, hidden, CONV2D_PROPERTY);
  }

  // in: [xn[B,hidden], eps, qscale, gated_norm_w[1,1,Dv], A_log[1,1,H], dt_bias[1,1,H],
  //      Wa[1,hidden,H], Wb[1,hidden,H], cw_q,cw_k,cw_v[1,4,*], cs_q,cs_k,cs_v[1,3,*],
  //      S0[H,Dk,Dv], Ltri,strict,eye[1,1,C,C]]
  // out: [y[1,1,B,hidden], Sp[1,H,Dk,Dv]]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    const int vd = H_ * Dv_, nc = B_ / C_;
    int p = 0;
    auto x = in[p++], eps = in[p++], qscale = in[p++], gated_w = in[p++];
    auto A_log = in[p++], dt_bias = in[p++], Wa = in[p++], Wb = in[p++];
    auto cw_q = in[p++], cw_k = in[p++], cw_v = in[p++];
    auto cs_q = in[p++], cs_k = in[p++], cs_v = in[p++];
    auto S0 = in[p++], Ltri = in[p++], strict = in[p++], eye = in[p++];
    (void)qscale;
    auto eps_f = eps.to(kFloat32);

    // in-graph gates (log-decay): g_log=-exp(A_log)*softplus(xn@Wa+dt_bias); beta=sigmoid(xn@Wb).
    auto x4 = x.view({1, 1, B_, hidden_}, true).to(kFloat32);
    auto a = F::matmul(x4, Wa.view({1, 1, hidden_, H_}, true).to(kFloat32));
    auto bb = F::matmul(x4, Wb.view({1, 1, hidden_, H_}, true).to(kFloat32));
    auto Al = A_log.view({1, 1, 1, H_}, true).to(kFloat32);
    auto db = dt_bias.view({1, 1, 1, H_}, true).to(kFloat32);
    auto g_log = F::neg(F::exp(Al) * F::softplus(a + db));   // [1,1,B,H]
    auto beta = F::sigmoid(bb);                              // [1,1,B,H]

    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);
    auto proj = [&](std::vector<nn::Conv2D>& pr, int h, int D, const std::string& tag) {
      return ptq::QDQ(this, pr[h](xq), tag + std::to_string(h)).to(kFloat32).view({1, 1, B_, D}, true);
    };
    auto conv_silu = [&](Tensor raw, Tensor cw_full, Tensor cs_full, int h, int D) {
      auto cw = cw_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true).to(kFloat32);
      auto cs = cs_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true).to(kFloat32).view({1, 1, 3, D}, true);
      auto full = F::concat({cs, raw}, 2);                                  // [1,1,B+3,D]
      Tensor acc;
      for (int j = 0; j < 4; ++j) {
        auto tap = cw.slice({kAll, {j, j + 1}, kAll}, true).view({1, 1, 1, D}, true);
        auto sl = full.slice({kAll, kAll, {j, j + B_}, kAll}, true);
        acc = (j == 0) ? (sl * tap) : (acc + sl * tap);
      }
      return acc * F::sigmoid(acc);
    };
    const float kClampLo = -50.0f, kClampHi = 0.0f;
    auto cexp = [&](Tensor t) { return F::exp(F::clip(t, kClampLo, kClampHi)); };

    std::vector<Tensor> y_heads, Sp_heads;
    for (int h = 0; h < H_; ++h) {
      auto q = conv_silu(proj(q_proj_, h, Dk_, "q_out_qdq_h"), cw_q, cs_q, h, Dk_);
      auto k = conv_silu(proj(k_proj_, h, Dk_, "k_out_qdq_h"), cw_k, cs_k, h, Dk_);
      auto v = conv_silu(proj(v_proj_, h, Dv_, "v_out_qdq_h"), cw_v, cs_v, h, Dv_);
      auto z = proj(z_proj_, h, Dv_, "z_out_qdq_h");
      auto eps_l2 = eps_f.view({1, 1, 1, 1}, true).mulConstant(Tensor::constant(1000.0f, kFloat32));  // ~1e-3
      const float qsv = 1.0f / std::sqrt((float)Dk_);
      q = (q * F::rsqrt(F::sum(q * q, -1, true) + eps_l2)).mulConstant(Tensor::constant(qsv, kFloat32));
      k = k * F::rsqrt(F::sum(k * k, -1, true) + eps_l2);
      auto gh = g_log.slice({kAll, kAll, kAll, {h, h + 1}}, true);   // [1,1,B,1]
      auto bh = beta.slice({kAll, kAll, kAll, {h, h + 1}}, true);

      Tensor S = S0.slice({{h, h + 1}, kAll, kAll}, true).view({1, 1, Dk_, Dv_}, true).to(kFloat32);
      std::vector<Tensor> outc;
      for (int c = 0; c < nc; ++c) {
        auto sl = [&](Tensor t, int D) { return t.slice({kAll, kAll, {c * C_, (c + 1) * C_}, kAll}, true); };
        auto qc = sl(q, Dk_), kc = sl(k, Dk_), vc = sl(v, Dv_);
        auto gc_in = gh.slice({kAll, kAll, {c * C_, (c + 1) * C_}, kAll}, true);
        auto bc = bh.slice({kAll, kAll, {c * C_, (c + 1) * C_}, kAll}, true);
        auto gc = F::matmul(Ltri, gc_in);                       // cumsum [1,1,C,1]
        auto gcT = gc.transpose(2, 3);
        auto egc = cexp(gc);
        auto ones = Ltri + strict.transpose(2, 3);              // all-ones
        auto gc_col = F::matmul(gc, ones.slice({kAll, kAll, {0, 1}, kAll}, true));
        auto gc_row = F::matmul(ones.slice({kAll, kAll, kAll, {0, 1}}, true), gcT);
        auto dm = cexp((gc_col - gc_row) * Ltri) * Ltri;        // decay mask [1,1,C,C]
        auto kb = kc * bc, vb = vc * bc;
        auto L = F::neg(F::matmul(kb, kc.transpose(2, 3)) * dm) * strict;
        Tensor T = eye, P = L;                                  // T = (I-L)^{-1} via doubling
        for (int pw = 1; pw < C_; pw *= 2) {
          T = F::matmul(T, eye + P);
          auto Pc = P * ones;
          P = F::matmul(P, Pc);
        }
        auto U = F::matmul(T, vb);
        auto Wk = F::matmul(T, kb * egc);
        auto attn = F::matmul(qc, kc.transpose(2, 3)) * dm;
        auto v_new = U - F::matmul(Wk, S);
        auto a_inter = F::matmul(qc * egc, S);
        outc.push_back(a_inter + F::matmul(attn, v_new));
        auto gc_last = gc.slice({kAll, kAll, {C_ - 1, C_}, kAll}, true);
        auto kdec = kc * cexp(gc_last - gc);
        S = S * cexp(gc_last) + F::matmul(kdec.transpose(2, 3), v_new);
      }
      auto core = (nc == 1) ? outc[0] : F::concat(outc, 2);     // [1,1,B,Dv]
      // K=200 scale-invariant gated RMSNorm * silu(z) (fp16 underflow guard).
      const float Kc = 200.0f;
      auto cs = core.mulConstant(Tensor::constant(Kc, kFloat32));
      auto inv = F::rsqrt(F::mean(cs * cs, -1, true) + eps_f.mulConstant(Tensor::constant(Kc * Kc, kFloat32)));
      auto normed = (cs * inv) * gated_w.to(kFloat32).view({1, 1, 1, Dv_}, true);
      y_heads.push_back((normed * (z * F::sigmoid(z))).to(kFloat16));
      Sp_heads.push_back(S);
    }
    auto cat = F::concat(y_heads, -1);                          // [1,1,B,H*Dv]
    auto cq = ptq::QDQ(this, cat, "out_proj_input_qdq").view({1, 1, -1, vd}, true);
    auto y = ptq::QDQ(this, out_proj_(cq), "out_proj_output_qdq").to(kFloat16).view({1, 1, B_, hidden_}, true);

    std::vector<Tensor> outs;
    outs.push_back(y);                                          // [1,1,B,hidden]
    outs.push_back(F::concat(Sp_heads, 1));                     // Sp [1,H,Dk,Dv]
    return outs;
  }
};

// ===========================================================================
// Full-attention block prefill (dense causal Sq=B).
// ===========================================================================
class AttnPrefillLPBQ final : public nn::Module {
  int H_ = 8, KV_ = 2, D_ = 256, hidden_ = 2048, rot_ = 64, B_ = 128, grp_ = 4;
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_;
  nn::Conv2D o_proj_;

 public:
  AttnPrefillLPBQ() = default;
  AttnPrefillLPBQ(const std::string& name, int H, int KV, int D, int hidden, int rot, int B)
      : nn::Module(name), H_(H), KV_(KV), D_(D), hidden_(hidden), rot_(rot), B_(B), grp_(H / KV) {
    for (int h = 0; h < H; ++h) q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden, 2 * D, CONV2D_PROPERTY));
    for (int kv = 0; kv < KV; ++kv) {
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(kv), hidden, D, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(kv), hidden, D, CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", H * D, hidden, CONV2D_PROPERTY);
  }

  // in: [xn[B,hidden], sin[1,B,rot], cos[1,B,rot], q_norm_w[1,1,D], k_norm_w[1,1,D],
  //      eps[1,1,1], cmask[1,1,B,B]]
  // out: [y[1,1,B,hidden], k_all[1,KV,D,B], v_all[1,KV,B,D]]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0], sin = in[1], cos = in[2], q_norm_w = in[3], k_norm_w = in[4], eps = in[5], cmask = in[6];
    const float scale = 1.0f / std::sqrt((float)D_);
    const int half = rot_ / 2;
    auto eps_f = eps.to(kFloat32).view({1, 1, 1, 1}, true);
    auto sin4 = sin.view({1, 1, B_, rot_}, true), cos4 = cos.view({1, 1, B_, rot_}, true);

    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);
    auto rmsnorm = [&](Tensor t, Tensor w) {
      auto tf = t.to(kFloat32);
      auto denom = F::clip(F::mean(tf * tf, -1, true) + eps_f, 1e-4f, 1e30f);  // floor (fp16 eps underflow)
      return ((tf * F::rsqrt(denom)) * w.to(kFloat32).view({1, 1, 1, D_}, true)).to(kFloat16);
    };
    auto rope = [&](Tensor t) {
      auto rotp = t.slice({kAll, kAll, kAll, {0, rot_}}, true);
      auto pass = t.slice({kAll, kAll, kAll, {rot_, D_}}, true);
      auto x1 = rotp.slice({kAll, kAll, kAll, {0, half}}, true);
      auto x2 = rotp.slice({kAll, kAll, kAll, {half, rot_}}, true);
      auto rh = F::concat({-x2, x1}, -1);
      return F::concat({rotp * cos4 + rh * sin4, pass}, -1);
    };
    auto proj = [&](std::vector<nn::Conv2D>& pr, int h, int outD, const std::string& tag) {
      return ptq::QDQ(this, pr[h](xq), tag + std::to_string(h)).to(kFloat16).view({1, 1, B_, outD}, true);
    };

    std::vector<Tensor> Kf, Vf, k_all, v_all;
    for (int kv = 0; kv < KV_; ++kv) {
      auto k_n = rope(rmsnorm(proj(k_proj_, kv, D_, "k_out_qdq_h"), k_norm_w));
      auto v_r = proj(v_proj_, kv, D_, "v_out_qdq_h");
      Kf.push_back(k_n); Vf.push_back(v_r);
      k_all.push_back(k_n.transpose(2, 3)); v_all.push_back(v_r);
    }
    std::vector<Tensor> head_outs, gates;
    for (int h = 0; h < H_; ++h) {
      auto qg = proj(q_proj_, h, 2 * D_, "q_out_qdq_h");
      gates.push_back(qg.slice({kAll, kAll, kAll, {D_, 2 * D_}}, true));
      auto q_n = rope(rmsnorm(qg.slice({kAll, kAll, kAll, {0, D_}}, true), q_norm_w)).to(kFloat32);
      int kv = h / grp_;
      auto sc = F::matmul(q_n, Kf[kv].transpose(2, 3).to(kFloat32));   // [1,1,B,B]
      sc = sc.mulConstant(Tensor::constant(scale, kFloat32)) + cmask.to(kFloat32);
      sc = F::softmax(sc, -1);
      head_outs.push_back(F::matmul(sc, Vf[kv].to(kFloat32)).to(kFloat16));
    }
    auto y = F::concat(head_outs, -1) * F::sigmoid(F::concat(gates, -1));
    auto cq = ptq::QDQ(this, y, "o_proj_input_qdq").view({1, 1, -1, H_ * D_}, true);
    auto o = ptq::QDQ(this, o_proj_(cq), "o_proj_output_qdq").to(kFloat16).view({1, 1, B_, hidden_}, true);
    std::vector<Tensor> outs;
    outs.push_back(o);
    outs.push_back(F::concat(k_all, 1));   // k_all [1,KV,D,B]
    outs.push_back(F::concat(v_all, 1));   // v_all [1,KV,B,D]
    return outs;
  }
};

// ===========================================================================
// MLP block prefill (Sq=B) — Qwen3MLP handles the seq dim; close with a QDQ.
// ===========================================================================
class FullMLPPrefill final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_ = 2048, B_ = 128;

 public:
  FullMLPPrefill() = default;
  FullMLPPrefill(const std::string& name, const Qwen3Config& cfg, int B)
      : nn::Module(name), hidden_(cfg.hidden_size), B_(B) {
    mlp_ = reg<Qwen3MLP>("", cfg);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = mlp_.forward(inputs, args)[0];   // [1,B,hidden]
    return {ptq::QDQ(this, o, "down_proj_output_qdq").to(kFloat16).view({B_, hidden_}, true)};
  }
};

}  // namespace mllm::models::qwen3::sha
