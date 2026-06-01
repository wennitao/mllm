// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Shared decode-layer modules for the Qwen3.5 whole-model NPU graph. These are the
// EXACT validated per-layer forwards (deltanet / attn / mlp), extracted verbatim from
// compile_{deltanet,attn,mlp}_decode.cpp so the monolithic Qwen3_5DecodeStack can chain
// them into one graph. Constants (norm weights, gates, eps, qscale, conv weights) are
// passed as input tensors by the stack (which sources them from registered params /
// graph inputs); the stack computes the deltanet gt/beta in-graph and slices per head.
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include <mllm/mllm.hpp>
#include "modeling_qwen_qnn_aot_sha.hpp"  // Qwen3MLP + CONV2D_PROPERTY + ptq::QDQ

namespace mllm::models::qwen3::sha {

using mllm::Tensor;

class DeltaNetDecodeLPBQ final : public nn::Module {
  int H_ = 0, Dk_ = 0, Dv_ = 0, hidden_ = 0;
  bool full_ = false, conv1d_ = false;
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_, z_proj_;
  nn::Conv2D out_proj_;

 public:
  DeltaNetDecodeLPBQ() = default;
  DeltaNetDecodeLPBQ(const std::string& name, int H, int Dk, int Dv, int hidden, bool full, bool conv1d)
      : nn::Module(name), H_(H), Dk_(Dk), Dv_(Dv), hidden_(hidden), full_(full), conv1d_(conv1d && full) {
    for (int h = 0; h < H; ++h) {
      auto hs = std::to_string(h);
      q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
      if (full) z_proj_.emplace_back(reg<nn::Conv2D>("z_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
    }
    if (full) out_proj_ = reg<nn::Conv2D>("out_proj", H * Dv, hidden, CONV2D_PROPERTY);
  }

  // recur inputs: [x[1,hidden], S_h0..[1,Dk,Dv], gt_h0..[1,1,1], beta_h0..[1,1,1]]
  // full  inputs: + eps[1,1,1], qscale[1,1,1], norm_w[1,1,Dv]
  // full+conv1d inputs: + cw_q[1,4,H*Dk], cw_k[1,4,H*Dk], cw_v[1,4,H*Dv],
  //                       cs_q[1,3,H*Dk], cs_k[1,3,H*Dk], cs_v[1,3,H*Dv]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0];
    Tensor eps, qscale, norm_w, cw_q, cw_k, cw_v, cs_q, cs_k, cs_v;
    if (full_) {
      eps = in[1 + 3 * H_];
      qscale = in[2 + 3 * H_];
      norm_w = in[3 + 3 * H_];
    }
    if (conv1d_) {
      cw_q = in[4 + 3 * H_]; cw_k = in[5 + 3 * H_]; cw_v = in[6 + 3 * H_];
      cs_q = in[7 + 3 * H_]; cs_k = in[8 + 3 * H_]; cs_v = in[9 + 3 * H_];
    }

    // Shared QDQ'd input to every per-head projection ([1,1,1,hidden] uint16).
    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);

    auto proj = [&](std::vector<nn::Conv2D>& p, int h, int D, const std::string& tag) {
      // LPBQ conv (uint16 out) -> QDQ (attach output scale) -> dequant to fp16.
      auto y = ptq::QDQ(this, p[h](xq), tag + std::to_string(h));
      return y.to(kFloat16).view({1, 1, D}, true);  // [1,1,D] fp16 for the recurrence
    };

    // Depthwise causal conv1d (kernel 4) + silu on a head's raw projection, BEFORE
    // l2norm — matches HF causal_conv1d_update: out = silu(sum_j w_j * window_j),
    // window = [s0,s1,s2,x_t]; new state = [s1,s2,x_t]. cw/cs are stacked over heads
    // ([1,4|3,H*D]); slice this head's D channels. new_cs collected for the output.
    std::vector<Tensor> new_cs_q, new_cs_k, new_cs_v;
    auto conv_silu = [&](Tensor proj_raw, Tensor cw_full, Tensor cs_full, int h, int D, std::vector<Tensor>& ncs) {
      auto cw = cw_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true);  // [1,4,D]
      auto cs = cs_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true);  // [1,3,D]
      auto win = F::concat({cs, proj_raw}, 1);                            // [1,4,D] = [s0,s1,s2,x_t]
      auto c = F::sum(win * cw, 1, true);                                 // [1,1,D] depthwise tap-sum
      c = c * F::sigmoid(c);                                              // silu
      ncs.push_back(win.slice({kAll, {1, 4}, kAll}, true));               // [1,3,D]
      return c;
    };

    std::vector<Tensor> gated_heads;
    std::vector<Tensor> outs;  // recur: {Sp_h, out_h}* ; full: {Sp_h}* then [new_cs_qkv] y
    for (int h = 0; h < H_; ++h) {
      Tensor S_h = in[1 + h];            // [1,Dk,Dv] recurrent state
      Tensor gt_h = in[1 + H_ + h];      // [1,1,1]
      Tensor beta_h = in[1 + 2 * H_ + h];// [1,1,1]
      auto q_raw = proj(q_proj_, h, Dk_, "q_out_qdq_h");
      auto k_raw = proj(k_proj_, h, Dk_, "k_out_qdq_h");
      auto v = proj(v_proj_, h, Dv_, "v_out_qdq_h");

      if (conv1d_) {  // conv1d+silu on q/k/v raw projections (z is NOT convolved)
        q_raw = conv_silu(q_raw, cw_q, cs_q, h, Dk_, new_cs_q);
        k_raw = conv_silu(k_raw, cw_k, cs_k, h, Dk_, new_cs_k);
        v = conv_silu(v, cw_v, cs_v, h, Dv_, new_cs_v);
      }

      // The recurrence runs in FP32 — HF's torch_recurrent_gated_delta_rule upcasts
      // q/k/v/g/beta/state to fp32. out=q@Sp is a near-cancellation (~1e-4 from terms
      // ~1e-2) whose direction fp16 can't hold (inputs are fp16-rounded); the gated
      // RMSNorm then amplifies that error ~1e3x. State is carried fp32 in/out too.
      auto S_f = S_h.to(kFloat32), gt_f = gt_h.to(kFloat32), beta_f = beta_h.to(kFloat32);
      auto eps_f = eps.to(kFloat32);
      Tensor q = q_raw.to(kFloat32), k = k_raw.to(kFloat32), v_f = v.to(kFloat32);
      if (full_) {
        // q/k L2-norm over Dk (fp32) + qscale on q.
        q = (q * F::rsqrt(F::sum(q * q, -1, true) + eps_f)) * qscale.to(kFloat32);
        k = k * F::rsqrt(F::sum(k * k, -1, true) + eps_f);
      }
      auto Ss = S_f * gt_f;                              // [1,Dk,Dv] fp32
      auto kv = F::matmul(k, Ss);                        // [1,1,Dv]
      auto delta = (v_f - kv) * beta_f;                  // [1,1,Dv]
      auto outer = F::matmul(k.transpose(1, 2), delta);  // [1,Dk,Dv]
      auto Sp = Ss + outer;                              // fp32 state out
      auto out = F::matmul(q, Sp);                       // [1,1,Dv] fp32
      outs.push_back(Sp);

      if (!full_) {
        outs.push_back(out.to(kFloat16));
        continue;
      }
      // Gated RMSNorm(out) over Dv * silu(z); gated -> fp16 for the LPBQ out_proj.
      // out=q@Sp is a near-cancellation and the elementwise ops run in fp16, so mean(out^2)
      // underflows to 0 and rsqrt(0+eps) -> inf (eps=1e-6 is an fp16 subnormal). RMSNorm is
      // scale-invariant, so scale out up by K and eps by K^2 to keep out^2 in fp16's normal
      // range while preserving the HF eps: normed = (out*K)*rsqrt(mean((out*K)^2)+eps*K^2).
      // K=200 (NOT 256: K^2=65536 > fp16 max 65504 -> inf); K^2=40000 is fp16-representable.
      const float Kc = 200.0f;
      auto K = Tensor::constant(Kc, kFloat32);
      auto K2 = Tensor::constant(Kc * Kc, kFloat32);
      auto os = out.mulConstant(K);                                          // out*K [1,1,Dv]
      auto inv = F::rsqrt(F::mean(os * os, -1, true) + eps_f.mulConstant(K2));
      auto z = proj(z_proj_, h, Dv_, "z_out_qdq_h").to(kFloat32);
      auto normed = (os * inv) * norm_w.to(kFloat32);
      auto gated = normed * (z * F::sigmoid(z));
      gated_heads.push_back(gated.to(kFloat16));
    }

    if (!full_) return outs;

    // Concat per-head gated outputs -> [1,1,H*Dv], LPBQ out_proj -> [1,hidden].
    auto cat = F::concat(gated_heads, -1);                                  // [1,1,H*Dv]
    auto cq = ptq::QDQ(this, cat, "out_proj_input_qdq").view({1, 1, -1, H_ * Dv_}, true);
    auto y = ptq::QDQ(this, out_proj_(cq), "out_proj_output_qdq").to(kFloat16).view({1, hidden_}, true);
    outs.push_back(y);  // y right after the H Sp states (index H)

    // Multi-token decode needs the UPDATED conv state carried to the next step.
    // Concat per-head new_cs ([1,3,D] each) over heads -> [1,3,H*D], appended after y
    // (indices H+1,H+2,H+3). Single-step validators just ignore these extra outputs.
    if (conv1d_) {
      outs.push_back(F::concat(new_cs_q, -1));   // new_cs_q [1,3,H*Dk]  (index H+1)
      outs.push_back(F::concat(new_cs_k, -1));   // new_cs_k [1,3,H*Dk]  (index H+2)
      outs.push_back(F::concat(new_cs_v, -1));   // new_cs_v [1,3,H*Dv]  (index H+3)
    }
    return outs;
  }
};

class AttnDecodeLPBQ final : public nn::Module {
  int H_ = 8, KV_ = 2, D_ = 256, hidden_ = 2048, rot_ = 64, ctx_ = 256;
  int grp_ = 4;  // H_/KV_
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_;  // q_proj: hidden->2D (query|gate)
  nn::Conv2D o_proj_;

 public:
  AttnDecodeLPBQ() = default;
  AttnDecodeLPBQ(const std::string& name, int H, int KV, int D, int hidden, int rot, int ctx, float eps)
      : nn::Module(name), H_(H), KV_(KV), D_(D), hidden_(hidden), rot_(rot), ctx_(ctx), grp_(H / KV) {
    for (int h = 0; h < H; ++h) {
      auto hs = std::to_string(h);
      q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + hs, hidden, 2 * D, CONV2D_PROPERTY));  // query|gate
    }
    for (int h = 0; h < KV; ++h) {
      auto hs = std::to_string(h);
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + hs, hidden, D, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + hs, hidden, D, CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", H * D, hidden, CONV2D_PROPERTY);
  }

  // inputs: x[1,hidden], sin[1,1,rot], cos[1,1,rot], q_norm_w[1,1,D], k_norm_w[1,1,D],
  //         eps[1,1,1], past_k[1,KV,D,P], past_v[1,KV,P,D], mask[1,1,1,ctx]
  // q/k-norm is computed by hand in fp32 (RMSNormOp requires a uint16 weight; the
  // by-hand form keeps it fp32 and accepts the norm weight as a graph input with
  // add_unit_offset (1+w) already baked in at export).
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0];
    auto sin = in[1];       // [1,1,rot]
    auto cos = in[2];       // [1,1,rot]
    auto q_norm_w = in[3];  // [1,1,D]  (1+w baked)
    auto k_norm_w = in[4];  // [1,1,D]
    auto eps = in[5];       // [1,1,1]
    auto past_k = in[6];    // [1,KV,D,P]
    auto past_v = in[7];    // [1,KV,P,D]
    auto mask = in[8];      // [1,1,1,ctx] additive fp16
    const float scale = 1.0f / std::sqrt((float)D_);
    const int half = rot_ / 2;
    auto eps_f = eps.to(kFloat32);

    // Shared QDQ'd input to every projection ([1,1,1,hidden] uint16).
    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);

    // hand-rolled RMSNorm over the last dim (D), fp32, weight = 1+w (baked). -> fp16.
    auto rmsnorm = [&](Tensor t, Tensor w) {
      auto tf = t.to(kFloat32);
      auto inv = F::rsqrt(F::mean(tf * tf, -1, true) + eps_f);
      return ((tf * inv) * w.to(kFloat32)).to(kFloat16);
    };

    // partial RoPE on the first rot_ channels of a [1,1,D] head (split-half conv).
    auto rope = [&](Tensor t) {
      auto rotp = t.slice({kAll, kAll, {0, rot_}}, true);     // [1,1,rot]
      auto pass = t.slice({kAll, kAll, {rot_, D_}}, true);    // [1,1,D-rot]
      auto x1 = rotp.slice({kAll, kAll, {0, half}}, true);    // [1,1,half]
      auto x2 = rotp.slice({kAll, kAll, {half, rot_}}, true); // [1,1,half]
      auto rh = F::concat({-x2, x1}, -1);                     // rotate_half [1,1,rot]
      auto roped = rotp * cos + rh * sin;                     // [1,1,rot]
      return F::concat({roped, pass}, -1);                    // [1,1,D]
    };

    // ---- K/V projections + norm + rope + cache concat (per kv head) ----
    std::vector<Tensor> Kfull, Vfull, k_new_heads, v_new_heads;
    for (int kv = 0; kv < KV_; ++kv) {
      auto ks = std::to_string(kv);
      auto k_raw = ptq::QDQ(this, k_proj_[kv](xq), "k_out_qdq_h" + ks).to(kFloat16).view({1, 1, D_}, true);
      auto v_raw = ptq::QDQ(this, v_proj_[kv](xq), "v_out_qdq_h" + ks).to(kFloat16).view({1, 1, D_}, true);
      auto k_n = rmsnorm(k_raw, k_norm_w);   // RMSNorm over D
      k_n = rope(k_n);                       // [1,1,D]
      // new k for cache: [1,1,D,1] ; new v: [1,1,1,D]
      auto k_t = k_n.view({1, 1, D_, 1}, true);
      auto v_t = v_raw.view({1, 1, 1, D_}, true);
      k_new_heads.push_back(k_t);
      v_new_heads.push_back(v_t);
      // K_full = concat(past_k[kv] [1,1,D,P], k_t [1,1,D,1]) -> [1,1,D,ctx]
      auto pk = past_k.slice({kAll, {kv, kv + 1}, kAll, kAll}, true).view({1, 1, D_, ctx_ - 1}, true);
      auto pv = past_v.slice({kAll, {kv, kv + 1}, kAll, kAll}, true).view({1, 1, ctx_ - 1, D_}, true);
      Kfull.push_back(F::concat({pk, k_t}, -1));   // [1,1,D,ctx]
      Vfull.push_back(F::concat({pv, v_t}, 2));    // [1,1,ctx,D]
    }

    // ---- per q-head attention ----
    std::vector<Tensor> head_outs, gates;
    for (int h = 0; h < H_; ++h) {
      auto hs = std::to_string(h);
      auto qg = ptq::QDQ(this, q_proj_[h](xq), "q_out_qdq_h" + hs).to(kFloat16).view({1, 1, 2 * D_}, true);
      auto q_raw = qg.slice({kAll, kAll, {0, D_}}, true);       // query [1,1,D]
      auto gate = qg.slice({kAll, kAll, {D_, 2 * D_}}, true);   // gate  [1,1,D]
      gates.push_back(gate);
      auto q_n = rmsnorm(q_raw, q_norm_w);
      q_n = rope(q_n);                                          // [1,1,D]
      int kv = h / grp_;
      // attn = q @ K_full in fp32 (scores/softmax precision; HTP supports fp32 matmul).
      auto q4 = q_n.view({1, 1, 1, D_}, true).to(kFloat32);     // [1,1,1,D]
      auto attn = F::matmul(q4, Kfull[kv].to(kFloat32));        // [1,1,1,ctx]
      attn = attn.mulConstant(Tensor::constant(scale, kFloat32)) + mask.to(kFloat32);  // [1,1,1,ctx]
      attn = F::softmax(attn, -1);
      auto out_h = F::matmul(attn, Vfull[kv].to(kFloat32)).to(kFloat16);  // [1,1,1,D]
      head_outs.push_back(out_h.view({1, 1, D_}, true));
    }

    // concat heads -> [1,1,H*D], apply output gate sigmoid, o_proj (LPBQ)
    auto y = F::concat(head_outs, -1);                          // [1,1,H*D]
    auto g = F::concat(gates, -1);                              // [1,1,H*D]
    y = y * F::sigmoid(g);
    auto cq = ptq::QDQ(this, y, "o_proj_input_qdq").view({1, 1, -1, H_ * D_}, true);
    auto o = ptq::QDQ(this, o_proj_(cq), "o_proj_output_qdq").to(kFloat16).view({1, hidden_}, true);

    std::vector<Tensor> outs;
    outs.push_back(o);                                  // y  (index 0)
    outs.push_back(F::concat(k_new_heads, 1));          // k_new [1,KV,D,1] (index 1)
    outs.push_back(F::concat(v_new_heads, 1));          // v_new [1,KV,1,D] (index 2)
    return outs;
  }
};

class FullMLPDecode final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_ = 2048;

 public:
  FullMLPDecode() = default;
  FullMLPDecode(const std::string& name, const Qwen3Config& cfg) : nn::Module(name), hidden_(cfg.hidden_size) {
    mlp_ = reg<Qwen3MLP>("", cfg);  // convs at model.gate_proj etc.
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = mlp_.forward(inputs, args)[0];
    return {ptq::QDQ(this, o, "down_proj_output_qdq").to(kFloat16).view({1, hidden_}, true)};
  }
};

}  // namespace mllm::models::qwen3::sha
