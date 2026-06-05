// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// LFM2.5-8B-A1B per-layer decode (Sq=1) modules for the NPU graph, mirroring the
// validated Qwen3.5 decode layers (qwen3_5_decode_layers.hpp) but for LFM2's mixers:
//   - Lfm2ShortConvDecodeLPBQ : double-gated short conv (in_proj -> B,C,x; Bx=B*x;
//       depthwise causal conv1d kernel=L_cache (NO activation); y=C*conv; out_proj).
//       conv weight cw[1,K,H] + conv state cs[1,K-1,H] are graph inputs; the updated
//       state is an output (threaded by the stack across decode steps).
//   - Lfm2AttnDecodeLPBQ : GQA attention, per-head LPBQ q/k/v + plain RMSNorm q/k-norm
//       + FULL RoPE + KV-cache concat + scaled-dot/softmax + o_proj. NO output gate.
//
// Weight-heavy projections are LPBQ Conv2D (w4a16); the cheap conv/gate/attention math
// runs fp16/fp32 with QDQ only at the projection boundaries — exactly the qwen3_5 recipe.
#pragma once
#include <cmath>
#include <string>
#include <vector>
#include <mllm/mllm.hpp>
#include "modeling_qwen_qnn_aot_sha.hpp"  // CONV2D_PROPERTY + ptq::QDQ

namespace mllm::models::lfm2::sha {

using mllm::Tensor;
using mllm::nn::Module;
using vi32 = std::vector<int32_t>;  // for CONV2D_PROPERTY
namespace ptq = mllm::models::qwen3::sha::ptq;

// ============================================================================
// Double-gated short convolution (LFM2 Lfm2MoeShortConv), decode step.
//   proj = in_proj(x)  [1,1,3H] = [B | C | xin]
//   Bx   = B * xin
//   win  = concat(cs[1,K-1,H], Bx[1,1,H])  ; conv = sum_j cw[:,j,:]*win[:,j,:]   (NO act)
//   y    = C * conv ; out = out_proj(y) ; new_cs = win[:, 1:K, :]
// ============================================================================
class Lfm2ShortConvDecodeLPBQ final : public Module {
  int hidden_ = 2048, K_ = 3;
  mllm::nn::Conv2D in_proj_, out_proj_;

 public:
  Lfm2ShortConvDecodeLPBQ() = default;
  Lfm2ShortConvDecodeLPBQ(const std::string& name, int hidden, int L_cache) : Module(name), hidden_(hidden), K_(L_cache) {
    in_proj_ = reg<mllm::nn::Conv2D>("in_proj", hidden, 3 * hidden, CONV2D_PROPERTY);
    out_proj_ = reg<mllm::nn::Conv2D>("out_proj", hidden, hidden, CONV2D_PROPERTY);
  }

  // inputs: x[1,hidden], cw[1,K,H], cs[1,K-1,H]   outputs: y[1,hidden], new_cs[1,K-1,H]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = mllm::nn::functional;
    auto x = in[0];
    auto cw = in[1];   // [1,K,H]
    auto cs = in[2];   // [1,K-1,H]
    const int H = hidden_;

    auto xq = ptq::QDQ(this, x, "in_proj_input_qdq").view({1, 1, -1, H}, true);
    auto proj = ptq::QDQ(this, in_proj_(xq), "in_proj_output_qdq").to(mllm::kFloat16).view({1, 1, 3 * H}, true);
    auto B = proj.slice({kAll, kAll, {0, H}}, true);          // [1,1,H]
    auto C = proj.slice({kAll, kAll, {H, 2 * H}}, true);      // [1,1,H]
    auto xin = proj.slice({kAll, kAll, {2 * H, 3 * H}}, true);// [1,1,H]
    auto Bx = B * xin;                                        // [1,1,H]

    auto win = F::concat({cs, Bx}, 1);                        // [1,K,H] = [s0..s_{K-2}, Bx]
    auto conv = F::sum(win * cw, 1, true);                    // [1,1,H] depthwise tap-sum (no activation)
    auto new_cs = win.slice({kAll, {1, K_}, kAll}, true);     // [1,K-1,H]
    auto y = C * conv;                                        // [1,1,H]

    auto yq = ptq::QDQ(this, y, "out_proj_input_qdq").view({1, 1, -1, H}, true);
    auto out = ptq::QDQ(this, out_proj_(yq), "out_proj_output_qdq").to(mllm::kFloat16).view({1, H}, true);

    return {out, new_cs};
  }
};

// ============================================================================
// GQA full attention (LFM2 Lfm2MoeAttention), decode step. Per-head LPBQ q/k/v,
// plain RMSNorm q/k-norm (weight = w, NO unit offset), FULL RoPE over head_dim,
// KV-cache concat, fp32 scaled-dot/softmax, LPBQ o_proj. NO output gate.
// ============================================================================
class Lfm2AttnDecodeLPBQ final : public Module {
  int H_ = 32, KV_ = 8, D_ = 64, hidden_ = 2048, rot_ = 64, ctx_ = 256, grp_ = 4;
  std::vector<mllm::nn::Conv2D> q_proj_, k_proj_, v_proj_;
  mllm::nn::Conv2D o_proj_;

 public:
  Lfm2AttnDecodeLPBQ() = default;
  Lfm2AttnDecodeLPBQ(const std::string& name, int H, int KV, int D, int hidden, int rot, int ctx)
      : Module(name), H_(H), KV_(KV), D_(D), hidden_(hidden), rot_(rot), ctx_(ctx), grp_(H / KV) {
    for (int h = 0; h < H; ++h) q_proj_.emplace_back(reg<mllm::nn::Conv2D>("q_proj." + std::to_string(h), hidden, D, CONV2D_PROPERTY));
    for (int kv = 0; kv < KV; ++kv) {
      k_proj_.emplace_back(reg<mllm::nn::Conv2D>("k_proj." + std::to_string(kv), hidden, D, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<mllm::nn::Conv2D>("v_proj." + std::to_string(kv), hidden, D, CONV2D_PROPERTY));
    }
    o_proj_ = reg<mllm::nn::Conv2D>("o_proj", H * D, hidden, CONV2D_PROPERTY);
  }

  // inputs: x[1,hidden], sin[1,1,rot], cos[1,1,rot], q_norm_w[1,1,D], k_norm_w[1,1,D],
  //         eps[1,1,1], past_k[1,KV,D,P], past_v[1,KV,P,D], mask[1,1,1,ctx]
  // outputs: y[1,hidden], k_new[1,KV,D,1], v_new[1,KV,1,D]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = mllm::nn::functional;
    auto x = in[0];
    auto sin = in[1], cos = in[2], q_norm_w = in[3], k_norm_w = in[4], eps = in[5];
    auto past_k = in[6], past_v = in[7], mask = in[8];
    const float scale = 1.0f / std::sqrt((float)D_);
    const int half = rot_ / 2;
    auto eps_f = eps.to(mllm::kFloat32);

    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);

    auto rmsnorm = [&](Tensor t, Tensor w) {  // plain RMSNorm over D (weight = w), fp32 -> fp16
      auto tf = t.to(mllm::kFloat32);
      auto inv = F::rsqrt(F::mean(tf * tf, -1, true) + eps_f);
      return ((tf * inv) * w.to(mllm::kFloat32)).to(mllm::kFloat16);
    };
    auto rope = [&](Tensor t) {  // FULL RoPE: rot_==D_, so the whole head rotates (no pass-through)
      auto x1 = t.slice({kAll, kAll, {0, half}}, true);
      auto x2 = t.slice({kAll, kAll, {half, rot_}}, true);
      auto rh = F::concat({-x2, x1}, -1);
      auto roped = t * cos + rh * sin;
      return roped;  // [1,1,D]
    };

    std::vector<Tensor> Kfull, Vfull, k_new_heads, v_new_heads;
    for (int kv = 0; kv < KV_; ++kv) {
      auto ks = std::to_string(kv);
      auto k_raw = ptq::QDQ(this, k_proj_[kv](xq), "k_out_qdq_h" + ks).to(mllm::kFloat16).view({1, 1, D_}, true);
      auto v_raw = ptq::QDQ(this, v_proj_[kv](xq), "v_out_qdq_h" + ks).to(mllm::kFloat16).view({1, 1, D_}, true);
      auto k_n = rope(rmsnorm(k_raw, k_norm_w));
      auto k_t = k_n.view({1, 1, D_, 1}, true);
      auto v_t = v_raw.view({1, 1, 1, D_}, true);
      k_new_heads.push_back(k_t);
      v_new_heads.push_back(v_t);
      auto pk = past_k.slice({kAll, {kv, kv + 1}, kAll, kAll}, true).view({1, 1, D_, ctx_ - 1}, true);
      auto pv = past_v.slice({kAll, {kv, kv + 1}, kAll, kAll}, true).view({1, 1, ctx_ - 1, D_}, true);
      Kfull.push_back(F::concat({pk, k_t}, -1));   // [1,1,D,ctx]
      Vfull.push_back(F::concat({pv, v_t}, 2));    // [1,1,ctx,D]
    }

    std::vector<Tensor> head_outs;
    for (int h = 0; h < H_; ++h) {
      auto q_raw = ptq::QDQ(this, q_proj_[h](xq), "q_out_qdq_h" + std::to_string(h)).to(mllm::kFloat16).view({1, 1, D_}, true);
      auto q_n = rope(rmsnorm(q_raw, q_norm_w));
      int kv = h / grp_;
      auto q4 = q_n.view({1, 1, 1, D_}, true).to(mllm::kFloat32);
      auto attn = F::matmul(q4, Kfull[kv].to(mllm::kFloat32));  // [1,1,1,ctx]
      attn = attn.mulConstant(Tensor::constant(scale, mllm::kFloat32)) + mask.to(mllm::kFloat32);
      attn = F::softmax(attn, -1);
      auto out_h = F::matmul(attn, Vfull[kv].to(mllm::kFloat32)).to(mllm::kFloat16);  // [1,1,1,D]
      head_outs.push_back(out_h.view({1, 1, D_}, true));
    }

    auto y = F::concat(head_outs, -1);  // [1,1,H*D]  (NO output gate for LFM2)
    auto cq = ptq::QDQ(this, y, "o_proj_input_qdq").view({1, 1, -1, H_ * D_}, true);
    auto o = ptq::QDQ(this, o_proj_(cq), "o_proj_output_qdq").to(mllm::kFloat16).view({1, hidden_}, true);

    return {o, F::concat(k_new_heads, 1), F::concat(v_new_heads, 1)};
  }
};

// ============================================================================
// Grouped MoE experts: N expert SwiGLU MLPs in ONE graph (cuts graph count 704->88 so
// the 24-layer stack fits the device's context/graph limits). Each expert e{j} is a
// Qwen3MLP (gate/up/down LPBQ) + its closing QDQ; returns N outputs [1,hidden]. The host
// dispatches only the groups containing a selected expert and weighted-sums the chosen ones.
//   inputs: x[1,hidden]   outputs: y0..y{N-1}[1,hidden]
// ============================================================================
class Lfm2MoeGroupLPBQ final : public Module {
  std::vector<mllm::models::qwen3::sha::Qwen3MLP> mlp_;
  int hidden_ = 2048, n_ = 8;

 public:
  Lfm2MoeGroupLPBQ() = default;
  Lfm2MoeGroupLPBQ(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg, int n)
      : Module(name), hidden_(cfg.hidden_size), n_(n) {
    for (int j = 0; j < n; ++j) mlp_.push_back(reg<mllm::models::qwen3::sha::Qwen3MLP>("e" + std::to_string(j), cfg));
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>& args) override {
    std::vector<Tensor> outs;
    for (int j = 0; j < n_; ++j) {
      auto o = mlp_[j].forward(in, args)[0];
      o = ptq::QDQ(this, o, "e" + std::to_string(j) + ".down_proj_output_qdq").to(mllm::kFloat16).view({1, hidden_}, true);
      outs.push_back(o);
    }
    return outs;
  }
};

// ============================================================================
// lm_head (tied embed_tokens): final RMSNorm (plain weight, in-graph fp32) + LPBQ proj.
//   inputs: x[1,hidden], norm_w[1,1,hidden], eps[1,1,1]   outputs: logits[1,vocab]
// ============================================================================
class Lfm2LmHeadLPBQ final : public Module {
  int hidden_ = 2048, vocab_ = 128000;
  mllm::nn::Conv2D lm_head_;

 public:
  Lfm2LmHeadLPBQ() = default;
  Lfm2LmHeadLPBQ(const std::string& name, int hidden, int vocab) : Module(name), hidden_(hidden), vocab_(vocab) {
    lm_head_ = reg<mllm::nn::Conv2D>("lm_head", hidden, vocab, CONV2D_PROPERTY);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = mllm::nn::functional;
    auto x = in[0], norm_w = in[1], eps = in[2];
    auto xf = x.to(mllm::kFloat32);
    auto inv = F::rsqrt(F::mean(xf * xf, -1, true) + eps.to(mllm::kFloat32));
    auto xn = ((xf * inv) * norm_w.to(mllm::kFloat32)).to(mllm::kFloat16);
    auto xq = ptq::QDQ(this, xn, "lmhead_input_qdq").view({1, 1, -1, hidden_}, true);
    auto logits = ptq::QDQ(this, lm_head_(xq), "lmhead_output_qdq").to(mllm::kFloat16).view({1, vocab_}, true);
    return {logits};
  }
};

}  // namespace mllm::models::lfm2::sha
