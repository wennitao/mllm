// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// LFM2.5-8B-A1B (LiquidAI `lfm2_moe`) — CPU / fp32 reference implementation.
//
// Architecture (HF `modeling_lfm2_moe.py`): a hybrid decoder that interleaves
//   - double-gated short-convolution layers (the majority), and
//   - grouped-query full-attention layers (the minority, per `layer_types`),
// each followed by a SwiGLU feed-forward that is a dense MLP for the first
// `num_dense_layers` layers and a sparse top-k Mixture-of-Experts block after.
// Pre-norm residuals; tied embeddings.
//
// This file is a *correctness* reference: linear projections, the SwiGLU MLPs and
// the expert routing scatter/gather use mllm ops, while the two op-coverage gaps —
// the depthwise causal short-conv and the sigmoid+bias MoE router — are hand-rolled
// as explicit fp32 CPU loops that mirror the HF reference line-for-line. The
// short-conv keeps its own per-layer conv window (reset between prompts on the
// prefill step), attention layers share a StaticCache. Batch size is assumed 1.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/lfm2_moe/configuration_lfm2_moe.hpp"
#include "mllm/models/ARGeneration.hpp"

namespace mllm::models::lfm2_moe {

// ============================================================================
// RoPE helpers (standard rotate_half / non-interleaved "GLM-style": emb = [f, f]).
// ============================================================================

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto p = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { p[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

// Returns {sin, cos}, each [B, S, dim] where dim == 2 * inv_freq_len.
inline auto makeRotaryPosEmbedding(const Tensor& position_ids, const Tensor& inv_freq)
    -> std::pair<Tensor, Tensor> {
  auto B = position_ids.shape()[0];
  auto S = position_ids.shape()[1];
  auto half = inv_freq.shape()[0];
  auto dim = half * 2;

  auto sin_emb = Tensor::empty({B, S, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({B, S, dim}, kFloat32, kCPU).alloc();
  auto* sp = sin_emb.ptr<float>();
  auto* cp = cos_emb.ptr<float>();
  const auto* pos = position_ids.ptr<int64_t>();
  const auto* inv = inv_freq.ptr<float>();

  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < S; ++s) {
      auto p = static_cast<float>(pos[b * S + s]);
      for (int d = 0; d < half; ++d) {
        float f = p * inv[d];
        float sv = std::sin(f);
        float cv = std::cos(f);
        auto base = (b * S + s) * dim;
        sp[base + d] = sv;
        sp[base + d + half] = sv;
        cp[base + d] = cv;
        cp[base + d + half] = cv;
      }
    }
  }
  return {sin_emb, cos_emb};
}

// Rotate the first `rotary_dim` channels of a contiguous [B, H, S, D] tensor in
// place; channels >= rotary_dim are passed through unchanged. sin/cos are
// [B, S, rotary_dim] with the [freqs, freqs] layout (so entry d == entry d+half).
inline void applyPartialRoPEInPlace(Tensor& x, const Tensor& sin, const Tensor& cos, int rotary_dim) {
  int B = x.shape()[0], H = x.shape()[1], S = x.shape()[2], D = x.shape()[3];
  int half = rotary_dim / 2;
  auto* xp = x.ptr<float>();
  const auto* sp = sin.ptr<float>();
  const auto* cp = cos.ptr<float>();
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s = 0; s < S; ++s) {
        float* row = xp + (((static_cast<int64_t>(b) * H + h) * S + s) * D);
        const float* srow = sp + ((static_cast<int64_t>(b) * S + s) * rotary_dim);
        const float* crow = cp + ((static_cast<int64_t>(b) * S + s) * rotary_dim);
        for (int d = 0; d < half; ++d) {
          float x1 = row[d];
          float x2 = row[d + half];
          float c = crow[d];
          float si = srow[d];
          row[d] = x1 * c - x2 * si;
          row[d + half] = x2 * c + x1 * si;
        }
      }
    }
  }
}

// ============================================================================
// SwiGLU MLP (HF Lfm2MoeMLP). Checkpoint Linears are named w1=gate, w3=up, w2=down.
// Reused both as the dense per-layer FFN and as each per-expert FFN.
// ============================================================================

class Lfm2MoeMLP final : public nn::Module {
  nn::Linear w1_;  // gate
  nn::Linear w3_;  // up
  nn::Linear w2_;  // down
  nn::SiLU act_;

  int hidden_size_ = 0;
  int intermediate_size_ = 0;

 public:
  Lfm2MoeMLP() = default;
  Lfm2MoeMLP(const std::string& name, const Lfm2MoeConfig& cfg,
             const std::optional<int>& intermediate_size = std::nullopt)
      : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = intermediate_size.value_or(cfg.intermediate_size);
    w1_ = reg<nn::Linear>("w1", hidden_size_, intermediate_size_, false, cfg.linear_impl_type);
    w3_ = reg<nn::Linear>("w3", hidden_size_, intermediate_size_, false, cfg.linear_impl_type);
    w2_ = reg<nn::Linear>("w2", intermediate_size_, hidden_size_, false, cfg.linear_impl_type);
    act_ = reg<nn::SiLU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    return {w2_(act_(w1_(inputs[0])) * w3_(inputs[0]))};
  }
};

// ============================================================================
// Grouped-query full attention (HF Lfm2MoeAttention).
//   - per-head RMSNorm (q_layernorm / k_layernorm, plain, over head_dim) applied
//     BEFORE RoPE, full rotary over head_dim.
//   - GQA: matmul broadcasts kv_heads -> q_heads (eager path, shared StaticCache).
//   - no output gate, no bias on any projection.
// ============================================================================

class Lfm2MoeAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear out_proj_;
  nn::RMSNorm q_norm_;
  nn::RMSNorm k_norm_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_ = 0;
  int head_dim_ = 0;
  int num_heads_ = 0;
  int num_kv_heads_ = 0;
  int rotary_dim_ = 0;

 public:
  int attn_layer_idx_ = 0;  // index among full-attention layers (StaticCache slot)

  Lfm2MoeAttention() = default;
  Lfm2MoeAttention(const std::string& name, const Lfm2MoeConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    head_dim_ = cfg.head_dim;
    num_heads_ = cfg.num_attention_heads;
    num_kv_heads_ = cfg.num_key_value_heads;
    rotary_dim_ = cfg.rotary_dim;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, num_heads_ * head_dim_, false, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, num_kv_heads_ * head_dim_, false, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, num_kv_heads_ * head_dim_, false, cfg.linear_impl_type);
    out_proj_ = reg<nn::Linear>("out_proj", num_heads_ * head_dim_, hidden_size_, false, cfg.linear_impl_type);

    q_norm_ = reg<nn::RMSNorm>("q_layernorm", cfg.norm_eps, /*add_unit_offset=*/false);
    k_norm_ = reg<nn::RMSNorm>("k_layernorm", cfg.norm_eps, /*add_unit_offset=*/false);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  // inputs: {x, sin, cos}   args: {StaticCache*}
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto sin = inputs[1];
    auto cos = inputs[2];
    auto* kv_cache = args[0].get<nn::StaticCache*>();

    int B = x.shape()[0];
    int S = x.shape()[1];

    // q/k norm over head_dim (4D, last-dim norm) BEFORE RoPE.
    auto query = q_norm_(nn::functional::view(q_proj_(x), {B, S, num_heads_, head_dim_}));
    auto key = k_norm_(nn::functional::view(k_proj_(x), {B, S, num_kv_heads_, head_dim_}));
    auto value = nn::functional::view(v_proj_(x), {B, S, num_kv_heads_, head_dim_});

    // [B,H,S,D]
    query = query.transpose(1, 2).contiguous();
    key = key.transpose(1, 2).contiguous();
    value = value.transpose(1, 2).contiguous();

    applyPartialRoPEInPlace(query, sin, cos, rotary_dim_);
    applyPartialRoPEInPlace(key, sin, cos, rotary_dim_);

    // KV cache (stores kv_heads); matmul broadcasts kv_heads -> q_heads (GQA).
    auto [k_full, v_full] = kv_cache->updateKVCache(attn_layer_idx_, key, value);

    auto attn = nn::functional::matmul(query, k_full, false, true) * (1.f / std::sqrt((float)head_dim_));
    attn = mask_(attn);
    attn = softmax_(attn);
    auto out = nn::functional::matmul(attn, v_full);  // [B,H,S,D]

    out = out.transpose(1, 2).contiguous();
    out = nn::functional::view(out, {B, S, num_heads_ * head_dim_});
    out = out_proj_(out);
    return {out};
  }
};

// ============================================================================
// Double-gated short convolution (HF Lfm2MoeShortConv).
//
//   proj = in_proj(x)                          # [B,S, 3H] = [B_gate | C_gate | x_in]
//   Bx   = B_gate * x_in
//   conv_out = depthwise_causal_conv1d(Bx)     # kernel = L_cache, NO activation
//   y    = C_gate * conv_out
//   out  = out_proj(y)
//
// conv.weight is loaded as a bare param [H, K] (squeezed from [H,1,K]) and the
// conv is hand-rolled; the per-channel window of the last (K-1) Bx frames is kept
// in conv_state_ so a 1-token decode step reproduces the prefill exactly.
// ============================================================================

class Lfm2MoeShortConv final : public nn::Module {
  nn::Linear in_proj_;     // hidden -> 3*hidden
  nn::Linear out_proj_;    // hidden -> hidden
  nn::Param conv_weight_;  // [hidden, K]  (loaded as a bare param)

  Tensor conv_state_;      // [hidden, K-1]

  int hidden_size_ = 0;
  int conv_k_ = 0;

 public:
  Lfm2MoeShortConv() = default;
  Lfm2MoeShortConv(const std::string& name, const Lfm2MoeConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    conv_k_ = cfg.conv_L_cache;

    in_proj_ = reg<nn::Linear>("in_proj", hidden_size_, 3 * hidden_size_, cfg.conv_bias, cfg.linear_impl_type);
    out_proj_ = reg<nn::Linear>("out_proj", hidden_size_, hidden_size_, cfg.conv_bias, cfg.linear_impl_type);
    conv_weight_ = reg<nn::Param>("conv.weight", getModuleName() + ".conv.weight");

    resetState();
  }

  void resetState() { conv_state_ = Tensor::zeros({hidden_size_, conv_k_ - 1}, kFloat32, kCPU); }

  // inputs: {x}   args: {} (state is internal)
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = inputs[0];
    int B = x.shape()[0];
    int S = x.shape()[1];
    MLLM_RT_ASSERT_EQ(B, 1);  // reference conv scan assumes batch size 1
    int H = hidden_size_;
    int Kc = conv_k_;

    auto proj = in_proj_(x).contiguous();  // [1,S,3H]
    auto cw_t = conv_weight_.weight();      // [H,K]
    const float* pj = proj.ptr<float>();
    const float* cw = cw_t.ptr<float>();
    float* cstate = conv_state_.ptr<float>();  // [H, K-1]

    auto y = Tensor::empty({1, S, H}, kFloat32, kCPU).alloc();
    float* yp = y.ptr<float>();

    std::vector<float> padded(static_cast<size_t>(Kc - 1 + S));
    for (int c = 0; c < H; ++c) {
      // padded[c] = [conv_state[c] (K-1) | Bx_time[c] (S)] ; Bx = B_gate * x_in
      for (int j = 0; j < Kc - 1; ++j) { padded[j] = cstate[c * (Kc - 1) + j]; }
      for (int s = 0; s < S; ++s) {
        float bg = pj[(size_t)s * 3 * H + c];          // B_gate
        float xin = pj[(size_t)s * 3 * H + 2 * H + c]; // x_in
        padded[Kc - 1 + s] = bg * xin;
      }
      const float* w = cw + (size_t)c * Kc;
      for (int s = 0; s < S; ++s) {
        float acc = 0.f;
        for (int j = 0; j < Kc; ++j) { acc += w[j] * padded[s + j]; }  // causal depthwise conv (no act)
        float cg = pj[(size_t)s * 3 * H + H + c];  // C_gate
        yp[(size_t)s * H + c] = cg * acc;
      }
      // new conv state = last (K-1) Bx frames
      for (int j = 0; j < Kc - 1; ++j) { cstate[c * (Kc - 1) + j] = padded[S + j]; }
    }

    return {out_proj_(y)};
  }
};

// ============================================================================
// Sparse top-k Mixture-of-Experts block (HF Lfm2MoeSparseMoeBlock).
//
// Router (hand-rolled, fp32):
//   logits = x @ gate.weight^T                 # [T, E]
//   rw     = sigmoid(logits)
//   score  = rw + expert_bias                  # bias used for SELECTION only
//   idx    = topk(score, k)                    # top-k experts
//   w      = rw[idx]                           # weights are the bare sigmoid values
//   w     /= (sum(w) + 1e-6)                   # if norm_topk_prob
//   w     *= routed_scaling_factor
// The token->expert scatter/gather is the same routine as Qwen3-MoE.
// ============================================================================

class Lfm2MoE final : public nn::Module {
  nn::ModuleList<Lfm2MoeMLP> experts_;
  nn::Param gate_weight_;   // [num_experts, hidden]
  nn::Param expert_bias_;   // [num_experts]

  int top_k_ = 4;
  int num_experts_ = 32;
  bool norm_topk_prob_ = true;
  bool use_expert_bias_ = true;
  float routed_scaling_factor_ = 1.0f;

 public:
  Lfm2MoE() = default;
  Lfm2MoE(const std::string& name, const Lfm2MoeConfig& cfg) : nn::Module(name) {
    top_k_ = cfg.num_experts_per_tok;
    num_experts_ = cfg.num_experts;
    norm_topk_prob_ = cfg.norm_topk_prob;
    use_expert_bias_ = cfg.use_expert_bias;
    routed_scaling_factor_ = cfg.routed_scaling_factor;

    experts_ = reg<nn::ModuleList<Lfm2MoeMLP>>("experts", cfg.num_experts, cfg, cfg.moe_intermediate_size);
    gate_weight_ = reg<nn::Param>("gate.weight", getModuleName() + ".gate.weight");
    expert_bias_ = reg<nn::Param>("expert_bias", getModuleName() + ".expert_bias");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_states = inputs[0];
    auto orig_shape = hidden_states.shape();
    int H = hidden_states.size(-1);
    auto x2d = hidden_states.view({-1, H});  // [T, H]
    int T = x2d.size(0);

    // Router logits (x @ gate.weight^T), both fp32.
    auto logits = nn::functional::matmul(x2d, gate_weight_.weight(), false, true);  // [T, E]

    auto topk_idx = Tensor::empty({T, top_k_}, kInt32, kCPU).alloc();
    auto topk_weight = Tensor::empty({T, top_k_}, kFloat32, kCPU).alloc();
    {
      const float* lg = logits.ptr<float>();
      const float* bias = use_expert_bias_ ? expert_bias_.weight().ptr<float>() : nullptr;
      int32_t* oi = topk_idx.ptr<int32_t>();
      float* ow = topk_weight.ptr<float>();
      std::vector<float> sig(num_experts_), score(num_experts_);
      std::vector<int> order(num_experts_);
      for (int t = 0; t < T; ++t) {
        const float* row = lg + (size_t)t * num_experts_;
        for (int e = 0; e < num_experts_; ++e) {
          float s = 1.f / (1.f + std::exp(-row[e]));
          sig[e] = s;
          score[e] = bias ? (s + bias[e]) : s;
          order[e] = e;
        }
        // top-k by score (desc); index tiebreak keeps selection deterministic.
        std::partial_sort(order.begin(), order.begin() + top_k_, order.end(), [&](int a, int b) {
          return score[a] > score[b] || (score[a] == score[b] && a < b);
        });
        float wsum = 0.f;
        for (int j = 0; j < top_k_; ++j) { wsum += sig[order[j]]; }
        float inv = norm_topk_prob_ ? (1.f / (wsum + 1e-6f)) : 1.f;
        for (int j = 0; j < top_k_; ++j) {
          int e = order[j];
          oi[(size_t)t * top_k_ + j] = e;
          ow[(size_t)t * top_k_ + j] = sig[e] * inv * routed_scaling_factor_;
        }
      }
    }

    auto y = moeInfer(x2d, topk_idx, topk_weight).view(orig_shape);
    return {y};
  }

 private:
  // Identical scatter/gather routing to Qwen3-MoE: sort tokens by selected expert,
  // run each expert's GEMM over its slice, scatter back, weighted-sum over k.
  Tensor moeInfer(const Tensor& x, Tensor& topk_ids, Tensor& topk_weights) {
    auto cnts = Tensor::zeros({topk_ids.size(0), (int32_t)experts_.list().size()});
    {
      const int32_t* idx_ptr = topk_ids.ptr<mllm_int32_t>();
      float* cnt_ptr = cnts.ptr<mllm_fp32_t>();
      const int batch = topk_ids.size(0);
      const int k = topk_ids.size(1);
      const int n_exp = cnts.size(1);
      for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < k; ++j) {
          int32_t e = idx_ptr[b * k + j];
          MLLM_RT_ASSERT(e >= 0 && e < n_exp);
          cnt_ptr[b * n_exp + e] += 1.f;
        }
      }
    }
    auto tokens_per_expert = cnts.sum(0);
    auto idxs = topk_ids.view({-1}).argsort();

    auto sorted_tokens = x[{idxs / topk_ids.size(1), {kAll}}];

    std::vector<Tensor> outputs;
    int start_idx = 0;
    for (int i = 0; i < experts_.list().size(); ++i) {
      auto num_tokens = tokens_per_expert.ptr<mllm_fp32_t>()[i];
      auto end_idx = start_idx + (int32_t)num_tokens;
      if (num_tokens == 0) { continue; }
      auto& expert = experts_.list()[i];
      auto tokens_for_this_expert = sorted_tokens[{{start_idx, end_idx}, kAll}];
      auto expert_out = expert(tokens_for_this_expert)[0];
      outputs.push_back(expert_out);
      start_idx = end_idx;
    }

    auto outs = nn::functional::concat(outputs, 0);
    auto new_x = Tensor::emptyLike(outs).alloc();
    {
      const int32_t* idx_ptr = idxs.ptr<mllm_int32_t>();
      float* outs_ptr = outs.ptr<mllm_fp32_t>();
      float* new_x_ptr = new_x.ptr<mllm_fp32_t>();
      MLLM_RT_ASSERT_EQ(new_x.rank(), 2);
      MLLM_RT_ASSERT_EQ(new_x.size(0), idxs.size(0));
      auto dim = new_x.size(1);
      for (int i = 0; i < idxs.size(0); ++i) {
        int32_t idx = idx_ptr[i];
        std::memcpy(new_x_ptr + idx * dim, outs_ptr + i * dim, dim * sizeof(float));
      }
    }

    auto final_out_shape = topk_ids.shape();
    final_out_shape.emplace_back(-1);
    auto final_out =
        new_x.view(final_out_shape).to(topk_weights.dtype()).mul_(topk_weights.unsqueeze(-1)).sum(1).to(new_x.dtype());
    return final_out;
  }
};

// ============================================================================
// Decoder layer: pre-norm residual around (conv | self_attn), then FFN (dense | MoE).
//   residual = x
//   h = operator_norm(x); h = (conv|attn)(h); x = h + residual
//   x = x + feed_forward(ffn_norm(x))
// ============================================================================

class Lfm2MoeDecoder final : public nn::Module {
 public:
  bool is_attention_ = false;
  bool is_dense_ = false;
  Lfm2MoeAttention self_attn_;
  Lfm2MoeShortConv conv_;
  std::optional<Lfm2MoeMLP> feed_forward_dense_;
  std::optional<Lfm2MoE> feed_forward_moe_;
  nn::RMSNorm operator_norm_;
  nn::RMSNorm ffn_norm_;

  Lfm2MoeDecoder() = default;
  Lfm2MoeDecoder(const std::string& name, const Lfm2MoeConfig& cfg, int layer_idx) : nn::Module(name) {
    is_attention_ = cfg.isAttentionLayer(layer_idx);
    is_dense_ = cfg.isDenseLayer(layer_idx);

    if (is_attention_) {
      self_attn_ = reg<Lfm2MoeAttention>("self_attn", cfg);
    } else {
      conv_ = reg<Lfm2MoeShortConv>("conv", cfg);
    }
    if (is_dense_) {
      feed_forward_dense_ = reg<Lfm2MoeMLP>("feed_forward", cfg);
    } else {
      feed_forward_moe_ = reg<Lfm2MoE>("feed_forward", cfg);
    }
    operator_norm_ = reg<nn::RMSNorm>("operator_norm", cfg.norm_eps, /*add_unit_offset=*/false);
    ffn_norm_ = reg<nn::RMSNorm>("ffn_norm", cfg.norm_eps, /*add_unit_offset=*/false);
  }

  // inputs: {x, sin, cos}   args: {StaticCache*}
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto residual = inputs[0];
    auto h = operator_norm_(inputs[0]);
    if (is_attention_) {
      h = self_attn_(h, inputs[1], inputs[2], args[0])[0];
    } else {
      h = conv_(h)[0];
    }
    auto x = h + residual;
    auto f = ffn_norm_(x);
    if (is_dense_) {
      f = feed_forward_dense_.value()(f)[0];
    } else {
      f = feed_forward_moe_.value()(f)[0];
    }
    return {x + f};
  }
};

// ============================================================================
// Text model: embedding + decoder stack + final norm (embedding_norm).
// ============================================================================

class Lfm2MoeText final : public nn::Module {
 public:
  nn::Embedding embedding_;
  std::vector<Lfm2MoeDecoder> decoders_;
  nn::RMSNorm embedding_norm_;

  Lfm2MoeText() = default;
  Lfm2MoeText(const std::string& name, const Lfm2MoeConfig& cfg) : nn::Module(name) {
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    decoders_.reserve(cfg.num_hidden_layers);
    int attn_idx = 0;
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      auto dec = reg<Lfm2MoeDecoder>("layers." + std::to_string(i), cfg, i);
      if (dec.is_attention_) { dec.self_attn_.attn_layer_idx_ = attn_idx++; }
      decoders_.push_back(dec);
    }
    embedding_norm_ = reg<nn::RMSNorm>("embedding_norm", cfg.norm_eps, /*add_unit_offset=*/false);
  }

  void resetState() {
    for (auto& d : decoders_) {
      if (!d.is_attention_) { d.conv_.resetState(); }
    }
  }

  // inputs: {ids, sin, cos}   args: {StaticCache*}
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = embedding_(inputs[0]);
    for (auto& d : decoders_) { x = d(x, inputs[1], inputs[2], args[0])[0]; }
    return {embedding_norm_(x)};
  }
};

// ============================================================================
// Causal LM head + autoregressive generation glue.
// ============================================================================

class Lfm2MoeForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Lfm2MoeForCausalLM(const Lfm2MoeConfig& cfg) : cfg_(cfg) {
    cfg_.recompute();
    kv_cache_ = nn::StaticCache(cfg_.max_cache_length, cfg_.num_attention_layers,
                                cfg_.num_attention_heads,   // q_heads
                                cfg_.num_key_value_heads,   // kv_heads
                                cfg_.head_dim,              // kv_dim
                                kFloat32, kFloat32, kCPU, /*use_fa2=*/false);
    eos_token_id_ = cfg_.end_of_text_token_id;
    max_length_ = cfg_.max_cache_length;
    tie_word_embeddings_ = cfg_.tie_word_embeddings;

    llm_ = reg<Lfm2MoeText>("model", cfg_);
    lm_head_ = reg<nn::Linear>("lm_head_out", cfg_.hidden_size, cfg_.vocab_size, false, cfg_.linear_impl_type);

    auto inv = makeRoPEInvFreq(cfg_.rotary_dim, cfg_.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  void load(const ParameterFile::ptr_t& param_file) {
    // lm_head: tied -> reuse embed_tokens; untied -> use the checkpoint's lm_head.
    if (!param_file->has("lm_head_out.weight")) {
      if (param_file->has("lm_head.weight")) {
        param_file->push("lm_head_out.weight", param_file->pull("lm_head.weight"));
      } else if (param_file->has("model.embed_tokens.weight")) {
        param_file->push("lm_head_out.weight", param_file->pull("model.embed_tokens.weight"));
      }
    }
    nn::Module::load(param_file);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs&) override {
    auto sequence = input.at("sequence");
    auto B = sequence.shape()[0];
    auto S = sequence.shape()[1];

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (S == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({B, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      // Prefill of a fresh prompt: start each independent sequence from clean state.
      llm_.resetState();
      kv_cache_.clearCache();
      position_ids = Tensor::empty({B, S}, kInt64, kCPU).alloc();
      auto* p = position_ids.ptr<int64_t>();
      for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) { p[b * S + s] = s; }
      }
    }

    auto [sin, cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"));

    sequence = llm_(sequence, sin, cos, AnyValue(&kv_cache_))[0];

    // keep only the last position, then project to vocab
    {
      auto Sout = sequence.shape()[1];
      sequence = sequence[{kAll, {Sout - 1}, kAll}];
    }
    sequence = lm_head_(sequence);

    return {{"sequence", sequence}, {"position_ids", position_ids}};
  }

  inline nn::StaticCache& kvCache() { return kv_cache_; }

 private:
  Lfm2MoeConfig cfg_;
  Lfm2MoeText llm_;
  nn::Linear lm_head_;
  bool tie_word_embeddings_ = true;
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::lfm2_moe
