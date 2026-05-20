// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// fp16 Qwen3 for QNN-AOT compile with **causal block-sparse attention**,
// dispatched **per q-block** (one BQ-row chunk per graphExecute).
//
// Compared to modeling_qwen_qnn_aot_fp16_blocksparse.hpp:
//   * Causal: applies the per-qb causal mask (padding slots + diagonal-slot
//     lower-triangle) inside the attention compute.
//   * Per-qb dispatch shape: the compiled graph processes BQ (= 32) tokens
//     per call. The runner re-binds Q/K/V/padding-mask and dispatches
//     num_qb times per prefill chunk.
//   * Mask handling = "split mask" design validated in tests/qnn/
//     BlockSparseAttentionCausalTest.cpp's PerQbStaticMask runner:
//       - STATIC triangle mask  [1, BQ, top_k*BK]   baked at compile time
//       - APP_WRITE padding mask [1, top_k*BK]      bound per dispatch
//     Two ElementWiseAdds before Softmax. Pure built-in QNN ops — no
//     custom-op-package dispatch on the per-qb hot path.
//   * Attention compute uses **rank-3 batched MatMul** over Hq, not the
//     per-head SHA loop the non-causal blocksparse file used. The per-head
//     pattern hits a 1.2–1.9× QNN MatMul-scheduling penalty (see
//     docs/qnn_backend/block_sparse_attention.md § "rank-4 vs rank-3").
//     Q/K/V projections remain SHA (per-head Linear) for AOT compile-time
//     savings; their outputs are concat'd into rank-3 tensors before the
//     attention matmul.
//
// Layer inputs (in order):
//   0 : hidden_states     [1, BQ, hidden_size]
//   1 : sin               [1, BQ, head_dim]            RoPE for this qb's positions
//   2 : cos               [1, BQ, head_dim]
//   3 : past_key          [1, num_kv_heads, head_dim, max_kv-BQ]   (for cache update)
//   4 : past_value        [1, num_kv_heads, max_kv-BQ, head_dim]
//   5 : K_arranged        [num_attention_heads, 1, (top_k-1)*BK, head_dim]  (HISTORICAL slots only — runner gather)
//   6 : V_arranged        [num_attention_heads, 1, (top_k-1)*BK, head_dim]
//   7 : padding_mask      [1, top_k*BK]                                     (qb-dependent)
//
// Layer outputs:
//   0 : hidden_states_next [1, BQ, hidden_size]
//   1 : present_key        [1, num_kv_heads, head_dim, BQ]
//   2 : present_value      [1, num_kv_heads, BQ, head_dim]
//
// Compile-time constants:
//   BQ = BK = 32, top_k = 8, top_k_BK = 256, hist_k_BK = (top_k-1)*BK = 224.
//
// Diagonal slot (chunk-local self-attention) is handled INSIDE the layer:
// the per-head current K/V (computed from this dispatch's hidden_states)
// is concat'd onto the historical K_arranged before the attention MatMul.
// This mirrors how the standard dense model handles per-chunk prefill
// (concat past_key + current k_h inside the layer — see
// modeling_qwen_qnn_aot_sha.hpp:372-377). The runner therefore only needs
// to gather HISTORICAL k-blocks (top_k - 1 of them), avoiding the
// chicken-and-egg problem where the diagonal slot's K wouldn't yet exist.

#pragma once

#include "mllm/core/TensorStorage.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

namespace mllm::models::qwen3::sha_fp16_blocksparse_causal {

constexpr int kBQ = 32;
constexpr int kBK = 32;
constexpr int kTopK = 8;
constexpr int kTopKBK = kTopK * kBK;            // 256 — full attention K width
constexpr int kHistKBK = (kTopK - 1) * kBK;     // 224 — runner-supplied historical width
constexpr float kMaskNeg = -1.0e4f;             // -inf surrogate; matches the test infra

inline Tensor rotateHalf(Tensor x) {
  auto D = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, /*ssa=*/true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, /*ssa=*/true);
  return nn::functional::concat({-x2, x1}, -1);
}

// MLP — same as the non-causal file.
class Qwen3MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;
  int hidden_size_;
  int intermediate_size_;

 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, /*bias=*/false);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, /*bias=*/false);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, /*bias=*/false);
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    // QNN AOT visitors implement Sigmoid but not SiLU directly; decompose
    // SiLU as `gate * sigmoid(gate)`, matching how modeling_qwen_qnn_aot_sha
    // handles it.
    auto gate_pre = gate_proj_(x);
    auto gate = gate_pre * nn::functional::sigmoid(gate_pre);
    auto up = up_proj_(x);
    auto y = down_proj_(gate * up);
    return {y};
  }
};

// ---------------------------------------------------------------------------
// Per-qb causal block-sparse attention.
//
// Q/K/V projections are SHA (one Linear per head — keeps AOT compile time
// reasonable). Their outputs are concat'd into rank-3 tensors so the
// attention MatMuls are one batched op over Hq (rank-3 form that QNN's
// MatMul schedules ~1.2–1.9× better than the rank-4 SHA loop).
// ---------------------------------------------------------------------------
class Qwen3CausalBlockSparseAttention final : public nn::Module {
  std::vector<nn::Linear> q_projs_;
  std::vector<nn::Linear> k_projs_;
  std::vector<nn::Linear> v_projs_;
  nn::Linear o_proj_;

  std::vector<nn::RMSNorm> rms_norm_q_;
  std::vector<nn::RMSNorm> rms_norm_k_;

  // Static causal triangle mask: [1, BQ, top_k*BK]. Zeros everywhere except
  // the diagonal slot's upper-triangle, which is kMaskNeg. Baked at compile
  // time and reused across all qb dispatches.
  nn::Param triangle_mask_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;

 public:
  Qwen3CausalBlockSparseAttention() = default;

  Qwen3CausalBlockSparseAttention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = 1.f / std::sqrt((float)head_dim_);

    for (int h = 0; h < num_attention_heads_; ++h) {
      q_projs_.emplace_back(reg<nn::Linear>("q_proj." + std::to_string(h), hidden_size_, head_dim_, /*bias=*/false));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      k_projs_.emplace_back(reg<nn::Linear>("k_proj." + std::to_string(h), hidden_size_, head_dim_, /*bias=*/false));
      v_projs_.emplace_back(reg<nn::Linear>("v_proj." + std::to_string(h), hidden_size_, head_dim_, /*bias=*/false));
    }
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, /*bias=*/false);

    for (int h = 0; h < num_attention_heads_; ++h) {
      rms_norm_q_.emplace_back(reg<nn::RMSNorm>("q_norm." + std::to_string(h), cfg.rms_norm_eps));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      rms_norm_k_.emplace_back(reg<nn::RMSNorm>("k_norm." + std::to_string(h), cfg.rms_norm_eps));
    }

    triangle_mask_ = reg<nn::Param>("triangle_mask", "model.causal_triangle_mask");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    // sin/cos arrive already 4D (1, 1, S, D) from the parent's gather —
    // GatherOp::reshape outputs table.shape[..dim] + indices.shape +
    // table.shape[dim+1..] = [1] + [1, S] + [D] = (1, 1, S, D). The earlier
    // unsqueeze(1) would have inflated this to 5D which (a) silently passed
    // ConcatOp::reshape's positional dim check at N=128 (since D==N==128)
    // but (b) crashed at N != 128.
    auto sin = inputs[1];
    auto cos = inputs[2];
    auto past_key     = inputs[3];
    auto past_value   = inputs[4];
    auto K_arranged   = inputs[5];   // [Hq, top_k*BK, D]
    auto V_arranged   = inputs[6];   // [Hq, top_k*BK, D]
    auto padding_mask = inputs[7];   // [1, top_k*BK] (qb-dependent)

    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // -----------------------------------------------------------------------
    // Per-head Q/K/V projections (SHA layout for AOT compile-time).
    // -----------------------------------------------------------------------
    std::vector<Tensor> Q_per_head, K_per_head, V_per_head;
    Q_per_head.reserve(num_attention_heads_);
    K_per_head.reserve(num_key_value_heads_);
    V_per_head.reserve(num_key_value_heads_);
    for (int h = 0; h < num_attention_heads_; ++h) {
      auto q = q_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, /*ssa=*/true);
      Q_per_head.push_back(q);
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto k = k_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, /*ssa=*/true);
      auto v = v_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, /*ssa=*/true);
      K_per_head.push_back(k);
      V_per_head.push_back(v);
    }

    // RMSNorm + RoPE.
    for (int h = 0; h < num_attention_heads_; ++h) {
      Q_per_head[h] = rms_norm_q_[h](Q_per_head[h]);
      Q_per_head[h] = Q_per_head[h] * cos + rotateHalf(Q_per_head[h]) * sin;
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      K_per_head[h] = rms_norm_k_[h](K_per_head[h]);
      K_per_head[h] = K_per_head[h] * cos + rotateHalf(K_per_head[h]) * sin;
    }

    // -----------------------------------------------------------------------
    // KV cache side-output: emit new_key / new_value for the runner to
    // append. The runner is responsible for using past_key/past_value +
    // these to build the next dispatch's K_arranged/V_arranged via gather.
    // -----------------------------------------------------------------------
    // Emit ONLY the just-computed K/V (delta), not past+current concat, so
    // present_key/present_value sit at [1, Hkv, D, BQ] / [1, Hkv, BQ, D]. The
    // runner appends the delta to its CPU-side cache between dispatches.
    // (Mirrors modeling_qwen_qnn_aot_sha.hpp's new_key handling.) past_key/
    // past_value are still inputs in case future variants need them; they're
    // unused by the block-sparse attention path itself (which uses K_arranged
    // / V_arranged) — leaving them in keeps the runner-side I/O layout
    // identical to the standard non-block-sparse model.
    (void)past_key;
    (void)past_value;
    std::vector<Tensor> new_key_per_head;
    std::vector<Tensor> new_value_per_head;
    new_key_per_head.reserve(num_key_value_heads_);
    new_value_per_head.reserve(num_key_value_heads_);
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto k_h = K_per_head[h].transpose(2, 3);   // [B, 1, D, BQ]
      auto v_h = V_per_head[h];                   // [B, 1, BQ, D]
      new_key_per_head.push_back(k_h);
      new_value_per_head.push_back(v_h);
    }
    auto new_key   = nn::functional::concat(new_key_per_head, 1);    // [1, Hkv, D, BQ]
    auto new_value = nn::functional::concat(new_value_per_head, 1);  // [1, Hkv, BQ, D]

    // -----------------------------------------------------------------------
    // Attention compute on the GATHERED K/V arrangements.
    // Q is reshaped into [Hq, BQ, D] via concat across heads.
    // K_arranged / V_arranged are already [Hq, top_k*BK, D].
    //
    //   QK = MatMul(Q3, K_arranged, transpose_in1=true)   [Hq, BQ, top_k*BK]
    //   QKs = Mul(QK, scale)
    //   QKpm = Add(QKs, padding_mask)        broadcast across Hq, BQ
    //   QKm  = Add(QKpm, triangle_mask)      broadcast across Hq
    //   P = Softmax(QKm, axis=-1)
    //   O = MatMul(P, V_arranged)                          [Hq, BQ, D]
    // -----------------------------------------------------------------------
    // Per-head attention. For each attention head h:
    //   1. Slice this head's HISTORICAL portion out of K_arranged:
    //        K_hist_h shape [1, 1, (top_k-1)*BK, D]
    //   2. Take this dispatch's just-computed K/V for the corresponding
    //      kv-head (GQA: kv_head_idx = h / num_key_value_groups_):
    //        K_curr_h shape [1, 1, BQ, D]   (= post-RoPE K from K_per_head)
    //   3. Concat into the full attention K (diagonal slot is the last BQ rows):
    //        K_for_attn shape [1, 1, top_k*BK, D]
    //   4. Same for V.
    //   5. Run the existing matmul → scale → +padding_mask → +triangle_mask
    //      → softmax → matmul_av chain.
    //
    // This mirrors the standard dense model's concat({past_k_h, k_h}, -1)
    // trick: the current chunk's K/V is created inside the layer and
    // stitched in before attention. No two-pass dispatch, no chicken-and-egg.
    std::vector<Tensor> attn_outputs;
    attn_outputs.reserve(num_attention_heads_);

    auto scale_const = Tensor::constant(scale_, kFloat32);
    // Views of the two mask tensors lifted out of the loop (same broadcast
    // across all heads).
    auto pm  = padding_mask.view({1, 1, 1, kTopKBK}, /*ssa=*/true);      // [1,1,1,top_k*BK]
    auto tri = triangle_mask_().view({1, 1, kBQ, kTopKBK}, /*ssa=*/true);// [1,1,BQ,top_k*BK]

    for (int h = 0; h < num_attention_heads_; ++h) {
      const int kv_head_idx = h / num_key_value_groups_;
      auto q_h = Q_per_head[h].view({1, 1, kBQ, head_dim_}, /*ssa=*/true);  // [1, 1, BQ, D]

      // Historical K/V from runner-side gather.
      auto K_hist_h = K_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, true);  // [1, 1, (top_k-1)*BK, D]
      auto V_hist_h = V_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, true);  // [1, 1, (top_k-1)*BK, D]

      // Diagonal slot: this dispatch's own K/V (post-RoPE, post-RMSNorm),
      // sliced per-kv-head for GQA. K_per_head[kv_head_idx] is [1, 1, BQ, D].
      auto K_curr_h = K_per_head[kv_head_idx];
      auto V_curr_h = V_per_head[kv_head_idx];

      // Stitch: [1, 1, top_k*BK, D].
      auto K_full_h = nn::functional::concat({K_hist_h, K_curr_h}, -2);
      auto V_full_h = nn::functional::concat({V_hist_h, V_curr_h}, -2);

      auto K_full_h_T = K_full_h.transpose(-1, -2);             // [1, 1, D, top_k*BK]
      auto attn  = nn::functional::matmul(q_h, K_full_h_T);     // [1, 1, BQ, top_k*BK]
      attn = attn.mulConstant(scale_const);
      attn = attn + pm;                                          // pad-slot mask (historical region)
      attn = attn + tri;                                         // diagonal-slot triangle
      attn = nn::functional::softmax(attn, -1);
      auto y_h = nn::functional::matmul(attn, V_full_h);         // [1, 1, BQ, D]

      y_h = y_h.view({1, 1, kBQ, head_dim_}, /*ssa=*/true);
      attn_outputs.push_back(y_h);
    }

    // Concat heads + output projection.
    auto y = nn::functional::concat(attn_outputs, 1);                                          // [1, Hq, BQ, D]
    y = y.transpose(1, 2).view({1, 1, kBQ, num_attention_heads_ * head_dim_}, /*ssa=*/true);   // [1, 1, BQ, Hq*D]
    y = o_proj_(y).view({1, kBQ, hidden_size_}, true);                                          // [1, BQ, hidden]

    return {y, new_key, new_value};
  }

  int layer_idx_;
};

// ---------------------------------------------------------------------------
// Decoder block. Forwards K_arranged / V_arranged / padding_mask through to
// the attention.
// ---------------------------------------------------------------------------
class Qwen3CausalBlockSparseDecoder final : public nn::Module {
 public:
  int layer_idx_;
  Qwen3CausalBlockSparseAttention self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3CausalBlockSparseDecoder() = default;

  // NOTE: arg order matches nn::ModuleListWithIdx convention.
  Qwen3CausalBlockSparseDecoder(const std::string& name, const Qwen3Config& cfg, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;
    self_attn_ = reg<Qwen3CausalBlockSparseAttention>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    Tensor hidden_states  = inputs[0];
    Tensor sin            = inputs[1];
    Tensor cos            = inputs[2];
    Tensor past_key       = inputs[3];
    Tensor past_value     = inputs[4];
    Tensor K_arranged     = inputs[5];
    Tensor V_arranged     = inputs[6];
    Tensor padding_mask   = inputs[7];

    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto attn_out = self_attn_(hidden_states, sin, cos, past_key, past_value, K_arranged, V_arranged, padding_mask);
    hidden_states = residual + attn_out[0];

    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + hidden_states;

    return {hidden_states, attn_out[1], attn_out[2]};
  }
};

// ---------------------------------------------------------------------------
// Text model. Input layout (per dispatch — one qb):
//   inputs[0]                    : token ids        [1, BQ]
//   inputs[1]                    : position_ids     [1, BQ]
//   inputs[2 .. 2+L)             : past_key per layer
//   inputs[2+L .. 2+2L)          : past_value per layer
//   inputs[2+2L .. 2+3L)         : K_arranged per layer
//   inputs[2+3L .. 2+4L)         : V_arranged per layer
//   inputs[2+4L]                 : padding_mask  (one shared across layers — same per qb)
// ---------------------------------------------------------------------------
class Qwen3CausalBlockSparseText final : public nn::Module {
  nn::ModuleListWithIdx<Qwen3CausalBlockSparseDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;

 public:
  Qwen3CausalBlockSparseText() = default;

  Qwen3CausalBlockSparseText(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3CausalBlockSparseDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    auto x = embedding_(inputs[0]);
    const auto& position_ids = inputs[1];

    auto sin = nn::functional::gather(rope_sin_(), 1, position_ids);
    auto cos = nn::functional::gather(rope_cos_(), 1, position_ids);

    const int L = num_hidden_layers_;
    const auto& padding_mask = inputs[2 + 4 * L];  // shared across layers

    std::vector<Tensor> new_keys;
    std::vector<Tensor> new_values;
    new_keys.reserve(L);
    new_values.reserve(L);

    for (auto [idx, block] : enumerate(blocks)) {
      auto past_k = inputs[2 + idx];
      auto past_v = inputs[2 + L + idx];
      auto K_arr  = inputs[2 + 2 * L + idx];
      auto V_arr  = inputs[2 + 3 * L + idx];

      auto out = block(x, sin, cos, past_k, past_v, K_arr, V_arr, padding_mask);
      x = out[0];
      new_keys.push_back(out[1]);
      new_values.push_back(out[2]);
    }

    x = norm_(x).view({1, 1, -1, hidden_size_}, true);

    std::vector<Tensor> ret = {x};
    ret.insert(ret.end(), new_keys.begin(), new_keys.end());
    ret.insert(ret.end(), new_values.begin(), new_values.end());
    return ret;
  }
};

// ---------------------------------------------------------------------------
// CausalLM wrapper.
// ---------------------------------------------------------------------------
class Qwen3ForCausalLM_CausalBlockSparse : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLM_CausalBlockSparse(const Qwen3Config& cfg) : cfg_(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<Qwen3CausalBlockSparseText>("model", cfg);

    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg.hidden_size, cfg.vocab_size, /*bias=*/false);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::IRContext::ptr_t llm_ir = nullptr;
    auto sequence = input.at("sequence");

    const int L = cfg_.num_hidden_layers;
    std::vector<Tensor> kv_caches;
    std::vector<Tensor> kv_arranged;

    for (int i = 0; i < L; ++i) {
      auto name = "past_key_" + std::to_string(i);
      if (!input.count(name)) throw std::runtime_error("Missing " + name);
      kv_caches.push_back(input.at(name));
    }
    for (int i = 0; i < L; ++i) {
      auto name = "past_value_" + std::to_string(i);
      if (!input.count(name)) throw std::runtime_error("Missing " + name);
      kv_caches.push_back(input.at(name));
    }
    for (int i = 0; i < L; ++i) {
      auto name = "k_arranged_" + std::to_string(i);
      if (!input.count(name)) throw std::runtime_error("Missing " + name);
      kv_arranged.push_back(input.at(name));
    }
    for (int i = 0; i < L; ++i) {
      auto name = "v_arranged_" + std::to_string(i);
      if (!input.count(name)) throw std::runtime_error("Missing " + name);
      kv_arranged.push_back(input.at(name));
    }
    if (!input.count("padding_mask")) throw std::runtime_error("Missing padding_mask");
    auto padding_mask = input.at("padding_mask");

    auto seq_len = sequence.shape()[1];
    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
    } else {
      position_ids = Tensor::empty({seq_len}, kInt32, kCPU).alloc();
      auto p = position_ids.ptr<int32_t>();
      for (int s = 0; s < seq_len; ++s) p[s] = s;
    }

    ir::lowlevel::traceStart();

    std::vector<Tensor> llm_inputs = {sequence, position_ids};
    llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());
    llm_inputs.insert(llm_inputs.end(), kv_arranged.begin(), kv_arranged.end());
    llm_inputs.push_back(padding_mask);

    sequence = llm_(llm_inputs)[0];
    sequence = lm_head_(sequence);

    llm_ir = ir::lowlevel::traceStop();
    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    return {};  // dispatched via QNN runtime, not the host forward path
  }

 private:
  const Qwen3Config cfg_;
  Qwen3CausalBlockSparseText llm_;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
};

// ---------------------------------------------------------------------------
// Build the static triangle mask [1, BQ, top_k*BK] as an fp16 Tensor and
// inject it into params under "model.causal_triangle_mask".
//
// Pattern (matches buildStaticTriangleMask in tests/qnn/
// BlockSparseAttentionCausalTest.cpp):
//   slots 0..top_k-2:  all zeros
//   diagonal slot (top_k-1): column c has value 0 if c <= q else kMaskNeg
// ---------------------------------------------------------------------------
inline void bakeCausalTriangleMask(const ParameterFile::ptr_t& params) {
  const int top_k_BK = kTopKBK;
  const int diag_off = (kTopK - 1) * kBK;
  std::vector<float> buf((size_t)kBQ * top_k_BK, 0.0f);
  for (int q = 0; q < kBQ; ++q) {
    for (int s = 0; s < kBK; ++s) {
      buf[(size_t)q * top_k_BK + diag_off + s] = (s <= q) ? 0.0f : kMaskNeg;
    }
  }
  auto t = Tensor::fromVector(buf, {1, kBQ, top_k_BK}, kFloat32).to(kFloat16);
  params->push("model.causal_triangle_mask",
               t.contiguous().setMemType(kParamsNormal).setName("model.causal_triangle_mask"));
}

// ---------------------------------------------------------------------------
// Parameter slicing for SHA — copied from the non-causal blocksparse file.
// ---------------------------------------------------------------------------
inline void prepareParametersForSHA_FP16(const ParameterFile::ptr_t& params, const Qwen3Config& cfg) {
  int num_heads = cfg.num_attention_heads;
  int num_kv_heads = cfg.num_key_value_heads;
  int head_dim = cfg.head_dim;
  int num_layers = cfg.num_hidden_layers;

  auto sliceLinear = [&](const std::string& orig_prefix, const std::string& new_prefix, int per_head_out,
                         int n_splits) {
    std::string orig_name = orig_prefix + ".weight";
    if (!params->has(orig_name)) return;
    auto orig = params->pull(orig_name);
    for (int h = 0; h < n_splits; ++h) {
      std::string new_name = new_prefix + "." + std::to_string(h) + ".weight";
      int start = h * per_head_out;
      int end = (h + 1) * per_head_out;
      auto sliced = orig.slice({{start, end}, kAll}, false);
      params->push(new_name, sliced.contiguous().setMemType(kParamsNormal).setName(new_name));
    }
  };

  for (int layer = 0; layer < num_layers; ++layer) {
    std::string p = "model.layers." + std::to_string(layer) + ".self_attn.";
    sliceLinear(p + "q_proj", p + "q_proj", head_dim, num_heads);
    sliceLinear(p + "k_proj", p + "k_proj", head_dim, num_kv_heads);
    sliceLinear(p + "v_proj", p + "v_proj", head_dim, num_kv_heads);

    auto replicateNorm = [&](const std::string& name, int count) {
      std::string orig_name = p + name + ".weight";
      if (!params->has(orig_name)) return;
      auto orig = params->pull(orig_name);
      for (int h = 0; h < count; ++h) {
        std::string new_name = p + name + "." + std::to_string(h) + ".weight";
        params->push(new_name, orig.contiguous().setMemType(kParamsNormal).setName(new_name));
      }
    };
    replicateNorm("q_norm", num_heads);
    replicateNorm("k_norm", num_kv_heads);
  }
}

}  // namespace mllm::models::qwen3::sha_fp16_blocksparse_causal
