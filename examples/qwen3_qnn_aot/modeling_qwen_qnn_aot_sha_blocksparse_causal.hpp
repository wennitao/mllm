// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// LPBQ SHA + per-qb CAUSAL block-sparse Qwen3 model.
//
// Combines the proven LPBQ SHA infrastructure (Conv2D w4a16 projections, per-head
// RMSNorm, QDQ chains, uint8-sym KV cache) from modeling_qwen_qnn_aot_sha.hpp /
// modeling_qwen_qnn_aot_sha_blocksparse.hpp with the per-qb causal block-sparse
// attention pattern from modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp.
//
// Differences from sha_blocksparse:
//   * chunk_size = kBQ = 32 (one q-block per graphExecute), num_q_blocks = 1.
//   * K_arranged / V_arranged carry only the (top_k - 1) HISTORICAL slots —
//     the diagonal slot's K/V is computed inside the layer from this qb's
//     own RoPE'd K/V and concat'd in. This mirrors the non-quantized
//     fp16_blocksparse_causal variant.
//   * past_key / past_value are NOT consumed by attention (the runner gathers
//     from them on CPU into K_arranged / V_arranged) but they're still graph
//     inputs because the runner registers the same cache buffers and we don't
//     want to invent a separate runtime layout. They go unused by the layer.
//   * A single combined uint16-quant `mask` graph input replaces dense
//     `causal_mask`. Shape [1, 1, kBQ, kTopKBK]. Runner builds it per qb:
//       - For historical slots that are real history (qb_global > slot):
//         all entries = 65535 (ACTIVE).
//       - For padding historical slots: all entries = 0 (MASKED).
//       - For the diagonal slot (last kBK columns): row q col c gets 65535
//         if c <= q else 0 (causal triangle).
//
// Attention math (per Q head h):
//   q_h        [1, 1, kBQ, head_dim]                (post-RoPE, uint16)
//   K_hist_h   [1, 1, head_dim, (top_k-1)*kBK]      (slice K_arranged, uint8)
//   K_curr_h   [1, 1, head_dim, kBQ]                (this qb's K, uint8, transposed)
//   K_full_h   [1, 1, head_dim, kTopKBK]            (concat on dim -1)
//   attn       [1, 1, kBQ, kTopKBK]                 (matmul(q_h, K_full_h))
//   attn       = mulConstant(attn, 1/sqrt(D))
//   attn       = where(mask == 0_dequantized, attn, attn_min - 20)
//   P          = softmax(attn, -1)
//   V_hist_h   [1, 1, (top_k-1)*kBK, head_dim]      (slice V_arranged, uint8)
//   V_curr_h   [1, 1, kBQ, head_dim]                (this qb's V, uint8)
//   V_full_h   [1, 1, kTopKBK, head_dim]            (concat on dim 2)
//   y_h        [1, 1, kBQ, head_dim]                (matmul(P, V_full_h))
//
// The runner is responsible for:
//   * Gathering historical K/V blocks from past_key/past_value into
//     K_arranged/V_arranged before each graphExecute (matches the pattern
//     in BlockSparsePromptProcessor for the fp16 variant, just uint8 dtype).
//   * Building the per-qb combined mask.

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

#include "modeling_qwen_qnn_aot_sha.hpp"  // sha::ptq, sha::Qwen3MLP, sha::rotateHalf, sha::prepareParametersForSHA

namespace mllm::models::qwen3::sha_blocksparse_causal {

namespace ptq = mllm::models::qwen3::sha::ptq;
using mllm::models::qwen3::sha::Qwen3MLP;
using mllm::models::qwen3::sha::rotateHalf;

using vi32 = std::vector<int32_t>;
#define BSC_CONV2D_PROPERTY vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G16

constexpr int kBQ = 32;
constexpr int kBK = 32;
constexpr int kTopK = 8;
constexpr int kTopKBK = kTopK * kBK;       // 256 — full attention K width per qb
constexpr int kHistKBK = (kTopK - 1) * kBK;  // 224 — runner-supplied historical slots

class Qwen3AttentionSHABlockSparseCausal final : public nn::Module {
  std::vector<nn::Conv2D> q_projs_;
  std::vector<nn::Conv2D> k_projs_;
  std::vector<nn::Conv2D> v_projs_;
  nn::Conv2D o_proj_;

  std::vector<nn::RMSNorm> rms_norm_q_;
  std::vector<nn::RMSNorm> rms_norm_k_;

  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;

 public:
  int layer_idx_ = 0;

  Qwen3AttentionSHABlockSparseCausal() = default;

  Qwen3AttentionSHABlockSparseCausal(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = (1.f / sqrtf((float)head_dim_));

    for (int h = 0; h < num_attention_heads_; ++h) {
      q_projs_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden_size_, head_dim_, BSC_CONV2D_PROPERTY));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      k_projs_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(h), hidden_size_, head_dim_, BSC_CONV2D_PROPERTY));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      v_projs_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(h), hidden_size_, head_dim_, BSC_CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, BSC_CONV2D_PROPERTY);

    for (int h = 0; h < num_attention_heads_; ++h) {
      rms_norm_q_.emplace_back(reg<nn::RMSNorm>("q_norm." + std::to_string(h), cfg.rms_norm_eps));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      rms_norm_k_.emplace_back(reg<nn::RMSNorm>("k_norm." + std::to_string(h), cfg.rms_norm_eps));
    }

    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  // inputs:
  //   0  hidden_states  [B=1, kBQ, hidden_size]
  //   1  sin            [B=1, kBQ, head_dim]
  //   2  cos            [B=1, kBQ, head_dim]
  //   3  mask           [1, 1, kBQ, kTopKBK]            uint16, runner per-qb
  //   4  K_arranged     [Hq, 1, head_dim, kHistKBK]     uint8 — historical only
  //   5  V_arranged     [Hq, 1, kHistKBK, head_dim]     uint8 — historical only
  // No past_key / past_value: the runner gathers historical K/V from its
  // CPU-side cache into K_arranged / V_arranged (any tensor that's a graph
  // input but is never consumed by an op never gets its quant spec solved,
  // and lowering crashes on the unsolved scale).
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto sin = inputs[1];
    auto cos = inputs[2];
    auto mask = inputs[3];
    auto K_arranged = inputs[4];
    auto V_arranged = inputs[5];

    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // Per-head Q/K/V projections.
    std::vector<Tensor> query_states_per_head;
    for (int h = 0; h < num_attention_heads_; ++h) {
      auto q_h = q_projs_[h](hidden_states);
      q_h = q_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      query_states_per_head.push_back(q_h);
    }
    std::vector<Tensor> key_states_per_head;
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto k_h = k_projs_[h](hidden_states);
      k_h = k_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      key_states_per_head.push_back(k_h);
    }
    std::vector<Tensor> value_states_per_head;
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto v_h = v_projs_[h](hidden_states);
      v_h = v_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      value_states_per_head.push_back(v_h);
    }

    // Per-head RMSNorm + RoPE. sin/cos arrive already 4D (1, 1, S, D) from
    // the parent's gather — position_ids is rank-2 [1, BQ] so gather output
    // is data[:1] + indices.shape + data[2:] = [1, 1, BQ, D]. No unsqueeze
    // (the standard SHA model passes rank-1 position_ids and gets rank-3
    // gather output, so it unsqueezes; we don't).

    for (int h = 0; h < num_attention_heads_; ++h) {
      std::string hs = std::to_string(h);
      query_states_per_head[h] = rms_norm_q_[h](ptq::QDQ(this, query_states_per_head[h], "q_norm_input_qdq_h" + hs));
      query_states_per_head[h] = ptq::QDQ(this, query_states_per_head[h], "q_norm_output_qdq_h" + hs);
      query_states_per_head[h] =
          ptq::QDQ(this,
                   ptq::QDQ(this, query_states_per_head[h] * cos, "q_rope_mul_0_output_qdq_h" + hs)
                       + ptq::QDQ(this, rotateHalf(query_states_per_head[h], this, "q_rope_neg_half_qdq_h" + hs) * sin,
                                  "q_rope_mul_1_output_qdq_h" + hs),
                   "q_rope_add_0_output_qdq_h" + hs);
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      std::string hs = std::to_string(h);
      key_states_per_head[h] = rms_norm_k_[h](ptq::QDQ(this, key_states_per_head[h], "k_norm_input_qdq_h" + hs));
      key_states_per_head[h] = ptq::QDQ(this, key_states_per_head[h], "k_norm_output_qdq_h" + hs);
      key_states_per_head[h] =
          ptq::QDQ(this,
                   ptq::QDQ(this, key_states_per_head[h] * cos, "k_rope_mul_0_output_qdq_h" + hs)
                       + ptq::QDQ(this, rotateHalf(key_states_per_head[h], this, "k_rope_neg_half_qdq_h" + hs) * sin,
                                  "k_rope_mul_1_output_qdq_h" + hs),
                   "k_rope_add_0_output_qdq_h" + hs);
    }

    // KV cache update outputs + uint8-sym K/V for in-layer concat.
    // Quantize K to uint8 (transposed [1,1,D,kBQ] to match cache layout).
    // Quantize V to uint8 ([1,1,kBQ,D]). These are both the runner's
    // present_key/present_value AND the "current" slot for the in-layer
    // attention concat below.
    std::vector<Tensor> new_key_per_head;
    std::vector<Tensor> new_value_per_head;
    std::vector<Tensor> k_curr_uint8_per_head;
    std::vector<Tensor> v_curr_uint8_per_head;
    new_key_per_head.reserve(num_key_value_heads_);
    new_value_per_head.reserve(num_key_value_heads_);
    k_curr_uint8_per_head.reserve(num_key_value_heads_);
    v_curr_uint8_per_head.reserve(num_key_value_heads_);
    for (int h = 0; h < num_key_value_heads_; ++h) {
      std::string hs = std::to_string(h);
      auto k_h = key_states_per_head[h].to(kUInt8PerTensorSym);
      k_h = ptq::QDQ_KV(this, k_h, "k_cast_to_int8_qdq_h" + hs);
      k_h = k_h.transpose(2, 3);  // [1,1,D,kBQ]
      auto v_h = ptq::QDQ(this, value_states_per_head[h], "v_cast_to_int16_qdq_h" + hs);
      v_h = v_h.to(kUInt8PerTensorSym);
      v_h = ptq::QDQ_KV(this, v_h, "v_cast_to_int8_qdq_h" + hs);
      new_key_per_head.push_back(k_h);
      new_value_per_head.push_back(v_h);
      k_curr_uint8_per_head.push_back(k_h);
      v_curr_uint8_per_head.push_back(v_h);
    }

    // Per-head causal block-sparse attention. GQA: each Q head uses its
    // corresponding KV head (kv_head_idx = h / num_kv_groups_).
    std::vector<Tensor> attn_outputs;
    attn_outputs.reserve(num_attention_heads_);
    for (int h = 0; h < num_attention_heads_; ++h) {
      std::string hs = std::to_string(h);
      const int kv_head_idx = h / num_key_value_groups_;
      const auto& q_h = query_states_per_head[h];

      // Historical K/V (uint8, runner-gathered) for THIS Q head.
      auto K_hist_h = K_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, /*ssa=*/true);  // [1,1,D,kHistKBK]
      auto V_hist_h = V_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, /*ssa=*/true);  // [1,1,kHistKBK,D]

      // Diagonal slot K/V — this dispatch's just-computed values.
      auto K_curr_h = k_curr_uint8_per_head[kv_head_idx];  // [1,1,D,kBQ]
      auto V_curr_h = v_curr_uint8_per_head[kv_head_idx];  // [1,1,kBQ,D]

      // Stitch: full attention K/V per head.
      auto K_full_h = nn::functional::concat({K_hist_h, K_curr_h}, -1);  // [1,1,D,kTopKBK]
      auto V_full_h = nn::functional::concat({V_hist_h, V_curr_h}, 2);   // [1,1,kTopKBK,D]

      // QK^T → scale → masked softmax → P @ V.
      auto attn = ptq::QDQ(this, nn::functional::matmul(q_h, K_full_h), "qk_matmul_output_qdq_h" + hs);

      auto scale = Tensor::constant(scale_, kFloat32);
      scale = ptq::QDQ(this, scale, "scaling_qdq_h" + hs);
      attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq_h" + hs);

      // Masked softmax via "min - 20" surrogate for -inf (matches sha.hpp).
      auto attn_min = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq_h" + hs);
      auto minus_value = Tensor::constant(-20, kFloat32);
      minus_value = ptq::QDQ(this, minus_value, "neg_20_qdq_h" + hs);
      auto attn_vv = ptq::QDQ(this, attn_min.addConstant(minus_value), "minus_0_output_qdq_h" + hs);
      auto zero_constant = Tensor::constant(0.f, kFloat32);
      zero_constant = ptq::QDQ_CONSTANT(this, zero_constant, "constant_zero");
      // mask == 0 (real, dequantized) → ACTIVE (take attn); != 0 → MASKED (take attn_vv).
      // Runner writes mask = 65535 for ACTIVE (zp = 65535 → real = 0) and 0 for MASKED.
      attn = nn::functional::where(mask.equalConstant(zero_constant), attn, attn_vv);
      attn = ptq::QDQ(this, attn, "where_attn_qdq_h" + hs);
      attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq_h" + hs);

      auto y_h = ptq::QDQ(this, nn::functional::matmul(attn, V_full_h), "attn_value_matmul_output_qdq_h" + hs);
      attn_outputs.push_back(y_h);
    }

    // Concat heads + O projection.
    auto y = nn::functional::concat(attn_outputs, 1);                                            // [1,Hq,kBQ,D]
    y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);      // [1,1,kBQ,Hq*D]
    y = o_proj_(y).view({1, -1, hidden_size_}, true);                                            // [1,kBQ,hidden]

    auto new_key = nn::functional::concat(new_key_per_head, 1);
    auto new_value = nn::functional::concat(new_value_per_head, 1);
    return {y, new_key, new_value};
  }
};

class Qwen3DecoderSHABlockSparseCausal final : public nn::Module {
 public:
  int layer_idx_;
  Qwen3AttentionSHABlockSparseCausal self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3DecoderSHABlockSparseCausal() = default;

  Qwen3DecoderSHABlockSparseCausal(const std::string& name, const Qwen3Config& cfg, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;
    self_attn_ = reg<Qwen3AttentionSHABlockSparseCausal>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto sin = inputs[1];
    auto cos = inputs[2];
    auto mask = inputs[3];
    auto K_arranged = inputs[4];
    auto V_arranged = inputs[5];

    if (layer_idx_ != 0) { hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq"); }
    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto _ = self_attn_(hidden_states, sin, cos, mask, K_arranged, V_arranged);
    hidden_states = _[0];
    hidden_states = ptq::QDQ(this, residual + ptq::QDQ(this, hidden_states, "add_0_lhs_input_qdq"), "add_0_output_qdq");
    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + ptq::QDQ(this, hidden_states, "add_1_lhs_input_qdq");
    return {hidden_states, _[1], _[2]};
  }
};

class Qwen3TextSHABlockSparseCausal final : public nn::Module {
  nn::ModuleListWithIdx<Qwen3DecoderSHABlockSparseCausal> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;

 public:
  Qwen3TextSHABlockSparseCausal() = default;

  Qwen3TextSHABlockSparseCausal(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3DecoderSHABlockSparseCausal>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
  }

  // Graph inputs (built by the compile-side trace driver):
  //   0          sequence       [1, kBQ] int32
  //   1          position_ids   [1, kBQ] int32
  //   2          mask           [1, 1, kBQ, kTopKBK] uint16
  //   3..3+L-1   k_arranged_i   [Hq, 1, D, kHistKBK] uint8
  //   3+L..      v_arranged_i   [Hq, 1, kHistKBK, D] uint8
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    auto x = embedding_(inputs[0]);
    const auto& position_ids = inputs[1];
    auto mask = inputs[2];

    // clang-format off
    auto sin = nn::functional::gather(ptq::QDQ_ROPE(this, rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto cos = nn::functional::gather(ptq::QDQ_ROPE(this, rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);
    // clang-format on

    const int L = num_hidden_layers_;
    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    for (auto [index, block] : enumerate(blocks)) {
      auto k_arr = inputs[3 + index];
      auto v_arr = inputs[3 + L + index];
      auto _ = block(x, sin, cos, mask, k_arr, v_arr);
      x = _[0];
      keys.push_back(_[1]);
      values.push_back(_[2]);
    }

    x = norm_(ptq::QDQ(this, x, "norm_input_qdq"));
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto ret = std::vector<Tensor>{x};
    for (const auto& item : keys) { ret.push_back(item); }
    for (const auto& item : values) { ret.push_back(item); }
    return ret;
  }
};

class Qwen3ForCausalLM_SHABlockSparseCausal : public ARGeneration, public nn::Module {
 public:
  Qwen3ForCausalLM_SHABlockSparseCausal(const Qwen3Config& cfg) : cfg_(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<Qwen3TextSHABlockSparseCausal>("model", cfg);
    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, BSC_CONV2D_PROPERTY);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::IRContext::ptr_t llm_ir = nullptr;

    auto sequence = input.at("sequence");
    auto mask = input.at("mask");

    const int L = cfg_.num_hidden_layers;
    std::vector<Tensor> k_arrs, v_arrs;
    k_arrs.reserve(L); v_arrs.reserve(L);
    for (int i = 0; i < L; ++i) {
      auto n = "k_arranged_" + std::to_string(i);
      if (!input.count(n)) throw std::runtime_error("Missing " + n);
      k_arrs.push_back(input.at(n));
    }
    for (int i = 0; i < L; ++i) {
      auto n = "v_arranged_" + std::to_string(i);
      if (!input.count(n)) throw std::runtime_error("Missing " + n);
      v_arrs.push_back(input.at(n));
    }

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

    std::vector<Tensor> llm_inputs = {sequence, position_ids, mask};
    llm_inputs.insert(llm_inputs.end(), k_arrs.begin(), k_arrs.end());
    llm_inputs.insert(llm_inputs.end(), v_arrs.begin(), v_arrs.end());

    sequence = llm_(llm_inputs)[0];
    sequence = lm_head_(ptq::QDQ(this, sequence, "lm_head_input_qdq"));
    sequence = ptq::QDQ(this, sequence, "lm_head_output_qdq");

    llm_ir = ir::lowlevel::traceStop();
    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

 private:
  const Qwen3Config& cfg_;
  Qwen3TextSHABlockSparseCausal llm_;
  nn::Conv2D lm_head_;
  bool tie_word_embeddings_;
};

#undef BSC_CONV2D_PROPERTY

}  // namespace mllm::models::qwen3::sha_blocksparse_causal
