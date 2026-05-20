// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Block-sparse variant of the SHA Qwen3 model.
//
// Differences from modeling_qwen_qnn_aot_sha.hpp:
//   * Adds two extra graph inputs per layer — `k_arranged_i` and `v_arranged_i`
//     of shape [Hq, num_q_blocks, top_k * BK, head_dim], fp16. The runner
//     selects (XAttention) and gathers these on CPU prior to the call.
//   * The per-head attention compute reads from K_arranged/V_arranged instead
//     of `concat(past_k, k_h)` / `concat(past_v, v_h)`. No causal mask is
//     applied inside attention — block selection produces the sparsity
//     pattern.
//   * `new_key` / `new_value` outputs are still produced (uint8 sym KV layout)
//     so the runner can write the just-computed K/V into its KV cache region
//     for the next chunk.
//
// Everything else (Conv2D LPBQ projections, per-head RMSNorm, RoPE, the QDQ
// chain) is identical to the SHA implementation.

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

#include "modeling_qwen_qnn_aot_sha.hpp"  // pulls in sha::ptq, sha::Qwen3MLP, sha::rotateHalf, sha::prepareParametersForSHA

namespace mllm::models::qwen3::sha_blocksparse {

namespace ptq = mllm::models::qwen3::sha::ptq;
using mllm::models::qwen3::sha::Qwen3MLP;
using mllm::models::qwen3::sha::rotateHalf;

using vi32 = std::vector<int32_t>;
#define BS_CONV2D_PROPERTY vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G16

// -----------------------------------------------------------------------------
// Block-sparse constants. Mirror the kernel-side sweep; chunk_size at trace
// time must be a multiple of kBQ.
// -----------------------------------------------------------------------------
constexpr int kBQ = 32;
constexpr int kBK = 32;
constexpr int kTopK = 8;
constexpr int kTopKBK = kTopK * kBK;  // 256

class Qwen3AttentionSHABlockSparse final : public nn::Module {
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

  int chunk_size_;
  int num_q_blocks_;

 public:
  int layer_idx_ = 0;

  Qwen3AttentionSHABlockSparse() = default;

  Qwen3AttentionSHABlockSparse(const std::string& name, const Qwen3Config& cfg, int chunk_size) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = (1.f / sqrtf((float)head_dim_));

    chunk_size_ = chunk_size;
    num_q_blocks_ = chunk_size / kBQ;
    MLLM_RT_ASSERT_EQ(chunk_size % kBQ, 0);

    for (int h = 0; h < num_attention_heads_; ++h) {
      q_projs_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden_size_, head_dim_, BS_CONV2D_PROPERTY));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      k_projs_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(h), hidden_size_, head_dim_, BS_CONV2D_PROPERTY));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      v_projs_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(h), hidden_size_, head_dim_, BS_CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, BS_CONV2D_PROPERTY);

    for (int h = 0; h < num_attention_heads_; ++h) {
      rms_norm_q_.emplace_back(reg<nn::RMSNorm>("q_norm." + std::to_string(h), cfg.rms_norm_eps));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      rms_norm_k_.emplace_back(reg<nn::RMSNorm>("k_norm." + std::to_string(h), cfg.rms_norm_eps));
    }

    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  // inputs:
  //   0  hidden_states           [B=1, S, hidden_size]
  //   1  llm_embedding_sin       [B=1, S, head_dim]
  //   2  llm_embedding_cos       [B=1, S, head_dim]
  //   3  K_arranged              [Hq, num_q_blocks, kTopKBK, head_dim]   (fp16)
  //   4  V_arranged              [Hq, num_q_blocks, kTopKBK, head_dim]   (fp16)
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto K_arranged = inputs[3];
    auto V_arranged = inputs[4];

    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // -------------------------------------------------------------------------
    // Per-head Q/K/V projections (identical to SHA).
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Per-head RMSNorm + RoPE (identical to SHA).
    // -------------------------------------------------------------------------
    auto cos = llm_embedding_cos.unsqueeze(1, true);
    auto sin = llm_embedding_sin.unsqueeze(1, true);

    for (int h = 0; h < num_attention_heads_; ++h) {
      std::string h_str = std::to_string(h);
      query_states_per_head[h] = rms_norm_q_[h](ptq::QDQ(this, query_states_per_head[h], "q_norm_input_qdq_h" + h_str));
      query_states_per_head[h] = ptq::QDQ(this, query_states_per_head[h], "q_norm_output_qdq_h" + h_str);
      query_states_per_head[h] =
          ptq::QDQ(this,
                   ptq::QDQ(this, query_states_per_head[h] * cos, "q_rope_mul_0_output_qdq_h" + h_str)
                       + ptq::QDQ(this, rotateHalf(query_states_per_head[h], this, "q_rope_neg_half_qdq_h" + h_str) * sin,
                                  "q_rope_mul_1_output_qdq_h" + h_str),
                   "q_rope_add_0_output_qdq_h" + h_str);
    }

    for (int h = 0; h < num_key_value_heads_; ++h) {
      std::string h_str = std::to_string(h);
      key_states_per_head[h] = rms_norm_k_[h](ptq::QDQ(this, key_states_per_head[h], "k_norm_input_qdq_h" + h_str));
      key_states_per_head[h] = ptq::QDQ(this, key_states_per_head[h], "k_norm_output_qdq_h" + h_str);
      key_states_per_head[h] =
          ptq::QDQ(this,
                   ptq::QDQ(this, key_states_per_head[h] * cos, "k_rope_mul_0_output_qdq_h" + h_str)
                       + ptq::QDQ(this, rotateHalf(key_states_per_head[h], this, "k_rope_neg_half_qdq_h" + h_str) * sin,
                                  "k_rope_mul_1_output_qdq_h" + h_str),
                   "k_rope_add_0_output_qdq_h" + h_str);
    }

    // -------------------------------------------------------------------------
    // KV-cache update outputs (identical to SHA, for runner cache write).
    // We do NOT read from past_key/past_value here; attention reads from
    // K_arranged/V_arranged.
    // -------------------------------------------------------------------------
    std::vector<Tensor> new_key_per_head;
    std::vector<Tensor> new_value_per_head;

    for (int h = 0; h < num_key_value_heads_; ++h) {
      std::string h_str = std::to_string(h);
      auto k_h = key_states_per_head[h].to(kUInt8PerTensorSym);
      k_h = ptq::QDQ_KV(this, k_h, "k_cast_to_int8_qdq_h" + h_str);
      k_h = k_h.transpose(2, 3);  // [B, 1, D, S]

      auto v_h = ptq::QDQ(this, value_states_per_head[h], "v_cast_to_int16_qdq_h" + h_str);
      v_h = v_h.to(kUInt8PerTensorSym);
      v_h = ptq::QDQ_KV(this, v_h, "v_cast_to_int8_qdq_h" + h_str);

      new_key_per_head.push_back(k_h);
      new_value_per_head.push_back(v_h);
    }

    // -------------------------------------------------------------------------
    // Block-sparse attention compute. Per Q head:
    //   q_h reshape [1, num_q_blocks, kBQ, head_dim]
    //   K_h = K_arranged[h]   [1, num_q_blocks, kTopKBK, head_dim]
    //   V_h = V_arranged[h]   [1, num_q_blocks, kTopKBK, head_dim]
    //   attn = matmul(q_h, K_h^T) * scale            [1, NQ, kBQ, kTopKBK]
    //   attn = softmax(attn, -1)
    //   y_h  = matmul(attn, V_h)                     [1, NQ, kBQ, head_dim]
    //   y_h  = view [1, 1, S, head_dim]
    // No causal mask here — block selection produces the sparsity pattern.
    // -------------------------------------------------------------------------
    std::vector<Tensor> attn_outputs;
    for (int h = 0; h < num_attention_heads_; ++h) {
      std::string h_str = std::to_string(h);

      auto q_h = query_states_per_head[h].view({1, num_q_blocks_, kBQ, head_dim_}, /*ssa=*/true);

      auto K_h = K_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, /*ssa=*/true);
      auto V_h = V_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, /*ssa=*/true);
      auto K_h_T = K_h.transpose(-1, -2);

      auto attn = ptq::QDQ(this, nn::functional::matmul(q_h, K_h_T), "qk_matmul_output_qdq_h" + h_str);

      auto scale = Tensor::constant(scale_, kFloat32);
      scale = ptq::QDQ(this, scale, "scaling_qdq_h" + h_str);
      attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq_h" + h_str);

      attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq_h" + h_str);

      auto y_h = ptq::QDQ(this, nn::functional::matmul(attn, V_h), "attn_value_matmul_output_qdq_h" + h_str);
      y_h = y_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);
      attn_outputs.push_back(y_h);
    }

    // -------------------------------------------------------------------------
    // Concat head outputs → O projection (identical to SHA).
    // -------------------------------------------------------------------------
    auto y = nn::functional::concat(attn_outputs, 1);
    y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);
    y = o_proj_(y).view({1, -1, hidden_size_}, true);

    auto new_key = nn::functional::concat(new_key_per_head, 1);
    auto new_value = nn::functional::concat(new_value_per_head, 1);

    return {y, new_key, new_value};
  }
};

class Qwen3DecoderSHABlockSparse final : public nn::Module {
 public:
  int layer_idx_;
  Qwen3AttentionSHABlockSparse self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3DecoderSHABlockSparse() = default;

  // NOTE: matches nn::ModuleListWithIdx convention (idx is appended at the end).
  Qwen3DecoderSHABlockSparse(const std::string& name, const Qwen3Config& cfg, int chunk_size, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;
    self_attn_ = reg<Qwen3AttentionSHABlockSparse>("self_attn", cfg, chunk_size);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto k_arranged = inputs[3];
    auto v_arranged = inputs[4];

    auto hidden_states = inputs[0];
    if (layer_idx_ != 0) { hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq"); }
    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto _ = self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, k_arranged, v_arranged);
    hidden_states = _[0];
    hidden_states = ptq::QDQ(this, residual + ptq::QDQ(this, hidden_states, "add_0_lhs_input_qdq"), "add_0_output_qdq");
    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + ptq::QDQ(this, hidden_states, "add_1_lhs_input_qdq");
    return {hidden_states, _[1], _[2]};
  }
};

class Qwen3TextSHABlockSparse final : public nn::Module {
  nn::ModuleListWithIdx<Qwen3DecoderSHABlockSparse> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;

 public:
  Qwen3TextSHABlockSparse() = default;

  Qwen3TextSHABlockSparse(const std::string& name, const Qwen3Config& cfg, int chunk_size) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3DecoderSHABlockSparse>>("layers", cfg.num_hidden_layers, cfg, chunk_size);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
  }

  // Graph inputs (built by the compile-side trace driver):
  //   0          sequence
  //   1          position_ids
  //   2..2+L-1   k_arranged_i
  //   2+L..2+2L-1 v_arranged_i
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    auto x = embedding_(inputs[0]);
    const auto& position_ids = inputs[1];

    // clang-format off
    auto llm_embedding_sin = nn::functional::gather(ptq::QDQ_ROPE(this, rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto llm_embedding_cos = nn::functional::gather(ptq::QDQ_ROPE(this, rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);
    // clang-format on

    const int L = num_hidden_layers_;
    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    for (auto [index, block] : enumerate(blocks)) {
      auto k_arr = inputs[2 + index];
      auto v_arr = inputs[2 + L + index];
      auto _ = block(x, llm_embedding_sin, llm_embedding_cos, k_arr, v_arr);
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

class Qwen3ForCausalLM_SHABlockSparse : public ARGeneration, public nn::Module {
 public:
  Qwen3ForCausalLM_SHABlockSparse(const Qwen3Config& cfg, int chunk_size) : cfg_(cfg), chunk_size_(chunk_size) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<Qwen3TextSHABlockSparse>("model", cfg, chunk_size);
    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, BS_CONV2D_PROPERTY);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::IRContext::ptr_t llm_ir = nullptr;

    auto sequence = input.at("sequence");

    std::vector<Tensor> k_arranged_inputs;
    std::vector<Tensor> v_arranged_inputs;
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
      auto kn = "k_arranged_" + std::to_string(i);
      if (!input.count(kn)) { throw std::runtime_error("Missing k_arranged for layer " + std::to_string(i)); }
      k_arranged_inputs.push_back(input.at(kn));
    }
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
      auto vn = "v_arranged_" + std::to_string(i);
      if (!input.count(vn)) { throw std::runtime_error("Missing v_arranged for layer " + std::to_string(i)); }
      v_arranged_inputs.push_back(input.at(vn));
    }

    auto seq_len = sequence.shape()[1];
    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int32_t>({position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({1}, kInt32, kCPU).alloc();
        *position_ids.offsettedPtr<int32_t>({0}) = last_pos + 1;
      }
    } else {
      position_ids = Tensor::empty({seq_len}, kInt32, kCPU).alloc();
      auto p = position_ids.ptr<int32_t>();
      for (int s = 0; s < seq_len; ++s) { p[s] = s; }
    }

    ir::lowlevel::traceStart();

    std::vector<Tensor> llm_inputs = {sequence, position_ids};
    llm_inputs.insert(llm_inputs.end(), k_arranged_inputs.begin(), k_arranged_inputs.end());
    llm_inputs.insert(llm_inputs.end(), v_arranged_inputs.begin(), v_arranged_inputs.end());

    sequence = llm_(llm_inputs)[0];
    sequence = lm_head_(ptq::QDQ(this, sequence, "lm_head_input_qdq"));
    sequence = ptq::QDQ(this, sequence, "lm_head_output_qdq");
    llm_ir = ir::lowlevel::traceStop();

    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

 private:
  const Qwen3Config& cfg_;
  int chunk_size_;
  Qwen3TextSHABlockSparse llm_;
  nn::Conv2D lm_head_;
  bool tie_word_embeddings_;
};

#undef BS_CONV2D_PROPERTY

}  // namespace mllm::models::qwen3::sha_blocksparse
