// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// fp16 Qwen3 (0.6B / 1.7B per-attention-shape, identical) for QNN-AOT compile,
// with **block-sparse attention** instead of dense at the attention step.
//
// Compared to modeling_qwen_qnn_aot_sha.hpp:
//   * No QDQ wrappers (this is plain fp16, no w4a16 / int8 KV).
//   * Q/K/V/O/MLP projections via nn::Linear (lowers to qti.aisw MatMul).
//   * Per-head attention loop replaced with block-sparse compute that takes
//     CPU-prepared K_arranged / V_arranged tensors as additional layer
//     inputs — see the BlockSparseAttention forward() docstring.
//
// Compile-time constants for block-sparse:
//   BK     = BQ          = 32           (HMX 32×32 tile)
//   top_k                = 8            (1/4 of 32 K-blocks for context_len=1024)
//   top_k_BK             = top_k · BK   = 256
// The runner is responsible for selecting and gathering exactly top_k blocks
// per (head, q_block) into K_arranged / V_arranged before each layer dispatch.
//
// Status: Phase 1+2 (compile-only). Runner integration with CPU gather +
// pipelining is Phase 3 — not done here.

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

namespace mllm::models::qwen3::sha_fp16_blocksparse {

// ---------------------------------------------------------------------------
// Block-sparse compile-time constants. Picked to match a 1024-context model
// with chunk_size=128 prefill: num_k_blocks_max = 1024/32 = 32, top_k = 1/4 = 8.
// ---------------------------------------------------------------------------
constexpr int kBQ = 32;
constexpr int kBK = 32;
constexpr int kTopK = 8;
constexpr int kTopKBK = kTopK * kBK;  // 256

// ---------------------------------------------------------------------------
// rotateHalf — plain fp16, no QDQ. X is [B, 1, S, D].
// ---------------------------------------------------------------------------
inline Tensor rotateHalf(Tensor x) {
  auto D = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, /*ssa=*/true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, /*ssa=*/true);
  return nn::functional::concat({-x2, x1}, -1);
}

// ---------------------------------------------------------------------------
// MLP — gate / up / down via fp16 Linear.
// ---------------------------------------------------------------------------
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
    auto gate = silu_(gate_proj_(x));
    auto up = up_proj_(x);
    auto y = down_proj_(gate * up);
    return {y};
  }
};

// ---------------------------------------------------------------------------
// Block-sparse attention layer (the focus of this file).
//
// Inputs (in order):
//   0 : hidden_states           [B=1, S_q=chunk_size, D_h=hidden_size]
//   1 : llm_embedding_sin       [B=1, S_q, head_dim]   (RoPE)
//   2 : llm_embedding_cos       [B=1, S_q, head_dim]
//   3 : past_key                [B=1, num_kv_heads, head_dim, S_kv_max]   (kept as-is for cache update)
//   4 : past_value              [B=1, num_kv_heads, S_kv_max, head_dim]
//   5 : K_arranged              [num_attention_heads, num_q_blocks, top_k·BK, head_dim]
//   6 : V_arranged              [num_attention_heads, num_q_blocks, top_k·BK, head_dim]
//
// Outputs:
//   0 : hidden_states_next      [B=1, S_q, D_h]                      (attention + O proj)
//   1 : new_key                 [B=1, num_kv_heads, head_dim, S_kv]  (for cache update)
//   2 : new_value               [B=1, num_kv_heads, S_kv, head_dim]
//
// Note: this layer does not include "current-chunk local attention" — the
// runner is responsible for ensuring K_arranged / V_arranged include any
// local current-chunk K/V that should be attended to. That keeps the model
// graph as one clean MatMul-Softmax-MatMul chain.
//
// num_q_blocks = chunk_size / BQ. For chunk_size=128, BQ=32 → num_q_blocks=4.
// ---------------------------------------------------------------------------
class Qwen3AttentionBlockSparse final : public nn::Module {
  // Per-head Q projections (SHA layout, kept for AOT compile-time savings).
  std::vector<nn::Linear> q_projs_;
  std::vector<nn::Linear> k_projs_;
  std::vector<nn::Linear> v_projs_;
  nn::Linear o_proj_;

  std::vector<nn::RMSNorm> rms_norm_q_;
  std::vector<nn::RMSNorm> rms_norm_k_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  int chunk_size_;
  int num_q_blocks_;
  float scale_;

 public:
  Qwen3AttentionBlockSparse() = default;

  Qwen3AttentionBlockSparse(const std::string& name, const Qwen3Config& cfg, int chunk_size) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    chunk_size_ = chunk_size;
    num_q_blocks_ = chunk_size / kBQ;
    scale_ = 1.f / std::sqrt((float)head_dim_);

    MLLM_RT_ASSERT_EQ(chunk_size % kBQ, 0);

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
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto sin_in = inputs[1];
    auto cos_in = inputs[2];
    auto sin = sin_in.unsqueeze(1, true);
    auto cos = cos_in.unsqueeze(1, true);
    const auto& past_key = inputs[3];
    const auto& past_value = inputs[4];
    const auto& K_arranged = inputs[5];  // [num_attention_heads, num_q_blocks, top_k·BK, head_dim]
    const auto& V_arranged = inputs[6];  // same

    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // -----------------------------------------------------------------------
    // Per-head Q/K/V projections.
    // -----------------------------------------------------------------------
    std::vector<Tensor> Q_per_head;
    Q_per_head.reserve(num_attention_heads_);
    for (int h = 0; h < num_attention_heads_; ++h) {
      auto q = q_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, /*ssa=*/true);
      Q_per_head.push_back(q);
    }
    std::vector<Tensor> K_per_head;
    std::vector<Tensor> V_per_head;
    K_per_head.reserve(num_key_value_heads_);
    V_per_head.reserve(num_key_value_heads_);
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto k = k_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, /*ssa=*/true);
      auto v = v_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, /*ssa=*/true);
      K_per_head.push_back(k);
      V_per_head.push_back(v);
    }

    // RMSNorm + RoPE per head (Q).
    for (int h = 0; h < num_attention_heads_; ++h) {
      Q_per_head[h] = rms_norm_q_[h](Q_per_head[h]);
      Q_per_head[h] = Q_per_head[h] * cos + rotateHalf(Q_per_head[h]) * sin;
    }
    // RMSNorm + RoPE per K head.
    for (int h = 0; h < num_key_value_heads_; ++h) {
      K_per_head[h] = rms_norm_k_[h](K_per_head[h]);
      K_per_head[h] = K_per_head[h] * cos + rotateHalf(K_per_head[h]) * sin;
    }

    // -----------------------------------------------------------------------
    // KV cache: emit new_key / new_value for the runner to append. We don't
    // actually use these inside the attention compute — that's K_arranged /
    // V_arranged from the runner. We just need them as side outputs.
    // -----------------------------------------------------------------------
    std::vector<Tensor> new_key_per_head;
    std::vector<Tensor> new_value_per_head;
    for (int h = 0; h < num_key_value_heads_; ++h) {
      auto k_h = K_per_head[h].transpose(2, 3);  // [B, 1, D, S_q]
      auto v_h = V_per_head[h];                  // [B, 1, S_q, D]
      auto past_k_h = past_key.slice({kAll, {h, h + 1}, kAll, kAll}, true);
      auto past_v_h = past_value.slice({kAll, {h, h + 1}, kAll, kAll}, true);
      new_key_per_head.push_back(nn::functional::concat({past_k_h, k_h}, -1));
      new_value_per_head.push_back(nn::functional::concat({past_v_h, v_h}, 2));
    }
    auto new_key = nn::functional::concat(new_key_per_head, 1);    // [1, Hkv, D, S_kv]
    auto new_value = nn::functional::concat(new_value_per_head, 1);  // [1, Hkv, S_kv, D]

    // -----------------------------------------------------------------------
    // Block-sparse attention compute. For each Q head h, attention is
    //   QK    = Q_h_4d · K_arranged[h]^T           [num_q_blocks, BQ, top_k·BK]
    //   P     = softmax(QK · scale)
    //   y_h   = P · V_arranged[h]                  [num_q_blocks, BQ, head_dim]
    // and y_h reshaped back to [1, 1, S_q, head_dim].
    //
    // The 4D batched MatMul (leading dims = [1, num_q_blocks]) is what HMX
    // engages on — each batch element is BQ × head_dim × top_k·BK.
    // -----------------------------------------------------------------------
    std::vector<Tensor> attn_outputs;
    attn_outputs.reserve(num_attention_heads_);

    for (int h = 0; h < num_attention_heads_; ++h) {
      auto q_h = Q_per_head[h].view({1, num_q_blocks_, kBQ, head_dim_}, /*ssa=*/true);

      auto K_h = K_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, true);  // [1, num_q_blocks, top_k·BK, D]
      auto V_h = V_arranged.slice({{h, h + 1}, kAll, kAll, kAll}, true);

      auto K_h_T = K_h.transpose(-1, -2);  // [1, num_q_blocks, D, top_k·BK]
      auto attn = nn::functional::matmul(q_h, K_h_T);
      auto scale_const = Tensor::constant(scale_, kFloat32);
      attn = attn.mulConstant(scale_const);
      attn = nn::functional::softmax(attn, -1);
      auto y_h = nn::functional::matmul(attn, V_h);  // [1, num_q_blocks, BQ, D]

      y_h = y_h.view({1, 1, -1, head_dim_}, /*ssa=*/true);  // [1, 1, S_q, D]
      attn_outputs.push_back(y_h);
    }

    // Concat heads + output projection.
    auto y = nn::functional::concat(attn_outputs, 1);                                         // [1, Hq, S_q, D]
    y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);  // [1, 1, S_q, Hq·D]
    y = o_proj_(y).view({1, -1, hidden_size_}, true);                                        // [1, S_q, D_h]

    return {y, new_key, new_value};
  }

  int layer_idx_;
};

// ---------------------------------------------------------------------------
// Decoder block: input_layernorm → attention → residual → post_attn_norm → MLP → residual.
// Forwards the K_arranged / V_arranged inputs through to the attention.
// ---------------------------------------------------------------------------
class Qwen3DecoderBlockSparse final : public nn::Module {
 public:
  int layer_idx_;
  Qwen3AttentionBlockSparse self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3DecoderBlockSparse() = default;

  // NOTE: arg order matches nn::ModuleListWithIdx convention, which appends
  // the layer index at the end of the forwarded args.
  Qwen3DecoderBlockSparse(const std::string& name, const Qwen3Config& cfg, int chunk_size, int layer_idx)
      : nn::Module(name) {
    layer_idx_ = layer_idx;
    self_attn_ = reg<Qwen3AttentionBlockSparse>("self_attn", cfg, chunk_size);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto sin = inputs[1];
    auto cos = inputs[2];
    auto past_key = inputs[3];
    auto past_value = inputs[4];
    auto K_arranged = inputs[5];
    auto V_arranged = inputs[6];

    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto attn_out = self_attn_(hidden_states, sin, cos, past_key, past_value, K_arranged, V_arranged);
    hidden_states = residual + attn_out[0];

    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + hidden_states;

    return {hidden_states, attn_out[1], attn_out[2]};
  }
};

// ---------------------------------------------------------------------------
// Text model: embedding + RoPE precompute + N decoder blocks + final norm.
// Input layout (matching the existing AOT model):
//   inputs[0]                          : token ids
//   inputs[1]                          : position_ids
//   inputs[2 .. 2+L)                   : past_key per layer
//   inputs[2+L .. 2+2L)                : past_value per layer
//   inputs[2+2L .. 2+3L)               : K_arranged per layer  (NEW)
//   inputs[2+3L .. 2+4L)               : V_arranged per layer  (NEW)
// ---------------------------------------------------------------------------
class Qwen3TextBlockSparse final : public nn::Module {
  nn::ModuleListWithIdx<Qwen3DecoderBlockSparse> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;
  int32_t chunk_size_;

 public:
  Qwen3TextBlockSparse() = default;

  Qwen3TextBlockSparse(const std::string& name, const Qwen3Config& cfg, int chunk_size) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    chunk_size_ = chunk_size;
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3DecoderBlockSparse>>("layers", cfg.num_hidden_layers, cfg, chunk_size);
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
    std::vector<Tensor> new_keys;
    std::vector<Tensor> new_values;
    new_keys.reserve(L);
    new_values.reserve(L);

    for (auto [idx, block] : enumerate(blocks)) {
      auto past_k = inputs[2 + idx];
      auto past_v = inputs[2 + L + idx];
      auto K_arr = inputs[2 + 2 * L + idx];
      auto V_arr = inputs[2 + 3 * L + idx];

      auto out = block(x, sin, cos, past_k, past_v, K_arr, V_arr);
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
// CausalLM wrapper: text model + lm_head.
// ---------------------------------------------------------------------------
class Qwen3ForCausalLM_BlockSparse : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLM_BlockSparse(const Qwen3Config& cfg, int chunk_size) : cfg_(cfg), chunk_size_(chunk_size) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<Qwen3TextBlockSparse>("model", cfg, chunk_size);

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

    // Position IDs: prefill = [0..S_q), decode = increment last.
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
      for (int s = 0; s < seq_len; ++s) p[s] = s;
    }

    ir::lowlevel::traceStart();

    std::vector<Tensor> llm_inputs = {sequence, position_ids};
    llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());
    llm_inputs.insert(llm_inputs.end(), kv_arranged.begin(), kv_arranged.end());

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
  int chunk_size_;
  Qwen3TextBlockSparse llm_;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
};

// ---------------------------------------------------------------------------
// Parameter slicing for SHA: same logic as the existing modeling_qwen_qnn_aot_sha.hpp
// (split MHA Q/K/V weights across heads), but for fp16 Linear layers the
// weight layout is [Out, In] (vs Conv2D's [1, 1, In, Out]). Slicing along
// dim 0 here, vs dim 3 in the Conv2D version.
// ---------------------------------------------------------------------------
inline void prepareParametersForSHA_FP16(const ParameterFile::ptr_t& params, const Qwen3Config& cfg) {
  int num_heads = cfg.num_attention_heads;
  int num_kv_heads = cfg.num_key_value_heads;
  int head_dim = cfg.head_dim;
  int num_layers = cfg.num_hidden_layers;

  auto sliceLinear = [&](const std::string& orig_prefix, const std::string& new_prefix, int per_head_out, int n_splits) {
    std::string orig_name = orig_prefix + ".weight";
    if (!params->has(orig_name)) return;
    auto orig = params->pull(orig_name);  // [num_total_out, in]
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

    // q_norm / k_norm: replicate (norm weights are scalars per head_dim, shared).
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

}  // namespace mllm::models::qwen3::sha_fp16_blocksparse
