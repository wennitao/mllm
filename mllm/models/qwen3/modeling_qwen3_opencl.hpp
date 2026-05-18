// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cstdio>
#include <memory>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/modeling_qwen3.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"

namespace mllm::models::qwen3 {

// OpenCL-specific attention block that swaps the eager
// (matmul + mask + softmax + matmul) chain for a single FlashAttention call.
// Layout stays BHSD throughout and the eager-mode StaticCache is reused
// (it pre-expands K/V from kv_heads to q_heads), so no cache rewrite is
// needed; the FA op simply consumes [B, H, S_kv, D] K/V tensors directly.
class Qwen3AttentionOpenCL final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  int layer_idx_ = 0;

  Qwen3AttentionOpenCL() = default;

  Qwen3AttentionOpenCL(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ =
        reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ =
        reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ =
        reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ =
        reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // Loud first-call trace via raw stderr — no buffering, no fmt deps.
    static bool s_fwd_logged = false;
    if (!s_fwd_logged) {
      s_fwd_logged = true;
      std::fprintf(stderr, "[OpenCL-FA1] Qwen3AttentionOpenCL::forward entered (layer_idx=%d)\n", layer_idx_);
      std::fflush(stderr);
    }

    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

    // [B, S, H * D]
    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // Eager-mode cache: K/V come back as [B, q_heads, S_kv, D] (GQA-expanded).
    auto [key_states_new, value_states_new] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = key_states_new;
    value_states = value_states_new;

    // FlashAttention (BHSD inputs). Returns [B, H, S_q, D].
    auto output = nn::functional::flashAttention2(query_states, key_states, value_states);

    // [B, H, S_q, D] -> [B, S_q, H, D] -> [B, S_q, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }
};

class Qwen3DecoderOpenCL final : public nn::Module {
 public:
  Qwen3AttentionOpenCL self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3DecoderOpenCL() = default;

  Qwen3DecoderOpenCL(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3AttentionOpenCL>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    static bool s_dec_logged = false;
    if (!s_dec_logged) {
      s_dec_logged = true;
      std::fprintf(stderr, "[OpenCL-FA1] Qwen3DecoderOpenCL::forward entered\n");
      std::fflush(stderr);
    }

    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    auto x = input_layer_norm_(inputs[0]);
    x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

class Qwen3TextOpenCL final : public nn::Module {
  nn::ModuleList<Qwen3DecoderOpenCL> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen3TextOpenCL() = default;

  Qwen3TextOpenCL(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen3DecoderOpenCL>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    static bool s_text_logged = false;
    if (!s_text_logged) {
      s_text_logged = true;
      std::fprintf(stderr, "[OpenCL-FA1] Qwen3TextOpenCL::forward entered, num_blocks=%zu\n", decode_blocks_.list().size());
      std::fflush(stderr);
    }

    auto& blocks = decode_blocks_.list();

    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& kv_cache = args[0];

    for (auto& block : blocks) { x = block(x, llm_embedding_sin, llm_embedding_cos, kv_cache)[0]; }
    x = norm_(x);

    return {x};
  }
};

class Qwen3ForCausalLMOpenCL : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLMOpenCL(const Qwen3Config& cfg) : cfg(cfg) {
    kv_cache_ = std::make_unique<nn::StaticCache>(cfg.max_cache_length, cfg.num_hidden_layers,
                                                  cfg.num_attention_heads,  // q_heads
                                                  cfg.num_key_value_heads,  // kv_heads
                                                  cfg.head_dim,             // kv_dim
                                                  kFloat32,                 // k_dtype
                                                  kFloat32,                 // v_dtype
                                                  kOpenCL,                  // device_type
                                                  false                     // use_fa2
    );
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    embedding_ = reg<nn::Embedding>("model.embed_tokens", cfg.vocab_size, cfg.hidden_size);
    llm = reg<Qwen3TextOpenCL>("model", cfg);

    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }

    // Keep inv_freq on CPU. RoPE sin/cos tables are generated host-side per step,
    // then copied to the current inference device.
    inv_freq_ = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
  }

  void toOpenCL() { llm.to(kOpenCL); }

  void to(DeviceTypes device_type) {
    if (device_type == kOpenCL) {
      toOpenCL();
      return;
    }
    nn::Module::to(device_type);
  }

  void load(const ParameterFile::ptr_t& param_file) {
    if (tie_word_embeddings_ && !param_file->has("lm_head_out.weight")) {
      if (param_file->has("lm_head.weight")) {
        param_file->push("lm_head_out.weight", param_file->pull("lm_head.weight"));
      } else if (param_file->has("model.embed_tokens.weight")) {
        param_file->push("lm_head_out.weight", param_file->pull("model.embed_tokens.weight"));
      }
    }
    nn::Module::load(param_file);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    static bool s_lm_logged = false;
    if (!s_lm_logged) {
      s_lm_logged = true;
      std::fprintf(stderr, "[OpenCL-FA1] Qwen3ForCausalLMOpenCL::forward (ARGeneration entry) entered\n");
      std::fflush(stderr);
    }

    auto sequence = input.at("sequence");
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      position_ids = input.at("position_ids");
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    auto hidden_states = embedding_(sequence);
    hidden_states = hidden_states.to(kOpenCL);

    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, inv_freq_, 1.0f);
    llm_embedding_sin = llm_embedding_sin.to(kOpenCL);
    llm_embedding_cos = llm_embedding_cos.to(kOpenCL);

    sequence = llm(hidden_states, llm_embedding_sin, llm_embedding_cos, AnyValue(kv_cache_.get()))[0];

    auto S = sequence.shape()[1];
    sequence = sequence[{kAll, {S - 1}, kAll}];
    sequence = sequence.to(kCPU);
    if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

    return {
        {"sequence", sequence},
        {"position_ids", position_ids},
    };
  }

  inline nn::StaticCache& kvCache() { return *kv_cache_; }

 private:
  const Qwen3Config& cfg;
  nn::Embedding embedding_;
  Qwen3TextOpenCL llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_ = true;
  Tensor inv_freq_;
  std::unique_ptr<nn::StaticCache> kv_cache_;
};

}  // namespace mllm::models::qwen3
