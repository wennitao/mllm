// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/modeling_qwen3.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"

namespace mllm::models::qwen3 {

class Qwen3TextOpenCL final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen3TextOpenCL() = default;

  Qwen3TextOpenCL(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
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
