#pragma once

#include <cmath>

#include "mllm/backends/qnn/CustomLayers.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/layers/Param.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::models::qwen3_npu {

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class Qwen3AttentionProjNPU final : public nn::Module {
  nn::RMSNorm input_layer_norm_;
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::Param quantize_scale_;
  nn::qnn::DequantizeAdd q_proj_dequantize_add_;
  nn::qnn::DequantizeAdd k_proj_dequantize_add_;
  nn::qnn::DequantizeAdd v_proj_dequantize_add_;

  int hidden_size_ = 0;
  int head_dim_ = 0;
  int num_attention_heads_ = 0;
  int num_key_value_heads_ = 0;

 public:
  Qwen3AttentionProjNPU() = default;

  Qwen3AttentionProjNPU(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;

    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);

    q_proj_ = reg<nn::Linear>("self_attn.q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias,
                              cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("self_attn.k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias,
                              cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("self_attn.v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias,
                              cfg.linear_impl_type);

    rms_norm_q_ = reg<nn::RMSNorm>("self_attn.q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("self_attn.k_norm", cfg.rms_norm_eps);

    quantize_scale_ = reg<nn::Param>("quantize_scale", getModuleName() + ".self_attn.q_proj.input_scale",
                                     Tensor::shape_t{1});

    q_proj_dequantize_add_ =
        reg<nn::qnn::DequantizeAdd>("self_attn.q_proj.dequantize", kFloat32, head_dim_ * num_attention_heads_);
    k_proj_dequantize_add_ =
        reg<nn::qnn::DequantizeAdd>("self_attn.k_proj.dequantize", kFloat32, head_dim_ * num_key_value_heads_);
    v_proj_dequantize_add_ =
        reg<nn::qnn::DequantizeAdd>("self_attn.v_proj.dequantize", kFloat32, head_dim_ * num_key_value_heads_);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    int B = x.shape()[0];
    int S = x.shape()[1];

    x = input_layer_norm_(x);
    x = x.view({B, S, 1, x.shape()[2]});
    x.attach("qnn_quant_scale", quantize_scale_.weight().impl());
    x = x.to(kInt16);

    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    query_states = q_proj_dequantize_add_(query_states);
    key_states = k_proj_dequantize_add_(key_states);
    value_states = v_proj_dequantize_add_(value_states);

    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    return {query_states, key_states, value_states};
  }
};

class Qwen3AttentionMatmul final : public nn::Module {
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;
  nn::KVCache kv_cache_;

  int head_dim_ = 0;
  int num_attention_heads_ = 0;
  int num_key_value_heads_ = 0;

 public:
  Qwen3AttentionMatmul() = default;

  Qwen3AttentionMatmul(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg) : nn::Module(name) {
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings);
    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
    kv_cache_ =
        reg<nn::KVCache>("kv_cache", 0, num_attention_heads_, num_key_value_heads_, head_dim_, false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto query_states = inputs[0];
    auto key_states = inputs[1];
    auto value_states = inputs[2];
    auto llm_embedding_sin = inputs[3];
    auto llm_embedding_cos = inputs[4];

    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    auto kv_outputs = kv_cache_(key_states, value_states);
    key_states = kv_outputs[0];
    value_states = kv_outputs[1];

    Tensor attn;
    if (key_states.dtype() == kFloat32) {
      attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFloat16) {
      attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true)
             * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      attn = attn.to(kFloat16);
    }

    auto output = nn::functional::matmul(attn, value_states);
    auto B = output.shape()[0];
    auto S = output.shape()[2];
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    return {output};
  }

  nn::KVCache& getKVCache() { return kv_cache_; }
};

class Qwen3OutProjAndMLP final : public nn::Module {
  nn::Linear o_proj_;
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;
  nn::RMSNorm post_attention_layer_norm_;
  nn::Param o_proj_quantize_scale_;
  nn::Param pre_mlp_proj_quantize_scale_;
  nn::Param down_proj_quantize_scale_;

  int hidden_size_ = 0;
  int intermediate_size_ = 0;

 public:
  Qwen3OutProjAndMLP() = default;

  Qwen3OutProjAndMLP(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;

    o_proj_ = reg<nn::Linear>("self_attn.o_proj", cfg.hidden_size, cfg.hidden_size, cfg.attention_bias,
                              cfg.linear_impl_type);
    gate_proj_ = reg<nn::Linear>("mlp.gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("mlp.act");
    up_proj_ = reg<nn::Linear>("mlp.up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("mlp.down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);

    o_proj_quantize_scale_ =
        reg<nn::Param>("o_proj_quantize_scale_", getModuleName() + ".self_attn.o_proj.input_scale", Tensor::shape_t{1});
    pre_mlp_proj_quantize_scale_ = reg<nn::Param>("pre_mlp_proj_quantize_scale_",
                                                  getModuleName() + ".mlp.gate_proj.input_scale", Tensor::shape_t{1});
    down_proj_quantize_scale_ = reg<nn::Param>("down_proj_quantize_scale_",
                                               getModuleName() + ".mlp.down_proj.input_scale", Tensor::shape_t{1});
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    auto input = inputs[0];
    auto res = inputs[1];
    MLLM_RT_ASSERT(res.device() == kQNN);

    input = input.view({B, S, 1, hidden_size_});
    input.attach("qnn_quant_scale", o_proj_quantize_scale_.weight().impl());
    input = input.to(kInt16);
    auto x = o_proj_(input);
    x = x.view({B, S, 1, hidden_size_}).to(kFloat32).view({B, S, hidden_size_});

    auto tmp = x + res;
    x = post_attention_layer_norm_(tmp);

    x = x.view({B, S, 1, hidden_size_});
    x.attach("qnn_quant_scale", pre_mlp_proj_quantize_scale_.weight().impl());
    x = x.to(kInt16);
    auto gate = gate_proj_(x);
    gate = gate.view({B, S, 1, intermediate_size_}).to(kFloat32);
    gate = silu_(gate);

    auto up = up_proj_(x);
    up = up.view({B, S, 1, intermediate_size_}).to(kFloat32);
    x = gate * up;

    x = x.view({B, S, 1, intermediate_size_});
    x.attach("qnn_quant_scale", down_proj_quantize_scale_.weight().impl());
    x = x.to(kInt16);
    x = down_proj_(x);
    x = x.view({B, S, 1, hidden_size_}).to(kFloat32).view({B, S, hidden_size_});

    x = x + tmp;
    return {x};
  }
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3AttentionProjNPU self_attn_proj_;
  Qwen3AttentionMatmul self_attn_matmul_;
  Qwen3OutProjAndMLP self_attn_out_mlp_;

  Qwen3Decoder() = default;

  Qwen3Decoder(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg) : nn::Module(name) {
    self_attn_proj_ = reg<Qwen3AttentionProjNPU>("", cfg);
    self_attn_proj_.to(kQNN);
    self_attn_matmul_ = reg<Qwen3AttentionMatmul>("self_attn", cfg);
    self_attn_out_mlp_ = reg<Qwen3OutProjAndMLP>("", cfg);
    self_attn_out_mlp_.to(kQNN);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    x = x.to(kQNN);
    auto states = self_attn_proj_(x);

    auto query_states = states[0].to(kCPU);
    auto key_states = states[1].to(kCPU);
    auto value_states = states[2].to(kCPU);

    x = self_attn_matmul_(query_states, key_states, value_states, llm_embedding_sin, llm_embedding_cos)[0];

    x = x.to(kQNN);
    auto res = inputs[0];
    res = res.to(kQNN);
    x = self_attn_out_mlp_(x, res)[0];

    return {x};
  }

  nn::KVCache& getKVCache() { return self_attn_matmul_.getKVCache(); }
};

class Qwen3Text final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;

 public:
  Qwen3Text() = default;

  Qwen3Text(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    embedding_.to(kQNN);

    auto inv = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];

    for (auto& block : decode_blocks_.list()) { x = block(x, llm_embedding_sin, llm_embedding_cos)[0]; }

    x = x.to(kCPU);
    x = norm_(x);
    return {x};
  }

  nn::Embedding embedding_;

  nn::ModuleList<Qwen3Decoder>& decode_blocks() { return decode_blocks_; }

  void clearKVCache() {
    for (auto& block : decode_blocks_.list()) { block.getKVCache().clearCache(); }
  }
};

class Qwen3ForCausalLMPrefill : public nn::Module, public ARGeneration {
 public:
  Qwen3ForCausalLMPrefill() = delete;

  explicit Qwen3ForCausalLMPrefill(const std::string& name, const mllm::models::qwen3::Qwen3Config& cfg)
      : cfg_(cfg), nn::Module(name) {
    model = reg<Qwen3Text>("model", cfg_);
    if (!cfg_.tie_word_embeddings) {
      lm_head_ = reg<nn::Linear>("lm_head", cfg_.hidden_size, cfg_.vocab_size, false, cfg_.linear_impl_type);
    }
    tie_word_embeddings_ = cfg_.tie_word_embeddings;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];
    auto real_seq = args.count("seq_len") ? args.at("seq_len").get<int>() : seq_len;

    Tensor position_ids;
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

    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, model.getBuffer("inv_freq"), 1.0f);

    auto input_embeddings = model.embedding_(sequence);
    auto hidden_states = model(input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];

    {
      auto S = hidden_states.shape()[1];
      hidden_states = hidden_states[{kAll, {real_seq - 1}, kAll}];
    }

    Tensor logits;
    if (!tie_word_embeddings_) {
      logits = lm_head_(hidden_states);
    } else {
      auto emb_w = model.embedding_.weight();
      logits = nn::functional::matmul(hidden_states, emb_w, false, true);
    }

    return {
        {"sequence", logits},
        {"position_ids", position_ids},
    };
  }

  Qwen3Text model;

 private:
  const mllm::models::qwen3::Qwen3Config& cfg_;
  nn::Linear lm_head_;
  bool tie_word_embeddings_ = true;
};

}  // namespace mllm::models::qwen3_npu
