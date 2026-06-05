// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm::models::lfm2_moe {

// LFM2.5-8B-A1B (LiquidAI `lfm2_moe`) configuration.
//
// Hybrid decoder that interleaves cheap double-gated short-conv layers with a
// minority of grouped-query attention layers, each followed by a SwiGLU FFN that
// is dense for the first `num_dense_layers` layers and a sparse top-k MoE block
// thereafter. Layer disposition (conv vs full_attention) is read verbatim from
// the `layer_types` array in config.json (it is irregular, not a fixed interval).
//
// LFM2.5-8B-A1B reference values:
//   hidden_size=2048, num_hidden_layers=24, intermediate_size=7168,
//   num_attention_heads=32, num_key_value_heads=8, head_dim=64,
//   conv_L_cache=3, conv_bias=false, num_dense_layers=2,
//   num_experts=32, num_experts_per_tok=4, moe_intermediate_size=1792,
//   use_expert_bias=true, routed_scaling_factor=1.0, norm_topk_prob=true,
//   rope_theta=5e6, norm_eps=1e-5, vocab_size=128000, tie_word_embeddings=true,
//   bos=124894, eos=124900 (<|im_end|>).
struct Lfm2MoeConfig : protected ConfigFile {
  Lfm2MoeConfig() = default;

  explicit Lfm2MoeConfig(const std::string& file_path) : ConfigFile(file_path) {
    const auto& root = data();
    // RoPE params live under a nested "rope_parameters" object.
    const nlohmann::json& rp = root.contains("rope_parameters") ? root["rope_parameters"] : root;

    hidden_size = root.value("hidden_size", hidden_size);
    intermediate_size = root.value("intermediate_size", intermediate_size);
    moe_intermediate_size = root.value("moe_intermediate_size", moe_intermediate_size);
    num_attention_heads = root.value("num_attention_heads", num_attention_heads);
    num_key_value_heads = root.value("num_key_value_heads", num_key_value_heads);
    num_hidden_layers = root.value("num_hidden_layers", num_hidden_layers);
    max_position_embeddings = root.value("max_position_embeddings", max_position_embeddings);
    norm_eps = root.value("norm_eps", norm_eps);
    vocab_size = root.value("vocab_size", vocab_size);

    // Short-conv params.
    conv_L_cache = root.value("conv_L_cache", conv_L_cache);
    conv_bias = root.value("conv_bias", conv_bias);

    // MoE params.
    num_dense_layers = root.value("num_dense_layers", num_dense_layers);
    num_experts = root.value("num_experts", num_experts);
    num_experts_per_tok = root.value("num_experts_per_tok", num_experts_per_tok);
    use_expert_bias = root.value("use_expert_bias", use_expert_bias);
    routed_scaling_factor = root.value("routed_scaling_factor", routed_scaling_factor);
    norm_topk_prob = root.value("norm_topk_prob", norm_topk_prob);

    rope_theta = rp.value("rope_theta", rope_theta);

    tie_word_embeddings = root.value("tie_word_embeddings", tie_word_embeddings);
    bos_token_id = root.value("bos_token_id", bos_token_id);
    eos_token_id = root.value("eos_token_id", eos_token_id);
    end_of_text_token_id = static_cast<int32_t>(eos_token_id);
    max_cache_length = root.value("max_cache_length", max_cache_length);

    // Layer disposition: "conv" | "full_attention", verbatim from config.json.
    layer_types.clear();
    if (root.contains("layer_types")) {
      layer_types = root["layer_types"].get<std::vector<std::string>>();
    }

    if (root.contains("linear_impl_type")) {
      linear_impl_type = aops::str2LinearImplTypes(root["linear_impl_type"]);
    }

    recompute();
  }

  // ---- raw config fields (defaults = LFM2.5-8B-A1B) ----
  int32_t hidden_size = 2048;
  int32_t intermediate_size = 7168;
  int32_t moe_intermediate_size = 1792;
  int32_t num_attention_heads = 32;
  int32_t num_key_value_heads = 8;
  int32_t num_hidden_layers = 24;
  int32_t max_position_embeddings = 128000;
  float norm_eps = 1e-5f;
  int32_t vocab_size = 128000;

  // Short conv
  int32_t conv_L_cache = 3;
  bool conv_bias = false;

  // MoE
  int32_t num_dense_layers = 2;
  int32_t num_experts = 32;
  int32_t num_experts_per_tok = 4;
  bool use_expert_bias = true;
  float routed_scaling_factor = 1.0f;
  bool norm_topk_prob = true;

  float rope_theta = 5000000.0f;

  bool tie_word_embeddings = true;
  int64_t bos_token_id = 124894;
  int64_t eos_token_id = 124900;
  int32_t end_of_text_token_id = 124900;
  int32_t max_cache_length = 2048;

  std::vector<std::string> layer_types;

  aops::LinearImplTypes linear_impl_type = aops::LinearImplTypes::kDefault;

  // ---- derived fields (computed in recompute()) ----
  int32_t head_dim = 64;             // hidden_size / num_attention_heads
  int32_t rotary_dim = 64;           // full RoPE over head_dim
  int32_t num_attention_layers = 6;  // count of full_attention entries in layer_types

  void recompute() {
    head_dim = (num_attention_heads > 0) ? (hidden_size / num_attention_heads) : head_dim;
    rotary_dim = head_dim;
    num_attention_layers = 0;
    for (int32_t i = 0; i < num_hidden_layers; ++i) {
      if (isAttentionLayer(i)) { ++num_attention_layers; }
    }
  }

  // Layer i is a full-attention layer iff layer_types[i] == "full_attention".
  // Falls back to "all conv" if layer_types is absent/short.
  [[nodiscard]] bool isAttentionLayer(int32_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int32_t>(layer_types.size())) { return false; }
    return layer_types[layer_idx] == "full_attention";
  }

  // Layers [0, num_dense_layers) use a dense MLP; the rest use a sparse MoE block.
  [[nodiscard]] bool isDenseLayer(int32_t layer_idx) const { return layer_idx < num_dense_layers; }
};

}  // namespace mllm::models::lfm2_moe
