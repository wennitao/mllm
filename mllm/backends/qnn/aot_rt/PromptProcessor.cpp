// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/aot_rt/PromptProcessor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>

namespace mllm::qnn::aot {

template<typename T>
PromptProcessor<T>::PromptProcessor(KVCacheManager<T>* kv_manager, QnnAOTConfig config)
    : kv_manager_(kv_manager), config_(config) {
  std::string graph_name = "model.0.s" + std::to_string(config_.ar_len);
  module_ = std::make_unique<QnnAOTModule>(graph_name);
  module_->to(kQNN);
}

template<typename T>
void PromptProcessor<T>::init_io() {
  input_tensors_.reserve(3 + 2 * config_.num_layers);

  // 1. Input IDs
  auto input_ids = Tensor::empty({1, config_.ar_len}, kInt32, kQNN).alloc();
  input_ids.setName("input_ids");
  input_tensors_.push_back(input_ids);

  // 2. Position IDs
  auto pos_ids = Tensor::empty({config_.ar_len}, kInt32, kQNN).alloc();
  pos_ids.setName("position_ids");
  input_tensors_.push_back(pos_ids);

  // 3. Attention Mask
  auto attn_mask = Tensor::empty({1, 1, config_.ar_len, config_.context_len}, kUInt16, kQNN).alloc();
  attn_mask.setName("attention_mask");
  input_tensors_.push_back(attn_mask);

  // 4. KV Caches
  const auto& k_caches = kv_manager_->getKCache();
  const auto& v_caches = kv_manager_->getVCache();
  // K
  for (int l = 0; l < config_.num_layers; ++l) {
    auto k_tensor = Tensor::empty({1, (int)config_.num_heads, config_.head_dim, config_.context_len - config_.ar_len},
                                  config_.kv_dtype, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("past_key_" + std::to_string(l));
    input_tensors_.push_back(k_tensor);
  }
  // V
  for (int l = 0; l < config_.num_layers; ++l) {
    auto v_tensor = Tensor::empty({1, (int)config_.num_heads, config_.context_len - config_.ar_len, config_.head_dim},
                                  config_.kv_dtype, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("past_value_" + std::to_string(l));
    input_tensors_.push_back(v_tensor);
  }

  // Output Tensors
  output_tensors_.reserve(1 + 2 * config_.num_layers);

  // 1. Logits. MLLM_DENSE_LASTTOK: bin computes lm_head for ONLY the last position
  // (see modeling_qwen_qnn_aot_sha.hpp), so logits is [1,1,1,vocab] not [1,1,ar_len,vocab].
  const int lm_positions = std::getenv("MLLM_DENSE_LASTTOK") ? 1 : config_.ar_len;
  auto logits = Tensor::empty({1, 1, lm_positions, config_.vocab_size}, kUInt16, kQNN).alloc();
  logits.setName("logits");
  output_tensors_.push_back(logits);

  // 2. KV Caches, should be consistant with modeling file, or it will cause error
  for (int l = 0; l < config_.num_layers; ++l) {
    // K Output
    auto k_tensor = Tensor::empty({1, (int)config_.num_heads, config_.head_dim, config_.ar_len}, config_.kv_dtype, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].output_buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("present_key_" + std::to_string(l));
    output_tensors_.push_back(k_tensor);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    // V Output
    auto v_tensor = Tensor::empty({1, (int)config_.num_heads, config_.ar_len, config_.head_dim}, config_.kv_dtype, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].output_buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("present_value_" + std::to_string(l));
    output_tensors_.push_back(v_tensor);
  }
}

template<typename T>
void PromptProcessor<T>::prepare_io(const std::vector<int64_t>& prompt_tokens, int64_t prompt_pos, int64_t start_pos) {
  int64_t num_tokens = prompt_tokens.size();
  int64_t chunk_size = std::min((int64_t)config_.ar_len, num_tokens - prompt_pos);

  int32_t* input_ids_ptr = input_tensors_[0].ptr<int32_t>();
  int32_t* pos_ids_ptr = input_tensors_[1].ptr<int32_t>();
  for (int i = 0; i < config_.ar_len; ++i) {
    // 1. Input IDs
    if (i < chunk_size) {
      input_ids_ptr[i] = (int32_t)prompt_tokens[prompt_pos + i];
    } else {
      input_ids_ptr[i] = 0;  // Padding
    }

    // 2. Position IDs
    pos_ids_ptr[i] = start_pos + i;
  }
}

template<typename T>
int64_t PromptProcessor<T>::prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos) {
  int64_t num_tokens = prompt_tokens.size();
  int64_t current_pos = start_pos;
  int64_t processed_tokens = 0;

  // Ensure KV cache is arranged for ar_len
  kv_manager_->rearrangeCache(config_.ar_len);

  std::vector<int32_t> attention_map(config_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  kv_manager_->initAttentionMask(input_tensors_[2].ptr<uint16_t>(), attention_map, config_.ar_len, start_pos);
  // init window attention mask with current position
  kv_manager_->initAttentionMask(input_tensors_[2].ptr<uint16_t>(), attention_map, config_.ar_len, start_pos,
                                 config_.sliding_window);

  module_->setOutputTensors(output_tensors_);

  MLLM_INFO("num_tokens: {}", num_tokens);

  while (processed_tokens < num_tokens) {
    int64_t chunk_size = std::min((int64_t)config_.ar_len, num_tokens - processed_tokens);

    prepare_io(prompt_tokens, processed_tokens, current_pos);

    std::vector<Tensor> module_input = input_tensors_;
    output_tensors_ = (*module_)(module_input);

    int32_t n_update = chunk_size;

    // Tap-point: dump layer-L K and V output bytes (post-RoPE for K,
    // post-projection for V, both pre-cache-write).
    // K layout: [num_heads, head_dim, ar_len] of sizeof(T) bytes each.
    // V layout: [num_heads, ar_len, head_dim] of sizeof(T) bytes each.
    if (const char* layer_env = std::getenv("MLLM_DUMP_K_LAYER")) {
      int layer = std::atoi(layer_env);
      if (layer >= 0 && layer < config_.num_layers) {
        size_t bytes = sizeof(T) * (size_t)config_.num_heads * (size_t)config_.head_dim * (size_t)config_.ar_len;
        if (const char* path = std::getenv("MLLM_DUMP_K_PATH")) {
          const auto& k_caches_dump = kv_manager_->getKCache();
          FILE* f = std::fopen(path, "wb");
          if (f) {
            std::fwrite(k_caches_dump[layer].output_buffer, 1, bytes, f);
            std::fclose(f);
          }
          MLLM_INFO("[dump_k] dense: layer={} wrote {} bytes to {}", layer, bytes, path);
        }
        if (const char* vpath = std::getenv("MLLM_DUMP_V_PATH")) {
          const auto& v_caches_dump = kv_manager_->getVCache();
          FILE* f = std::fopen(vpath, "wb");
          if (f) {
            std::fwrite(v_caches_dump[layer].output_buffer, 1, bytes, f);
            std::fclose(f);
          }
          MLLM_INFO("[dump_v] dense: layer={} wrote {} bytes to {}", layer, bytes, vpath);
        }
      }
    }

    kv_manager_->updateCache(config_.ar_len, current_pos, n_update, {});

    kv_manager_->updateAttentionMask(input_tensors_[2].ptr<uint16_t>(), config_.ar_len, current_pos, n_update);
    kv_manager_->updateAttentionMask(input_tensors_[2].ptr<uint16_t>(), config_.ar_len, current_pos, n_update,
                                     config_.sliding_window);

    processed_tokens += chunk_size;
    current_pos += chunk_size;
  }

  // MLLM_DENSE_LASTTOK: graph emits only the last position -> read index 0.
  const int logit_row =
      std::getenv("MLLM_DENSE_LASTTOK") ? 0 : (((int)num_tokens + config_.ar_len - 1) % config_.ar_len);
  auto logits = output_tensors_[0].to(kCPU).squeeze(0)[{kAll, logit_row, kAll}];

  if (const char* p = std::getenv("MLLM_DUMP_LOGITS")) {
    auto* data = logits.ptr<uint16_t>();
    size_t n = (size_t)config_.vocab_size;
    FILE* f = std::fopen(p, "wb");
    if (f) { std::fwrite(data, sizeof(uint16_t), n, f); std::fclose(f); }
    MLLM_INFO("[dump_logits] dense: wrote {} u16 to {}", n, p);
  }

  auto cur_token = module_->sampleGreedy(logits);

  return cur_token;
}

// Explicit instantiations
template class PromptProcessor<uint16_t>;
template class PromptProcessor<uint8_t>;

}  // namespace mllm::qnn::aot
