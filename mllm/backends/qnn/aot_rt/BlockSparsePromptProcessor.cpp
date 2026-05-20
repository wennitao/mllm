// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/aot_rt/BlockSparsePromptProcessor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>

namespace mllm::qnn::aot {

// Graph name for the per-qb causal block-sparse model. Must match what the
// compile script registered (see compile_fp16_blocksparse_causal.cpp:
// graph_name == "model.0.s32" — config_.ar_len would be the qb size here).
BlockSparsePromptProcessor::BlockSparsePromptProcessor(KVCacheManager<uint16_t>* kv_manager, QnnAOTConfig config)
    : kv_manager_(kv_manager), config_(config) {
  // Per-qb graph. ar_len in this processor's config is the BQ size (= kBQ).
  std::string graph_name = "model.0.s" + std::to_string(kBQ);
  module_ = std::make_unique<QnnAOTModule>(graph_name);
  module_->to(kQNN);
}

void BlockSparsePromptProcessor::init_io() {
  // Layout (mirrors what the compile script declared in trace_inputs):
  //   [0]   input_ids       [1, BQ]                            int32
  //   [1]   position_ids    [BQ]                               int32
  //   [2..L)             past_key  per layer [1, Hkv, D, CL-BQ]   fp16
  //   [L..2L)            past_value per layer [1, Hkv, CL-BQ, D]  fp16
  //   [2L..3L)           k_arranged per layer [Hq, 1, top_k*BK, D] fp16
  //   [3L..4L)           v_arranged per layer [Hq, 1, top_k*BK, D] fp16
  //   [4L]               padding_mask [1, top_k*BK]               fp16

  const int L = config_.num_layers;
  input_tensors_.clear();
  input_tensors_.reserve(2 + 4 * L + 1);

  auto input_ids = Tensor::empty({1, kBQ}, kInt32, kQNN).alloc();
  input_ids.setName("input_ids");
  input_tensors_.push_back(input_ids);

  auto pos_ids = Tensor::empty({kBQ}, kInt32, kQNN).alloc();
  pos_ids.setName("position_ids");
  input_tensors_.push_back(pos_ids);

  // past_key / past_value share storage with the KV cache manager.
  const auto& k_caches = kv_manager_->getKCache();
  const auto& v_caches = kv_manager_->getVCache();
  for (int l = 0; l < L; ++l) {
    auto k_tensor = Tensor::empty({1, (int)config_.num_heads, config_.head_dim, config_.context_len - kBQ},
                                  config_.kv_dtype, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("past_key_" + std::to_string(l));
    input_tensors_.push_back(k_tensor);
  }
  for (int l = 0; l < L; ++l) {
    auto v_tensor = Tensor::empty({1, (int)config_.num_heads, config_.context_len - kBQ, config_.head_dim},
                                  config_.kv_dtype, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("past_value_" + std::to_string(l));
    input_tensors_.push_back(v_tensor);
  }

  // K_arranged / V_arranged buffers — HISTORICAL slots only.
  // Shape: [num_attention_heads, 1, (top_k-1)*BK, head_dim].
  // The diagonal slot is concat'd in by the model itself; runner doesn't
  // need to (and can't, chicken-and-egg) supply it.
  //
  // Compile side declares K_arranged with num_attention_heads (Hq), not
  // num_kv_heads. For Qwen3-1.7B GQA this is 16 vs 8 — a 2x buffer-size
  // mismatch if confused, which trips QNN's memRegister buffer check.
  const int Hq = (int)config_.num_attention_heads;
  k_arranged_.clear();
  v_arranged_.clear();
  k_arranged_.reserve(L);
  v_arranged_.reserve(L);
  for (int l = 0; l < L; ++l) {
    auto k = Tensor::empty({Hq, 1, kHistKBK, config_.head_dim}, kFloat16, kQNN).alloc();
    std::memset(k.ptr<void>(), 0, k.bytes());
    k.setName("k_arranged_" + std::to_string(l));
    auto v = Tensor::empty({Hq, 1, kHistKBK, config_.head_dim}, kFloat16, kQNN).alloc();
    std::memset(v.ptr<void>(), 0, v.bytes());
    v.setName("v_arranged_" + std::to_string(l));
    k_arranged_.push_back(k);
    v_arranged_.push_back(v);
    input_tensors_.push_back(k);
    input_tensors_.push_back(v);
  }

  padding_mask_ = Tensor::empty({1, kTopKBK}, kFloat16, kQNN).alloc();
  std::memset(padding_mask_.ptr<void>(), 0, padding_mask_.bytes());
  padding_mask_.setName("padding_mask");
  input_tensors_.push_back(padding_mask_);

  // Outputs: logits [1, 1, BQ, vocab] + present_key/present_value per layer.
  output_tensors_.clear();
  output_tensors_.reserve(1 + 2 * L);
  // lm_head is fp16 in this model — match the dtype so sampleGreedy
  // dispatches to the fp16 path.
  auto logits = Tensor::empty({1, 1, kBQ, config_.vocab_size}, kFloat16, kQNN).alloc();
  logits.setName("logits");
  output_tensors_.push_back(logits);
  for (int l = 0; l < L; ++l) {
    auto k_tensor = Tensor::empty({1, (int)config_.num_heads, config_.head_dim, kBQ}, config_.kv_dtype, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].output_buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("present_key_" + std::to_string(l));
    output_tensors_.push_back(k_tensor);
  }
  for (int l = 0; l < L; ++l) {
    auto v_tensor = Tensor::empty({1, (int)config_.num_heads, kBQ, config_.head_dim}, config_.kv_dtype, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].output_buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("present_value_" + std::to_string(l));
    output_tensors_.push_back(v_tensor);
  }
}

void BlockSparsePromptProcessor::selectTopKBlocks(int qb_global, std::vector<int>& sel) {
  // Initial policy: most-recent (top_k - 1) historical blocks.
  // Selection is the same across heads (no content-aware scoring yet).
  // Diagonal slot is handled by the model layer, not selected here.
  const int Hq = (int)config_.num_attention_heads;
  const int n_hist = kTopK - 1;
  sel.assign((size_t)Hq * n_hist, 0);
  for (int h = 0; h < Hq; ++h) {
    int* hs = sel.data() + (size_t)h * n_hist;
    const int hist_slots = std::min(n_hist, qb_global);
    // Slots 0..hist_slots-1 → most recent historical k-blocks.
    for (int s = 0; s < hist_slots; ++s) {
      // Slot 0 = most recent (qb_global - 1), slot 1 = (qb_global - 2), ...
      hs[s] = qb_global - 1 - s;
    }
    // Slots hist_slots..n_hist-1 → padding (will be masked; index doesn't matter).
    for (int s = hist_slots; s < n_hist; ++s) hs[s] = 0;
  }
}

void BlockSparsePromptProcessor::gather_one_qb(int layer, int qb_global, const std::vector<int>& sel) {
  // Gather (top_k - 1) historical k-blocks per attention head from the live
  // KV cache into k_arranged_[layer] and v_arranged_[layer]. The diagonal
  // slot is NOT this function's responsibility — model.layer's in-layer
  // concat({K_hist, K_curr}, -2) supplies it from this dispatch's own
  // K-projection.
  //
  // KV cache buffers (from KVCacheManager — uint16 storage = fp16 bytes):
  //   past_key   shape [1, Hkv, D, max_cache_len]   K is transposed for HMX
  //   past_value shape [1, Hkv, max_cache_len, D]
  //
  // K_arranged target shape [Hq, 1, (top_k-1)*BK, D] — i.e., for attention
  // head h, slot s, row r in [0, BK), column c in [0, D):
  //   K_arranged[h][0][s*BK + r][c] = past_key[0][kv_head_idx][c][k_pos*BK + r]
  // (K's transposed layout means we walk past_key along its dim 2=D for each
  // output column; that's a strided copy, not a contiguous memcpy.)
  //
  // V is contiguous-friendly:
  //   V_arranged[h][0][s*BK + r][c] = past_value[0][kv_head_idx][k_pos*BK + r][c]
  // The BK rows of V_arranged for one slot are BK contiguous rows of D
  // elements in past_value — one memcpy per (head, slot).
  //
  // GQA: kv_head_idx = h / num_key_value_groups where Hq = num_kv_heads * group.
  const int Hkv = (int)config_.num_heads;            // num_kv_heads
  const int Hq  = (int)config_.num_attention_heads;  // num_attention_heads
  const int group = Hq / Hkv;                        // = 1 for MHA, 2 for Qwen3-1.7B GQA
  const int D  = (int)config_.head_dim;
  const int CL = (int)config_.context_len;
  const int n_hist = kTopK - 1;

  // past_key/past_value pointers for THIS layer.
  const uint16_t* pk = kv_manager_->getKCache()[layer].buffer;  // size: Hkv * D * (CL - BQ)
  const uint16_t* pv = kv_manager_->getVCache()[layer].buffer;  // size: Hkv * (CL - BQ) * D
  uint16_t* kdst = (uint16_t*)k_arranged_[layer].ptr<mllm_fp16_t>();
  uint16_t* vdst = (uint16_t*)v_arranged_[layer].ptr<mllm_fp16_t>();

  // Strides for past_key (K is [Hkv, D, max_cache_len_K]) and past_value
  // (V is [Hkv, max_cache_len_V, D]). max_cache_len for both is CL - BQ
  // (matches how the input tensors are declared).
  const int max_kv_len = CL - kBQ;
  const size_t past_key_head_stride   = (size_t)D * max_kv_len;
  const size_t past_value_head_stride = (size_t)max_kv_len * D;

  // Strides for K_arranged / V_arranged ([Hq, 1, n_hist*BK, D]).
  const size_t arr_head_stride = (size_t)n_hist * kBK * D;

  for (int h = 0; h < Hq; ++h) {
    const int kv_head_idx = h / group;
    const int* hs = sel.data() + (size_t)h * n_hist;

    const uint16_t* pk_head = pk + kv_head_idx * past_key_head_stride;       // [D, max_kv_len]
    const uint16_t* pv_head = pv + kv_head_idx * past_value_head_stride;     // [max_kv_len, D]
    uint16_t* kdst_head = kdst + (size_t)h * arr_head_stride;
    uint16_t* vdst_head = vdst + (size_t)h * arr_head_stride;

    for (int s = 0; s < n_hist; ++s) {
      const int k_pos = hs[s];                          // k-block index (0..)
      const int k_off = k_pos * kBK;                    // row offset in KV cache space

      // V: BK contiguous rows of D elements. Single memcpy.
      // past_value rows in [k_off, k_off+BK), each D elements.
      // Destination: v_arranged[h][0][s*BK : s*BK+BK][:].
      const uint16_t* vsrc = pv_head + (size_t)k_off * D;
      uint16_t*       vds  = vdst_head + (size_t)s * kBK * D;
      if (k_pos >= 0 && k_off + kBK <= max_kv_len) {
        std::memcpy(vds, vsrc, (size_t)kBK * D * sizeof(uint16_t));
      } else {
        // Out-of-range (e.g., padding slot pointing at index 0 with no history)
        // — leave as zeros; padding_mask will mask these out.
        std::fill_n(vds, (size_t)kBK * D, (uint16_t)0);
      }

      // K: transposed layout. For destination row r=0..BK-1, column c=0..D-1:
      //   K_arranged[h][0][s*BK + r][c] = past_key[0][kv][c][k_off + r]
      // The destination row stride is D; the source has D rows of (max_kv_len)
      // elements. Easiest: per output row r, gather D elements by walking
      // past_key[kv][:][k_off + r] — D strided loads. We do this as D
      // single-element copies per row (could SIMD later).
      if (k_pos >= 0 && k_off + kBK <= max_kv_len) {
        for (int r = 0; r < kBK; ++r) {
          uint16_t* krow_dst = kdst_head + ((size_t)s * kBK + r) * D;
          for (int c = 0; c < D; ++c) {
            krow_dst[c] = pk_head[(size_t)c * max_kv_len + (k_off + r)];
          }
        }
      } else {
        uint16_t* kds = kdst_head + (size_t)s * kBK * D;
        std::fill_n(kds, (size_t)kBK * D, (uint16_t)0);
      }
    }
  }
}

void BlockSparsePromptProcessor::build_padding_mask(int qb_global) {
  // Slots 0..hist_slots-1 = real history (mask = 0)
  // Slots hist_slots..top_k-2 = padding (mask = kMaskNeg)
  // Slot top_k-1 = diagonal (mask = 0; diagonal triangle is in the static mask)
  const int hist_slots = std::min(kTopK - 1, qb_global);
  auto p = padding_mask_.ptr<mllm_fp16_t>();
  for (int s = 0; s < kTopK; ++s) {
    const bool is_padding = (s >= hist_slots) && (s < kTopK - 1);
    const mllm_fp16_t v = is_padding ? (mllm_fp16_t)kMaskNeg : (mllm_fp16_t)0.0f;
    for (int j = 0; j < kBK; ++j) p[s * kBK + j] = v;
  }
}

void BlockSparsePromptProcessor::prepare_one_qb(const std::vector<int64_t>& prompt_tokens, int64_t prompt_pos,
                                                int64_t global_pos, int qb_in_chunk, int qb_global) {
  // 1. Input IDs & position IDs for this qb's BQ tokens.
  int32_t* input_ids_ptr = input_tensors_[0].ptr<int32_t>();
  int32_t* pos_ids_ptr   = input_tensors_[1].ptr<int32_t>();
  for (int i = 0; i < kBQ; ++i) {
    int64_t src_idx = prompt_pos + (int64_t)qb_in_chunk * kBQ + i;
    if (src_idx < (int64_t)prompt_tokens.size()) {
      input_ids_ptr[i] = (int32_t)prompt_tokens[src_idx];
    } else {
      input_ids_ptr[i] = 0;
    }
    pos_ids_ptr[i] = (int32_t)(global_pos + (int64_t)qb_in_chunk * kBQ + i);
  }

  // 2. Padding mask.
  build_padding_mask(qb_global);

  // 3. Selection + per-layer K_arranged/V_arranged gather (historical only).
  std::vector<int> sel;
  selectTopKBlocks(qb_global, sel);
  for (int l = 0; l < config_.num_layers; ++l) {
    gather_one_qb(l, qb_global, sel);
  }
}

int64_t BlockSparsePromptProcessor::prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos) {
  int64_t num_tokens = prompt_tokens.size();
  int64_t current_pos = start_pos;
  int64_t processed_tokens = 0;

  module_->setOutputTensors(output_tensors_);

  MLLM_INFO("BlockSparsePromptProcessor: num_tokens={} start_pos={}", num_tokens, start_pos);

  // Process the prompt one qb (= BQ tokens) at a time. No "chunk of N qbs"
  // grouping here — each graphExecute IS one qb.
  while (processed_tokens < num_tokens) {
    int64_t remaining = num_tokens - processed_tokens;
    if (remaining < kBQ) {
      // Tail of the prompt that's smaller than BQ. Pad with zeros; the
      // logits for the trailing padded positions are discarded.
    }

    const int qb_global = (int)(current_pos / kBQ);
    prepare_one_qb(prompt_tokens, processed_tokens, current_pos, /*qb_in_chunk=*/0, qb_global);

    std::vector<Tensor> module_input = input_tensors_;
    output_tensors_ = (*module_)(module_input);

    const int32_t n_update = (int32_t)std::min<int64_t>(kBQ, remaining);
    kv_manager_->updateCache(kBQ, current_pos, n_update, {});

    processed_tokens += n_update;
    current_pos += n_update;
  }

  // Sample the last produced position from the last qb's output.
  auto logits = output_tensors_[0]
                    .to(kCPU)
                    .squeeze(0)[{kAll, ((int)num_tokens + kBQ - 1) % kBQ, kAll}];
  auto cur_token = module_->sampleGreedy(logits);
  return cur_token;
}

}  // namespace mllm::qnn::aot
