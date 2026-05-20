// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/core/Tensor.hpp"
#include <vector>
#include <memory>

namespace mllm::qnn::aot {

// PromptProcessor for the per-qb CAUSAL block-sparse Qwen3 model
// (compile target: mllm-qwen3-aot-fp16-blocksparse-causal-c).
//
// Differences from the standard PromptProcessor:
//   * Per-dispatch shape is BQ tokens (= 32), NOT ar_len. The graph runs
//     once per q-block.
//   * One prefill "chunk" of N tokens (N = num_qb * BQ) is materialised as
//     num_qb sequential graphExecute calls.
//   * No attention_mask input; instead, padding_mask [1, top_k*BK] is
//     APP_WRITE per qb (qb-dependent: which slots are padding vs real
//     history depends on the qb's global index).
//   * Two extra per-layer inputs vs standard processor:
//       K_arranged [Hq, 1, (top_k-1)*BK, D]   HISTORICAL slots only
//       V_arranged [Hq, 1, (top_k-1)*BK, D]   ditto
//     The DIAGONAL slot's K/V (current chunk's own self-attention K) is
//     stitched in INSIDE the model layer via concat({K_arranged, K_curr},
//     -2). The runner therefore never has to worry about supplying the
//     diagonal slot — that data doesn't exist until the model has run
//     its K-projection on the current chunk's hidden_states. This mirrors
//     the standard dense path's concat({past_k_h, k_h}, -1).
//   * The static causal triangle mask is BAKED into the graph at compile
//     time (no per-dispatch DMA cost).
//
// Per-qb dispatch flow:
//   1. CPU: select top_k - 1 historical k-block indices for each (head, qb)
//   2. CPU: gather K/V from the live KV cache into K_arranged/V_arranged
//   3. CPU: build the per-qb padding_mask
//   4. graphExecute(per-qb graph) — model concats current K/V onto K_arranged
//      to form the full top_k*BK attention K
//   5. Update KV cache from the model's present_key/present_value outputs
//
// Top-k selection policy: initial version uses "most recent top_k-1
// historical blocks" (sliding window). Real production would use a
// content-aware selector.
class BlockSparsePromptProcessor {
 public:
  BlockSparsePromptProcessor(KVCacheManager<uint16_t>* kv_manager, QnnAOTConfig config);

  ~BlockSparsePromptProcessor() {
    if (module_) { module_->setOutputTensors({}); }
    output_tensors_.clear();
    input_tensors_.clear();
  }

  /**
   * Prefill the per-qb causal block-sparse LLM module.
   * @param prompt_tokens The text prompt tokens.
   * @param start_pos     Starting position in KV cache (0 for fresh prompt).
   * @return Last logits' argmax token id.
   */
  int64_t prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos = 0);

  void init_io();

  // Per-qb constants (must match the compiled graph).
  static constexpr int kBQ = 32;
  static constexpr int kBK = 32;
  static constexpr int kTopK = 8;
  static constexpr int kTopKBK = kTopK * kBK;            // 256 — full attention K width
  static constexpr int kHistKBK = (kTopK - 1) * kBK;     // 224 — runner-supplied historical width
  static constexpr float kMaskNeg = -1.0e4f;

 private:
  // Fill input_tensors_ for one qb dispatch.
  //   qb_in_chunk  = qb index within the current prefill chunk (0..num_qb-1)
  //   qb_global    = global qb index (start_pos/BQ + qb_in_chunk)
  void prepare_one_qb(const std::vector<int64_t>& prompt_tokens, int64_t prompt_pos, int64_t global_pos,
                      int qb_in_chunk, int qb_global);

  // CPU-side block selection: pick top_k - 1 HISTORICAL k-block indices
  // per (head, this qb). The diagonal slot is handled by the model
  // (concat in-layer), not by this selector. Initial policy: most-recent
  // historical blocks (sliding window).
  void selectTopKBlocks(int qb_global, std::vector<int>& sel /* [num_heads * (top_k-1)] */);

  // Gather K and V from live cache into K_arranged_[layer] and
  // V_arranged_[layer] for this qb's HISTORICAL slots only.
  void gather_one_qb(int layer, int qb_global, const std::vector<int>& sel);

  // Build the per-qb padding mask:
  //   slot s in [hist_slots .. top_k-2]: kMaskNeg
  //   else: 0
  // where hist_slots = min(top_k-1, qb_global).
  void build_padding_mask(int qb_global);

 private:
  std::unique_ptr<QnnAOTModule> module_;
  KVCacheManager<uint16_t>* kv_manager_;
  QnnAOTConfig config_;

  // I/O tensors (registered with the module).
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;

  // Per-layer K_arranged / V_arranged buffers (kept alive between dispatches).
  // Historical slots only — diagonal slot is stitched in by the model.
  std::vector<Tensor> k_arranged_;  // shape [num_heads, 1, (top_k-1)*BK, head_dim] fp16
  std::vector<Tensor> v_arranged_;
  // Per-dispatch padding mask buffer.
  Tensor padding_mask_;             // shape [1, top_k*BK] fp16
};

}  // namespace mllm::qnn::aot
