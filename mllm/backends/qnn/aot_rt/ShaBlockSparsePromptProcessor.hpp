// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/core/Tensor.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace mllm::qnn::aot {

// PromptProcessor for the LPBQ SHA per-qb CAUSAL block-sparse Qwen3 model.
// (compile target: mllm-qwen3-aot-sha-blocksparse-causal-c)
//
// Differences from the fp16 BlockSparsePromptProcessor:
//   * KV cache dtype is uint8 (sym, zp=128) instead of fp16. K cache is
//     stored transposed [1, Hkv, D, S]; V cache is [1, Hkv, S, D].
//   * mask is a single uint16-quant graph input (combined padding +
//     diagonal-triangle), not separate fp16 mask + baked param.
//   * past_key / past_value are NOT graph inputs (the graph reads only from
//     K_arranged / V_arranged, which the runner gathers from its CPU-side
//     uint8 KV cache).
//   * K_arranged is in TRANSPOSED layout [Hq, 1, D, hist_k_BK] to match
//     the cache layout for memcpy-friendly gather.
//
// Per-qb dispatch flow (same as the fp16 variant):
//   1. CPU: select (top_k - 1) historical k-block indices for this qb
//   2. CPU: gather K/V from CPU-side uint8 KV cache into K_arranged /
//      V_arranged
//   3. CPU: build the per-qb combined mask (uint16: 65535 = ACTIVE,
//      0 = MASKED). Slots 0..top_k-2 active when qb has that many real
//      historical blocks; slot top_k-1 is the causal triangle.
//   4. graphExecute(per-qb LPBQ SHA graph) — model concats current K/V
//      onto K_arranged in-layer to form full top_k*BK attention K
//   5. Update CPU-side KV cache from the model's uint8
//      present_key / present_value outputs
class ShaBlockSparsePromptProcessor {
 public:
  ShaBlockSparsePromptProcessor(KVCacheManager<uint8_t>* kv_manager, QnnAOTConfig config);

  ~ShaBlockSparsePromptProcessor() {
    if (module_) { module_->setOutputTensors({}); }
    output_tensors_.clear();
    input_tensors_.clear();
  }

  int64_t prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos = 0);

  // Run greedy decode for up to max_new_tokens steps. `all_tokens` is the
  // running token history (prompt + first sampled token from prefill). Each
  // decode step appends one new token to `all_tokens` and invokes
  // `token_callback(token_id)` for it. Stops on eos_token_id or after
  // max_new_tokens new tokens, whichever comes first.
  //
  // Implementation note: we don't have an ar_len=1 graph, so each step
  // re-feeds the current qb's token slice (positions 0..position_in_qb) to
  // recompute K_curr at all already-decoded positions of this qb. Slightly
  // wasteful (extra K/V compute) but no separate compile needed.
  void decode(std::vector<int64_t>& all_tokens, int max_new_tokens, int64_t eos_token_id,
              const std::function<void(int64_t)>& token_callback);

  void init_io();

  // Per-qb constants (must match the compiled graph).
  static constexpr int kBQ = 32;
  static constexpr int kBK = 32;
  static constexpr int kTopK = 8;
  static constexpr int kTopKBK = kTopK * kBK;             // 256
  static constexpr int kHistKBK = (kTopK - 1) * kBK;      // 224
  // Mask convention: ACTIVE = quantized value 65535 (zero_point = 65535 → real 0);
  // MASKED = 0 (real = -scale * 65535, very negative).
  static constexpr uint16_t kMaskActive = 65535;
  static constexpr uint16_t kMaskMasked = 0;

 private:
  // Fill input_tensors_ for one qb dispatch.
  //   qb_in_chunk  = qb index within the current prefill chunk (0..num_qb-1)
  //   qb_global    = global qb index (start_pos/BQ + qb_in_chunk)
  void prepare_one_qb(const std::vector<int64_t>& prompt_tokens, int64_t prompt_pos, int64_t global_pos,
                      int qb_in_chunk, int qb_global);

  // CPU-side block selection: pick top_k - 1 HISTORICAL k-block indices per
  // (head, this qb). Sliding-window "most-recent historical blocks" policy.
  void selectTopKBlocks(int qb_global, std::vector<int>& sel /* [num_attention_heads * (top_k-1)] */);

  // Gather K and V from CPU-side uint8 KV cache into K_arranged_[layer] and
  // V_arranged_[layer] for this qb's HISTORICAL slots only. Diagonal slot
  // is handled in-layer by the model (concat with this dispatch's own K/V).
  void gather_one_qb(int layer, int qb_global, const std::vector<int>& sel);

  // Build the per-qb combined mask:
  //   slot s in [0, top_k-1):
  //     - if qb_global > s (real history): all entries kMaskActive
  //     - else (padding): all entries kMaskMasked
  //   slot top_k-1 (diagonal): row q, col c in [0, kBK):
  //     - kMaskActive if c <= q else kMaskMasked
  // mask shape: [1, 1, kBQ, kTopKBK].
  void build_mask(int qb_global);

 private:
  std::unique_ptr<QnnAOTModule> module_;
  KVCacheManager<uint8_t>* kv_manager_;
  QnnAOTConfig config_;

  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;

  std::vector<Tensor> k_arranged_;  // shape [Hq, 1, D, (top_k-1)*kBK] uint8
  std::vector<Tensor> v_arranged_;  // shape [Hq, 1, (top_k-1)*kBK, D] uint8
  Tensor mask_;                      // shape [1, 1, kBQ, kTopKBK]      uint16
};

}  // namespace mllm::qnn::aot
