// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/core/Tensor.hpp"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace mllm::qnn::aot {

// Prompt processor for the LPBQ SHA per-qb CAUSAL block-sparse Qwen3 model,
// SPLIT-PREFILL variant. Drives the 2L+1 chunked graph context emitted by
// mllm-qwen3-aot-sha-blocksparse-causal-split-c.
//
// Dispatch flow at prefill:
//   chunk_0(input_ids[1,Sq], position_ids[1,Sq])
//       → residual_0[1,Sq,H], q_0[1,Hq,Sq,D], k_curr_0[1,Hkv,D,Sq], v_curr_0[1,Hkv,Sq,D]
//
//   for i in [0, L):
//       (cache write) copy k_curr_i / v_curr_i into KV cache for use by decode
//       for qb in [0, num_qb):
//           CPU: build per-qb mask, gather K_hist/V_hist into K_arranged_i/V_arranged_i
//           CPU: stage q_i / k_curr_i / v_curr_i per-qb slices into [1,Hq,BQ,D] /
//                [1,Hkv,D,BQ] / [1,Hkv,BQ,D] staging buffers (Sq isn't a contiguous
//                slice dim, so strided memcpy)
//           NPU: attn_i(q_qb, k_curr_qb, v_curr_qb, K_arranged_i, V_arranged_i, mask)
//                → attn_output_qb [1,Hq,BQ,D]
//           CPU: copy attn_output_qb back into attn_output_i at qb offset
//       chunk_{i+1}(residual_i, attn_output_i, position_ids)
//           → residual_{i+1}, q_{i+1}, k_curr_{i+1}, v_curr_{i+1}
//
//   chunk_L (last) outputs only logits[1,Sq,vocab].
//
// Memory:
//   * Per-layer full-Sq chunk-boundary buffers (residual, q, k_curr, v_curr,
//     attn_output) live in rpcmem and are re-used across dispatches.
//   * Per-qb staging buffers (q_qb, k_curr_qb, v_curr_qb, attn_output_qb) are
//     also rpcmem; populated by strided memcpy from the full-Sq buffers.
//   * K_arranged / V_arranged per-layer (runner-gathered, same as monolithic).
//   * One uint16-quant mask shared across all attn_i dispatches in a qb.
//
// Decode is NOT implemented yet — the split graphs are compiled for full Sq.
// For decode we need either (a) a parallel ar_len=1 split context, or (b) a
// fallback to the monolithic per-qb decode graph. Left as follow-up.
class ShaBlockSparsePromptProcessorSplit {
 public:
  ShaBlockSparsePromptProcessorSplit(KVCacheManager<uint8_t>* kv_manager, QnnAOTConfig config, int Sq);

  ~ShaBlockSparsePromptProcessorSplit();

  void init_io();

  // Runs the split-prefill chunk dispatch flow and returns the first sampled
  // token (greedy from chunk_L's logits at the last position).
  int64_t prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos = 0);

  static constexpr int kBQ = 32;
  static constexpr int kBK = 32;
  static constexpr int kTopK = 8;
  static constexpr int kTopKBK = kTopK * kBK;        // 256
  static constexpr int kHistKBK = (kTopK - 1) * kBK; // 224
  static constexpr uint16_t kMaskActive = 65535;
  static constexpr uint16_t kMaskMasked = 0;

 private:
  // Build mask + select historical block indices + gather K/V (matches the
  // monolithic processor's helpers; copied here for self-containedness).
  void build_mask(int qb_global);
  void selectTopKBlocks(int qb_global, std::vector<int>& sel);
  void gather_one_qb(int layer, int qb_global, const std::vector<int>& sel);

  // Strided copy: full-Sq chunk-boundary buffer ↔ per-qb staging.
  // qb = global qb index (always within this prefill chunk for now).
  void stage_q_qb(int layer, int qb);                          // q_i[1,Hq,Sq,D]      → q_qb[1,Hq,BQ,D]      (uint16)
  void stage_k_curr_qb(int layer, int qb);                     // k_curr_i[1,Hkv,D,Sq] → k_curr_qb[1,Hkv,D,BQ] (uint8)
  void stage_v_curr_qb(int layer, int qb);                     // v_curr_i[1,Hkv,Sq,D] → v_curr_qb[1,Hkv,BQ,D] (uint8)
  void writeback_attn_output_qb(int layer, int qb);            // attn_output_qb[1,Hq,BQ,D] → attn_output_i[1,Hq,Sq,D] (uint16)

  // Copy per-layer K_curr / V_curr (full-Sq) into the runner's uint8 KV cache
  // so subsequent dispatches' gathers can pull historical blocks. Run once
  // per layer at the end of its per-qb attention loop.
  void copy_kv_to_cache(int layer, int64_t base_pos, int64_t n_tokens);

  // Module getters: the runner names QnnAOTModule instances after their
  // compile-time graph names ("chunk_0", "chunk_1", ..., "attn_0", ...).
  QnnAOTModule& chunk_module(int i) { return *chunks_[i]; }   // i in [0, L]
  QnnAOTModule& attn_module(int i)  { return *attns_[i]; }    // i in [0, L)

 private:
  KVCacheManager<uint8_t>* kv_manager_;
  QnnAOTConfig config_;
  int Sq_;           // full prefill sequence length (compile-time --sq value)
  int num_qb_;       // = Sq_ / kBQ
  int L_;            // num_hidden_layers
  int Hq_;           // num_attention_heads
  int Hkv_;          // num_key_value_heads
  int D_;            // head_dim
  int hidden_;       // hidden_size
  int vocab_;        // vocab_size

  // 2L+1 graphs as QnnAOTModule instances. Each instance carries its
  // compile-time graph name.
  std::vector<std::unique_ptr<QnnAOTModule>> chunks_;  // size L+1
  std::vector<std::unique_ptr<QnnAOTModule>> attns_;   // size L

  // ----- Per-chunk I/O tensor lists ----------------------------------------
  // Indexing convention matches the compile driver's trace_inputs map order:
  //   chunk_0 inputs: [input_ids, position_ids]
  //   chunk_i (1..L-1) inputs: [residual_pre_attn_{i-1}, attn_output_{i-1}, position_ids]
  //   chunk_L inputs: [residual_pre_attn_{L-1}, attn_output_{L-1}]
  //   attn_i inputs: [q_i_qb, k_curr_i_qb, v_curr_i_qb, K_arranged_i, V_arranged_i, mask]
  std::vector<std::vector<Tensor>> chunk_in_;   // size L+1
  std::vector<std::vector<Tensor>> chunk_out_;  // size L+1
  std::vector<std::vector<Tensor>> attn_in_;    // size L
  std::vector<std::vector<Tensor>> attn_out_;   // size L

  // Top-level inputs (chunk_0 inputs) and shared position_ids.
  Tensor input_ids_;
  Tensor position_ids_;
  Tensor last_token_index_;  // [1, 1] int32 — chunk_L gathers this position pre-lm_head

  // Mask (shared across all attn dispatches within a qb).
  Tensor mask_;

  // Per-layer full-Sq chunk-boundary buffers.
  std::vector<Tensor> residual_full_;     // [1, Sq, hidden]            uint16  size L
  std::vector<Tensor> q_full_;            // [1, Hq, Sq, D]             uint16  size L
  std::vector<Tensor> k_curr_full_;       // [1, Hkv, D, Sq]            uint8   size L
  std::vector<Tensor> v_curr_full_;       // [1, Hkv, Sq, D]            uint8   size L
  std::vector<Tensor> attn_output_full_;  // [1, Hq, Sq, D]             uint16  size L

  // Per-layer per-qb staging buffers. Originally these were 4 shared rpcmem
  // regions used across all 28 attn dispatches (renamed each iteration to
  // match the layer's compile-time tensor name), but sharing storage across
  // distinct compiled graph inputs/outputs and renaming each layer triggered
  // QNNAllocator warnings and is suspected of causing degenerate model output.
  // Allocating one buffer per layer per kind sidesteps the rename + any
  // hidden allocator-state aliasing. Cost: 28 × (~256 KB + ~32 KB + ~32 KB
  // + ~256 KB) ≈ 16 MB total rpcmem, trivial.
  std::vector<Tensor> q_qb_;             // [1, Hq, BQ, D]       uint16  size L
  std::vector<Tensor> k_curr_qb_;        // [1, Hkv, D, BQ]      uint8   size L
  std::vector<Tensor> v_curr_qb_;        // [1, Hkv, BQ, D]      uint8   size L
  std::vector<Tensor> attn_output_qb_;   // [1, Hq, BQ, D]       uint16  size L

  // Per-layer K_arranged / V_arranged (runner-gathered, same shape as
  // monolithic).
  std::vector<Tensor> k_arranged_;  // [Hq, 1, D, kHistKBK]    uint8   size L
  std::vector<Tensor> v_arranged_;  // [Hq, 1, kHistKBK, D]    uint8   size L

  // Logits output of chunk_L.
  Tensor logits_;          // [1, Sq, vocab]        uint16
};

}  // namespace mllm::qnn::aot
