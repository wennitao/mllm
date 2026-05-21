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

namespace mllm::qnn { class QNNBackend; }  // fwd-decl for the runtime score context

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

  // Enable attention-score-based block selection (XAttention-style). Per layer:
  //   q_zp     = q_rope_add_0_output_qdq.fake_quant.zero_point  (center uint16 Q)
  //   q_scale  = q_rope_add_0_output_qdq.fake_quant.scale       (dequant Q)
  //   k_scale  = k_cast_to_int8_qdq.fake_quant.scale            (dequant K, zp=128)
  // Scales calibrate the softmax temperature so per-query normalization can
  // surface the relevant block over token-norm bulk. If never called, selection
  // falls back to the sink+recent+random policy. See block_selection.md.
  void enableScoreBasedSelection(const std::vector<int32_t>& q_zp, const std::vector<float>& q_scale,
                                 const std::vector<float>& k_scale);

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
  // `slot` selects which double-buffer the per-qb output goes into (for the
  // pipelined prefill the worker fills slot (qb+1)%2 while the NPU consumes
  // slot qb%2).
  void build_mask(int qb_global, int slot);
  void selectTopKBlocks(int layer, int qb_global, std::vector<int>& sel);
  void gather_one_qb(int layer, int qb_global, const std::vector<int>& sel, int slot);

  // Attention-score-based selection (XAttention-style, CPU). For query block
  // qb_global of `layer`, score each historical block kb in [0, qb_global) per
  // head into scores[h*qb_global + kb] via a reduced antidiagonal Qr·Krᵀ +
  // history-only softmax + block-pool (kBLAS on the small causal slice). Uses
  // q_full_ (Q), the KV-cache K, the per-layer Q zero-point/scale + K scale.
  // Returns false if score params absent.
  bool computeBlockScores(int layer, int qb_global, std::vector<float>& scores);

  // Big-M path helper (MLLM_BLOCKSEL_BIGM): build qr_layer_ + kc_layer_ and the
  // full reduced logits grid logits_full_ once for `layer` (cached by
  // logits_layer_idx_). Returns false if score params absent / no history.
  bool computeLayerLogits(int layer);

  // Strided copy: full-Sq chunk-boundary buffer ↔ per-qb staging `slot`.
  // qb = global qb index (always within this prefill chunk for now).
  void stage_q_qb(int qb, int slot);                           // q_full[1,Hq,Sq,D]      → q_qb[slot][1,Hq,BQ,D]      (uint16)
  void stage_k_curr_qb(int qb, int slot);                      // k_curr_full[1,Hkv,D,Sq] → k_curr_qb[slot][1,Hkv,D,BQ] (uint8)
  void stage_v_curr_qb(int qb, int slot);                      // v_curr_full[1,Hkv,Sq,D] → v_curr_qb[slot][1,Hkv,BQ,D] (uint8)
  void writeback_attn_output_qb(int qb, int slot);             // attn_output_qb[slot][1,Hq,BQ,D] → attn_output_full[1,Hq,Sq,D] (uint16)

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
  // NPU block-selection scoring (env MLLM_BLOCKSEL_NPU; needs the baked "score"
  // graph, FIXED stride S=8). One shared graph dispatched per layer: logits =
  // qr · kc. qr [Hq,Lr,SD] fp16, kc PRE-TRANSPOSED [Hq,SD,Lr] fp16, logits
  // [Hq,Lr,Lr] fp16 — all kQNN (model context, no 2nd-context issue).
  // The score graph is built at RUNTIME in a 2nd HTP context (beginAuxContext)
  // that shares the model context's spill-fill group — so it stays out of the
  // PD-saturated model context and needs no AOT recompile. qnn_backend_ is the
  // QNN backend used to build + dispatch it.
  mllm::qnn::QNNBackend* qnn_backend_ = nullptr;
  Tensor score_qr_, score_kc_, score_logits_;
  bool npu_score_ = false;
  static constexpr int kScoreStride = 8;
  // The score graph processes Hq/kScoreHeadSplit heads per dispatch so its
  // working set (qr+kc+out) fits the 8 MB VTCM → ~0 DDR spill-fill (a full-Hq
  // dispatch spilled 43.8 MB, the largest graph in the context, blowing the V79
  // PD cap). The runner loops the splits per layer. See block_selection.md.
  static constexpr int kScoreHeadSplit = 2;

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

  // Mask, double-buffered (slot qb%2). One per pipeline slot so the worker can
  // build qb+1's mask while the NPU consumes qb's.
  std::vector<Tensor> mask_;  // size 2

  // Per-layer full-Sq chunk-boundary buffers.
  std::vector<Tensor> residual_full_;     // [1, Sq, hidden]            uint16  size L
  std::vector<Tensor> q_full_;            // [1, Hq, Sq, D]             uint16  size L
  std::vector<Tensor> k_curr_full_;       // [1, Hkv, D, Sq]            uint8   size L
  std::vector<Tensor> v_curr_full_;       // [1, Hkv, Sq, D]            uint8   size L
  std::vector<Tensor> attn_output_full_;  // [1, Hq, Sq, D]             uint16  size L

  // Per-qb staging buffers, DOUBLE-BUFFERED (size 2, indexed by qb%2). The NPU
  // attn dispatch for qb reads slot qb%2 while the (optional) pipeline worker
  // fills slot (qb+1)%2 for the next qb — the two never touch the same slot, so
  // CPU gather/stage overlaps the NPU compute. Bound to the attn graph by
  // POSITION (no per-iter rename), so 2 stable buffers shared across all 28
  // layers' attn graphs is safe (same as the mask). Also cheaper than the old
  // per-layer (L×) layout: BQ-sized, ~1 MB total.
  std::vector<Tensor> q_qb_;             // [1, Hq, BQ, D]       uint16  size 2
  std::vector<Tensor> k_curr_qb_;        // [1, Hkv, D, BQ]      uint8   size 2
  std::vector<Tensor> v_curr_qb_;        // [1, Hkv, BQ, D]      uint8   size 2
  std::vector<Tensor> attn_output_qb_;   // [1, Hq, BQ, D]       uint16  size 2

  // K_arranged / V_arranged gather targets, double-buffered (size 2).
  std::vector<Tensor> k_arranged_;  // [Hq, 1, D, kHistKBK]    uint8   size 2
  std::vector<Tensor> v_arranged_;  // [Hq, 1, kHistKBK, D]    uint8   size 2

  // Logits output of chunk_L.
  Tensor logits_;          // [1, Sq, vocab]        uint16

  // ----- Score-based block selection (optional) ----------------------------
  bool score_based_ = false;          // set by enableScoreBasedSelection
  std::vector<int32_t> q_zp_;         // per-layer Q zero-point (size L)
  std::vector<float> q_scale_;        // per-layer Q dequant scale (size L)
  std::vector<float> k_scale_;        // per-layer K dequant scale (size L)
  // Per-layer centered+de-transposed K cache, rebuilt lazily when the layer
  // changes: [Hq, Sq, D] fp32, kc[h,pos,d] = K_cache[h/group, d, pos] - 128.
  // Per-layer de-transposed+centered+reshaped K as an mllm Tensor
  // [1, Hq, Lr, S·D] (Lr = Sq/S), built once per layer and fed (whole) to every
  // qb's score matmul — so the per-qb redundant K-slice memcpy is gone. Q is
  // rebuilt fresh per qb, so the (Q,K) matmul key is unique → no engine
  // memoization staleness from reusing K.
  Tensor kc_layer_;
  int kc_layer_idx_ = -1;
  // Big-M scoring path (env MLLM_BLOCKSEL_BIGM): instead of one small matmul per
  // qb (M=BQr tiny → poor gemm efficiency, 24 calls/layer), build the reduced
  // full-sequence Q once (qr_layer_ [1,Hq,Lr,SD], same reversal as the per-qb Q)
  // and do ONE big matmul/layer qr_layer_·kc_layerᵀ → logits_full_ [1,Hq,Lr,Lr]
  // (M=Lr=128, efficient). Each qb then just slices+softmax+pools logits_full_
  // (no matmul). ~2× the FLOPs (full square incl. upper triangle) but M=128 vs 4.
  Tensor qr_layer_;
  Tensor logits_full_;
  int logits_layer_idx_ = -1;
  // Scoring instrumentation (µs, summed; printed if MLLM_BLOCKSEL_TIMING set).
  // tot = whole computeBlockScores; kc = per-layer K de-transpose; prep = Q+K
  // build; mm = matmul; sm = softmax+pool.
  long long score_mm_us_ = 0, score_tot_us_ = 0, score_kc_us_ = 0, score_prep_us_ = 0, score_sm_us_ = 0;
  // Per-qb stage profiling (µs, summed across all layers; printed if
  // MLLM_QB_PROFILE set). sel = build_mask+selectTopKBlocks (CPU selection),
  // gat = gather_one_qb+stage_* (K/V/Q staging), npu = dispatch_attn (HTP).
  long long qp_sel_us_ = 0, qp_gat_us_ = 0, qp_npu_us_ = 0;
  // NOTE: two scoring optimizations were tried and REVERTED (see
  // docs/qnn_backend/block_selection.md):
  //  - reusing fixed scratch Tensors across calls → mllm engine memoizes matmul
  //    by input-tensor identity → stale cached result → wrong selection.
  //  - "Lever 1" full-Sq-grid single matmul per layer → 2× FLOPs (masked upper
  //    triangle) and kBLAS is buggy at that large batched shape; scoring is
  //    FLOP-bound, so it was slower. The per-qb kBLAS scorer is the fast+correct
  //    choice on CPU; real speedup needs the NPU (HMX) for the score matmul.
};

}  // namespace mllm::qnn::aot
