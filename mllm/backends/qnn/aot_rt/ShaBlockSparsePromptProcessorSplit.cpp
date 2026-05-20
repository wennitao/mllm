// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mllm::qnn::aot {

ShaBlockSparsePromptProcessorSplit::ShaBlockSparsePromptProcessorSplit(KVCacheManager<uint8_t>* kv_manager,
                                                                       QnnAOTConfig config, int Sq)
    : kv_manager_(kv_manager), config_(config), Sq_(Sq) {
  L_ = config_.num_layers;
  Hq_ = config_.num_attention_heads;
  Hkv_ = config_.num_heads;
  D_ = config_.head_dim;
  hidden_ = Hq_ * D_;  // Qwen3 has hidden_size == Hq * D
  vocab_ = config_.vocab_size;
  num_qb_ = Sq_ / kBQ;
  MLLM_RT_ASSERT_EQ(Sq_ % kBQ, 0);

  chunks_.reserve(L_ + 1);
  attns_.reserve(L_);
  for (int i = 0; i <= L_; ++i) {
    chunks_.push_back(std::make_unique<QnnAOTModule>("chunk_" + std::to_string(i)));
    chunks_.back()->to(kQNN);
  }
  for (int i = 0; i < L_; ++i) {
    attns_.push_back(std::make_unique<QnnAOTModule>("attn_" + std::to_string(i)));
    attns_.back()->to(kQNN);
  }
}

ShaBlockSparsePromptProcessorSplit::~ShaBlockSparsePromptProcessorSplit() {
  // Same cleanup discipline as the monolithic processor: drop owned output
  // tensor refs first, then clear input lists.
  for (auto& m : chunks_) {
    if (m) m->setOutputTensors({});
  }
  for (auto& m : attns_) {
    if (m) m->setOutputTensors({});
  }
  chunk_in_.clear();
  chunk_out_.clear();
  attn_in_.clear();
  attn_out_.clear();
  residual_full_.clear();
  q_full_.clear();
  k_curr_full_.clear();
  v_curr_full_.clear();
  attn_output_full_.clear();
  k_arranged_.clear();
  v_arranged_.clear();
}

// ============================================================================
// init_io: allocate every rpcmem buffer once + build per-module I/O lists.
// ============================================================================
void ShaBlockSparsePromptProcessorSplit::init_io() {
  // ----- top-level (chunk_0) inputs ----------------------------------------
  input_ids_ = Tensor::empty({1, Sq_}, kInt32, kQNN).alloc();
  input_ids_.setName("input_ids");
  position_ids_ = Tensor::empty({1, Sq_}, kInt32, kQNN).alloc();
  position_ids_.setName("position_ids");

  // ----- mask + K/V_arranged (per layer, runner-gathered) ------------------
  mask_ = Tensor::empty({1, 1, kBQ, kTopKBK}, kUInt16, kQNN).alloc();
  std::memset(mask_.ptr<void>(), 0, mask_.bytes());
  mask_.setName("mask");

  k_arranged_.clear();
  v_arranged_.clear();
  k_arranged_.reserve(L_);
  v_arranged_.reserve(L_);
  for (int i = 0; i < L_; ++i) {
    auto k_arr = Tensor::empty({Hq_, 1, D_, kHistKBK}, kUInt8, kQNN).alloc();
    std::memset(k_arr.ptr<void>(), 0, k_arr.bytes());
    k_arr.setName("K_arranged_" + std::to_string(i));
    k_arranged_.push_back(k_arr);

    auto v_arr = Tensor::empty({Hq_, 1, kHistKBK, D_}, kUInt8, kQNN).alloc();
    std::memset(v_arr.ptr<void>(), 0, v_arr.bytes());
    v_arr.setName("V_arranged_" + std::to_string(i));
    v_arranged_.push_back(v_arr);
  }

  // ----- full-Sq chunk-boundary buffers (reused, not L× — see below) -------
  // The chunks run strictly sequentially, so these don't need L distinct
  // copies (which cost ~410 MB at Sq=1024 and overflow the PD I/O-registration
  // budget). Live-range analysis:
  //   * q/k_curr/v_curr_full : produced by chunk_i, consumed by attn_i in the
  //     same layer iteration → ONE buffer, reused every layer.
  //   * attn_output_full      : written by attn_i (per-qb writeback), read by
  //     chunk_{i+1}; never read+written concurrently → ONE buffer.
  //   * residual_full         : chunk_{i+1} reads R[i] and writes R[i+1] in the
  //     SAME graphExecute (read-write hazard) → TWO buffers, ping-pong on i%2.
  residual_full_.reserve(2);
  for (int p = 0; p < 2; ++p) {
    auto t_res = Tensor::empty({1, Sq_, hidden_}, kFloat16, kQNN).alloc();
    t_res.setName("residual_pre_attn_pp" + std::to_string(p));
    residual_full_.push_back(t_res);
  }
  {
    auto t_q = Tensor::empty({1, Hq_, Sq_, D_}, kUInt16, kQNN).alloc();
    t_q.setName("q_full");
    q_full_.push_back(t_q);

    auto t_kc = Tensor::empty({1, Hkv_, D_, Sq_}, kUInt8, kQNN).alloc();
    t_kc.setName("k_curr_full");
    k_curr_full_.push_back(t_kc);

    auto t_vc = Tensor::empty({1, Hkv_, Sq_, D_}, kUInt8, kQNN).alloc();
    t_vc.setName("v_curr_full");
    v_curr_full_.push_back(t_vc);

    auto t_ao = Tensor::empty({1, Hq_, Sq_, D_}, kFloat16, kQNN).alloc();
    t_ao.setName("attn_output_full");
    attn_output_full_.push_back(t_ao);
  }

  // ----- per-layer per-qb staging buffers ----------------------------------
  q_qb_.clear();
  k_curr_qb_.clear();
  v_curr_qb_.clear();
  attn_output_qb_.clear();
  q_qb_.reserve(L_);
  k_curr_qb_.reserve(L_);
  v_curr_qb_.reserve(L_);
  attn_output_qb_.reserve(L_);
  for (int i = 0; i < L_; ++i) {
    auto si = std::to_string(i);
    auto t_q = Tensor::empty({1, Hq_, kBQ, D_}, kUInt16, kQNN).alloc();
    t_q.setName("q_" + si + "_qb");
    q_qb_.push_back(t_q);

    auto t_kc = Tensor::empty({1, Hkv_, D_, kBQ}, kUInt8, kQNN).alloc();
    t_kc.setName("k_curr_" + si + "_qb");
    k_curr_qb_.push_back(t_kc);

    auto t_vc = Tensor::empty({1, Hkv_, kBQ, D_}, kUInt8, kQNN).alloc();
    t_vc.setName("v_curr_" + si + "_qb");
    v_curr_qb_.push_back(t_vc);

    // attn_output_i_qb mirrors attn_output_i (boundary) — fp16.
    auto t_ao = Tensor::empty({1, Hq_, kBQ, D_}, kFloat16, kQNN).alloc();
    t_ao.setName("attn_output_" + si + "_qb");
    attn_output_qb_.push_back(t_ao);
  }

  // ----- last-token index (chunk_L gathers this position before lm_head) ----
  last_token_index_ = Tensor::empty({1, 1}, kInt32, kQNN).alloc();
  last_token_index_.setName("last_token_index");

  // ----- logits output of chunk_L ------------------------------------------
  // lm_head now runs at M=1 (gathered last position), so logits is [1,1,1,vocab].
  logits_ = Tensor::empty({1, 1, 1, vocab_}, kUInt16, kQNN).alloc();
  logits_.setName("logits");

  // ============================================================================
  // Build per-module I/O lists.
  // ============================================================================
  chunk_in_.assign(L_ + 1, {});
  chunk_out_.assign(L_ + 1, {});
  attn_in_.assign(L_, {});
  attn_out_.assign(L_, {});

  // Reused buffers: q/k/v_curr/attn_output are single (index 0); residual
  // ping-pongs on layer parity. R[i] (residual produced by chunk_i) lives in
  // residual_full_[i % 2]; chunk_{i+1} reads R[i] and writes R[i+1] into the
  // other slot, avoiding the in-place read-write hazard.
  // chunk_0: inputs = [input_ids, position_ids]; outputs = [R0, q, k_curr, v_curr]
  chunk_in_[0]  = {input_ids_, position_ids_};
  chunk_out_[0] = {residual_full_[0], q_full_[0], k_curr_full_[0], v_curr_full_[0]};

  // chunk_i (1..L-1): inputs = [R_{i-1}, attn_output, position_ids];
  //                   outputs = [R_i, q, k_curr, v_curr]
  for (int i = 1; i < L_; ++i) {
    chunk_in_[i]  = {residual_full_[(i - 1) % 2], attn_output_full_[0], position_ids_};
    chunk_out_[i] = {residual_full_[i % 2], q_full_[0], k_curr_full_[0], v_curr_full_[0]};
  }
  // chunk_L: inputs = [R_{L-1}, attn_output, last_token_index]; outputs = [logits]
  chunk_in_[L_]  = {residual_full_[(L_ - 1) % 2], attn_output_full_[0], last_token_index_};
  chunk_out_[L_] = {logits_};

  // attn_i: inputs = [q_qb_i, k_curr_qb_i, v_curr_qb_i, K_arranged_i, V_arranged_i, mask];
  //         outputs = [attn_output_qb_i].
  // Each layer gets its OWN per-qb staging buffers — distinct rpcmem regions
  // and stable names, so QNN's per-graph tensor wrappers don't share storage.
  for (int i = 0; i < L_; ++i) {
    attn_in_[i]  = {q_qb_[i], k_curr_qb_[i], v_curr_qb_[i], k_arranged_[i], v_arranged_[i], mask_};
    attn_out_[i] = {attn_output_qb_[i]};
  }

  // Bind output tensors on each module (the QnnAOTModule treats output_tensors_
  // as a fixed list to fill on graphExecute).
  for (int i = 0; i <= L_; ++i) chunks_[i]->setOutputTensors(chunk_out_[i]);
  for (int i = 0; i < L_; ++i) attns_[i]->setOutputTensors(attn_out_[i]);
}

// ============================================================================
// Helpers — same logic as the monolithic processor, copied for self-containedness.
// ============================================================================
void ShaBlockSparsePromptProcessorSplit::selectTopKBlocks(int qb_global, std::vector<int>& sel) {
  const int n_hist = kTopK - 1;
  sel.assign((size_t)Hq_ * n_hist, 0);
  for (int h = 0; h < Hq_; ++h) {
    int* hs = sel.data() + (size_t)h * n_hist;
    if (qb_global <= n_hist) {
      // All historical blocks fit — include them all, pad the rest.
      const int hist_slots = qb_global;
      for (int s = 0; s < hist_slots; ++s) hs[s] = qb_global - 1 - s;
      for (int s = hist_slots; s < n_hist; ++s) hs[s] = 0;  // padding slot (masked)
    } else {
      // Selection needed. Mirror ShaBlockSparsePromptProcessor (mono):
      //   slot 0       = block 0      (ATTENTION SINK — dropping it collapses
      //                                the softmax once qb_global crosses n_hist).
      //   slot 1       = block qb-1   (most-recent / sliding window).
      //   slots 2..    = (n_hist-2) distinct random blocks from middle [1, qb-2],
      //                  per-head, (qb_global, head)-seeded for reproducibility.
      hs[0] = 0;              // sink
      hs[1] = qb_global - 1;  // most-recent
      const int mid_lo = 1;
      const int mid_hi = qb_global - 2;  // inclusive
      const int mid_count = mid_hi - mid_lo + 1;
      const int need = n_hist - 2;
      std::mt19937 rng((uint32_t)(qb_global * 1315423911u + (uint32_t)h * 2654435761u + 0x9e3779b9u));
      std::vector<int> cand(mid_count);
      for (int i = 0; i < mid_count; ++i) cand[i] = mid_lo + i;
      for (int i = 0; i < need; ++i) {
        if (i < mid_count) {
          std::uniform_int_distribution<int> dist(i, mid_count - 1);
          int j = dist(rng);
          std::swap(cand[i], cand[j]);
          hs[2 + i] = cand[i];
        } else {
          hs[2 + i] = 0;
        }
      }
    }
  }
}

void ShaBlockSparsePromptProcessorSplit::gather_one_qb(int layer, int qb_global, const std::vector<int>& sel) {
  const int Hkv = Hkv_;
  const int Hq = Hq_;
  const int group = Hq / Hkv;
  const int D = D_;
  const int CL = config_.context_len;
  const int n_hist = kTopK - 1;
  const int max_kv_len = CL - kBQ;

  const uint8_t* pk = kv_manager_->getKCache()[layer].buffer;
  const uint8_t* pv = kv_manager_->getVCache()[layer].buffer;
  uint8_t* kdst = (uint8_t*)k_arranged_[layer].ptr<uint8_t>();
  uint8_t* vdst = (uint8_t*)v_arranged_[layer].ptr<uint8_t>();

  const size_t past_key_head_stride = (size_t)D * max_kv_len;
  const size_t past_value_head_stride = (size_t)max_kv_len * D;
  const size_t k_arr_head_stride = (size_t)D * n_hist * kBK;
  const size_t v_arr_head_stride = (size_t)n_hist * kBK * D;

  for (int h = 0; h < Hq; ++h) {
    const int kv_head_idx = h / group;
    const int* hs = sel.data() + (size_t)h * n_hist;
    const uint8_t* pk_head = pk + kv_head_idx * past_key_head_stride;
    const uint8_t* pv_head = pv + kv_head_idx * past_value_head_stride;
    uint8_t* kdst_head = kdst + (size_t)h * k_arr_head_stride;
    uint8_t* vdst_head = vdst + (size_t)h * v_arr_head_stride;

    for (int s = 0; s < n_hist; ++s) {
      const int k_pos = hs[s];
      const int k_off = k_pos * kBK;
      const bool valid = (k_pos >= 0) && (k_off + kBK <= max_kv_len);
      uint8_t* vds = vdst_head + (size_t)s * kBK * D;
      if (valid) {
        std::memcpy(vds, pv_head + (size_t)k_off * D, (size_t)kBK * D);
      } else {
        std::fill_n(vds, (size_t)kBK * D, (uint8_t)128);
      }
      if (valid) {
        for (int c = 0; c < D; ++c) {
          uint8_t* kdst_row = kdst_head + (size_t)c * n_hist * kBK + (size_t)s * kBK;
          const uint8_t* kpk_row = pk_head + (size_t)c * max_kv_len + k_off;
          std::memcpy(kdst_row, kpk_row, (size_t)kBK);
        }
      } else {
        for (int c = 0; c < D; ++c) {
          uint8_t* kdst_row = kdst_head + (size_t)c * n_hist * kBK + (size_t)s * kBK;
          std::fill_n(kdst_row, (size_t)kBK, (uint8_t)128);
        }
      }
    }
  }
}

void ShaBlockSparsePromptProcessorSplit::build_mask(int qb_global) {
  const int hist_slots = std::min(kTopK - 1, qb_global);
  uint16_t* p = mask_.ptr<uint16_t>();
  for (int q = 0; q < kBQ; ++q) {
    uint16_t* row = p + (size_t)q * kTopKBK;
    for (int s = 0; s < kTopK - 1; ++s) {
      const uint16_t v = (s < hist_slots) ? kMaskActive : kMaskMasked;
      for (int j = 0; j < kBK; ++j) row[s * kBK + j] = v;
    }
    uint16_t* diag = row + (kTopK - 1) * kBK;
    for (int c = 0; c < kBK; ++c) diag[c] = (c <= q) ? kMaskActive : kMaskMasked;
  }
}

// ============================================================================
// Strided staging copies between full-Sq buffers and per-qb staging.
// (Sq is the second-to-last dim for Q / attn_output, so a per-qb slice
// isn't a single contiguous memcpy.)
// ============================================================================
void ShaBlockSparsePromptProcessorSplit::stage_q_qb(int layer, int qb) {
  // q_full: [1, Hq, Sq, D] uint16. Per-head h: rows [qb*BQ, (qb+1)*BQ) contiguous BQ*D uint16.
  // q_qb:   [1, Hq, BQ, D] uint16.
  const uint16_t* src = q_full_[0].ptr<uint16_t>();  // single reused buffer
  uint16_t* dst = q_qb_[layer].ptr<uint16_t>();
  const size_t head_stride_src = (size_t)Sq_ * D_;
  const size_t head_stride_dst = (size_t)kBQ * D_;
  const size_t bytes_per_head = (size_t)kBQ * D_ * sizeof(uint16_t);
  for (int h = 0; h < Hq_; ++h) {
    std::memcpy(dst + h * head_stride_dst,
                src + h * head_stride_src + (size_t)qb * kBQ * D_,
                bytes_per_head);
  }
}

void ShaBlockSparsePromptProcessorSplit::stage_k_curr_qb(int layer, int qb) {
  // k_curr_full: [1, Hkv, D, Sq] uint8. Per (h, d): Sq columns, take [qb*BQ, (qb+1)*BQ) — BQ contiguous bytes.
  // k_curr_qb:   [1, Hkv, D, BQ] uint8.
  const uint8_t* src = k_curr_full_[0].ptr<uint8_t>();  // single reused buffer
  uint8_t* dst = k_curr_qb_[layer].ptr<uint8_t>();
  const size_t head_stride_src = (size_t)D_ * Sq_;
  const size_t head_stride_dst = (size_t)D_ * kBQ;
  for (int h = 0; h < Hkv_; ++h) {
    for (int c = 0; c < D_; ++c) {
      std::memcpy(dst + h * head_stride_dst + (size_t)c * kBQ,
                  src + h * head_stride_src + (size_t)c * Sq_ + (size_t)qb * kBQ,
                  (size_t)kBQ);
    }
  }
}

void ShaBlockSparsePromptProcessorSplit::stage_v_curr_qb(int layer, int qb) {
  // v_curr_full: [1, Hkv, Sq, D] uint8. Per head: BQ contiguous rows of D bytes.
  // v_curr_qb:   [1, Hkv, BQ, D] uint8.
  const uint8_t* src = v_curr_full_[0].ptr<uint8_t>();  // single reused buffer
  uint8_t* dst = v_curr_qb_[layer].ptr<uint8_t>();
  const size_t head_stride_src = (size_t)Sq_ * D_;
  const size_t head_stride_dst = (size_t)kBQ * D_;
  const size_t bytes_per_head = (size_t)kBQ * D_;
  for (int h = 0; h < Hkv_; ++h) {
    std::memcpy(dst + h * head_stride_dst,
                src + h * head_stride_src + (size_t)qb * kBQ * D_,
                bytes_per_head);
  }
}

void ShaBlockSparsePromptProcessorSplit::writeback_attn_output_qb(int layer, int qb) {
  // attn_output_full: [1, Hq, Sq, D] uint16. Write qb-slice from attn_output_qb_[layer] [1, Hq, BQ, D].
  const uint16_t* src = attn_output_qb_[layer].ptr<uint16_t>();
  uint16_t* dst = attn_output_full_[0].ptr<uint16_t>();  // single reused buffer
  const size_t head_stride_dst = (size_t)Sq_ * D_;
  const size_t head_stride_src = (size_t)kBQ * D_;
  const size_t bytes_per_head = (size_t)kBQ * D_ * sizeof(uint16_t);
  for (int h = 0; h < Hq_; ++h) {
    std::memcpy(dst + h * head_stride_dst + (size_t)qb * kBQ * D_,
                src + h * head_stride_src,
                bytes_per_head);
  }
}

// ============================================================================
// Copy the layer's just-computed full-Sq K_curr/V_curr into the runner's KV
// cache (same layout as the monolithic cache: K is [1,Hkv,D,max_kv_len];
// V is [1,Hkv,max_kv_len,D]). Used by decode to access historical positions.
// `base_pos` is the absolute position of the first token in this prefill
// chunk; `n_tokens` is how many tokens of K_curr/V_curr to commit (≤ Sq_).
// ============================================================================
void ShaBlockSparsePromptProcessorSplit::copy_kv_to_cache(int layer, int64_t base_pos, int64_t n_tokens) {
  const int CL = config_.context_len;
  const int max_kv_len = CL - kBQ;
  if (base_pos + n_tokens > max_kv_len) {
    // Truncate; the prompt exceeds the KV cache. Caller should clamp prompts.
    n_tokens = std::max<int64_t>(0, (int64_t)max_kv_len - base_pos);
  }

  // K: src [1, Hkv, D, Sq] uint8 → dst [1, Hkv, D, max_kv_len] uint8.
  //    For each head h and row c: copy n_tokens bytes from src row to
  //    dst row starting at column base_pos. Strided write.
  const uint8_t* k_src = k_curr_full_[0].ptr<uint8_t>();  // single reused buffer
  uint8_t* k_dst = kv_manager_->getKCache()[layer].buffer;
  for (int h = 0; h < Hkv_; ++h) {
    for (int c = 0; c < D_; ++c) {
      std::memcpy(k_dst + ((size_t)h * D_ + c) * max_kv_len + base_pos,
                  k_src + ((size_t)h * D_ + c) * Sq_,
                  (size_t)n_tokens);
    }
  }

  // V: src [1, Hkv, Sq, D] uint8 → dst [1, Hkv, max_kv_len, D] uint8.
  //    Per head: n_tokens contiguous rows of D bytes; dst rows offset by base_pos.
  const uint8_t* v_src = v_curr_full_[0].ptr<uint8_t>();  // single reused buffer
  uint8_t* v_dst = kv_manager_->getVCache()[layer].buffer;
  for (int h = 0; h < Hkv_; ++h) {
    std::memcpy(v_dst + ((size_t)h * max_kv_len + base_pos) * D_,
                v_src + (size_t)h * Sq_ * D_,
                (size_t)n_tokens * D_);
  }
}

// ============================================================================
// Prefill: chunk_0 → for each layer (per-qb attn loop + chunk_{i+1}) → chunk_L.
// Returns the first sampled token from chunk_L's logits at the last prompt
// position.
//
// Assumes start_pos == 0 (single-chunk prefill at Sq_). Multi-chunk prefill
// (prompt longer than Sq_) is a future extension.
// ============================================================================
int64_t ShaBlockSparsePromptProcessorSplit::prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos) {
  MLLM_RT_ASSERT_EQ(start_pos, 0);  // multi-chunk prefill TBD
  const int64_t num_tokens = (int64_t)prompt_tokens.size();
  MLLM_RT_ASSERT(num_tokens > 0 && num_tokens <= Sq_);
  MLLM_INFO("ShaBlockSparsePromptProcessorSplit: prefill num_tokens={} Sq={} num_qb={}", num_tokens, Sq_, num_qb_);

  // ---- fill input_ids + position_ids (padded with 0 past prompt end) ------
  {
    int32_t* ip = input_ids_.ptr<int32_t>();
    int32_t* pp = position_ids_.ptr<int32_t>();
    for (int i = 0; i < Sq_; ++i) {
      ip[i] = (i < num_tokens) ? (int32_t)prompt_tokens[i] : 0;
      pp[i] = (int32_t)(start_pos + i);
    }
    // chunk_L gathers this position out of [1, Sq, hidden] before lm_head.
    last_token_index_.ptr<int32_t>()[0] = (int32_t)(num_tokens - 1);
  }

  // ---- chunk_0 ------------------------------------------------------------
  {
    std::vector<Tensor> ins = chunk_in_[0];
    chunk_out_[0] = (*chunks_[0])(ins);
  }

  // Tap-point helper: dump layer `dl` K_curr/V_curr (post-RoPE K, post-proj V),
  // first kBQ positions, in dense's [Hkv, D, kBQ] / [Hkv, kBQ, D] layout for a
  // byte-compare with the dense dumps. Called after each chunk produces its
  // layer's k_curr_full/v_curr_full so any layer can be bisected.
  auto dump_layer_kv = [&](int produced_layer) {
    const char* layer_env = std::getenv("MLLM_DUMP_K_LAYER");
    if (!layer_env) return;
    if (std::atoi(layer_env) != produced_layer) return;
    const size_t bytes = (size_t)Hkv_ * D_ * kBQ;
    if (const char* path = std::getenv("MLLM_DUMP_K_PATH")) {
      const uint8_t* src = k_curr_full_[0].ptr<uint8_t>();  // [Hkv, D, Sq] (single reused; holds produced_layer's K)
      std::vector<uint8_t> buf(bytes);
      for (int h = 0; h < Hkv_; ++h)
        for (int c = 0; c < D_; ++c)
          std::memcpy(buf.data() + ((size_t)h * D_ + c) * kBQ, src + ((size_t)h * D_ + c) * Sq_, kBQ);
      FILE* f = std::fopen(path, "wb");
      if (f) { std::fwrite(buf.data(), 1, bytes, f); std::fclose(f); }
      MLLM_INFO("[dump_k] split: layer={} wrote {} bytes to {}", produced_layer, bytes, path);
    }
    if (const char* vpath = std::getenv("MLLM_DUMP_V_PATH")) {
      const uint8_t* src = v_curr_full_[0].ptr<uint8_t>();  // [Hkv, Sq, D] (single reused; holds produced_layer's V)
      std::vector<uint8_t> buf(bytes);
      for (int h = 0; h < Hkv_; ++h)
        std::memcpy(buf.data() + (size_t)h * kBQ * D_, src + (size_t)h * Sq_ * D_, (size_t)kBQ * D_);
      FILE* f = std::fopen(vpath, "wb");
      if (f) { std::fwrite(buf.data(), 1, bytes, f); std::fclose(f); }
      MLLM_INFO("[dump_v] split: layer={} wrote {} bytes to {}", produced_layer, bytes, vpath);
    }
  };
  dump_layer_kv(0);  // chunk_0 produced layer 0's K/V

  // ---- for each layer: per-qb attn loop, then chunk_{i+1} -----------------
  // (No per-layer setName needed: QNN binds dispatch inputs/outputs by
  // POSITION in the input/output vector, not by tensor name.)
  std::vector<int> sel;
  for (int i = 0; i < L_; ++i) {
    // Commit this layer's full-Sq K/V into the cache BEFORE the per-qb loop
    // so per-qb gathers for qb >= 1 see freshly-computed K/V at positions
    // 0 .. (qb*BQ - 1) instead of stale/uninitialised cache bytes. The
    // sliding-window selectTopKBlocks policy reads block indices [0, qb_global)
    // out of the cache; that range overlaps THIS prefill's tokens, so the
    // commit must precede the gather.
    copy_kv_to_cache(i, /*base_pos=*/start_pos, /*n_tokens=*/num_tokens);

    for (int qb = 0; qb < num_qb_; ++qb) {
      const int qb_global = qb;  // start_pos == 0
      // CPU prep: mask + gather (sliding-window top-k policy, same as monolithic)
      build_mask(qb_global);
      selectTopKBlocks(qb_global, sel);
      gather_one_qb(i, qb_global, sel);
      // CPU prep: stage per-qb slices into the small attention buffers.
      stage_q_qb(i, qb);
      stage_k_curr_qb(i, qb);
      stage_v_curr_qb(i, qb);

      // NPU dispatch: attn_i.
      std::vector<Tensor> ins = attn_in_[i];
      attn_out_[i] = (*attns_[i])(ins);

      // CPU writeback: per-qb attn output → full-Sq attn_output_i at qb offset.
      writeback_attn_output_qb(i, qb);
    }

    // Tap: dump this layer's fp16 boundary buffers (residual_pre_attn_i,
    // attn_output_i) for the requested layer, before chunk_{i+1} consumes them.
    if (const char* bl = std::getenv("MLLM_DUMP_BND_LAYER")) {
      if (std::atoi(bl) == i) {
        if (const char* rp = std::getenv("MLLM_DUMP_RES_PATH")) {
          auto t = residual_full_[i % 2].to(kCPU);  // R[i] lives in the ping-pong slot i%2
          FILE* f = std::fopen(rp, "wb");
          if (f) { std::fwrite(t.ptr<uint16_t>(), sizeof(uint16_t), (size_t)Sq_ * hidden_, f); std::fclose(f); }
          MLLM_INFO("[dump_bnd] split: residual layer={} ({}x{} fp16) to {}", i, Sq_, hidden_, rp);
        }
        if (const char* ap = std::getenv("MLLM_DUMP_AO_PATH")) {
          auto t = attn_output_full_[0].to(kCPU);  // single reused buffer
          FILE* f = std::fopen(ap, "wb");
          if (f) { std::fwrite(t.ptr<uint16_t>(), sizeof(uint16_t), (size_t)Hq_ * Sq_ * D_, f); std::fclose(f); }
          MLLM_INFO("[dump_bnd] split: attn_output layer={} ({}x{}x{} fp16) to {}", i, Hq_, Sq_, D_, ap);
        }
      }
    }

    // chunk_{i+1}: post_attn_i + (pre_attn_{i+1} or final norm + lm_head).
    std::vector<Tensor> ins = chunk_in_[i + 1];
    chunk_out_[i + 1] = (*chunks_[i + 1])(ins);
    if (i + 1 < L_) dump_layer_kv(i + 1);  // chunk_{i+1} produced layer (i+1)'s K/V
  }

  // ---- sample logits --------------------------------------------------------
  // chunk_L already gathered the last real token before lm_head, so logits_ is
  // [1, 1, 1, vocab] — the single row we want.
  auto logits = logits_.to(kCPU);  // [1, 1, 1, vocab]
  auto row = logits.squeeze(0)[{kAll, 0, kAll}];  // [1, vocab]
  if (const char* p = std::getenv("MLLM_DUMP_LOGITS")) {
    auto* data = row.ptr<uint16_t>();
    size_t n = (size_t)config_.vocab_size;
    FILE* f = std::fopen(p, "wb");
    if (f) { std::fwrite(data, sizeof(uint16_t), n, f); std::fclose(f); }
    MLLM_INFO("[dump_logits] split: wrote {} u16 to {}", n, p);
  }
  return chunks_[L_]->sampleGreedy(row);
}

}  // namespace mllm::qnn::aot
