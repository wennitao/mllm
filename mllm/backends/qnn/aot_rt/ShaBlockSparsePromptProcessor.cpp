// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace mllm::qnn::aot {

ShaBlockSparsePromptProcessor::ShaBlockSparsePromptProcessor(KVCacheManager<uint8_t>* kv_manager, QnnAOTConfig config)
    : kv_manager_(kv_manager), config_(config) {
  std::string graph_name = "model.0.s" + std::to_string(kBQ);
  module_ = std::make_unique<QnnAOTModule>(graph_name);
  module_->to(kQNN);
}

void ShaBlockSparsePromptProcessor::init_io() {
  // Inputs:
  //   [0]            input_ids       [1, kBQ]                          int32
  //   [1]            position_ids    [1, kBQ]                          int32
  //   [2]            mask            [1, 1, kBQ, kTopKBK]              uint16
  //   [3..3+L)       k_arranged_l    [Hq, 1, head_dim, kHistKBK]       uint8
  //   [3+L..3+2L)    v_arranged_l    [Hq, 1, kHistKBK, head_dim]       uint8

  const int L = config_.num_layers;
  const int Hq = config_.num_attention_heads;
  const int D = config_.head_dim;

  input_tensors_.clear();
  input_tensors_.reserve(3 + 2 * L);

  auto input_ids = Tensor::empty({1, kBQ}, kInt32, kQNN).alloc();
  input_ids.setName("input_ids");
  input_tensors_.push_back(input_ids);

  auto pos_ids = Tensor::empty({1, kBQ}, kInt32, kQNN).alloc();
  pos_ids.setName("position_ids");
  input_tensors_.push_back(pos_ids);

  mask_ = Tensor::empty({1, 1, kBQ, kTopKBK}, kUInt16, kQNN).alloc();
  std::memset(mask_.ptr<void>(), 0, mask_.bytes());
  mask_.setName("mask");
  input_tensors_.push_back(mask_);

  // The compiled graph orders K-arranged inputs first (all L layers' K),
  // then V-arranged inputs (all L layers' V) — see
  // Qwen3TextSHABlockSparseCausal::forward which iterates k_arrs then v_arrs.
  // Earlier we interleaved (K0, V0, K1, V1, ...) which silently worked for
  // qb_global=0 (K_arranged content fully masked out) but corrupted the qb
  // 0→1 transition (K slots received V data and vice versa).
  k_arranged_.clear();
  v_arranged_.clear();
  k_arranged_.reserve(L);
  v_arranged_.reserve(L);
  for (int l = 0; l < L; ++l) {
    auto k = Tensor::empty({Hq, 1, D, kHistKBK}, kUInt8, kQNN).alloc();
    std::memset(k.ptr<void>(), 0, k.bytes());
    k.setName("k_arranged_" + std::to_string(l));
    k_arranged_.push_back(k);
    input_tensors_.push_back(k);
  }
  for (int l = 0; l < L; ++l) {
    auto v = Tensor::empty({Hq, 1, kHistKBK, D}, kUInt8, kQNN).alloc();
    std::memset(v.ptr<void>(), 0, v.bytes());
    v.setName("v_arranged_" + std::to_string(l));
    v_arranged_.push_back(v);
    input_tensors_.push_back(v);
  }

  // Outputs: logits [1, 1, kBQ, vocab] uint16, plus present_key/value per layer.
  output_tensors_.clear();
  output_tensors_.reserve(1 + 2 * L);
  auto logits = Tensor::empty({1, 1, kBQ, config_.vocab_size}, kUInt16, kQNN).alloc();
  logits.setName("logits");
  output_tensors_.push_back(logits);

  const auto& k_caches = kv_manager_->getKCache();
  const auto& v_caches = kv_manager_->getVCache();
  for (int l = 0; l < L; ++l) {
    auto k_tensor = Tensor::empty({1, config_.num_heads, D, kBQ}, kUInt8, kQNN);
    k_tensor.impl()->storage()->ptr_ = k_caches[l].output_buffer;
    k_tensor.impl()->storage()->mem_type_ = kManual;
    k_tensor.setName("present_key_" + std::to_string(l));
    output_tensors_.push_back(k_tensor);
  }
  for (int l = 0; l < L; ++l) {
    auto v_tensor = Tensor::empty({1, config_.num_heads, kBQ, D}, kUInt8, kQNN);
    v_tensor.impl()->storage()->ptr_ = v_caches[l].output_buffer;
    v_tensor.impl()->storage()->mem_type_ = kManual;
    v_tensor.setName("present_value_" + std::to_string(l));
    output_tensors_.push_back(v_tensor);
  }
}

void ShaBlockSparsePromptProcessor::selectTopKBlocks(int qb_global, std::vector<int>& sel) {
  const int Hq = config_.num_attention_heads;
  const int n_hist = kTopK - 1;
  sel.assign((size_t)Hq * n_hist, 0);
  for (int h = 0; h < Hq; ++h) {
    int* hs = sel.data() + (size_t)h * n_hist;
    if (qb_global <= n_hist) {
      // All historical blocks fit — include them all, pad the rest.
      const int hist_slots = qb_global;
      for (int s = 0; s < hist_slots; ++s) hs[s] = qb_global - 1 - s;
      for (int s = hist_slots; s < n_hist; ++s) hs[s] = 0;  // padding slot (masked)
    } else {
      // Selection needed. Forced anchors:
      //   slot 0          = block 0       (ATTENTION SINK — dropping it collapses
      //                                     the softmax, the StreamingLLM result;
      //                                     this is what produced token garbage
      //                                     once qb_global crossed n_hist).
      //   slot 1          = block qb-1    (most-recent / sliding-window block,
      //                                     needed for local coherence).
      // Remaining (n_hist-2) slots: random distinct blocks from the middle range
      // [1, qb-2], sampled per-head. RNG seeded by (qb_global, h) so a given
      // prompt reproduces across runs while still spreading coverage over the
      // middle context.
      hs[0] = 0;             // sink
      hs[1] = qb_global - 1;  // most-recent
      const int mid_lo = 1;
      const int mid_hi = qb_global - 2;  // inclusive
      const int mid_count = mid_hi - mid_lo + 1;
      const int need = n_hist - 2;
      std::mt19937 rng((uint32_t)(qb_global * 1315423911u + (uint32_t)h * 2654435761u + 0x9e3779b9u));
      std::vector<int> cand(mid_count);
      for (int i = 0; i < mid_count; ++i) cand[i] = mid_lo + i;
      // Partial Fisher-Yates: pick `need` distinct middle blocks.
      for (int i = 0; i < need; ++i) {
        if (i < mid_count) {
          std::uniform_int_distribution<int> dist(i, mid_count - 1);
          int j = dist(rng);
          std::swap(cand[i], cand[j]);
          hs[2 + i] = cand[i];
        } else {
          hs[2 + i] = 0;  // middle range smaller than need → pad with sink dup
        }
      }
    }
  }
}

void ShaBlockSparsePromptProcessor::gather_one_qb(int layer, int qb_global, const std::vector<int>& sel) {
  // CPU-side uint8 KV cache layout matches past_key/past_value:
  //   K cache: [1, Hkv, D, max_kv_len]                — transposed K
  //   V cache: [1, Hkv, max_kv_len, D]
  // K_arranged dst: [Hq, 1, D, hist_k_BK]              — same transposed layout
  //   K_arranged[h, 0, c, s*BK + r] = past_key[0, kv_head, c, k_pos*BK + r]
  //   Per (head, slot): D rows of BK contiguous bytes  (D memcpys of BK bytes)
  // V_arranged dst: [Hq, 1, hist_k_BK, D]
  //   V_arranged[h, 0, s*BK + r, c] = past_value[0, kv_head, k_pos*BK + r, c]
  //   Per (head, slot): BK contiguous rows of D bytes  (single memcpy of BK*D)
  const int Hkv = config_.num_heads;
  const int Hq = config_.num_attention_heads;
  const int group = Hq / Hkv;
  const int D = config_.head_dim;
  const int CL = config_.context_len;
  const int n_hist = kTopK - 1;
  const int max_kv_len = CL - kBQ;

  const uint8_t* pk = kv_manager_->getKCache()[layer].buffer;
  const uint8_t* pv = kv_manager_->getVCache()[layer].buffer;
  uint8_t* kdst = (uint8_t*)k_arranged_[layer].ptr<uint8_t>();
  uint8_t* vdst = (uint8_t*)v_arranged_[layer].ptr<uint8_t>();

  const size_t past_key_head_stride = (size_t)D * max_kv_len;
  const size_t past_value_head_stride = (size_t)max_kv_len * D;
  const size_t k_arr_head_stride = (size_t)D * n_hist * kBK;  // D rows, hist_k_BK columns
  const size_t v_arr_head_stride = (size_t)n_hist * kBK * D;  // hist_k_BK rows, D columns

  for (int h = 0; h < Hq; ++h) {
    const int kv_head_idx = h / group;
    const int* hs = sel.data() + (size_t)h * n_hist;
    const uint8_t* pk_head = pk + kv_head_idx * past_key_head_stride;       // [D, max_kv_len]
    const uint8_t* pv_head = pv + kv_head_idx * past_value_head_stride;     // [max_kv_len, D]
    uint8_t* kdst_head = kdst + (size_t)h * k_arr_head_stride;              // [D, hist_k_BK]
    uint8_t* vdst_head = vdst + (size_t)h * v_arr_head_stride;              // [hist_k_BK, D]

    for (int s = 0; s < n_hist; ++s) {
      const int k_pos = hs[s];
      const int k_off = k_pos * kBK;
      const bool valid = (k_pos >= 0) && (k_off + kBK <= max_kv_len);

      // V: BK contiguous rows × D bytes — single memcpy.
      uint8_t* vds = vdst_head + (size_t)s * kBK * D;
      if (valid) {
        std::memcpy(vds, pv_head + (size_t)k_off * D, (size_t)kBK * D);
      } else {
        std::fill_n(vds, (size_t)kBK * D, (uint8_t)128);  // zp = 128 → dequant ≈ 0
      }

      // K: per-row memcpy of BK bytes (D rows in transposed layout).
      // K_arranged_head shape [D, hist_k_BK]: row c is D-th row.
      // K cache head shape [D, max_kv_len]: row c slice [k_off, k_off+BK).
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

void ShaBlockSparsePromptProcessor::build_mask(int qb_global) {
  // mask shape: [1, 1, kBQ, kTopKBK]
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

void ShaBlockSparsePromptProcessor::prepare_one_qb(const std::vector<int64_t>& prompt_tokens, int64_t prompt_pos,
                                                   int64_t global_pos, int qb_in_chunk, int qb_global) {
  int32_t* input_ids_ptr = input_tensors_[0].ptr<int32_t>();
  int32_t* pos_ids_ptr = input_tensors_[1].ptr<int32_t>();
  for (int i = 0; i < kBQ; ++i) {
    int64_t src_idx = prompt_pos + (int64_t)qb_in_chunk * kBQ + i;
    input_ids_ptr[i] = (src_idx < (int64_t)prompt_tokens.size()) ? (int32_t)prompt_tokens[src_idx] : 0;
    pos_ids_ptr[i] = (int32_t)(global_pos + (int64_t)qb_in_chunk * kBQ + i);
  }
  build_mask(qb_global);
  std::vector<int> sel;
  selectTopKBlocks(qb_global, sel);
  for (int l = 0; l < config_.num_layers; ++l) { gather_one_qb(l, qb_global, sel); }
}

int64_t ShaBlockSparsePromptProcessor::prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos) {
  int64_t num_tokens = prompt_tokens.size();
  int64_t current_pos = start_pos;
  int64_t processed_tokens = 0;

  module_->setOutputTensors(output_tensors_);

  MLLM_INFO("ShaBlockSparsePromptProcessor: num_tokens={} start_pos={}", num_tokens, start_pos);

  while (processed_tokens < num_tokens) {
    int64_t remaining = num_tokens - processed_tokens;
    const int qb_global = (int)(current_pos / kBQ);
    prepare_one_qb(prompt_tokens, processed_tokens, current_pos, /*qb_in_chunk=*/0, qb_global);

    std::vector<Tensor> module_input = input_tensors_;
    output_tensors_ = (*module_)(module_input);

    const int32_t n_update = (int32_t)std::min<int64_t>(kBQ, remaining);

    // Tap-point: dump layer-L K and V output bytes (post-RoPE for K,
    // post-projection for V, both pre-cache-write).
    // K layout: [num_heads, head_dim, kBQ] of uint8 each (LPBQ KV).
    // V layout: [num_heads, kBQ, head_dim] of uint8 each.
    if (const char* layer_env = std::getenv("MLLM_DUMP_K_LAYER")) {
      int layer = std::atoi(layer_env);
      if (layer >= 0 && layer < config_.num_layers) {
        size_t bytes = sizeof(uint8_t) * (size_t)config_.num_heads * (size_t)config_.head_dim * (size_t)kBQ;
        if (const char* path = std::getenv("MLLM_DUMP_K_PATH")) {
          const auto& k_caches_dump = kv_manager_->getKCache();
          FILE* f = std::fopen(path, "wb");
          if (f) {
            std::fwrite(k_caches_dump[layer].output_buffer, 1, bytes, f);
            std::fclose(f);
          }
          MLLM_INFO("[dump_k] mono: layer={} wrote {} bytes to {}", layer, bytes, path);
        }
        if (const char* vpath = std::getenv("MLLM_DUMP_V_PATH")) {
          const auto& v_caches_dump = kv_manager_->getVCache();
          FILE* f = std::fopen(vpath, "wb");
          if (f) {
            std::fwrite(v_caches_dump[layer].output_buffer, 1, bytes, f);
            std::fclose(f);
          }
          MLLM_INFO("[dump_v] mono: layer={} wrote {} bytes to {}", layer, bytes, vpath);
        }
      }
    }

    kv_manager_->updateCache(kBQ, current_pos, n_update, {});

    processed_tokens += n_update;
    current_pos += n_update;
  }

  auto logits = output_tensors_[0]
                    .to(kCPU)
                    .squeeze(0)[{kAll, ((int)num_tokens + kBQ - 1) % kBQ, kAll}];
  if (const char* p = std::getenv("MLLM_DUMP_LOGITS")) {
    auto* data = logits.ptr<uint16_t>();
    size_t n = (size_t)config_.vocab_size;
    FILE* f = std::fopen(p, "wb");
    if (f) { std::fwrite(data, sizeof(uint16_t), n, f); std::fclose(f); }
    MLLM_INFO("[dump_logits] mono: wrote {} u16 to {}", n, p);
  }
  auto cur_token = module_->sampleGreedy(logits);
  return cur_token;
}

// Copy ONE position (src_pos in output_buffer) to ONE cache slot (dst_pos in
// cache buffer) for every layer. Used by decode where each step adds exactly
// one new token to the KV cache.
//   K cache layout per head: [D, max_kv_len]; source per head: [D, kBQ]
//     for each c in [0, D): cache[h, c, dst_pos] = out[h, c, src_pos]   (D scalar writes)
//   V cache layout per head: [max_kv_len, D]; source per head: [kBQ, D]
//     for each h: memcpy(cache[h, dst_pos, :], out[h, src_pos, :], D)   (one memcpy)
static void copy_one_kv_position(KVCacheManager<uint8_t>* kv_manager, const QnnAOTConfig& config, int64_t dst_pos,
                                 int src_pos, int kBQ) {
  const int Hkv = config.num_heads;
  const int D = config.head_dim;
  const int max_kv_len = config.context_len - kBQ;
  for (int layer = 0; layer < config.num_layers; ++layer) {
    uint8_t* kcache = kv_manager->getKCache()[layer].buffer;
    const uint8_t* kout = kv_manager->getKCache()[layer].output_buffer;
    uint8_t* vcache = kv_manager->getVCache()[layer].buffer;
    const uint8_t* vout = kv_manager->getVCache()[layer].output_buffer;
    for (int h = 0; h < Hkv; ++h) {
      // K: scattered write of D bytes
      for (int c = 0; c < D; ++c) {
        kcache[((size_t)h * D + c) * max_kv_len + dst_pos] = kout[((size_t)h * D + c) * kBQ + src_pos];
      }
      // V: contiguous memcpy of D bytes
      std::memcpy(vcache + ((size_t)h * max_kv_len + dst_pos) * D, vout + ((size_t)h * kBQ + src_pos) * D, D);
    }
  }
}

// Fill input_tensors_[0..1] (input_ids + position_ids) for a single qb,
// pulling token data directly from `all_tokens`. Tokens beyond all_tokens.size()
// are padded with 0. Used by both prefill and decode.
static void fill_qb_inputs(std::vector<Tensor>& input_tensors, const std::vector<int64_t>& all_tokens, int64_t qb_start_pos,
                           int kBQ) {
  int32_t* ids_ptr = input_tensors[0].ptr<int32_t>();
  int32_t* pos_ptr = input_tensors[1].ptr<int32_t>();
  for (int i = 0; i < kBQ; ++i) {
    int64_t idx = qb_start_pos + i;
    ids_ptr[i] = (idx < (int64_t)all_tokens.size()) ? (int32_t)all_tokens[idx] : 0;
    pos_ptr[i] = (int32_t)idx;
  }
}

void ShaBlockSparsePromptProcessor::decode(std::vector<int64_t>& all_tokens, int max_new_tokens, int64_t eos_token_id,
                                           const std::function<void(int64_t)>& token_callback) {
  // Precondition: `all_tokens` already includes the first sampled token from
  // prefill. Its KV state at all positions < all_tokens.size() - 1 is in the
  // cache; the LAST token in all_tokens has NOT been written to cache yet
  // (it was just sampled from prefill's logits, not run through the model).
  //
  // First decode step: write that last token to cache + sample the next one.
  // Subsequent steps: same pattern.
  module_->setOutputTensors(output_tensors_);

  // pending_pos = the position whose token's KV needs to be written to the
  // cache before we can sample the NEXT token. After prefill, the first
  // sampled token sits at all_tokens.back() at position all_tokens.size()-1.
  for (int step = 0; step < max_new_tokens; ++step) {
    int64_t pending_pos = (int64_t)all_tokens.size() - 1;
    int qb_global = (int)(pending_pos / kBQ);
    int position_in_qb = (int)(pending_pos % kBQ);
    int64_t qb_start_pos = (int64_t)qb_global * kBQ;

    // Fill input_ids: positions 0..position_in_qb get the actual tokens of
    // this qb (which may include some prefill tokens if the qb straddles the
    // prefill boundary). Positions > position_in_qb are padding zeros.
    fill_qb_inputs(input_tensors_, all_tokens, qb_start_pos, kBQ);
    // Build mask + gather historical K/V for this qb.
    build_mask(qb_global);
    std::vector<int> sel;
    selectTopKBlocks(qb_global, sel);
    for (int l = 0; l < config_.num_layers; ++l) { gather_one_qb(l, qb_global, sel); }

    std::vector<Tensor> module_input = input_tensors_;
    output_tensors_ = (*module_)(module_input);

    // Write only the new token's KV (position position_in_qb of the qb) to
    // cache slot pending_pos.
    copy_one_kv_position(kv_manager_, config_, pending_pos, position_in_qb, kBQ);

    // Sample next token from logits at position_in_qb.
    auto logits = output_tensors_[0]
                      .to(kCPU)
                      .squeeze(0)[{kAll, position_in_qb, kAll}];
    int64_t next_token = module_->sampleGreedy(logits);
    all_tokens.push_back(next_token);
    token_callback(next_token);
    if (next_token == eos_token_id) break;
  }
}

}  // namespace mllm::qnn::aot
