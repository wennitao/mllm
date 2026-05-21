// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/nn/Functional.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace mllm::qnn::aot {

// Fast approximate expf (~10× faster than libm expf) for the softmax in block
// scoring. Selection only needs the block RANKING, and exp is monotonic, so a
// ~1e-3 relative error is irrelevant here. exp(x) = 2^(x·log2 e): split into
// integer (via float bit exponent) and fractional (degree-3 poly) parts.
static inline float fast_exp(float x) {
  x = x < -80.0f ? -80.0f : (x > 80.0f ? 80.0f : x);
  const float t = x * 1.4426950408889634f;  // log2(e)
  const float fi = std::floor(t);
  const float f = t - fi;
  // 2^f ≈ poly(f) on [0,1)
  const float p = 1.0f + f * (0.6958444f + f * (0.2247086f + f * 0.0791402f));
  union { float fl; int32_t i; } u;
  u.i = (int32_t)((int)fi + 127) << 23;  // 2^(int)
  return u.fl * p;
}

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

  // ----- mask + K/V_arranged, DOUBLE-BUFFERED (2 slots, runner-gathered) ----
  mask_.clear();
  k_arranged_.clear();
  v_arranged_.clear();
  for (int s = 0; s < 2; ++s) {
    auto m = Tensor::empty({1, 1, kBQ, kTopKBK}, kUInt16, kQNN).alloc();
    std::memset(m.ptr<void>(), 0, m.bytes());
    m.setName("mask_s" + std::to_string(s));
    mask_.push_back(m);

    auto k_arr = Tensor::empty({Hq_, 1, D_, kHistKBK}, kUInt8, kQNN).alloc();
    std::memset(k_arr.ptr<void>(), 0, k_arr.bytes());
    k_arr.setName("K_arranged_s" + std::to_string(s));
    k_arranged_.push_back(k_arr);

    auto v_arr = Tensor::empty({Hq_, 1, kHistKBK, D_}, kUInt8, kQNN).alloc();
    std::memset(v_arr.ptr<void>(), 0, v_arr.bytes());
    v_arr.setName("V_arranged_s" + std::to_string(s));
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

  // ----- per-qb staging buffers, DOUBLE-BUFFERED (2 slots) -----------------
  q_qb_.clear();
  k_curr_qb_.clear();
  v_curr_qb_.clear();
  attn_output_qb_.clear();
  for (int s = 0; s < 2; ++s) {
    auto ss = std::to_string(s);
    auto t_q = Tensor::empty({1, Hq_, kBQ, D_}, kUInt16, kQNN).alloc();
    t_q.setName("q_qb_s" + ss);
    q_qb_.push_back(t_q);

    auto t_kc = Tensor::empty({1, Hkv_, D_, kBQ}, kUInt8, kQNN).alloc();
    t_kc.setName("k_curr_qb_s" + ss);
    k_curr_qb_.push_back(t_kc);

    auto t_vc = Tensor::empty({1, Hkv_, kBQ, D_}, kUInt8, kQNN).alloc();
    t_vc.setName("v_curr_qb_s" + ss);
    v_curr_qb_.push_back(t_vc);

    auto t_ao = Tensor::empty({1, Hq_, kBQ, D_}, kFloat16, kQNN).alloc();
    t_ao.setName("attn_output_qb_s" + ss);
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

  // attn_i I/O is built per dispatch from the active double-buffer slot (see
  // dispatch_attn in prefill), so attn_in_/attn_out_ aren't pre-populated here.

  // Bind chunk output tensors (QnnAOTModule fills output_tensors_ on execute).
  for (int i = 0; i <= L_; ++i) chunks_[i]->setOutputTensors(chunk_out_[i]);

  // ----- NPU block-selection scoring graph (opt-in, needs baked "score") -----
  npu_score_ = std::getenv("MLLM_BLOCKSEL_NPU") != nullptr;
  if (npu_score_) {
    const int S = kScoreStride, Lr = Sq_ / S, SD = S * D_;
    const int Hh = Hq_ / kScoreHeadSplit;  // heads per dispatch
    uint64_t sf_mb = 128;
    if (const char* e = std::getenv("MLLM_QNN_SPILLFILL_MB")) sf_mb = std::strtoull(e, nullptr, 10);
    qnn_backend_ = std::static_pointer_cast<mllm::qnn::QNNBackend>(Context::instance().getBackend(kQNN)).get();
    // Build the score matmul graph in a 2nd HTP context sharing the model
    // context's spill-fill group (so it stays out of the saturated model
    // context). logits = score_qr · score_kcᵀ (transpose_in1=true, runtime
    // MatMul honors transpose flags — no pre-transpose needed).
    if (!qnn_backend_ || !qnn_backend_->beginAuxContext(sf_mb)) {
      MLLM_ERROR("NPU score: failed to open aux context — disabling NPU scoring");
      npu_score_ = false;
    } else {
      score_qr_ = Tensor::empty({Hh, Lr, SD}, kFloat16, kQNN).alloc();
      score_qr_.setName("score_qr");
      score_kc_ = Tensor::empty({Hh, Lr, SD}, kFloat16, kQNN).alloc();
      score_kc_.setName("score_kc");
      score_logits_ = Tensor::empty({Hh, Lr, Lr}, kFloat16, kQNN).alloc();
      score_logits_.setName("score_logits");
      const std::string g = "score";
      qnn_backend_->createQnnGraph(g);
      qnn_backend_->addTensor(g, "score_qr", QNN_TENSOR_TYPE_APP_WRITE, score_qr_);
      qnn_backend_->addTensor(g, "score_kc", QNN_TENSOR_TYPE_APP_WRITE, score_kc_);
      qnn_backend_->addTensor(g, "score_logits", QNN_TENSOR_TYPE_APP_READ, score_logits_);
      std::vector<std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>> mm_params;
      mm_params.push_back(mllm::qnn::QNNParamScalarWrapper::create<bool>("transpose_in1", true));
      qnn_backend_->graphAddNode(g, "matmul", "MatMul", {"score_qr", "score_kc"}, {"score_logits"}, {}, mm_params,
                                 "qti.aisw");
      if (!qnn_backend_->graphFinalize(g)) {
        MLLM_ERROR("NPU score: graphFinalize failed — disabling NPU scoring");
        npu_score_ = false;
      }
      qnn_backend_->endAuxContext();
    }
  }
}

// ============================================================================
// Helpers — same logic as the monolithic processor, copied for self-containedness.
// ============================================================================
void ShaBlockSparsePromptProcessorSplit::enableScoreBasedSelection(const std::vector<int32_t>& q_zp,
                                                                   const std::vector<float>& q_scale,
                                                                   const std::vector<float>& k_scale) {
  q_zp_ = q_zp;
  q_scale_ = q_scale;
  k_scale_ = k_scale;
  score_based_ = !q_zp_.empty() && q_scale_.size() == q_zp_.size() && k_scale_.size() == q_zp_.size();
  MLLM_INFO("ShaBlockSparsePromptProcessorSplit: score-based block selection {} ({} layers)",
            score_based_ ? "ENABLED" : "disabled", q_zp_.size());
}

// XAttention-style scoring (CPU), per (layer, qb). Reduced antidiagonal scoring:
// pack S consecutive positions into the feature dim (D→S·D) and subsample the
// sequence; the dot over the packed feature is the antidiagonal sum. Q packs in
// REVERSED order ("inverse" mode). softmax(/√D/S) over the reduced HISTORY keys,
// then pool BKr reduced cols per block → scores[h*qb_global + kb]. S from
// MLLM_BLOCKSEL_STRIDE (default 1 = exact). Uses kBLAS on the small causal
// slice — fast + correct (NOTE: the full-Sq-grid "Lever 1" variant was tried
// and reverted: it's 2× the FLOPs (computes the masked upper triangle) AND
// kBLAS is buggy at that large batched shape; scoring is FLOP-bound, not
// per-call-overhead-bound, so consolidating matmuls made it slower).
// Big-M path: build qr_layer_ (reduced full-seq Q) + kc_layer_ (reduced full-seq
// K) once per layer and do ONE matmul → logits_full_ [1,Hq,Lr,Lr]. Cached by
// logits_layer_idx_. The per-qb softmax+pool then just slices logits_full_.
bool ShaBlockSparsePromptProcessorSplit::computeLayerLogits(int layer) {
  if (!score_based_ || layer < 0 || layer >= (int)q_zp_.size()) return false;
  const int Hq = Hq_, D = D_, BQ = kBQ, group = Hq_ / Hkv_, Sq = Sq_;
  const int max_kv_len = config_.context_len - kBQ;
  int S = 1;
  if (const char* e = std::getenv("MLLM_BLOCKSEL_STRIDE")) S = std::atoi(e);
  if (S < 1 || BQ % S != 0 || kBK % S != 0) S = 1;
  const int SD = S * D, Lr = Sq / S;
  if (logits_layer_idx_ == layer) return true;  // cached

  using clk = std::chrono::high_resolution_clock;
  auto us_since = [](const std::chrono::time_point<clk>& t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t).count();
  };

  // ----- NPU path: build fp16 qr/kc, dispatch baked "score" graph -----------
  if (npu_score_) {
    const int Sn = kScoreStride, SDn = Sn * D, Lrn = Sq / Sn;
    const int Hh = Hq / kScoreHeadSplit;  // heads per dispatch (graph is baked at Hh)
    const bool slash = std::getenv("MLLM_BLOCKSEL_SLASH") != nullptr;
    // Pre-scale BOTH qr and kc by sqrt(temp) so the fp16 matmul output lands in
    // fp16 range: the raw qr·kc is ~1e7 (qr,kc ~O(100), summed over SD=1024) and
    // overflows fp16's 65504 max → inf → garbage scores. With each input scaled
    // by sqrt(temp), the product is the properly-scaled attention logit (O(1-10)),
    // so the CPU softmax below uses temp=1. temp = q_scale·k_scale/(√D·S).
    const float temp_layer = q_scale_[layer] * k_scale_[layer] / (std::sqrt((float)D) * (float)Sn);
    const float sqrt_temp = std::sqrt(temp_layer > 0.f ? temp_layer : 0.f);
    logits_full_ = Tensor::empty({1, Hq, Lrn, Lrn}, kFloat32, kCPU).alloc();
    // Process the heads in kScoreHeadSplit groups of Hh so each dispatch's
    // working set fits the 8 MB VTCM (→ ~0 spill-fill). hbase = first head.
    for (int g = 0; g < kScoreHeadSplit; ++g) {
      const int hbase = g * Hh;
      const auto t_b0 = clk::now();
      // qr [Hh,Lrn,SDn] fp16: qr[h,rg,s*D+d] = Q[hbase+h, rg*Sn+(Sn-1-s), d] - zp.
      {
        const uint16_t* qsrc = q_full_[0].ptr<uint16_t>();  // [Hq, Sq, D]
        const float zp = (float)q_zp_[layer];
        mllm_fp16_t* qd = score_qr_.ptr<mllm_fp16_t>();
#pragma omp parallel for schedule(static)
        for (int h = 0; h < Hh; ++h)
          for (int rg = 0; rg < Lrn; ++rg)
            for (int s = 0; s < Sn; ++s) {
              const int qpos = rg * Sn + (slash ? s : (Sn - 1 - s));
              const uint16_t* src = qsrc + ((size_t)(hbase + h) * Sq + qpos) * D;
              mllm_fp16_t* dst = qd + (((size_t)h * Lrn + rg) * SDn) + (size_t)s * D;
              for (int d = 0; d < D; ++d) dst[d] = (mllm_fp16_t)(((float)src[d] - zp) * sqrt_temp);
            }
      }
      // kc NATURAL [Hh,Lrn,SDn]: kc[h,rg,s*D+d] = K_cache[(hbase+h)/g,d,rg*Sn+s]-128.
      // (Runtime MatMul transpose_in1=true does the transpose, so kc isn't
      // pre-transposed — same packed layout as qr.) ~22 ms warm; the larger
      // variance seen at first is cold-start / CPU-frequency ramp, not this loop
      // (an A/B with a sequential-read order made no difference).
      {
        const uint8_t* pk = kv_manager_->getKCache()[layer].buffer;
        mllm_fp16_t* kd = score_kc_.ptr<mllm_fp16_t>();
#pragma omp parallel for schedule(static)
        for (int h = 0; h < Hh; ++h) {
          const int kv = (hbase + h) / group;
          const uint8_t* pk_head = pk + (size_t)kv * D * max_kv_len;  // [D, max_kv_len]
          mllm_fp16_t* base = kd + (size_t)h * Lrn * SDn;
          for (int rg = 0; rg < Lrn; ++rg)
            for (int s = 0; s < Sn; ++s)
              for (int d = 0; d < D; ++d)
                base[(size_t)rg * SDn + (s * D + d)] =
                    (mllm_fp16_t)(((float)pk_head[(size_t)d * max_kv_len + (rg * Sn + s)] - 128.0f) * sqrt_temp);
        }
      }
      score_kc_us_ += us_since(t_b0);
      // Dispatch the runtime score graph for this head group: logits = qr · kcᵀ.
      const auto t_mm0 = clk::now();
      std::vector<Tensor> score_ins = {score_qr_, score_kc_};
      std::vector<Tensor> score_outs = {score_logits_};
      qnn_backend_->graphExecute("score", score_ins, score_outs);
      // Convert fp16 logits [Hh,Lrn,Lrn] → logits_full_ at head offset hbase.
      {
        const mllm_fp16_t* sl = score_logits_.ptr<mllm_fp16_t>();
        float* lf = logits_full_.ptr<float>() + (size_t)hbase * Lrn * Lrn;
        const size_t n = (size_t)Hh * Lrn * Lrn;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) lf[i] = (float)sl[i];
      }
      score_mm_us_ += us_since(t_mm0);
    }
    logits_layer_idx_ = layer;
    return true;
  }

  // Reduced de-transposed centered K: kc[h,r,s*D+d] = K_cache[h/group,d,r*S+s]-128.
  const auto t_kc0 = clk::now();
  kc_layer_ = Tensor::empty({1, Hq, Lr, SD}, kFloat32, kCPU).alloc();
  {
    float* kc = kc_layer_.ptr<float>();
    const uint8_t* pk = kv_manager_->getKCache()[layer].buffer;
#pragma omp parallel for schedule(static)
    for (int h = 0; h < Hq; ++h) {
      const int kv = h / group;
      const uint8_t* pk_head = pk + (size_t)kv * D * max_kv_len;
      float* dst = kc + (size_t)h * Sq * D;
      for (int pos = 0; pos < Sq; ++pos)
        for (int d = 0; d < D; ++d) dst[(size_t)pos * D + d] = (float)pk_head[(size_t)d * max_kv_len + pos] - 128.0f;
    }
  }
  // Reduced Q for the WHOLE sequence: qr[h,rg,s*D+d] = Qc[h, rg*S+(S-1-s), d]-zp
  // (same antidiagonal reversal as the per-qb build; slash flips the order).
  qr_layer_ = Tensor::empty({1, Hq, Lr, SD}, kFloat32, kCPU).alloc();
  {
    const uint16_t* qsrc = q_full_[0].ptr<uint16_t>();  // [Hq, Sq, D]
    const float zp = (float)q_zp_[layer];
    const bool slash = std::getenv("MLLM_BLOCKSEL_SLASH") != nullptr;
    float* qd = qr_layer_.ptr<float>();
#pragma omp parallel for schedule(static)
    for (int h = 0; h < Hq; ++h)
      for (int rg = 0; rg < Lr; ++rg)
        for (int s = 0; s < S; ++s) {
          const int qpos = rg * S + (slash ? s : (S - 1 - s));
          const uint16_t* src = qsrc + ((size_t)h * Sq + qpos) * D;
          float* dst = qd + (((size_t)h * Lr + rg) * SD) + (size_t)s * D;
          for (int d = 0; d < D; ++d) dst[d] = (float)src[d] - zp;
        }
  }
  score_kc_us_ += us_since(t_kc0);
  // ONE big matmul/layer: logits_full_ [1,Hq,Lr,Lr] = qr_layer_ · kc_layerᵀ.
  const auto t_mm0 = clk::now();
  logits_full_ = mllm::nn::functional::matmul(qr_layer_, kc_layer_, /*transpose_A=*/false, /*transpose_B=*/true,
                                              mllm::aops::MatMulOpType::kMllmBlas)
                     .to(kCPU);
  score_mm_us_ += us_since(t_mm0);
  logits_layer_idx_ = layer;
  return true;
}

bool ShaBlockSparsePromptProcessorSplit::computeBlockScores(int layer, int qb_global, std::vector<float>& scores) {
  if (!score_based_ || layer < 0 || layer >= (int)q_zp_.size()) return false;
  const auto t_start = std::chrono::high_resolution_clock::now();
  const int Hq = Hq_, D = D_, BQ = kBQ, BK = kBK, group = Hq_ / Hkv_, Sq = Sq_;
  const int max_kv_len = config_.context_len - kBQ;
  const int hist = qb_global * BK;  // historical key positions [0, hist)
  if (hist <= 0) return false;

  int S = 1;
  if (const char* e = std::getenv("MLLM_BLOCKSEL_STRIDE")) S = std::atoi(e);
  if (S < 1 || BQ % S != 0 || BK % S != 0) S = 1;  // S must divide both block dims
  if (npu_score_) S = kScoreStride;                // NPU graph is baked at fixed S
  const int BQr = BQ / S, BKr = BK / S, histr = hist / S, SD = S * D;

  using clk = std::chrono::high_resolution_clock;
  auto us_since = [](const std::chrono::time_point<clk>& t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t).count();
  };
  const int Lr = Sq / S;  // reduced full key length

  // Two scoring layouts share the softmax+pool below via (lg, lg_row_stride,
  // lg_head_stride): `lg + h*lg_head_stride + r*lg_row_stride + k` is logit
  // (head h, query row r of this qb, history key r-row k) for k in [0, histr).
  const float* lg;
  int lg_row_stride;
  size_t lg_head_stride;
  Tensor logits;  // keeps the per-qb path's result alive

  if (npu_score_ || std::getenv("MLLM_BLOCKSEL_BIGM")) {
    // BIG-M / NPU: one matmul/layer (built+cached in computeLayerLogits), this qb
    // just slices logits_full_ [1,Hq,Lr,Lr]: rows [qb*BQr,+BQr), cols [0,histr).
    if (!computeLayerLogits(layer)) return false;
    lg = logits_full_.ptr<float>() + (size_t)(qb_global * BQr) * Lr;
    lg_row_stride = Lr;
    lg_head_stride = (size_t)Lr * Lr;
  } else {
    // PER-QB: rebuild de-transposed centered K once/layer, reduced Q per qb,
    // zero-copy K prefix view, one small matmul (M=BQr).
    const auto t_kc0 = clk::now();
    if (kc_layer_idx_ != layer) {
      // Fresh handle per layer: the mllm engine memoizes matmul by input-tensor
      // identity, so reusing the same kc_layer_ handle with refilled data returns
      // a STALE result. A new handle per layer avoids that; within a layer the
      // data is constant so reuse across qbs is safe.
      kc_layer_ = Tensor::empty({1, Hq, Lr, SD}, kFloat32, kCPU).alloc();
      float* kc = kc_layer_.ptr<float>();
      const uint8_t* pk = kv_manager_->getKCache()[layer].buffer;
#pragma omp parallel for schedule(static)
      for (int h = 0; h < Hq; ++h) {
        const int kv = h / group;
        const uint8_t* pk_head = pk + (size_t)kv * D * max_kv_len;  // [D, max_kv_len]
        float* dst = kc + (size_t)h * Sq * D;  // [Sq,D] == [Lr,SD] contiguous
        for (int pos = 0; pos < Sq; ++pos)
          for (int d = 0; d < D; ++d)
            dst[(size_t)pos * D + d] = (float)pk_head[(size_t)d * max_kv_len + pos] - 128.0f;
      }
      kc_layer_idx_ = layer;
    }
    score_kc_us_ += us_since(t_kc0);

    // Reduced Q for the current qb: Qr[h,r,s*D+d] = Qc[h, qb*BQ + r*S + (S-1-s), d].
    const auto t_prep0 = clk::now();
    Tensor Q = Tensor::empty({1, Hq, BQr, SD}, kFloat32, kCPU).alloc();
    {
      const uint16_t* qsrc = q_full_[0].ptr<uint16_t>();  // [Hq, Sq, D]
      const float zp = (float)q_zp_[layer];
      float* qd = Q.ptr<float>();
      const bool slash = std::getenv("MLLM_BLOCKSEL_SLASH") != nullptr;
#pragma omp parallel for schedule(static)
      for (int h = 0; h < Hq; ++h)
        for (int r = 0; r < BQr; ++r)
          for (int s = 0; s < S; ++s) {
            const int qpos = qb_global * BQ + r * S + (slash ? s : (S - 1 - s));
            const uint16_t* src = qsrc + ((size_t)h * Sq + qpos) * D;
            float* dst = qd + (((size_t)h * BQr + r) * SD) + (size_t)s * D;
            for (int d = 0; d < D; ++d) dst[d] = (float)src[d] - zp;
          }
    }
    // Zero-copy K view of kc_layer_'s first `histr` rows: shape [1,Hq,histr,SD]
    // but head/batch stride kept at Sq*D so each head reads its contiguous prefix
    // in place (kMllmBlas gemm path honors rhs.stride()[-3] + ldb=SD).
    Tensor K(mllm::TensorViewImpl::create(
        /*storage_offset=*/0, {1, Hq, histr, SD},
        /*stride=*/{Hq * Sq * D, Sq * D, SD, 1}, kc_layer_.impl()->storage()));
    score_prep_us_ += us_since(t_prep0);

    // Logits [1, Hq, BQr, histr] = Qr · Krᵀ. kMllmBlas (NOT kBLAS — kBLAS is a
    // NYI no-op on this MLLM_USE_BLAS=OFF build; see block_selection.md).
    const auto t_mm0 = clk::now();
    logits = mllm::nn::functional::matmul(Q, K, /*transpose_A=*/false, /*transpose_B=*/true,
                                          mllm::aops::MatMulOpType::kMllmBlas)
                 .to(kCPU);
    score_mm_us_ += us_since(t_mm0);
    lg = logits.ptr<float>();
    lg_row_stride = histr;
    lg_head_stride = (size_t)BQr * histr;
  }
  // NPU path pre-scales qr/kc by sqrt(temp) (fp16 range), so logits are already
  // temp-scaled → softmax with temp=1. CPU paths apply the full temp here.
  const float temp = npu_score_ ? 1.0f : (q_scale_[layer] * k_scale_[layer] / (std::sqrt((float)D) * (float)S));

  // Per head: history-only softmax + block-pool. The exp and the block-pool are
  // FUSED into one pass over histr (no prob[] scratch): per kb-block, exp+sum
  // its BKr logits straight into the block partial-sum, accumulating the row
  // total in parallel; then scores += block_sum/total. For the NPU path the
  // logits are pre-scaled to O(1-10), so the max-subtraction (needed only for
  // exp overflow) is skipped — one fewer pass over histr.
  const auto t_sm0 = clk::now();
  scores.assign((size_t)Hq * qb_global, 0.0f);
  const bool need_max = !npu_score_;
#pragma omp parallel for schedule(static)
  for (int h = 0; h < Hq; ++h) {
    std::vector<float> bsum(qb_global);
    const float* lh = lg + (size_t)h * lg_head_stride;
    float* sc = scores.data() + (size_t)h * qb_global;
    for (int r = 0; r < BQr; ++r) {
      const float* row = lh + (size_t)r * lg_row_stride;
      float mx = 0.0f;
      if (need_max) {
        mx = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < histr; ++k) mx = std::max(mx, row[k] * temp);
      }
      float sum = 0.0f;
      for (int kb = 0; kb < qb_global; ++kb) {
        const float* pk = row + (size_t)kb * BKr;
        float m = 0.0f;
        for (int k = 0; k < BKr; ++k) m += fast_exp(pk[k] * temp - mx);
        bsum[kb] = m;
        sum += m;
      }
      const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
      for (int kb = 0; kb < qb_global; ++kb) sc[kb] += bsum[kb] * inv;
    }
  }
  score_sm_us_ += us_since(t_sm0);
  score_tot_us_ += std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - t_start).count();
  return true;
}

void ShaBlockSparsePromptProcessorSplit::selectTopKBlocks(int layer, int qb_global, std::vector<int>& sel) {
  const int n_hist = kTopK - 1;
  sel.assign((size_t)Hq_ * n_hist, 0);

  if (qb_global <= n_hist) {
    // All historical blocks fit — include them all, pad the rest (no selection).
    for (int h = 0; h < Hq_; ++h) {
      int* hs = sel.data() + (size_t)h * n_hist;
      for (int s = 0; s < qb_global; ++s) hs[s] = qb_global - 1 - s;
      for (int s = qb_global; s < n_hist; ++s) hs[s] = 0;  // padding slot (masked)
    }
    return;
  }

  // Selection needed. Anchor slot 0 = block 0 (attention SINK), slot 1 =
  // block qb-1 (most-recent / sliding window); fill the remaining (n_hist-2)
  // slots from the middle range [1, qb-2].
  std::vector<float> scores;
  const bool have_scores = computeBlockScores(layer, qb_global, scores);  // scores[h*qb_global + kb]

  const int mid_lo = 1, mid_hi = qb_global - 2;  // inclusive
  const int mid_count = mid_hi - mid_lo + 1;
  const int need = n_hist - 2;
  for (int h = 0; h < Hq_; ++h) {
    int* hs = sel.data() + (size_t)h * n_hist;
    hs[0] = 0;              // sink
    hs[1] = qb_global - 1;  // most-recent

    std::vector<int> cand(std::max(0, mid_count));
    std::iota(cand.begin(), cand.end(), mid_lo);
    if (have_scores) {
      // Top-`need` middle blocks by descending score.
      const float* sc = scores.data() + (size_t)h * qb_global;
      std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return sc[a] > sc[b]; });
    } else {
      // Fallback: per-(qb,head)-seeded random middle (original policy).
      std::mt19937 rng((uint32_t)(qb_global * 1315423911u + (uint32_t)h * 2654435761u + 0x9e3779b9u));
      for (int i = 0; i < (int)cand.size() && i < need; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)cand.size() - 1);
        std::swap(cand[i], cand[dist(rng)]);
      }
    }
    for (int i = 0; i < need; ++i) hs[2 + i] = (i < (int)cand.size()) ? cand[i] : 0;
  }

  // Oracle override: MLLM_FORCE_BLOCKS="8,9" forces those blocks into the
  // middle slots (after sink+recent) for every layer/head/qb where they're
  // valid history. Lets us A/B "does the model retrieve when the needle block
  // is guaranteed attended?" vs the scorer's choices.
  if (const char* fb = std::getenv("MLLM_FORCE_BLOCKS")) {
    std::vector<int> forced;
    for (const char* p = fb; *p;) {
      forced.push_back(std::atoi(p));
      while (*p && *p != ',') ++p;
      while (*p == ',') ++p;
    }
    for (int h = 0; h < Hq_; ++h) {
      int* hs = sel.data() + (size_t)h * n_hist;
      int slot = 2;  // keep slot0=sink, slot1=recent
      for (int b : forced) {
        if (slot >= n_hist) break;
        if (b >= 0 && b < qb_global) hs[slot++] = b;
      }
    }
  }

  // Debug: MLLM_DUMP_SEL=<qb> dumps layer-0 head-0 block scores + selected
  // historical blocks for that query block, so we can see whether the needle
  // block is being ranked/selected.
  if (const char* e = std::getenv("MLLM_DUMP_SEL")) {
    const char* le = std::getenv("MLLM_DUMP_SEL_LAYER");
    const int dl = le ? std::atoi(le) : 0;
    if (layer == dl && qb_global == std::atoi(e)) {
      fprintf(stderr, "[sel] layer=%d qb=%d scored=%d  head0 selected:", layer, qb_global, (int)have_scores);
      for (int s = 0; s < n_hist; ++s) fprintf(stderr, " %d", sel[s]);
      if (have_scores) {
        std::vector<int> order(qb_global);
        std::iota(order.begin(), order.end(), 0);
        const float* sc = scores.data();  // head 0
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] > sc[b]; });
        fprintf(stderr, "  | top-8 blocks by score:");
        for (int i = 0; i < std::min(8, qb_global); ++i) fprintf(stderr, " b%d(%.0f)", order[i], sc[order[i]]);
      }
      fprintf(stderr, "\n");
    }
  }
}

void ShaBlockSparsePromptProcessorSplit::gather_one_qb(int layer, int qb_global, const std::vector<int>& sel, int slot) {
  const int Hkv = Hkv_;
  const int Hq = Hq_;
  const int group = Hq / Hkv;
  const int D = D_;
  const int CL = config_.context_len;
  const int n_hist = kTopK - 1;
  const int max_kv_len = CL - kBQ;

  const uint8_t* pk = kv_manager_->getKCache()[layer].buffer;
  const uint8_t* pv = kv_manager_->getVCache()[layer].buffer;
  uint8_t* kdst = (uint8_t*)k_arranged_[slot].ptr<uint8_t>();
  uint8_t* vdst = (uint8_t*)v_arranged_[slot].ptr<uint8_t>();

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

void ShaBlockSparsePromptProcessorSplit::build_mask(int qb_global, int slot) {
  const int hist_slots = std::min(kTopK - 1, qb_global);
  uint16_t* p = mask_[slot].ptr<uint16_t>();
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
void ShaBlockSparsePromptProcessorSplit::stage_q_qb(int qb, int slot) {
  // q_full: [1, Hq, Sq, D] uint16. Per-head h: rows [qb*BQ, (qb+1)*BQ) contiguous BQ*D uint16.
  // q_qb:   [1, Hq, BQ, D] uint16.
  const uint16_t* src = q_full_[0].ptr<uint16_t>();  // single reused buffer
  uint16_t* dst = q_qb_[slot].ptr<uint16_t>();
  const size_t head_stride_src = (size_t)Sq_ * D_;
  const size_t head_stride_dst = (size_t)kBQ * D_;
  const size_t bytes_per_head = (size_t)kBQ * D_ * sizeof(uint16_t);
  for (int h = 0; h < Hq_; ++h) {
    std::memcpy(dst + h * head_stride_dst,
                src + h * head_stride_src + (size_t)qb * kBQ * D_,
                bytes_per_head);
  }
}

void ShaBlockSparsePromptProcessorSplit::stage_k_curr_qb(int qb, int slot) {
  // k_curr_full: [1, Hkv, D, Sq] uint8. Per (h, d): Sq columns, take [qb*BQ, (qb+1)*BQ) — BQ contiguous bytes.
  // k_curr_qb:   [1, Hkv, D, BQ] uint8.
  const uint8_t* src = k_curr_full_[0].ptr<uint8_t>();  // single reused buffer
  uint8_t* dst = k_curr_qb_[slot].ptr<uint8_t>();
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

void ShaBlockSparsePromptProcessorSplit::stage_v_curr_qb(int qb, int slot) {
  // v_curr_full: [1, Hkv, Sq, D] uint8. Per head: BQ contiguous rows of D bytes.
  // v_curr_qb:   [1, Hkv, BQ, D] uint8.
  const uint8_t* src = v_curr_full_[0].ptr<uint8_t>();  // single reused buffer
  uint8_t* dst = v_curr_qb_[slot].ptr<uint8_t>();
  const size_t head_stride_src = (size_t)Sq_ * D_;
  const size_t head_stride_dst = (size_t)kBQ * D_;
  const size_t bytes_per_head = (size_t)kBQ * D_;
  for (int h = 0; h < Hkv_; ++h) {
    std::memcpy(dst + h * head_stride_dst,
                src + h * head_stride_src + (size_t)qb * kBQ * D_,
                bytes_per_head);
  }
}

void ShaBlockSparsePromptProcessorSplit::writeback_attn_output_qb(int qb, int slot) {
  // attn_output_full: [1, Hq, Sq, D] uint16. Write qb-slice from attn_output_qb_[slot] [1, Hq, BQ, D].
  const uint16_t* src = attn_output_qb_[slot].ptr<uint16_t>();
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
  //
  // Optional CPU/NPU PIPELINE (env MLLM_SPLIT_PIPELINE=1): the per-qb CPU prep
  // (mask + block-select + K/V gather + staging) of qb+1 runs on a worker
  // thread while the NPU computes attn for qb. Safe because copy_kv_to_cache()
  // ran BEFORE this loop, so gather(qb+1) only reads already-committed cache —
  // independent of attn(qb) — and the two use opposite double-buffer slots.
  const bool pipeline = [] {
    const char* e = std::getenv("MLLM_SPLIT_PIPELINE");
    return e && std::atoi(e) != 0;
  }();

  // CPU prep for one qb into double-buffer `slot`. selectTopKBlocks needs a
  // per-slot scratch `sel` so the worker (qb+1) and any main-thread prep don't
  // alias. computeBlockScores/kc_layer_ are only ever touched by a single
  // active prep at a time (main does qb=0, then the worker does qb>=1 for the
  // same layer), so no extra locking is needed.
  std::vector<int> sel0, sel1;
  auto prep_qb = [&](int layer, int qb_global, int slot) {
    std::vector<int>& sel = (slot == 0) ? sel0 : sel1;
    build_mask(qb_global, slot);
    selectTopKBlocks(layer, qb_global, sel);
    gather_one_qb(layer, qb_global, sel, slot);
    stage_q_qb(qb_global, slot);
    stage_k_curr_qb(qb_global, slot);
    stage_v_curr_qb(qb_global, slot);
  };

  // ---- Option-2 pipeline split: only the SCORING (selectTopKBlocks, the heavy
  // computeBlockScores matmul+softmax) runs on the worker; it produces just the
  // per-head `sel` int array (no QNN buffer), so it sidesteps the bind-once
  // graphExecute issue. The cheap gather+stage and the NPU dispatch stay on the
  // main thread into the SINGLE slot-0 buffers. `sel` is double-buffered
  // (sel0/sel1) so the worker writes sel[qb+1] while main consumes sel[qb].
  auto score_qb = [&](int layer, int qb_global, int sel_slot) {
    selectTopKBlocks(layer, qb_global, (sel_slot == 0) ? sel0 : sel1);
  };
  auto apply_qb = [&](int layer, int qb_global, int sel_slot) {
    std::vector<int>& sel = (sel_slot == 0) ? sel0 : sel1;
    build_mask(qb_global, /*slot=*/0);
    gather_one_qb(layer, qb_global, sel, /*slot=*/0);
    stage_q_qb(qb_global, /*slot=*/0);
    stage_k_curr_qb(qb_global, /*slot=*/0);
    stage_v_curr_qb(qb_global, /*slot=*/0);
  };
  auto dispatch_attn = [&](int layer, int slot) {
    std::vector<Tensor> ins = {q_qb_[slot], k_curr_qb_[slot], v_curr_qb_[slot],
                               k_arranged_[slot], v_arranged_[slot], mask_[slot]};
    attns_[layer]->setOutputTensors({attn_output_qb_[slot]});
    (void)(*attns_[layer])(ins);
  };

  // Persistent prep worker for the pipeline (created once, not per qb — a fresh
  // std::async per qb costs more than the overlap saves). Single-slot handoff:
  // main submits (layer,qb,slot), worker runs prep_qb, main waits.
  std::mutex pp_mtx;
  std::condition_variable pp_cv;
  bool pp_has_job = false, pp_done = true, pp_stop = false;
  int pp_l = 0, pp_q = 0, pp_s = 0;
  std::thread pp_worker;
  if (pipeline) {
    pp_worker = std::thread([&] {
      // Pin the prep worker to the prime cores (configurable via
      // MLLM_PIPELINE_WORKER_CPUS, e.g. "6,7"). Default {6,7}: on SM8750 those
      // are the 4.09 GHz prime cores; without pinning the scheduler parks the
      // worker on a 2.78 GHz perf core and the concurrent gather runs ~1.5×
      // slower than inline, defeating the overlap.
      {
        cpu_set_t set;
        CPU_ZERO(&set);
        bool any = false;
        if (const char* e = std::getenv("MLLM_PIPELINE_WORKER_CPUS")) {
          for (const char* p = e; *p;) { CPU_SET(std::atoi(p), &set); any = true; while (*p && *p != ',') ++p; while (*p == ',') ++p; }
        } else {
          CPU_SET(6, &set); CPU_SET(7, &set); any = true;
        }
        if (any) sched_setaffinity(0, sizeof(set), &set);
      }
      for (;;) {
        int l, q, s;
        {
          std::unique_lock<std::mutex> lk(pp_mtx);
          pp_cv.wait(lk, [&] { return pp_has_job || pp_stop; });
          if (pp_stop) return;
          l = pp_l; q = pp_q; s = pp_s; pp_has_job = false;
        }
        score_qb(l, q, s);  // worker does ONLY scoring → sel[s] (option 2)
        {
          std::lock_guard<std::mutex> lk(pp_mtx);
          pp_done = true;
        }
        pp_cv.notify_all();
      }
    });
  }
  auto pp_submit = [&](int l, int q, int s) {
    std::unique_lock<std::mutex> lk(pp_mtx);
    pp_cv.wait(lk, [&] { return pp_done; });
    pp_l = l; pp_q = q; pp_s = s; pp_has_job = true; pp_done = false;
    pp_cv.notify_all();
  };
  auto pp_wait = [&] {
    std::unique_lock<std::mutex> lk(pp_mtx);
    pp_cv.wait(lk, [&] { return pp_done; });
  };

  // Timing accumulators (µs, summed over all layers/qbs) — printed if
  // MLLM_SPLIT_PIPELINE_TIMING is set. prep = CPU gather/stage (main or worker),
  // disp = NPU attn dispatch (main), wait = main blocked in pp_wait for worker.
  long tp_prep_ = 0, tp_disp_ = 0, tp_wait_ = 0;

  for (int i = 0; i < L_; ++i) {
    // Commit this layer's full-Sq K/V into the cache BEFORE the per-qb loop
    // so per-qb gathers for qb >= 1 see freshly-computed K/V at positions
    // 0 .. (qb*BQ - 1) instead of stale/uninitialised cache bytes. The
    // sliding-window selectTopKBlocks policy reads block indices [0, qb_global)
    // out of the cache; that range overlaps THIS prefill's tokens, so the
    // commit must precede the gather.
    copy_kv_to_cache(i, /*base_pos=*/start_pos, /*n_tokens=*/num_tokens);

    using clk = std::chrono::high_resolution_clock;
    auto now_us = [] { return std::chrono::duration_cast<std::chrono::microseconds>(clk::now().time_since_epoch()).count(); };
    if (pipeline) {
      // Option 2: worker scores sel(qb+1) while main applies (gather+stage) +
      // dispatches attn(qb) into the SINGLE slot-0 buffers. sel is double-buffered
      // (qb&1) so worker(qb+1) and main(qb) don't alias. dispatch is synchronous,
      // so slot-0 is free to be refilled by apply(qb+1) after dispatch(qb) returns.
      auto t0 = now_us();
      score_qb(i, /*qb_global=*/0, /*sel_slot=*/0);  // prime sel[0] on main (qb 0: trivial, no scoring)
      tp_prep_ += now_us() - t0;
      for (int qb = 0; qb < num_qb_; ++qb) {
        const int cur = qb & 1;
        if (qb + 1 < num_qb_) pp_submit(i, qb + 1, (qb + 1) & 1);  // worker scores sel(qb+1)
        auto t1 = now_us();
        apply_qb(i, qb, cur);                                      // main: mask+gather+stage → slot 0
        tp_prep_ += now_us() - t1;
        auto td = now_us();
        dispatch_attn(i, /*slot=*/0);                              // NPU, concurrent with worker scoring
        tp_disp_ += now_us() - td;
        writeback_attn_output_qb(qb, /*slot=*/0);
        if (qb + 1 < num_qb_) { auto tw = now_us(); pp_wait(); tp_wait_ += now_us() - tw; }  // sel(qb+1) ready
      }
    } else {
      // Non-pipeline (serial): force slot 0 every qb. QNN graphExecute binds a
      // graph's I/O buffer only on its FIRST dispatch (QNNBackend.cpp `!isAlloc()`),
      // so alternating the double-buffer pointer (qb&1) makes attn read a stale
      // slot for qb>0 → garbage. A single fixed buffer, refilled before each
      // serial dispatch, is what bind-once expects. (Double-buffering needs a
      // graphExecute that rebinds per call — deferred with the pipeline path.)
      const char* qbprof = std::getenv("MLLM_QB_PROFILE");
      const int prof_layer = std::getenv("MLLM_QB_PROFILE_LAYER") ? std::atoi(std::getenv("MLLM_QB_PROFILE_LAYER")) : 0;
      for (int qb = 0; qb < num_qb_; ++qb) {
        const int s = 0;
        if (qbprof) {
          // Stage-resolved timing: sel (mask+select) | gather+stage | NPU attn.
          auto ta = now_us();
          build_mask(qb, s);
          selectTopKBlocks(i, qb, sel0);
          auto tb = now_us();
          gather_one_qb(i, qb, sel0, s);
          stage_q_qb(qb, s);
          stage_k_curr_qb(qb, s);
          stage_v_curr_qb(qb, s);
          auto tc = now_us();
          dispatch_attn(i, s);
          auto td2 = now_us();
          writeback_attn_output_qb(qb, s);
          const long long sel_us = tb - ta, gat_us = tc - tb, npu_us = td2 - tc;
          qp_sel_us_ += sel_us;
          qp_gat_us_ += gat_us;
          qp_npu_us_ += npu_us;
          if (i == prof_layer)
            MLLM_INFO("[qb prof] L{} qb{} sel={}us gather={}us npu={}us", i, qb, sel_us, gat_us, npu_us);
          continue;
        }
        auto t0 = now_us();
        prep_qb(i, qb, s);
        tp_prep_ += now_us() - t0;
        auto td = now_us();
        dispatch_attn(i, s);
        tp_disp_ += now_us() - td;
        writeback_attn_output_qb(qb, s);
      }
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

  // Stop the pipeline worker.
  if (pp_worker.joinable()) {
    {
      std::lock_guard<std::mutex> lk(pp_mtx);
      pp_stop = true;
    }
    pp_cv.notify_all();
    pp_worker.join();
  }

  if (std::getenv("MLLM_SPLIT_PIPELINE_TIMING")) {
    MLLM_INFO("[split timing] pipeline={} prep(main)={} ms  dispatch={} ms  pp_wait={} ms",
              pipeline, tp_prep_ / 1000.0, tp_disp_ / 1000.0, tp_wait_ / 1000.0);
  }
  if (std::getenv("MLLM_BLOCKSEL_TIMING")) {
    MLLM_INFO("[score timing] total={} ms | kc(K de-transpose)={} ms  prep(Q+K build)={} ms  "
              "matmul={} ms  softmax+pool={} ms",
              score_tot_us_ / 1000.0, score_kc_us_ / 1000.0, score_prep_us_ / 1000.0, score_mm_us_ / 1000.0,
              score_sm_us_ / 1000.0);
    score_mm_us_ = score_tot_us_ = score_kc_us_ = score_prep_us_ = score_sm_us_ = 0;
  }
  if (std::getenv("MLLM_QB_PROFILE")) {
    const long long tot = qp_sel_us_ + qp_gat_us_ + qp_npu_us_;
    MLLM_INFO("[qb prof TOTAL across {} layers] sel(select)={} ms  gather+stage={} ms  npu(attn)={} ms  | sum={} ms",
              L_, qp_sel_us_ / 1000.0, qp_gat_us_ / 1000.0, qp_npu_us_ / 1000.0, tot / 1000.0);
    qp_sel_us_ = qp_gat_us_ = qp_npu_us_ = 0;
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
