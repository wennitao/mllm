// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Causal block-sparse attention test. Companion to BlockSparseAttentionTest.cpp,
// which validated the non-causal sparse path. Before integrating block-sparse
// into prefill code we have to handle two things on top of the non-causal
// graph:
//
//   1. The diagonal k-block (k-block i for q-block i) must be present, and an
//      intra-block triangular mask must zero out the k > q positions inside it.
//   2. K-blocks > i (future tokens) must never be selected.
//
// Selection convention (random-causal + diagonal forced):
//   For q-block i, slot top_k-1 of the selection is forced to be k-block i
//   (the diagonal). Slots 0..top_k-2 are filled as follows:
//     - i >= top_k-1: random distinct draw of size top_k-1 from [0, i-1].
//     - i <  top_k-1: only i historical blocks available; slots 0..i-1 hold
//       a shuffle of {0..i-1}; slots i..top_k-2 are PADDING (gather fills
//       with k-block 0, but the mask zero-attentions them).
//
// Mask [1, num_q_blocks, BQ, top_k*BK] (broadcasts over Hq):
//   - diagonal slot (blk_idx == top_k-1): 0 if col_within_blk <= q else -1e4.
//   - historical slot (real, non-padding): 0.
//   - padding slot (i < top_k-1 and blk_idx in [i, top_k-2]): -1e4 for all
//     BK cols.
//
// Graph:
//   Q4d  = Reshape(Q, [Hq, num_q_blocks, BQ, D])
//   QK   = MatMul(Q4d, K_arr, transpose_in1=true)
//   QKs  = ElementWiseMul(QK, scale)
//   QKm  = ElementWiseAdd(QKs, mask)               <- new vs non-causal
//   P    = Softmax(QKm, axis=3)
//   O4d  = MatMul(P, V_arr)
//   O    = Reshape(O4d, [Hq, Sq, D])
//
// References:
//   - naiveCausalDenseFp16            : dense attention with [1, Sq, Skv] causal mask.
//   - naiveCausalBlockSparseFromArranged : block-sparse attention on the gathered
//     K_arr/V_arr, applying the same per-block mask used by the QNN graph.
//
// The dense NPU baseline is built and timed for comparison (same shape graph as
// FlashAttentionDecomposedTest's causal prefill case).

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

#include "mllm/backends/cpu/CPUBackend.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "QnnBackend.h"

using namespace mllm;
using namespace mllm::qnn;

namespace {

constexpr float kMaskNeg = -1.0e4f;  // -inf surrogate: avoids NaN if a row is fully masked.

void unbufferOutput() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
}

int timingRunsFromEnv() {
  const char* v = std::getenv("MLLM_QNN_FA_TIMING_RUNS");
  if (!v || v[0] == '\0') return 1;
  char* end = nullptr;
  long r = std::strtol(v, &end, 10);
  return (end != v && r > 0) ? (int)r : 1;
}

}  // namespace

class BlockSparseAttentionCausalTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    unbufferOutput();
    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string p = existing && existing[0] != '\0' ? (std::string(".;/data/local/tmp;") + existing) : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", p.c_str(), /*overwrite=*/1);
    }
    ASSERT_TRUE(isQnnAvailable());

    auto& ctx = Context::instance();
    auto host_backend = cpu::createCPUBackend();
    ctx.registerBackend(host_backend);
    ctx.memoryManager()->registerAllocator(kCPU, host_backend->allocator(), MemoryManagerOptions());

    backend_ = std::make_shared<QNNBackend>();
    ASSERT_TRUE(backend_->createContext());
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(
        kQNN, backend_->allocator(), {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
    ctx.dispatcherManager()->registerDispatcher(
        createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), QNNDispatcherOptions()));
  }
  static void TearDownTestSuite() { backend_.reset(); }
  static std::shared_ptr<QNNBackend> backend_;
};
std::shared_ptr<QNNBackend> BlockSparseAttentionCausalTest::backend_ = nullptr;

// ---------------------------------------------------------------------------
// Selection: random-causal with diagonal forced at slot top_k-1.
//
// Layout: sel[(h * num_q_blocks + i) * top_k + kk] = k-block index for q-block
// i, head h, slot kk.
//
// Selection is generated per-head so the gather call is unchanged. The mask
// is broadcast over Hq, so we keep selections consistent ACROSS heads (same
// k-block layout for every head); only the inner randomness varies per (h, i).
//
// For q-block i:
//   slot top_k-1 = i                                  (diagonal — always)
//   slot kk in [0, top_k-2]:
//     if i >= top_k-1: distinct random in [0, i-1]
//     if i <  top_k-1:
//       slot < i        : shuffle of {0..i-1}
//       slot >= i       : 0      (padding — k-block 0, will be fully masked)
// ---------------------------------------------------------------------------
static void causalSelection(std::vector<int>& sel, int Hq, int num_q_blocks, int top_k, std::mt19937& rng) {
  sel.resize((size_t)Hq * num_q_blocks * top_k);
  std::vector<int> pool;
  for (int h = 0; h < Hq; ++h) {
    for (int i = 0; i < num_q_blocks; ++i) {
      int* row = sel.data() + ((size_t)h * num_q_blocks + i) * top_k;
      // Diagonal at the last slot.
      row[top_k - 1] = i;
      const int n_hist_avail = i;                                  // historical blocks: [0, i-1]
      const int n_hist_slots = std::min(top_k - 1, n_hist_avail);  // slots that get a real historical block
      // Real historical: distinct random.
      pool.resize(n_hist_avail);
      std::iota(pool.begin(), pool.end(), 0);
      std::shuffle(pool.begin(), pool.end(), rng);
      for (int kk = 0; kk < n_hist_slots; ++kk) row[kk] = pool[kk];
      // Padding slots (only present if i < top_k-1): set to 0 (will be masked).
      for (int kk = n_hist_slots; kk < top_k - 1; ++kk) row[kk] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// CPU block gather (same as BlockSparseAttentionTest; copied here so this
// file is self-contained).
//   src : K [Hq, Skv, D]   dst : K_arr [Hq, num_q_blocks, top_k*BK, D]
// ---------------------------------------------------------------------------
static void cpuGather(const __fp16* src, __fp16* dst, const std::vector<int>& sel, int Hq, int Skv, int D, int BK,
                      int num_q_blocks, int top_k) {
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      __fp16* dst_qb = dst + (((size_t)h * num_q_blocks + q) * top_k * BK) * D;
      for (int kk = 0; kk < top_k; ++kk) {
        int j = sel[((size_t)h * num_q_blocks + q) * top_k + kk];
        const __fp16* s = src + ((size_t)h * Skv + j * BK) * D;
        std::memcpy(dst_qb + kk * BK * D, s, per_chunk);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Build the [BQ, top_k*BK] causal mask for a single q-block i. Used as the
// per-q-block building block for all three layouts below.
// ---------------------------------------------------------------------------
static void buildCausalMaskOneQb(__fp16* mask_qb, int i, int BQ, int BK, int top_k) {
  const int top_k_BK = top_k * BK;
  const int hist_slots = std::min(top_k - 1, i);  // slots that hold real historical blocks
  for (int q = 0; q < BQ; ++q) {
    __fp16* row = mask_qb + (size_t)q * top_k_BK;
    for (int slot = 0; slot < top_k - 1; ++slot) {
      const bool is_padding = (slot >= hist_slots);
      for (int s = 0; s < BK; ++s) {
        row[slot * BK + s] = is_padding ? (__fp16)kMaskNeg : (__fp16)0.0f;
      }
    }
    const int diag_off = (top_k - 1) * BK;
    for (int s = 0; s < BK; ++s) {
      row[diag_off + s] = (s <= q) ? (__fp16)0.0f : (__fp16)kMaskNeg;
    }
  }
}

// Rank-4 mask: [1, num_q_blocks, BQ, top_k*BK]. Broadcast over Hq via the
// leading 1.
static void buildCausalMask(__fp16* mask, int num_q_blocks, int BQ, int BK, int top_k) {
  const int top_k_BK = top_k * BK;
  for (int i = 0; i < num_q_blocks; ++i) {
    buildCausalMaskOneQb(mask + (size_t)i * BQ * top_k_BK, i, BQ, BK, top_k);
  }
}

// Big-batch rank-3 mask: [big_batch=Hq*num_qb, BQ, top_k*BK]. Reshape would
// give the same byte layout as the rank-4 mask only if mask depended solely
// on (i, q) — which it does — but the Hq leading dim is flattened *into*
// big_batch, so we tile across Hq.
static void buildCausalMaskBigBatch(__fp16* mask, int Hq, int num_q_blocks, int BQ, int BK, int top_k) {
  const int top_k_BK = top_k * BK;
  const size_t per_qb_bytes = (size_t)BQ * top_k_BK * sizeof(__fp16);
  // Build one block-of-num_qb section, then tile.
  std::vector<__fp16> single(num_q_blocks * BQ * top_k_BK);
  for (int i = 0; i < num_q_blocks; ++i) {
    buildCausalMaskOneQb(single.data() + (size_t)i * BQ * top_k_BK, i, BQ, BK, top_k);
  }
  for (int h = 0; h < Hq; ++h) {
    for (int i = 0; i < num_q_blocks; ++i) {
      std::memcpy(mask + ((size_t)h * num_q_blocks + i) * BQ * top_k_BK,
                  single.data() + (size_t)i * BQ * top_k_BK, per_qb_bytes);
    }
  }
}

// ---------------------------------------------------------------------------
// Host references.
// ---------------------------------------------------------------------------

// Dense causal attention: full Skv columns, mask k > (Skv - Sq) + q.
static void naiveCausalDenseFp16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, int Hq, int Sq, int Skv,
                                 int D, float scale) {
  std::vector<float> scores((size_t)Skv);
  for (int h = 0; h < Hq; ++h) {
    for (int s = 0; s < Sq; ++s) {
      const int kv_lim = (Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      const __fp16* qrow = Q + ((size_t)h * Sq + s) * D;
      __fp16* orow = O + ((size_t)h * Sq + s) * D;
      float row_max = -INFINITY;
      for (int j = 0; j < Skv; ++j) {
        if (j >= kv_lim) {
          scores[j] = kMaskNeg;
          continue;
        }
        const __fp16* krow = K + ((size_t)h * Skv + j) * D;
        float dot = 0.f;
        for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)krow[d];
        scores[j] = dot * scale;
        if (scores[j] > row_max) row_max = scores[j];
      }
      float denom = 0.f;
      for (int j = 0; j < Skv; ++j) {
        scores[j] = std::exp(scores[j] - row_max);
        denom += scores[j];
      }
      const float inv = denom > 0.f ? 1.f / denom : 0.f;
      std::vector<float> acc(D, 0.f);
      for (int j = 0; j < Skv; ++j) {
        if (j >= kv_lim) continue;
        const __fp16* vrow = V + ((size_t)h * Skv + j) * D;
        const float p = scores[j] * inv;
        for (int d = 0; d < D; ++d) acc[d] += p * (float)vrow[d];
      }
      for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
    }
  }
}

// Host reference for the sparse path: matches what the QNN graph computes.
// Operates on the gathered K_arr/V_arr and the same causal mask.
static void naiveCausalBlockSparseFromArranged(const __fp16* Q, const __fp16* K_arr, const __fp16* V_arr,
                                               const __fp16* mask, __fp16* O, int Hq, int Sq, int D, int BQ,
                                               int top_k_BK, int num_q_blocks, float scale) {
  std::vector<float> scores((size_t)top_k_BK);
  for (int h = 0; h < Hq; ++h) {
    for (int i = 0; i < num_q_blocks; ++i) {
      const __fp16* K_qb = K_arr + (((size_t)h * num_q_blocks + i) * top_k_BK) * D;
      const __fp16* V_qb = V_arr + (((size_t)h * num_q_blocks + i) * top_k_BK) * D;
      for (int q = 0; q < BQ; ++q) {
        const int row = i * BQ + q;
        const __fp16* qrow = Q + ((size_t)h * Sq + row) * D;
        __fp16* orow = O + ((size_t)h * Sq + row) * D;
        const __fp16* mrow = mask + ((size_t)i * BQ + q) * top_k_BK;

        float row_max = -INFINITY;
        for (int c = 0; c < top_k_BK; ++c) {
          const __fp16* kc = K_qb + (size_t)c * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)kc[d];
          scores[c] = dot * scale + (float)mrow[c];
          if (scores[c] > row_max) row_max = scores[c];
        }
        float denom = 0.f;
        for (int c = 0; c < top_k_BK; ++c) {
          scores[c] = std::exp(scores[c] - row_max);
          denom += scores[c];
        }
        const float inv = denom > 0.f ? 1.f / denom : 0.f;
        std::vector<float> acc(D, 0.f);
        for (int c = 0; c < top_k_BK; ++c) {
          const __fp16* vc = V_qb + (size_t)c * D;
          const float p = scores[c] * inv;
          for (int d = 0; d < D; ++d) acc[d] += p * (float)vc[d];
        }
        for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Graph runner helper (mirrors BlockSparseAttentionTest's runOneGraph).
// ---------------------------------------------------------------------------
struct GraphTime {
  double avg_ms;
  float max_abs_err;
  size_t mismatch;
};

template <typename BuildFn>
static GraphTime runOneGraph(const std::shared_ptr<QNNBackend>& backend, const std::string& gname,
                             const std::vector<Tensor>& ins, const Tensor& O, const std::vector<__fp16>& ref,
                             size_t numel, float tol, BuildFn build) {
  EXPECT_NE(backend->createQnnGraph(gname), nullptr) << gname;
  build(gname);
  EXPECT_TRUE(backend->graphFinalize(gname)) << "finalize failed: " << gname;

  std::vector<Tensor> ins_copy = ins;
  std::vector<Tensor> outs = {O};
  backend->graphExecute(gname, ins_copy, outs);  // warmup

  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(gname, ins_copy, outs);
  const auto t1 = std::chrono::steady_clock::now();

  GraphTime g{};
  g.avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
  const __fp16* got = O.ptr<__fp16>();
  g.mismatch = 0;
  g.max_abs_err = 0.f;
  for (size_t i = 0; i < numel; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > g.max_abs_err) g.max_abs_err = diff;
    if (diff > tol) ++g.mismatch;
  }
  return g;
}

// ---------------------------------------------------------------------------
// Main comparison entry point.
// ---------------------------------------------------------------------------
static void runCausalComparison(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                                int BK, int top_k, bool with_dense_baseline, const std::string& tag) {
  fprintf(stderr, "[CASE] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d (causal, dense_baseline=%d)\n", tag.c_str(),
          Sq, Hq, Skv, Hkv, D, BK, top_k, (int)with_dense_baseline);
  ASSERT_EQ(Sq, Skv) << "causal prefill expects Sq == Skv";
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  // ----- Allocate tensors -----
  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O_sparse = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto O_dense = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  // ----- Fill Q, K, V (GQA pre-expanded so the test data path matches the
  // production layout). -----
  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* vp = V.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  // ----- Causal selection + CPU gather -----
  std::vector<int> sel;
  causalSelection(sel, Hq, num_q_blocks, top_k, rng);
  cpuGather(kp, K_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);

  // ----- Static causal mask for the sparse graph: [1, num_q_blocks, BQ, top_k*BK] -----
  auto sparse_mask_t = Tensor::empty({1, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMask(sparse_mask_t.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k);

  // ----- Static causal mask for the dense graph: [1, Sq, Skv]. Only built if
  // the dense baseline is requested — at Sq=2048 the dense decomposed graph
  // OOMs the QNN PD on the [Hq, Sq, Skv] intermediate score tensor (128 MB just
  // for QK at fp16; QKs/QKm/P each add another 128 MB). The dense
  // graphFinalize then silently produces a graph that emits zeros at execute
  // time, which would fail the dense host-ref check unrelated to the sparse
  // path we're validating. -----
  Tensor dense_mask_t;
  if (with_dense_baseline) {
    dense_mask_t = Tensor::empty({1, Sq, Skv}, kFloat16, kQNN).alloc();
    __fp16* mp = dense_mask_t.ptr<__fp16>();
    for (int s = 0; s < Sq; ++s) {
      const int kv_lim = (Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      for (int j = 0; j < Skv; ++j) mp[s * Skv + j] = (j < kv_lim) ? (__fp16)0.0f : (__fp16)kMaskNeg;
    }
  }

  // ----- Static scale tensors. Dense uses rank-3, sparse rank-4. -----
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;
  auto scale_4d = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_4d.ptr<__fp16>()[0] = (__fp16)scale;

  // ----- Host references -----
  std::vector<__fp16> ref_dense(out_numel);
  std::vector<__fp16> ref_sparse(out_numel);
  if (with_dense_baseline) naiveCausalDenseFp16(qp, kp, vp, ref_dense.data(), Hq, Sq, Skv, D, scale);
  naiveCausalBlockSparseFromArranged(qp, K_arr.ptr<__fp16>(), V_arr.ptr<__fp16>(), sparse_mask_t.ptr<__fp16>(),
                                     ref_sparse.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks, scale);

  // ---------------------------------------------------------------------
  // Graph 1 (optional): dense causal attention (rank-3, MatMul → Mul →
  // Add(mask) → Softmax → MatMul). Mirrors FlashAttentionDecomposedTest's
  // causal path.
  // ---------------------------------------------------------------------
  GraphTime r_dense{};
  if (with_dense_baseline) {
    const std::string g_dense = "dense_causal_" + tag;
    r_dense = runOneGraph(backend, g_dense, {Q, K, V}, O_dense, ref_dense, out_numel, /*tol=*/5e-2f,
                          [&](const std::string& g) {
      backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
      backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
      backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
      backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_dense);
      backend->addStaticTensor(g, "scale", scale_3d);
      backend->addStaticTensor(g, "mask", dense_mask_t);

      auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
      auto QKs_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
      auto QKm_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
      auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
      backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
      backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
      backend->addTensor(g, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);
      backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);

      std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
          QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
      backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
      backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
      backend->graphAddNode(g, "add_mask", "ElementWiseAdd", {"QKs", "mask"}, {"QKm"}, {}, {}, "qti.aisw");
      std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
          QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
      backend->graphAddNode(g, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
      backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    });
  }

  // ---------------------------------------------------------------------
  // Graph 2: causal block-sparse. Adds ElementWiseAdd(mask) before Softmax
  // to inject the intra-block triangular mask on the diagonal tile and
  // fully-mask any padding slots.
  // ---------------------------------------------------------------------
  const std::string g_sparse = "sparse_causal_" + tag;
  GraphTime r_sparse = runOneGraph(backend, g_sparse, {Q, K_arr, V_arr}, O_sparse, ref_sparse, out_numel,
                                   /*tol=*/5e-2f, [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
    backend->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_sparse);
    backend->addStaticTensor(g, "scale", scale_4d);
    backend->addStaticTensor(g, "mask", sparse_mask_t);

    auto Q4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKm_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto O4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    backend->addTensor(g, "Q4d", QNN_TENSOR_TYPE_NATIVE, Q4d_t);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(g, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(g, "O4d", QNN_TENSOR_TYPE_NATIVE, O4d_t);

    backend->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q4d"}, {}, {}, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q4d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "add_mask", "ElementWiseAdd", {"QKs", "mask"}, {"QKm"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
        QNNParamScalarWrapper::create<uint32_t>("axis", 3u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O4d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "reshape_o", "Reshape", {"O4d"}, {"O"}, {}, {}, "qti.aisw");
  });

  // ----- Cross-check (only when dense baseline ran): sparse-vs-dense output
  // divergence. Sparse is an approximation of dense — informational, not
  // asserted. -----
  if (with_dense_baseline) {
    float xerr_max = 0.f;
    double xerr_rms = 0.0;
    const __fp16* gs = O_sparse.ptr<__fp16>();
    for (size_t i = 0; i < out_numel; ++i) {
      float diff = std::fabs((float)gs[i] - (float)ref_dense[i]);
      if (diff > xerr_max) xerr_max = diff;
      xerr_rms += (double)diff * (double)diff;
    }
    xerr_rms = std::sqrt(xerr_rms / out_numel);
    fprintf(stderr,
            "[CMP-CAUSAL %-14s]\n"
            "  NPU dense  causal       : %7.2f ms  err_vs_ref=%.4f miss=%zu\n"
            "  NPU sparse causal       : %7.2f ms  err_vs_ref=%.4f miss=%zu\n"
            "  Sparse vs dense         : max=%.4f  rms=%.4f  (sparse is an approximation)\n",
            tag.c_str(), r_dense.avg_ms, r_dense.max_abs_err, r_dense.mismatch, r_sparse.avg_ms, r_sparse.max_abs_err,
            r_sparse.mismatch, xerr_max, xerr_rms);
    EXPECT_EQ(r_dense.mismatch, 0u) << "dense causal NPU diverged from host reference";
  } else {
    fprintf(stderr,
            "[CMP-CAUSAL %-14s] (dense baseline skipped — Sq²·Hq intermediate would OOM the QNN PD)\n"
            "  NPU sparse causal       : %7.2f ms  err_vs_ref=%.4f miss=%zu\n",
            tag.c_str(), r_sparse.avg_ms, r_sparse.max_abs_err, r_sparse.mismatch);
  }
  EXPECT_EQ(r_sparse.mismatch, 0u) << "sparse causal NPU diverged from host reference (mask wiring?)";
}

// ===========================================================================
// Method 2: big-batch rank-3 causal sparse.
//
// Same data as rank-4, just declared shape changes:
//   K_arr [Hq, num_qb, top_k*BK, D]  →  K_arr3d [Hq*num_qb, top_k*BK, D]
// The graph:
//   Q (rank-3 [Hq, Sq, D]) → Reshape → Q3d [Hq*num_qb, BQ, D]
//   QK = MatMul(Q3d, K_arr3d, transpose_in1) → [big_batch, BQ, top_k*BK]
//   ... Mul → Add(mask_bb) → Softmax → MatMul(V) → O3d
//   Reshape O3d → O [Hq, Sq, D]
//
// mask_bb shape: [big_batch, BQ, top_k*BK]. Selection is the same across
// heads, so the [num_qb, BQ, top_k*BK] base mask is tiled Hq times along the
// leading dim.
// ===========================================================================
static void runCausalBigBatch(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                              int BK, int top_k, const std::string& tag) {
  fprintf(stderr, "[CASE big-batch rank-3] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d (causal)\n", tag.c_str(),
          Sq, Hq, Skv, Hkv, D, BK, top_k);
  ASSERT_EQ(Sq, Skv);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;
  const int big_batch = Hq * num_q_blocks;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  // Rank-3 layout. Bytes identical to [Hq, num_qb, top_k_BK, D].
  auto K_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* vp = V.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  std::vector<int> sel;
  causalSelection(sel, Hq, num_q_blocks, top_k, rng);
  cpuGather(kp, K_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);

  // Mask: [big_batch, BQ, top_k*BK]. Tile the per-q-block mask Hq times
  // along the leading dim — same selection layout per head so this is exact.
  //
  // We tried a smaller [1, num_qb, BQ, top_k*BK] mask bracketed by
  // rank-3↔rank-4 Reshape nodes around the add: it works when it works but
  // the QNN PD silently fails graphFinalize ~1 in 3 runs across all Sq with
  // that pattern. The bigger tiled mask is stable; the trade-off is that
  // big-batch causal OOMs the PD at Sq≥1024 (same ceiling as rank-4),
  // matching the doc's note that only per-qb pipelined dodges this wall.
  auto mask_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMaskBigBatch(mask_t.ptr<__fp16>(), Hq, num_q_blocks, BQ, BK, top_k);

  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;

  // Host reference: use the per-qb mask layout we already have (the byte
  // layout K_arr/V_arr is identical between rank-3 and rank-4 forms).
  auto mask_ref = Tensor::empty({num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMask(mask_ref.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k);
  std::vector<__fp16> ref(out_numel);
  naiveCausalBlockSparseFromArranged(qp, K_arr.ptr<__fp16>(), V_arr.ptr<__fp16>(), mask_ref.ptr<__fp16>(), ref.data(),
                                     Hq, Sq, D, BQ, top_k_BK, num_q_blocks, scale);

  const std::string gn = "bb_causal_" + tag;
  GraphTime r = runOneGraph(backend, gn, {Q, K_arr, V_arr}, O, ref, out_numel, /*tol=*/5e-2f,
                            [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
    backend->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O);
    backend->addStaticTensor(g, "scale", scale_3d);
    backend->addStaticTensor(g, "mask", mask_t);

    auto Q3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKm_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto O3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
    backend->addTensor(g, "Q3d", QNN_TENSOR_TYPE_NATIVE, Q3d_t);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(g, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(g, "O3d", QNN_TENSOR_TYPE_NATIVE, O3d_t);

    backend->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q3d"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q3d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "add_mask", "ElementWiseAdd", {"QKs", "mask"}, {"QKm"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
        QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O3d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "reshape_o", "Reshape", {"O3d"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[BB-CAUSAL %-14s] Sq=%-4d top_k=%-3d big_batch=%-4d\n"
          "  NPU big-batch rank-3 sparse causal : %7.2f ms  err_vs_ref=%.4f miss=%zu\n",
          tag.c_str(), Sq, top_k, big_batch, r.avg_ms, r.max_abs_err, r.mismatch);
  EXPECT_EQ(r.mismatch, 0u) << "big-batch rank-3 sparse causal NPU diverged from host reference";
}

// ===========================================================================
// Method 3: per-q-block pipelined rank-3 causal sparse.
//
// num_q_blocks rank-3 dispatches per layer with CPU prep ∥ NPU execute via a
// persistent worker thread (cv-based). Each per-qb shape:
//   Q_buf  [Hq, BQ, D]
//   K_buf  [Hq, top_k*BK, D]
//   V_buf  [Hq, top_k*BK, D]
//   mask   [1, BQ, top_k*BK]      ← dynamic; CPU writes per dispatch
//   O_buf  [Hq, BQ, D]
//
// The mask varies per qb (intra-block triangular position is i-relative for
// early q-blocks), so it MUST be APP_WRITE not static. Two graphs ping-pong
// across (Q,K,V,mask,O) buffer pairs to allow concurrent CPU prep + NPU exec.
//
// Correctness: sync mode writes O_buf[b] → output[h, i*BQ:(i+1)*BQ, :] for
// each qb. Final stitched O is checked against the host reference.
// Timing: separately runs a pipelined version with persistent cv worker.
// ===========================================================================

// Copy Q[h, i*BQ:(i+1)*BQ, :] for one q-block.
static void copyQSlice(const __fp16* Q_full, __fp16* Q_qb, int qb, int Hq, int Sq, int D, int BQ) {
  const size_t row_bytes = (size_t)BQ * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    std::memcpy(Q_qb + (size_t)h * BQ * D, Q_full + ((size_t)h * Sq + (size_t)qb * BQ) * D, row_bytes);
  }
}
// Gather K (or V) for one q-block into [Hq, top_k*BK, D].
static void gatherOneQb(const __fp16* src, __fp16* dst, const int* sel_full, int qb, int Hq, int Skv, int D, int BK,
                        int num_q_blocks, int top_k) {
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    __fp16* dst_h = dst + (size_t)h * top_k * BK * D;
    for (int kk = 0; kk < top_k; ++kk) {
      int j = sel_full[((size_t)h * num_q_blocks + qb) * top_k + kk];
      std::memcpy(dst_h + (size_t)kk * BK * D, src + ((size_t)h * Skv + (size_t)j * BK) * D, per_chunk);
    }
  }
}
// Per-qb PADDING mask shape [1, top_k*BK] — uniform across q-rows within a
// q-block. Slots 0..hist_slots-1 are real (mask=0), slots hist_slots..
// top_k-2 are padding (mask=kMaskNeg), slot top_k-1 is diagonal (mask=0
// here; the per-row triangle is handled by the static mask below).
//
// Tiny: 512 bytes at top_k=8. Bound APP_WRITE per dispatch.
//
// (We tried CPU-masking K_padding to a large negative constant to
// avoid this Add entirely. It doesn't work robustly: QK[padding] =
// Σ_d Q_d · (-L) = -L · Σ_d Q_d, which is large NEGATIVE only when
// Σ_d Q_d > 0. For random Q ~50% of rows have Σ < 0 → QK becomes
// large POSITIVE, softmax weight → 1 at padding columns → wrong
// output. Sign of Σ Q_d can't be controlled CPU-side.)
static void buildPaddingMaskOneQb(__fp16* mask_qb, int qb, int BK, int top_k) {
  const int hist_slots = std::min(top_k - 1, qb);
  const int diag_slot  = top_k - 1;
  for (int kk = 0; kk < top_k; ++kk) {
    const bool is_padding = (kk >= hist_slots) && (kk < diag_slot);
    const __fp16 v = is_padding ? (__fp16)kMaskNeg : (__fp16)0.0f;
    for (int s = 0; s < BK; ++s) mask_qb[kk * BK + s] = v;
  }
}

// Static triangle mask shape [1, BQ, top_k * BK]. Zeros everywhere
// except the diagonal slot (slot top_k - 1) which carries the
// lower-triangular causal pattern: column c is 0 if c <= q else kMaskNeg.
//
// This mask is the same for every q-block (the diagonal triangle pattern
// doesn't depend on qb), so it's bound STATIC at graph build time —
// no per-dispatch DMA cost.
static void buildStaticTriangleMask(__fp16* mask, int BQ, int BK, int top_k) {
  const int top_k_BK = top_k * BK;
  const int diag_off = (top_k - 1) * BK;
  for (int q = 0; q < BQ; ++q) {
    __fp16* row = mask + (size_t)q * top_k_BK;
    std::fill(row, row + (size_t)diag_off, (__fp16)0.0f);
    for (int s = 0; s < BK; ++s) {
      row[diag_off + s] = (s <= q) ? (__fp16)0.0f : (__fp16)kMaskNeg;
    }
  }
}

// Stitch O_buf [Hq, BQ, D] back into O_full [Hq, Sq, D] at q-block qb.
static void stitchO(__fp16* O_full, const __fp16* O_qb, int qb, int Hq, int Sq, int D, int BQ) {
  const size_t row_bytes = (size_t)BQ * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    std::memcpy(O_full + ((size_t)h * Sq + (size_t)qb * BQ) * D, O_qb + (size_t)h * BQ * D, row_bytes);
  }
}

static void runCausalPerQbPipelined(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                    int D, int BK, int top_k, const std::string& tag) {
  fprintf(stderr, "[CASE per-qb pipelined rank-3] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d (causal)\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k);
  ASSERT_EQ(Sq, Skv);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

  std::array<Tensor, 2> Q_buf, K_buf, V_buf, M_buf, O_buf;
  for (int b = 0; b < 2; ++b) {
    Q_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
    K_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    M_buf[b] = Tensor::empty({1, BQ, top_k_BK}, kFloat16, kQNN).alloc();
    O_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  }

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q_full.ptr<__fp16>();
  __fp16* kp = K_full.ptr<__fp16>();
  __fp16* vp = V_full.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  std::vector<int> sel;
  causalSelection(sel, Hq, num_q_blocks, top_k, rng);

  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  // Host reference. Need full [Hq, num_qb, top_k_BK, D] K_arr/V_arr to call
  // the existing reference; build them once.
  auto K_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  cpuGather(kp, K_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  auto mask_ref_t = Tensor::empty({num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMask(mask_ref_t.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k);
  std::vector<__fp16> ref(out_numel);
  naiveCausalBlockSparseFromArranged(qp, K_arr_ref.ptr<__fp16>(), V_arr_ref.ptr<__fp16>(),
                                     mask_ref_t.ptr<__fp16>(), ref.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks,
                                     scale);

  // ----- Build 2 rank-3 graphs (one per buffer set), with APP_WRITE mask. -----
  auto buildGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    backend->addTensor(gname, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_buf[b]);
    backend->addTensor(gname, "K", QNN_TENSOR_TYPE_APP_WRITE, K_buf[b]);
    backend->addTensor(gname, "V", QNN_TENSOR_TYPE_APP_WRITE, V_buf[b]);
    backend->addTensor(gname, "M", QNN_TENSOR_TYPE_APP_WRITE, M_buf[b]);
    backend->addTensor(gname, "O", QNN_TENSOR_TYPE_APP_READ, O_buf[b]);
    backend->addStaticTensor(gname, "scale", scale_t);

    auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKm_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    backend->addTensor(gname, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gname, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);
    backend->addTensor(gname, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gname, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gname, "add_mask", "ElementWiseAdd", {"QKs", "M"}, {"QKm"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
        QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gname, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gname, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  const std::string g0 = "perqb_b0_" + tag;
  const std::string g1 = "perqb_b1_" + tag;
  buildGraph(g0, 0);
  buildGraph(g1, 1);

  auto prep_qb = [&](int qb, int b) {
    copyQSlice(qp, Q_buf[b].ptr<__fp16>(), qb, Hq, Sq, D, BQ);
    gatherOneQb(kp, K_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    gatherOneQb(vp, V_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    buildCausalMaskOneQb(M_buf[b].ptr<__fp16>(), qb, BQ, BK, top_k);
  };

  // Warmup: prep + execute both graphs once on qb=0 (so their JIT-compile cost
  // is paid before timing).
  prep_qb(0, 0);
  prep_qb(0, 1);
  std::vector<Tensor> ins0 = {Q_buf[0], K_buf[0], V_buf[0], M_buf[0]};
  std::vector<Tensor> outs0 = {O_buf[0]};
  std::vector<Tensor> ins1 = {Q_buf[1], K_buf[1], V_buf[1], M_buf[1]};
  std::vector<Tensor> outs1 = {O_buf[1]};
  backend->graphExecute(g0, ins0, outs0);
  backend->graphExecute(g1, ins1, outs1);

  // ============================== SYNC (correctness) ==============================
  // prep_qb → execute serially for each qb, stitch O_buf into a host buffer,
  // then compare against the reference.
  std::vector<__fp16> got(out_numel, (__fp16)0.0f);
  const auto sync_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < num_q_blocks; ++i) {
    int b = i % 2;
    prep_qb(i, b);
    std::vector<Tensor> ins = {Q_buf[b], K_buf[b], V_buf[b], M_buf[b]};
    std::vector<Tensor> outs = {O_buf[b]};
    backend->graphExecute(b == 0 ? g0 : g1, ins, outs);
    stitchO(got.data(), O_buf[b].ptr<__fp16>(), i, Hq, Sq, D, BQ);
  }
  const auto sync_t1 = std::chrono::steady_clock::now();
  const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

  size_t miss = 0;
  float max_abs_err = 0.f;
  for (size_t i = 0; i < out_numel; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > max_abs_err) max_abs_err = diff;
    if (diff > 5e-2f) ++miss;
  }

  // ============================== PIPELINED (timing) ==============================
  // Persistent cv-based worker thread: NPU dispatch of qb i on the main
  // thread runs concurrently with CPU prep_qb of qb i+1 on the worker.
  std::mutex mu;
  std::condition_variable cv_req, cv_done;
  int req_qb = 0, req_b = 0;
  std::atomic<bool> has_req{false};
  std::atomic<bool> worker_done_pending{false};
  std::atomic<bool> stop_worker{false};
  std::thread worker([&]() {
    while (true) {
      std::unique_lock<std::mutex> lk(mu);
      cv_req.wait(lk, [&]() { return has_req.load() || stop_worker.load(); });
      if (stop_worker.load()) break;
      int qb_local = req_qb, b_local = req_b;
      has_req.store(false);
      lk.unlock();
      prep_qb(qb_local, b_local);
      {
        std::lock_guard<std::mutex> lk2(mu);
        worker_done_pending.store(true);
      }
      cv_done.notify_one();
    }
  });
  auto submit_prep = [&](int qb, int b) {
    {
      std::lock_guard<std::mutex> lk(mu);
      req_qb = qb;
      req_b = b;
      has_req.store(true);
    }
    cv_req.notify_one();
  };
  auto wait_prep = [&]() {
    std::unique_lock<std::mutex> lk(mu);
    cv_done.wait(lk, [&]() { return worker_done_pending.load(); });
    worker_done_pending.store(false);
  };

  prep_qb(0, 0);  // first qb sync on main thread
  const auto pipe_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < num_q_blocks; ++i) {
    int cur = i % 2;
    bool has_next = (i + 1 < num_q_blocks);
    if (has_next) submit_prep(i + 1, (i + 1) % 2);
    std::vector<Tensor> ins = {Q_buf[cur], K_buf[cur], V_buf[cur], M_buf[cur]};
    std::vector<Tensor> outs = {O_buf[cur]};
    backend->graphExecute(cur == 0 ? g0 : g1, ins, outs);
    if (has_next) wait_prep();
  }
  const auto pipe_t1 = std::chrono::steady_clock::now();
  const double pipe_ms = std::chrono::duration<double, std::milli>(pipe_t1 - pipe_t0).count();
  stop_worker.store(true);
  cv_req.notify_one();
  worker.join();

  fprintf(stderr,
          "[PERQB-CAUSAL %-10s] Sq=%-4d top_k=%-3d num_qb=%-3d\n"
          "  sync (prep→exec serially) : %7.2f ms  err_vs_ref=%.4f miss=%zu\n"
          "  pipelined (CPU∥NPU)       : %7.2f ms\n"
          "  pipeline speedup vs sync  : %.2fx\n",
          tag.c_str(), Sq, top_k, num_q_blocks, sync_ms, max_abs_err, miss, pipe_ms,
          (pipe_ms > 0.0) ? sync_ms / pipe_ms : 0.0);
  EXPECT_EQ(miss, 0u) << "per-qb pipelined rank-3 sparse causal NPU diverged from host reference";
}

// ===========================================================================
// Method 3b: per-q-block rank-3 sparse with split mask — TINY per-qb padding
// mask + STATIC triangle mask.
//
// Splits the per-qb dynamic mask into two pieces:
//   1. PADDING mask [1, top_k*BK] — uniform across q-rows within a q-block,
//      bound APP_WRITE per dispatch. 512 bytes (vs 16 KB for the original
//      per-qb [1, BQ, top_k*BK] mask).
//   2. TRIANGLE mask [1, BQ, top_k*BK] — same for every q-block, bound STATIC
//      at graph build. 16 KB but no per-dispatch DMA.
//
// Graph applies them as two Adds:
//   QK = MatMul(Q, K)
//   QKs = Mul(QK, scale)
//   QKpm = Add(QKs, padding_mask_per_qb)   # broadcast across BQ
//   QKm  = Add(QKpm, triangle_mask_static) # broadcast across Hq
//   P = Softmax(QKm)
//   O = MatMul(P, V)
//
// Pure built-in QNN ops. No custom op, no q_block_idx scalar input. The
// per-qb prep_qb still binds Q, K, V (data) and now a tiny padding_mask
// vector, instead of Q, K, V + a 16 KB dense mask.
// ===========================================================================
static void runCausalPerQbStaticMask(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                     int D, int BK, int top_k, const std::string& tag) {
  fprintf(stderr, "[CASE per-qb static-mask] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d (causal)\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k);
  ASSERT_EQ(Sq, Skv);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

  // Mp_buf: tiny per-qb padding mask [1, 1, top_k_BK] (rank-3 → backfill
  // [1, 1, 1, top_k_BK]). Broadcast across Hq and BQ via leading 1s.
  std::array<Tensor, 2> Q_buf, K_buf, V_buf, Mp_buf, O_buf;
  for (int b = 0; b < 2; ++b) {
    Q_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
    K_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    Mp_buf[b] = Tensor::empty({1, 1, top_k_BK}, kFloat16, kQNN).alloc();
    O_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  }

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q_full.ptr<__fp16>();
  __fp16* kp = K_full.ptr<__fp16>();
  __fp16* vp = V_full.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  std::vector<int> sel;
  causalSelection(sel, Hq, num_q_blocks, top_k, rng);

  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  // Static triangle mask. Same for every qb, broadcast across Hq via leading 1.
  auto mask_static = Tensor::empty({1, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildStaticTriangleMask(mask_static.ptr<__fp16>(), BQ, BK, top_k);

  // Host reference. Unchanged: uses the per-qb dynamic mask for correctness.
  auto K_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  cpuGather(kp, K_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  auto mask_ref_t = Tensor::empty({num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMask(mask_ref_t.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k);
  std::vector<__fp16> ref(out_numel);
  naiveCausalBlockSparseFromArranged(qp, K_arr_ref.ptr<__fp16>(), V_arr_ref.ptr<__fp16>(),
                                     mask_ref_t.ptr<__fp16>(), ref.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks,
                                     scale);

  // Build 2 ping-pong graphs. Triangle mask is STATIC (bound once); padding
  // mask is APP_WRITE (tiny — 512 bytes per qb).
  auto buildGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    backend->addTensor(gname, "Q",  QNN_TENSOR_TYPE_APP_WRITE, Q_buf[b]);
    backend->addTensor(gname, "K",  QNN_TENSOR_TYPE_APP_WRITE, K_buf[b]);
    backend->addTensor(gname, "V",  QNN_TENSOR_TYPE_APP_WRITE, V_buf[b]);
    backend->addTensor(gname, "Mp", QNN_TENSOR_TYPE_APP_WRITE, Mp_buf[b]);
    backend->addTensor(gname, "O",  QNN_TENSOR_TYPE_APP_READ,  O_buf[b]);
    backend->addStaticTensor(gname, "scale", scale_t);
    backend->addStaticTensor(gname, "Mt",    mask_static);  // static triangle mask

    auto QK_t   = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t  = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKpm_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKm_t  = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t    = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    backend->addTensor(gname, "QK",   QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "QKs",  QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gname, "QKpm", QNN_TENSOR_TYPE_NATIVE, QKpm_t);
    backend->addTensor(gname, "QKm",  QNN_TENSOR_TYPE_NATIVE, QKm_t);
    backend->addTensor(gname, "P",    QNN_TENSOR_TYPE_NATIVE, P_t);
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk",   "MatMul",                {"Q", "K"},      {"QK"},   {}, mm, "qti.aisw");
    backend->graphAddNode(gname, "scale_qk",    "ElementWiseMultiply",   {"QK", "scale"}, {"QKs"},  {}, {}, "qti.aisw");
    backend->graphAddNode(gname, "add_padding", "ElementWiseAdd",        {"QKs", "Mp"},   {"QKpm"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gname, "add_triangle","ElementWiseAdd",        {"QKpm", "Mt"},  {"QKm"},  {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
        QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gname, "softmax",   "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gname, "matmul_av", "MatMul",  {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  const std::string g0 = "perqb_sm_b0_" + tag;
  const std::string g1 = "perqb_sm_b1_" + tag;
  buildGraph(g0, 0);
  buildGraph(g1, 1);

  auto prep_qb = [&](int qb, int b) {
    copyQSlice(qp, Q_buf[b].ptr<__fp16>(), qb, Hq, Sq, D, BQ);
    gatherOneQb(kp, K_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    gatherOneQb(vp, V_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    buildPaddingMaskOneQb(Mp_buf[b].ptr<__fp16>(), qb, BK, top_k);
  };

  // Warmup.
  prep_qb(0, 0);
  prep_qb(0, 1);
  std::vector<Tensor> ins0 = {Q_buf[0], K_buf[0], V_buf[0], Mp_buf[0]};
  std::vector<Tensor> outs0 = {O_buf[0]};
  std::vector<Tensor> ins1 = {Q_buf[1], K_buf[1], V_buf[1], Mp_buf[1]};
  std::vector<Tensor> outs1 = {O_buf[1]};
  backend->graphExecute(g0, ins0, outs0);
  backend->graphExecute(g1, ins1, outs1);

  // Sync dispatch (correctness + per-qb wall-clock).
  std::vector<__fp16> got(out_numel, (__fp16)0.0f);
  const auto sync_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < num_q_blocks; ++i) {
    int b = i % 2;
    prep_qb(i, b);
    std::vector<Tensor> ins = {Q_buf[b], K_buf[b], V_buf[b], Mp_buf[b]};
    std::vector<Tensor> outs = {O_buf[b]};
    backend->graphExecute(b == 0 ? g0 : g1, ins, outs);
    stitchO(got.data(), O_buf[b].ptr<__fp16>(), i, Hq, Sq, D, BQ);
  }
  const auto sync_t1 = std::chrono::steady_clock::now();
  const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

  size_t miss = 0;
  float max_abs_err = 0.f;
  for (size_t i = 0; i < out_numel; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > max_abs_err) max_abs_err = diff;
    if (diff > 5e-2f) ++miss;
  }

  fprintf(stderr,
          "[PERQB-STATIC %-10s] Sq=%-4d top_k=%-3d num_qb=%-3d\n"
          "  sync (prep→exec serially) : %7.2f ms (%.3f ms/qb) err=%.4f miss=%zu\n",
          tag.c_str(), Sq, top_k, num_q_blocks, sync_ms, sync_ms / num_q_blocks, max_abs_err, miss);
  EXPECT_EQ(miss, 0u) << "per-qb static-mask sparse causal NPU diverged from host reference";
}

// ===========================================================================
// Method 4: per-q-block fixed-shape rank-3 sparse — one compiled graph
// reused across variable input length.
//
// In a real prefill server we can't recompile the QNN graph for every prompt
// length. The per-q-block runner above already has a fixed dispatch shape
// per layer, but each Sq case there compiles its own graph at a different
// top_k_BK = top_k * BK. This runner pins top_k_BK at compile time once
// (top_k_FIXED, default 8 → 256) and sweeps Sq through {128, 256, 512,
// 1024, 2048}, dispatching num_qb = Sq/BQ chunks per Sq through the SAME
// pre-compiled graph.
//
// Per-qb buffers (Q [Hq, BQ, D], K/V [Hq, top_k_FIXED*BK, D], M [1, BQ,
// top_k_FIXED*BK], O [Hq, BQ, D]) are also Sq-independent, so they're
// allocated once and reused.
//
// At Sq < top_k_FIXED * BK the K/V tile is over-provisioned: only the first
// (i+1) slots hold real history, the remaining slots are masked-out
// padding. This is intentional — it's exactly the cost we'd pay in
// production, where the graph shape is fixed but the prompt is short.
// ===========================================================================
static void runFixedShapeCausalPerQbSweep(const std::shared_ptr<QNNBackend>& backend, int Hq, int Hkv, int D, int BK,
                                          int top_k_fixed, const std::vector<int>& Sq_list, const std::string& tag) {
  fprintf(stderr,
          "[CASE fixed-shape per-qb] %s: Hq=%d Hkv=%d D=%d BK=%d top_k_fixed=%d (%d Sq values; one compiled graph)\n",
          tag.c_str(), Hq, Hkv, D, BK, top_k_fixed, (int)Sq_list.size());
  ASSERT_EQ(Hq % Hkv, 0);
  const int BQ = BK;
  const int top_k_BK = top_k_fixed * BK;
  const int group = Hq / Hkv;

  // ---- Allocate Sq-independent per-qb buffers (double-buffered) ----
  std::array<Tensor, 2> Q_buf, K_buf, V_buf, M_buf, O_buf;
  for (int b = 0; b < 2; ++b) {
    Q_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
    K_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    M_buf[b] = Tensor::empty({1, BQ, top_k_BK}, kFloat16, kQNN).alloc();
    O_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  }

  // Static scale (compile-time constant — same for all Sq).
  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  // Use the largest Sq's scale (1/sqrt(D) is Sq-independent anyway).
  scale_t.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));

  // ---- Compile two ping-pong graphs ONCE. Shape: top_k_FIXED · BK columns. ----
  auto buildGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    backend->addTensor(gname, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_buf[b]);
    backend->addTensor(gname, "K", QNN_TENSOR_TYPE_APP_WRITE, K_buf[b]);
    backend->addTensor(gname, "V", QNN_TENSOR_TYPE_APP_WRITE, V_buf[b]);
    backend->addTensor(gname, "M", QNN_TENSOR_TYPE_APP_WRITE, M_buf[b]);
    backend->addTensor(gname, "O", QNN_TENSOR_TYPE_APP_READ, O_buf[b]);
    backend->addStaticTensor(gname, "scale", scale_t);

    auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKm_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    backend->addTensor(gname, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gname, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);
    backend->addTensor(gname, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gname, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gname, "add_mask", "ElementWiseAdd", {"QKs", "M"}, {"QKm"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
        QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gname, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gname, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  const std::string g0 = "fixedshape_b0_" + tag;
  const std::string g1 = "fixedshape_b1_" + tag;
  buildGraph(g0, 0);
  buildGraph(g1, 1);

  // Per-Sq table rows (printed at the end so the format isn't interleaved
  // with QNN's per-execute logs).
  struct Row {
    int Sq;
    int num_qb;
    double sync_ms;
    double pipe_ms;
    float max_abs_err;
    size_t miss;
  };
  std::vector<Row> rows;

  // ---- Sweep Sq through the requested list ----
  for (int Sq : Sq_list) {
    ASSERT_EQ(Sq % BK, 0);
    const int Skv = Sq;  // causal prefill
    const int num_q_blocks = Sq / BQ;

    // Per-Sq full-source allocations (these DO depend on Sq).
    auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
    auto K_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
    auto V_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

    std::mt19937 rng((uint32_t)(0xCA0501u + Sq));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    __fp16* qp = Q_full.ptr<__fp16>();
    __fp16* kp = K_full.ptr<__fp16>();
    __fp16* vp = V_full.ptr<__fp16>();
    for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
    std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
    std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
    for (auto& x : K_unique) x = (__fp16)dist(rng);
    for (auto& x : V_unique) x = (__fp16)dist(rng);
    for (int h = 0; h < Hq; ++h) {
      int hkv = h / group;
      std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
      std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    }

    const float scale = 1.0f / std::sqrt((float)D);
    const size_t out_numel = (size_t)Hq * Sq * D;

    // Causal selection. top_k_fixed may exceed num_k_blocks at small Sq
    // (e.g. Sq=128 → 4 k-blocks vs top_k=8); causalSelection's per-q-block
    // path already pads the unused slots with k-block 0, and the mask zeroes
    // them out via -1e4. So no special handling needed beyond skipping the
    // ASSERT_LE(top_k, num_k_blocks) that the other runners apply.
    std::vector<int> sel;
    causalSelection(sel, Hq, num_q_blocks, top_k_fixed, rng);

    // Host reference (uses the same fixed shape K_arr/V_arr).
    auto K_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
    auto V_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
    cpuGather(kp, K_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k_fixed);
    cpuGather(vp, V_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k_fixed);
    auto mask_ref_t = Tensor::empty({num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
    buildCausalMask(mask_ref_t.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k_fixed);
    std::vector<__fp16> ref(out_numel);
    naiveCausalBlockSparseFromArranged(qp, K_arr_ref.ptr<__fp16>(), V_arr_ref.ptr<__fp16>(),
                                       mask_ref_t.ptr<__fp16>(), ref.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks,
                                       scale);

    auto prep_qb = [&](int qb, int b) {
      copyQSlice(qp, Q_buf[b].ptr<__fp16>(), qb, Hq, Sq, D, BQ);
      gatherOneQb(kp, K_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k_fixed);
      gatherOneQb(vp, V_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k_fixed);
      buildCausalMaskOneQb(M_buf[b].ptr<__fp16>(), qb, BQ, BK, top_k_fixed);
    };

    // Warmup both graphs once on this Sq (graph-prepare is paid only once
    // per process, but the first execute on a new buffer-binding has a
    // first-touch cost we want out of the timed loop).
    prep_qb(0, 0);
    prep_qb(0, 1);
    std::vector<Tensor> ins0 = {Q_buf[0], K_buf[0], V_buf[0], M_buf[0]};
    std::vector<Tensor> outs0 = {O_buf[0]};
    std::vector<Tensor> ins1 = {Q_buf[1], K_buf[1], V_buf[1], M_buf[1]};
    std::vector<Tensor> outs1 = {O_buf[1]};
    backend->graphExecute(g0, ins0, outs0);
    backend->graphExecute(g1, ins1, outs1);

    // ============================== SYNC (correctness + per-Sq latency) ==============================
    std::vector<__fp16> got(out_numel, (__fp16)0.0f);
    const auto sync_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < num_q_blocks; ++i) {
      int b = i % 2;
      prep_qb(i, b);
      std::vector<Tensor> ins = {Q_buf[b], K_buf[b], V_buf[b], M_buf[b]};
      std::vector<Tensor> outs = {O_buf[b]};
      backend->graphExecute(b == 0 ? g0 : g1, ins, outs);
      stitchO(got.data(), O_buf[b].ptr<__fp16>(), i, Hq, Sq, D, BQ);
    }
    const auto sync_t1 = std::chrono::steady_clock::now();
    const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

    size_t miss = 0;
    float max_abs_err = 0.f;
    for (size_t i = 0; i < out_numel; ++i) {
      float diff = std::fabs((float)got[i] - (float)ref[i]);
      if (diff > max_abs_err) max_abs_err = diff;
      if (diff > 5e-2f) ++miss;
    }
    EXPECT_EQ(miss, 0u) << "fixed-shape per-qb sparse causal NPU diverged from host reference at Sq=" << Sq;

    // ============================== PIPELINED (timing only) ==============================
    std::mutex mu;
    std::condition_variable cv_req, cv_done;
    int req_qb = 0, req_b = 0;
    std::atomic<bool> has_req{false};
    std::atomic<bool> worker_done_pending{false};
    std::atomic<bool> stop_worker{false};
    std::thread worker([&]() {
      while (true) {
        std::unique_lock<std::mutex> lk(mu);
        cv_req.wait(lk, [&]() { return has_req.load() || stop_worker.load(); });
        if (stop_worker.load()) break;
        int qb_local = req_qb, b_local = req_b;
        has_req.store(false);
        lk.unlock();
        prep_qb(qb_local, b_local);
        {
          std::lock_guard<std::mutex> lk2(mu);
          worker_done_pending.store(true);
        }
        cv_done.notify_one();
      }
    });
    auto submit_prep = [&](int qb, int b) {
      {
        std::lock_guard<std::mutex> lk(mu);
        req_qb = qb;
        req_b = b;
        has_req.store(true);
      }
      cv_req.notify_one();
    };
    auto wait_prep = [&]() {
      std::unique_lock<std::mutex> lk(mu);
      cv_done.wait(lk, [&]() { return worker_done_pending.load(); });
      worker_done_pending.store(false);
    };

    prep_qb(0, 0);  // first qb sync on main thread
    const auto pipe_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < num_q_blocks; ++i) {
      int cur = i % 2;
      bool has_next = (i + 1 < num_q_blocks);
      if (has_next) submit_prep(i + 1, (i + 1) % 2);
      std::vector<Tensor> ins = {Q_buf[cur], K_buf[cur], V_buf[cur], M_buf[cur]};
      std::vector<Tensor> outs = {O_buf[cur]};
      backend->graphExecute(cur == 0 ? g0 : g1, ins, outs);
      if (has_next) wait_prep();
    }
    const auto pipe_t1 = std::chrono::steady_clock::now();
    const double pipe_ms = std::chrono::duration<double, std::milli>(pipe_t1 - pipe_t0).count();
    stop_worker.store(true);
    cv_req.notify_one();
    worker.join();

    rows.push_back({Sq, num_q_blocks, sync_ms, pipe_ms, max_abs_err, miss});
    fprintf(stderr,
            "[FIXEDSHAPE %-10s] Sq=%-4d num_qb=%-3d  sync=%7.2f ms  pipe=%7.2f ms  speedup=%.2fx  err=%.4f miss=%zu\n",
            tag.c_str(), Sq, num_q_blocks, sync_ms, pipe_ms, (pipe_ms > 0.0) ? sync_ms / pipe_ms : 0.0, max_abs_err,
            miss);
  }

  // ---- Final summary table ----
  fprintf(stderr,
          "\n[FIXEDSHAPE-SUMMARY %s] one compiled graph (top_k_fixed=%d, top_k_BK=%d)\n"
          "  Sq    | num_qb | sync ms | pipe ms | sync/qb (us) | pipe/qb (us)\n"
          "  ------+--------+---------+---------+--------------+-------------\n",
          tag.c_str(), top_k_fixed, top_k_BK);
  for (const auto& r : rows) {
    fprintf(stderr, "  %4d  | %5d  | %7.2f | %7.2f |   %8.1f   |   %8.1f\n", r.Sq, r.num_qb, r.sync_ms, r.pipe_ms,
            r.sync_ms * 1000.0 / r.num_qb, r.pipe_ms * 1000.0 / r.num_qb);
  }
  fprintf(stderr, "\n");
}

// ---------------------------------------------------------------------------
// Test cases. Each TEST_F should be run in a fresh process (see comment in
// BlockSparseAttentionTest.cpp about dual-graph state accumulation).
// ---------------------------------------------------------------------------
TEST_F(BlockSparseAttentionCausalTest, Sq128_TopK2) {
  // num_q_blocks=4. q-block 0 has 0 historical → slots 0..0 padded; q-block 1
  // exercises the "slot 0 = real historical, diagonal at slot 1" case.
  runCausalComparison(backend_, 128, 16, 128, 8, 128, 32, 2, /*with_dense_baseline=*/true, "sq128_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, Sq256_TopK2) {
  runCausalComparison(backend_, 256, 16, 256, 8, 128, 32, 2, true, "sq256_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, Sq512_TopK4) {
  runCausalComparison(backend_, 512, 16, 512, 8, 128, 32, 4, true, "sq512_tk4");
}
TEST_F(BlockSparseAttentionCausalTest, Sq1024_TopK8) {
  runCausalComparison(backend_, 1024, 16, 1024, 8, 128, 32, 8, true, "sq1024_tk8");
}
TEST_F(BlockSparseAttentionCausalTest, Sq2048_TopK8) {
  // Dense baseline skipped: at Sq=2048 the dense decomposed graph materialises
  // a 128 MB Hq×Sq×Skv fp16 score tensor (×4 native copies for QK/QKs/QKm/P)
  // and the QNN PD silently OOMs at graphFinalize, emitting zeros at execute
  // time. This is the O(Sq²) wall block-sparse is supposed to avoid — the
  // sparse path itself is what we're validating here.
  //
  // top_k=8 (1/8 sparsity) instead of top_k=16 (1/4 sparsity) to keep K_arr
  // under the PD ceiling. The doc's measured K_arranged at Sq=2048, top_k=16
  // is 128 MB and labelled "OOM-borderline" — at top_k=16 the sparse execute
  // succeeds ~1/3 of the time on this device; top_k=8 (64 MB K_arr) runs
  // reliably. The mask wiring is the same in both cases.
  runCausalComparison(backend_, 2048, 16, 2048, 8, 128, 32, 8, /*with_dense_baseline=*/false, "sq2048_tk8");
}

// ----- Big-batch rank-3 (Method 2) -----
TEST_F(BlockSparseAttentionCausalTest, BigBatch_Sq128_TopK2) {
  runCausalBigBatch(backend_, 128, 16, 128, 8, 128, 32, 2, "sq128_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, BigBatch_Sq256_TopK2) {
  runCausalBigBatch(backend_, 256, 16, 256, 8, 128, 32, 2, "sq256_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, BigBatch_Sq512_TopK4) {
  runCausalBigBatch(backend_, 512, 16, 512, 8, 128, 32, 4, "sq512_tk4");
}
TEST_F(BlockSparseAttentionCausalTest, BigBatch_Sq1024_TopK8) {
  runCausalBigBatch(backend_, 1024, 16, 1024, 8, 128, 32, 8, "sq1024_tk8");
}
TEST_F(BlockSparseAttentionCausalTest, BigBatch_Sq2048_TopK8) {
  runCausalBigBatch(backend_, 2048, 16, 2048, 8, 128, 32, 8, "sq2048_tk8");
}

// ----- Per-qb pipelined rank-3 (Method 3) -----
TEST_F(BlockSparseAttentionCausalTest, PerQb_Sq128_TopK2) {
  runCausalPerQbPipelined(backend_, 128, 16, 128, 8, 128, 32, 2, "sq128_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, PerQb_Sq256_TopK2) {
  runCausalPerQbPipelined(backend_, 256, 16, 256, 8, 128, 32, 2, "sq256_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, PerQb_Sq512_TopK4) {
  runCausalPerQbPipelined(backend_, 512, 16, 512, 8, 128, 32, 4, "sq512_tk4");
}
TEST_F(BlockSparseAttentionCausalTest, PerQb_Sq1024_TopK8) {
  runCausalPerQbPipelined(backend_, 1024, 16, 1024, 8, 128, 32, 8, "sq1024_tk8");
}
TEST_F(BlockSparseAttentionCausalTest, PerQb_Sq2048_TopK8) {
  runCausalPerQbPipelined(backend_, 2048, 16, 2048, 8, 128, 32, 8, "sq2048_tk8");
}

// ----- Per-qb static-mask (Method 3b) -----
// Same dispatch shape as PerQb_Sq*, but no per-qb mask binding —
// CPU-side gather masks padding-slot K/V, and a tiny static [1, BQ, top_k*BK]
// triangle mask covers the diagonal-slot triangle. Pure built-in QNN ops.
TEST_F(BlockSparseAttentionCausalTest, PerQbStaticMask_Sq128_TopK2) {
  runCausalPerQbStaticMask(backend_, 128, 16, 128, 8, 128, 32, 2, "sm_sq128_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, PerQbStaticMask_Sq256_TopK2) {
  runCausalPerQbStaticMask(backend_, 256, 16, 256, 8, 128, 32, 2, "sm_sq256_tk2");
}
TEST_F(BlockSparseAttentionCausalTest, PerQbStaticMask_Sq512_TopK4) {
  runCausalPerQbStaticMask(backend_, 512, 16, 512, 8, 128, 32, 4, "sm_sq512_tk4");
}
TEST_F(BlockSparseAttentionCausalTest, PerQbStaticMask_Sq1024_TopK8) {
  runCausalPerQbStaticMask(backend_, 1024, 16, 1024, 8, 128, 32, 8, "sm_sq1024_tk8");
}
TEST_F(BlockSparseAttentionCausalTest, PerQbStaticMask_Sq2048_TopK8) {
  runCausalPerQbStaticMask(backend_, 2048, 16, 2048, 8, 128, 32, 8, "sm_sq2048_tk8");
}

// ----- Fixed-shape per-qb sweep (Method 4) -----
// One compiled graph (top_k=8, top_k_BK=256) reused across all Sq. Models the
// production case where the QNN graph is compiled once at server start and
// the same graph processes prompts of varying lengths in chunks of BQ=32.
TEST_F(BlockSparseAttentionCausalTest, FixedShape_TopK8_Sweep) {
  runFixedShapeCausalPerQbSweep(backend_, /*Hq=*/16, /*Hkv=*/8, /*D=*/128, /*BK=*/32,
                                /*top_k_fixed=*/8, /*Sq_list=*/{128, 256, 512, 1024, 2048}, "tk8");
}

// ===========================================================================
// Method 5: per-q-block dispatch using the fused SoftmaxBlockSparseCausal
// custom op (replaces Mul + Add(causal mask) + Softmax with a single op).
//
// End-to-end integration test: validates that the custom op slots cleanly
// into a per-qb causal block-sparse attention pipeline and produces output
// matching the host-reference dense-then-mask-then-softmax computation.
//
// Per-qb graph (vs Method 3's 5-op graph):
//   matmul_qk: MatMul(Q, K, transpose_in1=true) -> QK
//   fused:     SoftmaxBlockSparseCausal(QK, q_block_idx, scale, bk) -> P
//   matmul_av: MatMul(P, V) -> O
//
// q_block_idx is now an APP_WRITE INPUT (uint32 scalar tensor), so a single
// compiled graph handles every q-block — host writes the new value into the
// q_block_idx tensor before each dispatch. We use the same 2-graph ping-pong
// pattern as runCausalPerQbPipelined (Method 3) for a fair perf comparison.
//
// No mask buffer needed (custom op handles the structural mask internally),
// so the per-qb working set drops by ~16 KB per buffer set vs Method 3.
// ===========================================================================

// Separate fixture: needs LLaMAPackage registered as a custom op package.
// The base BlockSparseAttentionCausalTest fixture's setUp doesn't load it,
// and registration must happen BEFORE QNN context creation.
class BlockSparseAttentionCausalFusedTest : public testing::Test {
 protected:
  static bool readableFile(const std::string& path) { return access(path.c_str(), R_OK) == 0; }
  static std::string findReadableFile(const std::vector<std::string>& candidates) {
    for (const auto& c : candidates)
      if (readableFile(c)) return c;
    return {};
  }
  static void SetUpTestSuite() {
    unbufferOutput();
    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string p = existing && existing[0] != '\0' ? (std::string(".;/data/local/tmp;") + existing) : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", p.c_str(), /*overwrite=*/1);
    }
    ASSERT_TRUE(isQnnAvailable());

    auto& ctx = Context::instance();
    auto host_backend = cpu::createCPUBackend();
    ctx.registerBackend(host_backend);
    ctx.memoryManager()->registerAllocator(kCPU, host_backend->allocator(), MemoryManagerOptions());

    backend_ = std::make_shared<QNNBackend>();

    // Register LLaMAPackage on both CPU (prepare-side) and HTP (runtime).
    const std::string cpu_pkg = findReadableFile({"./libQnnLLaMAPackage_CPU.so", "/data/local/tmp/libQnnLLaMAPackage_CPU.so"});
    ASSERT_FALSE(cpu_pkg.empty()) << "Missing ARM-side libQnnLLaMAPackage_CPU.so on device";
    const std::string htp_pkg = findReadableFile(
        {"./libQnnLLaMAPackage.so", "/data/local/tmp/libQnnLLaMAPackage.so", "./libQnnLLaMAPackage_HTP.so", "/data/local/tmp/libQnnLLaMAPackage_HTP.so"});
    ASSERT_FALSE(htp_pkg.empty()) << "Missing Hexagon HTP libQnnLLaMAPackage.so on device";
    auto reg = [&](const std::string& path, const char* target) {
      auto ret = backend_->qnnInterface().backendRegisterOpPackage(backend_->backendHandle(), path.c_str(),
                                                                   "LLaMAPackageInterfaceProvider", target);
      ASSERT_EQ(QNN_BACKEND_NO_ERROR, (int)(ret & 0xFFFF))
          << "backendRegisterOpPackage(" << target << ") failed err=" << (int)(ret & 0xFFFF);
    };
    reg(cpu_pkg, "CPU");
    reg(htp_pkg, "HTP");

    ASSERT_TRUE(backend_->createContext());
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(
        kQNN, backend_->allocator(), {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
    ctx.dispatcherManager()->registerDispatcher(
        createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), QNNDispatcherOptions()));
  }
  static void TearDownTestSuite() { backend_.reset(); }
  static std::shared_ptr<QNNBackend> backend_;
};
std::shared_ptr<QNNBackend> BlockSparseAttentionCausalFusedTest::backend_ = nullptr;

static void runCausalPerQbFused(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                int D, int BK, int top_k, const std::string& tag) {
  fprintf(stderr, "[CASE per-qb fused] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d (causal)\n", tag.c_str(),
          Sq, Hq, Skv, Hkv, D, BK, top_k);
  ASSERT_EQ(Sq, Skv);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  // The fused op (SoftmaxBlockSparseCausal) requires bk == 32 and top_k must
  // be even (so W = top_k*bk is a multiple of 64 — pair-processing assumption
  // in the HVX path). Both hold for the standard test sizes (top_k 2, 4, 8).
  ASSERT_EQ(BK, 32) << "fused op requires BK == 32";
  ASSERT_EQ(top_k % 2, 0) << "fused op requires top_k even";

  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

  // Double-buffered per-qb dispatch (matches runCausalPerQbPipelined): two
  // sets of (Q, K, V, q_block_idx, O) buffers + two graphs (g0, g1) so a
  // future pipelined harness can overlap CPU prep of qb i+1 with NPU exec
  // of qb i. Sync mode used here for correctness + apples-to-apples timing
  // vs the existing Method 3 sync numbers.
  std::array<Tensor, 2> Q_buf, K_buf, V_buf, QBlockIdx_buf, O_buf;
  for (int b = 0; b < 2; ++b) {
    Q_buf[b]         = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
    K_buf[b]         = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_buf[b]         = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    QBlockIdx_buf[b] = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
    O_buf[b]         = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  }

  // Same seed/data layout as runCausalPerQbPipelined so the host reference
  // is bit-identical to that test's reference.
  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q_full.ptr<__fp16>();
  __fp16* kp = K_full.ptr<__fp16>();
  __fp16* vp = V_full.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  std::vector<int> sel;
  causalSelection(sel, Hq, num_q_blocks, top_k, rng);

  // ----- Host reference (matches runCausalPerQbPipelined) -----
  auto K_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr_ref = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  cpuGather(kp, K_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr_ref.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  auto mask_ref_t = Tensor::empty({num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMask(mask_ref_t.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k);
  std::vector<__fp16> ref(out_numel);
  naiveCausalBlockSparseFromArranged(qp, K_arr_ref.ptr<__fp16>(), V_arr_ref.ptr<__fp16>(),
                                     mask_ref_t.ptr<__fp16>(), ref.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks,
                                     scale);

  // ----- Build 2 ping-pong graphs (one per buffer set) -----
  // q_block_idx is now an APP_WRITE input — value is written by the host
  // before each dispatch, so a single graph handles every q-block.
  auto buildGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    backend->addTensor(gname, "Q",           QNN_TENSOR_TYPE_APP_WRITE, Q_buf[b]);
    backend->addTensor(gname, "K",           QNN_TENSOR_TYPE_APP_WRITE, K_buf[b]);
    backend->addTensor(gname, "V",           QNN_TENSOR_TYPE_APP_WRITE, V_buf[b]);
    backend->addTensor(gname, "q_block_idx", QNN_TENSOR_TYPE_APP_WRITE, QBlockIdx_buf[b]);
    backend->addTensor(gname, "O",           QNN_TENSOR_TYPE_APP_READ,  O_buf[b]);

    auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t  = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    backend->addTensor(gname, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "P",  QNN_TENSOR_TYPE_NATIVE, P_t);

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> fp = {
        QNNParamScalarWrapper::create<float>("softmax_scale", scale),
        QNNParamScalarWrapper::create<uint32_t>("bk", (uint32_t)BK),
    };
    backend->graphAddNode(gname, "fused_softmax", "SoftmaxBlockSparseCausal",
                          {"QK", "q_block_idx"}, {"P"}, {}, fp, "LLaMAPackage");

    backend->graphAddNode(gname, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  const std::string g0 = "perqb_fused_b0_" + tag;
  const std::string g1 = "perqb_fused_b1_" + tag;
  const auto compile_t0 = std::chrono::steady_clock::now();
  buildGraph(g0, 0);
  buildGraph(g1, 1);
  const auto compile_t1 = std::chrono::steady_clock::now();
  const double compile_ms = std::chrono::duration<double, std::milli>(compile_t1 - compile_t0).count();

  // CPU prep: copy Q[:, qb*BQ:(qb+1)*BQ, :] into Q_buf[b], gather K/V per qb,
  // write q_block_idx into the APP_WRITE scalar tensor for buffer set b.
  // No mask build — the fused op handles the structural mask internally.
  auto prep_qb = [&](int qb, int b) {
    copyQSlice(qp, Q_buf[b].ptr<__fp16>(), qb, Hq, Sq, D, BQ);
    gatherOneQb(kp, K_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    gatherOneQb(vp, V_buf[b].ptr<__fp16>(), sel.data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    QBlockIdx_buf[b].ptr<uint32_t>()[0] = (uint32_t)qb;
  };

  // Warmup both graphs once (so JIT cost is paid before timing).
  prep_qb(0, 0);
  prep_qb(0, 1);
  std::vector<Tensor> ins0 = {Q_buf[0], K_buf[0], V_buf[0], QBlockIdx_buf[0]};
  std::vector<Tensor> outs0 = {O_buf[0]};
  std::vector<Tensor> ins1 = {Q_buf[1], K_buf[1], V_buf[1], QBlockIdx_buf[1]};
  std::vector<Tensor> outs1 = {O_buf[1]};
  backend->graphExecute(g0, ins0, outs0);
  backend->graphExecute(g1, ins1, outs1);

  // ============================== SYNC dispatch ==============================
  std::vector<__fp16> got(out_numel, (__fp16)0.0f);
  const auto sync_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < num_q_blocks; ++i) {
    int b = i % 2;
    prep_qb(i, b);
    std::vector<Tensor> ins = {Q_buf[b], K_buf[b], V_buf[b], QBlockIdx_buf[b]};
    std::vector<Tensor> outs = {O_buf[b]};
    backend->graphExecute(b == 0 ? g0 : g1, ins, outs);
    stitchO(got.data(), O_buf[b].ptr<__fp16>(), i, Hq, Sq, D, BQ);
  }
  const auto sync_t1 = std::chrono::steady_clock::now();
  const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

  size_t miss = 0;
  float max_abs_err = 0.f;
  for (size_t i = 0; i < out_numel; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > max_abs_err) max_abs_err = diff;
    if (diff > 5e-2f) ++miss;
  }

  fprintf(stderr,
          "[PERQB-FUSED %-10s] Sq=%-4d top_k=%-3d num_qb=%-3d (2 ping-pong graphs)\n"
          "  graph compile (one-shot) : %7.2f ms total (%.2f ms/graph)\n"
          "  sync dispatch (per layer): %7.2f ms (%.3f ms/qb)\n"
          "  err_vs_host=%.4f miss=%zu\n",
          tag.c_str(), Sq, top_k, num_q_blocks, compile_ms, compile_ms / 2.0, sync_ms,
          sync_ms / num_q_blocks, max_abs_err, miss);
  EXPECT_EQ(miss, 0u) << "per-qb fused-op causal NPU diverged from host reference";
}

// ===========================================================================
// Big-batch rank-3 causal sparse with the fused
// SoftmaxBlockSparseCausalBigBatch op replacing Mul + Add(mask) + Softmax.
//
// Same graph skeleton as runCausalBigBatch:
//   Q [Hq, Sq, D] → Reshape → Q3d [big_batch, BQ, D]
//   QK = MatMul(Q3d, K_arr3d, transpose_in1) → [big_batch, BQ, top_k*BK]
//   P  = SoftmaxBlockSparseCausalBigBatch(QK, slice_base, scale, bk, num_qb)
//                      ← replaces Mul + Add(mask_8MB) + Softmax in decomp
//   O3d = MatMul(P, V_arr3d) → [big_batch, BQ, D]
//   O   = Reshape(O3d) → [Hq, Sq, D]
//
// slice_base is a uint32 scalar STATIC tensor = 0 from the user side. The
// AUTOSPLIT optimization rule REPLACES it per-slice with a synthesized
// Const(SPLIT_START), so each split-op gets its own global row offset.
// Inside the kernel: qb_idx_for_row(h_local) = (slice_base + h_local) % num_qb.
//
// num_qb is a regular scalar param. The kernel uses it for the modulo;
// chunk=16 in the AUTOSPLIT directive ensures the modulo is correct for
// every num_qb in {4,8,16,32,64} (each either divides 16 or is divisible
// by 16).
//
// Pure NPU win: kills the materialised [big_batch, BQ, top_k*BK] fp16 mask
// (8 MB at Sq=1024) and folds three QNN ops (Mul/Add/Softmax) into one.
// ===========================================================================
static void runCausalBigBatchFused(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                   int D, int BK, int top_k, const std::string& tag) {
  fprintf(stderr, "[CASE big-batch fused] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d (causal)\n", tag.c_str(),
          Sq, Hq, Skv, Hkv, D, BK, top_k);
  ASSERT_EQ(Sq, Skv);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  ASSERT_EQ(BK, 32) << "fused op requires BK == 32";
  ASSERT_EQ(top_k % 2, 0) << "fused op requires top_k even";

  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;
  const int big_batch = Hq * num_q_blocks;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* vp = V.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  std::vector<int> sel;
  causalSelection(sel, Hq, num_q_blocks, top_k, rng);
  cpuGather(kp, K_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);

  // slice_base = 0 STATIC. Const-zero scalar; AUTOSPLIT replaces per-slice
  // with gen_ConstScalar_i32(SPLIT_START("I")). For the un-split fallback
  // (rule doesn't fire), slice_base=0 + h_local ranging over full big_batch
  // gives the correct global qb index via (0 + h) % num_qb.
  auto slice_base_t = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
  *slice_base_t.ptr<uint32_t>() = 0;

  // Host reference uses the per-qb mask + arranged K/V (byte-identical to the
  // big-batch layout — same K_arr storage).
  auto mask_ref = Tensor::empty({num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN).alloc();
  buildCausalMask(mask_ref.ptr<__fp16>(), num_q_blocks, BQ, BK, top_k);
  std::vector<__fp16> ref(out_numel);
  naiveCausalBlockSparseFromArranged(qp, K_arr.ptr<__fp16>(), V_arr.ptr<__fp16>(), mask_ref.ptr<__fp16>(), ref.data(),
                                     Hq, Sq, D, BQ, top_k_BK, num_q_blocks, scale);

  const std::string gn = "bb_fused_" + tag;
  GraphTime r = runOneGraph(backend, gn, {Q, K_arr, V_arr}, O, ref, out_numel, /*tol=*/5e-2f,
                            [&](const std::string& g) {
    backend->addTensor(g, "Q",     QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
    backend->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
    backend->addTensor(g, "O",     QNN_TENSOR_TYPE_APP_READ,  O);
    // slice_base is STATIC so the optimizer sees Const(0) and the AUTOSPLIT
    // rule can replace it per-slice.
    backend->addStaticTensor(g, "slice_base", slice_base_t);

    auto Q3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
    auto QK_t  = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t   = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto O3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
    backend->addTensor(g, "Q3d", QNN_TENSOR_TYPE_NATIVE, Q3d_t);
    backend->addTensor(g, "QK",  QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "P",   QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(g, "O3d", QNN_TENSOR_TYPE_NATIVE, O3d_t);

    backend->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q3d"}, {}, {}, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
        QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q3d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> fp = {
        QNNParamScalarWrapper::create<float>("softmax_scale", scale),
        QNNParamScalarWrapper::create<uint32_t>("bk",     (uint32_t)BK),
        QNNParamScalarWrapper::create<uint32_t>("num_qb", (uint32_t)num_q_blocks),
    };
    backend->graphAddNode(g, "fused_softmax", "SoftmaxBlockSparseCausalBigBatch",
                          {"QK", "slice_base"}, {"P"}, {}, fp, "LLaMAPackage");

    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O3d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "reshape_o", "Reshape", {"O3d"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[BB-FUSED %-14s] Sq=%-4d top_k=%-3d big_batch=%-4d\n"
          "  NPU big-batch r3 fused causal : %7.2f ms  err_vs_ref=%.4f miss=%zu\n",
          tag.c_str(), Sq, top_k, big_batch, r.avg_ms, r.max_abs_err, r.mismatch);
  EXPECT_EQ(r.mismatch, 0u) << "big-batch r3 fused causal NPU diverged from host reference";
}

// Test cases — match the existing PerQb_Sq*_TopK* set so per-qb fused vs
// per-qb pipelined-decomposed numbers can be compared at the same shapes.
TEST_F(BlockSparseAttentionCausalFusedTest, FusedPerQb_Sq128_TopK2) {
  runCausalPerQbFused(backend_, 128, 16, 128, 8, 128, 32, 2, "fused_sq128_tk2");
}
TEST_F(BlockSparseAttentionCausalFusedTest, FusedPerQb_Sq256_TopK2) {
  runCausalPerQbFused(backend_, 256, 16, 256, 8, 128, 32, 2, "fused_sq256_tk2");
}
TEST_F(BlockSparseAttentionCausalFusedTest, FusedPerQb_Sq512_TopK4) {
  runCausalPerQbFused(backend_, 512, 16, 512, 8, 128, 32, 4, "fused_sq512_tk4");
}
TEST_F(BlockSparseAttentionCausalFusedTest, FusedPerQb_Sq1024_TopK8) {
  runCausalPerQbFused(backend_, 1024, 16, 1024, 8, 128, 32, 8, "fused_sq1024_tk8");
}
TEST_F(BlockSparseAttentionCausalFusedTest, FusedPerQb_Sq2048_TopK8) {
  runCausalPerQbFused(backend_, 2048, 16, 2048, 8, 128, 32, 8, "fused_sq2048_tk8");
}

// Big-batch rank-3 with the fused op replacing Mul+Add+Softmax. Compare
// against the matching BigBatch_Sq*_TopK* (decomposed) numbers in the
// non-fused fixture.
TEST_F(BlockSparseAttentionCausalFusedTest, BigBatchFused_Sq128_TopK2) {
  runCausalBigBatchFused(backend_, 128, 16, 128, 8, 128, 32, 2, "bbfused_sq128_tk2");
}
TEST_F(BlockSparseAttentionCausalFusedTest, BigBatchFused_Sq256_TopK2) {
  runCausalBigBatchFused(backend_, 256, 16, 256, 8, 128, 32, 2, "bbfused_sq256_tk2");
}
TEST_F(BlockSparseAttentionCausalFusedTest, BigBatchFused_Sq512_TopK4) {
  runCausalBigBatchFused(backend_, 512, 16, 512, 8, 128, 32, 4, "bbfused_sq512_tk4");
}
TEST_F(BlockSparseAttentionCausalFusedTest, BigBatchFused_Sq1024_TopK8) {
  runCausalBigBatchFused(backend_, 1024, 16, 1024, 8, 128, 32, 8, "bbfused_sq1024_tk8");
}
TEST_F(BlockSparseAttentionCausalFusedTest, BigBatchFused_Sq2048_TopK8) {
  runCausalBigBatchFused(backend_, 2048, 16, 2048, 8, 128, 32, 8, "bbfused_sq2048_tk8");
}

// Build the same per-qb fused graph (Sq=1024, top_k=8) as a single ping-pong
// pair, warmup-execute it, save context binary + raw inputs + input list so
// it can be re-run under qnn-net-run --profiling_option optrace for a full
// per-HW-unit (HMX/HVX) breakdown of the 3-op pipeline.
TEST_F(BlockSparseAttentionCausalFusedTest, DumpContext_FusedPerQb_Sq1024_TopK8) {
  const int Sq = 1024, Hq = 16, Hkv = 8, D = 128, BK = 32, top_k = 8;
  const int BQ = BK;
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto V_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  auto Q_buf  = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  auto K_buf  = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_buf  = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto QBlockIdx_buf = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
  auto O_buf  = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q_full.ptr<__fp16>();
  __fp16* kp = K_full.ptr<__fp16>();
  __fp16* vp = V_full.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Sq * D), V_unique((size_t)Hkv * Sq * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(kp + (size_t)h * Sq * D, K_unique.data() + (size_t)hkv * Sq * D, (size_t)Sq * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Sq * D, V_unique.data() + (size_t)hkv * Sq * D, (size_t)Sq * D * sizeof(__fp16));
  }
  const float scale = 1.0f / std::sqrt((float)D);

  std::vector<int> sel;
  causalSelection(sel, Hq, Sq / BQ, top_k, rng);

  // Fill Q_buf/K_buf/V_buf for q_block_idx=Sq/BQ-1 (last qb — exercises the
  // "all real history" mid-prefill path, the most common case).
  const int qb = Sq / BQ - 1;
  copyQSlice(qp, Q_buf.ptr<__fp16>(), qb, Hq, Sq, D, BQ);
  gatherOneQb(kp, K_buf.ptr<__fp16>(), sel.data(), qb, Hq, Sq, D, BK, Sq / BQ, top_k);
  gatherOneQb(vp, V_buf.ptr<__fp16>(), sel.data(), qb, Hq, Sq, D, BK, Sq / BQ, top_k);
  QBlockIdx_buf.ptr<uint32_t>()[0] = (uint32_t)qb;

  // Build single graph (no ping-pong needed for context dump).
  const std::string g = "perqb_fused_dump_sq1024";
  ASSERT_NE(backend_->createQnnGraph(g), nullptr);
  backend_->addTensor(g, "Q",           QNN_TENSOR_TYPE_APP_WRITE, Q_buf);
  backend_->addTensor(g, "K",           QNN_TENSOR_TYPE_APP_WRITE, K_buf);
  backend_->addTensor(g, "V",           QNN_TENSOR_TYPE_APP_WRITE, V_buf);
  backend_->addTensor(g, "q_block_idx", QNN_TENSOR_TYPE_APP_WRITE, QBlockIdx_buf);
  backend_->addTensor(g, "O",           QNN_TENSOR_TYPE_APP_READ,  O_buf);
  auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_t  = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  backend_->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend_->addTensor(g, "P",  QNN_TENSOR_TYPE_NATIVE, P_t);
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
      QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend_->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> fp = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<uint32_t>("bk", (uint32_t)BK),
  };
  backend_->graphAddNode(g, "fused_softmax", "SoftmaxBlockSparseCausal",
                         {"QK", "q_block_idx"}, {"P"}, {}, fp, "LLaMAPackage");
  backend_->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend_->graphFinalize(g));

  // Warmup execute (pays JIT cost so the saved context is "warmed").
  std::vector<Tensor> ins = {Q_buf, K_buf, V_buf, QBlockIdx_buf};
  std::vector<Tensor> outs = {O_buf};
  backend_->graphExecute(g, ins, outs);

  // Save context + inputs.
  backend_->saveContext("perqb_ctx.bin");
  auto dump = [&](const std::string& path, const void* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)p, n);
  };
  dump("perqb_Q.raw", Q_buf.ptr<__fp16>(), (size_t)Hq * BQ * D * sizeof(__fp16));
  dump("perqb_K.raw", K_buf.ptr<__fp16>(), (size_t)Hq * top_k_BK * D * sizeof(__fp16));
  dump("perqb_V.raw", V_buf.ptr<__fp16>(), (size_t)Hq * top_k_BK * D * sizeof(__fp16));
  uint32_t qb_val = (uint32_t)qb;
  dump("perqb_qbidx.raw", &qb_val, sizeof(qb_val));
  std::ofstream il("perqb_inputs.txt");
  il << "Q:=perqb_Q.raw K:=perqb_K.raw V:=perqb_V.raw q_block_idx:=perqb_qbidx.raw\n";
  fprintf(stderr,
          "[DUMP] context=perqb_ctx.bin inputs=perqb_inputs.txt schematic=%s_schematic.bin\n",
          g.c_str());
}

// Same idea as DumpContext_FusedPerQb but for the big-batch fused op.
// Used to inspect the post-EARLY-pass schematic and see whether AUTOSPLIT
// actually produced split-ops (look for `splithist` in the dumped python
// pprint) — the key question for "why is big-batch fused single-threaded".
TEST_F(BlockSparseAttentionCausalFusedTest, DumpContext_BigBatchFused_Sq1024_TopK8) {
  const int Sq = 1024, Hq = 16, Hkv = 8, D = 128, BK = 32, top_k = 8;
  const int BQ = BK;
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;
  const int num_qb = Sq / BQ;
  const int big_batch = Hq * num_qb;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>();
  __fp16* karp = K_arr.ptr<__fp16>();
  __fp16* varp = V_arr.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < (size_t)big_batch * top_k_BK * D; ++i) karp[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < (size_t)big_batch * top_k_BK * D; ++i) varp[i] = (__fp16)dist(rng);

  const float scale = 1.0f / std::sqrt((float)D);

  auto slice_base_t = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
  *slice_base_t.ptr<uint32_t>() = 0;

  const std::string g = "bbfused_dump_sq1024";
  ASSERT_NE(backend_->createQnnGraph(g), nullptr);
  backend_->addTensor(g, "Q",     QNN_TENSOR_TYPE_APP_WRITE, Q);
  backend_->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
  backend_->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
  backend_->addTensor(g, "O",     QNN_TENSOR_TYPE_APP_READ,  O);
  backend_->addStaticTensor(g, "slice_base", slice_base_t);

  auto Q3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
  auto QK_t  = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_t   = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto O3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
  backend_->addTensor(g, "Q3d", QNN_TENSOR_TYPE_NATIVE, Q3d_t);
  backend_->addTensor(g, "QK",  QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend_->addTensor(g, "P",   QNN_TENSOR_TYPE_NATIVE, P_t);
  backend_->addTensor(g, "O3d", QNN_TENSOR_TYPE_NATIVE, O3d_t);

  backend_->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q3d"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {
      QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend_->graphAddNode(g, "matmul_qk", "MatMul", {"Q3d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> fp = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<uint32_t>("bk",     (uint32_t)BK),
      QNNParamScalarWrapper::create<uint32_t>("num_qb", (uint32_t)num_qb),
  };
  backend_->graphAddNode(g, "fused_softmax", "SoftmaxBlockSparseCausalBigBatch",
                         {"QK", "slice_base"}, {"P"}, {}, fp, "LLaMAPackage");

  backend_->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O3d"}, {}, {}, "qti.aisw");
  backend_->graphAddNode(g, "reshape_o", "Reshape", {"O3d"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend_->graphFinalize(g));

  std::vector<Tensor> ins = {Q, K_arr, V_arr};
  std::vector<Tensor> outs = {O};
  backend_->graphExecute(g, ins, outs);

  backend_->saveContext("bbfused_ctx.bin");
  fprintf(stderr,
          "[DUMP] context=bbfused_ctx.bin schematic=%s_schematic.bin\n",
          g.c_str());
}
