// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Path B end-to-end: block-sparse attention with CPU-side selection + gather,
// NPU-side compute via batched MatMul + Softmax + MatMul on the per-q-block
// rearranged tile.
//
// Layout:
//   Q [Hq, Sq, D]              ← reshape→ [Hq, num_q_blocks, BQ, D]
//   K_arranged [Hq, num_q_blocks, top_k*BK, D]   ← CPU prepares this from K
//   V_arranged [Hq, num_q_blocks, top_k*BK, D]
//
// Graph (6 ops total — much smaller than the unrolled 64-block alternative):
//   Q4d  = Reshape(Q, [Hq, num_q_blocks, BQ, D])
//   QK   = MatMul(Q4d, K_arranged, transpose_in1=true)  → [Hq, num_q_blocks, BQ, top_k*BK]
//   QKs  = ElementWiseMultiply(QK, scale)                 batched 4-D MatMul
//   P    = Softmax(QKs, axis=3)
//   O4d  = MatMul(P, V_arranged)                          → [Hq, num_q_blocks, BQ, D]
//   O    = Reshape(O4d, [Hq, Sq, D])
//
// HMX reach: each batch element of the matmul is BQ × D × (top_k·BK) which
// matches HMX's 32-aligned tile shape exactly (BQ=BK=32, D=128 → tiles of
// (32, 128, 512)). Batch dim = Hq · num_q_blocks → all batches scheduled
// together; HMX should keep utilisation high.
//
// CPU side (timed separately):
//   - random per-(Hq, q_block) selection (stand-in for a real heuristic)
//   - block gather K[selected_blocks] → K_arranged, V → V_arranged

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <random>
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

static void unbufferOutput() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
}
static int timingRunsFromEnv() {
  const char* v = std::getenv("MLLM_QNN_FA_TIMING_RUNS");
  if (!v || v[0] == '\0') return 1;
  char* end = nullptr;
  long r = std::strtol(v, &end, 10);
  return (end != v && r > 0) ? (int)r : 1;
}

class BlockSparseAttentionTest : public testing::Test {
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
std::shared_ptr<QNNBackend> BlockSparseAttentionTest::backend_ = nullptr;

// XAttention-style block selection (Algorithm 1 from the paper).
//
// Uses the antidiagonal-stride trick: sum-pool Q within each block and K
// over the full sequence with stride S, compute attention on the pooled
// representation, then sum per original K block to score them. Top-k
// scoring K blocks per (head, q_block) → selection.
//
// The scaling factor sqrt(d_h · S) accounts for the variance growth from
// summing S rows pre-softmax (variance scales linearly with S).
//
// Cost per layer (Hq, num_q_blocks, top_k):
//   K pool   :  Hq · Skv · D                    MACs (one-shot per layer)
//   Q pool   :  Hq · Sq · D                     MACs
//   Matmul   :  Hq · num_q_blocks · (B/S) · (Skv/S) · D
//   Softmax+aggr+top-k: small
// At B=32, S=8 (B/S=4 sub-pools per block), this is ~500M MACs/layer at
// Sq=2048 — compute-heavy on CPU but still tractable when pipelined.
static void blockSelectionXAttn(const __fp16* Q, const __fp16* K, int Hq, int Sq, int Skv, int D, int B, int S,
                                int top_k, std::vector<int>& selections) {
  const int num_q_blocks = Sq / B;
  const int num_k_blocks = Skv / B;
  const int B_per_S = B / S;
  const int Skv_per_S = Skv / S;
  const float scale = 1.0f / std::sqrt((float)(D * S));

  selections.resize((size_t)Hq * num_q_blocks * top_k);

  std::vector<float> K_pooled((size_t)Skv_per_S * D);
  std::vector<float> Q_pooled((size_t)B_per_S * D);
  std::vector<float> A_pooled((size_t)B_per_S * Skv_per_S);
  std::vector<float> block_scores(num_k_blocks);
  std::vector<int> idx(num_k_blocks);

  for (int h = 0; h < Hq; ++h) {
    const __fp16* K_h = K + (size_t)h * Skv * D;

    // Sum-pool K with stride S: K_pooled[i, :] = sum over j in [0, S) of K[i*S + j, :].
    std::fill(K_pooled.begin(), K_pooled.end(), 0.0f);
    for (int i = 0; i < Skv_per_S; ++i) {
      for (int j = 0; j < S; ++j) {
        const __fp16* row = K_h + (i * S + j) * D;
        for (int d = 0; d < D; ++d) K_pooled[(size_t)i * D + d] += (float)row[d];
      }
    }

    for (int qb = 0; qb < num_q_blocks; ++qb) {
      const __fp16* Q_block = Q + ((size_t)h * Sq + qb * B) * D;

      // Sum-pool Q within the block, stride S.
      std::fill(Q_pooled.begin(), Q_pooled.end(), 0.0f);
      for (int i = 0; i < B_per_S; ++i) {
        for (int j = 0; j < S; ++j) {
          const __fp16* row = Q_block + (i * S + j) * D;
          for (int d = 0; d < D; ++d) Q_pooled[(size_t)i * D + d] += (float)row[d];
        }
      }

      // A_pooled = Q_pooled · K_pooled^T * scale.
      for (int i = 0; i < B_per_S; ++i) {
        for (int j = 0; j < Skv_per_S; ++j) {
          float sum = 0.f;
          for (int d = 0; d < D; ++d) sum += Q_pooled[(size_t)i * D + d] * K_pooled[(size_t)j * D + d];
          A_pooled[(size_t)i * Skv_per_S + j] = sum * scale;
        }
      }

      // Row-wise softmax over Skv_per_S.
      for (int i = 0; i < B_per_S; ++i) {
        float row_max = -INFINITY;
        for (int j = 0; j < Skv_per_S; ++j) row_max = std::max(row_max, A_pooled[(size_t)i * Skv_per_S + j]);
        float row_sum = 0.f;
        for (int j = 0; j < Skv_per_S; ++j) {
          A_pooled[(size_t)i * Skv_per_S + j] = std::exp(A_pooled[(size_t)i * Skv_per_S + j] - row_max);
          row_sum += A_pooled[(size_t)i * Skv_per_S + j];
        }
        const float inv = (row_sum > 0.f) ? 1.f / row_sum : 0.f;
        for (int j = 0; j < Skv_per_S; ++j) A_pooled[(size_t)i * Skv_per_S + j] *= inv;
      }

      // Aggregate to per-original-K-block: K block b covers pooled cols
      // [b·B_per_S, (b+1)·B_per_S). Sum across all pooled queries and within
      // the block's pooled cols.
      std::fill(block_scores.begin(), block_scores.end(), 0.0f);
      for (int i = 0; i < B_per_S; ++i) {
        for (int b = 0; b < num_k_blocks; ++b) {
          float s = 0.f;
          for (int j = 0; j < B_per_S; ++j) s += A_pooled[(size_t)i * Skv_per_S + b * B_per_S + j];
          block_scores[b] += s;
        }
      }

      // Top-k.
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                        [&](int a, int c) { return block_scores[a] > block_scores[c]; });
      for (int kk = 0; kk < top_k; ++kk) {
        selections[((size_t)h * num_q_blocks + qb) * top_k + kk] = idx[kk];
      }
    }
  }
}

// CPU side: gather selected K blocks into per-q-block layout.
//   src    : K [Hq, Skv, D]
//   sel[h, q, kk] : block idx in [0, num_k_blocks) for query block q, head h
//   dst    : K_arranged [Hq, num_q_blocks, top_k*BK, D]
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

// OpenMP-parallel version: split the (h) axis across threads. Per the
// MemBandwidth tests, this is 3-4× faster than the serial version at
// Sq=256-512 (the regime where multi-threading helps the most) and
// matches serial at Sq=128 and Sq=2048 (no regression).
static void cpuGatherOMP(const __fp16* src, __fp16* dst, const std::vector<int>& sel, int Hq, int Skv, int D, int BK,
                         int num_q_blocks, int top_k, int n_threads = 8) {
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);
  #pragma omp parallel for num_threads(n_threads) schedule(static)
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

// Host reference: block-sparse attention from arranged K/V.
// Operates on the same data the NPU sees.
static void naiveBlockSparseFromArranged(const __fp16* Q, const __fp16* K_arr, const __fp16* V_arr, __fp16* O, int Hq,
                                         int Sq, int D, int BQ, int top_k_BK, int num_q_blocks, float scale) {
  std::vector<float> scores(top_k_BK);
  for (int h = 0; h < Hq; ++h) {
    for (int i = 0; i < num_q_blocks; ++i) {
      const __fp16* K_qb = K_arr + (((size_t)h * num_q_blocks + i) * top_k_BK) * D;
      const __fp16* V_qb = V_arr + (((size_t)h * num_q_blocks + i) * top_k_BK) * D;
      for (int q = 0; q < BQ; ++q) {
        const int row = i * BQ + q;
        const __fp16* qrow = Q + ((size_t)h * Sq + row) * D;
        __fp16* orow = O + ((size_t)h * Sq + row) * D;

        float row_max = -INFINITY;
        for (int c = 0; c < top_k_BK; ++c) {
          const __fp16* kc = K_qb + (size_t)c * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)kc[d];
          scores[c] = dot * scale;
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

// Dense baseline (full attention) on the same Q, K, V — for timing comparison.
static void naiveDenseFp16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, int Hq, int Sq, int Skv, int D,
                           float scale) {
  std::vector<float> scores(Skv);
  for (int h = 0; h < Hq; ++h) {
    for (int s = 0; s < Sq; ++s) {
      const __fp16* qrow = Q + ((size_t)h * Sq + s) * D;
      __fp16* orow = O + ((size_t)h * Sq + s) * D;
      float row_max = -INFINITY;
      for (int j = 0; j < Skv; ++j) {
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
        const __fp16* vrow = V + ((size_t)h * Skv + j) * D;
        const float p = scores[j] * inv;
        for (int d = 0; d < D; ++d) acc[d] += p * (float)vrow[d];
      }
      for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
    }
  }
}

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

static void runComparison(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D, int BK,
                          int top_k, const std::string& tag) {
  fprintf(stderr, "[CASE] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d\n", tag.c_str(), Sq, Hq, Skv, Hkv, D, BK,
          top_k);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Skv % BK, 0);
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

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* vp = V.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  // Generate Hkv unique heads, replicate to Hq slots (GQA pre-expansion).
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

  // ----- Random selection (stand-in for a real heuristic). Same seed across
  // both runs so K_arranged is deterministic for verification. -----
  std::vector<int> sel((size_t)Hq * num_q_blocks * top_k);
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
      std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) sel[((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
    }
  }

  // ----- CPU gather (timed) -----
  __fp16* karr_p = K_arr.ptr<__fp16>();
  __fp16* varr_p = V_arr.ptr<__fp16>();
  // warmup
  cpuGather(kp, karr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, varr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);

  const int gather_runs = 10;
  const auto gt0 = std::chrono::steady_clock::now();
  for (int i = 0; i < gather_runs; ++i) {
    cpuGather(kp, karr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);
    cpuGather(vp, varr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  }
  const auto gt1 = std::chrono::steady_clock::now();
  const double t_cpu_gather = std::chrono::duration<double, std::milli>(gt1 - gt0).count() / gather_runs;

  // ----- Static scale tensor -----
  auto scale_t = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  // ----- Host references -----
  std::vector<__fp16> ref_sparse(out_numel);
  std::vector<__fp16> ref_dense(out_numel);
  naiveBlockSparseFromArranged(qp, karr_p, varr_p, ref_sparse.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks, scale);
  naiveDenseFp16(qp, kp, vp, ref_dense.data(), Hq, Sq, Skv, D, scale);

  // ---------------------------------------------------------------------
  // Graph 1: dense attention baseline (the same shape graph used in
  // FlashAttentionDecomposedTest at this Sq).
  // ---------------------------------------------------------------------
  std::string g_dense = "dense_" + tag;
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;
  GraphTime r_dense = runOneGraph(backend, g_dense, {Q, K, V}, O_dense, ref_dense, out_numel, /*tol=*/5e-2f,
                                  [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
    backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_dense);
    backend->addStaticTensor(g, "scale", scale_3d);

    auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  });

  // ---------------------------------------------------------------------
  // Graph 2: block-sparse attention. The graph consumes K_arranged and
  // V_arranged that were prepared by the CPU above. Q is reshaped into
  // [Hq, num_q_blocks, BQ, D] inside the graph.
  // ---------------------------------------------------------------------
  std::string g_sparse = "sparse_" + tag;
  GraphTime r_sparse = runOneGraph(backend, g_sparse, {Q, K_arr, V_arr}, O_sparse, ref_sparse, out_numel,
                                   /*tol=*/5e-2f, [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
    backend->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_sparse);
    backend->addStaticTensor(g, "scale", scale_t);

    auto Q4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto O4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    backend->addTensor(g, "Q4d", QNN_TENSOR_TYPE_NATIVE, Q4d_t);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(g, "O4d", QNN_TENSOR_TYPE_NATIVE, O4d_t);

    backend->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q4d"}, {}, {}, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q4d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 3u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O4d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "reshape_o", "Reshape", {"O4d"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[CMP %-22s]\n"
          "  CPU gather (K+V)        : %7.2f ms\n"
          "  NPU dense  (full attn)  : %7.2f ms  err=%.4f miss=%zu\n"
          "  NPU sparse (block-sparse): %7.2f ms  err=%.4f miss=%zu\n"
          "  Path B total (CPU+NPU sync)        : %7.2f ms\n"
          "  Path B total (CPU pipelined-hidden): %7.2f ms\n"
          "  Speedup vs dense (sync)            : %.2fx\n"
          "  Speedup vs dense (pipelined)       : %.2fx\n",
          tag.c_str(), t_cpu_gather, r_dense.avg_ms, r_dense.max_abs_err, r_dense.mismatch, r_sparse.avg_ms,
          r_sparse.max_abs_err, r_sparse.mismatch, t_cpu_gather + r_sparse.avg_ms,
          std::max(t_cpu_gather, r_sparse.avg_ms), r_dense.avg_ms / (t_cpu_gather + r_sparse.avg_ms),
          r_dense.avg_ms / std::max(t_cpu_gather, r_sparse.avg_ms));
}

// ---------------------------------------------------------------------------
// Sweep Sq from 128 → 2048 at constant 1/4 sparsity (top_k = num_k_blocks/4).
// Shape: Qwen3-0.6B / Qwen3-1.7B (identical attention: Hq=16, Hkv=8, D=128).
//
// Note: each TEST_F should be run in a fresh process (via --gtest_filter)
// because the dual-graph build state accumulates within one binary
// invocation and corrupts the dense graph after a few cases. Use the
// helper script in tests/qnn/run_block_sparse_sweep.sh which iterates.
// ---------------------------------------------------------------------------

#define QWEN3_PREFILL_SHAPE 16, /*Skv=*/0, 8, 128, 32  // Hq, ., Hkv, D, BK (Skv filled per case)

TEST_F(BlockSparseAttentionTest, Sq128_BK32_TopK1) {
  // num_k_blocks = 4, top_k = 1.  K_arranged = 16·4·1·32·128·2 = 512 KB.
  runComparison(backend_, 128, 16, 128, 8, 128, 32, 1, "sq128");
}
TEST_F(BlockSparseAttentionTest, Sq256_BK32_TopK2) {
  runComparison(backend_, 256, 16, 256, 8, 128, 32, 2, "sq256");
}
TEST_F(BlockSparseAttentionTest, Sq384_BK32_TopK3) {
  runComparison(backend_, 384, 16, 384, 8, 128, 32, 3, "sq384");
}
TEST_F(BlockSparseAttentionTest, Sq512_BK32_TopK4) {
  runComparison(backend_, 512, 16, 512, 8, 128, 32, 4, "sq512");
}
TEST_F(BlockSparseAttentionTest, Sq768_BK32_TopK6) {
  runComparison(backend_, 768, 16, 768, 8, 128, 32, 6, "sq768");
}
TEST_F(BlockSparseAttentionTest, Sq1024_BK32_TopK8) {
  runComparison(backend_, 1024, 16, 1024, 8, 128, 32, 8, "sq1024");
}
TEST_F(BlockSparseAttentionTest, Sq1536_BK32_TopK12) {
  runComparison(backend_, 1536, 16, 1536, 8, 128, 32, 12, "sq1536");
}
TEST_F(BlockSparseAttentionTest, Sq2048_BK32_TopK16) {
  // Qwen3-0.6B / 1.7B prefill regime. K_arranged = 128 MB.
  runComparison(backend_, 2048, 16, 2048, 8, 128, 32, 16, "sq2048");
}
// Sq=4096 with full Hq=16 OOMs on the QNN PD (K_arranged would be 512 MB).
// Skipping for the 1/4-sparsity sweep.

// ---------------------------------------------------------------------------
// Pipelined: simulates Qwen3 prefill with 28 layers, with CPU gather of
// layer L+1 overlapping NPU sparse compute of layer L. Two K_arr/V_arr
// buffers ping-pong, owned by two separate QNN graphs to avoid the
// runtime's tensor wrapper aliasing across re-execute.
//
// Compares wall-clock total of:
//   - sync   : cpuGather(L) → npu(L)  serially per layer
//   - async  : npu(L) || cpuGather(L+1)  via std::async on a worker thread
//
// In production, the worker thread would be mllm::async::fork (with the
// gather wrapped as a Module). std::async here is just to measure the
// orchestration's wall-clock benefit without that integration.
// ---------------------------------------------------------------------------
#include <future>

static void runPipelined(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D, int BK,
                         int top_k, int num_layers, const std::string& tag) {
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  fprintf(stderr,
          "[CASE] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d num_layers=%d\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k, num_layers);

  // ----- Single shared inputs (Q, K, V) — same data across all simulated layers.
  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

  // Double-buffered K_arr / V_arr / O for ping-pong between graphs.
  std::array<Tensor, 2> K_arr;
  std::array<Tensor, 2> V_arr;
  std::array<Tensor, 2> O_buf;
  for (int b = 0; b < 2; ++b) {
    K_arr[b] = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_arr[b] = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
    O_buf[b] = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  }

  // Static scale const.
  auto scale_t = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));

  // ----- Random Q, K, V (GQA pre-expanded) -----
  std::mt19937 rng(0xA77E0001u);
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

  // ----- Pre-generate per-layer random selections.
  std::vector<std::vector<int>> selections(num_layers);
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int L = 0; L < num_layers; ++L) {
    selections[L].resize((size_t)Hq * num_q_blocks * top_k);
    for (int h = 0; h < Hq; ++h) {
      for (int q = 0; q < num_q_blocks; ++q) {
        for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
        std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
        for (int kk = 0; kk < top_k; ++kk) {
          selections[L][((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
        }
      }
    }
  }

  // ----- Build two parallel sparse graphs, each bound to one buffer pair.
  auto buildSparseGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    EXPECT_TRUE(backend->addTensor(gname, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q));
    EXPECT_TRUE(backend->addTensor(gname, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr[b]));
    EXPECT_TRUE(backend->addTensor(gname, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr[b]));
    EXPECT_TRUE(backend->addTensor(gname, "O", QNN_TENSOR_TYPE_APP_READ, O_buf[b]));
    EXPECT_TRUE(backend->addStaticTensor(gname, "scale", scale_t));

    auto Q4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto O4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    backend->addTensor(gname, "Q4d", QNN_TENSOR_TYPE_NATIVE, Q4d_t);
    backend->addTensor(gname, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gname, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(gname, "O4d", QNN_TENSOR_TYPE_NATIVE, O4d_t);

    backend->graphAddNode(gname, "reshape_q", "Reshape", {"Q"}, {"Q4d"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk", "MatMul", {"Q4d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gname, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 3u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gname, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gname, "matmul_av", "MatMul", {"P", "V_arr"}, {"O4d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gname, "reshape_o", "Reshape", {"O4d"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  std::string gname0 = "sparse_buf0_" + tag;
  std::string gname1 = "sparse_buf1_" + tag;
  buildSparseGraph(gname0, 0);
  buildSparseGraph(gname1, 1);

  // Warmup each graph once with valid data.
  cpuGather(kp, K_arr[0].ptr<__fp16>(), selections[0], Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr[0].ptr<__fp16>(), selections[0], Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(kp, K_arr[1].ptr<__fp16>(), selections[0], Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr[1].ptr<__fp16>(), selections[0], Hq, Skv, D, BK, num_q_blocks, top_k);
  std::vector<Tensor> ins0 = {Q, K_arr[0], V_arr[0]};
  std::vector<Tensor> outs0 = {O_buf[0]};
  std::vector<Tensor> ins1 = {Q, K_arr[1], V_arr[1]};
  std::vector<Tensor> outs1 = {O_buf[1]};
  backend->graphExecute(gname0, ins0, outs0);
  backend->graphExecute(gname1, ins1, outs1);

  // ============================== SYNCHRONOUS ==============================
  // For each layer: CPU gather → NPU dispatch (serially).
  const auto sync_t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < num_layers; ++L) {
    int b = L % 2;
    cpuGather(kp, K_arr[b].ptr<__fp16>(), selections[L], Hq, Skv, D, BK, num_q_blocks, top_k);
    cpuGather(vp, V_arr[b].ptr<__fp16>(), selections[L], Hq, Skv, D, BK, num_q_blocks, top_k);
    std::vector<Tensor> ins = {Q, K_arr[b], V_arr[b]};
    std::vector<Tensor> outs = {O_buf[b]};
    backend->graphExecute(b == 0 ? gname0 : gname1, ins, outs);
  }
  const auto sync_t1 = std::chrono::steady_clock::now();
  const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

  // =============================== PIPELINED ===============================
  // Layer L+1's gather runs on the worker thread concurrently with layer
  // L's NPU dispatch on the main thread. Two buffers ping-pong: layer L
  // reads buffer L%2, while CPU writes buffer (L+1)%2. Two graphs ensure
  // the QNN runtime keeps each buffer's tensor wrapper bindings stable.
  cpuGather(kp, K_arr[0].ptr<__fp16>(), selections[0], Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr[0].ptr<__fp16>(), selections[0], Hq, Skv, D, BK, num_q_blocks, top_k);

  const auto async_t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < num_layers; ++L) {
    int cur = L % 2;

    std::future<void> next_prep;
    if (L + 1 < num_layers) {
      int next = (L + 1) % 2;
      next_prep = std::async(std::launch::async, [&, L, next]() {
        cpuGather(kp, K_arr[next].ptr<__fp16>(), selections[L + 1], Hq, Skv, D, BK, num_q_blocks, top_k);
        cpuGather(vp, V_arr[next].ptr<__fp16>(), selections[L + 1], Hq, Skv, D, BK, num_q_blocks, top_k);
      });
    }

    // NPU dispatch for layer L (sync; CPU is preparing L+1 in parallel).
    std::vector<Tensor> ins = {Q, K_arr[cur], V_arr[cur]};
    std::vector<Tensor> outs = {O_buf[cur]};
    backend->graphExecute(cur == 0 ? gname0 : gname1, ins, outs);

    if (L + 1 < num_layers) next_prep.wait();
  }
  const auto async_t1 = std::chrono::steady_clock::now();
  const double async_ms = std::chrono::duration<double, std::milli>(async_t1 - async_t0).count();

  fprintf(stderr,
          "[PIPE %-12s] Sq=%4d top_k=%-3d  layers=%2d  sync=%8.2f ms  "
          "pipe=%8.2f ms  per-layer sync=%6.2f ms  per-layer pipe=%6.2f ms  speedup=%.2fx\n",
          tag.c_str(), Sq, top_k, num_layers, sync_ms, async_ms, sync_ms / num_layers, async_ms / num_layers,
          sync_ms / async_ms);
}

TEST_F(BlockSparseAttentionTest, Pipelined_Sq128) { runPipelined(backend_, 128, 16, 128, 8, 128, 32, 1, 28, "sq128"); }
TEST_F(BlockSparseAttentionTest, Pipelined_Sq256) { runPipelined(backend_, 256, 16, 256, 8, 128, 32, 2, 28, "sq256"); }
TEST_F(BlockSparseAttentionTest, Pipelined_Sq384) { runPipelined(backend_, 384, 16, 384, 8, 128, 32, 3, 28, "sq384"); }
TEST_F(BlockSparseAttentionTest, Pipelined_Sq512) { runPipelined(backend_, 512, 16, 512, 8, 128, 32, 4, 28, "sq512"); }
TEST_F(BlockSparseAttentionTest, Pipelined_Sq768) { runPipelined(backend_, 768, 16, 768, 8, 128, 32, 6, 28, "sq768"); }
TEST_F(BlockSparseAttentionTest, Pipelined_Sq1024) { runPipelined(backend_, 1024, 16, 1024, 8, 128, 32, 8, 28, "sq1024"); }
TEST_F(BlockSparseAttentionTest, Pipelined_Sq1536) {
  runPipelined(backend_, 1536, 16, 1536, 8, 128, 32, 12, 28, "sq1536");
}
TEST_F(BlockSparseAttentionTest, Pipelined_Sq2048) {
  runPipelined(backend_, 2048, 16, 2048, 8, 128, 32, 16, 28, "sq2048");
}

// ---------------------------------------------------------------------------
// XAttention block selection: the algorithm from the paper. Compare:
//   - selection time   : random (negligible) vs XAttention CPU
//   - output quality   : both vs dense full attention (which is the ground truth)
//
// XAttention should produce output much closer to dense, because it picks
// the K blocks with the highest approximate attention mass for each Q block,
// not random ones. Random selection is the floor; XAttention should
// approach the ceiling set by the dense baseline.
// ---------------------------------------------------------------------------

// Host-side sparse attention given an explicit selection — used to measure
// output quality (max_abs_err vs dense) for each selection method without
// going through the QNN graph (which avoids the dual-graph harness flake).
static void hostSparseAttention(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O,
                                const std::vector<int>& sel, int Hq, int Sq, int Skv, int D, int BK, int top_k,
                                float scale) {
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int top_k_BK = top_k * BK;

  std::vector<float> scores((size_t)top_k_BK);
  std::vector<float> acc((size_t)D);

  for (int h = 0; h < Hq; ++h) {
    for (int i = 0; i < num_q_blocks; ++i) {
      const int* sel_row = sel.data() + ((size_t)h * num_q_blocks + i) * top_k;
      for (int q = 0; q < BQ; ++q) {
        const __fp16* qrow = Q + ((size_t)h * Sq + i * BQ + q) * D;
        __fp16* orow = O + ((size_t)h * Sq + i * BQ + q) * D;

        // Compute scores against the top_k·BK selected K rows.
        float row_max = -INFINITY;
        for (int kk = 0; kk < top_k; ++kk) {
          const int blk = sel_row[kk];
          for (int s = 0; s < BK; ++s) {
            const __fp16* krow = K + ((size_t)h * Skv + blk * BK + s) * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)krow[d];
            const int c = kk * BK + s;
            scores[c] = dot * scale;
            if (scores[c] > row_max) row_max = scores[c];
          }
        }
        float denom = 0.f;
        for (int c = 0; c < top_k_BK; ++c) {
          scores[c] = std::exp(scores[c] - row_max);
          denom += scores[c];
        }
        const float inv = (denom > 0.f) ? 1.f / denom : 0.f;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int kk = 0; kk < top_k; ++kk) {
          const int blk = sel_row[kk];
          for (int s = 0; s < BK; ++s) {
            const __fp16* vrow = V + ((size_t)h * Skv + blk * BK + s) * D;
            const float p = scores[kk * BK + s] * inv;
            for (int d = 0; d < D; ++d) acc[d] += p * (float)vrow[d];
          }
        }
        for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
      }
    }
  }
}

static void runXAttnComparison(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                               int BK, int top_k, int xattn_S, const std::string& tag) {
  fprintf(stderr,
          "[CASE] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d xattn_S=%d\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k, xattn_S);
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(BK % xattn_S, 0);
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  // ----- Allocate -----
  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u);
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
  auto scale_t = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;

  // ----- Random selection (baseline for selection quality) -----
  std::vector<int> sel_random((size_t)Hq * num_q_blocks * top_k);
  std::vector<int> shuffle_buf(num_k_blocks);
  std::mt19937 rng2(0xBEEF0001u);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
      std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng2);
      for (int kk = 0; kk < top_k; ++kk) sel_random[((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
    }
  }

  // ----- XAttention selection (timed) -----
  std::vector<int> sel_xattn;
  blockSelectionXAttn(qp, kp, Hq, Sq, Skv, D, BK, xattn_S, top_k, sel_xattn);  // warmup
  const int sel_runs = 5;
  const auto sel_t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < sel_runs; ++r) {
    blockSelectionXAttn(qp, kp, Hq, Sq, Skv, D, BK, xattn_S, top_k, sel_xattn);
  }
  const auto sel_t1 = std::chrono::steady_clock::now();
  const double t_xattn = std::chrono::duration<double, std::milli>(sel_t1 - sel_t0).count() / sel_runs;

  // ----- Dense reference (ground truth, host-side) -----
  std::vector<__fp16> ref_dense_host(out_numel);
  naiveDenseFp16(qp, kp, vp, ref_dense_host.data(), Hq, Sq, Skv, D, scale);

  // ----- Compute host-side sparse outputs for both selections -----
  // (Uses the same data path the QNN graph would, but on CPU. Avoids the
  // dual-graph harness flake while measuring actual selection quality.)
  std::vector<__fp16> out_random(out_numel);
  std::vector<__fp16> out_xattn(out_numel);
  hostSparseAttention(qp, kp, vp, out_random.data(), sel_random, Hq, Sq, Skv, D, BK, top_k, scale);
  hostSparseAttention(qp, kp, vp, out_xattn.data(), sel_xattn, Hq, Sq, Skv, D, BK, top_k, scale);

  // Quality metrics: compare each sparse output to dense ground-truth.
  auto computeErr = [&](const std::vector<__fp16>& out) {
    float max_abs = 0.f, sum_abs = 0.f, sum_sq = 0.f;
    for (size_t i = 0; i < out_numel; ++i) {
      float diff = std::fabs((float)out[i] - (float)ref_dense_host[i]);
      max_abs = std::max(max_abs, diff);
      sum_abs += diff;
      sum_sq += diff * diff;
    }
    return std::tuple<float, float, float>(max_abs, sum_abs / out_numel, std::sqrt(sum_sq / out_numel));
  };
  auto [r_max, r_mean, r_rms] = computeErr(out_random);
  auto [x_max, x_mean, x_rms] = computeErr(out_xattn);

  // ----- Time the NPU sparse compute once (same for any selection) -----
  cpuGather(kp, K_arr.ptr<__fp16>(), sel_xattn, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr.ptr<__fp16>(), sel_xattn, Hq, Skv, D, BK, num_q_blocks, top_k);
  auto O_sparse = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  std::string g_sparse = "sparse_npu_" + tag;
  std::vector<__fp16> dummy_ref(out_numel);
  GraphTime r_npu = runOneGraph(backend, g_sparse, {Q, K_arr, V_arr}, O_sparse, dummy_ref, out_numel, /*tol=*/1e9f,
                                [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
    backend->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_sparse);
    backend->addStaticTensor(g, "scale", scale_t);
    auto Q4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
    auto O4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    backend->addTensor(g, "Q4d", QNN_TENSOR_TYPE_NATIVE, Q4d_t);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(g, "O4d", QNN_TENSOR_TYPE_NATIVE, O4d_t);
    backend->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q4d"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q4d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 3u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O4d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "reshape_o", "Reshape", {"O4d"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[XATTN %-14s] Sq=%-4d top_k=%-3d xattn_S=%d\n"
          "  Selection time (CPU)   : random %.3f ms      XAttention %.3f ms\n"
          "  Output error vs dense  : random max=%.4f rms=%.4f      XAttention max=%.4f rms=%.4f\n"
          "  Quality improvement (rms ratio random/xattn): %.2fx better with XAttention\n"
          "  NPU sparse compute time (one-shot, for context): %.2f ms\n",
          tag.c_str(), Sq, top_k, xattn_S, /*random sel*/ 0.0, t_xattn, r_max, r_rms, x_max, x_rms,
          (x_rms > 0.f) ? r_rms / x_rms : 0.f, r_npu.avg_ms);
}

TEST_F(BlockSparseAttentionTest, XAttn_Sq2048_BK32_S8) {
  runXAttnComparison(backend_, 2048, 16, 2048, 8, 128, 32, 16, /*xattn_S=*/8, "sq2048_S8");
}
TEST_F(BlockSparseAttentionTest, XAttn_Sq1024_BK32_S8) {
  runXAttnComparison(backend_, 1024, 16, 1024, 8, 128, 32, 8, /*xattn_S=*/8, "sq1024_S8");
}
TEST_F(BlockSparseAttentionTest, XAttn_Sq512_BK32_S4) {
  runXAttnComparison(backend_, 512, 16, 512, 8, 128, 32, 4, /*xattn_S=*/4, "sq512_S4");
}

// ---------------------------------------------------------------------------
// MICRO: per-dispatch QNN overhead at the per-q-block shape.
//
// Intra-layer pipelining would dispatch the attention graph num_q_blocks times
// per layer, each on a [Hq, BQ, D] Q slice. The viability of that scheme
// depends on (per-dispatch overhead) × num_q_blocks vs the cost of doing all
// q-blocks in one batched call.
//
// Compares, at the same total work (num_q_blocks q-blocks):
//   A) one rank-3 [Hq, BQ, D] graph dispatched num_q_blocks times
//   B) one rank-4 [Hq, num_q_blocks, BQ, D] graph dispatched once
//
// (B / num_q_blocks) is the pure compute per q-block.
// A − (B / num_q_blocks) is the per-dispatch overhead.
// (A × num_q_blocks) / B is the slowdown factor of the per-qb scheme.
// ---------------------------------------------------------------------------
static void runDispatchMicro(const std::shared_ptr<QNNBackend>& backend, int top_k, int num_q_blocks,
                             const std::string& tag) {
  const int Hq = 16, BQ = 32, D = 128, BK = 32;
  const int top_k_BK = top_k * BK;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto fill = [&](Tensor& t) {
    __fp16* p = t.ptr<__fp16>();
    for (size_t i = 0; i < (size_t)t.numel(); ++i) p[i] = (__fp16)dist(rng);
  };

  // ----- Rank-3 graph: one q-block per dispatch -----
  auto Q_qb = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  auto K_qb = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_qb = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O_qb = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));
  fill(Q_qb); fill(K_qb); fill(V_qb);

  std::string g_qb = "micro_qb_" + tag;
  ASSERT_NE(backend->createQnnGraph(g_qb), nullptr);
  backend->addTensor(g_qb, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_qb);
  backend->addTensor(g_qb, "K", QNN_TENSOR_TYPE_APP_WRITE, K_qb);
  backend->addTensor(g_qb, "V", QNN_TENSOR_TYPE_APP_WRITE, V_qb);
  backend->addTensor(g_qb, "O", QNN_TENSOR_TYPE_APP_READ, O_qb);
  backend->addStaticTensor(g_qb, "scale", scale_3d);

  auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  auto QKs_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  backend->addTensor(g_qb, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend->addTensor(g_qb, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend->addTensor(g_qb, "P", QNN_TENSOR_TYPE_NATIVE, P_t);

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend->graphAddNode(g_qb, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g_qb, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm3 = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                             QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g_qb, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm3, "qti.aisw");
  backend->graphAddNode(g_qb, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g_qb));

  std::vector<Tensor> qb_ins = {Q_qb, K_qb, V_qb};
  std::vector<Tensor> qb_outs = {O_qb};
  for (int i = 0; i < 5; ++i) backend->graphExecute(g_qb, qb_ins, qb_outs);  // warmup

  const int N = 200;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) backend->graphExecute(g_qb, qb_ins, qb_outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double qb_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

  // ----- Rank-4 graph: num_q_blocks q-blocks per dispatch (matches the existing sparse graph) -----
  auto Q_bt = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN).alloc();
  auto K_bt = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_bt = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O_bt = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN).alloc();
  auto scale_4d = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_4d.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));
  fill(Q_bt); fill(K_bt); fill(V_bt);

  std::string g_bt = "micro_bt_" + tag;
  ASSERT_NE(backend->createQnnGraph(g_bt), nullptr);
  backend->addTensor(g_bt, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_bt);
  backend->addTensor(g_bt, "K", QNN_TENSOR_TYPE_APP_WRITE, K_bt);
  backend->addTensor(g_bt, "V", QNN_TENSOR_TYPE_APP_WRITE, V_bt);
  backend->addTensor(g_bt, "O", QNN_TENSOR_TYPE_APP_READ, O_bt);
  backend->addStaticTensor(g_bt, "scale", scale_4d);

  auto QK_bt = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
  auto QKs_bt = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_bt = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
  backend->addTensor(g_bt, "QK", QNN_TENSOR_TYPE_NATIVE, QK_bt);
  backend->addTensor(g_bt, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_bt);
  backend->addTensor(g_bt, "P", QNN_TENSOR_TYPE_NATIVE, P_bt);

  backend->graphAddNode(g_bt, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g_bt, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm4 = {QNNParamScalarWrapper::create<uint32_t>("axis", 3u),
                                                             QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g_bt, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm4, "qti.aisw");
  backend->graphAddNode(g_bt, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g_bt));

  std::vector<Tensor> bt_ins = {Q_bt, K_bt, V_bt};
  std::vector<Tensor> bt_outs = {O_bt};
  for (int i = 0; i < 5; ++i) backend->graphExecute(g_bt, bt_ins, bt_outs);  // warmup

  const auto u0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) backend->graphExecute(g_bt, bt_ins, bt_outs);
  const auto u1 = std::chrono::steady_clock::now();
  const double bt_us = std::chrono::duration<double, std::micro>(u1 - u0).count() / N;

  const double per_qb_compute = bt_us / num_q_blocks;
  const double overhead = qb_us - per_qb_compute;
  const double per_qb_total = qb_us * num_q_blocks;

  fprintf(stderr,
          "[MICRO %-12s] top_k=%-3d num_q_blocks=%-3d  (Hq=%d BQ=%d D=%d top_k_BK=%d)\n"
          "  per-qb dispatch (rank-3, 1 qb)        : %7.1f us\n"
          "  batched dispatch (rank-4, %2d qb)     : %7.1f us\n"
          "  pure compute per q-block (= batched/N): %7.1f us\n"
          "  per-dispatch overhead (qb - compute) : %7.1f us\n"
          "  N×per-qb vs 1×batched                  : %7.1f us  vs  %7.1f us  (slowdown %.2fx)\n",
          tag.c_str(), top_k, num_q_blocks, Hq, BQ, D, top_k_BK, qb_us, num_q_blocks, bt_us, per_qb_compute, overhead,
          per_qb_total, bt_us, per_qb_total / bt_us);
}

TEST_F(BlockSparseAttentionTest, MicroDispatch_TopK1_Nq4) { runDispatchMicro(backend_, 1, 4, "tk1_nq4"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_TopK4_Nq16) { runDispatchMicro(backend_, 4, 16, "tk4_nq16"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_TopK8_Nq32) { runDispatchMicro(backend_, 8, 32, "tk8_nq32"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_TopK16_Nq64) { runDispatchMicro(backend_, 16, 64, "tk16_nq64"); }

// ---------------------------------------------------------------------------
// MICRO (split): rank-3 and rank-4 timed separately so each fits in the QNN
// PD memory budget at Sq=2048 (where the combined test OOMs the rank-4 path).
// ---------------------------------------------------------------------------
static void runDispatchMicroPerQb(const std::shared_ptr<QNNBackend>& backend, int top_k, int num_q_blocks,
                                  const std::string& tag) {
  const int Hq = 16, BQ = 32, D = 128, BK = 32;
  const int top_k_BK = top_k * BK;
  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto fill = [&](Tensor& t) {
    __fp16* p = t.ptr<__fp16>();
    for (size_t i = 0; i < (size_t)t.numel(); ++i) p[i] = (__fp16)dist(rng);
  };

  auto Q_qb = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  auto K_qb = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_qb = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O_qb = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));
  fill(Q_qb); fill(K_qb); fill(V_qb);

  std::string g = "micro_qb_only_" + tag;
  ASSERT_NE(backend->createQnnGraph(g), nullptr);
  backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_qb);
  backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K_qb);
  backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V_qb);
  backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_qb);
  backend->addStaticTensor(g, "scale", scale_3d);
  auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  auto QKs_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
  backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm3 = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                             QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm3, "qti.aisw");
  backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g));
  std::vector<Tensor> ins = {Q_qb, K_qb, V_qb}, outs = {O_qb};
  for (int i = 0; i < 5; ++i) backend->graphExecute(g, ins, outs);  // warmup

  const int N = 200;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) backend->graphExecute(g, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  fprintf(stderr,
          "[MICRO-QB %-12s] top_k=%-3d num_q_blocks=%-3d  per-qb dispatch (rank-3): %7.1f us  "
          "N×per-qb=%8.1f us\n",
          tag.c_str(), top_k, num_q_blocks, us, us * num_q_blocks);
}

static void runDispatchMicroBatched(const std::shared_ptr<QNNBackend>& backend, int top_k, int num_q_blocks,
                                    const std::string& tag) {
  const int Hq = 16, BQ = 32, D = 128, BK = 32;
  const int top_k_BK = top_k * BK;
  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto fill = [&](Tensor& t) {
    __fp16* p = t.ptr<__fp16>();
    for (size_t i = 0; i < (size_t)t.numel(); ++i) p[i] = (__fp16)dist(rng);
  };

  auto Q_bt = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN).alloc();
  auto K_bt = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_bt = Tensor::empty({Hq, num_q_blocks, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O_bt = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN).alloc();
  auto scale_4d = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_4d.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));
  fill(Q_bt); fill(K_bt); fill(V_bt);

  std::string g = "micro_bt_only_" + tag;
  ASSERT_NE(backend->createQnnGraph(g), nullptr);
  backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_bt);
  backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K_bt);
  backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V_bt);
  backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_bt);
  backend->addStaticTensor(g, "scale", scale_4d);
  auto QK_bt = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
  auto QKs_bt = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_bt = Tensor::empty({Hq, num_q_blocks, BQ, top_k_BK}, kFloat16, kQNN);
  backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_bt);
  backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_bt);
  backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_bt);
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm4 = {QNNParamScalarWrapper::create<uint32_t>("axis", 3u),
                                                             QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm4, "qti.aisw");
  backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g));
  std::vector<Tensor> ins = {Q_bt, K_bt, V_bt}, outs = {O_bt};
  for (int i = 0; i < 5; ++i) backend->graphExecute(g, ins, outs);  // warmup

  const int N = 200;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) backend->graphExecute(g, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  fprintf(stderr,
          "[MICRO-BT %-12s] top_k=%-3d num_q_blocks=%-3d  batched dispatch (rank-4): %8.1f us  "
          "per-qb-compute=%7.1f us\n",
          tag.c_str(), top_k, num_q_blocks, us, us / num_q_blocks);
}

TEST_F(BlockSparseAttentionTest, MicroDispatch_PerQb_TopK1_Nq4) { runDispatchMicroPerQb(backend_, 1, 4, "tk1_nq4"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_PerQb_TopK4_Nq16) { runDispatchMicroPerQb(backend_, 4, 16, "tk4_nq16"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_PerQb_TopK8_Nq32) { runDispatchMicroPerQb(backend_, 8, 32, "tk8_nq32"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_PerQb_TopK16_Nq64) { runDispatchMicroPerQb(backend_, 16, 64, "tk16_nq64"); }
TEST_F(BlockSparseAttentionTest, MicroDispatch_Batched_TopK16_Nq64) { runDispatchMicroBatched(backend_, 16, 64, "tk16_nq64"); }

// ===========================================================================
// INTRA-LAYER q-block pipeline.
//
// Each layer has num_q_blocks q-blocks. We use 2 rank-3 graphs, ping-pong
// over Q_buf / K_buf / V_buf / O_buf (one buffer set per graph). Within a
// layer:
//
//   sync mode      : for qb in 0..num_q_blocks-1:
//                      prep_qb(qb, buf[qb%2])    # CPU: Q slice + K/V gather
//                      execute(graph[qb%2])      # NPU
//   pipelined mode : prep_qb(0, buf[0])
//                    for qb in 0..num_q_blocks-1:
//                      cur = qb % 2
//                      async: prep_qb(qb+1, buf[(qb+1)%2])   # CPU runs in parallel
//                      execute(graph[cur])                     # NPU runs concurrently
//                      wait(next_prep)
//
// Compared to the existing per-layer pipeline (runPipelined), this overlaps
// CPU and NPU at a finer granularity, *within* a single layer. It also uses
// the per-qb (rank-3) dispatch shape, which is itself faster than rank-4 at
// long Sq per the MicroDispatch results.
// ===========================================================================

// Copy Q[*, qb*BQ:(qb+1)*BQ, *] -> Q_qb[*, *, *].
static void cpuCopyQSlice(const __fp16* Q_full, __fp16* Q_qb, int qb, int Hq, int Sq, int D, int BQ) {
  const size_t row_bytes = (size_t)BQ * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    std::memcpy(Q_qb + (size_t)h * BQ * D, Q_full + ((size_t)h * Sq + (size_t)qb * BQ) * D, row_bytes);
  }
}

// Gather one q-block's worth of selected K (or V) blocks into the per-qb dst layout.
//   src    : K_full [Hq, Skv, D]
//   dst    : K_qb   [Hq, top_k*BK, D]
//   sel_layer : flat selection for one layer, indexed sel_layer[(h*num_q_blocks + qb)*top_k + kk]
static void cpuGatherOneQb(const __fp16* src, __fp16* dst, const int* sel_layer, int qb, int Hq, int Skv, int D,
                           int BK, int num_q_blocks, int top_k) {
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    __fp16* dst_h = dst + (size_t)h * top_k * BK * D;
    for (int kk = 0; kk < top_k; ++kk) {
      int j = sel_layer[((size_t)h * num_q_blocks + qb) * top_k + kk];
      std::memcpy(dst_h + (size_t)kk * BK * D, src + ((size_t)h * Skv + (size_t)j * BK) * D, per_chunk);
    }
  }
}

static void runIntraLayerPipelined(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                                   int BK, int top_k, int num_layers, const std::string& tag) {
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  fprintf(stderr,
          "[CASE intra-layer pipe] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d num_layers=%d num_q_blocks=%d\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k, num_layers, num_q_blocks);

  // ----- Inputs (full Q/K/V owned by the caller, never written by the graph) -----
  auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

  // ----- Double-buffered per-qb scratch (one set per graph) -----
  std::array<Tensor, 2> Q_buf, K_buf, V_buf, O_buf;
  for (int b = 0; b < 2; ++b) {
    Q_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
    K_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    O_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
  }

  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));

  // ----- Random Q/K/V (GQA pre-expanded) -----
  std::mt19937 rng(0xA77E0001u);
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

  // ----- Per-layer random selections -----
  std::vector<std::vector<int>> selections(num_layers);
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int L = 0; L < num_layers; ++L) {
    selections[L].resize((size_t)Hq * num_q_blocks * top_k);
    for (int h = 0; h < Hq; ++h) {
      for (int q = 0; q < num_q_blocks; ++q) {
        for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
        std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
        for (int kk = 0; kk < top_k; ++kk) {
          selections[L][((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
        }
      }
    }
  }

  // ----- Build 2 rank-3 graphs (one per buffer set) -----
  auto buildGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    backend->addTensor(gname, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_buf[b]);
    backend->addTensor(gname, "K", QNN_TENSOR_TYPE_APP_WRITE, K_buf[b]);
    backend->addTensor(gname, "V", QNN_TENSOR_TYPE_APP_WRITE, V_buf[b]);
    backend->addTensor(gname, "O", QNN_TENSOR_TYPE_APP_READ, O_buf[b]);
    backend->addStaticTensor(gname, "scale", scale_t);
    auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    backend->addTensor(gname, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gname, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gname, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gname, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gname, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  std::string g0 = "intra_buf0_" + tag;
  std::string g1 = "intra_buf1_" + tag;
  buildGraph(g0, 0);
  buildGraph(g1, 1);

  // Warmup both graphs with valid data (Q + K/V for qb=0 of layer 0).
  cpuCopyQSlice(qp, Q_buf[0].ptr<__fp16>(), 0, Hq, Sq, D, BQ);
  cpuGatherOneQb(kp, K_buf[0].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGatherOneQb(vp, V_buf[0].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuCopyQSlice(qp, Q_buf[1].ptr<__fp16>(), 0, Hq, Sq, D, BQ);
  cpuGatherOneQb(kp, K_buf[1].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGatherOneQb(vp, V_buf[1].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  std::vector<Tensor> ins0 = {Q_buf[0], K_buf[0], V_buf[0]};
  std::vector<Tensor> outs0 = {O_buf[0]};
  std::vector<Tensor> ins1 = {Q_buf[1], K_buf[1], V_buf[1]};
  std::vector<Tensor> outs1 = {O_buf[1]};
  backend->graphExecute(g0, ins0, outs0);
  backend->graphExecute(g1, ins1, outs1);

  auto prep_qb = [&](int L, int qb, int b) {
    cpuCopyQSlice(qp, Q_buf[b].ptr<__fp16>(), qb, Hq, Sq, D, BQ);
    cpuGatherOneQb(kp, K_buf[b].ptr<__fp16>(), selections[L].data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    cpuGatherOneQb(vp, V_buf[b].ptr<__fp16>(), selections[L].data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
  };

  // ============================== SYNC ==============================
  const auto sync_t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < num_layers; ++L) {
    for (int qb = 0; qb < num_q_blocks; ++qb) {
      int b = qb % 2;
      prep_qb(L, qb, b);
      std::vector<Tensor> ins = {Q_buf[b], K_buf[b], V_buf[b]};
      std::vector<Tensor> outs = {O_buf[b]};
      backend->graphExecute(b == 0 ? g0 : g1, ins, outs);
    }
  }
  const auto sync_t1 = std::chrono::steady_clock::now();
  const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

  // ============================== PIPELINED ==============================
  // Persistent worker thread to avoid std::async per-iteration thread-spawn
  // overhead (which dominated an earlier std::async-based version).
  std::mutex mu;
  std::condition_variable cv_req, cv_done;
  int req_L = 0, req_qb = 0, req_b = 0;
  std::atomic<bool> has_req{false};
  std::atomic<bool> worker_done_pending{false};
  std::atomic<bool> stop_worker{false};

  std::thread worker([&]() {
    while (true) {
      std::unique_lock<std::mutex> lk(mu);
      cv_req.wait(lk, [&]() { return has_req.load() || stop_worker.load(); });
      if (stop_worker.load()) break;
      int L_local = req_L, qb_local = req_qb, b_local = req_b;
      has_req.store(false);
      lk.unlock();

      prep_qb(L_local, qb_local, b_local);

      {
        std::lock_guard<std::mutex> lk2(mu);
        worker_done_pending.store(true);
      }
      cv_done.notify_one();
    }
  });

  auto submit_prep = [&](int L, int qb, int b) {
    {
      std::lock_guard<std::mutex> lk(mu);
      req_L = L; req_qb = qb; req_b = b;
      has_req.store(true);
    }
    cv_req.notify_one();
  };
  auto wait_prep = [&]() {
    std::unique_lock<std::mutex> lk(mu);
    cv_done.wait(lk, [&]() { return worker_done_pending.load(); });
    worker_done_pending.store(false);
  };

  const auto async_t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < num_layers; ++L) {
    prep_qb(L, 0, 0);  // first qb sync

    for (int qb = 0; qb < num_q_blocks; ++qb) {
      int cur = qb % 2;

      bool has_next = (qb + 1 < num_q_blocks);
      if (has_next) submit_prep(L, qb + 1, (qb + 1) % 2);

      std::vector<Tensor> ins = {Q_buf[cur], K_buf[cur], V_buf[cur]};
      std::vector<Tensor> outs = {O_buf[cur]};
      backend->graphExecute(cur == 0 ? g0 : g1, ins, outs);

      if (has_next) wait_prep();
    }
  }
  const auto async_t1 = std::chrono::steady_clock::now();
  const double async_ms = std::chrono::duration<double, std::milli>(async_t1 - async_t0).count();

  stop_worker.store(true);
  cv_req.notify_one();
  worker.join();

  const double per_layer_sync = sync_ms / num_layers;
  const double per_layer_pipe = async_ms / num_layers;
  const double per_qb_sync = per_layer_sync * 1000.0 / num_q_blocks;
  const double per_qb_pipe = per_layer_pipe * 1000.0 / num_q_blocks;

  fprintf(stderr,
          "[INTRA %-12s] Sq=%-4d top_k=%-3d num_q_blocks=%-3d layers=%d\n"
          "  sync (gather→exec per qb)  : %8.2f ms total  %6.2f ms/layer  %6.1f us/qb\n"
          "  pipelined (CPU∥NPU per qb) : %8.2f ms total  %6.2f ms/layer  %6.1f us/qb\n"
          "  pipeline speedup           : %.2fx\n",
          tag.c_str(), Sq, top_k, num_q_blocks, num_layers, sync_ms, per_layer_sync, per_qb_sync, async_ms,
          per_layer_pipe, per_qb_pipe, sync_ms / async_ms);
}

TEST_F(BlockSparseAttentionTest, IntraLayer_Sq128) { runIntraLayerPipelined(backend_, 128, 16, 128, 8, 128, 32, 1, 28, "sq128"); }
TEST_F(BlockSparseAttentionTest, IntraLayer_Sq256) { runIntraLayerPipelined(backend_, 256, 16, 256, 8, 128, 32, 2, 28, "sq256"); }
TEST_F(BlockSparseAttentionTest, IntraLayer_Sq512) { runIntraLayerPipelined(backend_, 512, 16, 512, 8, 128, 32, 4, 28, "sq512"); }
TEST_F(BlockSparseAttentionTest, IntraLayer_Sq1024) { runIntraLayerPipelined(backend_, 1024, 16, 1024, 8, 128, 32, 8, 28, "sq1024"); }
TEST_F(BlockSparseAttentionTest, IntraLayer_Sq2048) { runIntraLayerPipelined(backend_, 2048, 16, 2048, 8, 128, 32, 16, 28, "sq2048"); }

#if 0  // OMP variant: tried but slower than cv-based. Kept disabled for reference.
// ---------------------------------------------------------------------------
// OpenMP variant of intra-layer pipelining. Replaces the cv-based worker with
// an OMP parallel region (2 threads) + #pragma omp barrier per q-block. The
// omp barrier is ~5-10 µs vs the cv-based ~30 µs, so each q-block saves
// ~25 µs of sync overhead.
//
// Pattern per q-block i:
//   tid=0: graphExecute(graphs[i % 2])      ← NPU dispatch
//   tid=1: prep_qb(L, i+1, (i+1) % 2)       ← CPU prep for next qb (if any)
//   omp barrier                             ← wait for both
// ---------------------------------------------------------------------------
static void runIntraLayerPipelinedOMP(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                      int D, int BK, int top_k, int num_layers, const std::string& tag) {
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;

  fprintf(stderr,
          "[CASE intra-layer pipe OMP] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d num_layers=%d num_q_blocks=%d\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k, num_layers, num_q_blocks);

  auto Q_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V_full = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();

  std::array<Tensor, 2> Q_buf, K_buf, V_buf, O_buf;
  for (int b = 0; b < 2; ++b) {
    Q_buf[b] = Tensor::empty({Hq, BQ, D}, kFloat16, kQNN).alloc();
    K_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    V_buf[b] = Tensor::empty({Hq, top_k_BK, D}, kFloat16, kQNN).alloc();
    O_buf[b] = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  }

  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q_full.ptr<__fp16>(); __fp16* kp = K_full.ptr<__fp16>(); __fp16* vp = V_full.ptr<__fp16>();
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

  std::vector<std::vector<int>> selections(num_layers);
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int L = 0; L < num_layers; ++L) {
    selections[L].resize((size_t)Hq * num_q_blocks * top_k);
    for (int h = 0; h < Hq; ++h) {
      for (int q = 0; q < num_q_blocks; ++q) {
        for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
        std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
        for (int kk = 0; kk < top_k; ++kk) {
          selections[L][((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
        }
      }
    }
  }

  auto buildGraph = [&](const std::string& gname, int b) {
    EXPECT_NE(backend->createQnnGraph(gname), nullptr);
    backend->addTensor(gname, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q_buf[b]);
    backend->addTensor(gname, "K", QNN_TENSOR_TYPE_APP_WRITE, K_buf[b]);
    backend->addTensor(gname, "V", QNN_TENSOR_TYPE_APP_WRITE, V_buf[b]);
    backend->addTensor(gname, "O", QNN_TENSOR_TYPE_APP_READ, O_buf[b]);
    backend->addStaticTensor(gname, "scale", scale_t);
    auto QK_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, BQ, top_k_BK}, kFloat16, kQNN);
    backend->addTensor(gname, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gname, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gname, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gname, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gname, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gname, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gname, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
    EXPECT_TRUE(backend->graphFinalize(gname));
  };
  std::string g0 = "intra_omp_buf0_" + tag;
  std::string g1 = "intra_omp_buf1_" + tag;
  buildGraph(g0, 0);
  buildGraph(g1, 1);

  cpuCopyQSlice(qp, Q_buf[0].ptr<__fp16>(), 0, Hq, Sq, D, BQ);
  cpuGatherOneQb(kp, K_buf[0].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGatherOneQb(vp, V_buf[0].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuCopyQSlice(qp, Q_buf[1].ptr<__fp16>(), 0, Hq, Sq, D, BQ);
  cpuGatherOneQb(kp, K_buf[1].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGatherOneQb(vp, V_buf[1].ptr<__fp16>(), selections[0].data(), 0, Hq, Skv, D, BK, num_q_blocks, top_k);
  std::vector<Tensor> ins0 = {Q_buf[0], K_buf[0], V_buf[0]};
  std::vector<Tensor> outs0 = {O_buf[0]};
  std::vector<Tensor> ins1 = {Q_buf[1], K_buf[1], V_buf[1]};
  std::vector<Tensor> outs1 = {O_buf[1]};
  backend->graphExecute(g0, ins0, outs0);
  backend->graphExecute(g1, ins1, outs1);

  auto prep_qb = [&](int L, int qb, int b) {
    cpuCopyQSlice(qp, Q_buf[b].ptr<__fp16>(), qb, Hq, Sq, D, BQ);
    cpuGatherOneQb(kp, K_buf[b].ptr<__fp16>(), selections[L].data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
    cpuGatherOneQb(vp, V_buf[b].ptr<__fp16>(), selections[L].data(), qb, Hq, Skv, D, BK, num_q_blocks, top_k);
  };

  // Warm the OMP pool so the first iteration doesn't pay thread-creation.
  #pragma omp parallel num_threads(2)
  { (void)0; }

  // ============================== SYNC (same as cv version) ==============================
  const auto sync_t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < num_layers; ++L) {
    for (int qb = 0; qb < num_q_blocks; ++qb) {
      int b = qb % 2;
      prep_qb(L, qb, b);
      std::vector<Tensor> ins = {Q_buf[b], K_buf[b], V_buf[b]};
      std::vector<Tensor> outs = {O_buf[b]};
      backend->graphExecute(b == 0 ? g0 : g1, ins, outs);
    }
  }
  const auto sync_t1 = std::chrono::steady_clock::now();
  const double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count();

  // ============================== PIPELINED via OMP ==============================
  // Single persistent parallel region for the entire timed loop — pulling
  // the `#pragma omp parallel` *outside* the layer loop avoids paying the
  // region-create cost 28 times. Two threads, explicit barriers per qb.
  //   tid=0: NPU dispatch for current qb
  //   tid=1: CPU prep for next qb
  const auto async_t0 = std::chrono::steady_clock::now();
  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    for (int L = 0; L < num_layers; ++L) {
      if (tid == 0) prep_qb(L, 0, 0);  // first qb prep on main thread
      #pragma omp barrier

      for (int qb = 0; qb < num_q_blocks; ++qb) {
        int cur = qb % 2;
        bool has_next = (qb + 1 < num_q_blocks);

        if (tid == 0) {
          std::vector<Tensor> ins = {Q_buf[cur], K_buf[cur], V_buf[cur]};
          std::vector<Tensor> outs = {O_buf[cur]};
          backend->graphExecute(cur == 0 ? g0 : g1, ins, outs);
        } else {
          if (has_next) {
            int next = (qb + 1) % 2;
            prep_qb(L, qb + 1, next);
          }
        }

        #pragma omp barrier
      }
    }
  }
  const auto async_t1 = std::chrono::steady_clock::now();
  const double async_ms = std::chrono::duration<double, std::milli>(async_t1 - async_t0).count();

  const double per_layer_sync = sync_ms / num_layers;
  const double per_layer_pipe = async_ms / num_layers;
  const double per_qb_sync = per_layer_sync * 1000.0 / num_q_blocks;
  const double per_qb_pipe = per_layer_pipe * 1000.0 / num_q_blocks;

  fprintf(stderr,
          "[INTRA-OMP %-12s] Sq=%-4d top_k=%-3d num_q_blocks=%-3d layers=%d\n"
          "  sync (gather→exec per qb)  : %8.2f ms total  %6.2f ms/layer  %6.1f us/qb\n"
          "  pipelined (CPU∥NPU per qb) : %8.2f ms total  %6.2f ms/layer  %6.1f us/qb\n"
          "  pipeline speedup           : %.2fx\n",
          tag.c_str(), Sq, top_k, num_q_blocks, num_layers, sync_ms, per_layer_sync, per_qb_sync, async_ms,
          per_layer_pipe, per_qb_pipe, sync_ms / async_ms);
}

TEST_F(BlockSparseAttentionTest, IntraLayerOMP_Sq128) { runIntraLayerPipelinedOMP(backend_, 128, 16, 128, 8, 128, 32, 1, 28, "sq128"); }
TEST_F(BlockSparseAttentionTest, IntraLayerOMP_Sq256) { runIntraLayerPipelinedOMP(backend_, 256, 16, 256, 8, 128, 32, 2, 28, "sq256"); }
TEST_F(BlockSparseAttentionTest, IntraLayerOMP_Sq512) { runIntraLayerPipelinedOMP(backend_, 512, 16, 512, 8, 128, 32, 4, 28, "sq512"); }
TEST_F(BlockSparseAttentionTest, IntraLayerOMP_Sq1024) { runIntraLayerPipelinedOMP(backend_, 1024, 16, 1024, 8, 128, 32, 8, 28, "sq1024"); }
TEST_F(BlockSparseAttentionTest, IntraLayerOMP_Sq2048) { runIntraLayerPipelinedOMP(backend_, 2048, 16, 2048, 8, 128, 32, 16, 28, "sq2048"); }
#endif  // OMP variant

// ---------------------------------------------------------------------------
// Dense-only attention (no sparse graph in the same process).
//
// The combined runComparison test corrupts the dense graph at small Sq via
// dual-graph build state — `NPU dense (full attn)` reports ~0.02 ms with
// large `miss` count instead of the real HMX compute time. This test builds
// only the dense graph so we can get a clean Sq=128 / Sq=256 / Sq=512 number.
// ---------------------------------------------------------------------------
static void runDenseOnly(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                         const std::string& tag) {
  ASSERT_EQ(Hq % Hkv, 0);
  const int group = Hq / Hkv;
  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>(); __fp16* kp = K.ptr<__fp16>(); __fp16* vp = V.ptr<__fp16>();
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

  std::vector<__fp16> ref_dense(out_numel);
  naiveDenseFp16(qp, kp, vp, ref_dense.data(), Hq, Sq, Skv, D, scale);

  std::string g = "dense_only_" + tag;
  GraphTime r = runOneGraph(backend, g, {Q, K, V}, O, ref_dense, out_numel, /*tol=*/5e-2f,
                            [&](const std::string& gn) {
    backend->addTensor(gn, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(gn, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
    backend->addTensor(gn, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
    backend->addTensor(gn, "O", QNN_TENSOR_TYPE_APP_READ, O);
    backend->addStaticTensor(gn, "scale", scale_3d);
    auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    backend->addTensor(gn, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gn, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gn, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gn, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gn, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gn, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gn, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[DENSE-ONLY %-8s] Sq=%-4d Hq=%d Hkv=%d D=%d  NPU dense (full attn): %7.3f ms  err=%.4f miss=%zu\n",
          tag.c_str(), Sq, Hq, Hkv, D, r.avg_ms, r.max_abs_err, r.mismatch);
}

TEST_F(BlockSparseAttentionTest, DenseOnly_Sq128) { runDenseOnly(backend_, 128, 16, 128, 8, 128, "sq128"); }
TEST_F(BlockSparseAttentionTest, DenseOnly_Sq256) { runDenseOnly(backend_, 256, 16, 256, 8, 128, "sq256"); }
TEST_F(BlockSparseAttentionTest, DenseOnly_Sq512) { runDenseOnly(backend_, 512, 16, 512, 8, 128, "sq512"); }
TEST_F(BlockSparseAttentionTest, DenseOnly_Sq1024) { runDenseOnly(backend_, 1024, 16, 1024, 8, 128, "sq1024"); }
TEST_F(BlockSparseAttentionTest, DenseOnly_Sq2048) { runDenseOnly(backend_, 2048, 16, 2048, 8, 128, "sq2048"); }

// ---------------------------------------------------------------------------
// RANK-4 DENSE — same FLOPs as rank-3 dense, but expressed as a rank-4 graph
// matching the topology of the sparse rank-4 graph (reshape Q to rank-4,
// matmul against full K replicated across the num_q_blocks outer batch dim).
//
// Purpose: isolate whether the rank-4 sparse slowdown vs rank-3 dense is due
// to (a) the rank-4 expression itself / QNN MatMul's handling of leading
// batch dims, or (b) something specific to sparse (e.g. different K per
// outer batch, gather memory layout).
//
// If rank-4 dense is also ~2x slower than rank-3 dense at the same FLOPs,
// the rank itself is the culprit. If rank-4 dense matches rank-3 dense, the
// slowdown is something specific to the sparse compute pattern.
//
// Memory: K_rep is [Hq, num_q_blocks, Skv, D] × 2B. At Sq=Skv=2048: 536 MB,
// OOMs the QNN PD. Tests cap at Sq=1024 (134 MB each for K_rep/V_rep).
// ---------------------------------------------------------------------------
static void runRank4DenseOnly(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D, int BK,
                              const std::string& tag) {
  ASSERT_EQ(Hq % Hkv, 0);
  ASSERT_EQ(Sq % BK, 0);
  const int group = Hq / Hkv;
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_rep = Tensor::empty({Hq, num_q_blocks, Skv, D}, kFloat16, kQNN).alloc();
  auto V_rep = Tensor::empty({Hq, num_q_blocks, Skv, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto scale_t = Tensor::empty({1, 1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);

  // Replicate K, V across num_q_blocks for each head — gives rank-4 K with
  // identical inner-tensor data across the outer batch dim.
  __fp16* krp = K_rep.ptr<__fp16>();
  __fp16* vrp = V_rep.ptr<__fp16>();
  const size_t per_head_bytes = (size_t)Skv * D * sizeof(__fp16);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    for (int qb = 0; qb < num_q_blocks; ++qb) {
      std::memcpy(krp + ((size_t)h * num_q_blocks + qb) * Skv * D,
                  K_unique.data() + (size_t)hkv * Skv * D, per_head_bytes);
      std::memcpy(vrp + ((size_t)h * num_q_blocks + qb) * Skv * D,
                  V_unique.data() + (size_t)hkv * Skv * D, per_head_bytes);
    }
  }

  // Build expanded K_full / V_full for the host reference.
  std::vector<__fp16> K_full((size_t)Hq * Skv * D), V_full((size_t)Hq * Skv * D);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(K_full.data() + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, per_head_bytes);
    std::memcpy(V_full.data() + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, per_head_bytes);
  }
  std::vector<__fp16> ref_dense(out_numel);
  naiveDenseFp16(qp, K_full.data(), V_full.data(), ref_dense.data(), Hq, Sq, Skv, D, scale);

  std::string g = "rank4_dense_" + tag;
  GraphTime r = runOneGraph(backend, g, {Q, K_rep, V_rep}, O, ref_dense, out_numel, /*tol=*/5e-2f,
                            [&](const std::string& gn) {
    backend->addTensor(gn, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(gn, "K_rep", QNN_TENSOR_TYPE_APP_WRITE, K_rep);
    backend->addTensor(gn, "V_rep", QNN_TENSOR_TYPE_APP_WRITE, V_rep);
    backend->addTensor(gn, "O", QNN_TENSOR_TYPE_APP_READ, O);
    backend->addStaticTensor(gn, "scale", scale_t);

    auto Q4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({Hq, num_q_blocks, BQ, Skv}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, num_q_blocks, BQ, Skv}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, num_q_blocks, BQ, Skv}, kFloat16, kQNN);
    auto O4d_t = Tensor::empty({Hq, num_q_blocks, BQ, D}, kFloat16, kQNN);
    backend->addTensor(gn, "Q4d", QNN_TENSOR_TYPE_NATIVE, Q4d_t);
    backend->addTensor(gn, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gn, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gn, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(gn, "O4d", QNN_TENSOR_TYPE_NATIVE, O4d_t);

    backend->graphAddNode(gn, "reshape_q", "Reshape", {"Q"}, {"Q4d"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gn, "matmul_qk", "MatMul", {"Q4d", "K_rep"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gn, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 3u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gn, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gn, "matmul_av", "MatMul", {"P", "V_rep"}, {"O4d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gn, "reshape_o", "Reshape", {"O4d"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[RANK4-DENSE %-8s] Sq=%-4d Hq=%d Hkv=%d D=%d num_q_blocks=%d  NPU rank-4 dense: %7.3f ms  err=%.4f miss=%zu\n",
          tag.c_str(), Sq, Hq, Hkv, D, num_q_blocks, r.avg_ms, r.max_abs_err, r.mismatch);
}

TEST_F(BlockSparseAttentionTest, Rank4Dense_Sq128) { runRank4DenseOnly(backend_, 128, 16, 128, 8, 128, 32, "sq128"); }
TEST_F(BlockSparseAttentionTest, Rank4Dense_Sq256) { runRank4DenseOnly(backend_, 256, 16, 256, 8, 128, 32, "sq256"); }
TEST_F(BlockSparseAttentionTest, Rank4Dense_Sq512) { runRank4DenseOnly(backend_, 512, 16, 512, 8, 128, 32, "sq512"); }
TEST_F(BlockSparseAttentionTest, Rank4Dense_Sq1024) { runRank4DenseOnly(backend_, 1024, 16, 1024, 8, 128, 32, "sq1024"); }

// ---------------------------------------------------------------------------
// RANK-3 vs RANK-4 with IDENTICAL DATA SIZE
//
// Same K_arr data layout as the sparse rank-4 graph, but the QNN tensors
// are declared with rank-3 shape [Hq*num_qb, ...] instead of rank-4
// [Hq, num_qb, ...]. Bytes in memory are identical; only QNN's shape
// interpretation differs. This isolates the rank-4 scheduling overhead from
// memory bandwidth confounders.
//
// If rank-3 is meaningfully faster than rank-4 at the same data size and
// compute, the rank-4 expression itself has a scheduling penalty in QNN's
// MatMul op (e.g. outer-batch serialisation).
// ---------------------------------------------------------------------------
static void runRank3SparseSameData(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                                   int BK, int top_k, const std::string& tag) {
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int group = Hq / Hkv;
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  const int top_k_BK = top_k * BK;
  const int big_batch = Hq * num_q_blocks;

  // Declare tensors with rank-3 shape [Hq*num_qb, ...] — same total bytes
  // as the rank-4 sparse graph's [Hq, num_qb, ...].
  auto Q = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN).alloc();
  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)(1.0f / std::sqrt((float)D));

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < (size_t)big_batch * BQ * D; ++i) Q.ptr<__fp16>()[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < (size_t)big_batch * top_k_BK * D; ++i) K_arr.ptr<__fp16>()[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < (size_t)big_batch * top_k_BK * D; ++i) V_arr.ptr<__fp16>()[i] = (__fp16)dist(rng);

  std::string g = "rank3_same_data_" + tag;
  EXPECT_NE(backend->createQnnGraph(g), nullptr);
  backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
  backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
  backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
  backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O);
  backend->addStaticTensor(g, "scale", scale_t);

  auto QK_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto QKs_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                            QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
  backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  EXPECT_TRUE(backend->graphFinalize(g));

  std::vector<Tensor> ins = {Q, K_arr, V_arr};
  std::vector<Tensor> outs = {O};
  for (int i = 0; i < 5; ++i) backend->graphExecute(g, ins, outs);  // warmup

  const int N = 50;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) backend->graphExecute(g, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;

  fprintf(stderr,
          "[RANK3-SAMEDATA %-8s] Sq=%-4d top_k=%-3d num_qb=%-3d big_batch=%-4d top_k_BK=%-4d "
          "(Q+K_arr+V_arr+O = %.1f MB)  NPU rank-3 same data: %7.3f ms\n",
          tag.c_str(), Sq, top_k, num_q_blocks, big_batch, top_k_BK,
          (big_batch * (2 * BQ * D + 2 * top_k_BK * D) * 2) / 1024.0 / 1024.0, ms);
}

TEST_F(BlockSparseAttentionTest, Rank3SameData_Sq128) { runRank3SparseSameData(backend_, 128, 16, 128, 8, 128, 32, 1, "sq128"); }
TEST_F(BlockSparseAttentionTest, Rank3SameData_Sq256) { runRank3SparseSameData(backend_, 256, 16, 256, 8, 128, 32, 2, "sq256"); }
TEST_F(BlockSparseAttentionTest, Rank3SameData_Sq512) { runRank3SparseSameData(backend_, 512, 16, 512, 8, 128, 32, 4, "sq512"); }
TEST_F(BlockSparseAttentionTest, Rank3SameData_Sq1024) { runRank3SparseSameData(backend_, 1024, 16, 1024, 8, 128, 32, 8, "sq1024"); }
TEST_F(BlockSparseAttentionTest, Rank3SameData_Sq2048) { runRank3SparseSameData(backend_, 2048, 16, 2048, 8, 128, 32, 16, "sq2048"); }

// ---------------------------------------------------------------------------
// BIG-BATCH RANK-3 SPARSE — end-to-end with real gather + correctness check.
//
// Same data path as the rank-4 sparse graph in runComparison, but the graph
// uses rank-3 tensor shapes [Hq*num_qb, BQ, D] / [Hq*num_qb, top_k·BK, D]
// instead of rank-4 [Hq, num_qb, BQ, D] / [Hq, num_qb, top_k·BK, D].
// Identical bytes in memory (cpuGather is byte-layout-compatible between
// rank-3 and rank-4 interpretations); only the QNN shape header differs.
// Softmax axis is 2 (last dim of rank-3) instead of 3.
//
// Reports CPU gather, NPU dispatch, and verifies output against the
// naiveBlockSparseFromArranged host reference (same one used by the
// runComparison sparse path).
// ---------------------------------------------------------------------------
static void runBigBatchSparseE2E(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                                 int BK, int top_k, const std::string& tag) {
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Skv % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int group = Hq / Hkv;
  const int big_batch = Hq * num_q_blocks;

  fprintf(stderr,
          "[CASE big-batch rank-3 e2e] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d BK=%d top_k=%d big_batch=%d\n",
          tag.c_str(), Sq, Hq, Skv, Hkv, D, BK, top_k, big_batch);

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  // big-batch rank-3 layout; byte-identical to rank-4 [Hq, num_qb, top_k·BK, D]
  auto K_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>(); __fp16* kp = K.ptr<__fp16>(); __fp16* vp = V.ptr<__fp16>();
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

  // Random selection — same logic as runComparison so the test is comparable.
  std::vector<int> sel((size_t)Hq * num_q_blocks * top_k);
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
      std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) sel[((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
    }
  }

  __fp16* karr_p = K_arr.ptr<__fp16>();
  __fp16* varr_p = V_arr.ptr<__fp16>();

  // Warmup OpenMP pool so the first timed iter doesn't pay thread-creation.
  #pragma omp parallel num_threads(8)
  { (void)0; }

  cpuGatherOMP(kp, karr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGatherOMP(vp, varr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);

  const int gather_runs = 10;
  const auto gt0 = std::chrono::steady_clock::now();
  for (int i = 0; i < gather_runs; ++i) {
    cpuGatherOMP(kp, karr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);
    cpuGatherOMP(vp, varr_p, sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  }
  const auto gt1 = std::chrono::steady_clock::now();
  const double t_cpu_gather = std::chrono::duration<double, std::milli>(gt1 - gt0).count() / gather_runs;

  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  // Host reference — naiveBlockSparseFromArranged treats K_arr/V_arr as
  // [Hq, num_q_blocks, top_k_BK, D]. Same bytes as the rank-3 layout, so
  // the reference is correct for both.
  std::vector<__fp16> ref_sparse(out_numel);
  naiveBlockSparseFromArranged(qp, karr_p, varr_p, ref_sparse.data(), Hq, Sq, D, BQ, top_k_BK, num_q_blocks, scale);

  std::string g = "bigbatch_sparse_" + tag;
  GraphTime r = runOneGraph(backend, g, {Q, K_arr, V_arr}, O, ref_sparse, out_numel, /*tol=*/5e-2f,
                            [&](const std::string& gn) {
    backend->addTensor(gn, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(gn, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
    backend->addTensor(gn, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
    backend->addTensor(gn, "O", QNN_TENSOR_TYPE_APP_READ, O);
    backend->addStaticTensor(gn, "scale", scale_t);

    auto Q3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
    auto QK_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto P_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
    auto O3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
    backend->addTensor(gn, "Q3d", QNN_TENSOR_TYPE_NATIVE, Q3d_t);
    backend->addTensor(gn, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gn, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gn, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(gn, "O3d", QNN_TENSOR_TYPE_NATIVE, O3d_t);

    backend->graphAddNode(gn, "reshape_q", "Reshape", {"Q"}, {"Q3d"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(gn, "matmul_qk", "MatMul", {"Q3d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
    backend->graphAddNode(gn, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gn, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gn, "matmul_av", "MatMul", {"P", "V_arr"}, {"O3d"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gn, "reshape_o", "Reshape", {"O3d"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[BIGBATCH-E2E %-8s] Sq=%-4d top_k=%-3d big_batch=%-4d\n"
          "  CPU gather (K+V)            : %7.2f ms\n"
          "  NPU big-batch rank-3 sparse : %7.3f ms  err=%.4f miss=%zu\n"
          "  total (gather + NPU)        : %7.2f ms\n",
          tag.c_str(), Sq, top_k, big_batch, t_cpu_gather, r.avg_ms, r.max_abs_err, r.mismatch,
          t_cpu_gather + r.avg_ms);
}

// ---------------------------------------------------------------------------
// DumpContext: write a QNN context binary + raw input tensor files for a
// given graph, so we can post-hoc profile it with qnn-net-run +
// qnn-profile-viewer (which expects FlatBuffer-wrapped optrace logs that
// the in-process event API doesn't produce).
//
// Outputs (in CWD):
//   <tag>.bin           QNN context binary
//   <tag>_Q.raw         Q tensor (fp16)
//   <tag>_K.raw         K tensor (fp16)
//   <tag>_V.raw         V tensor (fp16)
//   <tag>_inputs.txt    input_list.txt for qnn-net-run
// ---------------------------------------------------------------------------
static void dumpDenseContextForNetRun(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                      int D, const std::string& tag) {
  ASSERT_EQ(Hq % Hkv, 0);
  const int group = Hq / Hkv;
  const float scale = 1.0f / std::sqrt((float)D);

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>(); __fp16* kp = K.ptr<__fp16>(); __fp16* vp = V.ptr<__fp16>();
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

  std::string g = "dump_dense_" + tag;
  ASSERT_NE(backend->createQnnGraph(g), nullptr);
  backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
  backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
  backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
  backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O);
  backend->addStaticTensor(g, "scale", scale_3d);
  auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
  auto QKs_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
  auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
  backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                            QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
  backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g));

  // Save context binary.
  std::string ctx_path = tag + ".bin";
  backend->saveContext(ctx_path);

  // Save raw input tensors.
  auto dump = [&](const std::string& path, const __fp16* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), n * sizeof(__fp16));
  };
  dump(tag + "_Q.raw", qp, (size_t)Hq * Sq * D);
  dump(tag + "_K.raw", kp, (size_t)Hq * Skv * D);
  dump(tag + "_V.raw", vp, (size_t)Hq * Skv * D);

  // input_list.txt format for qnn-net-run: each line lists one set of inputs
  // separated by spaces, with name=path syntax. One line = one execute call.
  std::ofstream input_list(tag + "_inputs.txt");
  input_list << "Q:=" << tag << "_Q.raw V:=" << tag << "_V.raw K:=" << tag << "_K.raw\n";

  fprintf(stderr, "[DUMP %s] context=%s inputs=%s_inputs.txt\n", tag.c_str(), ctx_path.c_str(), tag.c_str());
}

TEST_F(BlockSparseAttentionTest, DumpDenseContext_Sq512) {
  dumpDenseContextForNetRun(backend_, 512, 16, 512, 8, 128, "dense_sq512");
}

// ---------------------------------------------------------------------------
// PRE-TRANSPOSED K experiment.
//
// The default dense graph computes QK = MatMul(Q, K, transpose_in1=true),
// which costs ~943 K cycles of q::Transpose_impl per dispatch (9% of total
// dense Sq=512). If we transpose K on the host once at chunk-prep time and
// feed it as K_T [Hq, D, Skv], we can drop transpose_in1 and let QNN do
// MatMul(Q, K_T) with K_T already in matmul-ready layout.
//
// This test mirrors runDenseOnly but with K transposed on the host. It
// also dumps a context binary for downstream qnn-net-run / qnn-profile-viewer
// inspection so we can verify q::Transpose_impl goes to zero.
// ---------------------------------------------------------------------------
static void runDenseOnlyPreTransposedK(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                       int D, const std::string& tag, bool dump_context = false) {
  ASSERT_EQ(Hq % Hkv, 0);
  const int group = Hq / Hkv;
  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K_T = Tensor::empty({Hq, D, Skv}, kFloat16, kQNN).alloc();  // pre-transposed
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto scale_3d = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_3d.ptr<__fp16>()[0] = (__fp16)scale;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>(); __fp16* vp = V.ptr<__fp16>(); __fp16* ktp = K_T.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);

  // Generate the original (non-transposed) K data so we can both feed it
  // transposed to the graph AND use it untransposed for the host reference.
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);
  std::vector<__fp16> K_full((size_t)Hq * Skv * D);
  for (int h = 0; h < Hq; ++h) {
    int hkv = h / group;
    std::memcpy(K_full.data() + (size_t)h * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)h * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  // Host-side transpose: K[h, s, d] -> K_T[h, d, s]. Stride-2 strided copy.
  // Time this so we know what we're "paying" on CPU to save on NPU.
  const int xpose_runs = 50;
  const auto xpose_t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < xpose_runs; ++r) {
    for (int h = 0; h < Hq; ++h) {
      const __fp16* src = K_full.data() + (size_t)h * Skv * D;
      __fp16* dst = ktp + (size_t)h * D * Skv;
      for (int s = 0; s < Skv; ++s) {
        for (int d = 0; d < D; ++d) {
          dst[(size_t)d * Skv + s] = src[(size_t)s * D + d];
        }
      }
    }
  }
  const auto xpose_t1 = std::chrono::steady_clock::now();
  const double t_xpose = std::chrono::duration<double, std::milli>(xpose_t1 - xpose_t0).count() / xpose_runs;

  std::vector<__fp16> ref_dense(out_numel);
  naiveDenseFp16(qp, K_full.data(), vp, ref_dense.data(), Hq, Sq, Skv, D, scale);

  std::string g = "dense_pretxk_" + tag;
  GraphTime r = runOneGraph(backend, g, {Q, K_T, V}, O, ref_dense, out_numel, /*tol=*/5e-2f,
                            [&](const std::string& gn) {
    backend->addTensor(gn, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(gn, "K_T", QNN_TENSOR_TYPE_APP_WRITE, K_T);
    backend->addTensor(gn, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
    backend->addTensor(gn, "O", QNN_TENSOR_TYPE_APP_READ, O);
    backend->addStaticTensor(gn, "scale", scale_3d);
    auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QKs_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    backend->addTensor(gn, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(gn, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
    backend->addTensor(gn, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    // NOTE: no transpose_in1 — K is already in (D, Skv) layout, so a plain
    // matmul Q[Hq, Sq, D] · K_T[Hq, D, Skv] yields QK[Hq, Sq, Skv].
    backend->graphAddNode(gn, "matmul_qk", "MatMul", {"Q", "K_T"}, {"QK"}, {}, {}, "qti.aisw");
    backend->graphAddNode(gn, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(gn, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(gn, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  });

  fprintf(stderr,
          "[DENSE-PRETXK %-8s] Sq=%-4d Hq=%d Hkv=%d D=%d  CPU K transpose: %6.3f ms  "
          "NPU dense (full attn, pre-transposed K): %7.3f ms  err=%.4f miss=%zu\n",
          tag.c_str(), Sq, Hq, Hkv, D, t_xpose, r.avg_ms, r.max_abs_err, r.mismatch);

  if (dump_context) {
    backend->saveContext(tag + ".bin");
    auto dump = [&](const std::string& path, const __fp16* p, size_t n) {
      std::ofstream f(path, std::ios::binary);
      f.write(reinterpret_cast<const char*>(p), n * sizeof(__fp16));
    };
    dump(tag + "_Q.raw", qp, (size_t)Hq * Sq * D);
    dump(tag + "_K_T.raw", ktp, (size_t)Hq * D * Skv);
    dump(tag + "_V.raw", vp, (size_t)Hq * Skv * D);
    std::ofstream input_list(tag + "_inputs.txt");
    input_list << "Q:=" << tag << "_Q.raw V:=" << tag << "_V.raw K_T:=" << tag << "_K_T.raw\n";
    fprintf(stderr, "[DUMP %s] context=%s.bin inputs=%s_inputs.txt\n", tag.c_str(), tag.c_str(), tag.c_str());
  }
}

TEST_F(BlockSparseAttentionTest, DenseOnly_PreTxK_Sq512) {
  runDenseOnlyPreTransposedK(backend_, 512, 16, 512, 8, 128, "pretxk_sq512", /*dump_context=*/false);
}
TEST_F(BlockSparseAttentionTest, DumpDenseContextPreTxK_Sq512) {
  runDenseOnlyPreTransposedK(backend_, 512, 16, 512, 8, 128, "pretxk_dump_sq512", /*dump_context=*/true);
}

// Dump the big-batch rank-3 sparse graph + already-gathered K_arr/V_arr so
// we can run it under qnn-net-run + qnn-profile-viewer for per-op profiling.
static void dumpBigBatchSparseForNetRun(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv,
                                       int D, int BK, int top_k, const std::string& tag) {
  ASSERT_EQ(Sq % BK, 0);
  ASSERT_EQ(Hq % Hkv, 0);
  const int group = Hq / Hkv;
  const int BQ = BK;
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;
  ASSERT_LE(top_k, num_k_blocks);
  const int top_k_BK = top_k * BK;
  const int big_batch = Hq * num_q_blocks;
  const float scale = 1.0f / std::sqrt((float)D);

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto V_arr = Tensor::empty({big_batch, top_k_BK, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qp = Q.ptr<__fp16>(); __fp16* kp = K.ptr<__fp16>(); __fp16* vp = V.ptr<__fp16>();
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

  // Random selection + gather (so K_arr/V_arr have realistic content).
  std::vector<int> sel((size_t)Hq * num_q_blocks * top_k);
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
      std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) sel[((size_t)h * num_q_blocks + q) * top_k + kk] = shuffle_buf[kk];
    }
  }
  cpuGather(kp, K_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);
  cpuGather(vp, V_arr.ptr<__fp16>(), sel, Hq, Skv, D, BK, num_q_blocks, top_k);

  std::string g = "dump_bigbatch_" + tag;
  ASSERT_NE(backend->createQnnGraph(g), nullptr);
  backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
  backend->addTensor(g, "K_arr", QNN_TENSOR_TYPE_APP_WRITE, K_arr);
  backend->addTensor(g, "V_arr", QNN_TENSOR_TYPE_APP_WRITE, V_arr);
  backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O);
  backend->addStaticTensor(g, "scale", scale_t);
  auto Q3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
  auto QK_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto QKs_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto P_t = Tensor::empty({big_batch, BQ, top_k_BK}, kFloat16, kQNN);
  auto O3d_t = Tensor::empty({big_batch, BQ, D}, kFloat16, kQNN);
  backend->addTensor(g, "Q3d", QNN_TENSOR_TYPE_NATIVE, Q3d_t);
  backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend->addTensor(g, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
  backend->addTensor(g, "O3d", QNN_TENSOR_TYPE_NATIVE, O3d_t);
  backend->graphAddNode(g, "reshape_q", "Reshape", {"Q"}, {"Q3d"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
  backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q3d", "K_arr"}, {"QK"}, {}, mm, "qti.aisw");
  backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                            QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g, "softmax", "Softmax", {"QKs"}, {"P"}, {}, sm, "qti.aisw");
  backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V_arr"}, {"O3d"}, {}, {}, "qti.aisw");
  backend->graphAddNode(g, "reshape_o", "Reshape", {"O3d"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g));

  backend->saveContext(tag + ".bin");
  auto dump = [&](const std::string& path, const __fp16* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), n * sizeof(__fp16));
  };
  dump(tag + "_Q.raw", qp, (size_t)Hq * Sq * D);
  dump(tag + "_K_arr.raw", K_arr.ptr<__fp16>(), (size_t)big_batch * top_k_BK * D);
  dump(tag + "_V_arr.raw", V_arr.ptr<__fp16>(), (size_t)big_batch * top_k_BK * D);
  std::ofstream input_list(tag + "_inputs.txt");
  input_list << "Q:=" << tag << "_Q.raw V_arr:=" << tag << "_V_arr.raw K_arr:=" << tag << "_K_arr.raw\n";
  fprintf(stderr, "[DUMP %s] big_batch=%d top_k_BK=%d context=%s.bin\n", tag.c_str(), big_batch, top_k_BK, tag.c_str());
}

TEST_F(BlockSparseAttentionTest, DumpBigBatchContext_Sq512) {
  dumpBigBatchSparseForNetRun(backend_, 512, 16, 512, 8, 128, 32, 4, "bigbatch_sq512");
}
TEST_F(BlockSparseAttentionTest, BigBatchE2E_Sq128) { runBigBatchSparseE2E(backend_, 128, 16, 128, 8, 128, 32, 1, "sq128"); }
TEST_F(BlockSparseAttentionTest, BigBatchE2E_Sq256) { runBigBatchSparseE2E(backend_, 256, 16, 256, 8, 128, 32, 2, "sq256"); }
TEST_F(BlockSparseAttentionTest, BigBatchE2E_Sq512) { runBigBatchSparseE2E(backend_, 512, 16, 512, 8, 128, 32, 4, "sq512"); }
TEST_F(BlockSparseAttentionTest, BigBatchE2E_Sq1024) { runBigBatchSparseE2E(backend_, 1024, 16, 1024, 8, 128, 32, 8, "sq1024"); }
TEST_F(BlockSparseAttentionTest, BigBatchE2E_Sq2048) { runBigBatchSparseE2E(backend_, 2048, 16, 2048, 8, 128, 32, 16, "sq2048"); }
