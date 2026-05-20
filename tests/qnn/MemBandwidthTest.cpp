// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Probes CPU↔rpcmem bandwidth and the actual data-rearrangement cost of the
// "CPU selects, NPU computes" sparse-attention design (Path B). Answers:
//
//   1. How fast can the CPU read/write a kQNN-allocated buffer (which is
//      rpcmem-backed and visible to both CPU and DSP)? Compared to a
//      regular kCPU buffer.
//   2. What's the throughput of the actual block-gather pattern that Path B
//      would use — many smallish memcpys from random offsets in K into a
//      contiguous "K_arranged" tile?
//   3. What's the cost of the heuristic itself (max-pool over BK rows per K
//      block, then a small importance MatMul)?
//
// All measurements are CPU-side only — no QNN graph executes. We just need
// the QNN backend to be initialised so the kQNN allocator is registered.

#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <omp.h>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

#include "mllm/backends/cpu/CPUBackend.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "QnnBackend.h"

using namespace mllm;
using namespace mllm::qnn;

static void unbufferOutput() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
}

class MemBandwidthTest : public testing::Test {
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
    // Register CPU host backend + allocator. The kCPU allocator is needed
    // by Tensor::empty(..., kCPU).alloc() (the baseline against rpcmem).
    auto host_backend = cpu::createCPUBackend();
    ctx.registerBackend(host_backend);
    ctx.memoryManager()->registerAllocator(kCPU, host_backend->allocator(), MemoryManagerOptions());

    // Register QNN backend + rpcmem-backed kQNN allocator.
    backend_ = std::make_shared<QNNBackend>();
    ASSERT_TRUE(backend_->createContext());
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(
        kQNN, backend_->allocator(), {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
  }
  static void TearDownTestSuite() { backend_.reset(); }
  static std::shared_ptr<QNNBackend> backend_;
};
std::shared_ptr<QNNBackend> MemBandwidthTest::backend_ = nullptr;

static double timeIt(std::function<void()> fn, int runs) {
  fn();  // warmup once (warms cache, may matter for small sizes)
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) fn();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / runs;
}

// ---------------------------------------------------------------------------
// Test 1 — straight memcpy: rpcmem ↔ rpcmem ↔ ordinary heap.
//
// The compiler lowers std::memcpy to NEON-vectorised loads/stores on
// aarch64-android, so we're measuring the memory subsystem, not the
// compiler. Buffers are 32 MB so they overflow L2 cache and force DRAM
// traffic — the regime that matters for our 8–32 MB per-layer K/V data.
// ---------------------------------------------------------------------------
static void runMemcpy(size_t bytes, const std::string& tag) {
  const int runs = 20;

  auto qnn1 = Tensor::empty({(int)bytes}, kInt8, kQNN).alloc();
  auto qnn2 = Tensor::empty({(int)bytes}, kInt8, kQNN).alloc();
  auto cpu1 = Tensor::empty({(int)bytes}, kInt8, kCPU).alloc();
  auto cpu2 = Tensor::empty({(int)bytes}, kInt8, kCPU).alloc();

  uint8_t* qp1 = qnn1.ptr<uint8_t>();
  uint8_t* qp2 = qnn2.ptr<uint8_t>();
  uint8_t* cp1 = cpu1.ptr<uint8_t>();
  uint8_t* cp2 = cpu2.ptr<uint8_t>();

  // Fill with non-trivial data so we can't be optimised to zero-page mappings.
  std::mt19937 rng(0xA77E0001u);
  for (size_t i = 0; i < bytes; ++i) {
    qp1[i] = (uint8_t)(rng() & 0xFF);
    cp1[i] = (uint8_t)(rng() & 0xFF);
  }

  const double t_qq = timeIt([&] { std::memcpy(qp2, qp1, bytes); }, runs);
  const double t_qc = timeIt([&] { std::memcpy(cp2, qp1, bytes); }, runs);  // CPU read of rpcmem
  const double t_cq = timeIt([&] { std::memcpy(qp2, cp1, bytes); }, runs);  // CPU write to rpcmem
  const double t_cc = timeIt([&] { std::memcpy(cp2, cp1, bytes); }, runs);

  const auto bw = [bytes](double t) { return (bytes / 1e9) / t; };

  fprintf(stderr,
          "[BW %-12s %3zu MB]  qnn->qnn %6.2f GB/s (%.2f ms)  "
          "qnn->cpu %6.2f GB/s (%.2f ms)  cpu->qnn %6.2f GB/s (%.2f ms)  "
          "cpu->cpu %6.2f GB/s (%.2f ms)\n",
          tag.c_str(), bytes / (1024 * 1024), bw(t_qq), t_qq * 1000, bw(t_qc), t_qc * 1000, bw(t_cq), t_cq * 1000,
          bw(t_cc), t_cc * 1000);
}

// ---------------------------------------------------------------------------
// Test 2 — the actual block-gather pattern Path B uses.
//
//   Source K: [Hq, Skv, D] fp16, allocated as kQNN.
//   For each (q_block, head, selected_block_idx), memcpy a contiguous BK*D
//   chunk into K_arranged at a deterministic offset.
//
// Selection indices are random per (q_block, head) — represents the
// worst-case where heads pick different K blocks (no sharing). If a real
// algorithm shares selections across heads, the actual cost is lower.
// ---------------------------------------------------------------------------
static void runBlockGather(int Hq, int Skv, int D, int BK, int num_q_blocks, int top_k, const std::string& tag) {
  const int num_k_blocks = Skv / BK;
  const size_t k_bytes = (size_t)Hq * Skv * D * sizeof(__fp16);
  const size_t arranged_bytes = (size_t)Hq * num_q_blocks * top_k * BK * D * sizeof(__fp16);
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);

  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({Hq, num_q_blocks, top_k * BK, D}, kFloat16, kQNN).alloc();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* ap = K_arr.ptr<__fp16>();

  // Fill K with random data.
  std::mt19937 rng(0xA77E0001u);
  for (size_t i = 0; i < (size_t)Hq * Skv * D; ++i) kp[i] = (__fp16)(((rng() % 1000) - 500) * 0.001f);

  // Pre-compute random selection: which top_k of num_k_blocks for each (h, q_block).
  std::vector<int> sel((size_t)Hq * num_q_blocks * top_k);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      // Distinct per-q-block selection (otherwise CPU prefetcher would overlap).
      std::vector<int> all(num_k_blocks);
      for (int j = 0; j < num_k_blocks; ++j) all[j] = j;
      std::shuffle(all.begin(), all.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) sel[((size_t)h * num_q_blocks + q) * top_k + kk] = all[kk];
    }
  }

  const int runs = 20;
  auto fn = [&] {
    for (int h = 0; h < Hq; ++h) {
      for (int q = 0; q < num_q_blocks; ++q) {
        __fp16* dst_qb = ap + (((size_t)h * num_q_blocks + q) * top_k * BK * D);
        for (int kk = 0; kk < top_k; ++kk) {
          int j = sel[((size_t)h * num_q_blocks + q) * top_k + kk];
          const __fp16* src = kp + ((size_t)h * Skv + j * BK) * D;
          std::memcpy(dst_qb + kk * BK * D, src, per_chunk);
        }
      }
    }
  };

  const double t = timeIt(fn, runs);

  // Write-side bandwidth: bytes written into K_arranged per second. One-sided
  // (no 2× read+write factor) so the number is directly comparable to the
  // pure-memcpy `bytes / time` figures.
  const double bw_gbs = (arranged_bytes / 1e9) / t;

  fprintf(stderr,
          "[BG %-18s]  Hq=%d Skv=%d BK=%d num_q=%d top_k=%d  src=%zu MB  dst=%zu MB  "
          "%zu chunks of %zu KB each  %.2f ms  %.2f GB/s write\n",
          tag.c_str(), Hq, Skv, BK, num_q_blocks, top_k, k_bytes / (1024 * 1024), arranged_bytes / (1024 * 1024),
          (size_t)Hq * num_q_blocks * top_k, per_chunk / 1024, t * 1000, bw_gbs);
}

// ---------------------------------------------------------------------------
// Test 3 — heuristic compute: max-pool over BK rows per K block, then a
// small importance MatMul. This is the CPU-side selection cost.
//
// For each (h, j_block): K_summary[h, j_block, d] = max over BK rows of K[h, j_block*BK + s, d].
// Then importance[h, i, j] = sum_d Q_summary[h, i, d] * K_summary[h, j, d].
// Top-k of importance is trivial in std::partial_sort, included here too.
// ---------------------------------------------------------------------------
static void runHeuristic(int Hq, int Sq, int Skv, int D, int BQ, int BK, int top_k, const std::string& tag) {
  const int num_q_blocks = Sq / BQ;
  const int num_k_blocks = Skv / BK;

  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  std::vector<__fp16> Q_sum((size_t)Hq * num_q_blocks * D);
  std::vector<__fp16> K_sum((size_t)Hq * num_k_blocks * D);
  std::vector<float> importance((size_t)Hq * num_q_blocks * num_k_blocks);
  std::vector<int> selected((size_t)Hq * num_q_blocks * top_k);

  __fp16* qp = Q.ptr<__fp16>();
  __fp16* kp = K.ptr<__fp16>();
  std::mt19937 rng(0xA77E0001u);
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)(((rng() % 1000) - 500) * 0.001f);
  for (size_t i = 0; i < (size_t)Hq * Skv * D; ++i) kp[i] = (__fp16)(((rng() % 1000) - 500) * 0.001f);

  const int runs = 20;
  auto fn = [&] {
    // (a) Q max-pool: Q_summary[h, i, d] = max over BQ rows. Read 8 MB, write tiny.
    for (int h = 0; h < Hq; ++h) {
      for (int i = 0; i < num_q_blocks; ++i) {
        __fp16* qs = Q_sum.data() + ((size_t)h * num_q_blocks + i) * D;
        const __fp16* qb = qp + ((size_t)h * Sq + i * BQ) * D;
        for (int d = 0; d < D; ++d) qs[d] = qb[d];
        for (int s = 1; s < BQ; ++s) {
          for (int d = 0; d < D; ++d) {
            __fp16 v = qb[s * D + d];
            if (v > qs[d]) qs[d] = v;
          }
        }
      }
    }
    // (b) K max-pool: K_summary[h, j, d] = max over BK rows. Read 8 MB on K.
    for (int h = 0; h < Hq; ++h) {
      for (int j = 0; j < num_k_blocks; ++j) {
        __fp16* ks = K_sum.data() + ((size_t)h * num_k_blocks + j) * D;
        const __fp16* kb = kp + ((size_t)h * Skv + j * BK) * D;
        for (int d = 0; d < D; ++d) ks[d] = kb[d];
        for (int s = 1; s < BK; ++s) {
          for (int d = 0; d < D; ++d) {
            __fp16 v = kb[s * D + d];
            if (v > ks[d]) ks[d] = v;
          }
        }
      }
    }
    // (c) importance = Q_summary · K_summary^T per head.
    for (int h = 0; h < Hq; ++h) {
      for (int i = 0; i < num_q_blocks; ++i) {
        const __fp16* qs = Q_sum.data() + ((size_t)h * num_q_blocks + i) * D;
        for (int j = 0; j < num_k_blocks; ++j) {
          const __fp16* ks = K_sum.data() + ((size_t)h * num_k_blocks + j) * D;
          float acc = 0.f;
          for (int d = 0; d < D; ++d) acc += (float)qs[d] * (float)ks[d];
          importance[((size_t)h * num_q_blocks + i) * num_k_blocks + j] = acc;
        }
      }
    }
    // (d) Top-k per (h, q_block). Partial sort.
    std::vector<int> idx(num_k_blocks);
    for (int h = 0; h < Hq; ++h) {
      for (int i = 0; i < num_q_blocks; ++i) {
        for (int j = 0; j < num_k_blocks; ++j) idx[j] = j;
        const float* row = importance.data() + ((size_t)h * num_q_blocks + i) * num_k_blocks;
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                          [&](int a, int b) { return row[a] > row[b]; });
        int* out = selected.data() + ((size_t)h * num_q_blocks + i) * top_k;
        for (int kk = 0; kk < top_k; ++kk) out[kk] = idx[kk];
      }
    }
  };

  const double t = timeIt(fn, runs);
  fprintf(stderr,
          "[H  %-18s]  Hq=%d Sq=%d Skv=%d BQ=%d BK=%d  num_q=%d num_k=%d top_k=%d  %.3f ms total\n",
          tag.c_str(), Hq, Sq, Skv, BQ, BK, num_q_blocks, num_k_blocks, top_k, t * 1000);
}

// ---------------------------------------------------------------------------
// Test cases — sweep the sizes that actually matter for Qwen3 prefill.
// ---------------------------------------------------------------------------

TEST_F(MemBandwidthTest, Memcpy_8MB) { runMemcpy(8 * 1024 * 1024, "8MB"); }
TEST_F(MemBandwidthTest, Memcpy_16MB) { runMemcpy(16 * 1024 * 1024, "16MB"); }
TEST_F(MemBandwidthTest, Memcpy_32MB) { runMemcpy(32 * 1024 * 1024, "32MB"); }
TEST_F(MemBandwidthTest, Memcpy_64MB) { runMemcpy(64 * 1024 * 1024, "64MB"); }

// Realistic-block sweep: BK=BQ=32 (matches HMX's 32×32 tile shape), top_k =
// num_k_blocks / 4 (a typical sparsity ratio for long-context inference).
//
// Note that smaller blocks → MORE chunks (smaller granularity of selection)
// AND larger total destination buffer (because num_q_blocks scales with
// 1/BQ). So this regime stresses the gather harder than BK=128 did.
//
//   Sq=2048, BK=32:  num_k_blocks=64, top_k=16, num_q_blocks=64
//                    chunks = 16·64·16 = 16384, each 32·128·2 = 8 KB
//                    dst    = 16·64·16·32·128·2 = 128 MB
//
//   Sq=4096, BK=32:  num_k_blocks=128, top_k=32, num_q_blocks=128
//                    chunks = 16·128·32 = 65536, each 8 KB
//                    dst    = 16·128·32·32·128·2 = 512 MB  (won't fit; trim Hq)
TEST_F(MemBandwidthTest, BlockGather_Sq128_BK32_TopK1) {
  runBlockGather(/*Hq=*/16, /*Skv=*/128, /*D=*/128, /*BK=*/32,
                 /*num_q_blocks=*/4, /*top_k=*/1, "sq128_bk32");
}
TEST_F(MemBandwidthTest, BlockGather_Sq256_BK32_TopK2) {
  runBlockGather(/*Hq=*/16, /*Skv=*/256, /*D=*/128, /*BK=*/32,
                 /*num_q_blocks=*/8, /*top_k=*/2, "sq256_bk32");
}
TEST_F(MemBandwidthTest, BlockGather_Sq512_BK32_TopK4) {
  runBlockGather(/*Hq=*/16, /*Skv=*/512, /*D=*/128, /*BK=*/32,
                 /*num_q_blocks=*/16, /*top_k=*/4, "sq512_bk32");
}
TEST_F(MemBandwidthTest, BlockGather_Sq1024_BK32_TopK8) {
  runBlockGather(/*Hq=*/16, /*Skv=*/1024, /*D=*/128, /*BK=*/32,
                 /*num_q_blocks=*/32, /*top_k=*/8, "sq1024_bk32");
}
TEST_F(MemBandwidthTest, BlockGather_Sq2048_BK32_TopK16) {
  runBlockGather(/*Hq=*/16, /*Skv=*/2048, /*D=*/128, /*BK=*/32,
                 /*num_q_blocks=*/64, /*top_k=*/16, "sq2048_bk32");
}
TEST_F(MemBandwidthTest, BlockGather_Sq4096_BK32_TopK32_Hq8) {
  // Drop Hq to 8 (Hkv-only) to keep dst under 256 MB.
  runBlockGather(/*Hq=*/8, /*Skv=*/4096, /*D=*/128, /*BK=*/32,
                 /*num_q_blocks=*/128, /*top_k=*/32, "sq4096_bk32");
}

// ---------------------------------------------------------------------------
// Multi-threaded variant of the block gather. Split the (h) axis across
// n_threads using a persistent worker pool with a condition_variable barrier
// — same overhead profile as the production IntraLayer worker. Compares
// against the single-threaded baseline at the same shape to see whether
// multi-threading helps at small Sq (smaller working sets, less likely to be
// DRAM-bound than the 128 MB cases above).
// ---------------------------------------------------------------------------
static void runBlockGatherMT(int Hq, int Skv, int D, int BK, int num_q_blocks, int top_k, int n_threads,
                             const std::string& tag) {
  const int num_k_blocks = Skv / BK;
  const size_t arranged_bytes = (size_t)Hq * num_q_blocks * top_k * BK * D * sizeof(__fp16);
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);

  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({Hq, num_q_blocks, top_k * BK, D}, kFloat16, kQNN).alloc();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* ap = K_arr.ptr<__fp16>();

  std::mt19937 rng(0xA77E0001u);
  for (size_t i = 0; i < (size_t)Hq * Skv * D; ++i) kp[i] = (__fp16)(((rng() % 1000) - 500) * 0.001f);

  std::vector<int> sel((size_t)Hq * num_q_blocks * top_k);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      std::vector<int> all(num_k_blocks);
      for (int j = 0; j < num_k_blocks; ++j) all[j] = j;
      std::shuffle(all.begin(), all.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) sel[((size_t)h * num_q_blocks + q) * top_k + kk] = all[kk];
    }
  }

  // Persistent worker pool; main thread joins as one worker.
  std::vector<std::thread> workers;
  std::atomic<int> generation{0};
  std::atomic<int> done_count{0};
  std::atomic<bool> stop{false};
  std::mutex mu;
  std::condition_variable cv_start, cv_done;

  auto worker_chunk = [&](int tid) {
    int last_gen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lk(mu);
        cv_start.wait(lk, [&]() { return generation.load() != last_gen || stop.load(); });
        if (stop.load()) return;
        last_gen = generation.load();
      }
      for (int h = tid; h < Hq; h += n_threads) {
        for (int q = 0; q < num_q_blocks; ++q) {
          __fp16* dst_qb = ap + (((size_t)h * num_q_blocks + q) * top_k * BK * D);
          for (int kk = 0; kk < top_k; ++kk) {
            int j = sel[((size_t)h * num_q_blocks + q) * top_k + kk];
            const __fp16* src = kp + ((size_t)h * Skv + j * BK) * D;
            std::memcpy(dst_qb + kk * BK * D, src, per_chunk);
          }
        }
      }
      {
        std::lock_guard<std::mutex> lk(mu);
        if (++done_count == n_threads - 1) cv_done.notify_one();
      }
    }
  };

  for (int t = 1; t < n_threads; ++t) workers.emplace_back(worker_chunk, t);

  int current_gen = 0;
  auto fn = [&] {
    {
      std::lock_guard<std::mutex> lk(mu);
      done_count.store(0);
      ++current_gen;
      generation.store(current_gen);
    }
    cv_start.notify_all();
    // Main thread does tid=0's share.
    for (int h = 0; h < Hq; h += n_threads) {
      for (int q = 0; q < num_q_blocks; ++q) {
        __fp16* dst_qb = ap + (((size_t)h * num_q_blocks + q) * top_k * BK * D);
        for (int kk = 0; kk < top_k; ++kk) {
          int j = sel[((size_t)h * num_q_blocks + q) * top_k + kk];
          const __fp16* src = kp + ((size_t)h * Skv + j * BK) * D;
          std::memcpy(dst_qb + kk * BK * D, src, per_chunk);
        }
      }
    }
    if (n_threads > 1) {
      std::unique_lock<std::mutex> lk(mu);
      cv_done.wait(lk, [&]() { return done_count.load() == n_threads - 1; });
    }
  };

  const int runs = 20;
  const double t = timeIt(fn, runs);

  stop.store(true);
  cv_start.notify_all();
  for (auto& w : workers) w.join();

  const double bw_gbs = (arranged_bytes / 1e9) / t;
  fprintf(stderr,
          "[BGmt %-18s] n_threads=%d  Hq=%d Skv=%d  dst=%zu KB  %.3f ms  %.2f GB/s write\n",
          tag.c_str(), n_threads, Hq, Skv, arranged_bytes / 1024, t * 1000, bw_gbs);
}

TEST_F(MemBandwidthTest, BlockGatherMT_Sq128_T1) { runBlockGatherMT(16, 128, 128, 32, 4, 1, 1, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq128_T2) { runBlockGatherMT(16, 128, 128, 32, 4, 1, 2, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq128_T4) { runBlockGatherMT(16, 128, 128, 32, 4, 1, 4, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq128_T8) { runBlockGatherMT(16, 128, 128, 32, 4, 1, 8, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq256_T1) { runBlockGatherMT(16, 256, 128, 32, 8, 2, 1, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq256_T2) { runBlockGatherMT(16, 256, 128, 32, 8, 2, 2, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq256_T4) { runBlockGatherMT(16, 256, 128, 32, 8, 2, 4, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq256_T8) { runBlockGatherMT(16, 256, 128, 32, 8, 2, 8, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq512_T1) { runBlockGatherMT(16, 512, 128, 32, 16, 4, 1, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq512_T2) { runBlockGatherMT(16, 512, 128, 32, 16, 4, 2, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq512_T4) { runBlockGatherMT(16, 512, 128, 32, 16, 4, 4, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq512_T8) { runBlockGatherMT(16, 512, 128, 32, 16, 4, 8, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq1024_T1) { runBlockGatherMT(16, 1024, 128, 32, 32, 8, 1, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq1024_T2) { runBlockGatherMT(16, 1024, 128, 32, 32, 8, 2, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq1024_T4) { runBlockGatherMT(16, 1024, 128, 32, 32, 8, 4, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq1024_T8) { runBlockGatherMT(16, 1024, 128, 32, 32, 8, 8, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq2048_T1) { runBlockGatherMT(16, 2048, 128, 32, 64, 16, 1, "sq2048"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq2048_T4) { runBlockGatherMT(16, 2048, 128, 32, 64, 16, 4, "sq2048"); }
TEST_F(MemBandwidthTest, BlockGatherMT_Sq2048_T8) { runBlockGatherMT(16, 2048, 128, 32, 64, 16, 8, "sq2048"); }

// ---------------------------------------------------------------------------
// OpenMP variant of the block gather. Same work distribution as runBlockGatherMT
// but uses OpenMP's runtime instead of our hand-written cv barrier. libomp's
// barrier is typically 5-10 µs vs our cv-based barrier at 30-50 µs, so this
// should win at small Sq where sync overhead dominates.
// ---------------------------------------------------------------------------
static void runBlockGatherOMP(int Hq, int Skv, int D, int BK, int num_q_blocks, int top_k, int n_threads,
                              const std::string& tag) {
  const int num_k_blocks = Skv / BK;
  const size_t arranged_bytes = (size_t)Hq * num_q_blocks * top_k * BK * D * sizeof(__fp16);
  const size_t per_chunk = (size_t)BK * D * sizeof(__fp16);

  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto K_arr = Tensor::empty({Hq, num_q_blocks, top_k * BK, D}, kFloat16, kQNN).alloc();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* ap = K_arr.ptr<__fp16>();

  std::mt19937 rng(0xA77E0001u);
  for (size_t i = 0; i < (size_t)Hq * Skv * D; ++i) kp[i] = (__fp16)(((rng() % 1000) - 500) * 0.001f);

  std::vector<int> sel((size_t)Hq * num_q_blocks * top_k);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      std::vector<int> all(num_k_blocks);
      for (int j = 0; j < num_k_blocks; ++j) all[j] = j;
      std::shuffle(all.begin(), all.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) sel[((size_t)h * num_q_blocks + q) * top_k + kk] = all[kk];
    }
  }

  // Warmup the OpenMP pool so we don't time thread-creation in the first iter.
  #pragma omp parallel num_threads(n_threads)
  { (void)0; }

  auto fn = [&] {
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int h = 0; h < Hq; ++h) {
      for (int q = 0; q < num_q_blocks; ++q) {
        __fp16* dst_qb = ap + (((size_t)h * num_q_blocks + q) * top_k * BK * D);
        for (int kk = 0; kk < top_k; ++kk) {
          int j = sel[((size_t)h * num_q_blocks + q) * top_k + kk];
          const __fp16* src = kp + ((size_t)h * Skv + j * BK) * D;
          std::memcpy(dst_qb + kk * BK * D, src, per_chunk);
        }
      }
    }
  };

  const int runs = 20;
  const double t = timeIt(fn, runs);

  const double bw_gbs = (arranged_bytes / 1e9) / t;
  fprintf(stderr,
          "[BGomp %-18s] n_threads=%d  Hq=%d Skv=%d  dst=%zu KB  %.3f ms  %.2f GB/s write\n",
          tag.c_str(), n_threads, Hq, Skv, arranged_bytes / 1024, t * 1000, bw_gbs);
}

TEST_F(MemBandwidthTest, BlockGatherOMP_Sq128_T1) { runBlockGatherOMP(16, 128, 128, 32, 4, 1, 1, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq128_T2) { runBlockGatherOMP(16, 128, 128, 32, 4, 1, 2, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq128_T4) { runBlockGatherOMP(16, 128, 128, 32, 4, 1, 4, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq128_T8) { runBlockGatherOMP(16, 128, 128, 32, 4, 1, 8, "sq128"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq256_T1) { runBlockGatherOMP(16, 256, 128, 32, 8, 2, 1, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq256_T2) { runBlockGatherOMP(16, 256, 128, 32, 8, 2, 2, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq256_T4) { runBlockGatherOMP(16, 256, 128, 32, 8, 2, 4, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq256_T8) { runBlockGatherOMP(16, 256, 128, 32, 8, 2, 8, "sq256"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq512_T1) { runBlockGatherOMP(16, 512, 128, 32, 16, 4, 1, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq512_T2) { runBlockGatherOMP(16, 512, 128, 32, 16, 4, 2, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq512_T4) { runBlockGatherOMP(16, 512, 128, 32, 16, 4, 4, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq512_T8) { runBlockGatherOMP(16, 512, 128, 32, 16, 4, 8, "sq512"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq1024_T1) { runBlockGatherOMP(16, 1024, 128, 32, 32, 8, 1, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq1024_T2) { runBlockGatherOMP(16, 1024, 128, 32, 32, 8, 2, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq1024_T4) { runBlockGatherOMP(16, 1024, 128, 32, 32, 8, 4, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq1024_T8) { runBlockGatherOMP(16, 1024, 128, 32, 32, 8, 8, "sq1024"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq2048_T1) { runBlockGatherOMP(16, 2048, 128, 32, 64, 16, 1, "sq2048"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq2048_T4) { runBlockGatherOMP(16, 2048, 128, 32, 64, 16, 4, "sq2048"); }
TEST_F(MemBandwidthTest, BlockGatherOMP_Sq2048_T8) { runBlockGatherOMP(16, 2048, 128, 32, 64, 16, 8, "sq2048"); }

TEST_F(MemBandwidthTest, Heuristic_Sq2048_BK32) {
  // num_q_blocks=64, num_k_blocks=64, top_k=16. The importance MatMul becomes
  // [Hq, 64] · [Hq, 64].T per head — still tiny.
  runHeuristic(/*Hq=*/16, /*Sq=*/2048, /*Skv=*/2048, /*D=*/128,
               /*BQ=*/32, /*BK=*/32, /*top_k=*/16, "sq2048_bk32");
}
TEST_F(MemBandwidthTest, Heuristic_Sq4096_BK32) {
  runHeuristic(16, 4096, 4096, 128, 32, 32, 32, "sq4096_bk32");
}

// ---------------------------------------------------------------------------
// Test 4 — same gather pattern as runBlockGather, but executed by QNN's
// built-in GatherNd op instead of CPU memcpy. Apples-to-apples Path A vs B
// for just the data-movement step.
//
// The graph has exactly one node: GatherNd(K, idx, batch_dims=1). Indices
// are a [Hq, num_q_blocks, top_k*BK, 1] uint32 tensor laid out so that each
// (h, q_block) selects top_k blocks of BK contiguous rows. This matches the
// CPU test's selection pattern exactly.
// ---------------------------------------------------------------------------
static void runQnnGather(const std::shared_ptr<QNNBackend>& backend, int Hq, int Skv, int D, int BK, int num_q_blocks,
                         int top_k, const std::string& tag) {
  const int num_k_blocks = Skv / BK;
  const int per_q_rows = top_k * BK;

  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto idx = Tensor::empty({Hq, num_q_blocks, per_q_rows, 1}, kUInt32, kQNN).alloc();
  auto K_arr = Tensor::empty({Hq, num_q_blocks, per_q_rows, D}, kFloat16, kQNN).alloc();

  __fp16* kp = K.ptr<__fp16>();
  std::mt19937 rng(0xA77E0001u);
  for (size_t i = 0; i < (size_t)Hq * Skv * D; ++i) kp[i] = (__fp16)(((rng() % 1000) - 500) * 0.001f);

  // Per-(h, q_block) random selection: pick top_k of num_k_blocks distinct
  // blocks, then expand to BK consecutive row indices each.
  uint32_t* ip = idx.ptr<uint32_t>();
  std::vector<int> shuffle_buf(num_k_blocks);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < num_q_blocks; ++q) {
      for (int j = 0; j < num_k_blocks; ++j) shuffle_buf[j] = j;
      std::shuffle(shuffle_buf.begin(), shuffle_buf.end(), rng);
      for (int kk = 0; kk < top_k; ++kk) {
        int blk = shuffle_buf[kk];
        for (int s = 0; s < BK; ++s) {
          size_t out_idx = (((size_t)h * num_q_blocks + q) * top_k + kk) * BK + s;
          ip[out_idx] = (uint32_t)(blk * BK + s);
        }
      }
    }
  }

  std::string graph = "gather_" + tag;
  EXPECT_NE(backend->createQnnGraph(graph), nullptr);
  EXPECT_TRUE(backend->addTensor(graph, "K", QNN_TENSOR_TYPE_APP_WRITE, K));
  EXPECT_TRUE(backend->addTensor(graph, "idx", QNN_TENSOR_TYPE_APP_WRITE, idx));
  EXPECT_TRUE(backend->addTensor(graph, "K_arr", QNN_TENSOR_TYPE_APP_READ, K_arr));

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> gnd_p = {QNNParamScalarWrapper::create<uint32_t>("batch_dims", 1u)};
  backend->graphAddNode(graph, "gather", "GatherNd", {"K", "idx"}, {"K_arr"}, {}, gnd_p, "qti.aisw");

  EXPECT_TRUE(backend->graphFinalize(graph));

  std::vector<Tensor> ins = {K, idx};
  std::vector<Tensor> outs = {K_arr};
  backend->graphExecute(graph, ins, outs);  // warmup

  const int runs = 10;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double t = std::chrono::duration<double>(t1 - t0).count() / runs;

  const size_t arranged_bytes = (size_t)Hq * num_q_blocks * per_q_rows * D * sizeof(__fp16);
  const double bw = (arranged_bytes / 1e9) / t;
  fprintf(stderr,
          "[QG %-18s]  Hq=%d Skv=%d BK=%d num_q=%d top_k=%d  per_q=%d rows  dst=%zu MB  %.2f ms  %.2f GB/s write\n",
          tag.c_str(), Hq, Skv, BK, num_q_blocks, top_k, per_q_rows, arranged_bytes / (1024 * 1024), t * 1000, bw);
}

TEST_F(MemBandwidthTest, QnnGather_Sq2048_BK32_TopK16) {
  runQnnGather(backend_, 16, 2048, 128, 32, 64, 16, "sq2048_bk32");
}
TEST_F(MemBandwidthTest, QnnGather_Sq4096_BK32_TopK32_Hq8) {
  runQnnGather(backend_, 8, 4096, 128, 32, 128, 32, "sq4096_bk32");
}
