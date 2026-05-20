// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Graph-switch latency benchmark on the QNN HTP backend.
//
// The "L+1 chunk" prefill split design (modeling pre-attention, attention, and
// post-attention layer chunks as separate QNN graphs in one context) only wins
// if switching graph_handle mid-context is cheap — comparable to re-executing
// the same graph. This benchmark bounds that cost directly.
//
// Setup: N identical tiny graphs (single fp16 [32,32] x [32,32] MatMul, no
// shared weights). All N live in the same QNN context. We measure:
//   1) Steady-state single-graph dispatch  (graphExecute(g0) repeatedly)
//   2) Round-robin across N graphs         (graphExecute(g[i % N]))
// Difference between the two at a fixed inner-work size tells us the per-switch
// overhead. We sweep N over the values that matter for the prefill split:
// N = 1 (baseline), 2, 4, 8, 16, 30 (Qwen3-1.7B L=28 → L+2 = 30), 64.
//
// Tune iteration count via MLLM_QNN_GRAPHSWITCH_RUNS=<n> (default 200).

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNModel.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"

using namespace mllm;
using namespace mllm::qnn;

static int runsFromEnv() {
  const char* v = std::getenv("MLLM_QNN_GRAPHSWITCH_RUNS");
  if (!v || v[0] == '\0') return 200;
  char* end = nullptr;
  long r = std::strtol(v, &end, 10);
  return (end != v && r > 0) ? (int)r : 200;
}

class GraphSwitchLatencyTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string p = existing && existing[0] != '\0' ? (std::string(".;/data/local/tmp;") + existing) : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", p.c_str(), 1);
    }
    ASSERT_TRUE(isQnnAvailable());

    auto& ctx = Context::instance();
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
std::shared_ptr<QNNBackend> GraphSwitchLatencyTest::backend_ = nullptr;

namespace {

struct TinyGraph {
  std::string name;
  Tensor A, B, O;
};

// Build one tiny graph: [M,K] x [K,N] fp16 matmul. Distinct A/B/O per graph so
// each has its own rpcmem-backed tensors (matches the real "different layers'
// weights live in different memory" pattern).
TinyGraph buildTinyGraph(const std::shared_ptr<QNNBackend>& backend, int M, int K, int N, const std::string& tag) {
  TinyGraph g;
  g.name = "graphswitch_" + tag;
  g.A = Tensor::empty({M, K}, kFloat16, kQNN).alloc();
  g.B = Tensor::empty({K, N}, kFloat16, kQNN).alloc();
  g.O = Tensor::empty({M, N}, kFloat16, kQNN).alloc();
  std::mt19937 rng(0xBADC0DEu + std::hash<std::string>{}(tag));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* ap = g.A.ptr<__fp16>();
  __fp16* bp = g.B.ptr<__fp16>();
  for (int i = 0; i < M * K; ++i) ap[i] = (__fp16)dist(rng);
  for (int i = 0; i < K * N; ++i) bp[i] = (__fp16)dist(rng);
  EXPECT_NE(backend->createQnnGraph(g.name), nullptr);
  EXPECT_TRUE(backend->addTensor(g.name, "A", QNN_TENSOR_TYPE_APP_WRITE, g.A));
  EXPECT_TRUE(backend->addTensor(g.name, "B", QNN_TENSOR_TYPE_APP_WRITE, g.B));
  EXPECT_TRUE(backend->addTensor(g.name, "O", QNN_TENSOR_TYPE_APP_READ, g.O));
  backend->graphAddNode(g.name, "matmul", "MatMul", {"A", "B"}, {"O"}, {}, {}, "qti.aisw");
  EXPECT_TRUE(backend->graphFinalize(g.name));
  // Warmup: first execute of every graph pays an engine power-on / VTCM-acquire
  // tax (~5 ms cold). Drain it here so the measured loop reflects steady state.
  std::vector<Tensor> ins = {g.A, g.B};
  std::vector<Tensor> outs = {g.O};
  backend->graphExecute(g.name, ins, outs);
  return g;
}

double timeRoundRobin(const std::shared_ptr<QNNBackend>& backend, std::vector<TinyGraph>& graphs, int runs) {
  const int N = (int)graphs.size();
  // One more warmup pass over every graph to ensure all are cache-warm and the
  // dispatcher's per-graph state has been touched recently.
  for (auto& g : graphs) {
    std::vector<Tensor> ins = {g.A, g.B};
    std::vector<Tensor> outs = {g.O};
    backend->graphExecute(g.name, ins, outs);
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) {
    auto& g = graphs[i % N];
    std::vector<Tensor> ins = {g.A, g.B};
    std::vector<Tensor> outs = {g.O};
    backend->graphExecute(g.name, ins, outs);
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
}

void runSweepFor(const std::shared_ptr<QNNBackend>& backend, int M, int K, int N_inner, const std::string& shape_tag,
                 const std::vector<int>& Ns) {
  const int runs = runsFromEnv();
  fprintf(stderr, "\n=== GraphSwitch sweep: inner matmul [%d,%d] x [%d,%d] fp16, %d dispatches per N ===\n", M, K, K,
          N_inner, runs);
  fprintf(stderr, "  %-6s  %-10s  %-12s  %-12s\n", "N", "avg_ms", "Δ_vs_N=1_µs", "switch_cost");
  double base_ms = 0.0;
  for (int N : Ns) {
    std::vector<TinyGraph> graphs;
    graphs.reserve(N);
    for (int i = 0; i < N; ++i) {
      graphs.push_back(buildTinyGraph(backend, M, K, N_inner, shape_tag + "_N" + std::to_string(N) + "_g" + std::to_string(i)));
    }
    const double avg_ms = timeRoundRobin(backend, graphs, runs);
    if (N == 1) base_ms = avg_ms;
    const double delta_us = (avg_ms - base_ms) * 1000.0;
    // Each round-robin dispatch is "one execute"; the switch happens between
    // consecutive dispatches of different graphs. For N>=2 every dispatch is a
    // switch (i and i+1 land on different graphs), so the per-dispatch delta
    // *is* the per-switch overhead. For N=1 there is no switch.
    fprintf(stderr, "  %-6d  %-10.4f  %+12.2f  %s\n", N, avg_ms, delta_us, N == 1 ? "(baseline)" : "per-dispatch");
  }
}

}  // namespace

// Tiny inner work: [32,32] x [32,32] fp16 is 65 KFLOPs — well under 50 µs HMX
// compute, so the per-dispatch cost is dominated by RPC + switch.
TEST_F(GraphSwitchLatencyTest, TinyMatMul_32x32x32) {
  runSweepFor(backend_, 32, 32, 32, "32x32x32", {1, 2, 4, 8, 16, 30, 64});
}
TEST_F(GraphSwitchLatencyTest, TinyMatMul_64x64x64) {
  runSweepFor(backend_, 64, 64, 64, "64x64x64", {1, 2, 4, 8, 16, 30, 64});
}
// Realistic per-chunk work (QKV-proj-ish, [1024, 2048] x [2048, 2048] fp16) —
// checks whether the switch cost is still visible against meaningful inner
// compute. Capped at N=30 (the actual L+2 number for Qwen3-1.7B L=28) because
// each graph at this shape pins ~16 MB of rpcmem and N=64 would request ~1 GB.
TEST_F(GraphSwitchLatencyTest, RealisticChunk_1024x2048x2048) {
  runSweepFor(backend_, 1024, 2048, 2048, "1024x2048x2048", {1, 2, 4, 8, 16, 30});
}
