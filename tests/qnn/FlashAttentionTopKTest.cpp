// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Top-k sparse attention vs full attention, both using QNN core ops
// (qti.aisw — get HMX). Goal: find the Sq where top-k starts winning.
//
// Top-k graph topology (per (head, query) row):
//   QK              = MatMul(Q, K, transpose_in1=1)        [Hq, Sq, Skv]
//   QK_scaled       = ElementWiseMultiply(QK, scale)
//   QK_masked       = ElementWiseAdd(QK_scaled, causal_mask)
//   (top_v, top_i)  = TopK(QK_masked, k, axis=2)            [Hq, Sq, k]
//   P               = Softmax(top_v, axis=2)                [Hq, Sq, k]
//   top_i_4d        = Reshape(top_i, [Hq, Sq, k, 1])
//   V_topk          = GatherNd(V, top_i_4d, batch_dims=1)   [Hq, Sq, k, D]
//   P_4d            = Reshape(P, [Hq, Sq, 1, k])
//   O_4d            = MatMul(P_4d, V_topk)                  [Hq, Sq, 1, D]
//   O               = Reshape(O_4d, [Hq, Sq, D])
//
// vs full attention (decomposed): MatMul + Mul + Add + Softmax + MatMul,
// no TopK / Gather. Both routed through qti.aisw → HMX-eligible matmuls.
//
// Compute breakdown (compute + memory):
//   Full     : QK matmul + AV matmul (both HMX, large GEMMs).
//   Top-k    : QK matmul (still full Skv) + TopK + Gather V (memory-bound)
//              + tiny Softmax + reduced AV matmul (Hq*Sq small mat-vecs).
// The Gather V step moves Hq·Sq·k·D fp16 = 16 MB at Sq=128 / k=32 — that
// alone is ~0.3 ms at typical DDR bandwidth, more than full's total.
// The crossover is expected somewhere around Sq ~ 1024 where the AV cost
// dominates and the Gather amortises.

#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "QnnBackend.h"

using namespace mllm;
using namespace mllm::qnn;

static bool readableFile(const std::string& p) { return access(p.c_str(), R_OK) == 0; }
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

class FlashAttentionTopKTest : public testing::Test {
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
std::shared_ptr<QNNBackend> FlashAttentionTopKTest::backend_ = nullptr;

// Host reference: full attention (no top-k). Used by the full graph.
static void naiveFullAttention(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, int Hq, int Sq, int Skv,
                               int D, float scale, bool causal) {
  std::vector<float> scores(Skv);
  for (int hq = 0; hq < Hq; ++hq) {
    for (int s = 0; s < Sq; ++s) {
      const int kv_lim = (causal && Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      const __fp16* qrow = Q + ((size_t)hq * Sq + s) * D;
      __fp16* orow = O + ((size_t)hq * Sq + s) * D;

      float row_max = -INFINITY;
      for (int j = 0; j < kv_lim; ++j) {
        const __fp16* krow = K + ((size_t)hq * Skv + j) * D;
        float dot = 0.f;
        for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)krow[d];
        scores[j] = dot * scale;
        if (scores[j] > row_max) row_max = scores[j];
      }
      float denom = 0.f;
      for (int j = 0; j < kv_lim; ++j) {
        scores[j] = std::exp(scores[j] - row_max);
        denom += scores[j];
      }
      const float inv = denom > 0.f ? (1.f / denom) : 0.f;

      std::vector<float> acc(D, 0.f);
      for (int j = 0; j < kv_lim; ++j) {
        const __fp16* vrow = V + ((size_t)hq * Skv + j) * D;
        const float p = scores[j] * inv;
        for (int d = 0; d < D; ++d) acc[d] += p * (float)vrow[d];
      }
      for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
    }
  }
}

// Host reference: top-k attention. Mirrors what the graph computes —
// softmax is over only the k highest-scoring K positions per query row,
// not over the full Skv axis. So it's a different operation than full
// attention (the tail mass is renormalised).
static void naiveTopKAttention(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, int Hq, int Sq, int Skv,
                               int D, int k, float scale, bool causal) {
  std::vector<float> scores(Skv);
  std::vector<int> idx(Skv);
  for (int hq = 0; hq < Hq; ++hq) {
    for (int s = 0; s < Sq; ++s) {
      const int kv_lim = (causal && Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      const __fp16* qrow = Q + ((size_t)hq * Sq + s) * D;
      __fp16* orow = O + ((size_t)hq * Sq + s) * D;

      // Scores with mask (for masked positions, very negative).
      for (int j = 0; j < Skv; ++j) {
        if (j < kv_lim) {
          const __fp16* krow = K + ((size_t)hq * Skv + j) * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)krow[d];
          scores[j] = dot * scale;
        } else {
          scores[j] = -1.0e4f;
        }
        idx[j] = j;
      }

      // Top-k by score (largest=true).
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int a, int b) { return scores[a] > scores[b]; });

      // Softmax over the k selected positions.
      std::vector<float> top_scores(k);
      float row_max = -INFINITY;
      for (int kk = 0; kk < k; ++kk) {
        top_scores[kk] = scores[idx[kk]];
        if (top_scores[kk] > row_max) row_max = top_scores[kk];
      }
      float denom = 0.f;
      for (int kk = 0; kk < k; ++kk) {
        top_scores[kk] = std::exp(top_scores[kk] - row_max);
        denom += top_scores[kk];
      }
      const float inv = denom > 0.f ? (1.f / denom) : 0.f;

      // Sum top-k weighted V rows.
      std::vector<float> acc(D, 0.f);
      for (int kk = 0; kk < k; ++kk) {
        const __fp16* vrow = V + ((size_t)hq * Skv + idx[kk]) * D;
        const float p = top_scores[kk] * inv;
        for (int d = 0; d < D; ++d) acc[d] += p * (float)vrow[d];
      }
      for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
    }
  }
}

struct OpResult {
  double avg_ms;
  float max_abs_err;
  size_t mismatch;
};

// Build + run a graph; one warmup, then `runs` timed iterations. Returns
// avg_ms and accuracy info against the supplied host reference.
template <typename BuildFn>
static OpResult runOneGraph(const std::shared_ptr<QNNBackend>& backend, const std::string& graph_name,
                            const std::vector<Tensor>& inputs, const Tensor& O, const std::vector<__fp16>& ref,
                            size_t numel, float tol, BuildFn build) {
  EXPECT_NE(backend->createQnnGraph(graph_name), nullptr) << graph_name;
  build(graph_name);
  EXPECT_TRUE(backend->graphFinalize(graph_name)) << "finalize failed: " << graph_name;

  std::vector<Tensor> ins = inputs;
  std::vector<Tensor> outs = {O};
  backend->graphExecute(graph_name, ins, outs);  // warmup

  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph_name, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();

  OpResult r{};
  r.avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

  const __fp16* got = O.ptr<__fp16>();
  r.mismatch = 0;
  r.max_abs_err = 0.f;
  for (size_t i = 0; i < numel; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > r.max_abs_err) r.max_abs_err = diff;
    if (diff > tol) ++r.mismatch;
  }
  return r;
}

static void runComparison(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D, int k,
                          bool causal, const std::string& tag) {
  fprintf(stderr, "[CASE] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d k=%d causal=%d\n", tag.c_str(), Sq, Hq, Skv, Hkv, D, k,
          (int)causal);
  ASSERT_EQ(Hq % Hkv, 0);
  ASSERT_LE(k, Skv);
  const int group = Hq / Hkv;

  // Allocate inputs/outputs (one set, shared between full and top-k graphs).
  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto O_full = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto O_topk = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

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
  for (int hq = 0; hq < Hq; ++hq) {
    int hkv = hq / group;
    std::memcpy(kp + (size_t)hq * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)hq * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);
  const size_t out_numel = (size_t)Hq * Sq * D;

  // Static scale tensor (scalar fp16, broadcast everywhere).
  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  // Static causal mask [1, Sq, Skv]. Non-causal cases use a zero mask so the
  // graph topology stays uniform between full and top-k.
  auto mask_t = Tensor::empty({1, Sq, Skv}, kFloat16, kQNN).alloc();
  __fp16* mp = mask_t.ptr<__fp16>();
  if (causal) {
    for (int s = 0; s < Sq; ++s) {
      int kv_lim = (Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      for (int j = 0; j < Skv; ++j) mp[s * Skv + j] = (j < kv_lim) ? (__fp16)0.0f : (__fp16)-1.0e4f;
    }
  } else {
    for (int i = 0; i < Sq * Skv; ++i) mp[i] = (__fp16)0.0f;
  }

  // Host references.
  std::vector<__fp16> ref_full(out_numel);
  std::vector<__fp16> ref_topk(out_numel);
  naiveFullAttention(qp, kp, vp, ref_full.data(), Hq, Sq, Skv, D, scale, causal);
  naiveTopKAttention(qp, kp, vp, ref_topk.data(), Hq, Sq, Skv, D, k, scale, causal);

  std::vector<Tensor> inputs = {Q, K, V};

  // -----------------------------------------------------------------------
  // Graph 1: full attention (decomposed).
  // -----------------------------------------------------------------------
  std::string g_full = "full_" + tag;
  OpResult r_full = runOneGraph(backend, g_full, inputs, O_full, ref_full, out_numel, /*tol=*/5e-2f, [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
    backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_full);
    backend->addStaticTensor(g, "scale", scale_t);
    backend->addStaticTensor(g, "mask", mask_t);

    auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QK_s_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QK_m_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QK_scaled", QNN_TENSOR_TYPE_NATIVE, QK_s_t);
    backend->addTensor(g, "QK_masked", QNN_TENSOR_TYPE_NATIVE, QK_m_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_qk = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm_qk, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QK_scaled"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "add_mask", "ElementWiseAdd", {"QK_scaled", "mask"}, {"QK_masked"}, {}, {}, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"QK_masked"}, {"P"}, {}, sm, "qti.aisw");
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");
  });
  fprintf(stderr, "[FULL ] %-22s  avg=%8.3f ms  err=%.4f miss=%zu\n", tag.c_str(), r_full.avg_ms, r_full.max_abs_err,
          r_full.mismatch);

  // -----------------------------------------------------------------------
  // Graph 2: top-k attention.
  // -----------------------------------------------------------------------
  std::string g_topk = "topk_" + tag;
  OpResult r_topk = runOneGraph(backend, g_topk, inputs, O_topk, ref_topk, out_numel, /*tol=*/5e-2f, [&](const std::string& g) {
    backend->addTensor(g, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
    backend->addTensor(g, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
    backend->addTensor(g, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
    backend->addTensor(g, "O", QNN_TENSOR_TYPE_APP_READ, O_topk);
    backend->addStaticTensor(g, "scale", scale_t);
    backend->addStaticTensor(g, "mask", mask_t);

    auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QK_s_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto QK_m_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    auto top_v_t = Tensor::empty({Hq, Sq, k}, kFloat16, kQNN);
    auto top_i_t = Tensor::empty({Hq, Sq, k}, kUInt32, kQNN);
    auto top_i_4d_t = Tensor::empty({Hq, Sq, k, 1}, kUInt32, kQNN);
    auto P_t = Tensor::empty({Hq, Sq, k}, kFloat16, kQNN);
    auto P_4d_t = Tensor::empty({Hq, Sq, 1, k}, kFloat16, kQNN);
    auto V_topk_t = Tensor::empty({Hq, Sq, k, D}, kFloat16, kQNN);
    auto O_4d_t = Tensor::empty({Hq, Sq, 1, D}, kFloat16, kQNN);
    backend->addTensor(g, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
    backend->addTensor(g, "QK_scaled", QNN_TENSOR_TYPE_NATIVE, QK_s_t);
    backend->addTensor(g, "QK_masked", QNN_TENSOR_TYPE_NATIVE, QK_m_t);
    backend->addTensor(g, "top_v", QNN_TENSOR_TYPE_NATIVE, top_v_t);
    backend->addTensor(g, "top_i", QNN_TENSOR_TYPE_NATIVE, top_i_t);
    backend->addTensor(g, "top_i_4d", QNN_TENSOR_TYPE_NATIVE, top_i_4d_t);
    backend->addTensor(g, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
    backend->addTensor(g, "P_4d", QNN_TENSOR_TYPE_NATIVE, P_4d_t);
    backend->addTensor(g, "V_topk", QNN_TENSOR_TYPE_NATIVE, V_topk_t);
    backend->addTensor(g, "O_4d", QNN_TENSOR_TYPE_NATIVE, O_4d_t);

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_qk = {QNNParamScalarWrapper::create<bool>("transpose_in1", true)};
    backend->graphAddNode(g, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm_qk, "qti.aisw");
    backend->graphAddNode(g, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QK_scaled"}, {}, {}, "qti.aisw");
    backend->graphAddNode(g, "add_mask", "ElementWiseAdd", {"QK_scaled", "mask"}, {"QK_masked"}, {}, {}, "qti.aisw");

    // TopK along Skv axis.
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> topk_p = {
        QNNParamScalarWrapper::create<uint32_t>("k", (uint32_t)k),
        QNNParamScalarWrapper::create<bool>("largest", true),
    };
    backend->graphAddNode(g, "topk", "TopK", {"QK_masked"}, {"top_v", "top_i"}, {}, topk_p, "qti.aisw");

    // Reshape top_i [Hq,Sq,k] → [Hq,Sq,k,1] so GatherNd treats it as
    // (P=1)-coordinate per output element. Reshape uses no params (target
    // shape comes from the output tensor).
    backend->graphAddNode(g, "reshape_topi", "Reshape", {"top_i"}, {"top_i_4d"}, {}, {}, "qti.aisw");

    // GatherNd(V, top_i_4d, batch_dims=1):
    //   V: [Hq, Skv, D]
    //   top_i_4d: [Hq, Sq, k, 1]   (per-batch: 1 coord into Skv axis)
    //   out: [Hq, Sq, k, D]
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> gnd_p = {QNNParamScalarWrapper::create<uint32_t>("batch_dims", 1u)};
    backend->graphAddNode(g, "gather_v", "GatherNd", {"V", "top_i_4d"}, {"V_topk"}, {}, gnd_p, "qti.aisw");

    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
                                                              QNNParamScalarWrapper::create<float>("beta", 1.0f)};
    backend->graphAddNode(g, "softmax", "Softmax", {"top_v"}, {"P"}, {}, sm, "qti.aisw");

    // Reshape P [Hq, Sq, k] → [Hq, Sq, 1, k] for the reduced AV matmul.
    backend->graphAddNode(g, "reshape_p", "Reshape", {"P"}, {"P_4d"}, {}, {}, "qti.aisw");

    // Reduced AV: per-(Hq, Sq) batched mat-vec.
    //   P_4d:    [Hq, Sq, 1, k]
    //   V_topk:  [Hq, Sq, k, D]
    //   O_4d:    [Hq, Sq, 1, D]
    backend->graphAddNode(g, "matmul_av", "MatMul", {"P_4d", "V_topk"}, {"O_4d"}, {}, {}, "qti.aisw");

    // Reshape [Hq, Sq, 1, D] → [Hq, Sq, D].
    backend->graphAddNode(g, "reshape_o", "Reshape", {"O_4d"}, {"O"}, {}, {}, "qti.aisw");
  });
  fprintf(stderr, "[TOP-K] %-22s  k=%-5d avg=%8.3f ms  err=%.4f miss=%zu\n", tag.c_str(), k, r_topk.avg_ms,
          r_topk.max_abs_err, r_topk.mismatch);

  const double speedup_full_to_topk = r_full.avg_ms / r_topk.avg_ms;
  fprintf(stderr, "[CMP  ] %-22s  full=%8.3f ms  topk(k=%d)=%8.3f ms  topk/full=%.2fx (%s)\n", tag.c_str(),
          r_full.avg_ms, k, r_topk.avg_ms, speedup_full_to_topk,
          speedup_full_to_topk > 1.0 ? "TOP-K WINS" : "FULL WINS");

  EXPECT_LE(r_full.mismatch, 0u) << "Full graph diverged from host reference";
  EXPECT_LE(r_topk.mismatch, 0u) << "Top-k graph diverged from host reference";
}

// ---------------------------------------------------------------------------
// Sweep Sq from typical (Qwen3 prefill) to long-context. K stays the same
// across the sweep so the sparsity ratio grows with Sq (more interesting
// regime — that's where top-k starts to win in theory).
//
// Hq=16, Hkv=8, D=128 matches Qwen3-0.6B. k=128 throughout.
// ---------------------------------------------------------------------------

TEST_F(FlashAttentionTopKTest, Sq128_K128) {
  // No sparsity (k == Skv). Pure overhead measurement for TopK + Gather.
  runComparison(backend_, 128, 16, 128, 8, 128, 128, true, "sq128_k128");
}
TEST_F(FlashAttentionTopKTest, Sq128_K64) {
  runComparison(backend_, 128, 16, 128, 8, 128, 64, true, "sq128_k64");
}
TEST_F(FlashAttentionTopKTest, Sq128_K32) {
  runComparison(backend_, 128, 16, 128, 8, 128, 32, true, "sq128_k32");
}
TEST_F(FlashAttentionTopKTest, Sq512_K128) {
  runComparison(backend_, 512, 16, 512, 8, 128, 128, true, "sq512_k128");
}
TEST_F(FlashAttentionTopKTest, Sq1024_K128) {
  runComparison(backend_, 1024, 16, 1024, 8, 128, 128, true, "sq1024_k128");
}
TEST_F(FlashAttentionTopKTest, Sq2048_K128) {
  runComparison(backend_, 2048, 16, 2048, 8, 128, 128, true, "sq2048_k128");
}
