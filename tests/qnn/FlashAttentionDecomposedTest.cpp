// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Decomposed attention via QNN core ops: MatMul + ElementWiseMultiply +
// ElementWiseAdd + Softmax + MatMul. The point is that QNN's first-party
// MatMul gets HMX (the matrix engine) on prefill, which custom op packages
// can't access. Compare timing against V1 / V2 FlashAttention from the
// FlashAttentionCompareTest.
//
// Graph topology:
//
//   Q [Hq, Sq, D]  ─┐
//                   MatMul(transpose_in1=1) → QK [Hq, Sq, Skv]
//   K [Hq, Skv, D] ─┘
//                                                ↓
//                                  ElementWiseMultiply(QK, scale)
//                                                ↓
//                              (causal) ElementWiseAdd(_, mask[Sq, Skv])
//                                                ↓
//                                        Softmax(axis=2)
//                                                ↓
//                                  MatMul(P, V) → O [Hq, Sq, D]
//
// GQA is handled by pre-expanding K/V on the host from H_kv heads to H_q
// (each KV head replicated `group = H_q / H_kv` times). A real model would
// do this in-graph via Tile / Repeat or just store K/V as H_kv heads and
// route group-attention; the host pre-expansion here keeps the graph clean
// for benchmarking.

#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool readableFile(const std::string& path) { return access(path.c_str(), R_OK) == 0; }
static std::string findReadableFile(const std::vector<std::string>& candidates) {
  for (const auto& c : candidates) {
    if (readableFile(c)) return c;
  }
  return {};
}
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

#define STEP(msg) fprintf(stderr, "[STEP] " msg "\n")

class FlashAttentionDecomposedTest : public testing::Test {
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

    // No LLaMAPackage registration needed — this test only uses qti.aisw
    // built-in ops, which live inside libQnnHtp.so itself.
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
std::shared_ptr<QNNBackend> FlashAttentionDecomposedTest::backend_ = nullptr;

// Naive host fp16 reference. Inputs are already laid out as [Hq, Sq, D] /
// [Hq, Skv, D] (GQA pre-expanded).
static void naiveAttentionFp16Flat(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, int Hq, int Sq, int Skv,
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

static void runDecomposed(const std::shared_ptr<QNNBackend>& backend, int Sq, int Hq, int Skv, int Hkv, int D,
                          bool causal, const std::string& tag) {
  fprintf(stderr, "[CASE] %s: Sq=%d Hq=%d Skv=%d Hkv=%d D=%d causal=%d\n", tag.c_str(), Sq, Hq, Skv, Hkv, D, (int)causal);
  ASSERT_EQ(Hq % Hkv, 0);
  const int group = Hq / Hkv;

  // Allocate Q/O as [Hq, Sq, D] and K/V as [Hq, Skv, D] (GQA pre-expanded).
  auto Q = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({Hq, Skv, D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({Hq, Sq, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Q: random per Q head.
  __fp16* qp = Q.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) qp[i] = (__fp16)dist(rng);

  // K/V: generate Hkv unique heads, replicate to Hq slots.
  std::vector<__fp16> K_unique((size_t)Hkv * Skv * D);
  std::vector<__fp16> V_unique((size_t)Hkv * Skv * D);
  for (auto& x : K_unique) x = (__fp16)dist(rng);
  for (auto& x : V_unique) x = (__fp16)dist(rng);

  __fp16* kp = K.ptr<__fp16>();
  __fp16* vp = V.ptr<__fp16>();
  for (int hq = 0; hq < Hq; ++hq) {
    int hkv = hq / group;
    std::memcpy(kp + (size_t)hq * Skv * D, K_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
    std::memcpy(vp + (size_t)hq * Skv * D, V_unique.data() + (size_t)hkv * Skv * D, (size_t)Skv * D * sizeof(__fp16));
  }

  const float scale = 1.0f / std::sqrt((float)D);

  // Static scale tensor — broadcast over the [Hq, Sq, Skv] score tensor.
  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<__fp16>()[0] = (__fp16)scale;

  // Static causal mask [1, Sq, Skv] (broadcast over Hq).
  // Use -1e4 instead of -inf to avoid NaN in softmax when an entire row is masked.
  Tensor mask_t;
  if (causal) {
    mask_t = Tensor::empty({1, Sq, Skv}, kFloat16, kQNN).alloc();
    __fp16* mp = mask_t.ptr<__fp16>();
    for (int s = 0; s < Sq; ++s) {
      int kv_lim = (Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      for (int j = 0; j < Skv; ++j) mp[s * Skv + j] = (j < kv_lim) ? (__fp16)0.0f : (__fp16)-1.0e4f;
    }
  }

  // Host reference.
  std::vector<__fp16> ref((size_t)Hq * Sq * D);
  naiveAttentionFp16Flat(qp, kp, vp, ref.data(), Hq, Sq, Skv, D, scale, causal);

  // Build the QNN graph.
  std::string graph = "decomp_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);

  ASSERT_TRUE(backend->addTensor(graph, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q));
  ASSERT_TRUE(backend->addTensor(graph, "K", QNN_TENSOR_TYPE_APP_WRITE, K));
  ASSERT_TRUE(backend->addTensor(graph, "V", QNN_TENSOR_TYPE_APP_WRITE, V));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));

  ASSERT_TRUE(backend->addStaticTensor(graph, "scale", scale_t));
  if (causal) ASSERT_TRUE(backend->addStaticTensor(graph, "mask", mask_t));

  // Native intermediate tensors — QNN allocates internal storage.
  auto QK_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
  auto QK_scaled_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
  auto P_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
  ASSERT_TRUE(backend->addTensor(graph, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t));
  ASSERT_TRUE(backend->addTensor(graph, "QK_scaled", QNN_TENSOR_TYPE_NATIVE, QK_scaled_t));
  ASSERT_TRUE(backend->addTensor(graph, "P", QNN_TENSOR_TYPE_NATIVE, P_t));
  Tensor QK_masked_t;
  if (causal) {
    QK_masked_t = Tensor::empty({Hq, Sq, Skv}, kFloat16, kQNN);
    ASSERT_TRUE(backend->addTensor(graph, "QK_masked", QNN_TENSOR_TYPE_NATIVE, QK_masked_t));
  }

  // 1) QK = MatMul(Q, K, transpose_in1=true) → [Hq, Sq, Skv]
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> matmul_qk_params = {
      QNNParamScalarWrapper::create<bool>("transpose_in1", true),
  };
  backend->graphAddNode(graph, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, matmul_qk_params, "qti.aisw");

  // 2) QK_scaled = QK * scale (broadcast)
  backend->graphAddNode(graph, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QK_scaled"}, {}, {}, "qti.aisw");

  // 3) (causal) QK_masked = QK_scaled + mask  (broadcast over Hq)
  std::string softmax_input = "QK_scaled";
  if (causal) {
    backend->graphAddNode(graph, "add_mask", "ElementWiseAdd", {"QK_scaled", "mask"}, {"QK_masked"}, {}, {}, "qti.aisw");
    softmax_input = "QK_masked";
  }

  // 4) P = Softmax(_, axis=2)
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> softmax_params = {
      QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
      QNNParamScalarWrapper::create<float>("beta", 1.0f),
  };
  backend->graphAddNode(graph, "softmax", "Softmax", {softmax_input}, {"P"}, {}, softmax_params, "qti.aisw");

  // 5) O = MatMul(P, V) → [Hq, Sq, D]
  backend->graphAddNode(graph, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");

  ASSERT_TRUE(backend->graphFinalize(graph));

  // Execute (warmup + timed loop).
  std::vector<Tensor> ins = {Q, K, V};
  std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup

  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double avg_ms = total_ms / runs;

  // Tolerance: same family as the V1/V2 fp16 tests — 5e-2 absolute.
  const __fp16* got = O.ptr<__fp16>();
  size_t mismatch = 0;
  float max_abs_err = 0.f;
  for (size_t i = 0; i < (size_t)Hq * Sq * D; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > max_abs_err) max_abs_err = diff;
    if (diff > 5e-2f) ++mismatch;
  }
  fprintf(stderr, "[DECOMP] %-22s  avg=%8.3f ms (err=%.4f, miss=%zu)\n", tag.c_str(), avg_ms, max_abs_err, mismatch);
  EXPECT_EQ(mismatch, 0u) << "Decomposed output diverged from host reference";
}

// ---------------------------------------------------------------------------
// Same shapes as FlashAttentionCompareTest for direct comparison.
// ---------------------------------------------------------------------------

TEST_F(FlashAttentionDecomposedTest, F16_Decode_NoCausal) {
  runDecomposed(backend_, /*Sq=*/1, /*Hq=*/4, /*Skv=*/16, /*Hkv=*/4, /*D=*/64, /*causal=*/false, "decode_nocausal");
}
TEST_F(FlashAttentionDecomposedTest, F16_SingleTile_NoCausal) {
  runDecomposed(backend_, 1, 4, 64, 4, 64, false, "single_tile_nocausal");
}
TEST_F(FlashAttentionDecomposedTest, F16_MultiTile_NoCausal) {
  runDecomposed(backend_, 1, 8, 200, 8, 64, false, "multi_tile_nocausal");
}
TEST_F(FlashAttentionDecomposedTest, F16_Prefill_Causal) {
  runDecomposed(backend_, 16, 4, 16, 4, 64, true, "prefill_causal");
}
TEST_F(FlashAttentionDecomposedTest, F16_GQA_Decode) {
  runDecomposed(backend_, 1, 8, 96, 2, 64, false, "gqa_decode");
}
TEST_F(FlashAttentionDecomposedTest, F16_Qwen3_Decode) {
  runDecomposed(backend_, 1, 16, 128, 8, 128, false, "qwen3_decode");
}
TEST_F(FlashAttentionDecomposedTest, F16_Qwen3_Prefill) {
  runDecomposed(backend_, 128, 16, 128, 8, 128, true, "qwen3_prefill");
}
