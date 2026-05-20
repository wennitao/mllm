// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Side-by-side timing of the original FlashAttention (fp16, scalar fallback
// for the hot path) against FlashAttentionV2 (fp16, rx32 dot + rx2 V mad +
// vectorised online softmax).
//
// Both ops have the same signature, so each test case builds two QNN graphs
// from the same Q/K/V buffers and runs them under identical conditions —
// comparison-by-construction. Outputs are checked against a host fp32
// reference.
//
// Set MLLM_QNN_FA_TIMING_RUNS to override the per-graph run count
// (default 1). Recommended ≥ 50 for stable averages.

#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
  for (const auto& candidate : candidates) {
    if (readableFile(candidate)) { return candidate; }
  }
  return {};
}

static void unbufferOutput() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
}

static int timingRunsFromEnv() {
  const char* value = std::getenv("MLLM_QNN_FA_TIMING_RUNS");
  if (!value || value[0] == '\0') return 1;
  char* end = nullptr;
  const long runs = std::strtol(value, &end, 10);
  return (end != value && runs > 0) ? static_cast<int>(runs) : 1;
}

#define STEP(msg) fprintf(stderr, "[STEP] " msg "\n")

class FlashAttentionCompareTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    unbufferOutput();
    STEP("SetUpTestSuite start");

    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string adsp_path = existing && existing[0] != '\0' ? (std::string(".;/data/local/tmp;") + existing)
                                                              : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), /*overwrite=*/1);
    }

    ASSERT_TRUE(isQnnAvailable()) << "QNN runtime libraries not found";

    auto& ctx = Context::instance();
    backend_ = std::make_shared<QNNBackend>();

    const std::string cpu_pkg = findReadableFile({"./libQnnLLaMAPackage_CPU.so", "/data/local/tmp/libQnnLLaMAPackage_CPU.so"});
    ASSERT_FALSE(cpu_pkg.empty()) << "Missing libQnnLLaMAPackage_CPU.so";
    const std::string htp_pkg = findReadableFile({"./libQnnLLaMAPackage.so", "/data/local/tmp/libQnnLLaMAPackage.so"});
    ASSERT_FALSE(htp_pkg.empty()) << "Missing libQnnLLaMAPackage.so";

    auto reg = [](const std::shared_ptr<QNNBackend>& b, const std::string& path, const char* tgt) {
      auto ret = b->qnnInterface().backendRegisterOpPackage(b->backendHandle(), path.c_str(),
                                                            "LLaMAPackageInterfaceProvider", tgt);
      fprintf(stderr, "[STEP] backendRegisterOpPackage(%s) = %d\n", tgt, static_cast<int>(ret & 0xFFFF));
      return ret;
    };
    auto ret = reg(backend_, cpu_pkg, "CPU");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, static_cast<int>(ret & 0xFFFF));
    ret = reg(backend_, htp_pkg, "HTP");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, static_cast<int>(ret & 0xFFFF));

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

std::shared_ptr<QNNBackend> FlashAttentionCompareTest::backend_ = nullptr;

// Naive host fp32 reference reading fp16 inputs (matches what both ops
// compute in the limit of infinite precision).
static void naiveAttentionFp16(const __fp16* Q, const __fp16* K, const __fp16* V, __fp16* O, int B, int Sq, int Hq,
                               int Skv, int Hkv, int D, float scale, bool causal) {
  const int group = Hq / Hkv;
  std::vector<float> scores(Skv);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < Sq; ++s) {
      const int kv_lim = (causal && Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      for (int hq = 0; hq < Hq; ++hq) {
        const int hkv = hq / group;
        const __fp16* qrow = Q + ((((size_t)b * Sq) + s) * Hq + hq) * D;
        __fp16* orow = O + ((((size_t)b * Sq) + s) * Hq + hq) * D;
        float row_max = -INFINITY;
        for (int j = 0; j < kv_lim; ++j) {
          const __fp16* krow = K + ((((size_t)b * Skv) + j) * Hkv + hkv) * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += (float)qrow[d] * (float)krow[d];
          float s_val = dot * scale;
          scores[j] = s_val;
          if (s_val > row_max) row_max = s_val;
        }
        float denom = 0.f;
        for (int j = 0; j < kv_lim; ++j) {
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }
        const float inv = denom > 0.f ? (1.f / denom) : 0.f;
        std::vector<float> acc(D, 0.f);
        for (int j = 0; j < kv_lim; ++j) {
          const __fp16* vrow = V + ((((size_t)b * Skv) + j) * Hkv + hkv) * D;
          const float p = scores[j] * inv;
          for (int d = 0; d < D; ++d) acc[d] += p * (float)vrow[d];
        }
        for (int d = 0; d < D; ++d) orow[d] = (__fp16)acc[d];
      }
    }
  }
}

struct OpResult {
  double avg_ms;
  double total_ms;
  float max_abs_err;
  size_t mismatch;
};

static OpResult runOneOp(const std::shared_ptr<QNNBackend>& backend, const std::string& op_type,
                         const std::string& graph_name, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                         float scale, bool causal, const std::vector<__fp16>& ref, size_t numel, int timing_runs) {
  EXPECT_NE(backend->createQnnGraph(graph_name), nullptr);
  EXPECT_TRUE(backend->addTensor(graph_name, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q));
  EXPECT_TRUE(backend->addTensor(graph_name, "K", QNN_TENSOR_TYPE_APP_WRITE, K));
  EXPECT_TRUE(backend->addTensor(graph_name, "V", QNN_TENSOR_TYPE_APP_WRITE, V));
  EXPECT_TRUE(backend->addTensor(graph_name, "O", QNN_TENSOR_TYPE_APP_READ, O));

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> scalar_params = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<uint32_t>("causal", causal ? 1u : 0u),
  };
  backend->graphAddNode(graph_name, /*node=*/op_type + "_node", /*type=*/op_type, {"Q", "K", "V"}, {"O"}, {},
                        scalar_params, "LLaMAPackage");
  EXPECT_TRUE(backend->graphFinalize(graph_name));

  std::vector<Tensor> ins = {Q, K, V};
  std::vector<Tensor> outs = {O};
  // Warmup once (first run incurs JIT / setup overhead).
  backend->graphExecute(graph_name, ins, outs);

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < timing_runs; ++i) backend->graphExecute(graph_name, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();

  OpResult r{};
  r.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  r.avg_ms = r.total_ms / timing_runs;

  const __fp16* got = O.ptr<__fp16>();
  r.mismatch = 0;
  r.max_abs_err = 0.f;
  for (size_t i = 0; i < numel; ++i) {
    float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > r.max_abs_err) r.max_abs_err = diff;
    if (diff > 5e-2f) ++r.mismatch;
  }
  return r;
}

static void runCompare(const std::shared_ptr<QNNBackend>& backend, int B, int Sq, int Hq, int Skv, int Hkv, int D,
                       bool causal, const std::string& tag) {
  fprintf(stderr, "[CASE] %s: B=%d Sq=%d Hq=%d Skv=%d Hkv=%d D=%d causal=%d\n", tag.c_str(), B, Sq, Hq, Skv, Hkv, D,
          (int)causal);
  ASSERT_EQ(Hq % Hkv, 0);
  ASSERT_EQ(D % 64, 0) << "D must be a multiple of 64 for V2 hf rx32 dot path";

  auto Q = Tensor::empty({B, Sq, Hq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({B, Skv, Hkv, D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({B, Skv, Hkv, D}, kFloat16, kQNN).alloc();
  auto O_v1 = Tensor::empty({B, Sq, Hq, D}, kFloat16, kQNN).alloc();
  auto O_v2 = Tensor::empty({B, Sq, Hq, D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t q_numel = (size_t)B * Sq * Hq * D;
  size_t kv_numel = (size_t)B * Skv * Hkv * D;
  __fp16* qp = Q.ptr<__fp16>();
  __fp16* kp = K.ptr<__fp16>();
  __fp16* vp = V.ptr<__fp16>();
  for (size_t i = 0; i < q_numel; ++i) qp[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < kv_numel; ++i) kp[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < kv_numel; ++i) vp[i] = (__fp16)dist(rng);

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  std::vector<__fp16> ref(q_numel);
  naiveAttentionFp16(qp, kp, vp, ref.data(), B, Sq, Hq, Skv, Hkv, D, scale, causal);

  const int runs = timingRunsFromEnv();

  OpResult r1 = runOneOp(backend, "FlashAttention", "fa1_" + tag, Q, K, V, O_v1, scale, causal, ref, q_numel, runs);
  OpResult r2 = runOneOp(backend, "FlashAttentionV2", "fa2_" + tag, Q, K, V, O_v2, scale, causal, ref, q_numel, runs);

  const double speedup = r1.avg_ms / r2.avg_ms;
  fprintf(stderr,
          "[COMPARE] %-28s  V1 avg=%8.3f ms (err=%.4f, miss=%zu)   V2 avg=%8.3f ms (err=%.4f, miss=%zu)   speedup=%.2fx\n",
          tag.c_str(), r1.avg_ms, r1.max_abs_err, r1.mismatch, r2.avg_ms, r2.max_abs_err, r2.mismatch, speedup);

  // Outputs should both match the host reference. fp16 attention with random
  // inputs has ~5e-3 quant noise per element in the V sum; 5e-2 absolute
  // gives a comfortable margin and still catches algorithmic bugs.
  EXPECT_LE(r1.mismatch, 0u) << "V1 output diverged from host reference";
  EXPECT_LE(r2.mismatch, 0u) << "V2 output diverged from host reference";
}

TEST_F(FlashAttentionCompareTest, F16_Decode_NoCausal) {
  runCompare(backend_, 1, 1, 4, 16, 4, 64, false, "decode_nocausal");
}
TEST_F(FlashAttentionCompareTest, F16_SingleTile_NoCausal) {
  runCompare(backend_, 1, 1, 4, 64, 4, 64, false, "single_tile_nocausal");
}
TEST_F(FlashAttentionCompareTest, F16_MultiTile_NoCausal) {
  runCompare(backend_, 1, 1, 8, 200, 8, 64, false, "multi_tile_nocausal");
}
TEST_F(FlashAttentionCompareTest, F16_Prefill_Causal) {
  runCompare(backend_, 1, 16, 4, 16, 4, 64, true, "prefill_causal");
}
TEST_F(FlashAttentionCompareTest, F16_GQA_Decode) {
  runCompare(backend_, 1, 1, 8, 96, 2, 64, false, "gqa_decode");
}
TEST_F(FlashAttentionCompareTest, F16_Qwen3_Decode) {
  runCompare(backend_, 1, 1, 16, 128, 8, 128, false, "qwen3_decode");
}
TEST_F(FlashAttentionCompareTest, F16_Qwen3_Prefill) {
  runCompare(backend_, 1, 128, 16, 128, 8, 128, true, "qwen3_prefill");
}
