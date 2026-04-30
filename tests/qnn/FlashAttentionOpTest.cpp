// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device integration test for the FlashAttention custom QNN op.
//
// Prerequisites — push ALL of these to the same directory before running,
// then set LD_LIBRARY_PATH=. and ADSP_LIBRARY_PATH=.;/data/local/tmp before
// executing:
//
//   libQnnHtp.so                              (QNN HTP runtime, ARM side)
//   libQnnSystem.so                           (QNN system interface)
//   libQnnLLaMAPackage_CPU.so                 (ARM/aarch64 package — registered as target CPU)
//   libQnnLLaMAPackage.so                     (hexagon-v79/v75 package — registered as target HTP)
//
// The test builds a single-node QNN graph containing only the FlashAttention
// op, runs it with random fp32 inputs, and checks that the output matches a
// naive host-side softmax(Q·Kᵀ * scale + mask) · V reference within tolerance.
// The naive reference is mathematically equivalent to the FlashAttention
// online recurrence implemented in src/ops/FlashAttention.cpp.

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
#include "QnnBackend.h"  // QNN_BACKEND_NO_ERROR

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
  if (!value || value[0] == '\0') { return 1; }

  char* end = nullptr;
  const long runs = std::strtol(value, &end, 10);
  return (end != value && runs > 0) ? static_cast<int>(runs) : 1;
}

#define STEP(msg) fprintf(stderr, "[STEP] " msg "\n")

class FlashAttentionOpTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    unbufferOutput();
    STEP("SetUpTestSuite start");

    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string adsp_path = existing && existing[0] != '\0' ? (std::string(".;/data/local/tmp;") + existing)
                                                              : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), /*overwrite=*/1);
      fprintf(stderr, "[STEP] ADSP_LIBRARY_PATH=%s\n", adsp_path.c_str());
    }

    ASSERT_TRUE(isQnnAvailable()) << "QNN runtime libraries not found";

    auto& ctx = Context::instance();

    STEP("Creating QNNBackend");
    backend_ = std::make_shared<QNNBackend>();

    const std::string cpu_package_path = findReadableFile({
        "./libQnnLLaMAPackage_CPU.so",
        "/data/local/tmp/libQnnLLaMAPackage_CPU.so",
    });
    ASSERT_FALSE(cpu_package_path.empty())
        << "Missing ARM/aarch64 op package. Push build/aarch64-android/libQnnLLaMAPackage.so "
           "as libQnnLLaMAPackage_CPU.so and register it with target CPU.";

    const std::string htp_package_path = findReadableFile({
        "./libQnnLLaMAPackage.so",
        "/data/local/tmp/libQnnLLaMAPackage.so",
        "./libQnnLLaMAPackage_HTP.so",
        "/data/local/tmp/libQnnLLaMAPackage_HTP.so",
    });
    ASSERT_FALSE(htp_package_path.empty())
        << "Missing Hexagon HTP op package. Push build/hexagon-vXX/libQnnLLaMAPackage.so "
           "as libQnnLLaMAPackage.so and include its directory in ADSP_LIBRARY_PATH.";

    auto registerPackage = [](const std::shared_ptr<QNNBackend>& backend, const std::string& package_path,
                              const char* target) {
      fprintf(stderr, "[STEP] Registering LLaMAPackage op package: target=%s path=%s\n", target, package_path.c_str());
      auto ret = backend->qnnInterface().backendRegisterOpPackage(backend->backendHandle(), package_path.c_str(),
                                                                  "LLaMAPackageInterfaceProvider", target);
      fprintf(stderr, "[STEP] backendRegisterOpPackage(%s) returned 0x%x (low16=%d)\n", target,
              static_cast<unsigned>(ret), static_cast<int>(ret & 0xFFFF));
      return ret;
    };

    auto ret = registerPackage(backend_, cpu_package_path, "CPU");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, static_cast<int>(ret & 0xFFFF))
        << "backendRegisterOpPackage CPU failed, error=" << static_cast<int>(ret & 0xFFFF);

    ret = registerPackage(backend_, htp_package_path, "HTP");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, static_cast<int>(ret & 0xFFFF))
        << "backendRegisterOpPackage HTP failed, error=" << static_cast<int>(ret & 0xFFFF);

    STEP("Creating QNN context");
    ASSERT_TRUE(backend_->createContext()) << "QNN context creation failed";

    STEP("Registering backend and allocator");
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(
        kQNN, backend_->allocator(), {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});

    STEP("Registering QNN dispatcher");
    ctx.dispatcherManager()->registerDispatcher(
        createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), QNNDispatcherOptions()));
    STEP("SetUpTestSuite complete");
  }

  static void TearDownTestSuite() { backend_.reset(); }

  static std::shared_ptr<QNNBackend> backend_;
};

std::shared_ptr<QNNBackend> FlashAttentionOpTest::backend_ = nullptr;

// Naive host-side reference: O[b,s,h,:] = softmax(s * (Q·Kᵀ + mask))[s,:] · V
// All buffers in NHWC = [B, S, H, D]. Group-query attention: H_q must be a
// multiple of H_kv; Q head h maps to KV head (h / (H_q / H_kv)).
static void naiveAttentionFp32(const float* Q, const float* K, const float* V, float* O, int B, int Sq, int Hq, int Skv,
                               int Hkv, int D, float scale, bool causal) {
  const int group = Hq / Hkv;
  std::vector<float> scores(Skv);
  for (int b = 0; b < B; ++b) {
    for (int s = 0; s < Sq; ++s) {
      const int kv_lim = (causal && Sq > 1) ? ((Skv - Sq) + s + 1) : Skv;
      for (int hq = 0; hq < Hq; ++hq) {
        const int hkv = hq / group;
        const float* qrow = Q + ((((size_t)b * Sq) + s) * Hq + hq) * D;
        float* orow = O + ((((size_t)b * Sq) + s) * Hq + hq) * D;

        // Q · Kᵀ * scale
        float row_max = -INFINITY;
        for (int j = 0; j < kv_lim; ++j) {
          const float* krow = K + ((((size_t)b * Skv) + j) * Hkv + hkv) * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += qrow[d] * krow[d];
          float s_val = dot * scale;
          scores[j] = s_val;
          if (s_val > row_max) row_max = s_val;
        }
        // softmax
        float denom = 0.f;
        for (int j = 0; j < kv_lim; ++j) {
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }
        const float inv = denom > 0.f ? (1.f / denom) : 0.f;

        // Σ p_j * V_j
        for (int d = 0; d < D; ++d) orow[d] = 0.f;
        for (int j = 0; j < kv_lim; ++j) {
          const float* vrow = V + ((((size_t)b * Skv) + j) * Hkv + hkv) * D;
          const float p = scores[j] * inv;
          for (int d = 0; d < D; ++d) orow[d] += p * vrow[d];
        }
      }
    }
  }
}

static void runFlashAttentionTest(const std::shared_ptr<QNNBackend>& backend, int B, int Sq, int Hq, int Skv, int Hkv,
                                  int D, bool causal, const std::string& graph_name) {
  fprintf(stderr, "[STEP] runFlashAttentionTest: graph=%s B=%d Sq=%d Hq=%d Skv=%d Hkv=%d D=%d causal=%d\n",
          graph_name.c_str(), B, Sq, Hq, Skv, Hkv, D, (int)causal);
  ASSERT_EQ(Hq % Hkv, 0) << "Hq must be a multiple of Hkv";
  ASSERT_EQ(D % 32, 0) << "D must be a multiple of 32 for the HVX path";

  // ------------------------------------------------------------------
  // 1. Allocate Q, K, V, O in QNN shared memory.
  // ------------------------------------------------------------------
  STEP("Allocating tensors");
  auto Q = Tensor::empty({B, Sq, Hq, D}, kFloat32, kQNN).alloc();
  auto K = Tensor::empty({B, Skv, Hkv, D}, kFloat32, kQNN).alloc();
  auto V = Tensor::empty({B, Skv, Hkv, D}, kFloat32, kQNN).alloc();
  auto O = Tensor::empty({B, Sq, Hq, D}, kFloat32, kQNN).alloc();

  // ------------------------------------------------------------------
  // 2. Fill inputs with deterministic pseudo-random values.
  // ------------------------------------------------------------------
  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t q_numel = (size_t)B * Sq * Hq * D;
  const size_t kv_numel = (size_t)B * Skv * Hkv * D;
  float* qp = Q.ptr<float>();
  float* kp = K.ptr<float>();
  float* vp = V.ptr<float>();
  for (size_t i = 0; i < q_numel; ++i) qp[i] = dist(rng);
  for (size_t i = 0; i < kv_numel; ++i) kp[i] = dist(rng);
  for (size_t i = 0; i < kv_numel; ++i) vp[i] = dist(rng);

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  // ------------------------------------------------------------------
  // 3. Compute the host reference.
  // ------------------------------------------------------------------
  std::vector<float> ref(q_numel);
  naiveAttentionFp32(qp, kp, vp, ref.data(), B, Sq, Hq, Skv, Hkv, D, scale, causal);

  // ------------------------------------------------------------------
  // 4. Build the QNN graph with one FlashAttention node.
  // ------------------------------------------------------------------
  STEP("Creating QNN graph");
  ASSERT_NE(backend->createQnnGraph(graph_name), nullptr) << "createQnnGraph failed for " << graph_name;

  ASSERT_TRUE(backend->addTensor(graph_name, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q));
  ASSERT_TRUE(backend->addTensor(graph_name, "K", QNN_TENSOR_TYPE_APP_WRITE, K));
  ASSERT_TRUE(backend->addTensor(graph_name, "V", QNN_TENSOR_TYPE_APP_WRITE, V));
  ASSERT_TRUE(backend->addTensor(graph_name, "O", QNN_TENSOR_TYPE_APP_READ, O));

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> scalar_params = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<uint32_t>("causal", causal ? 1u : 0u),
  };

  STEP("Adding FlashAttention node (validation happens here)");
  backend->graphAddNode(graph_name,
                        /*nodeName=*/"fa0",
                        /*nodeType=*/"FlashAttention",
                        /*inputs=*/{"Q", "K", "V"},
                        /*outputs=*/{"O"},
                        /*tensorParams=*/{},
                        /*scalarParams=*/scalar_params,
                        /*packageName=*/"LLaMAPackage");

  STEP("Finalizing graph");
  ASSERT_TRUE(backend->graphFinalize(graph_name)) << "graphFinalize failed";

  // ------------------------------------------------------------------
  // 5. Execute and compare.
  // ------------------------------------------------------------------
  STEP("Executing graph");
  std::vector<Tensor> inputs = {Q, K, V};
  std::vector<Tensor> outputs = {O};
  const int timing_runs = timingRunsFromEnv();
  const auto execute_start = std::chrono::steady_clock::now();
  for (int i = 0; i < timing_runs; ++i) {
    backend->graphExecute(graph_name, inputs, outputs);
  }
  const auto execute_end = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(execute_end - execute_start).count();
  fprintf(stderr,
          "[TIMING] FlashAttention graph=%s runs=%d total_ms=%.3f avg_ms=%.3f shape=[B=%d,Sq=%d,Hq=%d,Skv=%d,Hkv=%d,D=%d]\n",
          graph_name.c_str(), timing_runs, total_ms, total_ms / timing_runs, B, Sq, Hq, Skv, Hkv, D);

  // Tolerance: dot products of D random uniforms yield O(√D) magnitude scores;
  // softmax is bounded but the weighted V sum accumulates Skv terms, so absolute
  // error grows mildly with Skv. 5e-3 is well within fp32 + Hexagon HVX qf32 noise.
  const float* got = O.ptr<float>();
  size_t mismatch = 0;
  float max_abs_err = 0.f;
  for (size_t i = 0; i < q_numel; ++i) {
    const float diff = std::fabs(got[i] - ref[i]);
    if (diff > max_abs_err) max_abs_err = diff;
    if (diff > 5e-3f) {
      if (mismatch < 8) {
        fprintf(stderr, "  mismatch[%zu]: got=%f ref=%f diff=%f\n", i, got[i], ref[i], diff);
      }
      ++mismatch;
    }
  }
  fprintf(stderr, "[STEP] max_abs_err=%g, mismatches=%zu / %zu\n", max_abs_err, mismatch, q_numel);
  EXPECT_EQ(mismatch, 0u);
}

// ---------------------------------------------------------------------------
// Test cases — sweep the small/typical/decode/prefill regimes.
// ---------------------------------------------------------------------------

// Decode step (Sq == 1) — exercises the no-causal-mask path on a single Q row.
TEST_F(FlashAttentionOpTest, Float32_Decode_NoCausal) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/4, /*Skv=*/16, /*Hkv=*/4, /*D=*/32, /*causal=*/false,
                        "fa_decode_nocausal");
}

// Single tile boundary — Skv == FA_BC, exercises a single online-softmax tile.
TEST_F(FlashAttentionOpTest, Float32_SingleTile_NoCausal) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/4, /*Skv=*/64, /*Hkv=*/4, /*D=*/64, /*causal=*/false,
                        "fa_single_tile_nocausal");
}

// Multi-tile, exercises the scale-and-rebase recurrence across FA_BC tiles.
TEST_F(FlashAttentionOpTest, Float32_MultiTile_NoCausal) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/8, /*Skv=*/200, /*Hkv=*/8, /*D=*/64, /*causal=*/false,
                        "fa_multi_tile_nocausal");
}

// Prefill with causal mask — the per-row kv_lim shrinks; checks masking logic.
TEST_F(FlashAttentionOpTest, Float32_Prefill_Causal) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/16, /*Hq=*/4, /*Skv=*/16, /*Hkv=*/4, /*D=*/64, /*causal=*/true,
                        "fa_prefill_causal");
}

// GQA: Hq=8, Hkv=2 (group size 4). Shared KV across query heads.
TEST_F(FlashAttentionOpTest, Float32_GQA_Decode) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/8, /*Skv=*/96, /*Hkv=*/2, /*D=*/64, /*causal=*/false,
                        "fa_gqa_decode");
}

// Typical Qwen3-0.6B-style head dim (128) + multi-head decode.
TEST_F(FlashAttentionOpTest, Float32_HeadDim128_Decode) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/16, /*Skv=*/128, /*Hkv=*/8, /*D=*/128, /*causal=*/false,
                        "fa_headdim128_decode");
}

// Typical Qwen3-0.6B-style head dim (128) + multi-head prefill with causal masking.
TEST_F(FlashAttentionOpTest, Float32_HeadDim128_Prefill) {
  runFlashAttentionTest(backend_, /*B=*/1, /*Sq=*/128, /*Hq=*/16, /*Skv=*/128, /*Hkv=*/8, /*D=*/128, /*causal=*/true,
                        "fa_headdim128_prefill");
}