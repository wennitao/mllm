// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device integration test for the FlashAttentionInt8KV custom QNN op.
//
// Compared to FlashAttentionOpTest.cpp, this exercises the asymmetric-precision
// path where Q is fp16 and K, V are u8 (per-tensor symmetric, zp = 128). The
// host reference dequantizes K/V the same way the kernel does, then runs naive
// softmax(Q · Kᵀ * scale) · V.
//
// Prerequisites — push ALL of these to the same directory before running, then
// set LD_LIBRARY_PATH=. and ADSP_LIBRARY_PATH=.;/data/local/tmp before executing:
//
//   libQnnHtp.so                              (QNN HTP runtime, ARM side)
//   libQnnSystem.so                           (QNN system interface)
//   libQnnLLaMAPackage_CPU.so                 (ARM/aarch64 package — registered as target CPU)
//   libQnnLLaMAPackage.so                     (hexagon-v79/v75 package — registered as target HTP)

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
  if (!value || value[0] == '\0') { return 1; }
  char* end = nullptr;
  const long runs = std::strtol(value, &end, 10);
  return (end != value && runs > 0) ? static_cast<int>(runs) : 1;
}

#define STEP(msg) fprintf(stderr, "[STEP] " msg "\n")

class FlashAttentionInt8KVOpTest : public testing::Test {
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
        << "Missing ARM/aarch64 op package. Push build/aarch64-android/libQnnLLaMAPackage.so as "
           "libQnnLLaMAPackage_CPU.so and register it with target CPU.";

    const std::string htp_package_path = findReadableFile({
        "./libQnnLLaMAPackage.so",
        "/data/local/tmp/libQnnLLaMAPackage.so",
        "./libQnnLLaMAPackage_HTP.so",
        "/data/local/tmp/libQnnLLaMAPackage_HTP.so",
    });
    ASSERT_FALSE(htp_package_path.empty())
        << "Missing Hexagon HTP op package. Push build/hexagon-vXX/libQnnLLaMAPackage.so as "
           "libQnnLLaMAPackage.so and include its directory in ADSP_LIBRARY_PATH.";

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

std::shared_ptr<QNNBackend> FlashAttentionInt8KVOpTest::backend_ = nullptr;

// QNN scale/offset convention: real = (quantized_value + offset) * scale.
// For u8 with zp=128 sym: real = (q - 128) * scale -> offset = -128.
static Qnn_QuantizeParams_t makeU8SymQuantParams(float scale) {
  Qnn_QuantizeParams_t qp{};
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = scale;
  qp.scaleOffsetEncoding.offset = -128;
  return qp;
}

// Naive host-side reference. Q is fp16, K and V are int real values
// (already dequantized with their respective scales). Computes
// softmax(Q · Kᵀ * scale) · V.
static void naiveAttentionHfRef(const __fp16* Q, const float* K_real, const float* V_real, __fp16* O, int B, int Sq,
                                int Hq, int Skv, int Hkv, int D, float scale, bool causal) {
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
          const float* krow = K_real + ((((size_t)b * Skv) + j) * Hkv + hkv) * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += (float)qrow[d] * krow[d];
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

        std::vector<float> o_acc(D, 0.f);
        for (int j = 0; j < kv_lim; ++j) {
          const float* vrow = V_real + ((((size_t)b * Skv) + j) * Hkv + hkv) * D;
          const float p = scores[j] * inv;
          for (int d = 0; d < D; ++d) o_acc[d] += p * vrow[d];
        }
        for (int d = 0; d < D; ++d) orow[d] = (__fp16)o_acc[d];
      }
    }
  }
}

static void runFlashAttentionInt8KVTest(const std::shared_ptr<QNNBackend>& backend, int B, int Sq, int Hq, int Skv,
                                        int Hkv, int D, bool causal, const std::string& graph_name) {
  fprintf(stderr, "[STEP] runFlashAttentionInt8KVTest: graph=%s B=%d Sq=%d Hq=%d Skv=%d Hkv=%d D=%d causal=%d\n",
          graph_name.c_str(), B, Sq, Hq, Skv, Hkv, D, (int)causal);
  ASSERT_EQ(Hq % Hkv, 0) << "Hq must be a multiple of Hkv";
  ASSERT_EQ(D % 128, 0) << "D must be a multiple of 128 for the u8 HVX dequant path";

  // ------------------------------------------------------------------
  // 1. Allocate Q, O (fp16) and K, V (u8) in QNN shared memory.
  // ------------------------------------------------------------------
  STEP("Allocating tensors");
  auto Q = Tensor::empty({B, Sq, Hq, D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({B, Skv, Hkv, D}, kUInt8, kQNN).alloc();
  auto V = Tensor::empty({B, Skv, Hkv, D}, kUInt8, kQNN).alloc();
  auto O = Tensor::empty({B, Sq, Hq, D}, kFloat16, kQNN).alloc();

  // ------------------------------------------------------------------
  // 2. Pick scales and quantize K, V.
  //    Real K/V values drawn from U(-1, 1); pick scale = 1/127 so the full
  //    range maps to u8 [1, 255] (zp=128, sym).
  // ------------------------------------------------------------------
  const float k_scale = 1.0f / 127.0f;
  const float v_scale = 1.0f / 127.0f;

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const size_t q_numel = (size_t)B * Sq * Hq * D;
  const size_t kv_numel = (size_t)B * Skv * Hkv * D;
  __fp16* qp = Q.ptr<__fp16>();
  uint8_t* kp = K.ptr<uint8_t>();
  uint8_t* vp = V.ptr<uint8_t>();

  // Hold the dequantized real values for the host reference.
  std::vector<float> k_real(kv_numel);
  std::vector<float> v_real(kv_numel);

  for (size_t i = 0; i < q_numel; ++i) qp[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < kv_numel; ++i) {
    float r = dist(rng);
    int q_val = (int)std::lround(r / k_scale) + 128;
    if (q_val < 0) q_val = 0;
    if (q_val > 255) q_val = 255;
    kp[i] = (uint8_t)q_val;
    k_real[i] = (q_val - 128) * k_scale;
  }
  for (size_t i = 0; i < kv_numel; ++i) {
    float r = dist(rng);
    int q_val = (int)std::lround(r / v_scale) + 128;
    if (q_val < 0) q_val = 0;
    if (q_val > 255) q_val = 255;
    vp[i] = (uint8_t)q_val;
    v_real[i] = (q_val - 128) * v_scale;
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  // ------------------------------------------------------------------
  // 3. Compute the host reference (operates on the dequantized K/V).
  // ------------------------------------------------------------------
  std::vector<__fp16> ref(q_numel);
  naiveAttentionHfRef(qp, k_real.data(), v_real.data(), ref.data(), B, Sq, Hq, Skv, Hkv, D, scale, causal);

  // ------------------------------------------------------------------
  // 4. Build the QNN graph with one FlashAttentionInt8KV node.
  // ------------------------------------------------------------------
  STEP("Creating QNN graph");
  ASSERT_NE(backend->createQnnGraph(graph_name), nullptr) << "createQnnGraph failed for " << graph_name;

  ASSERT_TRUE(backend->addTensor(graph_name, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q));
  ASSERT_TRUE(backend->addTensor(graph_name, "K", QNN_TENSOR_TYPE_APP_WRITE, K, makeU8SymQuantParams(k_scale)));
  ASSERT_TRUE(backend->addTensor(graph_name, "V", QNN_TENSOR_TYPE_APP_WRITE, V, makeU8SymQuantParams(v_scale)));
  ASSERT_TRUE(backend->addTensor(graph_name, "O", QNN_TENSOR_TYPE_APP_READ, O));

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> scalar_params = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<float>("k_scale", k_scale),
      QNNParamScalarWrapper::create<float>("v_scale", v_scale),
      QNNParamScalarWrapper::create<uint32_t>("causal", causal ? 1u : 0u),
  };

  STEP("Adding FlashAttentionInt8KV node (validation happens here)");
  backend->graphAddNode(graph_name,
                        /*nodeName=*/"faq0",
                        /*nodeType=*/"FlashAttentionInt8KV",
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
  for (int i = 0; i < timing_runs; ++i) { backend->graphExecute(graph_name, inputs, outputs); }
  const auto execute_end = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(execute_end - execute_start).count();
  fprintf(stderr,
          "[TIMING] FlashAttentionInt8KV graph=%s runs=%d total_ms=%.3f avg_ms=%.3f shape=[B=%d,Sq=%d,Hq=%d,Skv=%d,Hkv=%d,D=%d]\n",
          graph_name.c_str(), timing_runs, total_ms, total_ms / timing_runs, B, Sq, Hq, Skv, Hkv, D);

  // Tolerance: u8 KV quant noise is 1/254 of the dynamic range, fp16 limits Q
  // precision, and the recurrence accumulates Skv terms. 2e-2 absolute is a
  // reasonable bar that catches algorithmic bugs without flagging quant noise.
  const __fp16* got = O.ptr<__fp16>();
  size_t mismatch = 0;
  float max_abs_err = 0.f;
  for (size_t i = 0; i < q_numel; ++i) {
    const float diff = std::fabs((float)got[i] - (float)ref[i]);
    if (diff > max_abs_err) max_abs_err = diff;
    if (diff > 2e-2f) {
      if (mismatch < 8) {
        fprintf(stderr, "  mismatch[%zu]: got=%f ref=%f diff=%f\n", i, (float)got[i], (float)ref[i], diff);
      }
      ++mismatch;
    }
  }
  fprintf(stderr, "[STEP] max_abs_err=%g, mismatches=%zu / %zu\n", max_abs_err, mismatch, q_numel);
  EXPECT_EQ(mismatch, 0u);
}

// ---------------------------------------------------------------------------
// Test cases — same regimes as FlashAttentionOpTest, sized to the kernel's
// D % 128 constraint.
// ---------------------------------------------------------------------------

TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_Decode_NoCausal) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/4, /*Skv=*/16, /*Hkv=*/4, /*D=*/128,
                              /*causal=*/false, "faq_decode_nocausal");
}

TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_SingleTile_NoCausal) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/4, /*Skv=*/64, /*Hkv=*/4, /*D=*/128,
                              /*causal=*/false, "faq_single_tile_nocausal");
}

TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_MultiTile_NoCausal) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/8, /*Skv=*/200, /*Hkv=*/8, /*D=*/128,
                              /*causal=*/false, "faq_multi_tile_nocausal");
}

TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_Prefill_Causal) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/16, /*Hq=*/4, /*Skv=*/16, /*Hkv=*/4, /*D=*/128,
                              /*causal=*/true, "faq_prefill_causal");
}

TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_GQA_Decode) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/8, /*Skv=*/96, /*Hkv=*/2, /*D=*/128,
                              /*causal=*/false, "faq_gqa_decode");
}

// Qwen3-0.6B decode shape (head dim 128, 16 heads / 8 KV heads).
TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_Qwen3_Decode) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/1, /*Hq=*/16, /*Skv=*/128, /*Hkv=*/8, /*D=*/128,
                              /*causal=*/false, "faq_qwen3_decode");
}

// Qwen3-0.6B prefill with causal masking.
TEST_F(FlashAttentionInt8KVOpTest, Hf_U8_Qwen3_Prefill) {
  runFlashAttentionInt8KVTest(backend_, /*B=*/1, /*Sq=*/128, /*Hq=*/16, /*Skv=*/128, /*Hkv=*/8, /*D=*/128,
                              /*causal=*/true, "faq_qwen3_prefill");
}
