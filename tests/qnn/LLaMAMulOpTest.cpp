// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device integration test for the LLaMAMul custom QNN op.
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
// Both ARM *and* DSP libraries are required for ARM prepare. Register the ARM
// package with target CPU and the hexagon package with target HTP, matching the
// qnn-net-run op package syntax documented by QAIRT.
//
// The test builds a single-node QNN graph containing only the LLaMAMul op,
// runs it with known float32 inputs, and checks that output == in0 * in1
// element-wise (which is exactly what LLaMAMul is defined to do).

#include <gtest/gtest.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "QnnBackend.h"   // QNN_BACKEND_NO_ERROR

using namespace mllm;
using namespace mllm::qnn;

static bool readableFile(const std::string& path) { return access(path.c_str(), R_OK) == 0; }

static std::string findReadableFile(const std::vector<std::string>& candidates) {
  for (const auto& candidate : candidates) {
    if (readableFile(candidate)) { return candidate; }
  }
  return {};
}

// Make every log message visible even when abort() fires before the buffer flushes.
// Called once from SetUpTestSuite before any QNN work.
static void unbufferOutput() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
}

#define STEP(msg) fprintf(stderr, "[STEP] " msg "\n")

// ---------------------------------------------------------------------------
// Test fixture — QNNBackend is expensive to create, share one instance across
// all test cases in this file.
// ---------------------------------------------------------------------------

class LLaMAMulOpTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    unbufferOutput();
    STEP("SetUpTestSuite start");

    // Ensure the DSP can find libQnnLLaMAPackage.so (hexagon variant). FastRPC
    // ADSP paths are semicolon-separated, unlike LD_LIBRARY_PATH.
    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string adsp_path = existing && existing[0] != '\0'
                                  ? (std::string(".;/data/local/tmp;") + existing)
                                  : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), /*overwrite=*/1);
      fprintf(stderr, "[STEP] ADSP_LIBRARY_PATH=%s\n", adsp_path.c_str());
    }

    ASSERT_TRUE(isQnnAvailable()) << "QNN runtime libraries not found";

    auto& ctx = Context::instance();

    // 1. Create QNNBackend — loads libQnnHtp.so, creates the runtime.
    //    Do NOT call createContext() yet; op packages must be registered first.
    STEP("Creating QNNBackend");
    backend_ = std::make_shared<QNNBackend>();
    STEP("QNNBackend created");

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

    // 2. Register the LLaMAPackage custom op package.
    //    The interface provider symbol is LLaMAPackageInterfaceProvider —
    //    see the extern "C" block at the bottom of LLaMAPackageInterface.cpp.
    //    This call must happen before contextCreate.
    auto registerPackage = [](const std::shared_ptr<QNNBackend>& backend,
                              const std::string& package_path,
                              const char* target) {
      fprintf(stderr, "[STEP] Registering LLaMAPackage op package: target=%s path=%s\n",
              target, package_path.c_str());
      auto ret = backend->qnnInterface().backendRegisterOpPackage(
          backend->backendHandle(),
          package_path.c_str(),
          "LLaMAPackageInterfaceProvider",
          target);
      fprintf(stderr, "[STEP] backendRegisterOpPackage(%s) returned 0x%x (low16=%d)\n",
              target, static_cast<unsigned>(ret), static_cast<int>(ret & 0xFFFF));
      return ret;
    };

    auto ret = registerPackage(backend_, cpu_package_path, "CPU");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, static_cast<int>(ret & 0xFFFF))
        << "backendRegisterOpPackage CPU failed, error=" << static_cast<int>(ret & 0xFFFF);

    ret = registerPackage(backend_, htp_package_path, "HTP");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, static_cast<int>(ret & 0xFFFF))
        << "backendRegisterOpPackage HTP failed, error=" << static_cast<int>(ret & 0xFFFF);

    // 3. Create the QNN context (after package registration).
    STEP("Creating QNN context");
    ASSERT_TRUE(backend_->createContext()) << "QNN context creation failed";
    STEP("QNN context created");

    // 4. Register backend and memory allocator with the engine context.
    STEP("Registering backend and allocator");
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(kQNN, backend_->allocator(),
                                           {.really_large_tensor_threshold = 0,
                                            .using_buddy_mem_pool = false});

    // 5. Register the QNN dispatcher so the engine can schedule QNN ops.
    STEP("Registering QNN dispatcher");
    ctx.dispatcherManager()->registerDispatcher(
        createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), QNNDispatcherOptions()));
    STEP("SetUpTestSuite complete");
  }

  static void TearDownTestSuite() { backend_.reset(); }

  static std::shared_ptr<QNNBackend> backend_;
};

std::shared_ptr<QNNBackend> LLaMAMulOpTest::backend_ = nullptr;

// ---------------------------------------------------------------------------
// Helper: build and execute a LLaMAMul graph, then verify the output.
// Every call uses a unique graphName so QNN does not reuse a stale graph.
// ---------------------------------------------------------------------------
static void runLLaMAMulTest(const std::shared_ptr<QNNBackend>& backend,
                             const std::vector<int32_t>& shape,
                             const std::string& graph_name) {
  fprintf(stderr, "[STEP] runLLaMAMulTest: graph=%s\n", graph_name.c_str());

  size_t numel = 1;
  for (auto d : shape) numel *= static_cast<size_t>(d);

  // ------------------------------------------------------------------
  // 1. Allocate input/output tensors in QNN shared memory (rpcmem).
  // ------------------------------------------------------------------
  STEP("Allocating tensors");
  auto in0 = Tensor::empty(shape, kFloat32, kQNN).alloc();
  STEP("in0 allocated");
  auto in1 = Tensor::empty(shape, kFloat32, kQNN).alloc();
  STEP("in1 allocated");
  auto out  = Tensor::empty(shape, kFloat32, kQNN).alloc();
  STEP("out allocated");

  // ------------------------------------------------------------------
  // 2. Fill inputs.
  // ------------------------------------------------------------------
  float* p0 = in0.ptr<float>();
  float* p1 = in1.ptr<float>();
  for (size_t i = 0; i < numel; ++i) {
    p0[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    p1[i] = static_cast<float>(i % 7)  * 0.2f - 0.3f;
  }

  // ------------------------------------------------------------------
  // 3. Build the QNN graph.
  // ------------------------------------------------------------------
  STEP("Creating QNN graph");
  ASSERT_NE(backend->createQnnGraph(graph_name), nullptr)
      << "createQnnGraph failed for " << graph_name;
  STEP("QNN graph created");

  STEP("Adding tensors");
  ASSERT_TRUE(backend->addTensor(graph_name, "in0",  QNN_TENSOR_TYPE_APP_WRITE, in0));
  ASSERT_TRUE(backend->addTensor(graph_name, "in1",  QNN_TENSOR_TYPE_APP_WRITE, in1));
  ASSERT_TRUE(backend->addTensor(graph_name, "out0", QNN_TENSOR_TYPE_APP_READ,  out));
  STEP("Tensors added");

  STEP("Adding LLaMAMul node (validation happens here)");
  backend->graphAddNode(graph_name,
                        /*nodeName=*/"mul0",
                        /*nodeType=*/"LLaMAMul",
                        /*inputs=*/{"in0", "in1"},
                        /*outputs=*/{"out0"},
                        /*tensorParams=*/{},
                        /*scalarParams=*/{},
                        /*packageName=*/"LLaMAPackage");
  STEP("LLaMAMul node added");

  STEP("Finalizing graph");
  ASSERT_TRUE(backend->graphFinalize(graph_name)) << "graphFinalize failed";
  STEP("Graph finalized");

  // ------------------------------------------------------------------
  // 4. Execute.
  // ------------------------------------------------------------------
  STEP("Executing graph");
  std::vector<Tensor> inputs  = {in0, in1};
  std::vector<Tensor> outputs = {out};
  backend->graphExecute(graph_name, inputs, outputs);
  STEP("Graph executed");

  // ------------------------------------------------------------------
  // 5. Verify output.
  // ------------------------------------------------------------------
  const float* got = out.ptr<float>();
  for (size_t i = 0; i < numel; ++i) {
    EXPECT_NEAR(got[i], p0[i] * p1[i], 1e-5f) << "mismatch at flat index " << i;
  }
  STEP("Verification complete");
}

// ---------------------------------------------------------------------------
// Test cases — different shapes exercise the scalar leftover path and the
// main vectorised loop in hvx_mul_af / hvx_mul_ahf.
// ---------------------------------------------------------------------------

// Small tensor — exercises the leftover (< 32 floats) path.
TEST_F(LLaMAMulOpTest, Float32_SmallLeftover) {
  runLLaMAMulTest(backend_, {1, 1, 1, 7}, "llamamul_small_leftover");
}

// Exactly one HVX vector (32 floats) — main loop once, no leftover.
TEST_F(LLaMAMulOpTest, Float32_OneVector) {
  runLLaMAMulTest(backend_, {1, 1, 1, 32}, "llamamul_one_vector");
}

// Typical attention head shape used in Qwen/LLaMA decode step.
TEST_F(LLaMAMulOpTest, Float32_DecodeHead) {
  runLLaMAMulTest(backend_, {1, 32, 1, 128}, "llamamul_decode_head");
}

// Larger prefill-like shape — exercises BLOCK_SIZE batching and l2fetch.
TEST_F(LLaMAMulOpTest, Float32_PrefillShape) {
  runLLaMAMulTest(backend_, {1, 32, 128, 128}, "llamamul_prefill");
}
