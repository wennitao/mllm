// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// GEMM latency benchmark on the QNN HTP backend. Builds a single-node graph
// containing only qti.aisw MatMul (which lowers to HMX on prefill shapes) and
// measures average wall time per execution for a sweep of (M, N, K) shapes
// that matter for Qwen3-class LLM inference.
//
// Both operands are fp16 APP_WRITE inputs (dynamic). Static-weight runs would
// give a tighter latency but a more confusing apples-to-apples comparison
// against shapes whose weight side is also a runtime tensor (e.g. Q·K^T).
//
// Shape categories:
//   - LinearPrefill : 2-D matmul [M, K] x [K, N], M = prefill seq len.
//   - LinearDecode  : 2-D matmul [1, K] x [K, N], decode step.
//   - AttnQK        : batched [Hq, Sq, D] x [Hq, Skv, D]^T  (transpose_in1=true).
//   - AttnAV        : batched [Hq, Sq, Skv] x [Hq, Skv, D].
//   - Square        : pure square [M, M] x [M, M] roofline points.
//
// Tune iteration count via MLLM_QNN_GEMM_TIMING_RUNS=<n> (default 10).

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

#include <list>

#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNModel.hpp"
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
  const char* v = std::getenv("MLLM_QNN_GEMM_TIMING_RUNS");
  if (!v || v[0] == '\0') return 10;
  char* end = nullptr;
  long r = std::strtol(v, &end, 10);
  return (end != v && r > 0) ? (int)r : 10;
}

class GemmLatencyTest : public testing::Test {
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
    // qti.aisw MatMul lives in libQnnHtp.so, no custom op package needed.
    ASSERT_TRUE(backend_->createContext());
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(
        kQNN, backend_->allocator(), {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
    ctx.dispatcherManager()->registerDispatcher(
        createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), QNNDispatcherOptions()));
  }
  static void TearDownTestSuite() { backend_.reset(); }
  static std::shared_ptr<QNNBackend> backend_;

  // Lifetime-extending storage for W4A32 static-weight buffers, per-channel
  // fp32 scales, and int32 offsets. QNN's BW_AXIS_SCALE_OFFSET encoding stores
  // raw pointers into these arrays until the graph is destroyed. std::list
  // never invalidates element addresses on push_back.
  static std::list<std::vector<float>> w4_scales_;
  static std::list<std::vector<int32_t>> w4_offsets_;
};
std::shared_ptr<QNNBackend> GemmLatencyTest::backend_ = nullptr;
std::list<std::vector<float>> GemmLatencyTest::w4_scales_;
std::list<std::vector<int32_t>> GemmLatencyTest::w4_offsets_;

// ---------------------------------------------------------------------------
// Single MatMul graph helper. Shapes A and B are full QNN tensor shapes,
// outShape is what MatMul will produce. transposeB=true forms A · B^T (useful
// for Q·K^T-style ops without an explicit transpose node).
// ---------------------------------------------------------------------------
static void runGemmTiming(const std::shared_ptr<QNNBackend>& backend,
                          const std::vector<int32_t>& aShape,
                          const std::vector<int32_t>& bShape,
                          const std::vector<int32_t>& outShape,
                          bool transposeB,
                          const std::string& tag) {
  size_t aNumel = 1;
  for (auto d : aShape) aNumel *= (size_t)d;
  size_t bNumel = 1;
  for (auto d : bShape) bNumel *= (size_t)d;

  auto A = Tensor::empty(aShape, kFloat16, kQNN).alloc();
  auto B = Tensor::empty(bShape, kFloat16, kQNN).alloc();
  auto O = Tensor::empty(outShape, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* ap = A.ptr<__fp16>();
  __fp16* bp = B.ptr<__fp16>();
  for (size_t i = 0; i < aNumel; ++i) ap[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < bNumel; ++i) bp[i] = (__fp16)dist(rng);

  const std::string graph = "gemm_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A));
  ASSERT_TRUE(backend->addTensor(graph, "B", QNN_TENSOR_TYPE_APP_WRITE, B));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_params;
  if (transposeB) {
    mm_params.push_back(QNNParamScalarWrapper::create<bool>("transpose_in1", true));
  }
  backend->graphAddNode(graph, "matmul", "MatMul", {"A", "B"}, {"O"}, {}, mm_params, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(graph));

  std::vector<Tensor> ins = {A, B};
  std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup

  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

  // Compute throughput: 2 * (#output elements) * K  FLOPs per matmul.
  size_t outNumel = 1;
  for (auto d : outShape) outNumel *= (size_t)d;
  const int K = transposeB ? bShape.back() : bShape[bShape.size() - 2];
  const double flops = 2.0 * (double)outNumel * (double)K;
  const double gflops_s = flops / (avg_ms * 1.0e-3) / 1.0e9;

  auto shapeStr = [](const std::vector<int32_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      r += std::to_string(s[i]);
      if (i + 1 < s.size()) r += ",";
    }
    return r + "]";
  };
  fprintf(stderr,
          "[GEMM %-32s]  A%s x B%s%s -> O%s  K=%d  avg=%9.4f ms  %8.2f GFLOP/s\n",
          tag.c_str(), shapeStr(aShape).c_str(), shapeStr(bShape).c_str(), transposeB ? "^T" : "",
          shapeStr(outShape).c_str(), K, avg_ms, gflops_s);
}

// ---------------------------------------------------------------------------
// Convenience wrappers.
// 2-D linear: [M, K] x [K, N] -> [M, N]
// ---------------------------------------------------------------------------
static void runLinear(const std::shared_ptr<QNNBackend>& backend, int M, int N, int K, const std::string& tag) {
  runGemmTiming(backend, {M, K}, {K, N}, {M, N}, /*transposeB=*/false, tag);
}
// Batched attention QK^T: [Hq, Sq, D] x [Hq, Skv, D]^T -> [Hq, Sq, Skv]
static void runAttnQK(const std::shared_ptr<QNNBackend>& backend, int Hq, int Sq, int Skv, int D,
                      const std::string& tag) {
  runGemmTiming(backend, {Hq, Sq, D}, {Hq, Skv, D}, {Hq, Sq, Skv}, /*transposeB=*/true, tag);
}
// Batched attention P·V: [Hq, Sq, Skv] x [Hq, Skv, D] -> [Hq, Sq, D]
static void runAttnAV(const std::shared_ptr<QNNBackend>& backend, int Hq, int Sq, int Skv, int D,
                      const std::string& tag) {
  runGemmTiming(backend, {Hq, Sq, Skv}, {Hq, Skv, D}, {Hq, Sq, D}, /*transposeB=*/false, tag);
}

// ---------------------------------------------------------------------------
// XAttention block-selection SCORE CORE on HTP: the reduced antidiagonal
// scoring matmul + softmax, matching the CPU/GPU "core" in
// examples/block_selection/bench_block_selection.cpp. Per head:
//   M = Qr @ Kr^T   ([Hq,Lr,SD] x [Hq,Lr,SD]^T -> [Hq,Lr,Lr], SD = S*d)
//   O = softmax(M)  over the last (key) axis
// One graph, two qti.aisw nodes; M is a NATIVE intermediate. Both inputs are
// dynamic fp16 (Q and K are both runtime tensors here). Reports matmul GFLOP/s
// from the Qr@Kr^T MAC count (softmax folded into the same wall time).
// ---------------------------------------------------------------------------
static void runBlockSelectScore(const std::shared_ptr<QNNBackend>& backend, int Hq, int Lr, int SD,
                                const std::string& tag) {
  auto A = Tensor::empty({Hq, Lr, SD}, kFloat16, kQNN).alloc();  // Qr
  auto B = Tensor::empty({Hq, Lr, SD}, kFloat16, kQNN).alloc();  // Kr
  auto M = Tensor::empty({Hq, Lr, Lr}, kFloat16, kQNN).alloc();  // scores (native)
  auto O = Tensor::empty({Hq, Lr, Lr}, kFloat16, kQNN).alloc();  // softmaxed (read)

  std::mt19937 rng(0xB10C5E1u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* ap = A.ptr<__fp16>();
  __fp16* bp = B.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * Lr * SD; ++i) ap[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < (size_t)Hq * Lr * SD; ++i) bp[i] = (__fp16)dist(rng);

  const std::string graph = "blocksel_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A));
  ASSERT_TRUE(backend->addTensor(graph, "B", QNN_TENSOR_TYPE_APP_WRITE, B));
  ASSERT_TRUE(backend->addTensor(graph, "M", QNN_TENSOR_TYPE_NATIVE, M));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_params;
  mm_params.push_back(QNNParamScalarWrapper::create<bool>("transpose_in1", true));
  backend->graphAddNode(graph, "matmul", "MatMul", {"A", "B"}, {"M"}, {}, mm_params, "qti.aisw");

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm_params;
  sm_params.push_back(QNNParamScalarWrapper::create("axis", (uint32_t)2));  // last axis of [Hq,Lr,Lr]
  sm_params.push_back(QNNParamScalarWrapper::create("beta", 1.0f));
  backend->graphAddNode(graph, "softmax", "Softmax", {"M"}, {"O"}, {}, sm_params, "qti.aisw");

  ASSERT_TRUE(backend->graphFinalize(graph));

  std::vector<Tensor> ins = {A, B};
  std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup

  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

  const double macs = (double)Hq * Lr * Lr * SD;
  const double gflops_s = 2.0 * macs / (avg_ms * 1.0e-3) / 1.0e9;
  fprintf(stderr, "[BLKSEL %-16s]  Hq=%d Lr=%d SD=%d -> O[%d,%d,%d]  avg=%9.4f ms  %8.2f GFLOP/s\n", tag.c_str(), Hq,
          Lr, SD, Hq, Lr, Lr, avg_ms, gflops_s);
}

// ---------------------------------------------------------------------------
// PER-QB granularity on HTP: instead of one [Hq,Lr,Lr] matmul/layer (big-batch),
// this issues the model's ACTUAL per-qb scoring — num_qb separate MatMul+Softmax
// dispatches of shape M=BQr(=BQ/S), N=histr(qb)=qb*BKr, K=SD — and reports the
// SUMMED per-layer time. Tests whether per-dispatch overhead × num_qb beats the
// single big-batch dispatch (which does ~2× the FLOPs but one launch).
// ---------------------------------------------------------------------------
static void runBlockSelectScorePerQb(const std::shared_ptr<QNNBackend>& backend, int Hq, int Lr, int SD,
                                     const std::string& tag) {
  const int D = 128, S = SD / D, BQ = 32, BK = 32;
  const int BQr = BQ / S, BKr = BK / S;
  const int num_qb = (Lr * S) / BQ;  // = L / BQ
  const int runs = timingRunsFromEnv();
  std::mt19937 rng(0xB10C5E1u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  double total_ms = 0.0, total_macs = 0.0;
  for (int qb = 1; qb < num_qb; ++qb) {
    const int M = BQr, N = qb * BKr, K = SD;
    auto A = Tensor::empty({Hq, M, K}, kFloat16, kQNN).alloc();
    auto B = Tensor::empty({Hq, N, K}, kFloat16, kQNN).alloc();
    auto Mt = Tensor::empty({Hq, M, N}, kFloat16, kQNN).alloc();
    auto O = Tensor::empty({Hq, M, N}, kFloat16, kQNN).alloc();
    __fp16* ap = A.ptr<__fp16>();
    for (size_t i = 0; i < (size_t)Hq * M * K; ++i) ap[i] = (__fp16)dist(rng);
    __fp16* bp = B.ptr<__fp16>();
    for (size_t i = 0; i < (size_t)Hq * N * K; ++i) bp[i] = (__fp16)dist(rng);
    const std::string graph = "blkselqb_" + tag + "_" + std::to_string(qb);
    ASSERT_NE(backend->createQnnGraph(graph), nullptr);
    ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A));
    ASSERT_TRUE(backend->addTensor(graph, "B", QNN_TENSOR_TYPE_APP_WRITE, B));
    ASSERT_TRUE(backend->addTensor(graph, "M", QNN_TENSOR_TYPE_NATIVE, Mt));
    ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_params;
    mm_params.push_back(QNNParamScalarWrapper::create<bool>("transpose_in1", true));
    backend->graphAddNode(graph, "matmul", "MatMul", {"A", "B"}, {"M"}, {}, mm_params, "qti.aisw");
    std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm_params;
    sm_params.push_back(QNNParamScalarWrapper::create("axis", (uint32_t)2));
    sm_params.push_back(QNNParamScalarWrapper::create("beta", 1.0f));
    backend->graphAddNode(graph, "softmax", "Softmax", {"M"}, {"O"}, {}, sm_params, "qti.aisw");
    ASSERT_TRUE(backend->graphFinalize(graph));
    std::vector<Tensor> ins = {A, B};
    std::vector<Tensor> outs = {O};
    backend->graphExecute(graph, ins, outs);  // warmup
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
    const auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
    total_macs += (double)Hq * M * N * K;
  }
  const double gflops_s = 2.0 * total_macs / (total_ms * 1.0e-3) / 1.0e9;
  fprintf(stderr,
          "[BLKSEL-PERQB %-10s] Hq=%d Lr=%d SD=%d num_qb=%d BQr=%d -> per-layer SUM avg=%9.4f ms (%d dispatches) %8.2f "
          "GFLOP/s\n",
          tag.c_str(), Hq, Lr, SD, num_qb, BQr, total_ms, num_qb - 1, gflops_s);
}

// ---------------------------------------------------------------------------
// W4A16: per-channel int4 static weight × fp16 activation. "Naive" path — no
// block scaling, just one fp32 scale per output channel.
//
// The test was written for W4A32 (fp32 activation), but HTP MatMul rejects
// the fp32-activation × int4-weight signature at graph-finalize time. fp16
// activation is the only combination this device accepts, and it's also what
// production Qwen3-NPU runs anyway. The W4 win lives in weight bandwidth, not
// in the compute path — HMX has no native int4-mac, so the weight is
// dequantized to fp16 inline before the matmul.
//
// QNN HTP also refuses native UFIXED_POINT_4 / SFIXED_POINT_4 tensors with
// the plain axis-scale-offset encoding. The supported path for sub-byte
// weights is BW_AXIS_SCALE_OFFSET: the tensor's storage dtype is
// UFIXED_POINT_8 (one nibble per byte, low 4 bits) and `bitwidth=4` lives in
// the encoding struct.
//
// The packed buffer is filled with zeros — numerical correctness isn't the
// point, the HMX dispatch cost depends on shape + encoding, not data.
// ---------------------------------------------------------------------------
static void runGemmW4A32(const std::shared_ptr<QNNBackend>& backend, int M, int K, int N,
                        std::list<std::vector<float>>& w4_scales_storage,
                        std::list<std::vector<int32_t>>& w4_offsets_storage,
                        const std::string& tag) {
  auto A = Tensor::empty({M, K}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({M, N}, kFloat16, kQNN).alloc();
  // Storage for the int4 weight is one byte per nibble — wasteful but matches
  // what QNN expects when bitwidth=4 is set in the encoding struct. The kQNN
  // allocator gives us rpcmem; the static weight doesn't strictly need that,
  // but it keeps the test using the same allocator throughout.
  auto W = Tensor::empty({K, N}, kInt8, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* ap = A.ptr<__fp16>();
  for (int i = 0; i < M * K; ++i) ap[i] = (__fp16)dist(rng);

  // Random non-zero nibbles in the low 4 bits — needed so the QNN compiler
  // doesn't fold the matmul away to zero. (At fp16 we got away with random
  // input here; the W4 path is more aggressive about constant-prop.)
  uint8_t* wp = W.ptr<uint8_t>();
  for (size_t i = 0; i < (size_t)K * N; ++i) wp[i] = (uint8_t)((rng() & 0x0F));

  // Per-channel scale + offset arrays for BW_AXIS_SCALE_OFFSET. axis = 1 (N).
  auto& scales = w4_scales_storage.emplace_back(N, 1.0f / 8.0f);
  auto& offsets = w4_offsets_storage.emplace_back(N, /*int4 symmetric offset=*/-8);

  const std::string graph = "gemm_w4a32_" + tag;
  auto model = backend->createQnnGraph(graph);
  ASSERT_NE(model, nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));

  // Build the int4 weight wrapper. Storage dtype is UFIXED_POINT_8, but the
  // BW_AXIS_SCALE_OFFSET encoding tells QNN to interpret the low 4 bits as
  // the actual quantized value.
  auto w_wrapper = std::make_shared<QNNTensorWrapper>(
      /*name=*/"W",
      /*type=*/QNN_TENSOR_TYPE_STATIC,
      /*dataType=*/QNN_DATATYPE_UFIXED_POINT_8,
      /*dimensions=*/std::vector<uint32_t>{static_cast<uint32_t>(K), static_cast<uint32_t>(N)},
      /*quantize=*/DEFAULT_QUANTIZE_PARAMS);

  Qnn_Tensor_t* native = w_wrapper->getNativeTensor();
  native->v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  native->v2.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  native->v2.quantizeParams.bwAxisScaleOffsetEncoding = Qnn_BwAxisScaleOffset_t{
      .bitwidth = 4u,
      .axis = 1,
      .numElements = static_cast<uint32_t>(N),
      .scales = scales.data(),
      .offsets = offsets.data(),
  };
  native->v2.clientBuf.data = W.ptr<void>();
  native->v2.clientBuf.dataSize = static_cast<uint32_t>((size_t)K * N);

  ASSERT_EQ(model->addTensorWrapper(w_wrapper), MODEL_NO_ERROR);

  backend->graphAddNode(graph, "matmul", "MatMul", {"A", "W"}, {"O"}, {}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(graph));

  std::vector<Tensor> ins = {A};
  std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup

  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

  const double flops = 2.0 * (double)M * N * K;
  const double gflops_s = flops / (avg_ms * 1.0e-3) / 1.0e9;
  fprintf(stderr,
          "[W4A16 %-31s]  A[%d,%d]fp16 x W[%d,%d]i4 -> O[%d,%d]fp16  K=%d  avg=%9.4f ms  %8.2f GFLOP/s\n",
          tag.c_str(), M, K, K, N, M, N, K, avg_ms, gflops_s);
}

// ===========================================================================
// LLM Linear projections — Qwen3-1.7B  (hidden=2048, intermediate=6144,
//                                       Hq=16, Hkv=8, head_dim=128)
// Per-head projection widths:
//   q_proj: hidden -> Hq*D     = 2048 -> 2048
//   k_proj: hidden -> Hkv*D    = 2048 -> 1024
//   v_proj: hidden -> Hkv*D    = 2048 -> 1024
//   o_proj: hidden -> hidden   = 2048 -> 2048
//   gate/up: hidden -> 6144
//   down  : 6144  -> 2048
// ===========================================================================

// ----- Prefill (M = sequence length) --------------------------------------
TEST_F(GemmLatencyTest, Qwen17B_Prefill512_QProj)  { runLinear(backend_,  512, 2048, 2048, "1p7B_pf512_qproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill512_KVProj) { runLinear(backend_,  512, 1024, 2048, "1p7B_pf512_kvproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill512_OProj)  { runLinear(backend_,  512, 2048, 2048, "1p7B_pf512_oproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill512_Gate)   { runLinear(backend_,  512, 6144, 2048, "1p7B_pf512_gate"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill512_Down)   { runLinear(backend_,  512, 2048, 6144, "1p7B_pf512_down"); }

TEST_F(GemmLatencyTest, Qwen17B_Prefill1024_QProj)  { runLinear(backend_, 1024, 2048, 2048, "1p7B_pf1024_qproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill1024_KVProj) { runLinear(backend_, 1024, 1024, 2048, "1p7B_pf1024_kvproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill1024_OProj)  { runLinear(backend_, 1024, 2048, 2048, "1p7B_pf1024_oproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill1024_Gate)   { runLinear(backend_, 1024, 6144, 2048, "1p7B_pf1024_gate"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill1024_Down)   { runLinear(backend_, 1024, 2048, 6144, "1p7B_pf1024_down"); }

TEST_F(GemmLatencyTest, Qwen17B_Prefill2048_QProj)  { runLinear(backend_, 2048, 2048, 2048, "1p7B_pf2048_qproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill2048_KVProj) { runLinear(backend_, 2048, 1024, 2048, "1p7B_pf2048_kvproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill2048_OProj)  { runLinear(backend_, 2048, 2048, 2048, "1p7B_pf2048_oproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill2048_Gate)   { runLinear(backend_, 2048, 6144, 2048, "1p7B_pf2048_gate"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill2048_Down)   { runLinear(backend_, 2048, 2048, 6144, "1p7B_pf2048_down"); }

TEST_F(GemmLatencyTest, Qwen17B_Prefill4096_QProj)  { runLinear(backend_, 4096, 2048, 2048, "1p7B_pf4096_qproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill4096_KVProj) { runLinear(backend_, 4096, 1024, 2048, "1p7B_pf4096_kvproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill4096_OProj)  { runLinear(backend_, 4096, 2048, 2048, "1p7B_pf4096_oproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill4096_Gate)   { runLinear(backend_, 4096, 6144, 2048, "1p7B_pf4096_gate"); }
TEST_F(GemmLatencyTest, Qwen17B_Prefill4096_Down)   { runLinear(backend_, 4096, 2048, 6144, "1p7B_pf4096_down"); }

// ----- Decode (M = 1) -----------------------------------------------------
TEST_F(GemmLatencyTest, Qwen17B_Decode_QProj)  { runLinear(backend_, 1, 2048, 2048, "1p7B_dec_qproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Decode_KVProj) { runLinear(backend_, 1, 1024, 2048, "1p7B_dec_kvproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Decode_OProj)  { runLinear(backend_, 1, 2048, 2048, "1p7B_dec_oproj"); }
TEST_F(GemmLatencyTest, Qwen17B_Decode_Gate)   { runLinear(backend_, 1, 6144, 2048, "1p7B_dec_gate"); }
TEST_F(GemmLatencyTest, Qwen17B_Decode_Down)   { runLinear(backend_, 1, 2048, 6144, "1p7B_dec_down"); }

// ===========================================================================
// Attention matmuls — per-layer Q·K^T and P·V.
//
// Qwen3-1.7B uses Hq=16, head_dim=128. We benchmark with Hq pre-expanded
// (GQA already replicated) since the QNN graph sees [Hq, *, *] tensors.
// ===========================================================================

TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill512_QK)  { runAttnQK(backend_, 16,  512,  512, 128, "1p7B_pf512_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill512_AV)  { runAttnAV(backend_, 16,  512,  512, 128, "1p7B_pf512_av"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill1024_QK) { runAttnQK(backend_, 16, 1024, 1024, 128, "1p7B_pf1024_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill1024_AV) { runAttnAV(backend_, 16, 1024, 1024, 128, "1p7B_pf1024_av"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill2048_QK) { runAttnQK(backend_, 16, 2048, 2048, 128, "1p7B_pf2048_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill2048_AV) { runAttnAV(backend_, 16, 2048, 2048, 128, "1p7B_pf2048_av"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill4096_QK) { runAttnQK(backend_, 16, 4096, 4096, 128, "1p7B_pf4096_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_Prefill4096_AV) { runAttnAV(backend_, 16, 4096, 4096, 128, "1p7B_pf4096_av"); }

// Decode attention (Sq=1, Skv=context length). Memory-bound — K^T fetches
// dominate. Q·K^T is [16, 1, 128] x [16, Skv, 128]^T → [16, 1, Skv].
TEST_F(GemmLatencyTest, Qwen17B_Attn_DecodeCtx512_QK)  { runAttnQK(backend_, 16, 1,  512, 128, "1p7B_dec512_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_DecodeCtx512_AV)  { runAttnAV(backend_, 16, 1,  512, 128, "1p7B_dec512_av"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_DecodeCtx2048_QK) { runAttnQK(backend_, 16, 1, 2048, 128, "1p7B_dec2048_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_DecodeCtx2048_AV) { runAttnAV(backend_, 16, 1, 2048, 128, "1p7B_dec2048_av"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_DecodeCtx4096_QK) { runAttnQK(backend_, 16, 1, 4096, 128, "1p7B_dec4096_qk"); }
TEST_F(GemmLatencyTest, Qwen17B_Attn_DecodeCtx4096_AV) { runAttnAV(backend_, 16, 1, 4096, 128, "1p7B_dec4096_av"); }

// ===========================================================================
// XAttention block-selection score core (matmul+softmax) at the model's
// Hq=16, d=128, stride S=8 (SD = S*d = 1024), Lr = L/S. Compare against the
// ARM CPU / Adreno GPU numbers in docs/qnn_backend/block_selection.md.
// ===========================================================================
TEST_F(GemmLatencyTest, BlockSelScore_S8_L256)  { runBlockSelectScore(backend_, 16,  32, 1024, "s8_L256"); }
TEST_F(GemmLatencyTest, BlockSelScore_S8_L512)  { runBlockSelectScore(backend_, 16,  64, 1024, "s8_L512"); }
TEST_F(GemmLatencyTest, BlockSelScore_S8_L1024) { runBlockSelectScore(backend_, 16, 128, 1024, "s8_L1024"); }
TEST_F(GemmLatencyTest, BlockSelScore_S8_L2048) { runBlockSelectScore(backend_, 16, 256, 1024, "s8_L2048"); }
TEST_F(GemmLatencyTest, BlockSelScore_S8_L4096) { runBlockSelectScore(backend_, 16, 512, 1024, "s8_L4096"); }

// Per-qb granularity (num_qb separate dispatches/layer) vs the big-batch above.
TEST_F(GemmLatencyTest, BlockSelScorePerQb_S8_L1024) { runBlockSelectScorePerQb(backend_, 16, 128, 1024, "s8_L1024"); }
TEST_F(GemmLatencyTest, BlockSelScorePerQb_S8_L2048) { runBlockSelectScorePerQb(backend_, 16, 256, 1024, "s8_L2048"); }

// ===========================================================================
// Square sweep — roofline reference, plus an upper bound on what the HMX
// matrix engine can actually deliver at each tile size.
// ===========================================================================

TEST_F(GemmLatencyTest, Square_256)  { runLinear(backend_,  256,  256,  256, "square_256"); }
TEST_F(GemmLatencyTest, Square_512)  { runLinear(backend_,  512,  512,  512, "square_512"); }
TEST_F(GemmLatencyTest, Square_1024) { runLinear(backend_, 1024, 1024, 1024, "square_1024"); }
TEST_F(GemmLatencyTest, Square_2048) { runLinear(backend_, 2048, 2048, 2048, "square_2048"); }
TEST_F(GemmLatencyTest, Square_4096) { runLinear(backend_, 4096, 4096, 4096, "square_4096"); }

// ===========================================================================
// W4A16 — same Qwen3-1.7B linear shapes, int4 static weight with per-channel
// fp32 scale + fp16 activation. No block scaling (LPBQ is a separate path).
// HTP MatMul rejects fp32×int4 outright, so the activation has to be fp16
// here even though the user asked for A32. Same M sweep as the fp16 prefill
// block above, so rows line up for direct comparison.
// ===========================================================================

// --- Prefill ---------------------------------------------------------------
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill512_QProj)  { runGemmW4A32(backend_,  512, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf512_qproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill512_KVProj) { runGemmW4A32(backend_,  512, 2048, 1024, w4_scales_, w4_offsets_, "1p7B_pf512_kvproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill512_OProj)  { runGemmW4A32(backend_,  512, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf512_oproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill512_Gate)   { runGemmW4A32(backend_,  512, 2048, 6144, w4_scales_, w4_offsets_, "1p7B_pf512_gate"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill512_Down)   { runGemmW4A32(backend_,  512, 6144, 2048, w4_scales_, w4_offsets_, "1p7B_pf512_down"); }

TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill1024_QProj)  { runGemmW4A32(backend_, 1024, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf1024_qproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill1024_KVProj) { runGemmW4A32(backend_, 1024, 2048, 1024, w4_scales_, w4_offsets_, "1p7B_pf1024_kvproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill1024_OProj)  { runGemmW4A32(backend_, 1024, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf1024_oproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill1024_Gate)   { runGemmW4A32(backend_, 1024, 2048, 6144, w4_scales_, w4_offsets_, "1p7B_pf1024_gate"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill1024_Down)   { runGemmW4A32(backend_, 1024, 6144, 2048, w4_scales_, w4_offsets_, "1p7B_pf1024_down"); }

TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill2048_QProj)  { runGemmW4A32(backend_, 2048, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf2048_qproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill2048_KVProj) { runGemmW4A32(backend_, 2048, 2048, 1024, w4_scales_, w4_offsets_, "1p7B_pf2048_kvproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill2048_OProj)  { runGemmW4A32(backend_, 2048, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf2048_oproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill2048_Gate)   { runGemmW4A32(backend_, 2048, 2048, 6144, w4_scales_, w4_offsets_, "1p7B_pf2048_gate"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill2048_Down)   { runGemmW4A32(backend_, 2048, 6144, 2048, w4_scales_, w4_offsets_, "1p7B_pf2048_down"); }

TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill4096_QProj)  { runGemmW4A32(backend_, 4096, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf4096_qproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill4096_KVProj) { runGemmW4A32(backend_, 4096, 2048, 1024, w4_scales_, w4_offsets_, "1p7B_pf4096_kvproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill4096_OProj)  { runGemmW4A32(backend_, 4096, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_pf4096_oproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill4096_Gate)   { runGemmW4A32(backend_, 4096, 2048, 6144, w4_scales_, w4_offsets_, "1p7B_pf4096_gate"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Prefill4096_Down)   { runGemmW4A32(backend_, 4096, 6144, 2048, w4_scales_, w4_offsets_, "1p7B_pf4096_down"); }

// --- Decode (M=1) — bandwidth-bound; the W4 win lives here -----------------
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Decode_QProj)  { runGemmW4A32(backend_, 1, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_dec_qproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Decode_KVProj) { runGemmW4A32(backend_, 1, 2048, 1024, w4_scales_, w4_offsets_, "1p7B_dec_kvproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Decode_OProj)  { runGemmW4A32(backend_, 1, 2048, 2048, w4_scales_, w4_offsets_, "1p7B_dec_oproj"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Decode_Gate)   { runGemmW4A32(backend_, 1, 2048, 6144, w4_scales_, w4_offsets_, "1p7B_dec_gate"); }
TEST_F(GemmLatencyTest, W4A16_Qwen17B_Decode_Down)   { runGemmW4A32(backend_, 1, 6144, 2048, w4_scales_, w4_offsets_, "1p7B_dec_down"); }

// --- Square roofline -------------------------------------------------------
TEST_F(GemmLatencyTest, W4A16_Square_256)  { runGemmW4A32(backend_,  256,  256,  256, w4_scales_, w4_offsets_, "square_256"); }
TEST_F(GemmLatencyTest, W4A16_Square_512)  { runGemmW4A32(backend_,  512,  512,  512, w4_scales_, w4_offsets_, "square_512"); }
TEST_F(GemmLatencyTest, W4A16_Square_1024) { runGemmW4A32(backend_, 1024, 1024, 1024, w4_scales_, w4_offsets_, "square_1024"); }
TEST_F(GemmLatencyTest, W4A16_Square_2048) { runGemmW4A32(backend_, 2048, 2048, 2048, w4_scales_, w4_offsets_, "square_2048"); }
TEST_F(GemmLatencyTest, W4A16_Square_4096) { runGemmW4A32(backend_, 4096, 4096, 4096, w4_scales_, w4_offsets_, "square_4096"); }
