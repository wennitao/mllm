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
#include <algorithm>
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
                          const std::string& tag,
                          bool transposeA = false) {
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
  if (transposeA) {
    mm_params.push_back(QNNParamScalarWrapper::create<bool>("transpose_in0", true));
  }
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
// Explicit Transpose-op latency: a real data-movement op that physically
// relays out [M,N] fp16 -> [N,M] in memory (contrast the matmul's fused
// transpose_in* flag, which shuffles tile-by-tile inside the matmul with no
// materialised intermediate). Bounds the cost of re-orienting a tensor.
// ---------------------------------------------------------------------------
static void runTransposeTiming(const std::shared_ptr<QNNBackend>& backend, int M, int N, const std::string& tag) {
  auto A = Tensor::empty({M, N}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({N, M}, kFloat16, kQNN).alloc();
  std::mt19937 rng(0x7AB50001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* ap = A.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)M * N; ++i) ap[i] = (__fp16)dist(rng);

  const std::string graph = "transpose_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));
  auto perm = QNNParamTensorWrapper::create("perm", graph + ".perm", QNN_DATATYPE_UINT_32, std::vector<int32_t>{2});
  auto* pd = reinterpret_cast<uint32_t*>(perm->alloc());
  pd[0] = 1; pd[1] = 0;
  backend->graphAddNode(graph, "transpose", "Transpose", {"A"}, {"O"}, {perm}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(graph));

  std::vector<Tensor> ins = {A}; std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup
  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
  const double bytes = 2.0 * (double)M * N * sizeof(__fp16);  // read once + write once
  const double gbps = bytes / (avg_ms * 1.0e-3) / 1.0e9;
  fprintf(stderr, "[TRANSPOSE %-18s] [%d,%d]fp16 -> [%d,%d]  avg=%9.4f ms  %7.2f GB/s (rd+wr)\n",
          tag.c_str(), M, N, N, M, avg_ms, gbps);
}

// ---------------------------------------------------------------------------
// "Swap then transpose-back": the fast swapMN matmul produces the gate result
// in the transposed orientation [6144,1024] (small free dim, under the VTCM
// cliff), then an explicit Transpose brings it to the conventional [1024,6144].
// One two-node graph (matmul -> NATIVE [6144,1024] -> transpose -> [1024,6144]),
// timed end to end. Comparison target: the baseline NN gate matmul ~9.8 ms.
// Reports effective GFLOP/s on the matmul FLOPs so it lines up with gate_NN.
// ---------------------------------------------------------------------------
static void runSwapThenTranspose(const std::shared_ptr<QNNBackend>& backend, const std::string& tag) {
  const int Mp = 6144, K = 2048, Np = 1024;  // swapped matmul output [Mp,Np]=[6144,1024]
  auto A = Tensor::empty({Mp, K}, kFloat16, kQNN).alloc();   // weight-as-leading
  auto B = Tensor::empty({K, Np}, kFloat16, kQNN).alloc();   // tokens-as-free
  auto T = Tensor::empty({Mp, Np}, kFloat16, kQNN).alloc();  // NATIVE intermediate [6144,1024]
  auto O = Tensor::empty({Np, Mp}, kFloat16, kQNN).alloc();  // conventional output [1024,6144]
  std::mt19937 rng(0x5A1A0001u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* ap = A.ptr<__fp16>(); __fp16* bp = B.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Mp * K; ++i) ap[i] = (__fp16)dist(rng);
  for (size_t i = 0; i < (size_t)K * Np; ++i) bp[i] = (__fp16)dist(rng);

  const std::string graph = "swap_then_t_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A));
  ASSERT_TRUE(backend->addTensor(graph, "B", QNN_TENSOR_TYPE_APP_WRITE, B));
  ASSERT_TRUE(backend->addTensor(graph, "T", QNN_TENSOR_TYPE_NATIVE, T));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O));
  backend->graphAddNode(graph, "matmul", "MatMul", {"A", "B"}, {"T"}, {}, {}, "qti.aisw");
  auto perm = QNNParamTensorWrapper::create("perm", graph + ".perm", QNN_DATATYPE_UINT_32, std::vector<int32_t>{2});
  auto* pd = reinterpret_cast<uint32_t*>(perm->alloc());
  pd[0] = 1; pd[1] = 0;
  backend->graphAddNode(graph, "transpose", "Transpose", {"T"}, {"O"}, {perm}, {}, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(graph));

  std::vector<Tensor> ins = {A, B}; std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup
  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
  const double flops = 2.0 * (double)Mp * Np * K;
  const double gflops_s = flops / (avg_ms * 1.0e-3) / 1.0e9;
  fprintf(stderr, "[SWAP+T %-18s] matmul[6144,1024] + transpose -> [1024,6144]  avg=%9.4f ms  %8.2f GFLOP/s\n",
          tag.c_str(), avg_ms, gflops_s);
}

// ---------------------------------------------------------------------------
// Uniform-uint8 quantization helper + buildability probe. The model's real
// scheme (4-bit LPBQ weight × uint16 activation) is rejected on the runtime
// path ("uniform sets only: FP16/INT16/INT8") — so the only runtime-buildable
// quantized GEMM is a *uniform* integer one. This probe just checks whether a
// plain qti.aisw MatMul over uniform uint8 I/O finalizes at all; if it does,
// the int8 proxy for the transpose trick is viable, otherwise we go to AOT.
// QNN convention: real = (quantized + offset) * scale; u8-sym zp=128 -> offset=-128.
// ---------------------------------------------------------------------------
static Qnn_QuantizeParams_t u8SymQuant(float scale) {
  Qnn_QuantizeParams_t qp = DEFAULT_QUANTIZE_PARAMS;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = scale;
  qp.scaleOffsetEncoding.offset = -128;
  return qp;
}

static Qnn_QuantizeParams_t u16SymQuant(float scale) {
  Qnn_QuantizeParams_t qp = DEFAULT_QUANTIZE_PARAMS;
  qp.encodingDefinition = QNN_DEFINITION_DEFINED;
  qp.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qp.scaleOffsetEncoding.scale = scale;
  qp.scaleOffsetEncoding.offset = -32768;  // uint16 symmetric, zp = 32768
  return qp;
}

// Uniform uint16 GEMM [M,K]x[K,N]->[M,N], both operands dynamic per-tensor (no
// static per-channel weight, so it sidesteps the AXIS-encoding backend crash).
// uint16 = 2-byte elements — the SAME output-panel-byte regime as the production
// uint16 gate output — so the VTCM cliff sits where it does for fp16, letting us
// see whether the *integer engine* reproduces the cliff and the swapMN escape.
// `free_dim` (the matmul's output-column dim N) is what the cliff keys off.
static void runUint16Gemm(const std::shared_ptr<QNNBackend>& backend, int M, int N, int K, const std::string& tag) {
  // Storage must carry a recognised quantized 16-bit dtype tag (plain kUInt16 has
  // no QNN mapping); kUInt16PerTensorSym -> UFIXED_POINT_16, the production activation type.
  auto A = Tensor::empty({M, K}, kUInt16PerTensorSym, kQNN).alloc();
  auto B = Tensor::empty({K, N}, kUInt16PerTensorSym, kQNN).alloc();
  auto O = Tensor::empty({M, N}, kUInt16PerTensorSym, kQNN).alloc();
  std::mt19937 rng(0x16B0u);
  uint16_t* ap = A.ptr<uint16_t>(); for (size_t i = 0; i < (size_t)M * K; ++i) ap[i] = (uint16_t)(rng() & 0xFFFF);
  uint16_t* bp = B.ptr<uint16_t>(); for (size_t i = 0; i < (size_t)K * N; ++i) bp[i] = (uint16_t)(rng() & 0xFFFF);

  const std::string graph = "u16gemm_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A, u16SymQuant(1.0f / 32767.0f)));
  ASSERT_TRUE(backend->addTensor(graph, "B", QNN_TENSOR_TYPE_APP_WRITE, B, u16SymQuant(1.0f / 32767.0f)));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O, u16SymQuant(std::sqrt((float)K) / 32767.0f)));
  backend->graphAddNode(graph, "matmul", "MatMul", {"A", "B"}, {"O"}, {}, {}, "qti.aisw");
  if (!backend->graphFinalize(graph)) {
    fprintf(stderr, "[U16GEMM %-12s] graphFinalize REJECTED (uniform uint16 matmul not lowered)\n", tag.c_str());
    return;
  }
  std::vector<Tensor> ins = {A, B}; std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup
  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;
  const double gflops_s = 2.0 * (double)M * N * K / (avg_ms * 1e-3) / 1e9;
  fprintf(stderr, "[U16GEMM %-12s] [%d,%d]x[%d,%d]->[%d,%d]  free_dim=%d  avg=%9.4f ms  %8.2f GFLOP/s\n",
          tag.c_str(), M, K, K, N, M, N, N, avg_ms, gflops_s);
}

static void runUniformInt8Probe(const std::shared_ptr<QNNBackend>& backend, int M, int K, int N, bool transposeB,
                                const std::string& tag) {
  std::vector<int32_t> wShape = transposeB ? std::vector<int32_t>{N, K} : std::vector<int32_t>{K, N};
  auto A = Tensor::empty({M, K}, kUInt8, kQNN).alloc();
  auto W = Tensor::empty(wShape, kUInt8, kQNN).alloc();
  auto O = Tensor::empty({M, N}, kUInt8, kQNN).alloc();
  std::mt19937 rng(0x1118u);
  uint8_t* ap = A.ptr<uint8_t>(); for (size_t i = 0; i < (size_t)M * K; ++i) ap[i] = (uint8_t)(rng() % 256);
  uint8_t* wp = W.ptr<uint8_t>(); for (size_t i = 0; i < (size_t)K * N; ++i) wp[i] = (uint8_t)(rng() % 256);

  const std::string graph = "u8probe_" + tag;
  ASSERT_NE(backend->createQnnGraph(graph), nullptr);
  ASSERT_TRUE(backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A, u8SymQuant(1.0f / 127.0f)));
  ASSERT_TRUE(backend->addTensor(graph, "W", QNN_TENSOR_TYPE_APP_WRITE, W, u8SymQuant(1.0f / 127.0f)));
  ASSERT_TRUE(backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O, u8SymQuant(1.0f / 16.0f)));
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm;
  if (transposeB) mm.push_back(QNNParamScalarWrapper::create<bool>("transpose_in1", true));
  backend->graphAddNode(graph, "matmul", "MatMul", {"A", "W"}, {"O"}, {}, mm, "qti.aisw");
  bool ok = backend->graphFinalize(graph);
  fprintf(stderr, "[U8PROBE %-10s] uniform-uint8 MatMul (M=%d K=%d N=%d tB=%d) finalize = %s\n",
          tag.c_str(), M, K, N, (int)transposeB, ok ? "OK -- uniform int8 lowers at runtime" : "REJECTED");
}

// Build + run ONE orientation of the int8 gate GEMM with a STATIC per-output-
// channel uint8 weight. The per-channel scale axis is the only thing that moves:
//   conventional  O[M,N]   = A[M,K] · W[K,N]            weight axis = 1 (the N dim)
//   swapMN        O'[N,M]   = Wᵀ[N,K] · Aᵀ[K,M]          weight axis = 0 (N is leading)
// Inputs are the conventional-layout quantized arrays; this relays them out for the
// chosen orientation. Returns avg latency; if `outMN` non-null, fills the dequantised
// result already re-oriented to [M,N] so conventional and swapMN are directly comparable.
static double runInt8GateOrientation(const std::shared_ptr<QNNBackend>& backend, int M, int K, int N, bool swap,
                                     const std::vector<uint8_t>& Wq_conv /*[K,N]*/, const std::vector<float>& sw /*[N]*/,
                                     const std::vector<uint8_t>& Aq_conv /*[M,K]*/, float sx, float so,
                                     const std::string& tag, std::vector<float>* outMN) {
  // Leading / free / contraction dims of the matmul as actually issued.
  const int rows = swap ? N : M;   // matmul leading (output-row) dim
  const int cols = swap ? M : N;   // matmul free (output-col) dim
  // Weight is the per-channel STATIC operand; in swap it is the FIRST operand stored [N,K].
  std::vector<uint8_t> Wbytes(swap ? std::vector<uint8_t>((size_t)N * K) : Wq_conv);
  if (swap) { for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) Wbytes[(size_t)n * K + k] = Wq_conv[(size_t)k * N + n]; }
  // Activation (per-tensor) is the other operand; in swap it is stored transposed [K,M].
  std::vector<uint8_t> Abytes(swap ? std::vector<uint8_t>((size_t)K * M) : Aq_conv);
  if (swap) { for (int k = 0; k < K; ++k) for (int m = 0; m < M; ++m) Abytes[(size_t)k * M + m] = Aq_conv[(size_t)m * K + k]; }

  auto Act = Tensor::empty(swap ? std::vector<int32_t>{K, M} : std::vector<int32_t>{M, K}, kUInt8, kQNN).alloc();
  auto O   = Tensor::empty({rows, cols}, kUInt8, kQNN).alloc();
  std::memcpy(Act.ptr<uint8_t>(), Abytes.data(), Abytes.size());

  // Per-output-channel scale-offset array (real = (q-128)*sw[n]); the wrapper copies it.
  std::vector<Qnn_ScaleOffset_t> soVec(N);
  for (int n = 0; n < N; ++n) soVec[n] = Qnn_ScaleOffset_t{sw[n], -128};

  const std::string graph = "int8gate_" + tag;
  auto model = backend->createQnnGraph(graph);
  if (!model) { fprintf(stderr, "[INT8GATE %-10s] createQnnGraph FAILED\n", tag.c_str()); return -1.0; }

  // Static per-channel weight wrapper (mirrors the W4 path, but plain 8-bit AXIS_SCALE_OFFSET).
  // The static-tensor data MUST live in a kQNN-allocated Tensor (the QNN allocator/registration
  // path expects a registered buffer; a raw host pointer segfaults at finalize).
  std::vector<int32_t> wdims_i32 = swap ? std::vector<int32_t>{N, K} : std::vector<int32_t>{K, N};
  auto Wt = Tensor::empty(wdims_i32, kUInt8, kQNN).alloc();
  std::memcpy(Wt.ptr<uint8_t>(), Wbytes.data(), Wbytes.size());
  std::vector<uint32_t> wdims = swap ? std::vector<uint32_t>{(uint32_t)N, (uint32_t)K} : std::vector<uint32_t>{(uint32_t)K, (uint32_t)N};
  auto w_wrapper = std::make_shared<QNNTensorWrapper>("W", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UFIXED_POINT_8, wdims,
                                                      DEFAULT_QUANTIZE_PARAMS);
  Qnn_Tensor_t* nativeW = w_wrapper->getNativeTensor();
  nativeW->v2.clientBuf.data = Wt.ptr<void>();
  nativeW->v2.clientBuf.dataSize = (uint32_t)Wbytes.size();
  // First-class per-channel API: the wrapper owns the scale-offset array and points the
  // encoding at its own storage (axis = the tensor dimension holding the N output channels).
  w_wrapper->setScaleOffsetQuantization(soVec, /*axis=*/swap ? 0 : 1);

  // The dynamic per-tensor activation, and the per-tensor output.
  if (!backend->addTensor(graph, "Act", QNN_TENSOR_TYPE_APP_WRITE, Act, u8SymQuant(sx))) return -1.0;
  if (model->addTensorWrapper(w_wrapper) != MODEL_NO_ERROR) { fprintf(stderr, "[INT8GATE %-10s] addTensorWrapper FAILED\n", tag.c_str()); return -1.0; }
  if (!backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O, u8SymQuant(so))) return -1.0;

  // Operand order: conventional {Act, W}; swap {W, Act} (weight is the leading operand).
  std::vector<std::string> mm_in = swap ? std::vector<std::string>{"W", "Act"} : std::vector<std::string>{"Act", "W"};
  backend->graphAddNode(graph, "matmul", "MatMul", mm_in, {"O"}, {}, {}, "qti.aisw");
  if (!backend->graphFinalize(graph)) { fprintf(stderr, "[INT8GATE %-10s] graphFinalize REJECTED\n", tag.c_str()); return -1.0; }

  std::vector<Tensor> ins = {Act}; std::vector<Tensor> outs = {O};
  backend->graphExecute(graph, ins, outs);  // warmup
  const int runs = timingRunsFromEnv();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) backend->graphExecute(graph, ins, outs);
  const auto t1 = std::chrono::steady_clock::now();
  const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

  if (outMN) {  // dequantise and re-orient to [M,N] for comparison
    outMN->assign((size_t)M * N, 0.0f);
    const uint8_t* op = O.ptr<uint8_t>();
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
      float val = ((int)op[(size_t)r * cols + c] - 128) * so;
      // conventional [r=M,c=N] -> [m,n]; swap [r=N,c=M] -> [n,m] which is [m,n] transposed.
      if (swap) (*outMN)[(size_t)c * N + r] = val; else (*outMN)[(size_t)r * N + c] = val;
    }
  }
  return avg_ms;
}

// Top-level int8-proxy study: conventional vs swapMN at a fixed gate-shaped GEMM.
// `verify` (small shape) computes an fp32 reference and reports the conventional-vs-
// swapMN agreement — the self-check on whether the per-channel axis flip is correct.
static void runInt8OrientationStudy(const std::shared_ptr<QNNBackend>& backend, int M, int K, int N, bool verify,
                                    const std::string& tag) {
  std::mt19937 rng(0x9CA1u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> Wr((size_t)K * N), Ar((size_t)M * K);
  for (auto& w : Wr) w = dist(rng);
  for (auto& a : Ar) a = dist(rng);

  // Per-output-channel symmetric uint8 weight scales (one per N column).
  std::vector<float> sw(N, 0.0f);
  for (int n = 0; n < N; ++n) { float mx = 1e-6f; for (int k = 0; k < K; ++k) mx = std::max(mx, std::fabs(Wr[(size_t)k * N + n])); sw[n] = mx / 127.0f; }
  std::vector<uint8_t> Wq((size_t)K * N);
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) {
    int q = (int)std::lround(Wr[(size_t)k * N + n] / sw[n]) + 128; Wq[(size_t)k * N + n] = (uint8_t)std::clamp(q, 0, 255);
  }
  float amx = 1e-6f; for (float a : Ar) amx = std::max(amx, std::fabs(a)); const float sx = amx / 127.0f;
  std::vector<uint8_t> Aq((size_t)M * K);
  for (size_t i = 0; i < Ar.size(); ++i) { int q = (int)std::lround(Ar[i] / sx) + 128; Aq[i] = (uint8_t)std::clamp(q, 0, 255); }

  // Output scale: from a host estimate of the true matmul magnitude when verifying; a fixed guess otherwise.
  std::vector<float> R;  float so;
  if (verify) {
    R.assign((size_t)M * N, 0.0f); float rmx = 1e-6f;
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) { double acc = 0; for (int k = 0; k < K; ++k) acc += (double)Ar[(size_t)m * K + k] * Wr[(size_t)k * N + n]; R[(size_t)m * N + n] = (float)acc; rmx = std::max(rmx, std::fabs((float)acc)); }
    so = rmx / 127.0f;
  } else { so = std::sqrt((float)K) / 127.0f; }

  std::vector<float> conv_out, swap_out;
  fprintf(stderr, "[INT8GATE %-10s] building conventional orientation...\n", tag.c_str());
  double t_conv = runInt8GateOrientation(backend, M, K, N, /*swap=*/false, Wq, sw, Aq, sx, so, tag + "_conv", verify ? &conv_out : nullptr);
  fprintf(stderr, "[INT8GATE %-10s] conventional done (%.4f ms). %s\n", tag.c_str(), t_conv,
          std::getenv("MLLM_INT8_CONV_ONLY") ? "skipping swapMN (conv-only gate)" : "building swapMN orientation...");
  double t_swap = -1.0;
  if (!std::getenv("MLLM_INT8_CONV_ONLY"))
    t_swap = runInt8GateOrientation(backend, M, K, N, /*swap=*/true,  Wq, sw, Aq, sx, so, tag + "_swap", verify ? &swap_out : nullptr);

  const double flops = 2.0 * (double)M * N * K;
  auto gf = [&](double ms) { return ms > 0 ? flops / (ms * 1e-3) / 1e9 : 0.0; };
  fprintf(stderr, "[INT8GATE %-10s] M=%d K=%d N=%d  conv=%8.4f ms (%7.1f GFLOP/s)  swapMN=%8.4f ms (%7.1f GFLOP/s)  speedup=%.2fx\n",
          tag.c_str(), M, K, N, t_conv, gf(t_conv), t_swap, gf(t_swap), (t_conv > 0 && t_swap > 0) ? t_conv / t_swap : 0.0);

  if (verify && !conv_out.empty() && !swap_out.empty()) {
    double max_cs = 0, max_cr = 0, max_sr = 0, denom = 0;  // conv-vs-swap, conv-vs-ref, swap-vs-ref (max abs diffs)
    for (size_t i = 0; i < conv_out.size(); ++i) {
      max_cs = std::max(max_cs, std::fabs((double)conv_out[i] - swap_out[i]));
      max_cr = std::max(max_cr, std::fabs((double)conv_out[i] - R[i]));
      max_sr = std::max(max_sr, std::fabs((double)swap_out[i] - R[i]));
      denom = std::max(denom, std::fabs((double)R[i]));
    }
    fprintf(stderr, "[INT8GATE %-10s] CORRECTNESS  max|conv-swap|=%.4f (rel %.1e)  max|conv-ref|=%.3f  max|swap-ref|=%.3f  ref_range=%.2f  output_quant_step(so)=%.4f\n",
            tag.c_str(), max_cs, denom > 0 ? max_cs / denom : 0.0, max_cr, max_sr, denom, so);
  }
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
// On-chip-memory (VTCM) sweeps — isolate why the MLP GEMMs (large N=6144 or
// K=6144) hit ~half the HMX throughput of the 2048-wide projections at the
// same fp16 precision. All fp16, so weight-precision is held constant; only
// the GEMM shape (working-set footprint and weight-reuse factor) varies.
//
//   N-sweep  (fix prefill M=1024, K=2048): grows the output/weight working
//            set. If VTCM capacity is the cause, GFLOP/s rolls off as N grows.
//   K-sweep  (fix M=1024, N=2048): grows the contraction depth. HMX amortises
//            per-tile setup over more accumulation, so GFLOP/s should *rise*
//            with K — the opposite trend, hard to explain by anything but the
//            on-chip tiling story.
//   M-sweep  (fix the gate shape N=6144, K=2048): the canonical roofline ridge.
//            Each weight is reused M times on-chip; small M (decode-like) is
//            weight-bandwidth-bound and low-throughput, large M (prefill)
//            amortises the off-chip weight stream and plateaus at the compute
//            ceiling. Directly shows the bandwidth→compute regime flip.
// ===========================================================================

TEST_F(GemmLatencyTest, VtcmSweep_N_M1024_K2048) {
  for (int N : {512, 1024, 2048, 3072, 4096, 6144, 8192, 12288})
    runLinear(backend_, 1024, N, 2048, "Nsweep_N" + std::to_string(N));
}
TEST_F(GemmLatencyTest, VtcmSweep_K_M1024_N2048) {
  for (int K : {512, 1024, 2048, 3072, 4096, 6144, 8192, 12288})
    runLinear(backend_, 1024, 2048, K, "Ksweep_K" + std::to_string(K));
}
TEST_F(GemmLatencyTest, VtcmSweep_M_N6144_K2048) {
  for (int M : {1, 16, 64, 128, 256, 512, 1024, 2048})
    runLinear(backend_, M, 6144, 2048, "Msweep_M" + std::to_string(M));
}

// ===========================================================================
// Transpose-layout study at the FIXED gate arithmetic M=1024, K=2048, N=6144.
// Same product, same 12 MB output panel (already past the VTCM cliff) — only
// the operand transpose flags vary. The four BLAS layouts (op(A)·op(B)=C[M,N])
// isolate the in-kernel transpose / input-layout overhead from the FLOP count
// (the doc found transpose_in1 on the QK matmul burns real HVX cycles). The
// fifth case swaps M<->N to produce the transposed output [6144,1024]: same
// FLOPs and same panel bytes, but the *free* output dimension shrinks to 1024
// (fits VTCM) while the leading dimension becomes 6144 — tests whether the
// cliff is governed specifically by the free-dimension width. Separate tests
// so an unsupported transpose layout can't abort the rest.
//   Tensor stored shapes:  op(A) must be [M,K]=[1024,2048], op(B)=[K,N]=[2048,6144].
//   NN: A[1024,2048],  B[2048,6144]               TN: A[2048,1024]^T, B[2048,6144]
//   NT: A[1024,2048],  B[6144,2048]^T             TT: A[2048,1024]^T, B[6144,2048]^T
// ===========================================================================
TEST_F(GemmLatencyTest, TransposeStudy_Gate_NN) {
  runGemmTiming(backend_, {1024,2048}, {2048,6144}, {1024,6144}, /*transposeB=*/false, "gate_NN", /*transposeA=*/false);
}
TEST_F(GemmLatencyTest, TransposeStudy_Gate_NT) {
  runGemmTiming(backend_, {1024,2048}, {6144,2048}, {1024,6144}, /*transposeB=*/true,  "gate_NT", /*transposeA=*/false);
}
TEST_F(GemmLatencyTest, TransposeStudy_Gate_TN) {
  runGemmTiming(backend_, {2048,1024}, {2048,6144}, {1024,6144}, /*transposeB=*/false, "gate_TN", /*transposeA=*/true);
}
TEST_F(GemmLatencyTest, TransposeStudy_Gate_TT) {
  runGemmTiming(backend_, {2048,1024}, {6144,2048}, {1024,6144}, /*transposeB=*/true,  "gate_TT", /*transposeA=*/true);
}
TEST_F(GemmLatencyTest, TransposeStudy_Gate_SwapMN) {  // transposed output [N,M]=[6144,1024], free dim shrinks to 1024
  runGemmTiming(backend_, {6144,2048}, {2048,1024}, {6144,1024}, /*transposeB=*/false, "gate_swapMN", /*transposeA=*/false);
}

// Manual (explicit) Transpose op vs the fused flag, and the swap-then-transpose
// path that delivers the conventional gate orientation while keeping the fast
// matmul. The standalone transpose numbers bound the boundary cost; the SWAP+T
// graph shows whether fast-matmul + a real transpose still beats the ~9.8 ms NN.
TEST_F(GemmLatencyTest, TransposeOnly_Out1024x6144) { runTransposeTiming(backend_, 6144, 1024, "to_1024x6144"); }  // [6144,1024]->[1024,6144]
TEST_F(GemmLatencyTest, TransposeOnly_Out6144x1024) { runTransposeTiming(backend_, 1024, 6144, "to_6144x1024"); }  // [1024,6144]->[6144,1024]
TEST_F(GemmLatencyTest, TransposeStudy_Gate_SwapThenTranspose) { runSwapThenTranspose(backend_, "gate"); }

// Buildability probe for the int8 proxy: does a plain uniform-uint8 MatMul lower
// at runtime at all? Gate for whether the quantized-GEMM proxy is even possible.
TEST_F(GemmLatencyTest, Int8Probe_NN) { runUniformInt8Probe(backend_, 64, 64, 64, /*transposeB=*/false, "nn64"); }
TEST_F(GemmLatencyTest, Int8Probe_NT) { runUniformInt8Probe(backend_, 64, 64, 64, /*transposeB=*/true,  "nt64"); }

// Int8 proxy for the transpose trick: per-channel-weight quantized GEMM, conventional
// vs swapMN. Verify shape checks the per-channel-axis-flip correctness; gate shape
// measures whether the int8 path shows the same cliff/win as fp16.
TEST_F(GemmLatencyTest, Int8Orientation_Verify)      { runInt8OrientationStudy(backend_, 256, 256, 512, /*verify=*/true,  "verify"); }
TEST_F(GemmLatencyTest, Int8Orientation_GateLatency) { runInt8OrientationStudy(backend_, 1024, 2048, 6144, /*verify=*/false, "gate"); }

// Uniform-uint16 quantized-engine proxy for the swapMN trick at the gate arithmetic
// (M=tokens, K=hidden, N=intermediate). Tests whether the integer engine shows the
// same VTCM output-panel cliff and the same swapMN escape as fp16. (M,N,K) order.
TEST_F(GemmLatencyTest, Uint16Gate_BelowCliffRef) { runUint16Gemm(backend_, 1024, 2048, 2048, "ref_N2048");   }  // free dim 2048 (under cliff)
TEST_F(GemmLatencyTest, Uint16Gate_NN)            { runUint16Gemm(backend_, 1024, 6144, 2048, "gate_NN");     }  // conventional gate, free dim 6144 (over cliff)
TEST_F(GemmLatencyTest, Uint16Gate_SwapMN)        { runUint16Gemm(backend_, 6144, 1024, 2048, "gate_swapMN"); }  // transposed output, free dim 1024 (under cliff)

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
