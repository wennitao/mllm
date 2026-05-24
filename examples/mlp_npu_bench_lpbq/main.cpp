// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Quantized variant of mlp_npu_bench. Mirrors the spirit of the shipped
// Qwen3-1.7B QNN AOT path (qwen3-1.7B-lpbq-sha.bin built from
// "linear_impl_type":"QNN_LPBQ_w4a16o16_G16") by running MatMul with
// quantized weights and quantized activations on the HTP, instead of the
// fp16 MatMul that the original mlp_npu_bench uses.
//
// IMPORTANT: this is a *w8a16-per-channel stand-in*, NOT real LPBQ.
//   - Weight  W : sfixed_point_8, per-output-channel scale (axis=0).
//   - Act     X : ufixed_point_16, per-tensor scale-offset.
//   - Output  Y : ufixed_point_16, per-tensor scale-offset.
// Real LPBQ w4 uses int4 weights with BLOCKWISE_EXPANSION (per-channel +
// per-block secondary scales) and is baked offline by compile_sha.cpp
// through LLMQuantRecipePass / LPBQCanonicalizePass; replicating it on
// the online graph-build path is significantly more plumbing. The point
// of this bench is to expose the order-of-magnitude difference between
// fp16 MatMul and *any* quantized MatMul on HTP.
//
// SiLU and Mul stay fp16 (matching the original bench) since only Linear
// changes in the deployed path.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "QnnBackend.h"

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Context;
using mllm::kFloat16;
using mllm::kInt8PerTensorSym;
using mllm::kQNN;
using mllm::kUInt16PerTensorAsy;
using mllm::mllm_fp16_t;
using mllm::Tensor;
using mllm::qnn::QNNBackend;
using mllm::qnn::QNNParamScalarWrapper;

namespace {

constexpr int kHidden = 2048;        // Qwen3-1.7B hidden_size
constexpr int kIntermediate = 6144;  // Qwen3-1.7B intermediate_size

// Synthetic quantization params. Values don't affect kernel cycle counts
// on HTP; only the encoding metadata matters for graph validation.
constexpr float kWeightScale = 1.0f / 127.0f;
constexpr float kActScale = 2.0f / 65535.0f;
constexpr int32_t kActOffset = -32768;     // f = scale * (q + offset)
constexpr float kOutScale = 1.0f / 256.0f;  // bigger range after accum
constexpr int32_t kOutOffset = -32768;

struct Stats {
  double min_ms;
  double median_ms;
  double mean_ms;
};

Stats summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Stats s{};
  s.min_ms = samples.front();
  s.median_ms = samples[samples.size() / 2];
  s.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  return s;
}

double linear_gflops(int S, int K, int N, double ms) {
  const double flops = 2.0 * S * K * N;
  return (flops / (ms * 1e-3)) / 1e9;
}
double elem_gflops(int S, int I, double ms) {
  const double flops = static_cast<double>(S) * I;
  return (flops / (ms * 1e-3)) / 1e9;
}

void fill_random_fp16(mllm_fp16_t* ptr, size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<mllm_fp16_t>(dist(rng));
}
void fill_random_i8(int8_t* ptr, size_t n, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist(-128, 127);
  for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<int8_t>(dist(rng));
}
void fill_random_u16(uint16_t* ptr, size_t n, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist(0, 65535);
  for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<uint16_t>(dist(rng));
}

std::vector<double> time_graph(const std::shared_ptr<QNNBackend>& backend, const std::string& graph,
                               std::vector<Tensor>& ins, std::vector<Tensor>& outs, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) backend->graphExecute(graph, ins, outs);
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    backend->graphExecute(graph, ins, outs);
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return samples;
}

// Lives for the whole bench so the per-channel scale-offset arrays we
// hand to QNN via Qnn_QuantizeParams_t.axisScaleOffsetEncoding outlive
// graphFinalize+graphExecute. (QNN may keep pointers into client memory
// until the graph is destroyed.)
struct BenchState {
  std::vector<std::vector<Qnn_ScaleOffset_t>> w_per_channel_arrays;
};

Qnn_QuantizeParams_t make_per_tensor(float scale, int32_t offset) {
  Qnn_QuantizeParams_t q{};
  q.encodingDefinition = QNN_DEFINITION_DEFINED;
  q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  q.scaleOffsetEncoding.scale = scale;
  q.scaleOffsetEncoding.offset = offset;
  return q;
}

Qnn_QuantizeParams_t make_per_channel(int N, int32_t axis, BenchState& state) {
  state.w_per_channel_arrays.emplace_back(N);
  auto& vec = state.w_per_channel_arrays.back();
  for (int n = 0; n < N; ++n) vec[n] = Qnn_ScaleOffset_t{.scale = kWeightScale, .offset = 0};

  Qnn_QuantizeParams_t q{};
  q.encodingDefinition = QNN_DEFINITION_DEFINED;
  q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  q.axisScaleOffsetEncoding.axis = axis;
  q.axisScaleOffsetEncoding.numScaleOffsets = static_cast<uint32_t>(N);
  q.axisScaleOffsetEncoding.scaleOffset = vec.data();
  return q;
}

// MatMul-as-Linear: Y = X @ W^T, X[S,K] u16-per-tensor, W[N,K] i8-per-channel(axis=0), Y[S,N] u16.
Stats bench_linear(const std::shared_ptr<QNNBackend>& backend, BenchState& state, const std::string& tag, int S, int K,
                   int N, int warmup, int iters, std::mt19937& rng) {
  auto X = Tensor::empty({S, K}, kUInt16PerTensorAsy, kQNN).alloc();
  auto W = Tensor::empty({N, K}, kInt8PerTensorSym, kQNN).alloc();
  auto Y = Tensor::empty({S, N}, kUInt16PerTensorAsy, kQNN).alloc();
  fill_random_u16(X.ptr<uint16_t>(), static_cast<size_t>(S) * K, rng);
  fill_random_i8(W.ptr<int8_t>(), static_cast<size_t>(N) * K, rng);

  const auto x_q = make_per_tensor(kActScale, kActOffset);
  const auto y_q = make_per_tensor(kOutScale, kOutOffset);
  const auto w_q = make_per_channel(N, /*axis=*/0, state);

  const std::string graph = tag + "_S" + std::to_string(S);
  backend->createQnnGraph(graph);
  backend->addTensor(graph, "X", QNN_TENSOR_TYPE_APP_WRITE, X, x_q);
  backend->addStaticTensor(graph, "W", W, w_q);
  backend->addTensor(graph, "Y", QNN_TENSOR_TYPE_APP_READ, Y, y_q);

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_params = {
      QNNParamScalarWrapper::create<bool>("transpose_in1", true),
  };
  backend->graphAddNode(graph, tag + "_mm", "MatMul", {"X", "W"}, {"Y"}, {}, mm_params, "qti.aisw");
  backend->graphFinalize(graph);

  std::vector<Tensor> ins{X}, outs{Y};
  return summarize(time_graph(backend, graph, ins, outs, warmup, iters));
}

// SiLU as Sigmoid + ElementWiseMultiply (qti.aisw has no native "SiLU";
// the LLaMAPackage custom op isn't registered in this build). Kept fp16
// to match the original bench -- the deployed model also runs activations
// fp16 outside the Linear blocks.
Stats bench_silu(const std::shared_ptr<QNNBackend>& backend, int S, int I, int warmup, int iters, std::mt19937& rng) {
  auto X = Tensor::empty({S, I}, kFloat16, kQNN).alloc();
  auto SIG = Tensor::empty({S, I}, kFloat16, kQNN);
  auto Y = Tensor::empty({S, I}, kFloat16, kQNN).alloc();
  fill_random_fp16(X.ptr<mllm_fp16_t>(), static_cast<size_t>(S) * I, rng);

  const std::string graph = "silu_S" + std::to_string(S);
  backend->createQnnGraph(graph);
  backend->addTensor(graph, "X", QNN_TENSOR_TYPE_APP_WRITE, X);
  backend->addTensor(graph, "SIG", QNN_TENSOR_TYPE_NATIVE, SIG);
  backend->addTensor(graph, "Y", QNN_TENSOR_TYPE_APP_READ, Y);
  backend->graphAddNode(graph, "sigmoid", "Sigmoid", {"X"}, {"SIG"}, {}, {}, "qti.aisw");
  backend->graphAddNode(graph, "silu_mul", "ElementWiseMultiply", {"SIG", "X"}, {"Y"}, {}, {}, "qti.aisw");
  backend->graphFinalize(graph);

  std::vector<Tensor> ins{X}, outs{Y};
  return summarize(time_graph(backend, graph, ins, outs, warmup, iters));
}

Stats bench_mul(const std::shared_ptr<QNNBackend>& backend, int S, int I, int warmup, int iters, std::mt19937& rng) {
  auto A = Tensor::empty({S, I}, kFloat16, kQNN).alloc();
  auto B = Tensor::empty({S, I}, kFloat16, kQNN).alloc();
  auto Y = Tensor::empty({S, I}, kFloat16, kQNN).alloc();
  fill_random_fp16(A.ptr<mllm_fp16_t>(), static_cast<size_t>(S) * I, rng);
  fill_random_fp16(B.ptr<mllm_fp16_t>(), static_cast<size_t>(S) * I, rng);

  const std::string graph = "mul_S" + std::to_string(S);
  backend->createQnnGraph(graph);
  backend->addTensor(graph, "A", QNN_TENSOR_TYPE_APP_WRITE, A);
  backend->addTensor(graph, "B", QNN_TENSOR_TYPE_APP_WRITE, B);
  backend->addTensor(graph, "Y", QNN_TENSOR_TYPE_APP_READ, Y);
  backend->graphAddNode(graph, "mul", "ElementWiseMultiply", {"A", "B"}, {"Y"}, {}, {}, "qti.aisw");
  backend->graphFinalize(graph);

  std::vector<Tensor> ins{A, B}, outs{Y};
  return summarize(time_graph(backend, graph, ins, outs, warmup, iters));
}

void announce(int S) {
  std::printf(">> mlp        S=%5d hidden=%d intermediate=%d ...\n", S, kHidden, kIntermediate);
  std::fflush(stdout);
}

void bench_shape(const std::shared_ptr<QNNBackend>& backend, int S, int warmup, int iters) {
  BenchState state;
  std::mt19937 rng(0xC0DEC0DEu ^ static_cast<unsigned>(S));

  Stats g = bench_linear(backend, state, "gate_proj", S, kHidden, kIntermediate, warmup, iters, rng);
  Stats si = bench_silu(backend, S, kIntermediate, warmup, iters, rng);
  Stats u = bench_linear(backend, state, "up_proj", S, kHidden, kIntermediate, warmup, iters, rng);
  Stats m = bench_mul(backend, S, kIntermediate, warmup, iters, rng);
  Stats d = bench_linear(backend, state, "down_proj", S, kIntermediate, kHidden, warmup, iters, rng);

  const double total_min = g.min_ms + si.min_ms + u.min_ms + m.min_ms + d.min_ms;
  const double total_med = g.median_ms + si.median_ms + u.median_ms + m.median_ms + d.median_ms;
  const double total_mean = g.mean_ms + si.mean_ms + u.mean_ms + m.mean_ms + d.mean_ms;

  auto print_row = [&](const char* name, const Stats& st, double gfps) {
    std::printf("  %-9s  min=%8.4f  med=%8.4f  mean=%8.4f ms   %7.1f GF/s\n", name, st.min_ms, st.median_ms,
                st.mean_ms, gfps);
  };
  std::printf("mlp  S=%d\n", S);
  print_row("gate_proj", g, linear_gflops(S, kHidden, kIntermediate, g.mean_ms));
  print_row("silu", si, elem_gflops(S, kIntermediate, si.mean_ms));
  print_row("up_proj", u, linear_gflops(S, kHidden, kIntermediate, u.mean_ms));
  print_row("mul", m, elem_gflops(S, kIntermediate, m.mean_ms));
  print_row("down_proj", d, linear_gflops(S, kIntermediate, kHidden, d.mean_ms));
  std::printf("  %-9s  min=%8.4f  med=%8.4f  mean=%8.4f ms\n", "TOTAL", total_min, total_med, total_mean);
  std::fflush(stdout);
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  if (const char* p = std::getenv("ADSP_LIBRARY_PATH"); !p || *p == '\0') {
    setenv("ADSP_LIBRARY_PATH", ".:/data/local/tmp", /*overwrite=*/1);
  }
  setenv("MLLM_QNN_PROFILE_OFF", "1", /*overwrite=*/1);

  mllm::initQnnBackend("/tmp/__mllm_mlp_npu_bench_lpbq_no_context__.bin");

  auto backend = std::static_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!backend) {
    std::fprintf(stderr, "QNN backend not available\n");
    return 1;
  }

  constexpr int kWarmup = 3;
  constexpr int kIters = 20;

  std::printf("[mlp-npu-bench-lpbq] w8a16 MatMul (per-channel int8 W, per-tensor u16 X/Y) -- LPBQ stand-in\n");
  std::printf("[mlp-npu-bench-lpbq] SiLU + EWMul still fp16 (qti.aisw)\n");
  std::printf("[mlp-npu-bench-lpbq] model=Qwen3-1.7B  hidden=%d  intermediate=%d  warmup=%d iters=%d\n\n", kHidden,
              kIntermediate, kWarmup, kIters);

  // Qwen3-1.7B decode shape: one token per forward pass.
  constexpr int kS = 1;
  announce(kS);
  bench_shape(backend, kS, kWarmup, kIters);
});
