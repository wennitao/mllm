// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Per-operator microbenchmark for the Qwen3-1.7B MLP block on the QNN HTP
// NPU through the online graph-build path.
//
// Block (mllm/models/qwen3/modeling_qwen3.hpp:87-94):
//
//   x = gate_proj_(inputs[0]);   //  [S, H] @ [I, H]^T -> [S, I]   (Linear)
//   x = silu_(x);                //  [S, I]                         (SiLU)
//   y = up_proj_(inputs[0]);     //  [S, H] @ [I, H]^T -> [S, I]   (Linear)
//   x = x * y;                   //  elementwise [S, I]             (Mul)
//   x = down_proj_(x);           //  [S, I] @ [H, I]^T -> [S, H]   (Linear)
//
// Five distinct ops fire; the user-facing "4 operators" are the four named
// layers (gate_proj, silu, up_proj, down_proj). We measure all five so the
// total wall-clock adds up.
//
// Qwen3-1.7B dims (examples/qwen3_qnn_aot/config_1.7B.json):
//   hidden_size       = 2048
//   intermediate_size = 6144

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
using mllm::kQNN;
using mllm::mllm_fp16_t;
using mllm::Tensor;
using mllm::qnn::QNNBackend;
using mllm::qnn::QNNParamScalarWrapper;

namespace {

constexpr int kHidden = 2048;        // Qwen3-1.7B hidden_size
constexpr int kIntermediate = 6144;  // Qwen3-1.7B intermediate_size

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

// MatMul-as-Linear graph: Y = X @ W^T,  X [S, K], W [N, K], Y [S, N].
Stats bench_linear(const std::shared_ptr<QNNBackend>& backend, const std::string& tag, int S, int K, int N, int warmup,
                   int iters, std::mt19937& rng) {
  auto X = Tensor::empty({S, K}, kFloat16, kQNN).alloc();
  auto W = Tensor::empty({N, K}, kFloat16, kQNN).alloc();
  auto Y = Tensor::empty({S, N}, kFloat16, kQNN).alloc();
  fill_random_fp16(X.ptr<mllm_fp16_t>(), static_cast<size_t>(S) * K, rng);
  fill_random_fp16(W.ptr<mllm_fp16_t>(), static_cast<size_t>(N) * K, rng);

  const std::string graph = tag + "_S" + std::to_string(S);
  backend->createQnnGraph(graph);
  backend->addTensor(graph, "X", QNN_TENSOR_TYPE_APP_WRITE, X);
  backend->addStaticTensor(graph, "W", W);
  backend->addTensor(graph, "Y", QNN_TENSOR_TYPE_APP_READ, Y);

  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_params = {
      QNNParamScalarWrapper::create<bool>("transpose_in1", true),
  };
  backend->graphAddNode(graph, tag + "_mm", "MatMul", {"X", "W"}, {"Y"}, {}, mm_params, "qti.aisw");
  backend->graphFinalize(graph);

  std::vector<Tensor> ins{X}, outs{Y};
  return summarize(time_graph(backend, graph, ins, outs, warmup, iters));
}

// SiLU implemented as a 2-op graph: Y = Sigmoid(X) * X.
// qti.aisw has no native "SiLU" -- that's a LLaMAPackage custom HVX op,
// which isn't registered at runtime in this build. Decomposing into
// Sigmoid + ElementWiseMultiply keeps us inside qti.aisw and reflects
// what a fused SiLU implementation would internally do anyway.
Stats bench_silu(const std::shared_ptr<QNNBackend>& backend, int S, int I, int warmup, int iters, std::mt19937& rng) {
  auto X = Tensor::empty({S, I}, kFloat16, kQNN).alloc();
  auto SIG = Tensor::empty({S, I}, kFloat16, kQNN);  // native intermediate
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
  std::mt19937 rng(0xC0DE0000u ^ static_cast<unsigned>(S));

  Stats g = bench_linear(backend, "gate_proj", S, kHidden, kIntermediate, warmup, iters, rng);
  Stats si = bench_silu(backend, S, kIntermediate, warmup, iters, rng);
  Stats u = bench_linear(backend, "up_proj", S, kHidden, kIntermediate, warmup, iters, rng);
  Stats m = bench_mul(backend, S, kIntermediate, warmup, iters, rng);
  Stats d = bench_linear(backend, "down_proj", S, kIntermediate, kHidden, warmup, iters, rng);

  const double total_min = g.min_ms + si.min_ms + u.min_ms + m.min_ms + d.min_ms;
  const double total_med = g.median_ms + si.median_ms + u.median_ms + m.median_ms + d.median_ms;
  const double total_mean = g.mean_ms + si.mean_ms + u.mean_ms + m.mean_ms + d.mean_ms;

  // One row per op with min/med/mean for that op (GF/s computed off mean).
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
  // Required: per-graph NSP profile buffers accumulate otherwise.
  setenv("MLLM_QNN_PROFILE_OFF", "1", /*overwrite=*/1);

  mllm::initQnnBackend("/tmp/__mllm_mlp_npu_bench_no_context__.bin");

  auto backend = std::static_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!backend) {
    std::fprintf(stderr, "QNN backend not available\n");
    return 1;
  }

  constexpr int kWarmup = 3;
  constexpr int kIters = 20;

  std::printf("[mlp-npu-bench] fp16 / qti.aisw (MatMul + SiLU + EWMul)\n");
  std::printf("[mlp-npu-bench] model=Qwen3-1.7B  hidden=%d  intermediate=%d  warmup=%d iters=%d\n\n", kHidden,
              kIntermediate, kWarmup, kIters);

  // Qwen3-1.7B decode shape: one token per forward pass.
  constexpr int kS = 1;
  announce(kS);
  bench_shape(backend, kS, kWarmup, kIters);
});
