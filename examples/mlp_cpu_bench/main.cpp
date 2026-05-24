// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// CPU counterpart to mllm-mlp-npu-bench. Per-op wall-clock of the Qwen3-1.7B
// MLP block (mllm/models/qwen3/modeling_qwen3.hpp:87-94):
//
//   x = gate_proj_(inputs[0]);   //  [S, H] -> [S, I]   (Linear)
//   x = silu_(x);                //  [S, I]             (SiLU)
//   y = up_proj_(inputs[0]);     //  [S, H] -> [S, I]   (Linear)
//   x = x * y;                   //  [S, I] elementwise (Mul)
//   x = down_proj_(x);           //  [S, I] -> [S, H]   (Linear)
//
// fp32 to match the CPU production path (and fa_cpu_bench). Weights are
// random -- their values don't affect kernel timing on the BLAS code path.
//
// Qwen3-1.7B dims: hidden_size=2048, intermediate_size=6144.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/ops/SiLUOp.hpp"
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"

using mllm::kCPU;
using mllm::kFloat32;
using mllm::Tensor;

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

// Time a fully-prepared op (reshape+setup already called) over warmup+iters.
template <typename Op>
Stats time_op(Op& op, std::vector<Tensor>& ins, std::vector<Tensor>& outs, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) op.forward(ins, outs);
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    op.forward(ins, outs);
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return summarize(std::move(samples));
}

// Linear via mllm::cpu::CPULinearOp. weight layout = [out, in], fp32. Uses
// kDefault impl_type (BLAS if available, otherwise the MllmBlas fallback).
Stats bench_linear(int S, int K, int N, int warmup, int iters) {
  mllm::aops::LinearOpOptions opts{};
  opts.in_channels = K;
  opts.out_channels = N;
  opts.bias = false;
  opts.impl_type = mllm::aops::LinearImplTypes::kDefault;
  mllm::cpu::CPULinearOp op(opts);
  op.weight() = Tensor::random({N, K}, -1.f, 1.f, kFloat32, kCPU);

  Tensor X = Tensor::random({S, K}, -1.f, 1.f, kFloat32, kCPU);
  std::vector<Tensor> ins{X};
  std::vector<Tensor> outs;
  op.reshape(ins, outs);
  op.setup(ins, outs);
  return time_op(op, ins, outs, warmup, iters);
}

Stats bench_silu(int S, int I, int warmup, int iters) {
  mllm::aops::SiLUOpOptions opts{};
  mllm::cpu::CPUSiLUOp op(opts);
  Tensor X = Tensor::random({S, I}, -1.f, 1.f, kFloat32, kCPU);
  std::vector<Tensor> ins{X};
  std::vector<Tensor> outs;
  op.reshape(ins, outs);
  op.setup(ins, outs);
  return time_op(op, ins, outs, warmup, iters);
}

Stats bench_mul(int S, int I, int warmup, int iters) {
  mllm::aops::MulOpOptions opts{};
  mllm::cpu::CPUMulOp op(opts);
  Tensor A = Tensor::random({S, I}, -1.f, 1.f, kFloat32, kCPU);
  Tensor B = Tensor::random({S, I}, -1.f, 1.f, kFloat32, kCPU);
  std::vector<Tensor> ins{A, B};
  std::vector<Tensor> outs;
  op.reshape(ins, outs);
  op.setup(ins, outs);
  return time_op(op, ins, outs, warmup, iters);
}

void announce(int S) {
  std::printf(">> mlp        S=%5d hidden=%d intermediate=%d ...\n", S, kHidden, kIntermediate);
  std::fflush(stdout);
}

void bench_shape(int S, int warmup, int iters) {
  Stats g = bench_linear(S, kHidden, kIntermediate, warmup, iters);
  Stats si = bench_silu(S, kIntermediate, warmup, iters);
  Stats u = bench_linear(S, kHidden, kIntermediate, warmup, iters);
  Stats m = bench_mul(S, kIntermediate, warmup, iters);
  Stats d = bench_linear(S, kIntermediate, kHidden, warmup, iters);

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

  constexpr int kWarmup = 3;
  constexpr int kIters = 20;

  std::printf("[mlp-cpu-bench] fp32 / mllm::cpu::CPULinearOp + CPUSiLUOp + CPUMulOp (impl_type=kDefault)\n");
  std::printf("[mlp-cpu-bench] model=Qwen3-1.7B  hidden=%d  intermediate=%d  threads=%u  warmup=%d iters=%d\n\n",
              kHidden, kIntermediate, std::thread::hardware_concurrency(), kWarmup, kIters);

  // Qwen3-1.7B decode shape: one token per forward pass.
  constexpr int kS = 1;
  announce(kS);
  bench_shape(kS, kWarmup, kIters);
});
