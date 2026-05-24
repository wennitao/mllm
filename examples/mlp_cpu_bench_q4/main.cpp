// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Quantized variant of mlp_cpu_bench. Mirrors the shipped Qwen3-1.7B CPU
// path (qwen3-1.7B-q4.mllm + "linear_impl_type":"GGUF") instead of the
// fp32-sgemm path the original bench uses.
//
//   x = gate_proj_(inputs[0]);   //  Linear, weight = Q4_0 block-quantized
//   x = silu_(x);                //  fp32 elementwise
//   y = up_proj_(inputs[0]);     //  Linear, weight = Q4_0
//   x = x * y;                   //  fp32 elementwise
//   x = down_proj_(x);           //  Linear, weight = Q4_0
//
// Activation stays fp32, output is fp32 -- this matches what
// kGGUF dispatches to in CPULinearOp::forward (ggml::mat_mul with the
// llamafile_sgemm fast path). Only the Linear weights change, so SiLU
// and Mul are identical to the fp32 bench and are kept for total wall
// clock parity.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"
#include "mllm/backends/cpu/ops/SiLUOp.hpp"
#include "mllm/backends/cpu/ops/ElewiseOps.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q4.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/aops/SiLUOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"

using mllm::kCPU;
using mllm::kFloat32;
using mllm::kGGUF_Q4_0;
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

// Build a Q4_0-quantized weight Tensor of logical shape {N, K}.
// Internally allocates N*K/QK4_0 blocks of 18 bytes each.
Tensor make_q4_0_weight(int N, int K, std::mt19937& rng) {
  // Q4_0 block size = 32. Both Qwen3 shapes (2048, 6144) are divisible.
  const int total = N * K;

  // 1) random fp32 source, range [-1, 1].
  std::vector<float> src(total);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& v : src) v = dist(rng);

  // 2) allocate Q4_0 tensor; storage = total/32 * sizeof(block_q4_0)=18B.
  auto W = Tensor::empty({N, K}, kGGUF_Q4_0, kCPU).alloc();

  // 3) pack fp32 into Q4_0 blocks row-by-row (function operates on a flat
  //    buffer; row-major layout is preserved because K is a multiple of 32).
  mllm::cpu::quantize_row_q4_0(src.data(), W.ptr<void>(), total);
  return W;
}

// Linear via mllm::cpu::CPULinearOp with kGGUF impl_type. Weight = Q4_0,
// input/output = fp32. Dispatch lands in ggml::mat_mul -> llamafile_sgemm
// (same code path as the production Qwen3 q4 CPU build).
Stats bench_linear(int S, int K, int N, int warmup, int iters, std::mt19937& rng) {
  mllm::aops::LinearOpOptions opts{};
  opts.in_channels = K;
  opts.out_channels = N;
  opts.bias = false;
  opts.impl_type = mllm::aops::LinearImplTypes::kGGUF;
  mllm::cpu::CPULinearOp op(opts);
  op.weight() = make_q4_0_weight(N, K, rng);

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
  std::mt19937 rng(0xC0DEF00Du ^ static_cast<unsigned>(S));

  Stats g = bench_linear(S, kHidden, kIntermediate, warmup, iters, rng);
  Stats si = bench_silu(S, kIntermediate, warmup, iters);
  Stats u = bench_linear(S, kHidden, kIntermediate, warmup, iters, rng);
  Stats m = bench_mul(S, kIntermediate, warmup, iters);
  Stats d = bench_linear(S, kIntermediate, kHidden, warmup, iters, rng);

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

  std::printf("[mlp-cpu-bench-q4] Q4_0 weight / fp32 act / mllm::cpu::CPULinearOp(kGGUF)\n");
  std::printf("[mlp-cpu-bench-q4] model=Qwen3-1.7B  hidden=%d  intermediate=%d  threads=%u  warmup=%d iters=%d\n\n",
              kHidden, kIntermediate, std::thread::hardware_concurrency(), kWarmup, kIters);

  // Qwen3-1.7B decode shape: one token per forward pass.
  constexpr int kS = 1;
  announce(kS);
  bench_shape(kS, kWarmup, kIters);
});
