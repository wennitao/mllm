// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Standalone microbenchmark for the CPU FlashAttention v2 kernel
// (mllm::cpu::CPUFlashAttention2Op  +  mllm/backends/cpu/kernels/common/fa2_1).
//
// CPU counterpart to mllm-fa-opencl-bench, sweeping the same (B, H, D) shapes
// across the same two regimes so OpenCL and CPU numbers line up apples-to-apples.
//
// IMPORTANT layout difference vs. the OpenCL bench:
//   OpenCL FA op:  Q = [B, H, S_q, D]   (BHSD)
//   CPU FA op:     Q = [B, S_q, H, D]   (BSHD)
// The model code transposes once between these conventions; we just allocate
// tensors in BSHD here.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <vector>

#include "mllm/mllm.hpp"
#include "mllm/backends/cpu/ops/FlashAttention2Op.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"

using mllm::DeviceTypes;
using mllm::kCPU;
using mllm::kFloat32;
using mllm::Tensor;

namespace {

struct Shape {
  int B;
  int H;
  int S_q;
  int S_kv;
  int D;
  const char* label;
};

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

double gflops_causal(const Shape& sh, double ms) {
  const double pairs = static_cast<double>(sh.S_q) * (static_cast<double>(sh.S_kv) + 1.0) * 0.5;
  const double flops = 4.0 * sh.B * sh.H * sh.D * pairs;
  return (flops / (ms * 1e-3)) / 1e9;
}

double gbps_optimal(const Shape& sh, double ms) {
  const double bytes_per_elem = 4.0;
  const double bh = static_cast<double>(sh.B) * sh.H * sh.D;
  const double bytes = bytes_per_elem * (2.0 * bh * sh.S_q + 2.0 * bh * sh.S_kv);
  return (bytes / (ms * 1e-3)) / 1e9;
}

void announce(const Shape& sh) {
  std::printf(">> %-10s B=%d H=%2d D=%3d S_q=%5d S_kv=%5d ...\n", sh.label, sh.B, sh.H, sh.D, sh.S_q, sh.S_kv);
  std::fflush(stdout);
}

void bench_one(const Shape& sh, int warmup, int iters) {
  // BSHD layout for the CPU FA op.
  Tensor Q = Tensor::random({sh.B, sh.S_q, sh.H, sh.D}, -1.f, 1.f, kFloat32, kCPU);
  Tensor K = Tensor::random({sh.B, sh.S_kv, sh.H, sh.D}, -1.f, 1.f, kFloat32, kCPU);
  Tensor V = Tensor::random({sh.B, sh.S_kv, sh.H, sh.D}, -1.f, 1.f, kFloat32, kCPU);

  mllm::aops::FlashAttention2OpOptions opts{};
  opts.B = sh.B;
  opts.q_head = sh.H;
  opts.kv_head = sh.H;
  opts.D = sh.D;
  opts.causal_mask = true;
  mllm::cpu::CPUFlashAttention2Op op(opts);

  std::vector<Tensor> inputs{Q, K, V};
  std::vector<Tensor> outputs;
  op.reshape(inputs, outputs);
  op.setup(inputs, outputs);

  for (int i = 0; i < warmup; ++i) op.forward(inputs, outputs);

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    op.forward(inputs, outputs);
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  Stats st = summarize(std::move(samples));
  std::printf("%-10s  B=%d H=%2d D=%3d  S_q=%5d S_kv=%5d  min=%8.3f ms  med=%8.3f ms  mean=%8.3f ms  %7.1f GF/s  %6.1f GB/s\n",
              sh.label, sh.B, sh.H, sh.D, sh.S_q, sh.S_kv, st.min_ms, st.median_ms, st.mean_ms,
              gflops_causal(sh, st.min_ms), gbps_optimal(sh, st.min_ms));
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  constexpr int kB = 1;
  constexpr int kH = 16;
  constexpr int kD = 128;
  constexpr int kWarmup = 3;
  constexpr int kIters = 20;

  std::printf("[cpu] threads default = %u\n", std::thread::hardware_concurrency());
  std::printf("\n[mode=prefill]  S_kv fixed = 128, S_q varies\n");
  {
    constexpr int kS_kv = 128;
    const int s_q_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048};
    for (int s_q : s_q_sizes) {
      Shape sh{kB, kH, s_q, kS_kv, kD, "prefill"};
      announce(sh);
      bench_one(sh, kWarmup, kIters);
    }
  }

  std::printf("\n[mode=decode]   S_q fixed = 1, S_kv varies\n");
  {
    constexpr int kS_q = 1;
    const int s_kv_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    for (int s_kv : s_kv_sizes) {
      Shape sh{kB, kH, kS_q, s_kv, kD, "decode"};
      announce(sh);
      bench_one(sh, kWarmup, kIters);
    }
  }
});
