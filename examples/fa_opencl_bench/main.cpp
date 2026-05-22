// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Standalone microbenchmark for the OpenCL FlashAttention v1 kernel
// (mllm::opencl::OpenCLFlashAttention2Op + flash_attention.cl).
//
// Only this single op is exercised -- no model load, no tokenizer, no graph.
// Q/K/V are random BHSD tensors created on CPU and copied to the OpenCL
// device, then the op is invoked in a tight warmup+timed loop, with
// commandQueue().finish() bracketing every iteration so the host-side
// std::chrono delta is the true GPU wall-clock for one kernel launch.
//
// Build (host config that targets android-arm64):
//   cmake --build build-android-arm64-v8a --target mllm-fa-opencl-bench
//
// Push and run on device (per the user's memory rule, prefer adb -P/-s):
//   adb -P $PORT -s $DEVICE push \
//       build-android-arm64-v8a/bin/mllm-fa-opencl-bench /data/local/tmp/
//   adb -P $PORT -s $DEVICE shell /data/local/tmp/mllm-fa-opencl-bench

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <CL/cl.h>

#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/ops/FlashAttention2Op.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Context;
using mllm::DeviceTypes;
using mllm::kCPU;
using mllm::kFloat32;
using mllm::kOpenCL;
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

// Causal-mask FA: only the lower triangle's S_q*(S_kv+1)/2 attn entries get
// real work (Q@K^T and attn@V each contribute ~2*D flops per entry). Non-causal
// would be 4*B*H*S_q*S_kv*D. For S_q==S_kv this halves to 2*B*H*S_q*S_kv*D.
double gflops_causal(const Shape& sh, double ms) {
  const double pairs = static_cast<double>(sh.S_q) * (static_cast<double>(sh.S_kv) + 1.0) * 0.5;
  const double flops = 4.0 * sh.B * sh.H * sh.D * pairs;
  return (flops / (ms * 1e-3)) / 1e9;
}

// Lower-bound traffic: read Q once, read K and V once each, write O once.
// Real FA1 re-reads K/V per Q-block, so the kernel's effective bandwidth is
// higher than this -- this number lets you compare against device peak BW.
double gbps_optimal(const Shape& sh, double ms) {
  const double bytes_per_elem = 4.0;  // fp32
  const double bh = static_cast<double>(sh.B) * sh.H * sh.D;
  const double bytes = bytes_per_elem * (2.0 * bh * sh.S_q + 2.0 * bh * sh.S_kv);
  return (bytes / (ms * 1e-3)) / 1e9;
}

void print_device_limits(mllm::opencl::OpenCLRuntime* rt) {
  const auto& devs = rt->getDevices();
  if (devs.empty()) {
    std::printf("[device] no OpenCL devices reported\n");
    return;
  }
  const auto& dev = devs.front();
  cl_ulong local_mem = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
  cl_ulong global_mem = dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  std::string name = dev.getInfo<CL_DEVICE_NAME>();
  std::printf("[device] %s  local_mem=%llu B  global_mem=%llu MB  CUs=%u  max_wg=%u\n", name.c_str(),
              static_cast<unsigned long long>(local_mem), static_cast<unsigned long long>(global_mem / (1024 * 1024)),
              rt->deviceComputeUnits(), rt->maxWorkGroupSize());
}

void print_kernel_local_mem(mllm::opencl::OpenCLRuntime* rt, int D, int kBr) {
  // Build the same kernel the op will JIT, so we can ask the driver how much
  // __local memory it actually allocates per work-group for this (D, Br).
  std::set<std::string> opts;
  opts.insert(std::string("-DFA_D=") + std::to_string(D));
  opts.insert(std::string("-DFA_BR=") + std::to_string(kBr));
  auto kw = rt->buildKernel("flash_attention", "flash_attention_fp32", opts);
  if (!kw) {
    std::printf("[kernel] buildKernel failed for D=%d Br=%d\n", D, kBr);
    return;
  }
  const auto& devs = rt->getDevices();
  if (devs.empty()) return;
  cl_ulong klocal = 0;
  kw->get().getWorkGroupInfo(devs.front(), CL_KERNEL_LOCAL_MEM_SIZE, &klocal);
  size_t kwgs = 0;
  kw->get().getWorkGroupInfo(devs.front(), CL_KERNEL_WORK_GROUP_SIZE, &kwgs);
  std::printf("[kernel] flash_attention_fp32 D=%d Br=%d  local_mem=%llu B  max_wg_size=%zu\n", D, kBr,
              static_cast<unsigned long long>(klocal), kwgs);
}

void bench_one(const Shape& sh, int warmup, int iters) {
  // BHSD layout, matches what OpenCLFlashAttention2Op::forward expects.
  // Build on CPU so Tensor::random (which goes through CPU FillOp) works,
  // then ship to the device.
  Tensor Q = Tensor::random({sh.B, sh.H, sh.S_q, sh.D}, -1.f, 1.f, kFloat32, kCPU).to(kOpenCL);
  Tensor K = Tensor::random({sh.B, sh.H, sh.S_kv, sh.D}, -1.f, 1.f, kFloat32, kCPU).to(kOpenCL);
  Tensor V = Tensor::random({sh.B, sh.H, sh.S_kv, sh.D}, -1.f, 1.f, kFloat32, kCPU).to(kOpenCL);

  mllm::aops::FlashAttention2OpOptions opts{};
  opts.B = sh.B;
  opts.q_head = sh.H;
  opts.kv_head = sh.H;  // GQA pre-expanded, kernel sees H_q == H_kv
  opts.D = sh.D;
  opts.causal_mask = true;
  mllm::opencl::OpenCLFlashAttention2Op op(opts);

  std::vector<Tensor> inputs{Q, K, V};
  std::vector<Tensor> outputs;
  op.reshape(inputs, outputs);
  op.setup(inputs, outputs);

  auto runtime = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(Context::instance().getBackend(kOpenCL))->runtime();
  auto& cq = runtime->commandQueue();

  for (int i = 0; i < warmup; ++i) {
    op.forward(inputs, outputs);
    cq.finish();
  }

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    op.forward(inputs, outputs);
    cq.finish();
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  Stats st = summarize(std::move(samples));
  std::printf("%-10s  B=%d H=%2d D=%3d  S_q=%5d S_kv=%5d  min=%8.3f ms  med=%8.3f ms  mean=%8.3f ms  %7.1f GF/s  %6.1f GB/s\n",
              sh.label, sh.B, sh.H, sh.D, sh.S_q, sh.S_kv, st.min_ms, st.median_ms, st.mean_ms,
              gflops_causal(sh, st.min_ms), gbps_optimal(sh, st.min_ms));
}

// Pre-announce a shape before any kernel runs, so if the Adreno watchdog
// kills us mid-launch we know exactly which config did it.
void announce(const Shape& sh) {
  std::printf(">> %-10s B=%d H=%2d D=%3d S_q=%5d S_kv=%5d ...\n", sh.label, sh.B, sh.H, sh.D, sh.S_q, sh.S_kv);
  std::fflush(stdout);
}

}  // namespace

MLLM_MAIN({
  // adb shell pipes stdout, so libc defaults to block buffering and we lose
  // every line before a GPU-watchdog SIGKILL. Force line buffering so each
  // printf flushes immediately.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  mllm::initOpenCLBackend();

  // Qwen3 default dims. Override if you want a different model's shape -- the
  // sweep itself only varies S_q/S_kv, which is what we care about for the
  // prefill-vs-decode regime question.
  constexpr int kB = 1;
  constexpr int kH = 16;
  constexpr int kD = 128;
  constexpr int kBr = 4;  // must match kBr in OpenCLFlashAttention2Op
  constexpr int kWarmup = 3;
  constexpr int kIters = 20;

  auto runtime = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(Context::instance().getBackend(kOpenCL))->runtime();
  print_device_limits(runtime.get());
  print_kernel_local_mem(runtime.get(), kD, kBr);
  std::printf("\n");

  std::printf("\n[mode=prefill]  S_q = S_kv = N\n");
  {
    // Skip N=4096 here: a single FA kernel launch at that shape runs for
    // ~13 s on Adreno 830, which trips the GPU watchdog (TDR) and the OS
    // SIGKILLs the process. NPU and CPU benches include 4096; OpenCL
    // cannot without splitting the kernel launch.
    const int n_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048};
    for (int n : n_sizes) {
      Shape sh{kB, kH, n, n, kD, "prefill"};
      announce(sh);
      bench_one(sh, kWarmup, kIters);
    }
  }

  std::printf("\n[mode=decode]   S_q fixed = 1, S_kv varies\n");
  {
    constexpr int kS_q = 1;
    const int s_kv_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (int s_kv : s_kv_sizes) {
      Shape sh{kB, kH, kS_q, s_kv, kD, "decode"};
      announce(sh);
      bench_one(sh, kWarmup, kIters);
    }
  }
});
