// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Op-level correctness test for the OpenCL LPBQ Linear path. Drives
// OpenCLLinearOp::setLPBQ + runLPBQ on synthetic w4a16 weights and compares the
// GPU output to the CPU reference (mllm::cpu::lpbq_matmul_fp16_packed). Mirrors
// the data flow validated in bench_opencl_lpbq.cpp, but exercises the actual
// backend op (prepack layout + activation transpose/image + v5/gemv dispatch).
//
// Usage:  ./mllm-qwen3-opencl-linear-lpbq-test

#include <CL/cl.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/ops/LinearOp.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/backends/cpu/kernels/common/lpbq_matmul.hpp"

using mllm::opencl::OpenCLLoader;

namespace {
#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fmt::print(stderr, "OpenCL error {} at {}:{}\n", _e, __FILE__, __LINE__); std::exit(1); } } while(0)

cl_mem upload(cl_context ctx, cl_command_queue q, const void* host, size_t bytes, cl_mem_flags f) {
  cl_int err; cl_mem b = OpenCLLoader::instance().clCreateBuffer(ctx, f, bytes, nullptr, &err); CL_CHECK(err);
  CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q, b, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
  return b;
}
}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  fmt::print("[lpbq-op-test] init OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  const int Bs = 16;
  struct Shape { int M, K, N; };
  std::vector<Shape> shapes = {
    {1, 2048, 2048},      // decode (gemv)
    {128, 2048, 2048},    // prefill
    {512, 2048, 2048},
    {1024, 2048, 6144},   // gate/up
    {1024, 6144, 2048},   // down
  };

  std::mt19937 rng(0x1234);
  std::uniform_real_distribution<float> ad(-0.1f, 0.1f);
  std::uniform_int_distribution<int> nibd(0, 15);
  std::uniform_int_distribution<int> s1d(1, 8);
  std::uniform_real_distribution<float> s2d(0.002f, 0.01f);

  int failures = 0;
  fmt::print("\n=== OpenCL LPBQ LinearOp vs CPU ===\n");
  fmt::print("  {:>5} {:>5} {:>5}   {:>10}   {:>10}   {:>8}\n", "M", "K", "N", "max_abs", "max_ref", "result");

  for (auto sh : shapes) {
    const int M = sh.M, K = sh.K, N = sh.N;
    const int nb = K / Bs;

    // Synthetic LPBQ tensors. nibble[n][k] (0..15) shared by both pipelines.
    std::vector<uint8_t> in_KN((size_t)K * N);   // [K,N] one nibble/byte (USHORT4 input)
    std::vector<uint8_t> w_packed((size_t)N * K); // [N,K] one nibble/byte (CPU-ref input)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        uint8_t nib = (uint8_t)nibd(rng);
        in_KN[(size_t)k * N + n] = nib;
        w_packed[(size_t)n * K + k] = nib;
      }
    std::vector<uint8_t> s1((size_t)N * nb);
    for (auto& x : s1) x = (uint8_t)s1d(rng);
    std::vector<float> s2((size_t)N);
    for (auto& x : s2) x = s2d(rng);
    std::vector<__fp16> act((size_t)M * K);
    for (auto& x : act) x = (__fp16)ad(rng);

    // CPU reference.
    std::vector<__fp16> out_cpu((size_t)M * N);
    mllm::cpu::lpbq_matmul_fp16_packed(M, N, K, Bs, act.data(), w_packed.data(),
                                       s1.data(), s2.data(), nullptr, out_cpu.data());

    // GPU prepack (the validated bench layouts).
    std::vector<uint16_t> w_ushort((size_t)K * N / 4);
    mllm::cpu::lpbq_prepack_weights_USHORT4(K, N, in_KN.data(), w_ushort.data());
    std::vector<uint16_t> comb((size_t)N * nb);
    mllm::cpu::lpbq_prepack_combined_scales(N, nb, s1.data(), s2.data(), comb.data());

    cl_mem d_W = upload(ctx, q, w_ushort.data(), w_ushort.size() * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_S = upload(ctx, q, comb.data(), comb.size() * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_in = upload(ctx, q, act.data(), act.size() * 2, CL_MEM_READ_ONLY);
    cl_int err;
    cl_mem d_out = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * N * 2, nullptr, &err);
    CL_CHECK(err);

    // Build the op and run the LPBQ path.
    mllm::aops::LinearOpOptions opt;
    opt.in_channels = K;
    opt.out_channels = N;
    opt.bias = false;
    opt.impl_type = mllm::aops::LinearImplTypes::kDefault;
    mllm::opencl::OpenCLLinearOp op(opt);
    op.setLPBQ(d_W, d_S, K, N, Bs);
    op.runLPBQ(d_in, d_out, M);

    std::vector<__fp16> out_gpu((size_t)M * N);
    CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, out_gpu.size() * 2, out_gpu.data(), 0, nullptr, nullptr));

    // Compare first row.
    double max_abs = 0.0, max_ref = 0.0;
    int N_check = std::min(64, N);
    for (int n = 0; n < N_check; ++n) {
      double g = (float)out_gpu[n], r = (float)out_cpu[n];
      max_abs = std::max(max_abs, std::abs(g - r));
      max_ref = std::max(max_ref, std::abs(r));
    }
    // fp16-accumulation tolerance: the kernel accumulates in half over K terms
    // while the CPU ref uses fp32, so divergence grows with K (~0.5% at
    // K=2048, ~1.6% at K=6144). 3% relative is the honest fp16-GEMM bound.
    bool ok = max_abs < 3e-2 * std::max(1e-3, max_ref) + 5e-3;
    if (!ok) ++failures;
    fmt::print("  {:>5} {:>5} {:>5}   {:>.3e}   {:>.3e}   {:>8}\n",
               M, K, N, max_abs, max_ref, ok ? "PASS" : "FAIL");

    OpenCLLoader::instance().clReleaseMemObject(d_W);
    OpenCLLoader::instance().clReleaseMemObject(d_S);
    OpenCLLoader::instance().clReleaseMemObject(d_in);
    OpenCLLoader::instance().clReleaseMemObject(d_out);
  }

  fmt::print("\n{}\n", failures == 0 ? "ALL PASS" : fmt::format("{} FAILURES", failures));
  return failures == 0 ? 0 : 1;
});
