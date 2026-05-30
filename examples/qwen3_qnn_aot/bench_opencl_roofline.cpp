// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// OpenCL roofline anchors for the Adreno 830 (SM8750):
//
//   1. fp16 FMA throughput  → the COMPUTE ceiling (TFLOP/s). A purely
//      register-resident half8 FMA recurrence with 8 independent
//      accumulators to hide FMA latency; zero memory traffic in the loop,
//      one write per thread. Whatever this reaches is the hard upper bound
//      for any fp16 matmul kernel on this device.
//
//   2. global-memory read bandwidth → the MEMORY ceiling (GB/s). A
//      grid-stride half8 read-reduce over a large buffer. Sets the roofline
//      slope; a weight-streaming GEMM that is memory-bound cannot beat
//      (this BW) × (arithmetic intensity).
//
// Usage:
//   ./mllm-qwen3-opencl-roofline [--reps 20] [--iters 8192] [--mb 256]

#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {

// Compute ceiling: 16 INDEPENDENT half2 accumulator chains, one fma() each
// per iteration. half2 is Adreno's native packed-fp16 width, so each chain is
// 1 register → 16-deep ILP at minimal register pressure (≈ max occupancy).
// 16 independent chains hide FMA latency within a single thread; combined with
// many resident waves this saturates the fp16 ALUs. Runtime b,c defeat
// constant folding; data-dependent recurrence prevents loop collapse; final
// reduction + write defeats DCE.
//
// FLOPs/iter/thread = 16 fma × 2 lanes × 2 FLOP = 64.
const char* kComputeSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define A(i) a##i = fma(a##i, b, c);
__kernel void fma_peak(__global half* out, const int iters,
                       const half bv, const half cv) {
  const int gid = get_global_id(0);
  half2 b = (half2)(bv);
  half2 c = (half2)(cv);
  half2 a0=(half2)(bv+(half)(gid&1)),  a1=(half2)(cv+(half)(gid&3)),
        a2=(half2)(bv+(half)2),  a3=(half2)(cv+(half)3),
        a4=(half2)(bv+(half)4),  a5=(half2)(cv+(half)5),
        a6=(half2)(bv+(half)6),  a7=(half2)(cv+(half)7),
        a8=(half2)(bv+(half)8),  a9=(half2)(cv+(half)9),
        a10=(half2)(bv+(half)10),a11=(half2)(cv+(half)11),
        a12=(half2)(bv+(half)12),a13=(half2)(cv+(half)13),
        a14=(half2)(bv+(half)14),a15=(half2)(cv+(half)15);
  for (int i = 0; i < iters; ++i) {
    A(0) A(1) A(2) A(3) A(4) A(5) A(6) A(7)
    A(8) A(9) A(10) A(11) A(12) A(13) A(14) A(15)
  }
  half2 s = ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))
          + ((a8+a9)+(a10+a11))+((a12+a13)+(a14+a15));
  out[gid] = s.s0 + s.s1;
}
)CL";

// Memory ceiling: grid-stride half8 read-reduce. Reads the whole buffer once.
const char* kBwSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void bw_read(__global const half8* in, __global half* out, const int n8) {
  const int gid = get_global_id(0);
  const int stride = get_global_size(0);
  half8 acc = (half8)((half)0);
  for (int i = gid; i < n8; i += stride) acc += in[i];
  half r = ((acc.s0 + acc.s1) + (acc.s2 + acc.s3)) + ((acc.s4 + acc.s5) + (acc.s6 + acc.s7));
  out[gid] = r;
}
)CL";

#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fmt::print(stderr, "OpenCL error {} at {}:{} ({})\n", _e, __FILE__, __LINE__, #expr); std::exit(1); } } while(0)

cl_kernel build(cl_context ctx, mllm::opencl::OpenCLRuntime* rt, const char* src, const char* name) {
  size_t len = std::strlen(src);
  cl_int err;
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(ctx, 1, &src, &len, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[16384] = {0}; size_t ls = 0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, rt->getDevices()[0](), CL_PROGRAM_BUILD_LOG, sizeof(log), log, &ls);
    fmt::print(stderr, "build failed ({}):\n{}\n", name, log);
    std::exit(1);
  }
  cl_kernel k = OpenCLLoader::instance().clCreateKernel(prog, name, &err);
  CL_CHECK(err);
  return k;
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(20);
  auto& iters_arg = Argparse::add<int>("--iters").help("FMA loop iters").def(8192);
  auto& mb_arg = Argparse::add<int>("--mb").help("BW buffer size (MB)").def(256);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  const int reps  = reps_arg.get();
  const int iters = iters_arg.get();
  const int mb    = mb_arg.get();

  fmt::print("[roofline] initializing OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();
  cl_int err;

  // ---- 1. fp16 FMA compute ceiling (delta-method) ------------------------
  {
    cl_kernel k = build(ctx, rt.get(), kComputeSrc, "fma_peak");
    const size_t threads = (size_t)1 << 16;   // 65,536 work-items (full occupancy)
    cl_mem out = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, threads * 2, nullptr, &err);
    CL_CHECK(err);
    cl_half bv = 0x3C00; // 1.0h  (identity mul; runtime arg so not folded)
    cl_half cv = 0x1400; // ~2.4e-4h — NORMAL half (smallest normal is 0x0400);
                         // avoids denormal-handling slowpath in the FMA loop.
    auto time_at = [&](int it) {
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 0, sizeof(cl_mem), &out));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 1, sizeof(int), &it));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 2, sizeof(cl_half), &bv));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 3, sizeof(cl_half), &cv));
      size_t g = threads, l = 128;
      auto one = [&]{
        CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
        CL_CHECK(OpenCLLoader::instance().clFinish(q));
      };
      one();  // warmup
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < reps; ++i) one();
      auto t1 = std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    };
    // Delta cancels fixed per-dispatch overhead: rate = dFLOP / dTime.
    double ms_lo = time_at(iters);
    double ms_hi = time_at(iters * 2);
    double dflops = (double)threads * (double)iters * 64.0;   // 64 FLOP/iter/thread
    double dms = ms_hi - ms_lo;
    double rate = dflops / (dms * 1e6);
    fmt::print("\n=== fp16 FMA compute ceiling (delta-method) ===\n");
    fmt::print("  threads={}  iters {}->{}  t_lo={:.3f} t_hi={:.3f} ms (delta={:.3f})\n",
               threads, iters, iters * 2, ms_lo, ms_hi, dms);
    fmt::print("  -> {:.1f} GFLOP/s  ({:.2f} TFLOP/s)\n", rate, rate / 1e3);
    OpenCLLoader::instance().clReleaseMemObject(out);
  }

  // ---- 2. global read bandwidth ceiling ----------------------------------
  {
    cl_kernel k = build(ctx, rt.get(), kBwSrc, "bw_read");
    const size_t bytes = (size_t)mb * 1024 * 1024;
    const int n8 = (int)(bytes / 16);   // # of half8 elements
    std::vector<__fp16> host(bytes / 2, (__fp16)1.0f);
    cl_mem in = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q, in, CL_TRUE, 0, bytes, host.data(), 0, nullptr, nullptr));
    const size_t threads = 1 << 16;
    cl_mem out = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, threads * 2, nullptr, &err);
    CL_CHECK(err);
    auto run = [&]{
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 0, sizeof(cl_mem), &in));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 1, sizeof(cl_mem), &out));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, 2, sizeof(int), &n8));
      size_t g = threads, l = 128;
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
    };
    run();  // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    fmt::print("\n=== global read bandwidth ceiling ===\n");
    fmt::print("  buffer={} MB  {:.3f} ms  -> {:.1f} GB/s\n", mb, ms, (double)bytes / (ms * 1e6));
    OpenCLLoader::instance().clReleaseMemObject(in);
    OpenCLLoader::instance().clReleaseMemObject(out);
  }
});
