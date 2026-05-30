// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// OpenCL fp32 GEMM microbench — variant A, fp32 twin of bench_opencl_fp16_va.cpp.
//
// IDENTICAL algorithm / tiling / access pattern to the fp16 variant A; the ONLY
// difference is the datatype (half -> float everywhere). This isolates the
// effect of arithmetic precision on this Adreno: if fp32 ~= fp16 GFLOP/s, the
// device gives no fp16 packing speedup (and the bottleneck is issue/structure,
// not the ALU's fp16 rate); if fp32 is ~2x slower, fp16 is genuinely faster.
//
//   acts (image1d_buffer of float4):  [K, M] float
//   weights (__global float*):        [K, N/4, 4] float (OC-interleaved)
//   dst (__global float*):            [N, M] float
//
// Bench shapes match the fp16 twin (LLM projections + cache-resident probes).
//
// Usage:  ./mllm-qwen3-fp32-gemm-va [--reps 30]

#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {

// Pure fp32 GEMM, structurally identical to fp16_gemm_va. 8 M-rows x 4 N-cols
// per thread; float8 register accumulators; broadcast-scalar FMA; no __local.
const char* kFp32VaKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_FULL __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_FULL
#endif

REQD_SUBGROUP_SIZE_FULL
__kernel void fp32_gemm_va(
    __read_only image1d_buffer_t acts,    // [K, M] float (1d image of float4)
    __global const float*        weights,  // [K, N/4, 4] float (OC-interleaved)
    __global       float*        dst,      // [N, M] float
    const int M, const int N, const int K) {
  const int gy   = get_global_id(0);     // M-tile index (8 M-rows / thread)
  const int gx   = get_global_id(1);     // N-tile index (4 N-cols / thread)
  const int gx_4 = gx << 2;
  const int M_4  = M >> 2;
  const int N_4  = N >> 2;

  float8 c0 = (float8)(0.0f);
  float8 c1 = (float8)(0.0f);
  float8 c2 = (float8)(0.0f);
  float8 c3 = (float8)(0.0f);
  float8 B;

  for (int i = 0; i < K; i += 4) {
    float4 w = vload4(0, weights + ((long)(i + 0) * N_4 + gx) * 4);
    B.s0123 = read_imagef(acts, gy * 2 + (i + 0) * M_4);
    B.s4567 = read_imagef(acts, gy * 2 + (i + 0) * M_4 + 1);
    c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;

    w = vload4(0, weights + ((long)(i + 1) * N_4 + gx) * 4);
    B.s0123 = read_imagef(acts, gy * 2 + (i + 1) * M_4);
    B.s4567 = read_imagef(acts, gy * 2 + (i + 1) * M_4 + 1);
    c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;

    w = vload4(0, weights + ((long)(i + 2) * N_4 + gx) * 4);
    B.s0123 = read_imagef(acts, gy * 2 + (i + 2) * M_4);
    B.s4567 = read_imagef(acts, gy * 2 + (i + 2) * M_4 + 1);
    c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;

    w = vload4(0, weights + ((long)(i + 3) * N_4 + gx) * 4);
    B.s0123 = read_imagef(acts, gy * 2 + (i + 3) * M_4);
    B.s4567 = read_imagef(acts, gy * 2 + (i + 3) * M_4 + 1);
    c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
  }

  #define WRITE_M(SM) {                                                       \
    dst[(gx_4 + 0) * M + (gy << 3) + (SM)] = c0.s##SM;                        \
    dst[(gx_4 + 1) * M + (gy << 3) + (SM)] = c1.s##SM;                        \
    dst[(gx_4 + 2) * M + (gy << 3) + (SM)] = c2.s##SM;                        \
    dst[(gx_4 + 3) * M + (gy << 3) + (SM)] = c3.s##SM;                        \
  }
  WRITE_M(0); WRITE_M(1); WRITE_M(2); WRITE_M(3);
  WRITE_M(4); WRITE_M(5); WRITE_M(6); WRITE_M(7);
  #undef WRITE_M
}
)CL";

#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fmt::print(stderr, "OpenCL error {} at {}:{} ({})\n", _e, __FILE__, __LINE__, #expr); std::exit(1); } } while(0)

cl_mem create_and_upload(cl_context ctx, cl_command_queue q, const void* host, size_t bytes, cl_mem_flags flags) {
  cl_int err;
  cl_mem buf = OpenCLLoader::instance().clCreateBuffer(ctx, flags, bytes, nullptr, &err);
  CL_CHECK(err);
  CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
  return buf;
}

void transpose_MK_to_KM(int M, int K, const float* src, float* dst) {
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) dst[(long)k * M + m] = src[(long)m * K + k];
}

void interleave_W_NK_to_KN4(int N, int K, const float* W_NK, float* W_il) {
  const int N_4 = N / 4;
  for (int k = 0; k < K; ++k)
    for (int ng = 0; ng < N_4; ++ng)
      for (int il = 0; il < 4; ++il)
        W_il[((long)k * N_4 + ng) * 4 + il] = W_NK[(long)(ng * 4 + il) * K + k];
}

void cpu_ref_first_row(int M, int K, int N, int N_check, const float* A_MK, const float* W_NK, float* out) {
  (void)M;
  for (int n = 0; n < N_check; ++n) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) acc += (double)A_MK[k] * (double)W_NK[(long)n * K + k];
    out[n] = (float)acc;
  }
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(30);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int reps = reps_arg.get();

  fmt::print("[opencl-fp32-va] initializing OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  size_t src_len = std::strlen(kFp32VaKernelSrc);
  cl_int err;
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(ctx, 1, &kFp32VaKernelSrc, &src_len, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[16384] = {0}; size_t ls = 0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, rt->getDevices()[0](), CL_PROGRAM_BUILD_LOG, sizeof(log), log, &ls);
    fmt::print(stderr, "OpenCL build failed:\n{}\n", log);
    return 1;
  }
  cl_kernel k_gemm = OpenCLLoader::instance().clCreateKernel(prog, "fp32_gemm_va", &err);
  CL_CHECK(err);

  struct Shape { int M, K, N; };
  std::vector<Shape> shapes = {
    {128,  2048, 2048},
    {512,  2048, 2048},
    {1024, 2048, 2048},
    {1024, 2048, 6144},
    {1024, 6144, 2048},
    // compute-bound probes (weights fit/near the 1 MB L2)
    {2048, 256,  256},
    {4096, 256,  256},
    {4096, 512,  512},
    {8192, 256,  256},
  };

  std::mt19937 rng(0xc0ffee);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  constexpr double kPeakFp32 = 1.5e12;  // Adreno 830 fp32 peak (half of fp16 3 TF, by assumption)

  fmt::print("\n=== OpenCL fp32-GEMM variant A bench (reps={}) ===\n", reps);
  fmt::print("  {:>5} {:>5} {:>5}   {:>8}   {:>10}   {:>8}   {:>9}   {:>7}\n",
             "M", "K", "N", "ms", "GFLOP/s", "GB/s", "max_rel", "%fp32pk");

  for (auto sh : shapes) {
    const int M = sh.M, K = sh.K, N = sh.N;
    if ((M % 8) != 0 || (N % 4) != 0 || (K % 4) != 0) continue;

    std::vector<float> A_MK((size_t)M * K), W_NK((size_t)N * K);
    for (auto& x : A_MK) x = dist(rng);
    for (auto& x : W_NK) x = dist(rng);

    std::vector<float> A_KM((size_t)M * K);
    transpose_MK_to_KM(M, K, A_MK.data(), A_KM.data());

    cl_mem d_A_buf = create_and_upload(ctx, q, A_KM.data(), A_KM.size() * 4, CL_MEM_READ_ONLY);
    cl_image_format img_fmt = {CL_RGBA, CL_FLOAT};
    cl_image_desc img_desc = {};
    img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc.image_width = (size_t)M * K / 4;
    img_desc.buffer      = d_A_buf;
    cl_mem d_A_img = OpenCLLoader::instance().clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &img_desc, nullptr, &err);
    CL_CHECK(err);

    std::vector<float> W_il((size_t)N * K);
    interleave_W_NK_to_KN4(N, K, W_NK.data(), W_il.data());
    cl_mem d_W = create_and_upload(ctx, q, W_il.data(), W_il.size() * 4, CL_MEM_READ_ONLY);
    cl_mem d_C_NM = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)M * N * 4, nullptr, &err);
    CL_CHECK(err);

    auto run_once = [&]() {
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 0, sizeof(cl_mem), &d_A_img));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 1, sizeof(cl_mem), &d_W));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 2, sizeof(cl_mem), &d_C_NM));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 3, sizeof(int), &M));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 4, sizeof(int), &N));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 5, sizeof(int), &K));
      size_t global[2] = {(size_t)M / 8, (size_t)N / 4};
      size_t local[2]  = {8, 16};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q, k_gemm, 2, nullptr, global, local, 0, nullptr, nullptr));
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
    };

    run_once();
    std::vector<float> out_NM((size_t)M * N);
    CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, d_C_NM, CL_TRUE, 0, out_NM.size() * 4, out_NM.data(), 0, nullptr, nullptr));
    const int N_check = std::min(64, N);
    std::vector<float> ref(N_check);
    cpu_ref_first_row(M, K, N, N_check, A_MK.data(), W_NK.data(), ref.data());
    double max_rel = 0.0;
    for (int n = 0; n < N_check; ++n) {
      float g = out_NM[(long)n * M + 0], r = ref[n];
      max_rel = std::max(max_rel, std::abs((double)g - (double)r) / std::max(1e-6, std::abs((double)r)));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double flops = 2.0 * M * (double)K * (double)N;
    double bytes = 4.0 * ((double)N * K + (double)M * K + (double)M * N);
    fmt::print("  {:>5} {:>5} {:>5}   {:>7.3f}    {:>9.2f}   {:>7.2f}   {:>.3e}   {:>6.1f}%\n",
               M, K, N, ms, flops / (ms * 1e6), bytes / (ms * 1e6), max_rel,
               100.0 * (flops / (ms * 1e-3)) / kPeakFp32);

    OpenCLLoader::instance().clReleaseMemObject(d_A_buf);
    OpenCLLoader::instance().clReleaseMemObject(d_A_img);
    OpenCLLoader::instance().clReleaseMemObject(d_W);
    OpenCLLoader::instance().clReleaseMemObject(d_C_NM);
  }

  OpenCLLoader::instance().clReleaseKernel(k_gemm);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
