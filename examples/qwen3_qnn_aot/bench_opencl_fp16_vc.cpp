// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// OpenCL fp16 GEMM microbench — variant C.
//
// Faithful fp16 port of llama.cpp's kernel_gemm_noshuffle_q4_0_f32 (the ~2 TF
// Adreno GEMM), with the int4 dequant removed and weights stored as real fp16.
//
// The ONE thing variant A was missing: amortized weight loads. The q4_0 kernel
// loads 4 consecutive K-values for 4 output channels in a SINGLE vector load
// (a packed ushort4) and issues 16 FMAs per load. Variant A instead did 4
// separate vload4's + four 64-bit address computations per 4 K-values, so it
// was issue-bound on the load/address path. Variant C replicates the q4_0
// amortization for fp16: weights are repacked to [K/4, N, 4] so ONE vload16
// pulls 4 channels × 4 K-values (16 halfs), feeding 16 FMAs.
//
//   acts (image1d_buffer of half4): [K, M] half (transposed; 8 tokens/thread)
//   weights (__global half*):       [K/4, N, 4] half  (4 K-values contiguous)
//   dst (__global half*):           [N, M] half
//
// Each thread computes an 8 M(token) × 4 N(out-channel) tile, exactly like v4.
//
// Usage:  ./mllm-qwen3-fp16-gemm-vc [--reps 30] [--lx 8] [--ly 16]

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

const char* kFp16VcKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_FULL __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_FULL
#endif

// Weights: [K/4, N, 4] half. For K-group g=i/4 and the 4 output channels this
// thread owns (gx*4 .. gx*4+3), the 16 halfs (4 chan × 4 K) are contiguous:
//   base = (g*N + gx*4)*4  → [c0k0,c0k1,c0k2,c0k3, c1k0..c1k3, c2.., c3..]
REQD_SUBGROUP_SIZE_FULL
__kernel void fp16_gemm_vc(
    __read_only image1d_buffer_t acts,    // [K, M] half (1d image of half4)
    __global const half*         weights,  // [K/4, N, 4] half
    __global       half*         dst,      // [N, M] half
    const int M, const int N, const int K) {
  const int gy   = get_global_id(0);     // M-tile index (8 M-rows / thread)
  const int gx   = get_global_id(1);     // N-tile index (4 N-cols / thread)
  const int gx_4 = gx << 2;
  const int M_4  = M >> 2;

  half8 c0 = (half8)((half)0), c1 = (half8)((half)0);
  half8 c2 = (half8)((half)0), c3 = (half8)((half)0);
  half8 B0, B1, B2, B3;

  for (int i = 0; i < K; i += 4) {
    // 8 tokens at each of K=i..i+3 (two half4 reads per K).
    B0.s0123 = read_imageh(acts, gy * 2 + (i + 0) * M_4);
    B0.s4567 = read_imageh(acts, gy * 2 + (i + 0) * M_4 + 1);
    B1.s0123 = read_imageh(acts, gy * 2 + (i + 1) * M_4);
    B1.s4567 = read_imageh(acts, gy * 2 + (i + 1) * M_4 + 1);
    B2.s0123 = read_imageh(acts, gy * 2 + (i + 2) * M_4);
    B2.s4567 = read_imageh(acts, gy * 2 + (i + 2) * M_4 + 1);
    B3.s0123 = read_imageh(acts, gy * 2 + (i + 3) * M_4);
    B3.s4567 = read_imageh(acts, gy * 2 + (i + 3) * M_4 + 1);

    // ONE load: 4 channels × 4 K-values = 16 halfs.
    half16 w = vload16(0, weights + ((long)(i >> 2) * N + gx_4) * 4);

    c0 += B0 * w.s0; c0 += B1 * w.s1; c0 += B2 * w.s2; c0 += B3 * w.s3;
    c1 += B0 * w.s4; c1 += B1 * w.s5; c1 += B2 * w.s6; c1 += B3 * w.s7;
    c2 += B0 * w.s8; c2 += B1 * w.s9; c2 += B2 * w.sa; c2 += B3 * w.sb;
    c3 += B0 * w.sc; c3 += B1 * w.sd; c3 += B2 * w.se; c3 += B3 * w.sf;
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

void transpose_MK_to_KM(int M, int K, const __fp16* src, __fp16* dst) {
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) dst[(long)k * M + m] = src[(long)m * K + k];
}

// Repack weights [N, K] -> [K/4, N, 4]: dst[(g*N + n)*4 + kk] = W_NK[n*K + g*4 + kk].
void pack_W_NK_to_K4N4(int N, int K, const __fp16* W_NK, __fp16* W_p) {
  const int K_4 = K / 4;
  for (int g = 0; g < K_4; ++g)
    for (int n = 0; n < N; ++n)
      for (int kk = 0; kk < 4; ++kk)
        W_p[((long)g * N + n) * 4 + kk] = W_NK[(long)n * K + g * 4 + kk];
}

void cpu_ref_first_row(int M, int K, int N, int N_check, const __fp16* A_MK, const __fp16* W_NK, float* out) {
  (void)M;
  for (int n = 0; n < N_check; ++n) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) acc += (double)(float)A_MK[k] * (double)(float)W_NK[(long)n * K + k];
    out[n] = (float)acc;
  }
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(30);
  auto& lx_arg = Argparse::add<int>("--lx").help("local dim0").def(8);
  auto& ly_arg = Argparse::add<int>("--ly").help("local dim1").def(16);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int reps = reps_arg.get(), lx = lx_arg.get(), ly = ly_arg.get();

  fmt::print("[opencl-fp16-vc] init; local={{{}, {}}}\n", lx, ly);
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  size_t src_len = std::strlen(kFp16VcKernelSrc);
  cl_int err;
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(ctx, 1, &kFp16VcKernelSrc, &src_len, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[16384] = {0}; size_t ls = 0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, rt->getDevices()[0](), CL_PROGRAM_BUILD_LOG, sizeof(log), log, &ls);
    fmt::print(stderr, "OpenCL build failed:\n{}\n", log);
    return 1;
  }
  cl_kernel k_gemm = OpenCLLoader::instance().clCreateKernel(prog, "fp16_gemm_vc", &err);
  CL_CHECK(err);

  struct Shape { int M, K, N; };
  std::vector<Shape> shapes = {
    {128,  2048, 2048}, {512, 2048, 2048}, {1024, 2048, 2048},
    {1024, 2048, 6144}, {1024, 6144, 2048},
    {2048, 256, 256}, {4096, 512, 512}, {8192, 512, 512},
  };

  std::mt19937 rng(0xc0ffee);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  constexpr double kPeak3 = 3.0e12;

  fmt::print("\n=== OpenCL fp16-GEMM variant C (q4_0-style packed loads, reps={}) ===\n", reps);
  fmt::print("  {:>5} {:>5} {:>5}   {:>8}   {:>10}   {:>9}   {:>7}\n", "M", "K", "N", "ms", "GFLOP/s", "max_rel", "%peak");

  for (auto sh : shapes) {
    const int M = sh.M, K = sh.K, N = sh.N;
    if ((M % 8) != 0 || (N % 4) != 0 || (K % 4) != 0) continue;

    std::vector<__fp16> A_MK((size_t)M * K), W_NK((size_t)N * K);
    for (auto& x : A_MK) x = (__fp16)dist(rng);
    for (auto& x : W_NK) x = (__fp16)dist(rng);

    std::vector<__fp16> A_KM((size_t)M * K);
    transpose_MK_to_KM(M, K, A_MK.data(), A_KM.data());
    cl_mem d_A_buf = create_and_upload(ctx, q, A_KM.data(), A_KM.size() * 2, CL_MEM_READ_ONLY);
    cl_image_format img_fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc img_desc = {};
    img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc.image_width = (size_t)M * K / 4;
    img_desc.buffer = d_A_buf;
    cl_mem d_A_img = OpenCLLoader::instance().clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &img_desc, nullptr, &err);
    CL_CHECK(err);

    std::vector<__fp16> W_p((size_t)N * K);
    pack_W_NK_to_K4N4(N, K, W_NK.data(), W_p.data());
    cl_mem d_W = create_and_upload(ctx, q, W_p.data(), W_p.size() * 2, CL_MEM_READ_ONLY);
    cl_mem d_C_NM = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)M * N * 2, nullptr, &err);
    CL_CHECK(err);

    auto run_once = [&]() {
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 0, sizeof(cl_mem), &d_A_img));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 1, sizeof(cl_mem), &d_W));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 2, sizeof(cl_mem), &d_C_NM));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 3, sizeof(int), &M));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 4, sizeof(int), &N));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 5, sizeof(int), &K));
      size_t global[2] = {(size_t)M / 8, (size_t)N / 4};
      size_t local[2]  = {(size_t)lx, (size_t)ly};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q, k_gemm, 2, nullptr, global, local, 0, nullptr, nullptr));
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
    };

    run_once();
    std::vector<__fp16> out_NM((size_t)M * N);
    CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, d_C_NM, CL_TRUE, 0, out_NM.size() * 2, out_NM.data(), 0, nullptr, nullptr));
    const int N_check = std::min(64, N);
    std::vector<float> ref(N_check);
    cpu_ref_first_row(M, K, N, N_check, A_MK.data(), W_NK.data(), ref.data());
    double max_rel = 0.0;
    for (int n = 0; n < N_check; ++n) {
      float g = (float)out_NM[(long)n * M + 0], r = ref[n];
      max_rel = std::max(max_rel, std::abs((double)g - (double)r) / std::max(1e-6, std::abs((double)r)));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double flops = 2.0 * M * (double)K * (double)N;
    fmt::print("  {:>5} {:>5} {:>5}   {:>7.3f}    {:>9.2f}   {:>.3e}   {:>6.1f}%\n",
               M, K, N, ms, flops / (ms * 1e6), max_rel, 100.0 * (flops / (ms * 1e-3)) / kPeak3);

    OpenCLLoader::instance().clReleaseMemObject(d_A_buf);
    OpenCLLoader::instance().clReleaseMemObject(d_A_img);
    OpenCLLoader::instance().clReleaseMemObject(d_W);
    OpenCLLoader::instance().clReleaseMemObject(d_C_NM);
  }
  OpenCLLoader::instance().clReleaseKernel(k_gemm);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
