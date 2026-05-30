// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// OpenCL fp16 GEMM microbench — variant A.
//
// Derived from the v4 LPBQ kernel (see bench_opencl_lpbq.cpp,
// lpbq_gemm_fp16_v4) with the 4-bit dequant logic stripped out: weights
// are real fp16 here. Inherits the Adreno-tuned v4 idioms:
//
//   * cl_qcom_reqd_sub_group_size("full")  → 128-wide subgroup
//   * Activations come from an image1d_buffer (texture cache) of half4
//   * 8 M-rows × 4 N-cols output tile per thread (32 outputs / thread)
//   * half8 register accumulators c0..c3 (one per N-col, 8 M-rows each)
//   * Broadcast scalar half8 FMA:  c0 += B * w_scalar  (the v4 trick)
//   * No __local memory
//
// Layouts:
//   acts (image1d_buffer of half4):  [K, M] row-major in half — so
//       read_imageh(acts, k * (M/4) + m/4) returns half4 of 4
//       consecutive M values at K=k.
//   weights (__global half*):        [N, K] row-major in half — one
//       contiguous K-row per output column.
//   dst (__global half*):            [N, M] row-major in half (same as
//       v4; transposed back to [M, N] on the host for the check).
//
// Bench shapes:
//   (M, K, N) ∈ { (128, 2048, 2048), (512, 2048, 2048),
//                 (1024, 2048, 2048), (1024, 2048, 6144),
//                 (1024, 6144, 2048) }
// Inputs are random fp16 ∈ [-0.1, 0.1]; first-row check vs fp32 CPU ref.
//
// Usage:
//   ./mllm-qwen3-fp16-gemm-va [--reps 5]

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

// ---------------------------------------------------------------------------
// Variant A: pure fp16 × fp16 GEMM. Same tile shape, same Adreno idioms,
// no dequant. One thread owns an 8 M × 4 N tile; the K-loop steps by 4 to
// mirror v4's BS=16-with-4-K-per-iter cadence, so the per-iter weight
// footprint (4 fp16s × 4 OCs = 32 B = one vload16 of half) is identical.
// ---------------------------------------------------------------------------
const char* kFp16VaKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_FULL __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_FULL
#endif

// Pure fp16 GEMM, v4-identical structure. The ONLY change from v4 is the
// weight source: direct fp16 read instead of int4-unpack+dequant.
//
// Weights use v4's OC-interleaved layout [K, N/4, 4] (fp16). At K-position i,
// the 4 OC weights this thread owns are CONTIGUOUS (one vload4), and adjacent
// threads (gx, gx+1) read adjacent 8-byte chunks → coalesced across the
// subgroup. (A naive [N, K] row-major layout makes the 4 OCs K-elements apart
// and adjacent threads stride by 4 rows → uncoalesced, ~2× slower.)
REQD_SUBGROUP_SIZE_FULL
__kernel void fp16_gemm_va(
    __read_only image1d_buffer_t acts,    // [K, M] half (1d image of half4)
    __global const half*         weights,  // [K, N/4, 4] half (OC-interleaved)
    __global       half*         dst,      // [N, M] half
    const int M, const int N, const int K) {
  const int gy   = get_global_id(0);     // M-tile index (8 M-rows / thread)
  const int gx   = get_global_id(1);     // N-tile index (4 N-cols / thread)
  const int gx_4 = gx << 2;               // base N for this thread
  const int M_4  = M >> 2;                // M/4 (image stride along K)
  const int N_4  = N >> 2;                // N/4 (weight inner stride)

  // 4 N-cols × 8 M-rows of half registers. c0.sM is OC=gx_4+0 at M=(gy*8)+M.
  half8 c0 = (half8)((half)0);
  half8 c1 = (half8)((half)0);
  half8 c2 = (half8)((half)0);
  half8 c3 = (half8)((half)0);
  half8 B;

  // Unroll K by 4 to match v4's cadence and amortize loop overhead.
  for (int i = 0; i < K; i += 4) {
    // j=0 (K = i+0): 4 OC weights, contiguous, coalesced across subgroup.
    half4 w = vload4(0, weights + ((long)(i + 0) * N_4 + gx) * 4);
    B.s0123 = read_imageh(acts, gy * 2 + (i + 0) * M_4);
    B.s4567 = read_imageh(acts, gy * 2 + (i + 0) * M_4 + 1);
    c0 += B * w.s0;
    c1 += B * w.s1;
    c2 += B * w.s2;
    c3 += B * w.s3;

    // j=1 (K = i+1)
    w = vload4(0, weights + ((long)(i + 1) * N_4 + gx) * 4);
    B.s0123 = read_imageh(acts, gy * 2 + (i + 1) * M_4);
    B.s4567 = read_imageh(acts, gy * 2 + (i + 1) * M_4 + 1);
    c0 += B * w.s0;
    c1 += B * w.s1;
    c2 += B * w.s2;
    c3 += B * w.s3;

    // j=2 (K = i+2)
    w = vload4(0, weights + ((long)(i + 2) * N_4 + gx) * 4);
    B.s0123 = read_imageh(acts, gy * 2 + (i + 2) * M_4);
    B.s4567 = read_imageh(acts, gy * 2 + (i + 2) * M_4 + 1);
    c0 += B * w.s0;
    c1 += B * w.s1;
    c2 += B * w.s2;
    c3 += B * w.s3;

    // j=3 (K = i+3)
    w = vload4(0, weights + ((long)(i + 3) * N_4 + gx) * 4);
    B.s0123 = read_imageh(acts, gy * 2 + (i + 3) * M_4);
    B.s4567 = read_imageh(acts, gy * 2 + (i + 3) * M_4 + 1);
    c0 += B * w.s0;
    c1 += B * w.s1;
    c2 += B * w.s2;
    c3 += B * w.s3;
  }

  // Store 8 M × 4 N into dst[N, M] (stride M between OCs). Same as v4.
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

cl_mem create_and_upload(cl_context ctx, cl_command_queue q, const void* host, size_t bytes,
                         cl_mem_flags flags) {
  cl_int err;
  cl_mem buf = OpenCLLoader::instance().clCreateBuffer(ctx, flags, bytes, nullptr, &err);
  CL_CHECK(err);
  CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
  return buf;
}

// Transpose acts [M, K] (host fp16) → [K, M] (host fp16). Used as the
// image1d_buffer backing storage, matching the v4 acts layout exactly.
void transpose_MK_to_KM(int M, int K, const __fp16* src, __fp16* dst) {
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      dst[(long)k * M + m] = src[(long)m * K + k];
}

// Interleave weights [N, K] → [K, N/4, 4] (OC-interleaved, fp16), so the
// kernel's per-K vload4 reads 4 OC weights contiguously and adjacent threads
// coalesce. dst index for (k, oc=ng*4+il): (k * (N/4) + ng) * 4 + il.
void interleave_W_NK_to_KN4(int N, int K, const __fp16* W_NK, __fp16* W_il) {
  const int N_4 = N / 4;
  for (int k = 0; k < K; ++k)
    for (int ng = 0; ng < N_4; ++ng)
      for (int il = 0; il < 4; ++il)
        W_il[((long)k * N_4 + ng) * 4 + il] = W_NK[(long)(ng * 4 + il) * K + k];
}

// fp32 CPU reference, first-row only (n = 0..N_check-1). Matches what the
// LPBQ bench does for correctness checks.
void cpu_ref_first_row(int M, int K, int N, int N_check,
                       const __fp16* A_MK,    // [M, K]
                       const __fp16* W_NK,    // [N, K]
                       float* out_first_row /*[N]*/) {
  (void)M;
  for (int n = 0; n < N_check; ++n) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      acc += (double)(float)A_MK[k] * (double)(float)W_NK[(long)n * K + k];
    }
    out_first_row[n] = (float)acc;
  }
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(5);
  auto& lx_arg = Argparse::add<int>("--lx").help("local size dim0 (M/8 dim)").def(8);
  auto& ly_arg = Argparse::add<int>("--ly").help("local size dim1 (N/4 dim)").def(16);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  const int reps = reps_arg.get();
  const int lx = lx_arg.get();
  const int ly = ly_arg.get();
  fmt::print("[opencl-fp16-va] local work size = {{{}, {}}}\n", lx, ly);

  fmt::print("[opencl-fp16-va] initializing OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  // Build the inline program.
  size_t src_len = std::strlen(kFp16VaKernelSrc);
  cl_int err;
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(
      ctx, 1, &kFp16VaKernelSrc, &src_len, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[16384] = {0};
    size_t log_size = 0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, rt->getDevices()[0](),
        CL_PROGRAM_BUILD_LOG, sizeof(log), log, &log_size);
    fmt::print(stderr, "OpenCL build failed:\n{}\n", log);
    return 1;
  }
  cl_kernel k_gemm = OpenCLLoader::instance().clCreateKernel(prog, "fp16_gemm_va", &err);
  CL_CHECK(err);

  struct Shape { int M, K, N; };
  std::vector<Shape> shapes = {
    {128,  2048, 2048},
    {512,  2048, 2048},
    {1024, 2048, 2048},
    {1024, 2048, 6144},
    {1024, 6144, 2048},
    // --- compute-bound probes: weight matrix N*K fits in the 1 MB L2 so
    // weight reuse hits cache, exposing the kernel's FMA/issue ceiling
    // rather than DRAM bandwidth. (N*K*2 bytes: 256x256=128KB, 512x512=512KB.)
    {2048, 256,  256},
    {4096, 256,  256},
    {4096, 512,  512},
    {8192, 512,  512},
    {8192, 256,  256},
  };

  std::mt19937 rng(0xc0ffee);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  constexpr double kPeakTflops3 = 3.0e12;  // Adreno 830 fp16 peak (12 CUs)

  fmt::print("\n=== OpenCL fp16-GEMM variant A bench (reps={}) ===\n", reps);
  fmt::print("  {:>5} {:>5} {:>5}   {:>8}   {:>10}   {:>8}   {:>9}   {:>7}\n",
             "M", "K", "N", "ms", "GFLOP/s", "GB/s", "max_rel", "%peak");

  for (auto sh : shapes) {
    const int M = sh.M, K = sh.K, N = sh.N;
    // Tile-divisibility constraints from the kernel.
    if ((M % 8) != 0 || (N % 4) != 0 || (K % 4) != 0) {
      fmt::print(stderr, "skip M={}, K={}, N={} (tile mismatch)\n", M, K, N);
      continue;
    }

    std::vector<__fp16> A_MK((size_t)M * K);
    std::vector<__fp16> W_NK((size_t)N * K);
    for (auto& x : A_MK) x = (__fp16)dist(rng);
    for (auto& x : W_NK) x = (__fp16)dist(rng);

    // Activations as image1d_buffer: backing is [K, M] in fp16.
    std::vector<__fp16> A_KM((size_t)M * K);
    transpose_MK_to_KM(M, K, A_MK.data(), A_KM.data());

    cl_mem d_A_buf = create_and_upload(ctx, q, A_KM.data(), A_KM.size() * 2, CL_MEM_READ_ONLY);
    cl_image_format img_fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc img_desc = {};
    img_desc.image_type   = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc.image_width  = (size_t)M * K / 4;     // # half4 pixels
    img_desc.buffer       = d_A_buf;
    cl_mem d_A_img = OpenCLLoader::instance().clCreateImage(
        ctx, CL_MEM_READ_ONLY, &img_fmt, &img_desc, nullptr, &err);
    CL_CHECK(err);

    // Interleave weights to v4's [K, N/4, 4] layout (one-time, not timed).
    std::vector<__fp16> W_il((size_t)N * K);
    interleave_W_NK_to_KN4(N, K, W_NK.data(), W_il.data());
    cl_mem d_W = create_and_upload(ctx, q, W_il.data(), W_il.size() * 2, CL_MEM_READ_ONLY);
    cl_mem d_C_NM = OpenCLLoader::instance().clCreateBuffer(
        ctx, CL_MEM_WRITE_ONLY, (size_t)M * N * 2, nullptr, &err);
    CL_CHECK(err);

    auto run_once = [&]() {
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 0, sizeof(cl_mem), &d_A_img));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 1, sizeof(cl_mem), &d_W));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 2, sizeof(cl_mem), &d_C_NM));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 3, sizeof(int), &M));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 4, sizeof(int), &N));
      CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 5, sizeof(int), &K));
      size_t global[2] = {(size_t)M / 8, (size_t)N / 4};
      size_t local[2]  = {(size_t)lx, (size_t)ly};   // configurable; 128 = full Adreno subgroup
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(
          q, k_gemm, 2, nullptr, global, local, 0, nullptr, nullptr));
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
    };

    // Warmup + correctness check (first row, against fp32 CPU ref).
    run_once();
    std::vector<__fp16> out_NM((size_t)M * N);
    CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(
        q, d_C_NM, CL_TRUE, 0, out_NM.size() * 2, out_NM.data(), 0, nullptr, nullptr));

    const int N_check = std::min(64, N);
    std::vector<float> ref(N_check);
    cpu_ref_first_row(M, K, N, N_check, A_MK.data(), W_NK.data(), ref.data());
    double max_rel = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (int n = 0; n < N_check; ++n) {
      // dst layout is [N, M] → first row m=0 lives at out_NM[n * M + 0].
      float g = (float)out_NM[(long)n * M + 0];
      float r = ref[n];
      double abs_err = std::abs((double)g - (double)r);
      double rel = abs_err / std::max(1e-6, std::abs((double)r));
      max_rel = std::max(max_rel, rel);
      max_abs = std::max(max_abs, abs_err);
      max_ref = std::max(max_ref, (double)std::abs(r));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double flops = 2.0 * M * (double)K * (double)N;
    double w_bytes   = 2.0 * (double)N * (double)K;   // fp16 weights
    double act_bytes = 2.0 * (double)M * (double)K;   // fp16 acts
    double out_bytes = 2.0 * (double)M * (double)N;
    double gflops = flops / (ms * 1e6);
    double gbps   = (w_bytes + act_bytes + out_bytes) / (ms * 1e6);
    double pct_peak = 100.0 * (flops / (ms * 1e-3)) / kPeakTflops3;

    fmt::print("  {:>5} {:>5} {:>5}   {:>7.3f}    {:>9.2f}   {:>7.2f}   {:>.3e}   {:>6.1f}%   max_abs={:.3e} max_ref={:.3e}\n",
               M, K, N, ms, gflops, gbps, max_rel, pct_peak, max_abs, max_ref);

    OpenCLLoader::instance().clReleaseMemObject(d_A_buf);
    OpenCLLoader::instance().clReleaseMemObject(d_A_img);
    OpenCLLoader::instance().clReleaseMemObject(d_W);
    OpenCLLoader::instance().clReleaseMemObject(d_C_NM);
  }

  OpenCLLoader::instance().clReleaseKernel(k_gemm);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
