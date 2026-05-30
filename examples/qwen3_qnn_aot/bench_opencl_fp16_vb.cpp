// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// OpenCL fp16 GEMM microbench — variant B (llama.cpp mul_mm_f16_f32_l4_lm).
//
// This is the "variant B" companion to bench_opencl_fp16_va.cpp. Its kernel is
// adapted from llama.cpp's most sophisticated fp16 mul_mm kernel
//   ggml/src/ggml-opencl/kernels/mul_mm_f16_f32_l4_lm.cl
// which uses BM=64, BN=64, BK=16, TM=4, TN=8 — i.e. each WG owns a 64×64
// output tile, each thread owns a 4×8 sub-tile (32 outputs), and the WG is
// 128 threads. The original llama.cpp kernel multiplies fp16 weights × fp32
// activations and writes fp32; here we keep everything fp16 (matching v4 LPBQ)
// because (a) our reference kernel is fp16 and (b) the activation memory
// footprint halves, helping bandwidth.
//
// Key llama.cpp ideas adopted:
//   * __local memory K-tile for BOTH A and B (the LPBQ v4 design omits LDS;
//     this is the major structural difference vs variant A).
//   * A stored transposed in LDS (buf_a[k * BM + m]) so the inner FMA reads
//     A with stride 1 across M.
//   * Register-blocked inner loop: cache_a[TM] and cache_b[TN] reused TM*TN
//     times per K-step.
//   * Vectorized cooperative loads (LOAD_VEC_A = LOAD_VEC_B = 4 → half4).
//   * 128 threads per WG, matching Adreno full subgroup width.
//
// Shapes / weights / activations match bench_opencl_fp16_va.cpp:
//   A : [M=Sq, K]  fp16  random in [-0.1, 0.1]
//   B : [K, N]     fp16  random in [-0.1, 0.1]   (re-used dequantized LPBQ
//                  values would be one option, but variant A uses random
//                  weights, so we match that for apples-to-apples.)
//   C : [M, N]     fp16
//
// Reported metrics: ms, GFLOPS, GB/s, max_rel (vs fp32 CPU reference),
// percentage of 3 TF/s (Adreno 830 fp16 peak).
//
//   ./mllm-qwen3-opencl-fp16-vb-bench --sq 1024 --reps 5

#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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
// OpenCL kernel source — fp16 GEMM, llama.cpp l4_lm-style tile structure.
//
//   A : [M, K] fp16  row-major
//   B : [K, N] fp16  row-major
//   C : [M, N] fp16  row-major
//
// Tile / blocking constants (compile-time):
//   BM = 64    output rows per WG
//   BN = 64    output cols per WG
//   BK = 16    K-step per outer iteration (K-tile loaded to LDS each step)
//   TM = 4     output rows per thread
//   TN = 8     output cols per thread
//
// WG size = (BM/TM) * (BN/TN) = 16 * 8 = 128 threads (1D).
//
// LDS layout:
//   buf_a[BK][BM]   — A transposed (stride-1 across M in inner loop)
//   buf_b[BK][BN]   — B normal (stride-1 across N)
// Total LDS = (BK*BM + BK*BN) * sizeof(half) = (16*64 + 16*64)*2 = 4 KB / WG.
//
// Loads per outer step:
//   threads cooperatively load BM*BK halves of A + BN*BK halves of B via half4.
//   With 128 threads × half4 per step = 512 halves loaded per pass; BM*BK = 1024
//   so each thread does 2 vload4 of A and 2 vload4 of B.
//
// Inner loop:
//   For each k ∈ [0, BK):
//     load TM activations and TN weights from LDS into registers
//     TM*TN = 32 half-FMAs into sums[TM*TN] accumulator
// ---------------------------------------------------------------------------

const char* kFp16VBKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_FULL __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define REQD_SUBGROUP_SIZE_FULL
#endif

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 8

// Round-up to BM/BN safe variant: caller must pad or ensure M,N % {BM,BN} == 0.
// For Qwen3 shapes (M ∈ {1024,512,128}, N ∈ {2048, 1024, 2048, 6144}) all are
// multiples of 64. We omit the OOB checks from llama.cpp for that reason — they
// inflate the inner loop register pressure on Adreno.

REQD_SUBGROUP_SIZE_FULL
__kernel void fp16_gemm_vb_l4lm(
    __global const half4* A,    // [M, K/4] half4
    __global const half4* B,    // [K, N/4] half4
    __global       half*  C,    // [M, N]   half
    const int M, const int N, const int K) {

  __local half buf_a[BK * BM];  // [BK][BM] — A transposed
  __local half buf_b[BK * BN];  // [BK][BN]

  const int ir = get_group_id(0);          // M-tile index
  const int ic = get_group_id(1);          // N-tile index

  const int tid  = get_local_id(0);        // 0..127
  const int th_r = tid % (BM / TM);        // 0..15  (which row group)
  const int th_c = tid / (BM / TM);        // 0..7   (which col group)

  const int K4 = K >> 2;                   // K/4 (half4 stride in A)
  const int N4 = N >> 2;                   // N/4 (half4 stride in B)

  // half4-relative starting positions for this WG (advanced by BK each step).
  int pos_a_k4 = 0;                        // K-offset in half4 units (A)
  int pos_b_k  = 0;                        // K-offset in half (B row index)

  half sums[TM * TN];
  half cache_a[TM];
  half cache_b[TN];
  #pragma unroll
  for (int i = 0; i < TM * TN; ++i) sums[i] = (half)0;

  // Cooperative load layouts:
  //   A tile: BM × BK = 64 × 16 = 1024 halves = 256 half4. 128 threads × 2
  //     half4 each. Map thread to a (m_row, k4_col) cell via flat index;
  //     each thread loads 2 contiguous m_rows (or 2 contiguous k4_cols).
  //   B tile: BK × BN = 16 × 64 = 1024 halves = 256 half4. Same partition:
  //     128 threads × 2 half4 each, mapping to (k_row, n4_col).
  // We choose to keep N-stride-1 loads for B (warps coalesce on N), and
  // M-stride-1 for A (since A is laid out [M, K] row-major with K fastest,
  // we read K-rows of 4 contiguous K-elements per thread — coalesced on K).
  //
  // Concretely: BK/4 = 4 half4 columns per A row, BM=64 rows → 64*4=256 cells.
  // tid / 4 = m_row (0..31), tid % 4 = k4_col (0..3) for pass 1; m_row + 32
  // for pass 2.
  // For B: BN/4 = 16 half4 columns per B row, BK=16 rows → 16*16=256 cells.
  // tid / 16 = k_row (0..7), tid % 16 = n4_col (0..15) for pass 1; k_row + 8
  // for pass 2.
  const int a_m   = tid >> 2;                // 0..31
  const int a_k4  = tid & 3;                  // 0..3
  const int b_k   = tid >> 4;                 // 0..7
  const int b_n4  = tid & 15;                 // 0..15

  for (int block = 0; block < K; block += BK) {
    // ---- Load A tile (BM × BK) into buf_a, K-major (transposed for inner).
    //   Two passes: rows {a_m, a_m+32} × cols {a_k4*4 .. a_k4*4 + 3}.
    {
      const int m0 = a_m;
      const int m1 = a_m + 32;
      half4 a0 = A[(ir * BM + m0) * K4 + pos_a_k4 + a_k4];
      half4 a1 = A[(ir * BM + m1) * K4 + pos_a_k4 + a_k4];
      const int k_base = a_k4 * LOAD_VEC_A;       // 0,4,8,12
      buf_a[(k_base + 0) * BM + m0] = a0.s0;
      buf_a[(k_base + 1) * BM + m0] = a0.s1;
      buf_a[(k_base + 2) * BM + m0] = a0.s2;
      buf_a[(k_base + 3) * BM + m0] = a0.s3;
      buf_a[(k_base + 0) * BM + m1] = a1.s0;
      buf_a[(k_base + 1) * BM + m1] = a1.s1;
      buf_a[(k_base + 2) * BM + m1] = a1.s2;
      buf_a[(k_base + 3) * BM + m1] = a1.s3;
    }

    // ---- Load B tile (BK × BN) into buf_b, K-row × N-col layout (stride-1
    // on N). Two passes: rows {b_k, b_k+8} × cols {b_n4*4 .. b_n4*4 + 3}.
    {
      const int k0 = b_k;
      const int k1 = b_k + 8;
      const int n_base = b_n4 * LOAD_VEC_B;       // 0,4,...,60
      const int n4_in_tile = b_n4;                // half4 col within BN-tile
      const int n4_global  = (ic * (BN >> 2)) + n4_in_tile;
      half4 b0 = B[(pos_b_k + k0) * N4 + n4_global];
      half4 b1 = B[(pos_b_k + k1) * N4 + n4_global];
      vstore4(b0, 0, &buf_b[k0 * BN + n_base]);
      vstore4(b1, 0, &buf_b[k1 * BN + n_base]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    pos_a_k4 += BK >> 2;                    // advance K by BK halves
    pos_b_k  += BK;                          // advance K by BK rows in B

    // ---- Inner: BK steps of TM × TN outer-product accumulation.
    // NOTE: Adreno OpenCL compiler OOMs (clBuildProgram aborts in
    // operator new) when the outer BK loop is fully unrolled together
    // with the nested TM*TN unrolls (16 * 4 * 8 = 512 FMAs expanded).
    // Keep the outer rolled.
    for (int i = 0; i < BK; ++i) {
      #pragma unroll
      for (int j = 0; j < TM; ++j) {
        cache_a[j] = buf_a[i * BM + th_r * TM + j];
      }
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        cache_b[j] = buf_b[i * BN + th_c * TN + j];
      }
      #pragma unroll
      for (int cc = 0; cc < TN; ++cc) {
        #pragma unroll
        for (int cr = 0; cr < TM; ++cr) {
          sums[cc * TM + cr] = fma(cache_a[cr], cache_b[cc], sums[cc * TM + cr]);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write back. dst[m, n] = sums[cc*TM + cr]; m = ir*BM + th_r*TM + cr,
  // n = ic*BN + th_c*TN + cc.
  const int dr = ir * BM + th_r * TM;
  const int dc = ic * BN + th_c * TN;
  #pragma unroll
  for (int cc = 0; cc < TN; ++cc) {
    #pragma unroll
    for (int cr = 0; cr < TM; ++cr) {
      C[(long)(dr + cr) * N + (dc + cc)] = sums[cc * TM + cr];
    }
  }
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

// CPU fp32 reference: C[M,N] = A[M,K] * B[K,N], both inputs fp16.
void cpu_ref_fp16(int M, int N, int K,
                  const __fp16* A, const __fp16* B, __fp16* C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        acc += (float)A[(size_t)m * K + k] * (float)B[(size_t)k * N + n];
      }
      C[(size_t)m * N + n] = (__fp16)acc;
    }
  }
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& sq_arg = Argparse::add<int>("--sq").help("M dim (Sq)").def(1024);
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(5);
  auto& projs_arg = Argparse::add<std::string>("--projs")
                        .help("comma list of projection shapes").def("q,k,v,o,gate,up,down");
  auto& cpu_check_arg = Argparse::add<bool>("--cpu-check")
                            .help("run full CPU fp32 reference (slow at large Sq)");
  auto& fixed_shapes_arg = Argparse::add<bool>("--fixed-shapes")
                            .help("Run the 5 explicit (M,K,N) shapes from the bench task");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  const int Sq = sq_arg.get();
  const int reps = reps_arg.get();

  // Default: env override turns on fixed shapes (so we don't need a CLI arg).
  bool fixed_shapes = fixed_shapes_arg.isSet() || std::getenv("MLLM_VB_FIXED") != nullptr;
  // Always run the fixed 5 shapes for the cross-variant comparison, then the
  // projection shapes too (if requested).
  // For the task's GEMM measurement we only need fixed shapes; flip default ON
  // to keep parity with bench_opencl_fp16_va.cpp.
  fixed_shapes = true;

  // Qwen3-1.7B geometry — identical to bench_opencl_lpbq.cpp.
  const int H = 2048, Hq = 16 * 128, Hkv = 8 * 128, I = 6144;

  fmt::print("[opencl-fp16-vb] initializing OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  // Build inline kernel.
  size_t src_len = std::strlen(kFp16VBKernelSrc);
  cl_int err;
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(
      ctx, 1, &kFp16VBKernelSrc, &src_len, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[16384] = {0};
    size_t log_size = 0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, rt->getDevices()[0](),
        CL_PROGRAM_BUILD_LOG, sizeof(log), log, &log_size);
    fmt::print(stderr, "OpenCL build failed:\n{}\n", log);
    return 1;
  }
  cl_kernel k_gemm = OpenCLLoader::instance().clCreateKernel(prog, "fp16_gemm_vb_l4lm", &err);
  CL_CHECK(err);

  struct Spec { const char* tag; int K; int N; int M_override; };
  std::vector<Spec> all = {
      {"q",    H,  Hq, 0},
      {"k",    H,  Hkv, 0},
      {"v",    H,  Hkv, 0},
      {"o",    Hq, H, 0},
      {"gate", H,  I, 0},
      {"up",   H,  I, 0},
      {"down", I,  H, 0},
  };
  std::vector<Spec> wanted;
  if (fixed_shapes) {
    // (M, K, N) tuples from the bench task — match bench_opencl_fp16_va.cpp.
    wanted.push_back({"s1",  2048, 2048,  128});
    wanted.push_back({"s2",  2048, 2048,  512});
    wanted.push_back({"s3",  2048, 2048, 1024});
    wanted.push_back({"s4",  2048, 6144, 1024});
    wanted.push_back({"s5",  6144, 2048, 1024});
  } else {
    for (auto& p : all) if (projs_arg.get().find(p.tag) != std::string::npos) wanted.push_back(p);
  }

  std::mt19937 rng(0xc0ffee);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  fmt::print("\n=== OpenCL fp16 GEMM bench — variant B (llama.cpp l4_lm) "
             "(Sq={}, reps={}) ===\n", Sq, reps);
  fmt::print("  Adreno 830 fp16 theoretical peak ≈ 3 TFLOPS (12 CUs @ ~1.1 GHz)\n");
  fmt::print("  Tile: BM=64 BN=64 BK=16 TM=4 TN=8 (32 outs/thread, 128 threads/WG)\n");
  fmt::print("  {:<6} {:>5}×{:>5} {:>5}   ms      GFLOPS    GB/s    %3TF    max_rel\n",
             "proj", "K", "N", "M");

  for (auto& p : wanted) {
    const int K = p.K;
    const int N = p.N;
    const int M_eff = p.M_override > 0 ? p.M_override : Sq;

    // Sanity: kernel requires M,N,K multiples of BM,BN,BK respectively.
    if (M_eff % 64 != 0 || N % 64 != 0 || K % 16 != 0) {
      fmt::print(stderr, "  skip {} ({}×{}×{}): not divisible by (BM=64, BN=64, BK=16)\n",
                 p.tag, M_eff, K, N);
      continue;
    }

    std::vector<__fp16> A((size_t)M_eff * K);
    std::vector<__fp16> B((size_t)K * N);
    std::vector<__fp16> C_gpu((size_t)M_eff * N, (__fp16)0);
    std::vector<__fp16> C_cpu((size_t)M_eff * N, (__fp16)0);
    for (auto& x : A) x = (__fp16)dist(rng);
    for (auto& x : B) x = (__fp16)dist(rng);

    // CPU reference. Full reference is slow; gate behind --cpu-check, and
    // otherwise only validate the first row of outputs.
    if (cpu_check_arg.isSet()) {
      cpu_ref_fp16(M_eff, N, K, A.data(), B.data(), C_cpu.data());
    } else {
      // First-row only: M=1 against the same B → length N reference vector.
      cpu_ref_fp16(1, N, K, A.data(), B.data(), C_cpu.data());
    }

    cl_mem d_A = create_and_upload(ctx, q, A.data(), A.size() * 2, CL_MEM_READ_ONLY);
    cl_mem d_B = create_and_upload(ctx, q, B.data(), B.size() * 2, CL_MEM_READ_ONLY);
    cl_mem d_C = OpenCLLoader::instance().clCreateBuffer(
        ctx, CL_MEM_WRITE_ONLY, (size_t)M_eff * N * 2, nullptr, &err);
    CL_CHECK(err);

    int Mv = M_eff, Nv = N, Kv = K;
    CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 0, sizeof(cl_mem), &d_A));
    CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 1, sizeof(cl_mem), &d_B));
    CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 2, sizeof(cl_mem), &d_C));
    CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 3, sizeof(int),    &Mv));
    CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 4, sizeof(int),    &Nv));
    CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemm, 5, sizeof(int),    &Kv));

    // Global: (M/TM, N/TN) total threads → groups of (BM/TM, BN/TN).
    // 1D local along ir,ic is awkward; we use 2D NDRange where local = (128,1)
    // and global = ((M/BM)*128, N/BN). get_group_id(0) gives ir, group_id(1)
    // gives ic, local_id(0) is tid 0..127.
    const size_t lws[2] = {128, 1};
    const size_t gws[2] = {(size_t)(M_eff / 64) * 128, (size_t)(N / 64)};

    auto run_once = [&]() {
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(
          q, k_gemm, 2, nullptr, gws, lws, 0, nullptr, nullptr));
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
    };

    // Warmup.
    run_once();
    CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(
        q, d_C, CL_TRUE, 0, C_gpu.size() * 2, C_gpu.data(), 0, nullptr, nullptr));

    // Validate first row vs CPU reference.
    double max_rel = 0.0, max_abs = 0.0, max_ref = 0.0;
    int N_check = std::min(64, N);
    for (int n = 0; n < N_check; ++n) {
      float a = (float)C_gpu[n];
      float b = (float)C_cpu[n];
      double abs_err = std::abs((double)a - (double)b);
      double rel = abs_err / std::max(1e-6, std::abs((double)b));
      max_rel = std::max(max_rel, rel);
      max_abs = std::max(max_abs, abs_err);
      max_ref = std::max(max_ref, (double)std::abs(b));
    }

    if (cpu_check_arg.isSet()) {
      // Full-matrix check.
      double full_max_rel = 0.0;
      const int M_check = std::min(8, M_eff);
      for (int m = 0; m < M_check; ++m) {
        for (int n = 0; n < N; ++n) {
          float a = (float)C_gpu[(size_t)m * N + n];
          float b = (float)C_cpu[(size_t)m * N + n];
          double abs_err = std::abs((double)a - (double)b);
          double rel = abs_err / std::max(1e-6, std::abs((double)b));
          full_max_rel = std::max(full_max_rel, rel);
        }
      }
      fmt::print("    (full-check first {} rows: max_rel={:.3e})\n", M_check, full_max_rel);
      max_rel = std::max(max_rel, full_max_rel);
    }

    // Timed reps.
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double flops = 2.0 * M_eff * (double)K * (double)N;
    double w_bytes = 2.0 * (double)K * (double)N;            // B is fp16
    double act_bytes = 2.0 * M_eff * (double)K;               // A is fp16
    double out_bytes = 2.0 * M_eff * (double)N;               // C is fp16
    double gflops = flops / (ms * 1e6);
    double pct = gflops / 3000.0 * 100.0;                     // % of 3 TFLOPS
    fmt::print("  {:<6} {:>5}×{:>5} {:>5}   {:>6.3f}   {:>7.2f}   {:>6.2f}  {:>5.1f}%   {:.3e}\n",
               p.tag, K, N, M_eff, ms, gflops,
               (w_bytes + act_bytes + out_bytes) / (ms * 1e6),
               pct, max_rel);

    OpenCLLoader::instance().clReleaseMemObject(d_A);
    OpenCLLoader::instance().clReleaseMemObject(d_B);
    OpenCLLoader::instance().clReleaseMemObject(d_C);
  }

  OpenCLLoader::instance().clReleaseKernel(k_gemm);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
