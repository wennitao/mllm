// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// OpenCL LPBQ matmul microbench.
//
// Loads real LPBQ q/k/v/o/gate/up/down weights from qwen3_1.7b_ptq_lpbq.mllm,
// uploads to GPU, runs an inline OpenCL kernel, validates against the CPU
// reference, and times prefill (Sq=128/512/1024) + decode (Sq=1).
//
// Weights are prepacked to [N, K] layout (one nibble per byte, 0..15 with
// signed int4 read v = u<8 ? u : u-16) so the K-axis is contiguous and the
// kernel can stream weights with vload8 / vload16. scale1[N, K/Bs] uchar,
// scale2[N] fp32.
//
//   ./mllm-qwen3-opencl-lpbq-bench --params qwen3_1.7b_ptq_lpbq.mllm \
//                                  --sq 1024 --reps 5

#include <CL/cl.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/cpu/kernels/common/lpbq_matmul.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {

// ---------------------------------------------------------------------------
// OpenCL LPBQ matmul kernel sources (inline).
//
// Two entry points:
//   lpbq_gemv_fp16 : Sq=1 (decode). One WG per output column; threads in the
//                    WG split the K dimension.
//   lpbq_gemm_fp16 : Sq>1 (prefill). One work-item per output element.
//
// Both expect packed weight in [N, K] layout, one nibble per uchar (0..15
// unsigned; reconstruct signed via v = u<8 ? u : u-16). Inputs/outputs in
// fp16; reductions in fp32 for stability with K up to 6144.
// ---------------------------------------------------------------------------

const char* kLpbqKernelSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_FULL __attribute__((qcom_reqd_sub_group_size("full")))
#define REQD_SUBGROUP_SIZE_HALF __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_FULL
#define REQD_SUBGROUP_SIZE_HALF
#endif

// ---------------- gemv (Sq=1) ----------------
//   A : [K]                fp16
//   W : [N, K]              uchar (nibble per byte)
//   S1: [N, K/Bs]           uchar (1..16)
//   S2: [N]                 float
//   C : [N]                 fp16
__kernel void lpbq_gemv_fp16(const int K, const int N, const int Bs,
                             __global const half*  A,
                             __global const uchar* W,
                             __global const uchar* S1,
                             __global const float* S2,
                             __global       half*  C) {
  const int n = get_group_id(0);
  if (n >= N) return;
  const int local_id = get_local_id(0);
  const int wg_size = get_local_size(0);
  const int num_blocks = K / Bs;

  __local float partials[128];

  const __global uchar* w_row = W + (long)n * K;
  const __global uchar* s1_row = S1 + (long)n * num_blocks;

  float acc = 0.0f;
  for (int b = local_id; b < num_blocks; b += wg_size) {
    float block_acc = 0.0f;
    const int k_base = b * Bs;
    for (int ki = 0; ki < Bs; ki += 16) {
      // Sign-extend 16 nibbles in parallel via shift-trick.
      uchar16 wu = vload16(0, w_row + k_base + ki);
      char16  ws_lo = convert_char16((wu << (uchar16)4)) >> (char16)4;
      half8 a0 = vload8(0, A + k_base + ki);
      half8 a1 = vload8(1, A + k_base + ki);
      half8 w0 = convert_half8(convert_short8(ws_lo.s01234567));
      half8 w1 = convert_half8(convert_short8(ws_lo.s89abcdef));
      float8 p0 = convert_float8(a0 * w0);
      float8 p1 = convert_float8(a1 * w1);
      block_acc += dot(p0.lo, (float4)(1.0f)) + dot(p0.hi, (float4)(1.0f))
                 + dot(p1.lo, (float4)(1.0f)) + dot(p1.hi, (float4)(1.0f));
    }
    acc += block_acc * (float)s1_row[b];
  }
  partials[local_id] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = wg_size / 2; off > 0; off >>= 1) {
    if (local_id < off) partials[local_id] += partials[local_id + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_id == 0) {
    C[n] = (half)(partials[0] * S2[n]);
  }
}

// ---------------- gemm (Sq>1) ----------------
//   A : [M, K]              fp16
//   W : [N, K]              uchar
//   S1: [N, K/Bs]           uchar
//   S2: [N]                 float
//   C : [M, N]              fp16
__kernel void lpbq_gemm_fp16(const int M, const int K, const int N, const int Bs,
                             __global const half*  A,
                             __global const uchar* W,
                             __global const uchar* S1,
                             __global const float* S2,
                             __global       half*  C) {
  const int m = get_global_id(0);
  const int n = get_global_id(1);
  if (m >= M || n >= N) return;
  const int num_blocks = K / Bs;

  const __global uchar* w_row = W + (long)n * K;
  const __global uchar* s1_row = S1 + (long)n * num_blocks;
  const __global half*  a_row  = A + (long)m * K;

  float total = 0.0f;
  for (int b = 0; b < num_blocks; ++b) {
    float block_acc = 0.0f;
    const int k_base = b * Bs;
    for (int ki = 0; ki < Bs; ki += 16) {
      uchar16 wu = vload16(0, w_row + k_base + ki);
      char16 ws_lo = convert_char16((wu << (uchar16)4)) >> (char16)4;
      half8 a0 = vload8(0, a_row + k_base + ki);
      half8 a1 = vload8(1, a_row + k_base + ki);
      half8 w0 = convert_half8(convert_short8(ws_lo.s01234567));
      half8 w1 = convert_half8(convert_short8(ws_lo.s89abcdef));
      float8 p0 = convert_float8(a0 * w0);
      float8 p1 = convert_float8(a1 * w1);
      block_acc += dot(p0.lo, (float4)(1.0f)) + dot(p0.hi, (float4)(1.0f))
                 + dot(p1.lo, (float4)(1.0f)) + dot(p1.hi, (float4)(1.0f));
    }
    total += block_acc * (float)s1_row[b];
  }
  C[(long)m * N + n] = (half)(total * S2[n]);
}

// ---------------- gemm v2: register-blocked + local-mem weight cache ----------
//
// Each work-item produces 1 × NG = 8 outputs (one M-row, 8 contiguous N-cols).
// WG is 1D of LOCAL_M = 64 threads along M → WG owns a 64×8 output tile.
// Per inner block of K (Bs elements):
//   * Cooperatively cache 8 weight rows × Bs cols (8 × 128 = 1024 bytes) into
//     local memory — 64 threads each load 16 bytes once.
//   * Each thread streams 1 M-row of activations from global, fmas against
//     the cached weights, accumulating 8 partial sums in registers.
// At end of block, multiply 8 partials by scale1[col, b] and add to totals.
//
// Compile-time constants encoded as #define so __local arrays can be sized.
// Bs MUST equal LPBQ_BS at runtime; runner sets LPBQ_BS via build options.
//
// Reuse won:
//   * Each weight byte read once per WG (vs once per output → 64× reduction).
//   * Activations: still one read per (m, K) pair (each thread owns its m).
// Bs is set by the model export side; for qwen3_1.7b_ptq_lpbq.mllm it's 16.
#ifndef LPBQ_BS
#define LPBQ_BS 16
#endif
#define LPBQ_NG 8
#define LPBQ_LOCAL_M 64
#define LPBQ_K_CHUNK 256           // 16 blocks worth of K cached per load
#define LPBQ_BLOCKS_PER_CHUNK (LPBQ_K_CHUNK / LPBQ_BS)   // 16
#define LPBQ_LOAD_BYTES (LPBQ_NG * LPBQ_BS)        // unused in v3


__kernel void lpbq_gemm_fp16_v2(const int M, const int K, const int N,
                                __global const half*  A,
                                __global const uchar* W,
                                __global const uchar* S1,
                                __global const float* S2,
                                __global       half*  C) {
  // Final v2 kernel: WG = 64 threads × 1 M-row × NG=8 N-cols output tile.
  // Local mem caches:
  //   * w_tile[NG][Bs] (256 B) — pre-dequantized fp16 weights, loaded by 8
  //     threads once per K-block.
  //   * s1_tile[NG * num_blocks] (up to 3KB) — block scales as fp32, loaded
  //     cooperatively once per kernel.
  // Inner reduction: convert acts and weights to float4 chunks, use 4× dot4
  // per N-col. Per-output total kept as fp32 scalar; final scale by s2 at end.
  const int m = get_global_id(0);
  const int n_group = get_global_id(1);
  const int n_base = n_group * LPBQ_NG;
  const int lm = get_local_id(0);
  if (n_base >= N) return;
  const int num_blocks = K / LPBQ_BS;

  __local half w_tile[LPBQ_NG][LPBQ_BS];
  __local float s1_tile[LPBQ_NG * 384];
  for (int i = lm; i < LPBQ_NG * num_blocks; i += LPBQ_LOCAL_M) {
    int row = i / num_blocks;
    int blk = i - row * num_blocks;
    s1_tile[i] = (float)S1[(long)(n_base + row) * num_blocks + blk];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float total[LPBQ_NG] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int b = 0; b < num_blocks; ++b) {
    if (lm < LPBQ_NG) {
      int n = n_base + lm;
      if (n < N) {
        uchar16 wu = vload16(0, W + (long)n * K + b * LPBQ_BS);
        char16 ws = convert_char16((wu << (uchar16)4)) >> (char16)4;
        half16 wh = convert_half16(convert_short16(ws));
        vstore16(wh, 0, (__local half*)&w_tile[lm][0]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (m < M) {
      half16 a = vload16(0, A + (long)m * K + b * LPBQ_BS);
      float4 af0 = convert_float4(a.s0123);
      float4 af1 = convert_float4(a.s4567);
      float4 af2 = convert_float4(a.s89ab);
      float4 af3 = convert_float4(a.scdef);
      #define LPBQ_RED_COL(J) {                                              \
        half16 wh = vload16(0, (__local half*)&w_tile[J][0]);                \
        float4 wf0 = convert_float4(wh.s0123);                               \
        float4 wf1 = convert_float4(wh.s4567);                               \
        float4 wf2 = convert_float4(wh.s89ab);                               \
        float4 wf3 = convert_float4(wh.scdef);                               \
        float sum = dot(af0, wf0) + dot(af1, wf1)                            \
                  + dot(af2, wf2) + dot(af3, wf3);                           \
        total[J] += sum * s1_tile[J * num_blocks + b];                       \
      }
      LPBQ_RED_COL(0); LPBQ_RED_COL(1); LPBQ_RED_COL(2); LPBQ_RED_COL(3);
      LPBQ_RED_COL(4); LPBQ_RED_COL(5); LPBQ_RED_COL(6); LPBQ_RED_COL(7);
      #undef LPBQ_RED_COL
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (m < M) {
    float8 t = (float8)(total[0], total[1], total[2], total[3],
                        total[4], total[5], total[6], total[7]);
    float8 s2_vec = vload8(0, S2 + n_base);
    half8 r = convert_half8(t * s2_vec);
    vstore8(r, 0, C + (long)m * N + n_base);
  }
}

// =============================================================================
// v4: llama.cpp-inspired Adreno-tuned GEMM
//
//   * cl_qcom_reqd_sub_group_size("full") → 128-wide subgroups for optimal scheduling
//   * Activations from image1d_buffer (texture cache) instead of __global buffer
//   * 8 M-rows × 4 N-cols output tile per work-item → 32 outputs/thread
//   * half8 accumulators (c0..c3, one per N-col), holds 8 M-rows of partial sums
//   * Broadcast-scalar half8 FMA: `c0 += B * dequant.s0` — works on Adreno
//     because the scalar dequant lives in a scalar register (not a constructed
//     vector). This was the key insight my v3 missed.
//   * No __local memory — relies on texture cache + register reuse
//   * Weights packed as ushort (4 nibbles per word, 4 K-positions × 1 OC)
//
// Format match-up vs llama.cpp's gemm_noshuffle_q4_0_f32:
//   Their M (output OC) = our N (output features)
//   Their N (batch)     = our M (Sq)
//   Block size: q4_0=32, ours=16 (LPBQ Bs). 4 K-positions per K-iter, so:
//     q4_0 has 8 K-iters per scale block
//     ours has 4 K-iters per scale block
//   Combined scale (s1 * s2 baked into single fp16) — kernel applies once.
//
// Input data layouts:
//   acts      : image1d_buffer of half4, content [K, M] row-major in half
//               (M = Sq), so read_imageh(acts, k*M/4 + m/4) returns half4 of
//               4 consecutive M values at K=k.
//   weights   : __global ushort*, [K/4, N/4, 4 ushorts] — each ushort packs
//               4 nibbles for 4 K-positions of one specific OC. Nibbles are
//               offset-binary (XOR'd in prepack).
//   scales    : __global half*, [num_blocks, N/4, 4 halves] — pre-multiplied
//               s1 * s2 per (block, OC).
//   dst       : __global half*, [N, M] row-major in half (transposed back
//               on the host if needed).
//
// Output: dst[N, M]. Each work-item writes 8 M-rows × 4 N-cols = 32 halves.
// =============================================================================
//
// Weights are pre-shuffled to [N/IL, K, IL] layout (IL = kInterleave = 4).
// Each vload16 of weights now spans 4 K × 4 OCs — the dequant work for ONE
// load now produces values for 4 output channels at once. The s1 scale is
// also pre-shuffled to [N/IL, num_blocks, IL].
//
// Fast dequant (from AWQ's dequantize.cuh, ported to OpenCL):
//   Given byte u (low 4 bits = nibble value 0..15), construct the fp16
//   representation of (1024 + nibble) by:
//      packed = (ushort)u | 0x6400
//      val_fp16 = as_half(packed) - 1032.0h        // signed -8..7
//   Total: ushort widen + or + fp16 sub = 3 instructions on a vector of
//   16 bytes (vs the old convert_char/<<4/>>4/convert_short/convert_half
//   chain = ~5 ops). Magic 1032 = 1024 + 8 folds the signed shift in.
//
// Per K-block inner loop now does HALF the local-memory weight loads vs v2
// (4 vload4 of half4 instead of 8 vload16 of half16), and the inner FMA is
// half4 broadcast-multiply against the dequanted weight, accumulating 4
// per-OC partials per K-step into one half4 register per OC-group.

#define LPBQ_IL 4

REQD_SUBGROUP_SIZE_FULL
__kernel void lpbq_gemm_fp16_v4(
    __read_only image1d_buffer_t acts,   // [K, M] half (1d image of half4)
    __global const ushort*       weights, // [K/4, N/4, 4 ushorts]
    __global const half*         scales,  // [num_blocks, N/4, 4 halves]
    __global       half*         dst,     // [N, M] half
    const int M, const int N, const int K) {
  const int gy = get_global_id(0);          // M-tile index, 8 M-rows
  const int gx = get_global_id(1);          // N-tile index, 4 N-cols
  const int gx_4 = gx << 2;                  // base N for this thread
  const int M_4 = M >> 2;                    // M/4 (image stride)
  const int N_4 = N >> 2;                    // N/4 (weight row stride)
  const int num_blocks = K >> 4;             // K/Bs, Bs=16

  half8 c0 = (half8)((half)0);
  half8 c1 = (half8)((half)0);
  half8 c2 = (half8)((half)0);
  half8 c3 = (half8)((half)0);
  half8 B;
  half4 dq;

  __global const ushort* w_ptr = weights + gx_4;       // step by N_4 * 4 per K/4
  __global const half*   s_ptr = scales  + gx_4;        // step by N_4 * 4 per block

  for (int b = 0; b < num_blocks; ++b) {
    // 4 combined-scale entries (s1*s2 for our 4 OCs at this block).
    half4 scale = vload4(0, s_ptr + (long)b * N_4 * 4);

    // 4 K-iters per scale block (Bs=16, 4 K-positions/iter).
    #pragma unroll
    for (int ji = 0; ji < 4; ++ji) {
      const int i = b * 16 + ji * 4;          // K-base for this iter
      // 4 ushorts: weights for 4 OCs, each packing 4 K-positions of one OC.
      ushort4 bits4 = vload4(0, w_ptr + (long)(i >> 2) * N_4 * 4);

      // j=0 (K = i+0)
      B.s0123 = read_imageh(acts, gy * 2 + i * M_4);
      B.s4567 = read_imageh(acts, gy * 2 + i * M_4 + 1);
      dq.s0 = ((half)((short)(bits4.s0 & (ushort)0x000F) - 8)) * scale.s0;
      dq.s1 = ((half)((short)(bits4.s1 & (ushort)0x000F) - 8)) * scale.s1;
      dq.s2 = ((half)((short)(bits4.s2 & (ushort)0x000F) - 8)) * scale.s2;
      dq.s3 = ((half)((short)(bits4.s3 & (ushort)0x000F) - 8)) * scale.s3;
      c0 += B * dq.s0;
      c1 += B * dq.s1;
      c2 += B * dq.s2;
      c3 += B * dq.s3;

      // j=1 (K = i+1)
      B.s0123 = read_imageh(acts, gy * 2 + (i + 1) * M_4);
      B.s4567 = read_imageh(acts, gy * 2 + (i + 1) * M_4 + 1);
      dq.s0 = ((half)((short)((bits4.s0 & (ushort)0x00F0) >> 4) - 8)) * scale.s0;
      dq.s1 = ((half)((short)((bits4.s1 & (ushort)0x00F0) >> 4) - 8)) * scale.s1;
      dq.s2 = ((half)((short)((bits4.s2 & (ushort)0x00F0) >> 4) - 8)) * scale.s2;
      dq.s3 = ((half)((short)((bits4.s3 & (ushort)0x00F0) >> 4) - 8)) * scale.s3;
      c0 += B * dq.s0;
      c1 += B * dq.s1;
      c2 += B * dq.s2;
      c3 += B * dq.s3;

      // j=2 (K = i+2)
      B.s0123 = read_imageh(acts, gy * 2 + (i + 2) * M_4);
      B.s4567 = read_imageh(acts, gy * 2 + (i + 2) * M_4 + 1);
      dq.s0 = ((half)((short)((bits4.s0 & (ushort)0x0F00) >> 8) - 8)) * scale.s0;
      dq.s1 = ((half)((short)((bits4.s1 & (ushort)0x0F00) >> 8) - 8)) * scale.s1;
      dq.s2 = ((half)((short)((bits4.s2 & (ushort)0x0F00) >> 8) - 8)) * scale.s2;
      dq.s3 = ((half)((short)((bits4.s3 & (ushort)0x0F00) >> 8) - 8)) * scale.s3;
      c0 += B * dq.s0;
      c1 += B * dq.s1;
      c2 += B * dq.s2;
      c3 += B * dq.s3;

      // j=3 (K = i+3)
      B.s0123 = read_imageh(acts, gy * 2 + (i + 3) * M_4);
      B.s4567 = read_imageh(acts, gy * 2 + (i + 3) * M_4 + 1);
      dq.s0 = ((half)((short)((bits4.s0 & (ushort)0xF000) >> 12) - 8)) * scale.s0;
      dq.s1 = ((half)((short)((bits4.s1 & (ushort)0xF000) >> 12) - 8)) * scale.s1;
      dq.s2 = ((half)((short)((bits4.s2 & (ushort)0xF000) >> 12) - 8)) * scale.s2;
      dq.s3 = ((half)((short)((bits4.s3 & (ushort)0xF000) >> 12) - 8)) * scale.s3;
      c0 += B * dq.s0;
      c1 += B * dq.s1;
      c2 += B * dq.s2;
      c3 += B * dq.s3;
    }
  }

  // Store 8 M × 4 N. dst is [N, M] row-major → dst[n*M + m].
  // 8 M-rows × 4 OCs: write 8 vstore4 of half4 (one per M-row).
  // c0..c3 each hold 8 M-rows of values for ONE N-col.
  // For M-row sM, write (c0.sM, c1.sM, c2.sM, c3.sM) packed.
  int base = gx_4 * M + (gy << 3);
  #define WRITE_M(SM) {                                              \
    half4 out = (half4)(c0.s##SM, c1.s##SM, c2.s##SM, c3.s##SM);     \
    /* layout: dst[n, m] with stride M between N-rows */              \
    dst[(gx_4 + 0) * M + (gy << 3) + (SM)] = out.s0;                  \
    dst[(gx_4 + 1) * M + (gy << 3) + (SM)] = out.s1;                  \
    dst[(gx_4 + 2) * M + (gy << 3) + (SM)] = out.s2;                  \
    dst[(gx_4 + 3) * M + (gy << 3) + (SM)] = out.s3;                  \
  }
  WRITE_M(0); WRITE_M(1); WRITE_M(2); WRITE_M(3);
  WRITE_M(4); WRITE_M(5); WRITE_M(6); WRITE_M(7);
  #undef WRITE_M
  (void)base;
}

// =============================================================================
// v2 GEMV: K-parallel within 64-wide subgroup, sub_group_reduce_add at end.
//
// Layout matches v4 GEMM:
//   acts    : image1d_buffer of half4 (1D image, K halves)
//   weights : __global ushort* in [K/4, N/4, 4 ushorts] layout
//   scales  : __global half*   in [num_blocks, N/4, 4 halves]
//
// One WG = 1 subgroup = 64 threads, handles 4 OCs.
// Each thread accumulates partials for K/64 K-elements, then the 64 threads
// reduce their partials via sub_group_reduce_add. Thread 0 writes the 4
// half outputs.
//
// Why this should be faster than v1 gemv:
//   * No __local mem barrier-reduction (sub_group_reduce_add is 1 op)
//   * Image1d_buffer texture cache for activations
//   * Pre-combined scales (s1·s2 in fp16) instead of separate s1+s2 fp32 path
//   * Subgroup-size-aware launch (no over/under-subscription)
// =============================================================================

REQD_SUBGROUP_SIZE_FULL
__kernel void lpbq_gemv_fp16_v2(
    __read_only image1d_buffer_t acts,    // [1, K] half, image of half4
    __global const ushort*       weights,  // [K/4, N/4, 4 ushorts]
    __global const half*         scales,   // [num_blocks, N/4, 4 halves]
    __global       half*         dst,      // [N] half
    const int K, const int N) {
  const int n_group = get_group_id(0);     // OC-group index
  const int slid    = get_sub_group_local_id();   // 0..127
  const int n_base  = n_group * 4;
  const int N_4     = N >> 2;
  const int num_blocks       = K >> 4;     // K/Bs, Bs=16
  const int K_per_thread     = K >> 7;     // K/128
  const int blocks_per_thread = K_per_thread >> 4; // (K/128)/16

  __global const ushort* w_ptr = weights + n_base;
  __global const half*   s_ptr = scales  + n_base;

  float4 acc = (float4)(0.0f);

  const int b_start = slid * blocks_per_thread;
  const int b_end   = b_start + blocks_per_thread;

  for (int b = b_start; b < b_end; ++b) {
    half4 scale = vload4(0, s_ptr + (long)b * N_4 * 4);

    #pragma unroll
    for (int ji = 0; ji < 4; ++ji) {
      const int i = b * 16 + ji * 4;
      ushort4 bits4 = vload4(0, w_ptr + (long)(i >> 2) * N_4 * 4);
      half4 a = read_imageh(acts, i >> 2);

      half4 dq;
      // j=0
      dq.s0 = ((half)((short)(bits4.s0 & (ushort)0x000F) - 8)) * scale.s0;
      dq.s1 = ((half)((short)(bits4.s1 & (ushort)0x000F) - 8)) * scale.s1;
      dq.s2 = ((half)((short)(bits4.s2 & (ushort)0x000F) - 8)) * scale.s2;
      dq.s3 = ((half)((short)(bits4.s3 & (ushort)0x000F) - 8)) * scale.s3;
      acc += convert_float4(dq * a.s0);
      // j=1
      dq.s0 = ((half)((short)((bits4.s0 & (ushort)0x00F0) >> 4) - 8)) * scale.s0;
      dq.s1 = ((half)((short)((bits4.s1 & (ushort)0x00F0) >> 4) - 8)) * scale.s1;
      dq.s2 = ((half)((short)((bits4.s2 & (ushort)0x00F0) >> 4) - 8)) * scale.s2;
      dq.s3 = ((half)((short)((bits4.s3 & (ushort)0x00F0) >> 4) - 8)) * scale.s3;
      acc += convert_float4(dq * a.s1);
      // j=2
      dq.s0 = ((half)((short)((bits4.s0 & (ushort)0x0F00) >> 8) - 8)) * scale.s0;
      dq.s1 = ((half)((short)((bits4.s1 & (ushort)0x0F00) >> 8) - 8)) * scale.s1;
      dq.s2 = ((half)((short)((bits4.s2 & (ushort)0x0F00) >> 8) - 8)) * scale.s2;
      dq.s3 = ((half)((short)((bits4.s3 & (ushort)0x0F00) >> 8) - 8)) * scale.s3;
      acc += convert_float4(dq * a.s2);
      // j=3
      dq.s0 = ((half)((short)((bits4.s0 & (ushort)0xF000) >> 12) - 8)) * scale.s0;
      dq.s1 = ((half)((short)((bits4.s1 & (ushort)0xF000) >> 12) - 8)) * scale.s1;
      dq.s2 = ((half)((short)((bits4.s2 & (ushort)0xF000) >> 12) - 8)) * scale.s2;
      dq.s3 = ((half)((short)((bits4.s3 & (ushort)0xF000) >> 12) - 8)) * scale.s3;
      acc += convert_float4(dq * a.s3);
    }
  }

  // Subgroup reduce: each lane's `acc.sN` summed across 64 lanes.
  float r0 = sub_group_reduce_add(acc.s0);
  float r1 = sub_group_reduce_add(acc.s1);
  float r2 = sub_group_reduce_add(acc.s2);
  float r3 = sub_group_reduce_add(acc.s3);

  if (slid == 0) {
    vstore4(convert_half4((float4)(r0, r1, r2, r3)), 0, dst + n_base);
  }
}

// Reference v3 kernel (kept for diff comparison; not dispatched)
__kernel void lpbq_gemm_fp16_v3(const int M, const int K, const int N,
                                __global const half*  A,
                                __global const uchar* W,      // [N/IL, K, IL]
                                __global const uchar* S1,     // [N/IL, num_blocks, IL]
                                __global const float* S2,     // [N]
                                __global       half*  C) {
  // WG = 64 threads × 1 M × 8 N output tile (NG=8 = 2 OC-groups of IL=4 each)
  const int m = get_global_id(0);
  const int n_group = get_global_id(1);          // index over 8-N-col tiles
  const int n_base = n_group * LPBQ_NG;
  const int lm = get_local_id(0);
  if (n_base >= N) return;
  const int num_blocks = K / LPBQ_BS;
  const int NG2 = LPBQ_NG / LPBQ_IL;             // = 2

  // Dequantized weight cache: [NG2 OC-groups][LPBQ_BS K][LPBQ_IL OCs] of half.
  // Size = 2 * 16 * 4 * 2 bytes = 256 bytes.
  __local half w_tile[LPBQ_NG / LPBQ_IL][LPBQ_BS][LPBQ_IL];
  __local float s1_tile[LPBQ_NG * 384];          // s1 same layout as v2 (per-OC scaled)

  // Load s1 once: N-cols of this WG × num_blocks. Interleave-aware: the
  // global s1 is shaped [N/IL, num_blocks, IL]. For our 2 OC-groups, that's
  // the two contiguous (num_blocks × IL) chunks starting at (n_base/IL).
  for (int i = lm; i < LPBQ_NG * num_blocks; i += LPBQ_LOCAL_M) {
    int row = i / num_blocks;                    // 0..NG-1, OC within tile
    int blk = i - row * num_blocks;
    int g = row / LPBQ_IL;                        // 0..NG2-1
    int il = row % LPBQ_IL;                       // 0..IL-1
    int gn = n_base / LPBQ_IL + g;                // global OC-group index
    s1_tile[i] = (float)S1[(long)gn * num_blocks * LPBQ_IL + (long)blk * LPBQ_IL + il];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float total[LPBQ_NG] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int b = 0; b < num_blocks; ++b) {
    // Cooperative weight load: 8 threads, each does one vload16 = 4 K × 4 OCs.
    // We need NG2*Bs*IL = 2*16*4 = 128 bytes per K-block. 8 vload16 each = 128 bytes.
    // wrow ∈ {0,1} (OC-group), kchunk ∈ {0,1,2,3} (4-K-element chunk).
    if (lm < NG2 * 4) {
      int wrow = lm / 4;                          // 0..1
      int kchunk = lm & 3;                        // 0..3
      int gn = n_base / LPBQ_IL + wrow;
      uchar16 wu = vload16(0,
          W + (long)gn * K * LPBQ_IL + (long)b * LPBQ_BS * LPBQ_IL + kchunk * 16);
      // Fast dequant: 16 bytes → 16 halves in [-8, 7].
      ushort16 packed = convert_ushort16(wu) | (ushort16)((ushort)0x6400);
      half16 wh = as_half16(packed) - (half16)((half)1032.0f);
      // Store as 16 halves into w_tile[wrow][kchunk*4..kchunk*4+3][0..3].
      vstore16(wh, kchunk, (__local half*)&w_tile[wrow][0][0]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (m < M) {
      half16 a = vload16(0, A + (long)m * K + b * LPBQ_BS);
      // v3 with 4 parallel half4 accumulators per group → breaks the single-
      // register RAW dependency chain, lets compiler pipeline the FMAs.
      // Same total work, more ILP.
      #define BCAST4(x) ((half4)((x), (x), (x), (x)))
      half4 a0 = (half4)((half)0), a1 = (half4)((half)0);
      half4 a2 = (half4)((half)0), a3 = (half4)((half)0);
      half4 b0 = (half4)((half)0), b1 = (half4)((half)0);
      half4 b2 = (half4)((half)0), b3 = (half4)((half)0);
      half16 wb0g = vload16(0, (__local half*)&w_tile[0][0][0]);
      half16 wb1g = vload16(1, (__local half*)&w_tile[0][0][0]);
      half16 wb2g = vload16(2, (__local half*)&w_tile[0][0][0]);
      half16 wb3g = vload16(3, (__local half*)&w_tile[0][0][0]);
      a0 = fma(wb0g.s0123, BCAST4(a.s0), a0);
      a0 = fma(wb0g.s4567, BCAST4(a.s1), a0);
      a1 = fma(wb0g.s89ab, BCAST4(a.s2), a1);
      a1 = fma(wb0g.scdef, BCAST4(a.s3), a1);
      a2 = fma(wb1g.s0123, BCAST4(a.s4), a2);
      a2 = fma(wb1g.s4567, BCAST4(a.s5), a2);
      a3 = fma(wb1g.s89ab, BCAST4(a.s6), a3);
      a3 = fma(wb1g.scdef, BCAST4(a.s7), a3);
      a0 = fma(wb2g.s0123, BCAST4(a.s8), a0);
      a0 = fma(wb2g.s4567, BCAST4(a.s9), a0);
      a1 = fma(wb2g.s89ab, BCAST4(a.sa), a1);
      a1 = fma(wb2g.scdef, BCAST4(a.sb), a1);
      a2 = fma(wb3g.s0123, BCAST4(a.sc), a2);
      a2 = fma(wb3g.s4567, BCAST4(a.sd), a2);
      a3 = fma(wb3g.s89ab, BCAST4(a.se), a3);
      a3 = fma(wb3g.scdef, BCAST4(a.sf), a3);
      half4 acc0 = (a0 + a1) + (a2 + a3);
      half16 wb0h = vload16(0, (__local half*)&w_tile[1][0][0]);
      half16 wb1h = vload16(1, (__local half*)&w_tile[1][0][0]);
      half16 wb2h = vload16(2, (__local half*)&w_tile[1][0][0]);
      half16 wb3h = vload16(3, (__local half*)&w_tile[1][0][0]);
      b0 = fma(wb0h.s0123, BCAST4(a.s0), b0);
      b0 = fma(wb0h.s4567, BCAST4(a.s1), b0);
      b1 = fma(wb0h.s89ab, BCAST4(a.s2), b1);
      b1 = fma(wb0h.scdef, BCAST4(a.s3), b1);
      b2 = fma(wb1h.s0123, BCAST4(a.s4), b2);
      b2 = fma(wb1h.s4567, BCAST4(a.s5), b2);
      b3 = fma(wb1h.s89ab, BCAST4(a.s6), b3);
      b3 = fma(wb1h.scdef, BCAST4(a.s7), b3);
      b0 = fma(wb2h.s0123, BCAST4(a.s8), b0);
      b0 = fma(wb2h.s4567, BCAST4(a.s9), b0);
      b1 = fma(wb2h.s89ab, BCAST4(a.sa), b1);
      b1 = fma(wb2h.scdef, BCAST4(a.sb), b1);
      b2 = fma(wb3h.s0123, BCAST4(a.sc), b2);
      b2 = fma(wb3h.s4567, BCAST4(a.sd), b2);
      b3 = fma(wb3h.s89ab, BCAST4(a.se), b3);
      b3 = fma(wb3h.scdef, BCAST4(a.sf), b3);
      half4 acc1 = (b0 + b1) + (b2 + b3);
      #undef BCAST4

      // Scale each group's 4-OC partials by their s1 entries and accumulate
      // into the running fp32 totals.
      float4 s1_g0 = (float4)(s1_tile[(0*LPBQ_IL + 0) * num_blocks + b],
                              s1_tile[(0*LPBQ_IL + 1) * num_blocks + b],
                              s1_tile[(0*LPBQ_IL + 2) * num_blocks + b],
                              s1_tile[(0*LPBQ_IL + 3) * num_blocks + b]);
      float4 s1_g1 = (float4)(s1_tile[(1*LPBQ_IL + 0) * num_blocks + b],
                              s1_tile[(1*LPBQ_IL + 1) * num_blocks + b],
                              s1_tile[(1*LPBQ_IL + 2) * num_blocks + b],
                              s1_tile[(1*LPBQ_IL + 3) * num_blocks + b]);
      float4 acc0_f = convert_float4(acc0) * s1_g0;
      float4 acc1_f = convert_float4(acc1) * s1_g1;
      total[0] += acc0_f.s0; total[1] += acc0_f.s1;
      total[2] += acc0_f.s2; total[3] += acc0_f.s3;
      total[4] += acc1_f.s0; total[5] += acc1_f.s1;
      total[6] += acc1_f.s2; total[7] += acc1_f.s3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (m < M) {
    float8 t = (float8)(total[0], total[1], total[2], total[3],
                        total[4], total[5], total[6], total[7]);
    float8 s2_vec = vload8(0, S2 + n_base);
    half8 r = convert_half8(t * s2_vec);
    vstore8(r, 0, C + (long)m * N + n_base);
  }
}

// =============================================================================
// v5: optimized LPBQ GEMM. Same int4 K-loop as v4, but two fixes that close the
// ~2x gap vs llama.cpp's q4_0 GEMM (measured on Adreno 830):
//   (1) COALESCED output in [M, N] (row = token) via vstore4 of half4 — v4 wrote
//       [N, M] with 32 scattered scalar half stores/thread (uncoalesced across
//       the subgroup). This store/layout fix is the DOMINANT ~2x.
//   (2) flat K-loop + simple `(nib-8)*scale` dequant (no (half)((short)…) casts).
// Weights/scales layouts are IDENTICAL to v4 (no re-prepack needed). Output is
// the natural [M=Sq, N] row-major layout (= CPU ref layout), no transpose.
__kernel void lpbq_gemm_fp16_v5(
    __read_only image1d_buffer_t acts,   // [K, M] half (1d image of half4)
    __global const ushort*       weights, // [K/4, N/4, 4 ushorts]  (== v4)
    __global const half*         scales,  // [num_blocks, N/4, 4 halves] (== v4)
    __global       half*         dst,     // [M, N] half (row = token)
    const int M, const int N, const int K) {
  const int gy   = get_global_id(0);     // M-tile index, 8 M-rows (tokens)
  const int gx   = get_global_id(1);     // N-tile index, 4 N-cols (out chans)
  const int gx_4 = gx << 2;
  const int M_4  = M >> 2;
  const int N_4  = N >> 2;

  half8 c0 = (half8)((half)0), c1 = (half8)((half)0);
  half8 c2 = (half8)((half)0), c3 = (half8)((half)0);
  half8 B; half4 dq;
  __global const ushort* w_ptr = weights + gx_4;
  __global const half*   s_ptr = scales  + gx_4;

  for (int i = 0; i < K; i += 4) {
    half4 scale  = vload4(0, s_ptr + (long)(i >> 4) * N_4 * 4);   // 1 scale / 16 K
    ushort4 bits = vload4(0, w_ptr + (long)(i >> 2) * N_4 * 4);   // 4 K x 4 OC

    #define V5_STEP(KK, MASK, SH) {                                          \
      B.s0123 = read_imageh(acts, gy*2 + (i+KK)*M_4);                        \
      B.s4567 = read_imageh(acts, gy*2 + (i+KK)*M_4 + 1);                    \
      dq.s0 = (((bits.s0 & (MASK)) >> SH) - 8) * scale.s0;                   \
      dq.s1 = (((bits.s1 & (MASK)) >> SH) - 8) * scale.s1;                   \
      dq.s2 = (((bits.s2 & (MASK)) >> SH) - 8) * scale.s2;                   \
      dq.s3 = (((bits.s3 & (MASK)) >> SH) - 8) * scale.s3;                   \
      c0 += B * dq.s0; c1 += B * dq.s1; c2 += B * dq.s2; c3 += B * dq.s3;    \
    }
    V5_STEP(0, 0x000F, 0); V5_STEP(1, 0x00F0, 4);
    V5_STEP(2, 0x0F00, 8); V5_STEP(3, 0xF000, 12);
    #undef V5_STEP
  }

  // Coalesced store: for token m = (gy*8 + SM), write the 4 OCs (gx_4..gx_4+3)
  // as a half4 to dst[m*N + gx_4]. Adjacent gx threads -> adjacent N -> coalesced.
  #define WRITE_M(SM) \
    vstore4((half4)(c0.s##SM, c1.s##SM, c2.s##SM, c3.s##SM), 0, \
            dst + (long)((gy << 3) + (SM)) * N + gx_4);
  WRITE_M(0); WRITE_M(1); WRITE_M(2); WRITE_M(3);
  WRITE_M(4); WRITE_M(5); WRITE_M(6); WRITE_M(7);
  #undef WRITE_M
}
)CL";

struct LPBQTensor {
  std::vector<uint8_t> w_packed;       // [N, K] (HWOI, for v2)
  std::vector<uint8_t> w_interleaved;  // [N/4, K, 4] (for v3)
  std::vector<uint16_t> w_ushort;      // [K/4, N/4, 4 ushorts] (for v4)
  std::vector<uint8_t> s1;             // [N, K/Bs]
  std::vector<uint8_t> s1_interleaved; // [N/4, K/Bs, 4] (for v3)
  std::vector<uint16_t> combined_scales_fp16; // [num_blocks, N/4, 4 halves] (for v4)
  std::vector<float>   s2;             // [N]
  int K, N, Bs;
};

LPBQTensor load_lpbq(const mllm::ParameterFile::ptr_t& params,
                     const std::string& prefix, int K, int N) {
  LPBQTensor t;
  t.K = K; t.N = N;
  auto w_t  = params->pull(prefix + ".weight");
  auto s1_t = params->pull(prefix + ".scale1");
  auto s2_t = params->pull(prefix + ".scale2");
  int num_blocks = (int)((size_t)s1_t.numel() / (size_t)N);
  t.Bs = K / num_blocks;
  t.s1.assign(s1_t.ptr<uint8_t>(), s1_t.ptr<uint8_t>() + (size_t)N * num_blocks);
  t.s2.assign(s2_t.ptr<float>(),  s2_t.ptr<float>()  + (size_t)N);
  t.w_packed.resize((size_t)K * (size_t)N);
  mllm::cpu::lpbq_prepack_weights_HWOI(K, N, w_t.ptr<uint8_t>(), t.w_packed.data());
  t.w_interleaved.resize((size_t)K * (size_t)N);
  mllm::cpu::lpbq_prepack_weights_NK4(K, N, w_t.ptr<uint8_t>(), t.w_interleaved.data());
  t.s1_interleaved.resize((size_t)N * (size_t)num_blocks);
  mllm::cpu::lpbq_prepack_scale1_NB4(N, num_blocks, t.s1.data(), t.s1_interleaved.data());
  // v4: ushort-packed weights (4 nibbles/ushort) + combined fp16 scales.
  t.w_ushort.resize((size_t)K * (size_t)N / 4);  // K*N nibbles = K*N/4 ushorts
  mllm::cpu::lpbq_prepack_weights_USHORT4(K, N, w_t.ptr<uint8_t>(), t.w_ushort.data());
  t.combined_scales_fp16.resize((size_t)N * num_blocks);
  mllm::cpu::lpbq_prepack_combined_scales(N, num_blocks, t.s1.data(), t.s2.data(),
                                          t.combined_scales_fp16.data());
  return t;
}

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

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& params_arg = Argparse::add<std::string>("--params").help(".mllm ptq lpbq file")
                         .def("qwen3_1.7b_ptq_lpbq.mllm");
  auto& layer_arg = Argparse::add<int>("--layer").help("layer index").def(0);
  auto& sq_arg = Argparse::add<int>("--sq").help("M dim").def(1024);
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(5);
  auto& projs_arg = Argparse::add<std::string>("--projs")
                        .help("comma list").def("q,k,v,o,gate,up,down");
  auto& ver_arg = Argparse::add<int>("--ver").help("prefill GEMM kernel: 4 or 5").def(5);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  const int Sq = sq_arg.get();
  const int reps = reps_arg.get();
  const int gemm_ver = ver_arg.get();
  const int L = layer_arg.get();

  // Qwen3-1.7B geometry.
  const int H = 2048, Hq = 16*128, Hkv = 8*128, I = 6144;

  fmt::print("[opencl-lpbq] loading {}\n", params_arg.get());
  auto params = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2,
                           mllm::kCPU, /*mmap=*/false);
  fmt::print("[opencl-lpbq] params loaded ({} tensors)\n", (int)params->dict().size());

  fmt::print("[opencl-lpbq] initializing OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  // Build inline kernels.
  size_t src_len = std::strlen(kLpbqKernelSrc);
  cl_int err;
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(
      ctx, 1, &kLpbqKernelSrc, &src_len, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[16384]={0};
    size_t log_size=0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, rt->getDevices()[0](),
        CL_PROGRAM_BUILD_LOG, sizeof(log), log, &log_size);
    fmt::print(stderr, "OpenCL build failed:\n{}\n", log);
    return 1;
  }
  cl_kernel k_gemv    = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemv_fp16",    &err); CL_CHECK(err);
  cl_kernel k_gemv_v2 = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemv_fp16_v2", &err); CL_CHECK(err);
  cl_kernel k_gemm    = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemm_fp16",    &err); CL_CHECK(err);
  cl_kernel k_gemm_v2 = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemm_fp16_v2", &err); CL_CHECK(err);
  cl_kernel k_gemm_v3 = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemm_fp16_v3", &err); CL_CHECK(err);
  cl_kernel k_gemm_v4 = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemm_fp16_v4", &err); CL_CHECK(err);
  cl_kernel k_gemm_v5 = OpenCLLoader::instance().clCreateKernel(prog, "lpbq_gemm_fp16_v5", &err); CL_CHECK(err);

  struct Spec { const char* tag; std::string prefix; int K; int N; };
  std::string base = "model.layers." + std::to_string(L);
  std::vector<Spec> all = {
      {"q", base + ".self_attn.q_proj", H, Hq},
      {"k", base + ".self_attn.k_proj", H, Hkv},
      {"v", base + ".self_attn.v_proj", H, Hkv},
      {"o", base + ".self_attn.o_proj", Hq, H},
      {"gate", base + ".mlp.gate_proj", H, I},
      {"up",   base + ".mlp.up_proj",   H, I},
      {"down", base + ".mlp.down_proj", I, H},
  };
  std::vector<Spec> wanted;
  for (auto& p : all) if (projs_arg.get().find(p.tag) != std::string::npos) wanted.push_back(p);

  std::mt19937 rng(0xc0ffee);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  fmt::print("\n=== OpenCL LPBQ bench (Sq={}, reps={}, layer={}) ===\n", Sq, reps, L);
  fmt::print("  {:<6} {:>5}×{:>5} {:>5}  Bs   ms      GFLOP/s    GB/s   max_rel(vs CPU)\n",
             "proj","K","N","M");
  for (auto& p : wanted) {
    auto T = load_lpbq(params, p.prefix, p.K, p.N);
    int num_blocks = T.K / T.Bs;

    std::vector<__fp16> act((size_t)Sq * (size_t)T.K);
    for (auto& x : act) x = (__fp16)dist(rng);
    std::vector<__fp16> out_cpu((size_t)Sq * (size_t)T.N);
    std::vector<__fp16> out_gpu((size_t)Sq * (size_t)T.N);

    // CPU reference using the prepacked weights (matches kernel input).
    mllm::cpu::lpbq_matmul_fp16_packed(Sq, T.N, T.K, T.Bs, act.data(),
                                       T.w_packed.data(), T.s1.data(), T.s2.data(),
                                       nullptr, out_cpu.data());

    // Upload to GPU.
    cl_mem d_A = create_and_upload(ctx, q, act.data(),  act.size()*2, CL_MEM_READ_ONLY);
    cl_mem d_W = create_and_upload(ctx, q, T.w_packed.data(), T.w_packed.size(), CL_MEM_READ_ONLY);
    cl_mem d_W3= create_and_upload(ctx, q, T.w_interleaved.data(), T.w_interleaved.size(), CL_MEM_READ_ONLY);
    cl_mem d_S1= create_and_upload(ctx, q, T.s1.data(), T.s1.size(),   CL_MEM_READ_ONLY);
    cl_mem d_S13= create_and_upload(ctx, q, T.s1_interleaved.data(), T.s1_interleaved.size(), CL_MEM_READ_ONLY);
    cl_mem d_S2= create_and_upload(ctx, q, T.s2.data(), T.s2.size()*4, CL_MEM_READ_ONLY);
    cl_mem d_C = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                    (size_t)Sq * (size_t)T.N * 2, nullptr, &err); CL_CHECK(err);

    // v4 buffers + image1d_buffer for activations.
    // 1. Transpose acts [M=Sq, K] → [K, M] fp16, upload as buffer, then wrap as image1d_buffer.
    std::vector<__fp16> act_KM((size_t)Sq * (size_t)T.K);
    mllm::cpu::lpbq_transpose_acts_MK_to_KM(Sq, T.K, act.data(), act_KM.data());
    cl_mem d_A_KM_buf = create_and_upload(ctx, q, act_KM.data(), act_KM.size()*2, CL_MEM_READ_ONLY);
    cl_image_format img_fmt = {CL_RGBA, CL_HALF_FLOAT};
    cl_image_desc img_desc = {};
    img_desc.image_type   = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc.image_width  = (size_t)Sq * T.K / 4;   // # half4 pixels
    img_desc.buffer       = d_A_KM_buf;
    cl_mem d_A_img = OpenCLLoader::instance().clCreateImage(ctx, CL_MEM_READ_ONLY,
                          &img_fmt, &img_desc, nullptr, &err); CL_CHECK(err);
    cl_mem d_W4 = create_and_upload(ctx, q, T.w_ushort.data(),
                    T.w_ushort.size() * sizeof(uint16_t), CL_MEM_READ_ONLY);
    cl_mem d_Sc4 = create_and_upload(ctx, q, T.combined_scales_fp16.data(),
                    T.combined_scales_fp16.size() * sizeof(uint16_t), CL_MEM_READ_ONLY);
    // v4 output is [N, M] layout (transposed vs CPU's [M, N]).
    cl_mem d_C_NM = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                      (size_t)Sq * (size_t)T.N * 2, nullptr, &err); CL_CHECK(err);

    auto run_once = [&]() {
      if (Sq == 1) {
        // GEMV path: kept on v1 (per-OC WG with local-mem reduction). The v2
        // GEMV (subgroup-reduce, image1d_buffer acts, ushort weights, fp16
        // combined scales) was correct but slower than v1 on this device —
        // at decode the Adreno kernel-launch floor (~140 μs) dominates total
        // wall time, and v2's larger setup cost (image+scale prepack reads)
        // doesn't pay off when compute is already <0.2 ms. v2 source kept
        // for reference.  See get_local_size(0)=64 in v1.
        int K = T.K, N = T.N, Bs = T.Bs;
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 0, sizeof(int), &K));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 1, sizeof(int), &N));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 2, sizeof(int), &Bs));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 3, sizeof(cl_mem), &d_A));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 4, sizeof(cl_mem), &d_W));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 5, sizeof(cl_mem), &d_S1));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 6, sizeof(cl_mem), &d_S2));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k_gemv, 7, sizeof(cl_mem), &d_C));
        const size_t wg = 64;
        size_t global = (size_t)T.N * wg;
        size_t local  = wg;
        CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q, k_gemv, 1, nullptr, &global, &local, 0, nullptr, nullptr));
      } else {
        // Tiled GEMM. Output tile per thread = 8 M × 4 N. Global = (M/8, N/4).
        // Local = (8, 16) → 128 threads/WG matches Adreno full subgroup.
        if (T.Bs != 16) {
          fmt::print(stderr, "GEMM requires Bs=16 (got {})\n", T.Bs);
        }
        int M = Sq, K = T.K, N = T.N;
        // v5 writes [M, N] (coalesced) into d_C; v4 writes [N, M] into d_C_NM.
        cl_kernel kg = (gemm_ver == 5) ? k_gemm_v5 : k_gemm_v4;
        cl_mem    dC = (gemm_ver == 5) ? d_C       : d_C_NM;
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 0, sizeof(cl_mem), &d_A_img));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 1, sizeof(cl_mem), &d_W4));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 2, sizeof(cl_mem), &d_Sc4));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 3, sizeof(cl_mem), &dC));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 4, sizeof(int), &M));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 5, sizeof(int), &N));
        CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kg, 6, sizeof(int), &K));
        size_t global[2] = {(size_t)M / 8, (size_t)T.N / 4};
        size_t local[2]  = {8, 16};      // 128 threads/WG
        CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q, kg, 2, nullptr, global, local, 0, nullptr, nullptr));
      }
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
    };

    // Warmup + correctness.
    run_once();
    if (Sq != 1 && gemm_ver == 4) {
      // v4 writes dst in [N, M] layout; transpose back to [M, N] for compare.
      std::vector<__fp16> out_NM((size_t)Sq * (size_t)T.N);
      CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, d_C_NM, CL_TRUE, 0,
          out_NM.size() * 2, out_NM.data(), 0, nullptr, nullptr));
      for (int m = 0; m < Sq; ++m)
        for (int n = 0; n < T.N; ++n)
          out_gpu[m * T.N + n] = out_NM[n * Sq + m];
    } else {
      // v5 (and GEMV) write [M, N] directly into d_C.
      CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, d_C, CL_TRUE, 0,
          out_gpu.size() * 2, out_gpu.data(), 0, nullptr, nullptr));
    }

    // Compare first row (vs CPU).
    double max_rel = 0.0, max_abs = 0.0, max_ref = 0.0;
    int N_check = std::min(32, (int)T.N);
    for (int n = 0; n < N_check; ++n) {
      float a = (float)out_gpu[n], b = (float)out_cpu[n];
      double abs_err = std::abs((double)a - (double)b);
      double rel = abs_err / std::max(1e-6, std::abs((double)b));
      max_rel = std::max(max_rel, rel);
      max_abs = std::max(max_abs, abs_err);
      max_ref = std::max(max_ref, (double)std::abs(b));
    }
    fmt::print("    (max_abs={:.3e}  max_ref={:.3e})\n", max_abs, max_ref);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    double flops = 2.0 * Sq * (double)T.K * (double)T.N;
    double w_bytes = 0.5 * T.K * (double)T.N + 0.5 * T.N * (double)T.K / T.Bs + 4.0 * T.N;
    double act_bytes = 2.0 * Sq * (double)T.K;
    fmt::print("  {:<6} {:>5}×{:>5} {:>5}  {:>2}  {:>6.3f}    {:>7.2f}   {:>6.2f}    {:.3e}\n",
               p.tag, p.K, p.N, Sq, T.Bs, ms,
               flops / (ms * 1e6), (w_bytes + act_bytes) / (ms * 1e6),
               max_rel);

    OpenCLLoader::instance().clReleaseMemObject(d_A);
    OpenCLLoader::instance().clReleaseMemObject(d_W);
    OpenCLLoader::instance().clReleaseMemObject(d_W3);
    OpenCLLoader::instance().clReleaseMemObject(d_S1);
    OpenCLLoader::instance().clReleaseMemObject(d_S13);
    OpenCLLoader::instance().clReleaseMemObject(d_S2);
    OpenCLLoader::instance().clReleaseMemObject(d_C);
    OpenCLLoader::instance().clReleaseMemObject(d_A_KM_buf);
    OpenCLLoader::instance().clReleaseMemObject(d_A_img);
    OpenCLLoader::instance().clReleaseMemObject(d_W4);
    OpenCLLoader::instance().clReleaseMemObject(d_Sc4);
    OpenCLLoader::instance().clReleaseMemObject(d_C_NM);
  }

  OpenCLLoader::instance().clReleaseKernel(k_gemv);
  OpenCLLoader::instance().clReleaseKernel(k_gemv_v2);
  OpenCLLoader::instance().clReleaseKernel(k_gemm);
  OpenCLLoader::instance().clReleaseKernel(k_gemm_v2);
  OpenCLLoader::instance().clReleaseKernel(k_gemm_v3);
  OpenCLLoader::instance().clReleaseKernel(k_gemm_v4);
  OpenCLLoader::instance().clReleaseKernel(k_gemm_v5);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
