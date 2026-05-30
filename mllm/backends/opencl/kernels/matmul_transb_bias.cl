// kernel/matmul_transb_bias.cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define LPBQ_REQD_SG __attribute__((qcom_reqd_sub_group_size("full")))
#else
#define LPBQ_REQD_SG
#endif
#define TILE_SIZE 16
#define QK4_0 32
#define QK8_0 32

typedef struct {
  half d;
  uchar qs[QK4_0 / 2];
} block_q4_0;

// ==================================================================
// 1. FP32 Fused GEMM + Bias Kernel
// ==================================================================
__kernel void gemm_fp32_transb_bias(__global const float *A,
                                    __global const float *B,
                                    __global const float *bias,
                                    __global float *C, const int M, const int K,
                                    const int N, const int has_bias,
                                    const int offset_a) {
  const int s = get_global_id(1);
  const int n = get_global_id(0);
  const int batch_idx = get_global_id(2);

  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  __local float a_tile[TILE_SIZE][TILE_SIZE];
  __local float b_tile[TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;
  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    const int k_start = t * TILE_SIZE;

    const int a_k_idx = k_start + local_col;
    if (s < M && a_k_idx < K) {
      a_tile[local_row][local_col] =
          A[(long)batch_idx * M * K + (long)s * K + a_k_idx + offset_a];
    } else {
      a_tile[local_row][local_col] = 0.0f;
    }

    const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
    const int b_k_idx = k_start + local_col;
    if (b_n_idx < N && b_k_idx < K) {
      b_tile[local_row][local_col] = B[(long)b_n_idx * K + b_k_idx];
    } else {
      b_tile[local_row][local_col] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
      acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (s < M && n < N) {
    if (has_bias != 0) {
      acc += bias[n];
    }
    C[(long)batch_idx * M * N + (long)s * N + n] = acc;
  }
}

__kernel void
gemm_fp32_fp16_transb_bias(__global const float *A, __global const half *B,
                           __global const float *bias, __global float *C,
                           const int M, const int K, const int N,
                           const int has_bias, const int offset_a) {
  const int s = get_global_id(1);
  const int n = get_global_id(0);
  const int batch_idx = get_global_id(2);

  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  __local float a_tile[TILE_SIZE][TILE_SIZE];
  __local float b_tile[TILE_SIZE][TILE_SIZE];

  float acc = 0.0f;
  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    const int k_start = t * TILE_SIZE;

    const int a_k_idx = k_start + local_col;
    if (s < M && a_k_idx < K) {
      a_tile[local_row][local_col] =
          A[(long)batch_idx * M * K + (long)s * K + a_k_idx + offset_a];
    } else {
      a_tile[local_row][local_col] = 0.0f;
    }

    const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
    const int b_k_idx = k_start + local_col;
    if (b_n_idx < N && b_k_idx < K) {
      b_tile[local_row][local_col] = (float)B[(long)b_n_idx * K + b_k_idx];
    } else {
      b_tile[local_row][local_col] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
      acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (s < M && n < N) {
    if (has_bias != 0) {
      acc += bias[n];
    }
    C[(long)batch_idx * M * N + (long)s * N + n] = acc;
  }
}

// ==================================================================
// 2. FP16 Fused GEMM + Bias Kernel
// ==================================================================
#if defined(SUPPORTS_FP16)
__kernel void gemm_fp16_transb_bias(__global const half *A,
                                    __global const half *B,
                                    __global const float *bias,
                                    __global half *C, const int M, const int K,
                                    const int N, const int has_bias,
                                    const int offset_a) {
  const int s = get_global_id(1);
  const int n = get_global_id(0);
  const int batch_idx = get_global_id(2);

  const int local_row = get_local_id(1);
  const int local_col = get_local_id(0);

  __local half a_tile[TILE_SIZE][TILE_SIZE];
  __local half b_tile[TILE_SIZE][TILE_SIZE];

  half acc = 0.0h;
  const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < num_tiles; ++t) {
    const int k_start = t * TILE_SIZE;
    const int a_k_idx = k_start + local_col;
    if (s < M && a_k_idx < K) {
      a_tile[local_row][local_col] =
          A[(long)batch_idx * M * K + (long)s * K + a_k_idx + offset_a];
    } else {
      a_tile[local_row][local_col] = 0.0h;
    }

    const int b_n_idx = get_group_id(0) * TILE_SIZE + local_row;
    const int b_k_idx = k_start + local_col;
    if (b_n_idx < N && b_k_idx < K) {
      b_tile[local_row][local_col] = B[(long)b_n_idx * K + b_k_idx];
    } else {
      b_tile[local_row][local_col] = 0.0h;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_tile = 0; k_tile < TILE_SIZE; ++k_tile) {
      acc += a_tile[local_row][k_tile] * b_tile[local_col][k_tile];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (s < M && n < N) {
    if (has_bias != 0) {
      acc += bias[n];
    }
    C[(long)batch_idx * M * N + (long)s * N + n] = acc;
  }
}
#else
__kernel void gemm_fp16_transb_bias(__global const half *A,
                                    __global const half *B,
                                    __global const float *bias,
                                    __global half *C, const int M, const int K,
                                    const int N, const int has_bias,
                                    const int offset_a) {
  const int s = get_global_id(1);
  const int n = get_global_id(0);
  const int batch_idx = get_global_id(2);

  if (s >= M || n >= N)
    return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    long a_idx = (long)batch_idx * M * K + (long)s * K + k + offset_a;
    long b_idx = (long)n * K + k;
    acc += (float)A[a_idx] * (float)B[b_idx];
  }

  if (has_bias != 0) {
    acc += bias[n];
  }
  long c_idx = (long)batch_idx * M * N + (long)s * N + n;
  C[c_idx] = (half)acc;
}
#endif // SUPPORTS_FP16

// ==================================================================
// 3. FP32 * Q4_0 Fused GEMV + Bias Kernels (for M = 1, Decoding)
// ==================================================================
__kernel void gemv_fp32_q4_0_transb_bias(__global const float *A,
                                         __global const block_q4_0 *B,
                                         __global const float *bias,
                                         __global float *C, const int K,
                                         const int N, const int has_bias,
                                         const int offset_a) {
  const int n = get_group_id(0);
  const int batch_idx = get_group_id(1);

  if (n >= N)
    return;

  const int local_id = get_local_id(0);
  const int wg_size = get_local_size(0);
  __local float partial_sums[256];

  float private_acc = 0.0f;
  const long a_base_idx = (long)batch_idx * K + offset_a;
  const long b_row_offset_blocks = (long)n * (K / QK4_0);

  for (int k = local_id; k < K; k += wg_size) {
    const int k_block_idx = k / QK4_0;
    const int k_in_block = k % QK4_0;

    const __global block_q4_0 *b_block_ptr =
        &B[b_row_offset_blocks + k_block_idx];
#if defined(SUPPORTS_FP16)
    const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
#else
    const float d_b = (float)(b_block_ptr->d); // TODO Change here [gemini]
#endif

    const uchar q_packed = b_block_ptr->qs[k_in_block % 16];
    char q_nibble = (k_in_block < 16) ? (q_packed & 0x0F) : (q_packed >> 4);
    const float b_val = (float)(q_nibble - 8) * d_b;

    private_acc += A[a_base_idx + k] * b_val;
  }

  partial_sums[local_id] = private_acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      partial_sums[local_id] += partial_sums[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    float final_val = partial_sums[0];
    if (has_bias != 0) {
      final_val += bias[n];
    }
    const long c_idx = (long)batch_idx * N + n;
    C[c_idx] = final_val;
  }
}
// ==================================================================
// 4. FP32 * Q4_0 Fused GEMM + Bias Kernels (for M > 1, Training)
// ==================================================================

__kernel void gemm_fp32_q4_0_transb_bias(
    __global const float *A, __global const block_q4_0 *B,
    __global const float *bias, __global float *C, const int M, const int K,
    const int N, const int has_bias, const int offset_a) {
  const int s = get_global_id(1);
  const int n = get_global_id(0);
  const int batch_idx = get_global_id(2);

  if (s >= M || n >= N) {
    return;
  }

  float acc = 0.0f;

  const long a_row_offset = (long)batch_idx * M * K + (long)s * K + offset_a;
  const long b_row_offset_blocks = (long)n * (K / QK4_0);

  for (int k_block_idx = 0; k_block_idx < K / QK4_0; ++k_block_idx) {
    const __global block_q4_0 *b_block_ptr =
        &B[b_row_offset_blocks + k_block_idx];

#if defined(SUPPORTS_FP16)
    const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
#else
    const float d_b = (float)(b_block_ptr->d); // TODO Change here [gemini]
#endif
    const __global float *a_ptr = A + a_row_offset + k_block_idx * QK4_0;

    for (int j = 0; j < QK4_0 / 2; j += 4) { // QK4_0/2 = 16

      const uchar q_packed0 = b_block_ptr->qs[j + 0];
      const uchar q_packed1 = b_block_ptr->qs[j + 1];
      const uchar q_packed2 = b_block_ptr->qs[j + 2];
      const uchar q_packed3 = b_block_ptr->qs[j + 3];

      const float4 a_vals_lo = vload4(0, a_ptr + j);
      const float4 a_vals_hi =
          vload4(0, a_ptr + j + (QK4_0 / 2)); // (QK4_0 / 2) = 16

      float4 b_dequant_lo;
      b_dequant_lo.x = (float)((q_packed0 & 0x0F) - 8) * d_b;
      b_dequant_lo.y = (float)((q_packed1 & 0x0F) - 8) * d_b;
      b_dequant_lo.z = (float)((q_packed2 & 0x0F) - 8) * d_b;
      b_dequant_lo.w = (float)((q_packed3 & 0x0F) - 8) * d_b;

      float4 b_dequant_hi;
      b_dequant_hi.x = (float)((q_packed0 >> 4) - 8) * d_b;
      b_dequant_hi.y = (float)((q_packed1 >> 4) - 8) * d_b;
      b_dequant_hi.z = (float)((q_packed2 >> 4) - 8) * d_b;
      b_dequant_hi.w = (float)((q_packed3 >> 4) - 8) * d_b;

      acc += dot(a_vals_lo, b_dequant_lo);
      acc += dot(a_vals_hi, b_dequant_hi);
    }
  }

  if (has_bias != 0) {
    acc += bias[n];
  }

  const long c_idx = (long)batch_idx * M * N + (long)s * N + n;
  C[c_idx] = acc;
}

// ==================================================================
// 5. FP16 * Q4_0 Fused GEMV + Bias Kernel (for M=1, Decoding)
// ==================================================================
#if defined(SUPPORTS_FP16)

__kernel void gemv_fp16_q4_0_transb_bias(__global const half *A,
                                         __global const block_q4_0 *B,
                                         __global const float *bias,
                                         __global half *C, const int K,
                                         const int N, const int has_bias,
                                         const int offset_a) {
  const int n = get_group_id(0);
  const int batch_idx = get_group_id(1);
  if (n >= N)
    return;
  const int local_id = get_local_id(0);
  const int wg_size = get_local_size(0);
  __local float partial_sums[256];
  float private_acc = 0.0f;
  const long a_base_idx = (long)batch_idx * K + offset_a;
  const long b_row_offset_blocks = (long)n * (K / QK4_0);
  const int num_k_blocks = K / QK4_0;
  for (int k_block_idx = local_id; k_block_idx < num_k_blocks;
       k_block_idx += wg_size) {
    const __global block_q4_0 *b_block_ptr =
        &B[b_row_offset_blocks + k_block_idx];
    const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
    const __global half *a_ptr = A + a_base_idx + k_block_idx * QK4_0;
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; j += 4) {
      const uchar q_packed0 = b_block_ptr->qs[j + 0];
      const uchar q_packed1 = b_block_ptr->qs[j + 1];
      const uchar q_packed2 = b_block_ptr->qs[j + 2];
      const uchar q_packed3 = b_block_ptr->qs[j + 3];

      float4 a_vals_lo, a_vals_hi;
      a_vals_lo.x = (float)a_ptr[j + 0];
      a_vals_lo.y = (float)a_ptr[j + 1];
      a_vals_lo.z = (float)a_ptr[j + 2];
      a_vals_lo.w = (float)a_ptr[j + 3];
      a_vals_hi.x = (float)a_ptr[j + 16 + 0];
      a_vals_hi.y = (float)a_ptr[j + 16 + 1];
      a_vals_hi.z = (float)a_ptr[j + 16 + 2];
      a_vals_hi.w = (float)a_ptr[j + 16 + 3];
      float4 b_dequant_lo, b_dequant_hi;
      b_dequant_lo.x = (float)((q_packed0 & 0x0F) - 8) * d_b;
      b_dequant_lo.y = (float)((q_packed1 & 0x0F) - 8) * d_b;
      b_dequant_lo.z = (float)((q_packed2 & 0x0F) - 8) * d_b;
      b_dequant_lo.w = (float)((q_packed3 & 0x0F) - 8) * d_b;
      b_dequant_hi.x = (float)((q_packed0 >> 4) - 8) * d_b;
      b_dequant_hi.y = (float)((q_packed1 >> 4) - 8) * d_b;
      b_dequant_hi.z = (float)((q_packed2 >> 4) - 8) * d_b;
      b_dequant_hi.w = (float)((q_packed3 >> 4) - 8) * d_b;
      private_acc += dot(a_vals_lo, b_dequant_lo);
      private_acc += dot(a_vals_hi, b_dequant_hi);
    }
  }
  partial_sums[local_id] = private_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      partial_sums[local_id] += partial_sums[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_id == 0) {
    float final_val = partial_sums[0];
    if (has_bias != 0) {
      final_val += bias[n];
    }
    const long c_idx = (long)batch_idx * N + n;
    vstore_half_rte(final_val, 0, &C[c_idx]);
  }
}

inline float hsum_half16(half16 v) {
  half8 r1 = v.lo + v.hi;
  half4 r2 = r1.lo + r1.hi;
  half2 r3 = r2.lo + r2.hi;
  return (float)(r3.x + r3.y);
}

__kernel void gemv_fp16_q4_0_transb_bias_half16(__global const half *A,
                                                __global const block_q4_0 *B,
                                                __global const float *bias,
                                                __global half *C, const int K,
                                                const int N, const int has_bias,
                                                const int offset_a) {
  const int n = get_group_id(0);
  const int bh_idx = get_group_id(1);
  const int batch_idx = get_group_id(2);
  if (n >= N)
    return;

  const int local_id = get_local_id(0);
  const int wg_size = get_local_size(0);

  float private_acc = 0.0f;

  __local float partial_sums[256];

  const long a_base_idx = (long)batch_idx * K + offset_a;
  const long b_row_offset_blocks = (long)n * (K / QK4_0);

  const int num_k_blocks = K / QK4_0;

  for (int k_block_idx = local_id; k_block_idx < num_k_blocks;
       k_block_idx += wg_size) {
    const __global block_q4_0 *b_block_ptr =
        &B[b_row_offset_blocks + k_block_idx];
    const half d_b = b_block_ptr->d;

    const __global half *a_ptr = A + a_base_idx + k_block_idx * QK4_0;

    const half16 a_vals_lo = vload16(0, a_ptr);
    const uchar8 q_packed_lo = vload8(0, b_block_ptr->qs);

    char16 b_s_lo;
    b_s_lo.s0 = (q_packed_lo.s0 & 0x0F) - 8;
    b_s_lo.s1 = (q_packed_lo.s0 >> 4) - 8;
    b_s_lo.s2 = (q_packed_lo.s1 & 0x0F) - 8;
    b_s_lo.s3 = (q_packed_lo.s1 >> 4) - 8;
    b_s_lo.s4 = (q_packed_lo.s2 & 0x0F) - 8;
    b_s_lo.s5 = (q_packed_lo.s2 >> 4) - 8;
    b_s_lo.s6 = (q_packed_lo.s3 & 0x0F) - 8;
    b_s_lo.s7 = (q_packed_lo.s3 >> 4) - 8;
    b_s_lo.s8 = (q_packed_lo.s4 & 0x0F) - 8;
    b_s_lo.s9 = (q_packed_lo.s4 >> 4) - 8;
    b_s_lo.sa = (q_packed_lo.s5 & 0x0F) - 8;
    b_s_lo.sb = (q_packed_lo.s5 >> 4) - 8;
    b_s_lo.sc = (q_packed_lo.s6 & 0x0F) - 8;
    b_s_lo.sd = (q_packed_lo.s6 >> 4) - 8;
    b_s_lo.se = (q_packed_lo.s7 & 0x0F) - 8;
    b_s_lo.sf = (q_packed_lo.s7 >> 4) - 8;

    const half16 b_vals_dequant_lo = convert_half16(b_s_lo) * d_b;
    private_acc += hsum_half16(a_vals_lo * b_vals_dequant_lo);

    const half16 a_vals_hi = vload16(0, a_ptr + 16);
    const uchar8 q_packed_hi = vload8(0, b_block_ptr->qs + 8);

    char16 b_s_hi;
    b_s_hi.s0 = (q_packed_hi.s0 & 0x0F) - 8;
    b_s_hi.s1 = (q_packed_hi.s0 >> 4) - 8;
    b_s_hi.s2 = (q_packed_hi.s1 & 0x0F) - 8;
    b_s_hi.s3 = (q_packed_hi.s1 >> 4) - 8;
    b_s_hi.s4 = (q_packed_hi.s2 & 0x0F) - 8;
    b_s_hi.s5 = (q_packed_hi.s2 >> 4) - 8;
    b_s_hi.s6 = (q_packed_hi.s3 & 0x0F) - 8;
    b_s_hi.s7 = (q_packed_hi.s3 >> 4) - 8;
    b_s_hi.s8 = (q_packed_hi.s4 & 0x0F) - 8;
    b_s_hi.s9 = (q_packed_hi.s4 >> 4) - 8;
    b_s_hi.sa = (q_packed_hi.s5 & 0x0F) - 8;
    b_s_hi.sb = (q_packed_hi.s5 >> 4) - 8;
    b_s_hi.sc = (q_packed_hi.s6 & 0x0F) - 8;
    b_s_hi.sd = (q_packed_hi.s6 >> 4) - 8;
    b_s_hi.se = (q_packed_hi.s7 & 0x0F) - 8;
    b_s_hi.sf = (q_packed_hi.s7 >> 4) - 8;

    const half16 b_vals_dequant_hi = convert_half16(b_s_hi) * d_b;
    private_acc += hsum_half16(a_vals_hi * b_vals_dequant_hi);
  }

  partial_sums[local_id] = private_acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      partial_sums[local_id] += partial_sums[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0) {
    float final_val = partial_sums[0];
    if (has_bias != 0) {
      final_val += bias[n];
    }
    const long c_idx = (long)batch_idx * N + n;
    vstore_half_rte(final_val, 0, &C[c_idx]);
  }
}

#else
__kernel void gemv_fp16_q4_0_transb_bias(__global const half *A,
                                         __global const block_q4_0 *B,
                                         __global const float *bias,
                                         __global half *C, const int K,
                                         const int N, const int has_bias,
                                         const int offset_a) {
  const int n = get_group_id(0);
  const int bh_idx = get_group_id(1);
  const int batch_idx = bh_idx;
  if (n >= N)
    return;
  const int local_id = get_local_id(0);
  const int wg_size = get_local_size(0);
  __local float partial_sums[256];
  float private_acc = 0.0f;
  const long a_base_idx = (long)batch_idx * K + offset_a;
  const long b_row_offset_blocks = (long)n * (K / QK4_0);
  const int num_k_blocks = K / QK4_0;
  for (int k_block_idx = local_id; k_block_idx < num_k_blocks;
       k_block_idx += wg_size) {
    const __global block_q4_0 *b_block_ptr =
        &B[b_row_offset_blocks + k_block_idx];
    const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
    const __global half *a_ptr = A + a_base_idx + k_block_idx * QK4_0;
#pragma unroll
    for (int j = 0; j < QK4_0 / 2; j += 4) {
      const uchar q_packed0 = b_block_ptr->qs[j + 0];
      const uchar q_packed1 = b_block_ptr->qs[j + 1];
      const uchar q_packed2 = b_block_ptr->qs[j + 2];
      const uchar q_packed3 = b_block_ptr->qs[j + 3];

      float4 a_vals_lo, a_vals_hi;
      a_vals_lo.x = (float)a_ptr[j + 0];
      a_vals_lo.y = (float)a_ptr[j + 1];
      a_vals_lo.z = (float)a_ptr[j + 2];
      a_vals_lo.w = (float)a_ptr[j + 3];
      a_vals_hi.x = (float)a_ptr[j + 16 + 0];
      a_vals_hi.y = (float)a_ptr[j + 16 + 1];
      a_vals_hi.z = (float)a_ptr[j + 16 + 2];
      a_vals_hi.w = (float)a_ptr[j + 16 + 3];
      float4 b_dequant_lo, b_dequant_hi;
      b_dequant_lo.x = (float)((q_packed0 & 0x0F) - 8) * d_b;
      b_dequant_lo.y = (float)((q_packed1 & 0x0F) - 8) * d_b;
      b_dequant_lo.z = (float)((q_packed2 & 0x0F) - 8) * d_b;
      b_dequant_lo.w = (float)((q_packed3 & 0x0F) - 8) * d_b;
      b_dequant_hi.x = (float)((q_packed0 >> 4) - 8) * d_b;
      b_dequant_hi.y = (float)((q_packed1 >> 4) - 8) * d_b;
      b_dequant_hi.z = (float)((q_packed2 >> 4) - 8) * d_b;
      b_dequant_hi.w = (float)((q_packed3 >> 4) - 8) * d_b;
      private_acc += dot(a_vals_lo, b_dequant_lo);
      private_acc += dot(a_vals_hi, b_dequant_hi);
    }
  }
  partial_sums[local_id] = private_acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
    if (local_id < offset) {
      partial_sums[local_id] += partial_sums[local_id + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (local_id == 0) {
    float final_val = partial_sums[0];
    if (has_bias != 0) {
      final_val += bias[n];
    }
    const long c_idx = (long)batch_idx * N + n;
    vstore_half_rte(final_val, 0, &C[c_idx]);
  }
}

#endif // SUPPORTS_FP16

// ==================================================================
// 6. FP16 * Q4_0 Fused GEMM + Bias Kernel (for M>1, Prefill)
// ==================================================================
#if defined(SUPPORTS_FP16)

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WPT_M 8
#define WPT_N 8
#define THREADS_X (TILE_N / WPT_N) // 8
#define THREADS_Y (TILE_M / WPT_M) // 8

__kernel void
gemm_fp16_q4_0_transb_bias(__global const half *A, __global const block_q4_0 *B,
                           __global const float *bias, __global half *C,
                           const int M, const int K, const int N,
                           const int has_bias, const int offset_a) {
  const int group_m_idx = get_group_id(1);
  const int group_n_idx = get_group_id(0);
  const int local_m_idx = get_local_id(1);
  const int local_n_idx = get_local_id(0);
  const int batch_idx = get_global_id(2);
  __local half a_tile[TILE_M][TILE_K + 1];
  __local half b_tile[TILE_K][TILE_N + 1];
  float acc[WPT_M][WPT_N];
#pragma unroll
  for (int i = 0; i < WPT_M; ++i) {
#pragma unroll
    for (int j = 0; j < WPT_N; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  const long base_a_offset = (long)batch_idx * M * K + offset_a;

  const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
  for (int t = 0; t < num_k_tiles; ++t) {
    const int k_start = t * TILE_K;
#pragma unroll
    for (int i = 0; i < WPT_M; ++i) {
      const int m_local = local_m_idx * WPT_M + i;
      const int k_local = local_n_idx;
      const int m_global = group_m_idx * TILE_M + m_local;
      if (m_global < M) {
        for (int k_load_step = 0; k_load_step < TILE_K / THREADS_X;
             ++k_load_step) {
          int k_global = k_start + k_local + k_load_step * THREADS_X;
          if (k_global < K) {
            a_tile[m_local][k_local + k_load_step * THREADS_X] =
                A[base_a_offset + m_global * K + k_global];
          } else {
            a_tile[m_local][k_local + k_load_step * THREADS_X] = 0.0h;
          }
        }
      } else {
        for (int k_load_step = 0; k_load_step < TILE_K / THREADS_X;
             ++k_load_step) {
          a_tile[m_local][k_local + k_load_step * THREADS_X] = 0.0h;
        }
      }
    }

#pragma unroll
    for (int i = 0; i < WPT_N; ++i) {
      const int n_local = local_n_idx * WPT_N + i;
      const int k_local = local_m_idx;
      const int n_global = group_n_idx * TILE_N + n_local;
      if (n_global < N) {
        for (int k_load_step = 0; k_load_step < TILE_K / THREADS_Y;
             ++k_load_step) {
          int k_global = k_start + k_local + k_load_step * THREADS_Y;
          if (k_global < K) {
            const int k_block_idx = k_global / QK4_0;
            const int k_in_block = k_global % QK4_0;
            const __global block_q4_0 *b_block_ptr =
                &B[n_global * (K / QK4_0) + k_block_idx];

            const float d_b =
                vload_half(0, (__global half *)(&(b_block_ptr->d)));

            const uchar qs_sub_idx = k_in_block % 16;

            const uchar q_packed = b_block_ptr->qs[qs_sub_idx];

            const bool is_low_nibble = (k_in_block < 16);
            char q_nibble =
                is_low_nibble ? ((q_packed & 0x0F) - 8) : ((q_packed >> 4) - 8);

            b_tile[k_local + k_load_step * THREADS_Y][n_local] =
                (half)((float)q_nibble * d_b);

          } else {
            b_tile[k_local + k_load_step * THREADS_Y][n_local] = 0.0h;
          }
        }
      } else {
        for (int k_load_step = 0; k_load_step < TILE_K / THREADS_Y;
             ++k_load_step) {
          b_tile[k_local + k_load_step * THREADS_Y][n_local] = 0.0h;
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int k_tile = 0; k_tile < TILE_K; ++k_tile) {
#pragma unroll
      for (int m = 0; m < WPT_M; ++m) {
        half a_val = a_tile[local_m_idx * WPT_M + m][k_tile];
#pragma unroll
        for (int n = 0; n < WPT_N; ++n) {
          half b_val = b_tile[k_tile][local_n_idx * WPT_N + n];
          acc[m][n] = mad((float)a_val, (float)b_val, acc[m][n]);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  long c_offset = (long)batch_idx * M * N;
#pragma unroll
  for (int m = 0; m < WPT_M; ++m) {
    int m_global = group_m_idx * TILE_M + local_m_idx * WPT_M + m;
    if (m_global < M) {
#pragma unroll
      for (int n = 0; n < WPT_N; ++n) {
        int n_global = group_n_idx * TILE_N + local_n_idx * WPT_N + n;
        if (n_global < N) {
          float result = acc[m][n];
          if (has_bias) {
            result += bias[n_global];
          }
          C[c_offset + m_global * N + n_global] = (half)result;
        }
      }
    }
  }
}

#else
// ---------- [Fallback] ----------
__kernel void
gemm_fp16_q4_0_transb_bias(__global const half *A, __global const block_q4_0 *B,
                           __global const float *bias, __global half *C,
                           const int M, const int K, const int N,
                           const int has_bias, const int offset_a) {

  const int s = get_global_id(1);
  const int n = get_global_id(0);
  const int batch_idx = get_global_id(2);

  if (s >= M || n >= N) {
    return;
  }

  float acc = 0.0f;
  const long a_row_offset = (long)batch_idx * M * K + (long)s * K + offset_a;
  const long b_row_offset_blocks = (long)n * (K / QK4_0);

  for (int k_block_idx = 0; k_block_idx < K / QK4_0; ++k_block_idx) {
    const __global block_q4_0 *b_block_ptr =
        &B[b_row_offset_blocks + k_block_idx];

    const float d_b = vload_half(0, (__global half *)(&(b_block_ptr->d)));
    const __global half *a_ptr = A + a_row_offset + k_block_idx * QK4_0;

    for (int j = 0; j < QK4_0 / 2; ++j) {
      const uchar q_packed = b_block_ptr->qs[j];
      const char q_lo = (q_packed & 0x0F) - 8;
      const char q_hi = (q_packed >> 4) - 8;

      acc += vload_half(j, a_ptr) * (float)q_lo * d_b;
      acc += vload_half(j + QK4_0 / 2, a_ptr) * (float)q_hi * d_b;
    }
  }

  if (has_bias != 0) {
    acc += bias[n];
  }
  const long c_idx = (long)batch_idx * M * N + (long)s * N + n;

  vstore_half_rte(acc, 0, &C[c_idx]);
}
#endif // SUPPORTS_FP16

#if defined(SUPPORTS_FP16)
// ==================================================================
// LPBQ (w4a16, two-scale int4) tuned GEMM/GEMV.
// Ported from the validated bench (bench_opencl_lpbq.cpp v5). Weights are
// ushort-packed [K/4, N/4, 4] (4 K-nibbles/OC, offset-binary), scales are
// combined s1*s2 fp16 [num_blocks, N/4, 4]. Activations come from a half
// image1d_buffer of [K, M] (transpose first via lpbq_transpose_mk_to_km_f16).
// Output is [M, N] half written COALESCED via vstore4 (the key vs the old
// scattered [N, M] store) — see commit history for the 1.8x measurement.
// ==================================================================

// [M,K] half -> [K,M] half (backing store for the activation image1d_buffer).
__kernel void lpbq_transpose_mk_to_km_f16(__global const half* src,
                                          __global half* dst,
                                          const int M, const int K) {
  const int m = get_global_id(0);
  const int k = get_global_id(1);
  if (m < M && k < K) dst[(long)k * M + m] = src[(long)m * K + k];
}

// Prefill GEMM (M % 8 == 0, N % 4 == 0). 8 tokens x 4 OCs per thread.
LPBQ_REQD_SG
__kernel void lpbq_gemm_fp16_v5(
    __read_only image1d_buffer_t acts,   // [K, M] half (1d image of half4)
    __global const ushort*       weights, // [K/4, N/4, 4 ushorts]
    __global const half*         scales,  // [num_blocks, N/4, 4 halves]
    __global       half*         dst,     // [M, N] half (row = token)
    const int M, const int N, const int K) {
  const int gy   = get_global_id(0);
  const int gx   = get_global_id(1);
  const int gx_4 = gx << 2;
  const int M_4  = M >> 2;
  const int N_4  = N >> 2;

  half8 c0 = (half8)((half)0), c1 = (half8)((half)0);
  half8 c2 = (half8)((half)0), c3 = (half8)((half)0);
  half8 B; half4 dq;
  __global const ushort* w_ptr = weights + gx_4;
  __global const half*   s_ptr = scales  + gx_4;

  for (int i = 0; i < K; i += 4) {
    half4 scale  = vload4(0, s_ptr + (long)(i >> 4) * N_4 * 4);
    ushort4 bits = vload4(0, w_ptr + (long)(i >> 2) * N_4 * 4);
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

  #define WRITE_M(SM) \
    vstore4((half4)(c0.s##SM, c1.s##SM, c2.s##SM, c3.s##SM), 0, \
            dst + (long)((gy << 3) + (SM)) * N + gx_4);
  WRITE_M(0); WRITE_M(1); WRITE_M(2); WRITE_M(3);
  WRITE_M(4); WRITE_M(5); WRITE_M(6); WRITE_M(7);
  #undef WRITE_M
}

// Decode GEMV (M == 1). One workgroup of LWS threads per output column;
// threads stride over K and reduce in local memory. Activation is the plain
// [1, K] = [K] half buffer (no transpose/image needed at M=1).
__kernel void lpbq_gemv_fp16_v5(
    __global const half*   act,     // [K] half
    __global const ushort* weights, // [K/4, N/4, 4]
    __global const half*   scales,  // [num_blocks, N/4, 4]
    __global       half*   dst,     // [N] half
    const int N, const int K) {
  const int n   = get_group_id(0);    // output column
  const int tid = get_local_id(0);
  const int lws = get_local_size(0);
  const int N_4 = N >> 2;
  const int ng  = n >> 2;
  const int il  = n & 3;
  __local float partial[256];

  float acc = 0.0f;
  for (int k = tid; k < K; k += lws) {
    const int kg = k >> 2;
    const int j  = k & 3;
    ushort packed = weights[((long)kg * N_4 + ng) * 4 + il];
    int nib = ((packed >> (j * 4)) & 0x0F) - 8;
    half sc = scales[((long)(k >> 4) * N_4 + ng) * 4 + il];
    acc += (float)act[k] * ((float)nib * (float)sc);
  }
  partial[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lws >> 1; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) dst[n] = (half)partial[0];
}
#endif // SUPPORTS_FP16
