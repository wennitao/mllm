#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rope_f32(__global const float *src, __global float *dst,
                       __global const float *sin_table,
                       __global const float *cos_table, const int H,
                       const int S, const int D) {
  int d_half = get_global_id(0);
  int s = get_global_id(1);
  int bh = get_global_id(2);

  if (d_half >= D / 2)
    return;

  int b = bh / H;
  int h = bh % H;

  int d = d_half;
  int d_pair = d + D / 2;

  // Input shape: [B, H, S, D]
  int in_idx1 = b * (H * S * D) + h * (S * D) + s * D + d;
  int in_idx2 = b * (H * S * D) + h * (S * D) + s * D + d_pair;

  // Sin/Cos shape: [B, S, D]
  int sc_idx1 = b * (S * D) + s * D + d;

  float v1 = src[in_idx1];
  float v2 = src[in_idx2];

  float s1 = sin_table[sc_idx1];
  float c1 = cos_table[sc_idx1];

  float out1 = v1 * c1 - v2 * s1;
  float out2 = v1 * s1 + v2 * c1;

  dst[in_idx1] = out1;
  dst[in_idx2] = out2;
}

__kernel void rope_f16(__global const half *src, __global half *dst,
                       __global const float *sin_table,
                       __global const float *cos_table, const int H,
                       const int S, const int D) {
  int d_half = get_global_id(0);
  int s = get_global_id(1);
  int bh = get_global_id(2);

  if (d_half >= D / 2)
    return;

  int b = bh / H;
  int h = bh % H;

  int d = d_half;
  int d_pair = d + D / 2;

  // Input shape: [B, H, S, D]
  int in_idx1 = b * (H * S * D) + h * (S * D) + s * D + d;
  int in_idx2 = b * (H * S * D) + h * (S * D) + s * D + d_pair;

  // Sin/Cos shape: [B, S, D]
  int sc_idx1 = b * (S * D) + s * D + d;

  float v1 = (float)src[in_idx1];
  float v2 = (float)src[in_idx2];

  float s1 = sin_table[sc_idx1];
  float c1 = cos_table[sc_idx1];

  float out1 = v1 * c1 - v2 * s1;
  float out2 = v1 * s1 + v2 * c1;

  dst[in_idx1] = (half)out1;
  dst[in_idx2] = (half)out2;
}

__kernel void rope_bshd_f32(__global const void *src_void, __global void *dst_void,
                            __global const float *sin_table,
                            __global const float *cos_table,
                            ulong src_offset_bytes, ulong dst_offset_bytes,
                            const int H, const int S, const int D) {
  int d_half = get_global_id(0);
  int s = get_global_id(1);
  int bh = get_global_id(2);

  if (d_half >= D / 2)
    return;

  int b = bh / H;
  int h = bh % H;
  int d = d_half;
  int d_pair = d + D / 2;

  const global float* src = (const global float*)((const global char*)src_void + src_offset_bytes);
  global float* dst = (global float*)((global char*)dst_void + dst_offset_bytes);

  // Input shape: [B, S, H, D]
  int in_idx1 = b * (S * H * D) + s * (H * D) + h * D + d;
  int in_idx2 = b * (S * H * D) + s * (H * D) + h * D + d_pair;

  // Sin/Cos shape: [B, S, D]
  int sc_idx1 = b * (S * D) + s * D + d;

  float v1 = src[in_idx1];
  float v2 = src[in_idx2];
  float s1 = sin_table[sc_idx1];
  float c1 = cos_table[sc_idx1];

  dst[in_idx1] = v1 * c1 - v2 * s1;
  dst[in_idx2] = v1 * s1 + v2 * c1;
}

__kernel void rope_bshd_f16(__global const void *src_void, __global void *dst_void,
                            __global const float *sin_table,
                            __global const float *cos_table,
                            ulong src_offset_bytes, ulong dst_offset_bytes,
                            const int H, const int S, const int D) {
  int d_half = get_global_id(0);
  int s = get_global_id(1);
  int bh = get_global_id(2);

  if (d_half >= D / 2)
    return;

  int b = bh / H;
  int h = bh % H;
  int d = d_half;
  int d_pair = d + D / 2;

  const global half* src = (const global half*)((const global char*)src_void + src_offset_bytes);
  global half* dst = (global half*)((global char*)dst_void + dst_offset_bytes);

  // Input shape: [B, S, H, D]
  int in_idx1 = b * (S * H * D) + s * (H * D) + h * D + d;
  int in_idx2 = b * (S * H * D) + s * (H * D) + h * D + d_pair;

  // Sin/Cos shape: [B, S, D]
  int sc_idx1 = b * (S * D) + s * D + d;

  float v1 = (float)src[in_idx1];
  float v2 = (float)src[in_idx2];
  float s1 = sin_table[sc_idx1];
  float c1 = cos_table[sc_idx1];

  dst[in_idx1] = (half)(v1 * c1 - v2 * s1);
  dst[in_idx2] = (half)(v1 * s1 + v2 * c1);
}
