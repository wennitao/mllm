//==============================================================================
// FlashAttentionInt8KV custom op for LLaMAPackage
//
// Scaled dot-product attention with FlashAttention online softmax, designed
// for the QDQ Qwen3 graph where:
//   * Q activations are u16-asym, dequantized by QNN to fp16 at the op
//     boundary (so the kernel sees Q as plain fp16),
//   * K and V live in shared memory as u8 with per-tensor symmetric quant
//     (zp = 128, scale supplied as a scalar param).
//
// Inner-loop precision: hf x hf -> qf16 -> qf32 accumulate. fp16 mpy gives
// 2x the throughput of the fp32 path used by FlashAttention.cpp. The online
// softmax stats (m, l) and the running output stay in fp32 for stability.
//
// Layout (NHWC, same as FlashAttention.cpp so the two ops are interchangeable
// at the graph level):
//   in[0] Q : [B, S_q,  H_q,  D]   FLOAT_16
//   in[1] K : [B, S_kv, H_kv, D]   UFIXED_POINT_8 (zp=128 sym)
//   in[2] V : [B, S_kv, H_kv, D]   UFIXED_POINT_8 (zp=128 sym)
//   out[0]  : [B, S_q,  H_q,  D]   FLOAT_16
//
// The model's existing K transpose to [B, H_kv, D, S] (used because matmul
// needed K^T) is dropped — this op consumes K in the same NHWC layout as V.
//
// GQA is supported: H_q must be a multiple of H_kv.
//
// Parameters (in order, see DEF_PACKAGE_PARAM_ORDER below):
//   softmax_scale : float scalar (== 1/sqrt(D) on the host side)
//   k_scale       : float scalar (per-tensor scale for K, real = (u8 - 128) * k_scale)
//   v_scale       : float scalar (per-tensor scale for V)
//   causal        : uint32 scalar (0 / 1)
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include <math.h>
#include <stddef.h>

BEGIN_PKG_OP_DEFINITION(PKG_FlashAttentionInt8KV);

template<typename TensorType>
GraphStatus flashattentionint8kvImpl(TensorType& out_0, const TensorType& q_in, const TensorType& k_in,
                                     const TensorType& v_in, const PlainFloatTensor& softmax_scale,
                                     const PlainFloatTensor& k_scale, const PlainFloatTensor& v_scale,
                                     const Tensor& causal, hnnx::op_slice_spec slice_spec);

DEF_PACKAGE_OP((flashattentionint8kvImpl<Tensor>), "FlashAttentionInt8KV")

DEF_PACKAGE_PARAM_ORDER("FlashAttentionInt8KV",
                        "softmax_scale", true, nullptr,
                        "k_scale",       true, nullptr,
                        "v_scale",       true, nullptr,
                        "causal",        true, nullptr)

// One Q row per slice keeps causal masking exact for prefill: slice_idx maps
// directly to the original S_q position.
#define FAQ_SQ_TILE 1

#ifndef REFERENCE_OP
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("FlashAttentionInt8KV", "Q", "K", "V", "Scale", "Kscale", "Vscale", "Causal"),
    AND(EQ(DTYPE_OF("Q"), DType::Float16),
        EQ(DTYPE_OF("*"), DType::Float16),
        GT(DIM_HEIGHT("*"), FAQ_SQ_TILE),
        CONSTVAL_INT_VALID("Causal", 0),
        EQ(CONSTVAL_INT("Causal", 0), 0)),
    AUTOTHREAD_HVX(1, "I", Op("FlashAttentionInt8KV", TYPICAL_SLICE("Q", "I"), "K", "V",
                               "Scale", "Kscale", "Vscale", "Causal")))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 1,
    Op("FlashAttentionInt8KV", "Q", "K", "V", "Scale", "Kscale", "Vscale", "Causal"),
    AND(EQ(DTYPE_OF("Q"), DType::Float16),
        EQ(DTYPE_OF("*"), DType::Float16),
        GT(DIM_HEIGHT("*"), FAQ_SQ_TILE),
        CONSTVAL_INT_VALID("Causal", 0),
        EQ(CONSTVAL_INT("Causal", 0), 1)),
    AUTOSPLIT(1, "I", FAQ_SQ_TILE, Op("FlashAttentionInt8KV", TYPICAL_SLICE("Q", "I"), "K", "V",
                                       "Scale", "Kscale", "Vscale", "Causal")))
#endif

// S_kv tile size. Same rationale as FlashAttention.cpp — small enough that
// the running output rescale on m updates is cheap, large enough to amortise
// L2 prefetch latency.
#define FAQ_BC 64

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"

#define FAQ_BLOCK_SIZE (8 * 1024 / VLEN)

static HVX_INLINE_ALWAYS uint32_t faq_float_to_bits(float x) {
  union {
    float f;
    uint32_t i;
  } u = {.f = x};
  return u.i;
}

// Pack a single fp16 value into a 32-bit splat of two fp16 values (used by
// Q6_V_vsplat_R to broadcast an hf scalar across an HVX vector).
static HVX_INLINE_ALWAYS uint32_t faq_float_to_fp16s(float x) {
  union {
    int32_t i;
    __fp16 f[2];
  } u = {.f = {(__fp16)x, (__fp16)x}};
  return u.i;
}

// Self-contained scalar expf — libm's expf returns 0 on the HTP at runtime
// (see custom_hvx_op_skill.md §3.1).
static inline float faq_expf(float x) {
  if (x < -87.336544f) return 0.0f;
  if (x > 88.722839f) x = 88.722839f;

  const float LOG2EF = 1.44269504088896341f;
  const float C1 = 0.693359375f;
  const float C2 = -2.12194440e-4f;

  float fx = x * LOG2EF + 0.5f;
  int32_t n = (int32_t)fx;
  if ((float)n > fx) n -= 1;
  float fn = (float)n;

  float r = x - fn * C1;
  r = r - fn * C2;

  float r2 = r * r;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r2 + r;
  p = p + 1.0f;

  union {
    float f;
    int32_t i;
  } u;
  u.i = (n + 127) << 23;
  return p * u.f;
}

// qf32 → sf → spill → 32 scalar adds. Avoid vlalign-based reductions
// (custom_hvx_op_skill.md §3.2).
static inline float faq_hvx_qf32_hsum(HVX_Vector v_qf32) {
  HVX_Vector acc_sf = Q6_Vsf_equals_Vqf32(v_qf32);
  union {
    HVX_Vector v;
    float f[32];
  } u;
  u.v = acc_sf;
  float s = 0.0f;
  for (int i = 0; i < 32; ++i) s += u.f[i];
  return s;
}

// Dequantize one D-element row of u8 (zp=128 sym) into two HVX hf vectors.
// Splits a 128-element u8 vector into a low half (elements 0..63) and a high
// half (elements 64..127). The shuffle reorders the unpacked int16 lanes back
// into source-order, matching the pattern in LLaMADequantize.cpp.
//
// out_lo / out_hi are pre-scale: callers fold the K/V scale into the dot or
// AV step where it is most efficient (one mpy per dot, not per element).
static HVX_INLINE_ALWAYS void faq_dequant_u8_row_to_hf(HVX_Vector u8_row, HVX_Vector* out_hf_lo,
                                                       HVX_Vector* out_hf_hi) {
  HVX_VectorPair widened = Q6_Wh_vadd_VubVub(u8_row, Q6_V_vzero());          // u8 -> u16, 256 lanes (interleaved)
  widened = Q6_W_vshuff_VVR(Q6_V_hi_W(widened), Q6_V_lo_W(widened), -2);     // re-order to source layout
  HVX_Vector zp_vec = Q6_V_vsplat_R(0x00800080);                             // 128 in each int16 lane
  HVX_Vector lo_h = Q6_Vh_vsub_VhVh(Q6_V_lo_W(widened), zp_vec);             // int16, lanes 0..63
  HVX_Vector hi_h = Q6_Vh_vsub_VhVh(Q6_V_hi_W(widened), zp_vec);             // int16, lanes 64..127
  *out_hf_lo = Q6_Vhf_equals_Vh(lo_h);                                       // hf, lanes 0..63
  *out_hf_hi = Q6_Vhf_equals_Vh(hi_h);                                       // hf, lanes 64..127
}

// Sum of (Q[d] * dequant(K[d])) over d in [0, D). D must be a multiple of 128
// (one u8 vector per row). Q is hf; K is u8 (zp=128 sym).
//
// The K dequant scale (k_scale) is applied once at the end on the scalar
// reduction — by linearity, sum(Q · k_scale·(K-128)) = k_scale · sum(Q·(K-128)).
static inline float faq_hvx_dot_hf_u8(const __fp16* restrict q, const uint8_t* restrict k_u8, uint32_t D,
                                      float k_scale) {
  const HVX_UVector* qp = (const HVX_UVector*)q;
  const HVX_UVector* kp = (const HVX_UVector*)k_u8;

  // Two qf32 accumulators — one for the lo and one for the hi half of each
  // hf vector pair. Q6_Wqf32_vmpy_VhfVhf returns 64 qf32 lanes split across
  // a VectorPair, so we cannot collapse to a single accumulator without an
  // extra add.
  HVX_Vector acc_lo = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
  HVX_Vector acc_hi = acc_lo;

  const uint32_t n_u8_vecs = D / 128;       // u8 row -> u8 vectors
  for (uint32_t i = 0; i < n_u8_vecs; ++i) {
    HVX_Vector k_hf_lo, k_hf_hi;
    faq_dequant_u8_row_to_hf(kp[i], &k_hf_lo, &k_hf_hi);

    // Q is two hf vectors per 128 elements (each holds 64 hf lanes).
    HVX_Vector q_hf_lo = qp[2 * i + 0];
    HVX_Vector q_hf_hi = qp[2 * i + 1];

    // hf * hf -> Wqf32 (64 qf32 lanes in a VectorPair).
    HVX_VectorPair p_lo = Q6_Wqf32_vmpy_VhfVhf(q_hf_lo, k_hf_lo);
    HVX_VectorPair p_hi = Q6_Wqf32_vmpy_VhfVhf(q_hf_hi, k_hf_hi);

    acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(p_lo));
    acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_hi_W(p_lo));
    acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_lo_W(p_hi));
    acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(p_hi));
  }

  HVX_Vector acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, acc_hi);
  return faq_hvx_qf32_hsum(acc) * k_scale;
}

// Widen one hf vector (64 lanes) into two sf vectors (32 lanes each), in
// source order. The SDK has no direct Wsf_equals_Vhf — go via Wqf32 by
// multiplying by fp16 1.0, shuffle the qf32 pair to source order, then narrow
// each half to sf. Same pattern as LLaMADequantize.cpp's "_af" helpers.
static HVX_INLINE_ALWAYS void faq_hf_to_sf_pair(HVX_Vector hf_in, HVX_Vector* out_sf_lo, HVX_Vector* out_sf_hi) {
  HVX_Vector one_hf = Q6_V_vsplat_R(0x3C003C00);  // hf 1.0 in both 16-bit halves of each 32-bit lane
  HVX_VectorPair qf32 = Q6_Wqf32_vmpy_VhfVhf(hf_in, one_hf);
  qf32 = Q6_W_vshuff_VVR(Q6_V_hi_W(qf32), Q6_V_lo_W(qf32), -4);
  *out_sf_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(qf32));
  *out_sf_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(qf32));
}

// o_tmp[0..D-1] += alpha * dequant(v_u8[0..D-1]). o_tmp is fp32 (the
// running output accumulator), v is u8. alpha already includes v_scale on
// the host side: callers pass `alpha * v_scale`.
static inline void faq_hvx_axpy_f32_from_u8(float* restrict o_tmp, const uint8_t* restrict v_u8, float alpha_v,
                                            uint32_t D) {
  HVX_UVector* op = (HVX_UVector*)o_tmp;          // fp32 output, unaligned
  const HVX_UVector* vp = (const HVX_UVector*)v_u8;
  HVX_Vector alpha_sf = Q6_V_vsplat_R(faq_float_to_bits(alpha_v));
  HVX_Vector zero = Q6_V_vzero();

  const uint32_t n_u8_vecs = D / 128;
  for (uint32_t i = 0; i < n_u8_vecs; ++i) {
    HVX_Vector v_hf_lo, v_hf_hi;
    faq_dequant_u8_row_to_hf(vp[i], &v_hf_lo, &v_hf_hi);

    // hf (64 lanes) -> two sf (32 lanes) so we can add into the fp32 o_tmp.
    HVX_Vector v_sf_q0, v_sf_q1, v_sf_q2, v_sf_q3;
    faq_hf_to_sf_pair(v_hf_lo, &v_sf_q0, &v_sf_q1);
    faq_hf_to_sf_pair(v_hf_hi, &v_sf_q2, &v_sf_q3);

    HVX_Vector* o_q0 = op + 4 * i + 0;
    HVX_Vector* o_q1 = op + 4 * i + 1;
    HVX_Vector* o_q2 = op + 4 * i + 2;
    HVX_Vector* o_q3 = op + 4 * i + 3;

    HVX_Vector d0 = *o_q0;
    HVX_Vector d1 = *o_q1;
    HVX_Vector d2 = *o_q2;
    HVX_Vector d3 = *o_q3;

    HVX_Vector p0 = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, v_sf_q0);
    HVX_Vector p1 = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, v_sf_q1);
    HVX_Vector p2 = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, v_sf_q2);
    HVX_Vector p3 = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, v_sf_q3);

    HVX_Vector dq0 = Q6_Vqf32_vadd_VsfVsf(d0, zero);
    HVX_Vector dq1 = Q6_Vqf32_vadd_VsfVsf(d1, zero);
    HVX_Vector dq2 = Q6_Vqf32_vadd_VsfVsf(d2, zero);
    HVX_Vector dq3 = Q6_Vqf32_vadd_VsfVsf(d3, zero);

    *o_q0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(p0, dq0));
    *o_q1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(p1, dq1));
    *o_q2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(p2, dq2));
    *o_q3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(p3, dq3));
  }
}

// o_tmp[0..D-1] *= alpha. fp32. D % 32 == 0.
static inline void faq_hvx_scale_f32(float* restrict o_tmp, float alpha, uint32_t D) {
  HVX_UVector* op = (HVX_UVector*)o_tmp;
  HVX_Vector alpha_sf = Q6_V_vsplat_R(faq_float_to_bits(alpha));
  uint32_t nv = D / 32;
  for (uint32_t i = 0; i < nv; ++i) {
    HVX_Vector s = op[i];
    op[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(alpha_sf, s));
  }
}

static inline void faq_hvx_zero_f32(float* restrict o_tmp, uint32_t D) {
  HVX_UVector* op = (HVX_UVector*)o_tmp;
  HVX_Vector zero = Q6_V_vzero();
  uint32_t nv = D / 32;
  for (uint32_t i = 0; i < nv; ++i) op[i] = zero;
}

// Cast fp32 buffer to fp16 over D elements. Scalar loop: this runs once per
// (Q row, head), so cost is negligible vs the inner D×Skv loops.
//
// An HVX-vectorised cast is tempting but tricky here: the only available
// narrow path is Q6_Vhf_equals_Wqf32, which expects its qf32 pair in the
// SDK's interleaved convention (lo = even source lanes, hi = odd source
// lanes — same form Q6_Wqf32_vmpy_VhfVhf produces). Our o_tmp lives in
// source order in memory (sp[0] = lanes 0..31, sp[1] = lanes 32..63, ...),
// so feeding adjacent sp[k], sp[k+1] to the narrow scrambles the output.
// Re-interleaving with vshuff would touch four sf vectors and isn't worth
// the bookkeeping for a single-row cast.
static inline void faq_cast_f32_to_f16(const float* restrict src_f32, __fp16* restrict dst_f16, uint32_t D) {
  for (uint32_t d = 0; d < D; ++d) dst_f16[d] = (__fp16)src_f32[d];
}

// One Q row (one head), full online softmax over [0, S_kv_lim).
// Q: hf [D]. K, V: u8 [S_kv, H_kv*D] block (head-strided rows of u8). O: hf [D].
static void faq_one_row_hf_u8(const __fp16* restrict q,             // [D]
                              const uint8_t* restrict k_base,        // [S_kv, H_kv*D]
                              const uint8_t* restrict v_base,        // [S_kv, H_kv*D]
                              uint32_t kv_stride_elems,              // == H_kv * D
                              __fp16* restrict o,                    // [D]
                              uint32_t S_kv_lim, uint32_t D, float scale, float k_scale, float v_scale) {
  // Running output accumulator stays in fp32 for stability across many
  // rescales. D <= 512 covers all current LLaMA/Qwen head dims.
  float o_tmp[512];
  faq_hvx_zero_f32(o_tmp, D);
  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j0 = 0; j0 < S_kv_lim; j0 += FAQ_BC) {
    uint32_t jN = j0 + FAQ_BC;
    if (jN > S_kv_lim) jN = S_kv_lim;

    if (jN < S_kv_lim) {
      uint32_t fetch_rows = (jN + FAQ_BC <= S_kv_lim) ? FAQ_BC : (S_kv_lim - jN);
      uint32_t k_bytes_per_row = D * sizeof(uint8_t);
      uint32_t v_bytes_per_row = D * sizeof(uint8_t);
      l2fetch(k_base + (size_t)jN * kv_stride_elems, k_bytes_per_row, k_bytes_per_row, fetch_rows, 0);
      l2fetch(v_base + (size_t)jN * kv_stride_elems, v_bytes_per_row, v_bytes_per_row, fetch_rows, 0);
    }

    // 1) S_ij = Q · K_j * scale * k_scale, plus rowmax for this tile.
    float s_buf[FAQ_BC];
    float m_tile = -INFINITY;
    const float qk_combined_scale = scale;  // k_scale folded inside the dot
    for (uint32_t j = j0; j < jN; ++j) {
      const uint8_t* kptr = k_base + (size_t)j * kv_stride_elems;
      float dot = faq_hvx_dot_hf_u8(q, kptr, D, k_scale);
      float s = dot * qk_combined_scale;
      s_buf[j - j0] = s;
      if (s > m_tile) m_tile = s;
    }

    // 2) Online softmax update.
    float m_new = (m > m_tile) ? m : m_tile;
    float alpha = (m == -INFINITY) ? 0.0f : faq_expf(m - m_new);
    float l_tile = 0.0f;
    for (uint32_t j = j0; j < jN; ++j) {
      float p = faq_expf(s_buf[j - j0] - m_new);
      s_buf[j - j0] = p;
      l_tile += p;
    }

    // 3) Rescale running output, then accumulate p_j * dequant(V_j).
    if (alpha != 1.0f) faq_hvx_scale_f32(o_tmp, alpha, D);
    for (uint32_t j = j0; j < jN; ++j) {
      const uint8_t* vptr = v_base + (size_t)j * kv_stride_elems;
      faq_hvx_axpy_f32_from_u8(o_tmp, vptr, s_buf[j - j0] * v_scale, D);
    }

    l = alpha * l + l_tile;
    m = m_new;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  faq_hvx_scale_f32(o_tmp, inv_l, D);
  faq_cast_f32_to_f16(o_tmp, o, D);
}

static void faq_flashattention_hf_u8(__fp16* restrict Obuf, const __fp16* restrict Qbuf, const uint8_t* restrict Kbuf,
                                     const uint8_t* restrict Vbuf, uint32_t Bq, uint32_t Sq, uint32_t Sk, uint32_t Hq,
                                     uint32_t Hkv, uint32_t Dq, uint32_t q_stride, uint32_t kv_stride,
                                     uint32_t group, float scale, float k_scale, float v_scale, uint32_t is_causal,
                                     uint32_t global_sq_start, uint32_t original_sq) {
  for (uint32_t b = 0; b < Bq; ++b) {
    const __fp16* Qb = Qbuf + (size_t)b * Sq * q_stride;
    const uint8_t* Kb = Kbuf + (size_t)b * Sk * kv_stride;
    const uint8_t* Vb = Vbuf + (size_t)b * Sk * kv_stride;
    __fp16* Ob = Obuf + (size_t)b * Sq * q_stride;
    for (uint32_t s = 0; s < Sq; ++s) {
      uint32_t kv_lim = Sk;
      if (is_causal && original_sq > 1) kv_lim = (Sk - original_sq) + global_sq_start + s + 1;
      for (uint32_t hq = 0; hq < Hq; ++hq) {
        uint32_t hkv = hq / group;
        const __fp16* qrow = Qb + (size_t)s * q_stride + hq * Dq;
        const uint8_t* krow = Kb + hkv * Dq;
        const uint8_t* vrow = Vb + hkv * Dq;
        __fp16* orow = Ob + (size_t)s * q_stride + hq * Dq;
        faq_one_row_hf_u8(qrow, krow, vrow, kv_stride, orow, kv_lim, Dq, scale, k_scale, v_scale);
      }
    }
  }
}

#endif  // !REFERENCE_OP

// Reference / aarch64 path. Plain C++; matches the HVX recurrence so the
// graph builder can validate the op and so the on-device test has a known-
// good baseline if HVX is disabled.
static inline void faq_one_row_ref_hf_u8(const __fp16* q, const uint8_t* k_base, const uint8_t* v_base,
                                         uint32_t kv_stride_elems, __fp16* o, uint32_t S_kv_lim, uint32_t D,
                                         float scale, float k_scale, float v_scale) {
  float o_tmp[512];
  for (uint32_t d = 0; d < D; ++d) o_tmp[d] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j0 = 0; j0 < S_kv_lim; j0 += FAQ_BC) {
    uint32_t jN = j0 + FAQ_BC;
    if (jN > S_kv_lim) jN = S_kv_lim;

    float s_buf[FAQ_BC];
    float m_tile = -INFINITY;
    for (uint32_t j = j0; j < jN; ++j) {
      const uint8_t* kptr = k_base + (size_t)j * kv_stride_elems;
      float dot = 0.0f;
      for (uint32_t d = 0; d < D; ++d) {
        float k_real = (float)((int32_t)kptr[d] - 128) * k_scale;
        dot += (float)q[d] * k_real;
      }
      float s = dot * scale;
      s_buf[j - j0] = s;
      if (s > m_tile) m_tile = s;
    }

    float m_new = (m > m_tile) ? m : m_tile;
    float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
    float l_tile = 0.0f;
    for (uint32_t j = j0; j < jN; ++j) {
      float p = expf(s_buf[j - j0] - m_new);
      s_buf[j - j0] = p;
      l_tile += p;
    }

    if (alpha != 1.0f) {
      for (uint32_t d = 0; d < D; ++d) o_tmp[d] *= alpha;
    }
    for (uint32_t j = j0; j < jN; ++j) {
      const uint8_t* vptr = v_base + (size_t)j * kv_stride_elems;
      float p = s_buf[j - j0];
      for (uint32_t d = 0; d < D; ++d) {
        float v_real = (float)((int32_t)vptr[d] - 128) * v_scale;
        o_tmp[d] += p * v_real;
      }
    }

    l = alpha * l + l_tile;
    m = m_new;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (uint32_t d = 0; d < D; ++d) o[d] = (__fp16)(o_tmp[d] * inv_l);
}

template<typename TensorType>
GraphStatus flashattentionint8kvImpl(TensorType& out_0, const TensorType& q_in, const TensorType& k_in,
                                     const TensorType& v_in, const PlainFloatTensor& softmax_scale,
                                     const PlainFloatTensor& k_scale, const PlainFloatTensor& v_scale,
                                     const Tensor& causal, hnnx::op_slice_spec slice_spec) {
  out_0.set_dims(q_in);

  auto [Bq, Sq, Hq, Dq] = q_in.dims();
  auto [Bk, Sk, Hkv, Dk] = k_in.dims();
  auto [Bv, Sv, Hkv2, Dv] = v_in.dims();

  if (Bq != Bk || Bq != Bv) return GraphStatus::ErrorDimensions;
  if (Sk != Sv) return GraphStatus::ErrorDimensions;
  if (Hkv != Hkv2) return GraphStatus::ErrorDimensions;
  if (Dq != Dk || Dq != Dv) return GraphStatus::ErrorDimensions;
  if (Hq % Hkv != 0) return GraphStatus::ErrorDimensions;
  // u8 dequant unrolls one HVX vector (128 lanes) per row.
  if ((Dq & 127) != 0) return GraphStatus::ErrorDimensions;

  const uint32_t slice_count = slice_spec.num_slices();
  const uint32_t slice_idx = slice_spec.slice_idx();
  const uint32_t original_sq = (slice_count > 1 && Sq == FAQ_SQ_TILE) ? slice_count : (uint32_t)Sq;
  const uint32_t global_sq_start = (slice_count > 1 && Sq == FAQ_SQ_TILE) ? slice_idx : 0;
  if (Sk < original_sq) return GraphStatus::ErrorDimensions;

  const uint32_t group = Hq / Hkv;
  const uint32_t kv_stride = Hkv * Dq;
  const uint32_t q_stride = Hq * Dq;
  const float scale = softmax_scale(0, 0, 0, 0);
  const float k_scale_v = k_scale(0, 0, 0, 0);
  const float v_scale_v = v_scale(0, 0, 0, 0);
  const uint32_t is_causal = (uint32_t)causal(0, 0, 0, 0);

  if (q_in.get_dtype() != DType::Float16 || out_0.get_dtype() != DType::Float16) {
    return GraphStatus::ErrorUnsupported;
  }
  if (k_in.get_dtype() != DType::QUInt8 || v_in.get_dtype() != DType::QUInt8) {
    return GraphStatus::ErrorUnsupported;
  }

  const __fp16* Qbuf = (const __fp16*)q_in.raw_data_const();
  const uint8_t* Kbuf = (const uint8_t*)k_in.raw_data_const();
  const uint8_t* Vbuf = (const uint8_t*)v_in.raw_data_const();
  __fp16* Obuf = (__fp16*)out_0.raw_data();

#ifndef REFERENCE_OP
  faq_flashattention_hf_u8(Obuf, Qbuf, Kbuf, Vbuf, (uint32_t)Bq, (uint32_t)Sq, (uint32_t)Sk, (uint32_t)Hq,
                           (uint32_t)Hkv, (uint32_t)Dq, q_stride, kv_stride, group, scale, k_scale_v,
                           v_scale_v, is_causal, global_sq_start, original_sq);
#else
  for (uint32_t b = 0; b < (uint32_t)Bq; ++b) {
    const __fp16* Qb = Qbuf + (size_t)b * Sq * q_stride;
    const uint8_t* Kb = Kbuf + (size_t)b * Sk * kv_stride;
    const uint8_t* Vb = Vbuf + (size_t)b * Sk * kv_stride;
    __fp16* Ob = Obuf + (size_t)b * Sq * q_stride;
    for (uint32_t s = 0; s < (uint32_t)Sq; ++s) {
      uint32_t kv_lim = (uint32_t)Sk;
      if (is_causal && original_sq > 1) kv_lim = (Sk - original_sq) + global_sq_start + s + 1;
      for (uint32_t hq = 0; hq < (uint32_t)Hq; ++hq) {
        uint32_t hkv = hq / group;
        const __fp16* qrow = Qb + (size_t)s * q_stride + hq * Dq;
        const uint8_t* krow = Kb + hkv * Dq;
        const uint8_t* vrow = Vb + hkv * Dq;
        __fp16* orow = Ob + (size_t)s * q_stride + hq * Dq;
        faq_one_row_ref_hf_u8(qrow, krow, vrow, kv_stride, orow, kv_lim, Dq, scale, k_scale_v, v_scale_v);
      }
    }
  }
#endif

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_FlashAttentionInt8KV);
