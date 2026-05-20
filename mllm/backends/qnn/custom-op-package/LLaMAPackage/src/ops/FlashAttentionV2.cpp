//==============================================================================
// FlashAttentionV2 custom op for LLaMAPackage
//
// fp16 FlashAttention with the kernel-level optimisations from llama.cpp's
// htp/flash-attn-ops.c:
//   * rx32 dot: 32 Q·K dot products per call, internally 8× rx4 (4 K rows
//     per Q load → reuses Q across 4 mpyacc accumulators).
//   * rx2 V mad: 2 V rows per call (acc += p0*V0 + p1*V1) — better register
//     allocation for the AV step.
//   * Vector M / S running stats — rescale of VKQ32 stays HVX-vectorised
//     instead of scalar splat-on-the-fly.
//   * Sub-block scores buffer reused for softmax (no recompute).
//
// Intentionally skipped vs flash-attn-ops.c (out of scope for a QNN custom
// op or non-portable):
//   * VTCM staging + DMA double-buffer (no direct VTCM API in QNN custom ops).
//   * dspqueue / worker pool (QNN handles threading via AUTOTHREAD).
//   * HMX dispatch (not exposed to custom ops).
//   * Mask tensor input, sinks, softcap (kept the V1 `causal` flag for parity).
//
// Layout matches V1 / FlashAttention.cpp (NHWC):
//   in[0] Q : [B, S_q,  H_q,  D]   FLOAT_16
//   in[1] K : [B, S_kv, H_kv, D]   FLOAT_16
//   in[2] V : [B, S_kv, H_kv, D]   FLOAT_16
//   out[0]  : [B, S_q,  H_q,  D]   FLOAT_16
//
// GQA supported (H_q must be a multiple of H_kv).
//
// Parameters:
//   softmax_scale : float scalar (== 1/sqrt(D))
//   causal        : uint32 scalar (0/1)
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

BEGIN_PKG_OP_DEFINITION(PKG_FlashAttentionV2);

template<typename TensorType>
GraphStatus flashattentionv2Impl(TensorType& out_0, const TensorType& q_in, const TensorType& k_in, const TensorType& v_in,
                                 const PlainFloatTensor& softmax_scale, const Tensor& causal,
                                 hnnx::op_slice_spec slice_spec);

DEF_PACKAGE_OP((flashattentionv2Impl<Tensor>), "FlashAttentionV2")

DEF_PACKAGE_PARAM_ORDER("FlashAttentionV2", "softmax_scale", true, nullptr, "causal", true, nullptr)

// Q-row tile: how many Q rows share each K/V tile load. The dominant cost in
// causal prefill is K-row reads (Q row q reads K[0..q]), so tiling BQ Q rows
// together reduces total K-row loads by ~BQ× when those rows fit in cache.
// BQ=4 keeps the per-tile state (4 × VKQ32[D] + sb_scores) under ~3 KB stack
// for D ≤ 128. Bump to 8 for longer head dims; mind FA2_VKQ32_D_MAX below.
#define FA2_BQ_TILE     4
#define FA2_VKQ32_D_MAX 512
#define FA2_BC          64

#ifndef REFERENCE_OP
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("FlashAttentionV2", "Q", "K", "V", "Scale", "Causal"),
    AND(EQ(DTYPE_OF("Q"), DType::Float16),
        EQ(DTYPE_OF("*"), DType::Float16),
        GT(DIM_HEIGHT("*"), FA2_BQ_TILE),
        CONSTVAL_INT_VALID("Causal", 0),
        EQ(CONSTVAL_INT("Causal", 0), 0)),
    AUTOTHREAD_HVX(1, "I", Op("FlashAttentionV2", TYPICAL_SLICE("Q", "I"), "K", "V", "Scale", "Causal")))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 1,
    Op("FlashAttentionV2", "Q", "K", "V", "Scale", "Causal"),
    AND(EQ(DTYPE_OF("Q"), DType::Float16),
        EQ(DTYPE_OF("*"), DType::Float16),
        GT(DIM_HEIGHT("*"), FA2_BQ_TILE),
        CONSTVAL_INT_VALID("Causal", 0),
        EQ(CONSTVAL_INT("Causal", 0), 1)),
    AUTOSPLIT(1, "I", FA2_BQ_TILE, Op("FlashAttentionV2", TYPICAL_SLICE("Q", "I"), "K", "V", "Scale", "Causal")))
#endif

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"

// ---------------------------------------------------------------------------
// Helpers — adapted from htp/hvx-base.h, htp/hvx-reduce.h, htp/hvx-exp.h,
// htp/hvx-floor.h. Inlined here to keep the op self-contained (the htp/
// folder lives in a separate runtime stack and isn't on this op's include
// path). All credit to the original ggml-hexagon authors.
// ---------------------------------------------------------------------------

#define FA2_VLEN      128
#define FA2_VLEN_FP32 32
#define FA2_VLEN_FP16 64

typedef struct {
  HVX_Vector v[4];
} FA2_HVX_Vector_x4;

static HVX_INLINE_ALWAYS HVX_Vector fa2_splat_f32(float v) {
  union {
    float f;
    uint32_t i;
  } u = {.f = v};
  return Q6_V_vsplat_R(u.i);
}

static HVX_INLINE_ALWAYS HVX_Vector fa2_splat_f16_bits(uint16_t bits) {
  return Q6_Vh_vsplat_R(bits);
}

// Splat a fp16 value loaded from memory (callers must keep the value
// addressable — __fp16 isn't allowed as a function parameter on this
// compiler, hence the pointer indirection).
static HVX_INLINE_ALWAYS HVX_Vector fa2_splat_f16_p(const __fp16* p) {
  union {
    __fp16 f;
    uint16_t i;
  } u = {.f = *p};
  return Q6_Vh_vsplat_R(u.i);
}

static HVX_INLINE_ALWAYS float fa2_get_f32(HVX_Vector v) {
  // Aligned spill of a single fp32 lane — matches hvx_vec_get_f32 in hvx-base.h.
  union {
    HVX_Vector v;
    float f[32];
  } u __attribute__((aligned(128)));
  u.v = v;
  return u.f[0];
}

// fp16 × fp16 → fp32 multiply-accumulate (ports hvx_vec_mpyacc_f32_f16).
// The pre-v79 path manually widens via Wqf32_vmpy + add to acc; v79+ has a
// direct fused intrinsic.
static HVX_INLINE_ALWAYS HVX_VectorPair fa2_mpyacc_f32_f16(HVX_VectorPair acc, HVX_Vector x, HVX_Vector y) {
#if __HVX_ARCH__ >= 79
  return Q6_Wsf_vmpyacc_WsfVhfVhf(acc, x, y);
#else
  HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(x, y);
  HVX_Vector a0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(m), Q6_V_lo_W(acc)));
  HVX_Vector a1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(m), Q6_V_hi_W(acc)));
  return Q6_W_vcombine_VV(a1, a0);
#endif
}

// 4-way reduce: each input vector is reduced to a scalar; 4 scalars packed
// into lanes 0..3 of the output, replicated every 4 lanes across the rest.
// Ports hvx_vec_reduce_sum_f32x4 (v76+ branch).
static HVX_INLINE_ALWAYS HVX_Vector fa2_reduce_sum_f32x4(FA2_HVX_Vector_x4 in) {
  HVX_VectorPair sum_p01 = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
  HVX_VectorPair sum_p23 = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
  HVX_Vector sum_sf01 = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
  HVX_Vector sum_sf23 = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

  HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(sum_sf23, sum_sf01, 8);
  HVX_Vector sum_sf = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

  sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, FA2_VLEN / 2));
  sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, FA2_VLEN / 4));
  sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, FA2_VLEN / 8));
  return sum_sf;
}

// Reduce-sum across 32 fp32 lanes via rotate-add tree.
static HVX_INLINE_ALWAYS HVX_Vector fa2_reduce_sum_f32(HVX_Vector in) {
  HVX_Vector sum = in;
  unsigned int width = 4;
  unsigned int total = FA2_VLEN_FP32 * 4;
  while (width < total) {
    HVX_Vector t = Q6_V_vror_VR(sum, width);
    sum = Q6_Vsf_vadd_VsfVsf(sum, t);
    width <<= 1;
  }
  return sum;
}

// Reduce-max-and-broadcast, accumulating against an existing running max.
static HVX_INLINE_ALWAYS HVX_Vector fa2_reduce_max2_f32(HVX_Vector in, HVX_Vector cur_max) {
  HVX_Vector m = Q6_Vsf_vmax_VsfVsf(in, cur_max);
  unsigned int width = 4;
  while (width < FA2_VLEN) {
    HVX_Vector t = Q6_V_vror_VR(m, width);
    m = Q6_Vsf_vmax_VsfVsf(t, m);
    width <<= 1;
  }
  return m;
}

// fp32 truncation / floor — needed by exp_f32. Adapted from hvx-floor.h.
#define FA2_VSF_EXPLEN  (8)
#define FA2_VSF_EXPBIAS (127)
#define FA2_VSF_EXPMASK (0xFF)
#define FA2_VSF_MANTLEN (23)
#define FA2_VSF_MANTMASK (0x7FFFFF)
#define FA2_VSF_MIMPMASK (0x800000)

static HVX_INLINE_ALWAYS HVX_Vector fa2_truncate_f32(HVX_Vector in_vec) {
  HVX_Vector mask_mant_v = Q6_V_vsplat_R(FA2_VSF_MANTMASK);
  HVX_Vector mask_impl_v = Q6_V_vsplat_R(FA2_VSF_MIMPMASK);
  HVX_Vector zero_v = Q6_V_vzero();

  HVX_VectorPred q_neg = Q6_Q_vcmp_gt_VwVw(zero_v, in_vec);
  HVX_Vector expval_v = in_vec >> FA2_VSF_MANTLEN;
  expval_v &= FA2_VSF_EXPMASK;
  expval_v -= FA2_VSF_EXPBIAS;
  HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(zero_v, expval_v);

  HVX_Vector rshift_v = FA2_VSF_MANTLEN - expval_v;
  HVX_Vector mant_v = in_vec & mask_mant_v;
  HVX_Vector vout = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);
  vout = Q6_Vw_vasr_VwVw(vout, rshift_v);
  vout = Q6_V_vmux_QVV(q_negexp, zero_v, vout);

  HVX_Vector neg_vout = -vout;
  vout = Q6_V_vmux_QVV(q_neg, neg_vout, vout);
  return vout;
}

static HVX_INLINE_ALWAYS HVX_Vector fa2_floor_f32(HVX_Vector in_vec) {
  HVX_Vector mask_mant_v = Q6_V_vsplat_R(FA2_VSF_MANTMASK);
  HVX_Vector mask_impl_v = Q6_V_vsplat_R(FA2_VSF_MIMPMASK);
  HVX_Vector mnlen_v = Q6_V_vsplat_R(FA2_VSF_MANTLEN);
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_Vector negone_v = Q6_V_vsplat_R(0xbf800000);

  HVX_VectorPred q_neg = Q6_Q_vcmp_gt_VwVw(zero_v, in_vec);
  HVX_Vector expval_v = in_vec >> FA2_VSF_MANTLEN;
  expval_v &= FA2_VSF_EXPMASK;
  expval_v -= FA2_VSF_EXPBIAS;

  HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(zero_v, expval_v);
  HVX_VectorPred q_expltmn = Q6_Q_vcmp_gt_VwVw(mnlen_v, expval_v);
  HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, in_vec, zero_v);
  HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, zero_v, in_vec);

  mask_mant_v >>= expval_v;
  HVX_Vector neg_addin_v = mask_impl_v >> expval_v;
  HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(in_vec, neg_addin_v);
  HVX_Vector vout = Q6_V_vmux_QVV(q_neg, vout_neg_addin, in_vec);

  HVX_Vector mask_chk_v = Q6_V_vand_VV(in_vec, mask_mant_v);
  HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(zero_v, mask_chk_v);

  HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);
  HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);

  vout = in_vec;
  vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);
  vout = Q6_V_vmux_QVV(q_integral, in_vec, vout);
  vout = Q6_V_vmux_QVV(q_negexp_pos, zero_v, vout);
  vout = Q6_V_vmux_QVV(q_negexp_neg, negone_v, vout);
  return vout;
}

// Vectorised exp_f32 — Cephes-style polynomial. Adapted from hvx-exp.h.
#define FA2_EXP_C5      (0x3D2AAAA0)
#define FA2_EXP_C4      (0x3D2AB327)
#define FA2_EXP_C3      (0x3E2AAAAA)
#define FA2_EXP_C2      (0x3E08CBB6)
#define FA2_EXP_C1      (0x3E2AAAAA)
#define FA2_EXP_C0      (0x3F000000)
#define FA2_EXP_LOGN2   (0x3F317218)
#define FA2_EXP_LOG2E   (0x3FB8AA3B)
#define FA2_EXP_ONE     (0x3F800000)
#define FA2_EXP_RANGE_R (0x42B17218)
#define FA2_EXP_RANGE_L (0xC2B00000)

static HVX_INLINE_ALWAYS HVX_Vector fa2_exp_f32(HVX_Vector in_vec) {
  HVX_Vector log2e = Q6_V_vsplat_R(FA2_EXP_LOG2E);
  HVX_Vector logn2 = Q6_V_vsplat_R(FA2_EXP_LOGN2);
  HVX_Vector zero_v = Q6_V_vzero();
  HVX_Vector temp_v = in_vec;

  HVX_VectorPred pred_cap_r = Q6_Q_vcmp_gt_VsfVsf(in_vec, Q6_V_vsplat_R(FA2_EXP_RANGE_R));
  HVX_VectorPred pred_cap_l = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(FA2_EXP_RANGE_L), in_vec);
  in_vec = Q6_V_vmux_QVV(pred_cap_r, Q6_V_vsplat_R(FA2_EXP_RANGE_R), temp_v);
  in_vec = Q6_V_vmux_QVV(pred_cap_l, Q6_V_vsplat_R(FA2_EXP_RANGE_L), in_vec);

  HVX_Vector epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, in_vec);
  epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

  HVX_Vector f_v = fa2_floor_f32(epsilon_v);
  HVX_Vector k_v = fa2_truncate_f32(f_v);

  HVX_Vector x_qf32_v = Q6_Vqf32_vadd_VsfVsf(in_vec, zero_v);
  epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
  x_qf32_v = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
  x_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);
  HVX_Vector x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

  HVX_Vector z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
  z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

  HVX_Vector E = Q6_V_vsplat_R(FA2_EXP_C5);
  HVX_Vector y_v = Q6_Vqf32_vmpy_VsfVsf(E, x_v);
  E = Q6_V_vsplat_R(FA2_EXP_C4);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  E = Q6_V_vsplat_R(FA2_EXP_C3);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  E = Q6_V_vsplat_R(FA2_EXP_C2);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  E = Q6_V_vsplat_R(FA2_EXP_C1);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  E = Q6_V_vsplat_R(FA2_EXP_C0);
  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

  y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);
  y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(FA2_EXP_ONE));

  y_v = Q6_Vsf_equals_Vqf32(y_v);

  HVX_Vector y_v_exp = Q6_Vw_vasl_VwR(y_v, 1);
  y_v_exp = Q6_Vuw_vlsr_VuwR(y_v_exp, FA2_VSF_MANTLEN + 1);
  y_v_exp = Q6_Vw_vadd_VwVw(k_v, y_v_exp);
  HVX_VectorPred q_neg_exp = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exp);

  y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, FA2_VSF_MANTLEN);
  y_v = Q6_V_vmux_QVV(q_neg_exp, zero_v, y_v);
  return y_v;
}

// fp32 pair → fp16 vector. Adapted from hvx_vec_f32_to_f16. Goes via qf32 to
// match the SDK's interleaved pair convention required by Vhf_equals_Wqf32,
// then a final vdeal to undo the interleave.
static HVX_INLINE_ALWAYS HVX_Vector fa2_f32_to_f16(HVX_Vector v0, HVX_Vector v1) {
#if __HVX_ARCH__ >= 81
  HVX_Vector q0 = Q6_Vqf32_equals_Vsf(v0);
  HVX_Vector q1 = Q6_Vqf32_equals_Vsf(v1);
#else
  const HVX_Vector zero = Q6_V_vzero();
  HVX_Vector q0 = Q6_Vqf32_vadd_VsfVsf(v0, zero);
  HVX_Vector q1 = Q6_Vqf32_vadd_VsfVsf(v1, zero);
#endif
  HVX_Vector hf = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
  return Q6_Vh_vdeal_Vh(hf);
}

// VKQ32 *= alpha — fp32 scale of the running output. From hvx_scale_vec_f32_aa.
static HVX_INLINE_ALWAYS void fa2_scale_vec_f32(float* restrict dst, const float* restrict src, uint32_t n,
                                                HVX_Vector vs) {
  const HVX_Vector* restrict vsrc = (const HVX_Vector*)src;
  HVX_Vector* restrict vdst = (HVX_Vector*)dst;
  uint32_t nvec = n / FA2_VLEN_FP32;
#pragma unroll(4)
  for (uint32_t i = 0; i < nvec; ++i) {
    vdst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs));
  }
}

// Splat-zero into an aligned fp32 buffer.
static HVX_INLINE_ALWAYS void fa2_zero_f32(float* restrict dst, uint32_t n) {
  HVX_Vector* vdst = (HVX_Vector*)dst;
  HVX_Vector zero = Q6_V_vzero();
  uint32_t nvec = n / FA2_VLEN_FP32;
  for (uint32_t i = 0; i < nvec; ++i) vdst[i] = zero;
}

// 4 K rows × 1 Q row dot product, returned as 4 reductions in lanes 0..3
// (and replicated every 4 lanes across the rest of the vector). x rows are
// strided by stride_x bytes. Adapted from hvx_dot_f16_f16_aa_rx4.
static HVX_INLINE_ALWAYS HVX_Vector fa2_dot_hf_rx4(const __fp16* restrict q,
                                                   const uint8_t* restrict k_base,
                                                   size_t stride_k, size_t nvec) {
  const HVX_Vector* restrict vq = (const HVX_Vector*)q;
  const HVX_Vector* restrict vk0 = (const HVX_Vector*)k_base;
  const HVX_Vector* restrict vk1 = (const HVX_Vector*)(k_base + stride_k);
  const HVX_Vector* restrict vk2 = (const HVX_Vector*)(k_base + stride_k * 2);
  const HVX_Vector* restrict vk3 = (const HVX_Vector*)(k_base + stride_k * 3);

  HVX_VectorPair s0 = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
  HVX_VectorPair s1 = s0, s2 = s0, s3 = s0;

  for (uint32_t i = 0; i < nvec; ++i) {
    HVX_Vector q_hf = vq[i];
    s0 = fa2_mpyacc_f32_f16(s0, vk0[i], q_hf);
    s1 = fa2_mpyacc_f32_f16(s1, vk1[i], q_hf);
    s2 = fa2_mpyacc_f32_f16(s2, vk2[i], q_hf);
    s3 = fa2_mpyacc_f32_f16(s3, vk3[i], q_hf);
  }

  HVX_Vector r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(s0), Q6_V_hi_W(s0)));
  HVX_Vector r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(s1), Q6_V_hi_W(s1)));
  HVX_Vector r2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(s2), Q6_V_hi_W(s2)));
  HVX_Vector r3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(s3), Q6_V_hi_W(s3)));

  FA2_HVX_Vector_x4 x4 = {.v = {r0, r1, r2, r3}};
  return fa2_reduce_sum_f32x4(x4);
}

// 32 K rows × 1 Q row, all 32 dots packed into one sf vector. Internally
// loops 8× rx4. Adapted from hvx_dot_f16_f16_aa_rx32.
static HVX_INLINE_ALWAYS HVX_Vector fa2_dot_hf_rx32(const __fp16* restrict q,
                                                    const uint8_t* restrict k_base,
                                                    size_t stride_k, size_t D, float scale) {
  size_t nvec = D / FA2_VLEN_FP16;          // hf vectors per K row
  size_t stride_k_4 = stride_k * 4;

  HVX_Vector sums = Q6_V_vzero();
  for (uint32_t j = 0; j < FA2_VLEN_FP32; j += 4) {
    HVX_Vector x4 = fa2_dot_hf_rx4(q, k_base, stride_k, nvec);
    HVX_VectorPred pred = Q6_Q_vsetq_R(j * sizeof(float));
    sums = Q6_V_vmux_QVV(pred, sums, x4);
    k_base += stride_k_4;
  }

  sums = Q6_Vqf32_vmpy_VsfVsf(fa2_splat_f32(scale), sums);
  return Q6_Vsf_equals_Vqf32(sums);
}

// VKQ32 += V0 (hf) * s0 + V1 (hf) * s1, both fp16. The mantissa shuffle
// `Q6_Vh_vshuff_Vh` reorders hf lanes into the SDK's interleaved convention
// expected by the qf32 mpyacc — matches hvx_mad_f32_f16_aa_rx2.
static HVX_INLINE_ALWAYS void fa2_mad_f32_hf_rx2(float* restrict y, const __fp16* restrict x0,
                                                 const __fp16* restrict x1, const __fp16* restrict s0,
                                                 const __fp16* restrict s1, uint32_t D) {
  const HVX_Vector* restrict vx0 = (const HVX_Vector*)x0;
  const HVX_Vector* restrict vx1 = (const HVX_Vector*)x1;
  HVX_VectorPair* restrict vy_p = (HVX_VectorPair*)y;

  HVX_Vector S0 = fa2_splat_f16_p(s0);
  HVX_Vector S1 = fa2_splat_f16_p(s1);

  uint32_t nvec = D / FA2_VLEN_FP16;
#pragma unroll(2)
  for (uint32_t i = 0; i < nvec; ++i) {
    vy_p[i] = fa2_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    vy_p[i] = fa2_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
  }
}

// BQ Q rows × one head — shared K/V loads for one (head, Q-tile) call.
//
// Each Q row q in the tile maintains its own (M[q], S[q], VKQ32[q]) online
// softmax state. K and V are read once per K-row position and the inner
// dot/mad calls reuse them across all BQ Q rows.
//
// Causal masking is per-Q-row: kv_lims[q] is the exclusive upper bound on
// K indices for Q row q. For sub-blocks fully past kv_lims[q], scores are
// set to -INF (P → 0, contributes nothing). For sub-blocks straddling the
// boundary, a vsetq predicate masks the trailing lanes.
static void fa2_q_tile_hf(const __fp16* restrict q_tile, const __fp16* restrict k_base,
                          const __fp16* restrict v_base, uint32_t q_stride_elems, uint32_t kv_stride_elems,
                          __fp16* restrict o_tile, const uint32_t* kv_lims, uint32_t kv_lim_max, uint32_t BQ,
                          uint32_t D, float scale) {
  // Per-Q-row running output (fp32). FA2_VKQ32_D_MAX × FA2_BQ_TILE = 8 KB
  // for D=128, BQ=4 — well under the HTP stack budget.
  float __attribute__((aligned(128))) VKQ32[FA2_BQ_TILE * FA2_VKQ32_D_MAX];
  HVX_Vector M_vec[FA2_BQ_TILE];
  HVX_Vector S_vec[FA2_BQ_TILE];

  // Sub-block scores: 2 sub-blocks (FA2_BC/32) × BQ Q rows.
  HVX_Vector sb_scores[FA2_BQ_TILE][FA2_BC / FA2_VLEN_FP32];

  const HVX_Vector neg_inf = fa2_splat_f32(-INFINITY);

  for (uint32_t q = 0; q < BQ; ++q) {
    fa2_zero_f32(VKQ32 + q * D, D);
    M_vec[q] = neg_inf;
    S_vec[q] = fa2_splat_f32(0.0f);
  }

  const size_t k_stride_bytes = kv_stride_elems * sizeof(__fp16);

  for (uint32_t j0 = 0; j0 < kv_lim_max; j0 += FA2_BC) {
    uint32_t jN = j0 + FA2_BC;
    if (jN > kv_lim_max) jN = kv_lim_max;

    if (jN < kv_lim_max) {
      uint32_t fetch_rows = (jN + FA2_BC <= kv_lim_max) ? FA2_BC : (kv_lim_max - jN);
      uint32_t bytes_per_row = D * sizeof(__fp16);
      l2fetch(k_base + (size_t)jN * kv_stride_elems, bytes_per_row, bytes_per_row, fetch_rows, 0);
      l2fetch(v_base + (size_t)jN * kv_stride_elems, bytes_per_row, bytes_per_row, fetch_rows, 0);
    }

    // 1) Score sub-blocks. The K/V data loaded for this tile is reused
    // across all BQ Q rows — that's the whole point of Q tiling.
    HVX_Vector v_max[FA2_BQ_TILE];
    for (uint32_t q = 0; q < BQ; ++q) v_max[q] = neg_inf;

    uint32_t ic = 0;
    for (uint32_t iv = 0; ic + FA2_VLEN_FP32 <= (jN - j0); ic += FA2_VLEN_FP32, ++iv) {
      const uint8_t* k_ptr = (const uint8_t*)(k_base + (size_t)(j0 + ic) * kv_stride_elems);
      const uint32_t k_first = j0 + ic;

      for (uint32_t q = 0; q < BQ; ++q) {
        const __fp16* qrow = q_tile + (size_t)q * q_stride_elems;
        HVX_Vector scores = fa2_dot_hf_rx32(qrow, k_ptr, k_stride_bytes, D, scale);

        // Causal mask: if this sub-block extends past kv_lims[q], blank the
        // trailing lanes with -INF. Whole sub-block past the limit ⇒ all -INF.
        if (k_first + FA2_VLEN_FP32 > kv_lims[q]) {
          if (k_first >= kv_lims[q]) {
            scores = neg_inf;
          } else {
            uint32_t valid = kv_lims[q] - k_first;
            HVX_VectorPred p = Q6_Q_vsetq_R(valid * (uint32_t)sizeof(float));
            scores = Q6_V_vmux_QVV(p, scores, neg_inf);
          }
        }

        sb_scores[q][iv] = scores;
        v_max[q] = fa2_reduce_max2_f32(scores, v_max[q]);
      }
    }

    // 2) Online softmax + AV per Q row. Each Q row has its own M, S and
    // VKQ32; rescaling is independent.
    for (uint32_t q = 0; q < BQ; ++q) {
      // Skip rows whose entire tile is past their causal limit — v_max is
      // -INF, no update is needed and exp(score - M) would underflow anyway.
      if (j0 >= kv_lims[q]) continue;

      HVX_Vector M_new = Q6_Vsf_vmax_VsfVsf(v_max[q], M_vec[q]);
      HVX_Vector diff = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(M_vec[q], M_new));
      HVX_Vector ms = fa2_exp_f32(diff);
      M_vec[q] = M_new;

      fa2_scale_vec_f32(VKQ32 + q * D, VKQ32 + q * D, D, ms);

      HVX_Vector p_sum = fa2_splat_f32(0.0f);
      for (uint32_t ic2 = 0, iv = 0; ic2 + FA2_VLEN_FP32 <= (jN - j0); ic2 += FA2_VLEN_FP32, ++iv) {
        HVX_Vector s_shifted = Q6_Vqf32_vsub_VsfVsf(sb_scores[q][iv], M_vec[q]);
        HVX_Vector P = fa2_exp_f32(Q6_Vsf_equals_Vqf32(s_shifted));
        p_sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p_sum, P));

        __fp16 __attribute__((aligned(128))) p_arr[FA2_VLEN_FP16];
        *(HVX_Vector*)p_arr = fa2_f32_to_f16(P, fa2_splat_f32(0));

        for (uint32_t k = 0; k < FA2_VLEN_FP32; k += 2) {
          const __fp16* v0 = v_base + (size_t)(j0 + ic2 + k) * kv_stride_elems;
          const __fp16* v1 = v_base + (size_t)(j0 + ic2 + k + 1) * kv_stride_elems;
          fa2_mad_f32_hf_rx2(VKQ32 + q * D, v0, v1, &p_arr[k], &p_arr[k + 1], D);
        }
      }

      p_sum = fa2_reduce_sum_f32(p_sum);
      S_vec[q] = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(S_vec[q], ms)), p_sum));

      // Scalar tail for any leftover K tokens, masked per kv_lims[q].
      if (ic < (jN - j0)) {
        float M_s = fa2_get_f32(M_vec[q]);
        float S_s = fa2_get_f32(S_vec[q]);
        const __fp16* qrow = q_tile + (size_t)q * q_stride_elems;
        for (uint32_t j = j0 + ic; j < jN; ++j) {
          if (j >= kv_lims[q]) break;
          const __fp16* kptr = k_base + (size_t)j * kv_stride_elems;
          float dot = 0.0f;
          for (uint32_t d = 0; d < D; ++d) dot += (float)qrow[d] * (float)kptr[d];
          float s_val = dot * scale;
          float Mold = M_s;
          __fp16 vs = 1.0f;
          if (s_val > M_s) {
            M_s = s_val;
            float msf = expf(Mold - M_s);
            for (uint32_t d = 0; d < D; ++d) VKQ32[q * D + d] *= msf;
            S_s = S_s * msf + (float)vs;
          } else {
            vs = (__fp16)expf(s_val - M_s);
            S_s += (float)vs;
          }
          const __fp16* vptr = v_base + (size_t)j * kv_stride_elems;
          for (uint32_t d = 0; d < D; ++d) VKQ32[q * D + d] += (float)vs * (float)vptr[d];
        }
        M_vec[q] = fa2_splat_f32(M_s);
        S_vec[q] = fa2_splat_f32(S_s);
      }
    }
  }

  // Final normalize and fp32 → fp16 cast per Q row.
  for (uint32_t q = 0; q < BQ; ++q) {
    float S_s = fa2_get_f32(S_vec[q]);
    float inv_S = (S_s == 0.0f) ? 0.0f : (1.0f / S_s);
    HVX_Vector inv_vec = fa2_splat_f32(inv_S);
    fa2_scale_vec_f32(VKQ32 + q * D, VKQ32 + q * D, D, inv_vec);

    HVX_Vector* vo = (HVX_Vector*)(o_tile + (size_t)q * q_stride_elems);
    const HVX_Vector* vacc = (const HVX_Vector*)(VKQ32 + q * D);
    uint32_t n_hf = D / FA2_VLEN_FP16;
    for (uint32_t i = 0; i < n_hf; ++i) {
      vo[i] = fa2_f32_to_f16(vacc[2 * i + 0], vacc[2 * i + 1]);
    }
  }
}

static void fa2_flashattention_hf(__fp16* restrict Obuf, const __fp16* restrict Qbuf, const __fp16* restrict Kbuf,
                                  const __fp16* restrict Vbuf, uint32_t Bq, uint32_t Sq, uint32_t Sk, uint32_t Hq,
                                  uint32_t Hkv, uint32_t Dq, uint32_t q_stride, uint32_t kv_stride, uint32_t group,
                                  float scale, uint32_t is_causal, uint32_t global_sq_start, uint32_t original_sq) {
  for (uint32_t b = 0; b < Bq; ++b) {
    const __fp16* Qb = Qbuf + (size_t)b * Sq * q_stride;
    const __fp16* Kb = Kbuf + (size_t)b * Sk * kv_stride;
    const __fp16* Vb = Vbuf + (size_t)b * Sk * kv_stride;
    __fp16* Ob = Obuf + (size_t)b * Sq * q_stride;
    for (uint32_t s = 0; s < Sq; s += FA2_BQ_TILE) {
      uint32_t bq = (s + FA2_BQ_TILE <= Sq) ? FA2_BQ_TILE : (Sq - s);

      uint32_t kv_lims[FA2_BQ_TILE];
      uint32_t kv_lim_max = 0;
      for (uint32_t q = 0; q < bq; ++q) {
        uint32_t lim = Sk;
        if (is_causal && original_sq > 1) lim = (Sk - original_sq) + global_sq_start + s + q + 1;
        kv_lims[q] = lim;
        if (lim > kv_lim_max) kv_lim_max = lim;
      }

      for (uint32_t hq = 0; hq < Hq; ++hq) {
        uint32_t hkv = hq / group;
        const __fp16* qtile = Qb + (size_t)s * q_stride + hq * Dq;
        const __fp16* krow = Kb + hkv * Dq;
        const __fp16* vrow = Vb + hkv * Dq;
        __fp16* otile = Ob + (size_t)s * q_stride + hq * Dq;
        fa2_q_tile_hf(qtile, krow, vrow, q_stride, kv_stride, otile, kv_lims, kv_lim_max, bq, Dq, scale);
      }
    }
  }
}

#endif  // !REFERENCE_OP

// Reference / aarch64 path — fp32-accumulator scalar reference. Only used by
// QNN's graph-builder validation (not on the hot path).
static inline void fa2_one_row_ref_hf(const __fp16* q, const __fp16* k_base, const __fp16* v_base,
                                      uint32_t kv_stride_elems, __fp16* o, uint32_t S_kv_lim, uint32_t D, float scale) {
  float o_tmp[2048];
  for (uint32_t d = 0; d < D; ++d) o_tmp[d] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j0 = 0; j0 < S_kv_lim; j0 += FA2_BC) {
    uint32_t jN = j0 + FA2_BC;
    if (jN > S_kv_lim) jN = S_kv_lim;

    float s_buf[FA2_BC];
    float m_tile = -INFINITY;
    for (uint32_t j = j0; j < jN; ++j) {
      const __fp16* kptr = k_base + (size_t)j * kv_stride_elems;
      float dot = 0.0f;
      for (uint32_t d = 0; d < D; ++d) dot += (float)q[d] * (float)kptr[d];
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
      const __fp16* vptr = v_base + (size_t)j * kv_stride_elems;
      float p = s_buf[j - j0];
      for (uint32_t d = 0; d < D; ++d) o_tmp[d] += p * (float)vptr[d];
    }
    l = alpha * l + l_tile;
    m = m_new;
  }
  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (uint32_t d = 0; d < D; ++d) o[d] = (__fp16)(o_tmp[d] * inv_l);
}

template<typename TensorType>
GraphStatus flashattentionv2Impl(TensorType& out_0, const TensorType& q_in, const TensorType& k_in,
                                 const TensorType& v_in, const PlainFloatTensor& softmax_scale, const Tensor& causal,
                                 hnnx::op_slice_spec slice_spec) {
  out_0.set_dims(q_in);

  auto [Bq, Sq, Hq, Dq] = q_in.dims();
  auto [Bk, Sk, Hkv, Dk] = k_in.dims();
  auto [Bv, Sv, Hkv2, Dv] = v_in.dims();

  if (Bq != Bk || Bq != Bv) return GraphStatus::ErrorDimensions;
  if (Sk != Sv) return GraphStatus::ErrorDimensions;
  if (Hkv != Hkv2) return GraphStatus::ErrorDimensions;
  if (Dq != Dk || Dq != Dv) return GraphStatus::ErrorDimensions;
  if (Hq % Hkv != 0) return GraphStatus::ErrorDimensions;
  // hf vectors hold 64 lanes; require D % 64 == 0 for the rx32 dot path.
  if ((Dq & 63) != 0) return GraphStatus::ErrorDimensions;

  const uint32_t slice_count = slice_spec.num_slices();
  const uint32_t slice_idx = slice_spec.slice_idx();
  // AUTOSPLIT(tile=FA2_BQ_TILE) gives each kernel call up to BQ_TILE Q rows.
  // The full prefill height is slice_count * BQ_TILE; the slice's first row
  // is at slice_idx * BQ_TILE in that original space. Assumes Sq divides
  // evenly by FA2_BQ_TILE for sliced ops (the AUTOSPLIT case); the test
  // shapes (Sq=128 with BQ=4) satisfy this.
  const uint32_t original_sq = (slice_count > 1) ? slice_count * FA2_BQ_TILE : (uint32_t)Sq;
  const uint32_t global_sq_start = (slice_count > 1) ? slice_idx * FA2_BQ_TILE : 0;
  if (Sk < original_sq) return GraphStatus::ErrorDimensions;

  const uint32_t group = Hq / Hkv;
  const uint32_t kv_stride = Hkv * Dq;
  const uint32_t q_stride = Hq * Dq;
  const float scale = softmax_scale(0, 0, 0, 0);
  const uint32_t is_causal = (uint32_t)causal(0, 0, 0, 0);

  if (q_in.get_dtype() != DType::Float16 || out_0.get_dtype() != DType::Float16 ||
      k_in.get_dtype() != DType::Float16 || v_in.get_dtype() != DType::Float16) {
    return GraphStatus::ErrorUnsupported;
  }

  const __fp16* Qbuf = (const __fp16*)q_in.raw_data_const();
  const __fp16* Kbuf = (const __fp16*)k_in.raw_data_const();
  const __fp16* Vbuf = (const __fp16*)v_in.raw_data_const();
  __fp16* Obuf = (__fp16*)out_0.raw_data();

#ifndef REFERENCE_OP
  fa2_flashattention_hf(Obuf, Qbuf, Kbuf, Vbuf, (uint32_t)Bq, (uint32_t)Sq, (uint32_t)Sk, (uint32_t)Hq, (uint32_t)Hkv,
                        (uint32_t)Dq, q_stride, kv_stride, group, scale, is_causal, global_sq_start, original_sq);
#else
  for (uint32_t b = 0; b < (uint32_t)Bq; ++b) {
    const __fp16* Qb = Qbuf + (size_t)b * Sq * q_stride;
    const __fp16* Kb = Kbuf + (size_t)b * Sk * kv_stride;
    const __fp16* Vb = Vbuf + (size_t)b * Sk * kv_stride;
    __fp16* Ob = Obuf + (size_t)b * Sq * q_stride;
    for (uint32_t s = 0; s < (uint32_t)Sq; ++s) {
      uint32_t kv_lim = (uint32_t)Sk;
      if (is_causal && original_sq > 1) kv_lim = (Sk - original_sq) + global_sq_start + s + 1;
      for (uint32_t hq = 0; hq < (uint32_t)Hq; ++hq) {
        uint32_t hkv = hq / group;
        const __fp16* qrow = Qb + (size_t)s * q_stride + hq * Dq;
        const __fp16* krow = Kb + hkv * Dq;
        const __fp16* vrow = Vb + hkv * Dq;
        __fp16* orow = Ob + (size_t)s * q_stride + hq * Dq;
        fa2_one_row_ref_hf(qrow, krow, vrow, kv_stride, orow, kv_lim, Dq, scale);
      }
    }
  }
#endif

  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_FlashAttentionV2);
