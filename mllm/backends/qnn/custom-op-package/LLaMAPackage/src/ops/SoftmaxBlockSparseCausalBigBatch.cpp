//==============================================================================
// SoftmaxBlockSparseCausalBigBatch — big-batch fused
// Mul + (causal mask) + Softmax replacement for the big-batch r3 sparse
// attention pipeline.
//
// QK shape: [B=1, H=Hq*num_qb, BQ, top_k*bk]   (rank-3 backfilled)
//
// Per-row qb index is computed from POSITION:
//   global_h     = slice_base + h_local
//   qb_idx_for_h = global_h % num_qb
//   (data layout is Hq-outer, qb-inner — see runCausalBigBatch())
//
// `slice_base` is a uint32 scalar input. The user sets it to 0 (graph-time
// STATIC). The AUTOSPLIT optimization rule REPLACES it per-slice with
// `gen_ConstScalar_i32(SPLIT_START("I"))` so each split-op gets a Const
// carrying its own global start row index. That dissolves the previous
// design's need for a [big_batch, 1, 1] q_block_idx vector input — which
// turned out to block AUTOSPLIT firing reliably (TYPICAL_SLICE on a
// degenerate [1, N, 1, 1] tensor is non-standard).
//
// `num_qb` is a regular uint32 scalar param.
//
// Layout (NHWC after rank backfill):
//   in[0] QK         : [B, big_batch, BQ, top_k*bk]   (fp16 or fp32)
//   in[1] slice_base : uint32 scalar (graph-time 0; replaced per-slice
//                                    by the AUTOSPLIT rule)
//   out[0] P         : same shape and dtype as in[0]
//
// Parameters:
//   softmax_scale : float scalar
//   bk            : uint32 scalar (k-block size; top_k = K / bk)
//   num_qb        : uint32 scalar (number of q-blocks; modulo divisor)
//
// Helpers below are duplicated from SoftmaxBlockSparseCausal.cpp; when
// improving kernel-level details (expf accuracy, pair processing), update
// both files.
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

// #define SBSC_BB_NO_EXP

BEGIN_PKG_OP_DEFINITION(PKG_SoftmaxBlockSparseCausalBigBatch);

template<typename TensorType, typename TensorType1>
GraphStatus softmaxblocksparsecausalbigbatchImpl(TensorType& out_0,
                                                 const TensorType& in_0,
                                                 const TensorType1& slice_base_t,
                                                 const PlainFloatTensor& softmax_scale,
                                                 const Tensor& bk,
                                                 const Tensor& num_qb);

static float softmaxblocksparsecausalbigbatchCostFunc(const Op* op);

DEF_PACKAGE_OP((softmaxblocksparsecausalbigbatchImpl<Tensor, Tensor>),
               "SoftmaxBlockSparseCausalBigBatch")

DEF_PACKAGE_PARAM_ORDER("SoftmaxBlockSparseCausalBigBatch",
                        "softmax_scale", true, nullptr,
                        "bk",            true, nullptr,
                        "num_qb",        true, nullptr)

// AUTOSPLIT chunk=16 on dim 1 (H = big_batch). For each slice the rule
// REPLACES the user-passed slice_base (a Const scalar = 0) with a synthesized
// Const scalar whose value is SPLIT_START("I") — the slice's global row offset.
//
// chunk=16 is universally safe for the modulo: any num_qb in {4,8,16,32,64}
// either divides 16 or is divisible by 16, so qb_idx = (slice_base + h_local)
// % num_qb cycles correctly within every slice. (Smaller values would also
// work; 16 matches "one Hq's worth" of qb-rows per slice for our standard
// Hq=16 attention shape.)
//
// The single TYPICAL_SLICE on QK (a normal 4D tensor) is the standard pattern
// QNN's optimizer handles best — the prior design's TYPICAL_SLICE on a
// degenerate [1, N, 1, 1] q_block_idx vector appeared to suppress firing.
// Schematic dump confirmed AUTOSPLIT fires with chunk=16, producing 32
// split-ops at Sq=1024. But each split paid full per-op fixed overhead
// (~500 µs each × 32 = 16 ms ≈ measured wall-clock), so chunk=16 was
// effectively running 32 per-qb dispatches in serial.
//
// Goal: pick chunk so num_splits ≈ num_HVX_threads = 6 — each split is
// fatter, so compute amortises the per-op overhead, and the 6 splits
// can fan out on the 6 HVX threads in one round.
//
//   Sq=128  → big_batch=64  → chunk=11 gives ceil(64/11)=6 splits
//   Sq=256  → big_batch=128 → chunk=22 gives 6 splits
//   Sq=512  → big_batch=256 → chunk=43 gives 6 splits
//   Sq=1024 → big_batch=512 → chunk=86 gives 6 splits
//   Sq=2048 → big_batch=1024 → chunk=171 gives 6 splits
//
// Single chunk constant has to compromise across Sq values. chunk=85
// optimises the larger Sq cases (Sq>=512) where per-dispatch overhead
// dominates, at the cost of small Sq cases having only 1-2 splits.
// (Small Sq cases were already fast enough not to need parallelism.)
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("SoftmaxBlockSparseCausalBigBatch", "QK", "slice_base",
       "softmax_scale", "bk", "num_qb"),
    GT(DIM_HEIGHT("QK"), 16),
    AUTOSPLIT(1, "I", 16,
              Op("SoftmaxBlockSparseCausalBigBatch",
                 TYPICAL_SLICE("QK", "I"),
                 gen_ConstScalar_i32(SPLIT_START("I")),
                 "softmax_scale", "bk", "num_qb")))

// Layout / placement directive — tell QNN our HVX-only kernel wants Flat
// layout (not Crouton, which is HMX-friendly), and try to keep the score
// tensor in TCM scratchpad. If QK fits in TCM (~8 MB on V79), this avoids
// main-memory bandwidth on the per-row stream. Per the implementing-ops
// doc, omitting this means default = MainMemory + whatever layout the
// upstream op left, which forces conversion at every dispatch.
// DEF_TENSOR_PROPERTIES(
//     Op("SoftmaxBlockSparseCausalBigBatch", "QK", "slice_base"),
//     Flat("*", "QK"),
//     MainMemory("slice_base"),
//     Tcm("QK"))

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"

static inline uint32_t sbsc_bb_float_to_bits(float x) {
  union { float f; uint32_t i; } u = {.f = x};
  return u.i;
}
static inline int32_t sbsc_bb_float_to_fp16x2(float x) {
  union { int32_t i; __fp16 f[2]; } u = {.f = {(__fp16)x, (__fp16)x}};
  return u.i;
}

static inline __attribute__((always_inline)) HVX_Vector sbsc_bb_hvx_expf_sf(HVX_Vector x) {
  HVX_Vector log2ef       = Q6_V_vsplat_R(sbsc_bb_float_to_bits(1.44269504088896341f));
  HVX_Vector half         = Q6_V_vsplat_R(sbsc_bb_float_to_bits(0.5f));
  HVX_Vector C1           = Q6_V_vsplat_R(sbsc_bb_float_to_bits(0.693359375f));
  HVX_Vector C2           = Q6_V_vsplat_R(sbsc_bb_float_to_bits(-2.12194440e-4f));
  HVX_Vector neg_lim      = Q6_V_vsplat_R(sbsc_bb_float_to_bits(-87.336544f));
  HVX_Vector pos_lim      = Q6_V_vsplat_R(sbsc_bb_float_to_bits(88.722839f));
  HVX_Vector one          = Q6_V_vsplat_R(sbsc_bb_float_to_bits(1.0f));
  HVX_Vector zero         = Q6_V_vzero();
  HVX_Vector vsplat_127   = Q6_V_vsplat_R(127);

  HVX_Vector c0 = Q6_V_vsplat_R(sbsc_bb_float_to_bits(1.9875691500e-4f));
  HVX_Vector c1 = Q6_V_vsplat_R(sbsc_bb_float_to_bits(1.3981999507e-3f));
  HVX_Vector c2 = Q6_V_vsplat_R(sbsc_bb_float_to_bits(8.3334519073e-3f));
  HVX_Vector c3 = Q6_V_vsplat_R(sbsc_bb_float_to_bits(4.1665795894e-2f));
  HVX_Vector c4 = Q6_V_vsplat_R(sbsc_bb_float_to_bits(1.6666665459e-1f));
  HVX_Vector c5 = Q6_V_vsplat_R(sbsc_bb_float_to_bits(5.0000001201e-1f));

  HVX_VectorPred too_low = Q6_Q_vcmp_gt_VsfVsf(neg_lim, x);
  x = Q6_Vsf_vmin_VsfVsf(x, pos_lim);

  HVX_Vector fx_sf = Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, log2ef)), half));

  HVX_Vector n_w  = Q6_Vw_equals_Vsf(fx_sf);
  HVX_Vector n_sf = Q6_Vsf_equals_Vw(n_w);
  HVX_VectorPred over = Q6_Q_vcmp_gt_VsfVsf(n_sf, fx_sf);
  HVX_Vector adj = Q6_V_vmux_QVV(over, Q6_V_vsplat_R(1), zero);
  n_w = Q6_Vw_vsub_VwVw(n_w, adj);
  HVX_Vector fn = Q6_Vsf_equals_Vw(n_w);

  HVX_Vector tmp1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(fn, C1));
  HVX_Vector r    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, tmp1));
  HVX_Vector tmp2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(fn, C2));
  r               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(r, tmp2));

  HVX_Vector r2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r, r));

  HVX_Vector p = c0;
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c4, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c5, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));

  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(r, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r2))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p, one));

  HVX_Vector exp_bits = Q6_Vw_vadd_VwVw(n_w, vsplat_127);
  HVX_Vector pow2     = Q6_Vw_vasl_VwR(exp_bits, 23);

  HVX_Vector result = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, pow2));
  result = Q6_V_vmux_QVV(too_low, zero, result);
  return result;
}

static inline float sbsc_bb_hvx_sf_hmax(HVX_Vector v_sf) {
  union { HVX_Vector v; float f[32]; } u;
  u.v = v_sf;
  float m = -INFINITY;
  for (int i = 0; i < 32; ++i) if (u.f[i] > m) m = u.f[i];
  return m;
}
static inline float sbsc_bb_hvx_sf_hsum(HVX_Vector v_sf) {
  union { HVX_Vector v; float f[32]; } u;
  u.v = v_sf;
  float s = 0.0f;
  for (int i = 0; i < 32; ++i) s += u.f[i];
  return s;
}

static inline __attribute__((always_inline)) HVX_VectorPair
sbsc_bb_hvx_load64_widen_pair(const __fp16* s) {
  HVX_Vector splat_1 = Q6_V_vsplat_R(sbsc_bb_float_to_fp16x2(1.0f));
  HVX_Vector hf = *(const HVX_UVector*)s;
  HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(hf, splat_1);
  HVX_Vector lo_il = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
  HVX_Vector hi_il = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
  return Q6_W_vshuff_VVR(hi_il, lo_il, -4);
}

static inline __attribute__((always_inline)) void
sbsc_bb_hvx_store64_narrow_pair_qf(__fp16* p, HVX_Vector qf_a_src, HVX_Vector qf_b_src) {
  HVX_VectorPair pair = Q6_W_vdeal_VVR(qf_b_src, qf_a_src, -4);
  *(HVX_UVector*)p = Q6_Vhf_equals_Wqf32(pair);
}

static inline __attribute__((always_inline)) HVX_Vector sbsc_bb_mul_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline __attribute__((always_inline)) HVX_Vector sbsc_bb_add_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

// One row HVX softmax — identical to sbsc_row_hvx_f16 in the per-qb op.
// q_block_idx is sourced per-row by the caller (caller does the modulo).
static inline void sbsc_bb_row_hvx_f16(const __fp16* s, __fp16* p, int W, int bk,
                                       int q_block_idx, int q_row, float scale) {
  const int top_k         = W / bk;
  const int diag_slot     = top_k - 1;
  const int n_real_hist   = (q_block_idx < diag_slot) ? q_block_idx : diag_slot;
  (void)top_k;

  HVX_Vector scale_v   = Q6_V_vsplat_R(sbsc_bb_float_to_bits(scale));
  HVX_Vector neg_inf_v = Q6_V_vsplat_R(sbsc_bb_float_to_bits(-INFINITY));
  HVX_Vector neg100_v  = Q6_V_vsplat_R(sbsc_bb_float_to_bits(-100.0f));
  HVX_Vector zero_v    = Q6_V_vzero();

  HVX_Vector triangle_mask;
  if (q_row + 1 >= bk) {
    triangle_mask = zero_v;
  } else {
    HVX_VectorPred valid_pred = Q6_Q_vsetq_R((q_row + 1) * 4);
    triangle_mask = Q6_V_vmux_QVV(valid_pred, zero_v, neg100_v);
  }

  enum PairKind : uint8_t { PK_REAL_REAL, PK_HAS_MASK, PK_SKIP };
  const int n_pairs = W / 64;
  HVX_Vector mask_a_arr[8];
  HVX_Vector mask_b_arr[8];
  PairKind kind[8];
  for (int pi = 0; pi < n_pairs; ++pi) {
    int slot_a = pi * 2;
    int slot_b = pi * 2 + 1;
    bool a_real = slot_a < n_real_hist;
    bool b_real = slot_b < n_real_hist;
    bool b_diag = slot_b == diag_slot;
    if (!a_real && !b_real && !b_diag) {
      kind[pi] = PK_SKIP;
    } else if (a_real && b_real) {
      kind[pi] = PK_REAL_REAL;
    } else {
      kind[pi] = PK_HAS_MASK;
      mask_a_arr[pi] = a_real ? zero_v : neg100_v;
      if (b_diag)        mask_b_arr[pi] = triangle_mask;
      else if (b_real)   mask_b_arr[pi] = zero_v;
      else               mask_b_arr[pi] = neg100_v;
    }
  }

  HVX_Vector sf_a[8], sf_b[8];
  HVX_Vector vmax = neg_inf_v;
  for (int pi = 0; pi < n_pairs; ++pi) {
    if (kind[pi] == PK_SKIP) {
      sf_a[pi] = zero_v;
      sf_b[pi] = zero_v;
      continue;
    }
    HVX_VectorPair w = sbsc_bb_hvx_load64_widen_pair(s + pi * 64);
    sf_a[pi] = sbsc_bb_mul_sf(Q6_V_lo_W(w), scale_v);
    sf_b[pi] = sbsc_bb_mul_sf(Q6_V_hi_W(w), scale_v);
    if (kind[pi] == PK_HAS_MASK) {
      sf_a[pi] = sbsc_bb_add_sf(sf_a[pi], mask_a_arr[pi]);
      sf_b[pi] = sbsc_bb_add_sf(sf_b[pi], mask_b_arr[pi]);
    }
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, sf_a[pi]);
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, sf_b[pi]);
  }
  const float row_max = sbsc_bb_hvx_sf_hmax(vmax);
  HVX_Vector neg_max_v = Q6_V_vsplat_R(sbsc_bb_float_to_bits(-row_max));

  HVX_Vector vsum = zero_v;
  for (int pi = 0; pi < n_pairs; ++pi) {
    if (kind[pi] == PK_SKIP) continue;
    sf_a[pi] = sbsc_bb_add_sf(sf_a[pi], neg_max_v);
    sf_b[pi] = sbsc_bb_add_sf(sf_b[pi], neg_max_v);
#ifdef SBSC_BB_NO_EXP
#else
    sf_a[pi] = sbsc_bb_hvx_expf_sf(sf_a[pi]);
    sf_b[pi] = sbsc_bb_hvx_expf_sf(sf_b[pi]);
#endif
    vsum = sbsc_bb_add_sf(vsum, sf_a[pi]);
    vsum = sbsc_bb_add_sf(vsum, sf_b[pi]);
  }

  const float row_sum = sbsc_bb_hvx_sf_hsum(vsum);
  const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
  HVX_Vector inv_v = Q6_V_vsplat_R(sbsc_bb_float_to_bits(inv));

  for (int pi = 0; pi < n_pairs; ++pi) {
    HVX_Vector q_a = Q6_Vqf32_vmpy_VsfVsf(sf_a[pi], inv_v);
    HVX_Vector q_b = Q6_Vqf32_vmpy_VsfVsf(sf_b[pi], inv_v);
    sbsc_bb_hvx_store64_narrow_pair_qf(p + pi * 64, q_a, q_b);
  }
}

#endif  // !REFERENCE_OP

template<typename T_in, typename T_out>
static inline void sbsc_bb_one_row_scalar(const T_in* __restrict__ s, T_out* __restrict__ p,
                                          int W, int bk, int q_block_idx, int q_row, float scale) {
  const int top_k = W / bk;
  const int diag_slot = top_k - 1;
  const int n_real_hist = (q_block_idx < top_k - 1) ? q_block_idx : (top_k - 1);

  float row_max = -INFINITY;
  for (int c = 0; c < W; ++c) {
    int slot = c / bk;
    int col_in_slot = c - slot * bk;
    bool valid;
    if (slot < n_real_hist)      valid = true;
    else if (slot < diag_slot)   valid = false;
    else                         valid = (col_in_slot <= q_row);
    if (!valid) continue;
    float v = (float)s[c] * scale;
    if (v > row_max) row_max = v;
  }
  if (row_max == -INFINITY) {
    for (int c = 0; c < W; ++c) p[c] = (T_out)0;
    return;
  }
  float denom = 0.0f;
  for (int c = 0; c < W; ++c) {
    int slot = c / bk;
    int col_in_slot = c - slot * bk;
    bool valid;
    if (slot < n_real_hist)      valid = true;
    else if (slot < diag_slot)   valid = false;
    else                         valid = (col_in_slot <= q_row);
    if (!valid) { p[c] = (T_out)0; continue; }
    float v = (float)s[c] * scale;
    float e = expf(v - row_max);
    denom += e;
    p[c] = (T_out)e;
  }
  float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int c = 0; c < W; ++c) p[c] = (T_out)((float)p[c] * inv);
}

template<typename TensorType, typename TensorType1>
GraphStatus softmaxblocksparsecausalbigbatchImpl(TensorType& out_0,
                                                 const TensorType& in_0,
                                                 const TensorType1& slice_base_t,
                                                 const PlainFloatTensor& softmax_scale,
                                                 const Tensor& bk_t,
                                                 const Tensor& num_qb_t) {
  out_0.set_dims(in_0);
  auto [B, H, Wq, K] = in_0.dims();

  const float scale = softmax_scale(0, 0, 0, 0);
  const uint32_t bk = (uint32_t)bk_t(0, 0, 0, 0);
  const uint32_t num_qb = (uint32_t)num_qb_t(0, 0, 0, 0);
  const uint32_t slice_base = (uint32_t)slice_base_t(0, 0, 0, 0);

  if (bk == 0 || (K % bk) != 0) return GraphStatus::ErrorDimensions;
  if (num_qb == 0) return GraphStatus::ErrorDimensions;
  const int top_k = (int)(K / bk);
  if (top_k < 1) return GraphStatus::ErrorDimensions;

  if (in_0.get_dtype() == DType::Float16 && out_0.get_dtype() == DType::Float16) {
    const __fp16* sp = (const __fp16*)in_0.raw_data_const();
    __fp16* pp = (__fp16*)out_0.raw_data();
#ifndef REFERENCE_OP
    const bool hvx_ok = (bk == 32) && ((K % 32) == 0);
    for (int b = 0; b < (int)B; ++b) {
      for (int h = 0; h < (int)H; ++h) {
        // global_h = slice_base + h_local;  qb_idx = global_h % num_qb
        const int qb_idx = (int)((slice_base + (uint32_t)h) % num_qb);
        for (int q = 0; q < (int)Wq; ++q) {
          const __fp16* srow = sp + (((size_t)b * H + h) * Wq + q) * K;
          __fp16* prow = pp + (((size_t)b * H + h) * Wq + q) * K;
          if (hvx_ok) {
            sbsc_bb_row_hvx_f16(srow, prow, (int)K, (int)bk, qb_idx, q, scale);
          } else {
            sbsc_bb_one_row_scalar<__fp16, __fp16>(srow, prow, (int)K, (int)bk, qb_idx, q, scale);
          }
        }
      }
    }
#else
    for (int b = 0; b < (int)B; ++b) {
      for (int h = 0; h < (int)H; ++h) {
        const int qb_idx = (int)((slice_base + (uint32_t)h) % num_qb);
        for (int q = 0; q < (int)Wq; ++q) {
          const __fp16* srow = sp + (((size_t)b * H + h) * Wq + q) * K;
          __fp16* prow = pp + (((size_t)b * H + h) * Wq + q) * K;
          sbsc_bb_one_row_scalar<__fp16, __fp16>(srow, prow, (int)K, (int)bk, qb_idx, q, scale);
        }
      }
    }
#endif
    return GraphStatus::Success;
  }

  if (in_0.get_dtype() == DType::Float32 && out_0.get_dtype() == DType::Float32) {
    const float* sp = (const float*)in_0.raw_data_const();
    float* pp = (float*)out_0.raw_data();
    for (int b = 0; b < (int)B; ++b) {
      for (int h = 0; h < (int)H; ++h) {
        const int qb_idx = (int)((slice_base + (uint32_t)h) % num_qb);
        for (int q = 0; q < (int)Wq; ++q) {
          const float* srow = sp + (((size_t)b * H + h) * Wq + q) * K;
          float* prow = pp + (((size_t)b * H + h) * Wq + q) * K;
          sbsc_bb_one_row_scalar<float, float>(srow, prow, (int)K, (int)bk, qb_idx, q, scale);
        }
      }
    }
    return GraphStatus::Success;
  }

  return GraphStatus::ErrorUnsupported;
}

__attribute__((unused)) static float softmaxblocksparsecausalbigbatchCostFunc(const Op* op) {
  return 0.0f;
}

END_PKG_OP_DEFINITION(PKG_SoftmaxBlockSparseCausalBigBatch);
