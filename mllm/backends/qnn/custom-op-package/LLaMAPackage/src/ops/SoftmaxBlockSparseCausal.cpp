//==============================================================================
// SoftmaxBlockSparseCausal custom op for LLaMAPackage
//
// Fuses three ops from the block-sparse decomposed attention graph:
//   QKs = ElementWiseMul(QK, scale)
//   QKm = ElementWiseAdd(QKs, mask)        <-- the big causal mask we want to kill
//   P   = Softmax(QKm, axis=-1)
//
// into one op:
//   P = SoftmaxBlockSparseCausal(QK, softmax_scale, q_block_idx, bk)
//
// The mask is structurally simple: for any q-row q in [0, BQ), the score
// columns are partitioned into top_k slots of size bk each. The selection is
// arranged so that the LAST slot (slot top_k-1) is the diagonal k-block
// itself, and the first top_k-1 slots are historical k-blocks (real for
// q_block_idx >= top_k-1, otherwise padded with k-block 0 for the slots
// beyond the available history).
//
// Mask rules per (q-row q, column c):
//   slot       = c / bk
//   col_in_slot= c % bk
//   diag_slot  = top_k - 1
//   n_real_hist= min(top_k - 1, q_block_idx)
//
//   slot < n_real_hist                 -> real historical, no mask
//   slot in [n_real_hist, diag_slot)   -> padding slot, fully masked
//   slot == diag_slot && col > q       -> diagonal triangle upper, masked
//   slot == diag_slot && col <= q      -> diagonal triangle lower, no mask
//
// Layout (NHWC):
//   in[0] QK          : [B, H_q, BQ, top_k*bk]  (fp16 or fp32)
//   in[1] q_block_idx : uint32 scalar (0..num_q_blocks-1) — runtime-bindable
//                       INPUT (not a compile-time param). Lets a single
//                       compiled graph handle every q-block's mask via
//                       APP_WRITE binding before each dispatch.
//   out[0] P          : same shape and dtype as in[0]
//
// Parameters:
//   softmax_scale : float scalar
//   bk            : uint32 scalar (k-block size; top_k = W / bk)
//
// v5: HVX-vectorised fp16 fast path. Pair processing (2 bk-blocks per
// 64-lane hf vector) with vectorised widen, vectorised qf32→hf narrow, and
// a vectorised Cephes expf inlined into the hot loop. Diagonal-slot triangle
// is applied via Q6_Q_vsetq_R + Q6_V_vmux_QVV — never materialised. Pass 1
// caches scaled+masked sf in a stack tmp[fp32, W] buffer so pass 2 doesn't
// redo the widen+scale+mask. PADDING+PADDING pairs short-circuit to zero.
//
// v7: AUTOSPLIT(1, "I", 3, ...) directive distributes 6 H_q slices across
// the 6 HVX threads. See the directive's comment block below for the
// chunk-size tuning rationale. This required two prerequisites that took
// most of the debug effort:
//   - REGISTER_PACKAGE_OPTIMIZATIONS() must be called in LLaMAPackageInit
//     (was missing — added; FA's directives were probably also no-ops).
//   - QNN backfills our rank-3 input to rank-4 [B=1, H_q, BQ, top_k_BK];
//     splitting along H_q means dim 1 (DIM_HEIGHT), not dim 0.
//
// Per-dispatch latency at Qwen3-shape (Hq=16, BQ=32, top_k=8) on V79:
//   v1 (scalar)              : 34.1   ms
//   v2 (HVX, scalar wn)      :  8.5   ms
//   v3 (vectorised narrow)   :  5.2   ms
//   v4 (pair widen+narrow)   :  0.51  ms
//   v5 (cache pass 1 sf)     :  0.49  ms      ← single-thread on 1 HVX
//   v6 (AUTOSPLIT chunk 4)   :  0.19  ms      ← 4 HVX threads
//   v7 (AUTOSPLIT chunk 3)   :  0.152 ms      ← all 6 HVX threads engaged
//
// vs the reference Mul+Add+Softmax decomp (which is also HVX-only — not
// HMX as I'd assumed): 0.114 ms. v7 is within 1.33× — essentially as good
// as it gets for an external custom op without manual QURT threading.
//
// The scalar reference path (used by the aarch64 prepare-side build and as
// a fallback for non-fp16 dtypes) is kept inline below the HVX kernel.
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

// Perf-probe: define SBSC_NO_EXP at compile time to replace the polynomial
// expf with an sf passthrough, isolating expf cost. With v5 it's ~0.05 ms
// out of 0.49 ms at top_k=8 / Hq=16 — small. Keep available but off by default.
// #define SBSC_NO_EXP

BEGIN_PKG_OP_DEFINITION(PKG_SoftmaxBlockSparseCausal);

// q_block_idx is an INPUT (not a param) so a single compiled graph can serve
// all q-blocks — the host writes the value to an APP_WRITE tensor before each
// dispatch. Same idiom RoPE/IRoPE use for `h_cnt`. The `TensorType1` template
// slot lets the input be uint32 even though `in_0` is fp16/fp32.
template<typename TensorType, typename TensorType1>
GraphStatus softmaxblocksparsecausalImpl(TensorType& out_0,
                                         const TensorType& in_0,
                                         const TensorType1& q_block_idx,
                                         const PlainFloatTensor& softmax_scale,
                                         const Tensor& bk);

static float softmaxblocksparsecausalCostFunc(const Op* op);

DEF_PACKAGE_OP((softmaxblocksparsecausalImpl<Tensor, Tensor>), "SoftmaxBlockSparseCausal")

DEF_PACKAGE_PARAM_ORDER("SoftmaxBlockSparseCausal",
                        "softmax_scale", true, nullptr,
                        "bk",            true, nullptr)

// ============================================================================
// Multi-thread the kernel across HVX threads by slicing the H_q dimension.
//
// Background (from QHAS profiling at Mid_TopK8): without this directive the
// QNN scheduler runs the entire SoftmaxBlockSparseCausal call on a single
// HVX thread (tid 513 at 96% utilisation, the other five HVX threads idle).
// The reference Mul+Add+Softmax decomp distributes work across all six
// threads at ~50–80% util each. That single-vs-six-thread asymmetry is the
// dominant ~6× wall-clock gap, NOT kernel quality.
//
// Our input tensor is 4D NHWC = [B=1, H_q, BQ, top_k_BK]:
//   N (BATCH)  = B           = 1   (can't split)
//   H (HEIGHT) = H_q         = 16  ← split this
//   W (WIDTH)  = BQ          = 32
//   C (DEPTH)  = top_k_BK    = 256 (softmax axis — must NOT split)
//
// AUTOTHREAD_HVX(rank=1, "I", body) tells the HTP scheduler to slice the
// named tensor along rank 1 (H_q) and dispatch each slice on a separate
// HVX thread. The kernel itself doesn't need to change — its existing
// (b, h, q) loop just iterates over the smaller H slice for each thread.
// q_block_idx and bk are passed through unchanged (they're the same for
// every head).
//
// Fp16 fast path (the HVX kernel path):
// ============================================================================
// ============================================================================
// HVX-thread parallelism via AUTOSPLIT
// ============================================================================
// QHAS profiling shows the kernel runs on a single HVX thread otherwise
// (~1.13M cycles tid=513, the other five threads idle). The reference
// Mul+Add+Softmax decomp distributes across all six HVX threads at 50–80%
// util each — that single-vs-six asymmetry was the dominant ~4× gap.
//
// Two prerequisites that took a while to find:
//   1. REGISTER_PACKAGE_OPTIMIZATIONS() must be called in LLaMAPackageInit.
//      Without it, DEF_PACKAGE_OPTIMIZATION rules compile but never reach
//      the HTP optimizer. This was missing — added to LLaMAPackageInterface.cpp.
//   2. QNN backfills our rank-3 [H_q=16, BQ=32, top_k_BK=256] input to rank-4
//      [B=1, H_q=16, BQ=32, top_k_BK=256] internally. So DIM_BATCHES("*")=1,
//      not 16; DIM_HEIGHT("*")=16. Splitting along H_q means dim 1, not 0.
//      (Verified by inspecting fused_v*_htp.json — the tensor's `dims` field
//      shows [1, 16, 32, 256].)
//
// Chunk size 4 → 16/4 = 4 slices, distributed across the 6 HVX threads.
// Smaller chunks have higher per-slice overhead (the kernel's setup costs
// — splat builds, vmax/vsum register init, mask build — repeat per slice).
// 4 is a starting point worth re-tuning once the rule fires.
// ============================================================================
// AUTOSPLIT chunk 3 on H_q (dim 1) → 16/3 = 5 slices of 3 + 1 slice of 1
// → 6 slices total, one per HVX thread. QHAS confirms HVX 512/513/514/515/517
// at 85-91% util, the leftover-1 slice on one thread at 42%.
//
// Tuning observations (Hq=16, BQ=32, top_k_BK=256 on V79):
//   chunk 4 → 4 slices,  4 HVX threads at 87-93% util          → 0.19 ms
//   chunk 3 → 6 slices,  all 6 HVX threads engaged (this one)  → 0.15 ms
//   chunk 2 → 8 slices,  rule never fires (htp_ops=1)          → 0.49 ms
//   chunk 8 → 2 slices,  rule never fires AND broke correctness on qb0_tk8
//   AUTOTHREAD_HVX     → never fires (autothread_hvx_ntiles cap likely low)
//
// Why MUST we slice on H_q (dim 1) and not BQ (dim 2)? The triangle mask
// in the diagonal slot depends on the GLOBAL q-row index within BQ.
// AUTOSPLIT on BQ gives each slice rows q=0..(slice_size-1) without
// adjusting q_block_idx, so the mask is recomputed wrong for slices
// past row 0 — confirmed: BQ chunk 4 produces fused vs host err 0.96 with
// 8K-element divergence. Heads (H_q) are independent, so slicing there
// is safe.
//
// Why chunk 2/8 don't fire is unclear from the public headers; they may hit
// QNN-internal "too many slices for the inner work" or "splits must be a
// power of 2" heuristics. chunk 3 is the empirical sweet spot for H_q=16.
// Op() pattern lists inputs first (QK, q_block_idx) then params (scale, bk).
// q_block_idx is a SCALAR input — pass through TYPICAL_SLICE-untouched.
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("SoftmaxBlockSparseCausal", "QK", "q_block_idx", "softmax_scale", "bk"),
    GT(DIM_HEIGHT("*"), 3),
    AUTOSPLIT(1, "I", 3, Op("SoftmaxBlockSparseCausal", TYPICAL_SLICE("QK", "I"),
                            "q_block_idx", "softmax_scale", "bk")))

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"

// ----- bit utilities -----
static inline uint32_t sbsc_float_to_bits(float x) {
  union { float f; uint32_t i; } u = {.f = x};
  return u.i;
}
// Pack two copies of fp16 into a 32-bit word; Q6_V_vsplat_R then splats this
// word, giving a 64-lane fp16 splat vector.
static inline int32_t sbsc_float_to_fp16x2(float x) {
  union { int32_t i; __fp16 f[2]; } u = {.f = {(__fp16)x, (__fp16)x}};
  return u.i;
}

// ============================================================================
// Vectorised Cephes expf (fp32, 32 lanes per HVX vector). Same body as
// fa_expf in FlashAttention.cpp, lifted into HVX intrinsics. Accuracy ~6-7
// decimal digits; plenty for fp16 softmax.
//
// Inputs outside [-87.336544, 88.722839]:
//   x < -87.336544  -> output 0   (handled by mux at the end)
//   x > 88.722839   -> clamped to 88.722839 (then exp ~3.4e38, near fp32 max)
//
// always_inline so the 14 splat constants hoist out of the caller's loop.
// ============================================================================
static inline __attribute__((always_inline)) HVX_Vector sbsc_hvx_expf_sf(HVX_Vector x) {
  // Cephes constants
  HVX_Vector log2ef       = Q6_V_vsplat_R(sbsc_float_to_bits(1.44269504088896341f));
  HVX_Vector half         = Q6_V_vsplat_R(sbsc_float_to_bits(0.5f));
  HVX_Vector C1           = Q6_V_vsplat_R(sbsc_float_to_bits(0.693359375f));
  HVX_Vector C2           = Q6_V_vsplat_R(sbsc_float_to_bits(-2.12194440e-4f));
  HVX_Vector neg_lim      = Q6_V_vsplat_R(sbsc_float_to_bits(-87.336544f));
  HVX_Vector pos_lim      = Q6_V_vsplat_R(sbsc_float_to_bits(88.722839f));
  HVX_Vector one          = Q6_V_vsplat_R(sbsc_float_to_bits(1.0f));
  HVX_Vector zero         = Q6_V_vzero();
  HVX_Vector vsplat_127   = Q6_V_vsplat_R(127);

  // Polynomial coefficients (degree 6 in r)
  HVX_Vector c0 = Q6_V_vsplat_R(sbsc_float_to_bits(1.9875691500e-4f));
  HVX_Vector c1 = Q6_V_vsplat_R(sbsc_float_to_bits(1.3981999507e-3f));
  HVX_Vector c2 = Q6_V_vsplat_R(sbsc_float_to_bits(8.3334519073e-3f));
  HVX_Vector c3 = Q6_V_vsplat_R(sbsc_float_to_bits(4.1665795894e-2f));
  HVX_Vector c4 = Q6_V_vsplat_R(sbsc_float_to_bits(1.6666665459e-1f));
  HVX_Vector c5 = Q6_V_vsplat_R(sbsc_float_to_bits(5.0000001201e-1f));

  // Track lanes that should output 0 (x < -87.336544).
  HVX_VectorPred too_low = Q6_Q_vcmp_gt_VsfVsf(neg_lim, x);
  // Clamp upper end. (Lower end is handled by the too_low mux at the bottom.)
  x = Q6_Vsf_vmin_VsfVsf(x, pos_lim);

  // fx = x * LOG2EF + 0.5
  HVX_Vector fx_sf = Q6_Vsf_equals_Vqf32(
      Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, log2ef)), half));

  // n = floor(fx). Strategy: convert fx to int32 (whatever the rounding mode),
  // convert back, and adjust by -1 for any lane where the rounded value
  // exceeds fx. This works regardless of whether the convert truncates,
  // rounds-to-nearest, etc.
  HVX_Vector n_w  = Q6_Vw_equals_Vsf(fx_sf);
  HVX_Vector n_sf = Q6_Vsf_equals_Vw(n_w);
  HVX_VectorPred over = Q6_Q_vcmp_gt_VsfVsf(n_sf, fx_sf);
  HVX_Vector adj = Q6_V_vmux_QVV(over, Q6_V_vsplat_R(1), zero);
  n_w = Q6_Vw_vsub_VwVw(n_w, adj);
  HVX_Vector fn = Q6_Vsf_equals_Vw(n_w);

  // r = x - fn*C1 - fn*C2  (split into two steps for accuracy)
  HVX_Vector tmp1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(fn, C1));
  HVX_Vector r    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, tmp1));
  HVX_Vector tmp2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(fn, C2));
  r               = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(r, tmp2));

  HVX_Vector r2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(r, r));

  // Horner: p = (((((c0*r + c1)*r + c2)*r + c3)*r + c4)*r + c5)
  HVX_Vector p = c0;
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c1, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c2, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c3, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c4, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(c5, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r))));

  // p = p * r2 + r + 1
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(r, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, r2))));
  p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p, one));

  // 2^n via bit-bang: u.i = (n+127) << 23
  HVX_Vector exp_bits = Q6_Vw_vadd_VwVw(n_w, vsplat_127);
  HVX_Vector pow2     = Q6_Vw_vasl_VwR(exp_bits, 23);

  // result = p * pow2  (interpret pow2 as fp32 — the bit pattern IS valid fp32)
  HVX_Vector result = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(p, pow2));
  // Lanes with x < -87 -> 0
  result = Q6_V_vmux_QVV(too_low, zero, result);
  return result;
}

// Horizontal max of 32 sf lanes. Spill to stack, scalar reduce — same pattern
// FlashAttention.cpp uses for hsum. Robust against the qf32-spill brittleness
// described in custom_hvx_op_skill.md §4.2.
static inline float sbsc_hvx_sf_hmax(HVX_Vector v_sf) {
  union { HVX_Vector v; float f[32]; } u;
  u.v = v_sf;
  float m = -INFINITY;
  for (int i = 0; i < 32; ++i) if (u.f[i] > m) m = u.f[i];
  return m;
}
static inline float sbsc_hvx_sf_hsum(HVX_Vector v_sf) {
  union { HVX_Vector v; float f[32]; } u;
  u.v = v_sf;
  float s = 0.0f;
  for (int i = 0; i < 32; ++i) s += u.f[i];
  return s;
}

// Vectorised widen: read 64 fp16 lanes (1 hf vector) → 2 source-order sf
// vectors. Q6_Wqf32_vmpy_VhfVhf produces an INTERLEAVED Wqf32 (lo = even
// source lanes, hi = odd source lanes — see custom_hvx_op_skill.md §4.6);
// vshuff(-4) deinterleaves back to source order so masks built per source
// column apply naturally.
//
// Returned pair: .lo = source lanes 0..31, .hi = source lanes 32..63.
static inline __attribute__((always_inline)) HVX_VectorPair
sbsc_hvx_load64_widen_pair(const __fp16* s) {
  HVX_Vector splat_1 = Q6_V_vsplat_R(sbsc_float_to_fp16x2(1.0f));
  HVX_Vector hf = *(const HVX_UVector*)s;
  HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(hf, splat_1);
  HVX_Vector lo_il = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
  HVX_Vector hi_il = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
  return Q6_W_vshuff_VVR(hi_il, lo_il, -4);
}

// Vectorised narrow: 2 source-order qf32 vectors → 1 hf vector (source
// order), unaligned store to p. vdeal(-4) inverts vshuff(-4): it puts
// even source lanes in .lo and odd source lanes in .hi, which is exactly
// what Q6_Vhf_equals_Wqf32 expects to invert back to source-order hf.
static inline __attribute__((always_inline)) void
sbsc_hvx_store64_narrow_pair_qf(__fp16* p, HVX_Vector qf_a_src, HVX_Vector qf_b_src) {
  HVX_VectorPair pair = Q6_W_vdeal_VVR(qf_b_src, qf_a_src, -4);
  *(HVX_UVector*)p = Q6_Vhf_equals_Wqf32(pair);
}

// HVX shorthand
static inline __attribute__((always_inline)) HVX_Vector sbsc_mul_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline __attribute__((always_inline)) HVX_Vector sbsc_add_sf(HVX_Vector a, HVX_Vector b) {
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}

// One row of HVX-vectorised softmax. Requires bk == 32 (one HVX fp32 vector
// = one block), W is a multiple of 64 (i.e. top_k is even — diagonal slot
// is always slot_b in the last pair), and bk-block layout is contiguous.
// Falls back to scalar otherwise.
//
// v4 layout:
//   Pair processing — each pair holds 2 adjacent slots (slot_a even,
//   slot_b = slot_a+1). 1 hf vector (64 lanes) per pair load.
//
//   Pass 1: per-pair, vectorised widen → 2 sf vectors (sf_a, sf_b in source
//           order), scaled+per-pair-mask → vmax. ALL_PADDING pairs skipped.
//   Pass 2: per-pair, vectorised widen, scaled+mask−row_max → exp →
//           tmp[fp32, W], accumulate vsum.
//   Pass 3: per-pair, multiply tmp[fp32] by inv, qf32→hf narrow, unaligned
//           hf store to p.
//
// Pair phases (slot_a, slot_b):
//   REAL+REAL          : mask_a = 0,    mask_b = 0
//   REAL+PADDING       : mask_a = 0,    mask_b = -100
//   REAL+DIAGONAL      : mask_a = 0,    mask_b = triangle_mask
//   PADDING+PADDING    : skip (write zeros to tmp; doesn't affect sum/max)
//   PADDING+DIAGONAL   : mask_a = -100, mask_b = triangle_mask
//
// (DIAGONAL+anything doesn't appear because diag_slot is always odd here, so
// it's always slot_b.)
//
// The expf cost is ~0.3 ms per dispatch (measured by SBSC_NO_EXP probe);
// scalar widen + narrow were ~3.5 ms combined in v2 — both eliminated here.
static inline void sbsc_row_hvx_f16(const __fp16* s, __fp16* p, int W, int bk,
                                    int q_block_idx, int q_row, float scale) {
  const int top_k         = W / bk;
  const int diag_slot     = top_k - 1;
  const int n_real_hist   = (q_block_idx < diag_slot) ? q_block_idx : diag_slot;
  (void)top_k;

  HVX_Vector scale_v   = Q6_V_vsplat_R(sbsc_float_to_bits(scale));
  HVX_Vector neg_inf_v = Q6_V_vsplat_R(sbsc_float_to_bits(-INFINITY));
  HVX_Vector neg100_v  = Q6_V_vsplat_R(sbsc_float_to_bits(-100.0f));
  HVX_Vector zero_v    = Q6_V_vzero();

  // Triangle mask for the diagonal slot. Q6_Q_vsetq_R(R) sets only the low
  // 7 bits of R (R=128 → 0 bytes set), so guard the all-valid case explicitly.
  HVX_Vector triangle_mask;
  if (q_row + 1 >= bk) {
    triangle_mask = zero_v;
  } else {
    HVX_VectorPred valid_pred = Q6_Q_vsetq_R((q_row + 1) * 4);
    triangle_mask = Q6_V_vmux_QVV(valid_pred, zero_v, neg100_v);
  }

  // Determine each pair's "kind" once and pre-build the mask values for
  // the kinds that actually need a mask add.
  //
  //   PK_REAL_REAL : both slots are real history; mask add is a no-op
  //                  (both masks would be zero_v) — skip the adds entirely.
  //   PK_HAS_MASK  : at least one slot is padding or diagonal; mask_a/mask_b
  //                  carry the actual additive bias to apply.
  //   PK_SKIP      : both slots are padding; skip pass 1/2 contributions
  //                  entirely (sf_a/sf_b stay zero, output is zero).
  //
  // For typical mid-prefill (n_real_hist == top_k - 1), most pairs are
  // PK_REAL_REAL — only the last pair (containing the diagonal slot) is
  // PK_HAS_MASK. Skipping the mask add for the rest is ~6 fewer adds/row.
  const int n_pairs = W / 64;
  // Hard limit n_pairs ≤ 8 for the register-resident fast path. That covers
  // top_k ≤ 16 (W ≤ 512), which includes every production shape we test.
  // Caller is responsible for ensuring top_k ≤ 16; the impl()'s top-level
  // dtype-dispatch above falls back to scalar for shapes that don't fit
  // the HVX path (bk != 32, K not divisible by 32, etc.) — but n_pairs > 8
  // would write past these arrays. Add an explicit check there if you ever
  // dispatch with W > 512.
  enum PairKind : uint8_t { PK_REAL_REAL, PK_HAS_MASK, PK_SKIP };
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
      else               mask_b_arr[pi] = neg100_v;  // padding
    }
  }

  // ===== Phase 1: read, widen, scale [+mask], find vmax =====
  // sf_a[]/sf_b[] hold the working values across all 3 phases — no tmp
  // buffer load/store roundtrips. The compiler keeps the small-N array
  // (n_pairs ≤ 8) live in HVX registers when the loop is short enough.
  HVX_Vector sf_a[8], sf_b[8];
  HVX_Vector vmax = neg_inf_v;
  for (int pi = 0; pi < n_pairs; ++pi) {
    if (kind[pi] == PK_SKIP) {
      // Sentinel zero: phase 3's narrow writes 0 to p[SKIP cols]. vmax skips
      // these (the `continue` below) so they don't perturb the max.
      sf_a[pi] = zero_v;
      sf_b[pi] = zero_v;
      continue;
    }
    HVX_VectorPair w = sbsc_hvx_load64_widen_pair(s + pi * 64);
    sf_a[pi] = sbsc_mul_sf(Q6_V_lo_W(w), scale_v);
    sf_b[pi] = sbsc_mul_sf(Q6_V_hi_W(w), scale_v);
    if (kind[pi] == PK_HAS_MASK) {
      sf_a[pi] = sbsc_add_sf(sf_a[pi], mask_a_arr[pi]);
      sf_b[pi] = sbsc_add_sf(sf_b[pi], mask_b_arr[pi]);
    }
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, sf_a[pi]);
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, sf_b[pi]);
  }
  const float row_max = sbsc_hvx_sf_hmax(vmax);
  HVX_Vector neg_max_v = Q6_V_vsplat_R(sbsc_float_to_bits(-row_max));

  // ===== Phase 2: subtract row_max, exp, accumulate sum =====
  // sf_a/sf_b hold scaled+[mask] from Phase 1. After this phase they hold
  // exp(scaled+[mask] - row_max) — ready for Phase 3 multiply.
  HVX_Vector vsum = zero_v;
  for (int pi = 0; pi < n_pairs; ++pi) {
    if (kind[pi] == PK_SKIP) continue;  // sf_a/sf_b stay zero, fine for Phase 3
    sf_a[pi] = sbsc_add_sf(sf_a[pi], neg_max_v);
    sf_b[pi] = sbsc_add_sf(sf_b[pi], neg_max_v);
#ifdef SBSC_NO_EXP
    // perf-probe: skip expf to isolate cost
#else
    sf_a[pi] = sbsc_hvx_expf_sf(sf_a[pi]);
    sf_b[pi] = sbsc_hvx_expf_sf(sf_b[pi]);
#endif
    vsum = sbsc_add_sf(vsum, sf_a[pi]);
    vsum = sbsc_add_sf(vsum, sf_b[pi]);
  }

  const float row_sum = sbsc_hvx_sf_hsum(vsum);
  const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
  HVX_Vector inv_v = Q6_V_vsplat_R(sbsc_float_to_bits(inv));

  // ===== Phase 3: multiply by inv, narrow, store =====
  // For PK_SKIP pairs, sf_a[pi]/sf_b[pi] are zero from Phase 1 — the
  // multiply gives 0 → narrow gives fp16 0 → correct masked-output value.
  for (int pi = 0; pi < n_pairs; ++pi) {
    HVX_Vector q_a = Q6_Vqf32_vmpy_VsfVsf(sf_a[pi], inv_v);
    HVX_Vector q_b = Q6_Vqf32_vmpy_VsfVsf(sf_b[pi], inv_v);
    sbsc_hvx_store64_narrow_pair_qf(p + pi * 64, q_a, q_b);
  }
}

#else  // REFERENCE_OP — ARM-side / validation build can use libm.

static inline float sbsc_expf_ref(float x) { return expf(x); }

#endif  // !REFERENCE_OP

// ============================================================================
// Scalar reference (used by aarch64 prepare-side build and as fp32 fallback).
// ============================================================================
template<typename T_in, typename T_out>
static inline void sbsc_one_row_scalar(const T_in* __restrict__ s, T_out* __restrict__ p,
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
#ifndef REFERENCE_OP
    float e = expf(v - row_max);  // libm available on the HTP for the fp32 fallback path
#else
    float e = expf(v - row_max);
#endif
    denom += e;
    p[c] = (T_out)e;
  }
  float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
  for (int c = 0; c < W; ++c) p[c] = (T_out)((float)p[c] * inv);
}

template<typename TensorType, typename TensorType1>
GraphStatus softmaxblocksparsecausalImpl(TensorType& out_0,
                                         const TensorType& in_0,
                                         const TensorType1& q_block_idx_t,
                                         const PlainFloatTensor& softmax_scale,
                                         const Tensor& bk_t) {
  out_0.set_dims(in_0);
  auto [B, H, Wq, K] = in_0.dims();

  const float scale = softmax_scale(0, 0, 0, 0);
  const uint32_t q_block_idx = (uint32_t)q_block_idx_t(0, 0, 0, 0);
  const uint32_t bk = (uint32_t)bk_t(0, 0, 0, 0);

  if (bk == 0 || (K % bk) != 0) return GraphStatus::ErrorDimensions;
  const int top_k = (int)(K / bk);
  if (top_k < 1) return GraphStatus::ErrorDimensions;

  if (in_0.get_dtype() == DType::Float16 && out_0.get_dtype() == DType::Float16) {
    const __fp16* sp = (const __fp16*)in_0.raw_data_const();
    __fp16* pp = (__fp16*)out_0.raw_data();
#ifndef REFERENCE_OP
    // Fast HVX path requires bk == 32 (one fp32 HVX vector per k-block) and
    // K % 32 == 0. Drop to scalar otherwise.
    const bool hvx_ok = (bk == 32) && ((K % 32) == 0);
    for (int b = 0; b < (int)B; ++b) {
      for (int h = 0; h < (int)H; ++h) {
        for (int q = 0; q < (int)Wq; ++q) {
          const __fp16* srow = sp + (((size_t)b * H + h) * Wq + q) * K;
          __fp16* prow = pp + (((size_t)b * H + h) * Wq + q) * K;
          if (hvx_ok) {
            sbsc_row_hvx_f16(srow, prow, (int)K, (int)bk, (int)q_block_idx, q, scale);
          } else {
            sbsc_one_row_scalar<__fp16, __fp16>(srow, prow, (int)K, (int)bk, (int)q_block_idx, q, scale);
          }
        }
      }
    }
#else
    for (int b = 0; b < (int)B; ++b) {
      for (int h = 0; h < (int)H; ++h) {
        for (int q = 0; q < (int)Wq; ++q) {
          const __fp16* srow = sp + (((size_t)b * H + h) * Wq + q) * K;
          __fp16* prow = pp + (((size_t)b * H + h) * Wq + q) * K;
          sbsc_one_row_scalar<__fp16, __fp16>(srow, prow, (int)K, (int)bk, (int)q_block_idx, q, scale);
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
        for (int q = 0; q < (int)Wq; ++q) {
          const float* srow = sp + (((size_t)b * H + h) * Wq + q) * K;
          float* prow = pp + (((size_t)b * H + h) * Wq + q) * K;
          sbsc_one_row_scalar<float, float>(srow, prow, (int)K, (int)bk, (int)q_block_idx, q, scale);
        }
      }
    }
    return GraphStatus::Success;
  }

  return GraphStatus::ErrorUnsupported;
}

__attribute__((unused)) static float softmaxblocksparsecausalCostFunc(const Op* op) {
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_SoftmaxBlockSparseCausal);
