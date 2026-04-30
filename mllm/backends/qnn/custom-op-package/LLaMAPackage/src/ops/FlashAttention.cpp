//==============================================================================
// FlashAttention custom op for LLaMAPackage
//
// Implements scaled dot-product attention using the FlashAttention online
// softmax algorithm (Tri Dao et al., 2022). Adapted for Hexagon HTP / HVX:
// the on-chip SRAM tiling pattern of the Nvidia kernel is replaced with
// register-resident accumulators + L2 prefetching of K/V tiles. The
// numerical recurrence is identical:
//
//   m_new = max(m, rowmax(S_ij))
//   l_new = exp(m - m_new) * l + rowsum(exp(S_ij - m_new))
//   O     = diag(l_new)^-1 * ( diag(l) * exp(m - m_new) * O + exp(S_ij - m_new) * V_j )
//
// Layout (NHWC, matching the rest of LLaMAPackage):
//   in[0] Q : [B, S_q,  H_q,  D]
//   in[1] K : [B, S_kv, H_kv, D]
//   in[2] V : [B, S_kv, H_kv, D]
//   out[0]  : [B, S_q,  H_q,  D]
//
// GQA is supported: H_q must be a multiple of H_kv.
//
// Parameters (in order, see DEF_PACKAGE_PARAM_ORDER below):
//   softmax_scale : float scalar (== 1/sqrt(D) when None on host side)
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

BEGIN_PKG_OP_DEFINITION(PKG_FlashAttention);

template<typename TensorType>
GraphStatus flashattentionImpl(TensorType& out_0, const TensorType& q_in, const TensorType& k_in, const TensorType& v_in,
                               const PlainFloatTensor& softmax_scale, const Tensor& causal,
                               hnnx::op_slice_spec slice_spec);

static float flashattentionCostFunc(const Op* op);

DEF_PACKAGE_OP((flashattentionImpl<Tensor>), "FlashAttention")

DEF_PACKAGE_PARAM_ORDER("FlashAttention", "softmax_scale", true, nullptr, "causal", true, nullptr)

// Causal split tile. One S_q position per slice keeps masking exact because
// slice_idx maps directly to the original S_q position.
#define FA_SQ_TILE 1

#ifndef REFERENCE_OP
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("FlashAttention", "Q", "K", "V", "Scale", "Causal"),
    AND(EQ(DTYPE_OF("Q"), DType::Float32),
        EQ(DTYPE_OF("*"), DType::Float32),
        GT(DIM_HEIGHT("*"), FA_SQ_TILE),
        CONSTVAL_INT_VALID("Causal", 0),
        EQ(CONSTVAL_INT("Causal", 0), 0)),
    AUTOTHREAD_HVX(1, "I", Op("FlashAttention", TYPICAL_SLICE("Q", "I"), "K", "V", "Scale", "Causal")))

DEF_PACKAGE_OPTIMIZATION(
    EARLY + 1,
    Op("FlashAttention", "Q", "K", "V", "Scale", "Causal"),
    AND(EQ(DTYPE_OF("Q"), DType::Float32),
        EQ(DTYPE_OF("*"), DType::Float32),
        GT(DIM_HEIGHT("*"), FA_SQ_TILE),
        CONSTVAL_INT_VALID("Causal", 0),
        EQ(CONSTVAL_INT("Causal", 0), 1)),
    AUTOSPLIT(1, "I", FA_SQ_TILE, Op("FlashAttention", TYPICAL_SLICE("Q", "I"), "K", "V", "Scale", "Causal")))
#endif

// Block size along S_kv. Picked to keep K_j / V_j tiles warm in L2 while
// the inner D loop runs.  Small enough to limit the cost of the
// re-scaling pass over O when m changes.
#define FA_BC 64

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"

#define BLOCK_SIZE (8 * 1024 / VLEN)
#define L2FETCH_AHEAD (BLOCK_SIZE)

static HVX_INLINE_ALWAYS uint32_t fa_float_to_bits(float x) {
  union {
    float f;
    uint32_t i;
  } u = {.f = x};
  return u.i;
}

// Self-contained scalar expf for the HVX path. libm's expf is not reliably
// available on Hexagon HTP — every existing op in this package only calls
// expf inside the REFERENCE_OP (aarch64) blocks. If we call libm's expf here
// it returns 0 at runtime, which silently zeroes the FlashAttention output.
//
// Cephes-style range-reduction: x = n*ln(2) + r, |r| <= ln(2)/2.
// exp(r) is then a degree-5 polynomial; 2^n is built by adjusting the
// IEEE-754 exponent. Accurate to ~6-7 decimal digits, more than enough for
// fp32 attention.
static inline float fa_expf(float x) {
  if (x < -87.336544f) return 0.0f;
  if (x > 88.722839f) x = 88.722839f;

  const float LOG2EF = 1.44269504088896341f;
  const float C1 = 0.693359375f;
  const float C2 = -2.12194440e-4f;

  // n = round(x * log2(e))
  float fx = x * LOG2EF + 0.5f;
  int32_t n = (int32_t)fx;
  if ((float)n > fx) n -= 1;  // floor for negative x
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
  u.i = (n + 127) << 23;  // 2^n
  return p * u.f;
}

// Convert qf32 → sf, spill to stack, sum the 32 lanes in scalar code. Avoids
// the brittle vlalign-based horizontal reduction.
static inline float fa_hvx_qf32_hsum(HVX_Vector v_qf32) {
  HVX_Vector acc_sf = Q6_Vsf_equals_Vqf32(v_qf32);
  union {
    HVX_Vector v;
    float f[32];
  } u;
  u.v = acc_sf;
  float sum = 0.0f;
  for (int i = 0; i < 32; ++i) sum += u.f[i];
  return sum;
}

// All HVX accesses use HVX_UVector (unaligned vmemu) — QNN output tensors
// are not guaranteed to be 128-byte aligned, and slicing per (head, seq)
// can land on smaller alignments. Aligned vmem silently truncates to the
// previous 128-byte boundary on Hexagon, which corrupts neighbouring rows.

// dst[0..D-1] += alpha * src[0..D-1], all fp32. D assumed multiple of 32.
static inline void fa_hvx_axpy_f32(float* restrict dst, const float* restrict src, float alpha, uint32_t D) {
  HVX_UVector* dptr = (HVX_UVector*)dst;
  const HVX_UVector* sptr = (const HVX_UVector*)src;
  HVX_Vector alpha_sf = Q6_V_vsplat_R(fa_float_to_bits(alpha));
  HVX_Vector zero = Q6_V_vzero();
  uint32_t nv = D / 32;
  for (uint32_t i = 0; i < nv; ++i) {
    HVX_Vector s = sptr[i];
    HVX_Vector d = dptr[i];
    HVX_Vector prod_qf = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, s);
    HVX_Vector d_qf = Q6_Vqf32_vadd_VsfVsf(d, zero);
    HVX_Vector sum_qf = Q6_Vqf32_vadd_Vqf32Vqf32(prod_qf, d_qf);
    dptr[i] = Q6_Vsf_equals_Vqf32(sum_qf);
  }
}

// dst[0..D-1] *= alpha. D assumed multiple of 32.
static inline void fa_hvx_scale_f32(float* restrict dst, float alpha, uint32_t D) {
  HVX_UVector* dptr = (HVX_UVector*)dst;
  HVX_Vector alpha_sf = Q6_V_vsplat_R(fa_float_to_bits(alpha));
  uint32_t nv = D / 32;
  for (uint32_t i = 0; i < nv; ++i) {
    HVX_Vector s = dptr[i];
    HVX_Vector prod_qf = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, s);
    dptr[i] = Q6_Vsf_equals_Vqf32(prod_qf);
  }
}

static inline void fa_hvx_zero_f32(float* restrict dst, uint32_t D) {
  HVX_UVector* dptr = (HVX_UVector*)dst;
  HVX_Vector zero = Q6_V_vzero();
  uint32_t nv = D / 32;
  for (uint32_t i = 0; i < nv; ++i) dptr[i] = zero;
}

// Returns sum(q[i] * k[i]) over D, fp32. D assumed multiple of 32.
static inline float fa_hvx_dot_f32(const float* restrict q, const float* restrict k, uint32_t D) {
  const HVX_UVector* qptr = (const HVX_UVector*)q;
  const HVX_UVector* kptr = (const HVX_UVector*)k;
  HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
  uint32_t nv = D / 32;
  for (uint32_t i = 0; i < nv; ++i) { acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(qptr[i], kptr[i])); }
  return fa_hvx_qf32_hsum(acc);
}

// One Q row * one head, fp32 path: full online softmax over [0, S_kv_lim).
static void fa_one_row_f32(const float* restrict q,        // [D]
                           const float* restrict k_base,   // [S_kv, H_kv*D] block contiguous over S_kv
                           const float* restrict v_base,   // [S_kv, H_kv*D]
                           uint32_t kv_stride_elems,       // == H_kv * D
                           float* restrict o,              // [D] (output)
                           uint32_t S_kv_lim, uint32_t D, float scale) {
  fa_hvx_zero_f32(o, D);
  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j0 = 0; j0 < S_kv_lim; j0 += FA_BC) {
    uint32_t jN = j0 + FA_BC;
    if (jN > S_kv_lim) jN = S_kv_lim;

    // L2-prefetch the next K and V tile in this attention row.
    if (jN < S_kv_lim) {
      uint32_t fetch_rows = (jN + FA_BC <= S_kv_lim) ? FA_BC : (S_kv_lim - jN);
      uint32_t bytes_per_row = D * sizeof(float);
      l2fetch(k_base + (size_t)jN * kv_stride_elems, bytes_per_row, bytes_per_row, fetch_rows, 0);
      l2fetch(v_base + (size_t)jN * kv_stride_elems, bytes_per_row, bytes_per_row, fetch_rows, 0);
    }

    // 1) Compute S_ij = Q · K_j^T * scale, find rowmax in this tile.
    float s_buf[FA_BC];
    float m_tile = -INFINITY;
    for (uint32_t j = j0; j < jN; ++j) {
      const float* kptr = k_base + (size_t)j * kv_stride_elems;
      float dot = fa_hvx_dot_f32(q, kptr, D);
      float s = dot * scale;
      s_buf[j - j0] = s;
      if (s > m_tile) m_tile = s;
    }

    // 2) Online softmax update.
    float m_new = (m > m_tile) ? m : m_tile;
    float alpha = (m == -INFINITY) ? 0.0f : fa_expf(m - m_new);  // O *= alpha
    float l_tile = 0.0f;
    for (uint32_t j = j0; j < jN; ++j) {
      float p = fa_expf(s_buf[j - j0] - m_new);
      s_buf[j - j0] = p;
      l_tile += p;
    }

    // 3) Re-scale running output, then accumulate p_j * V_j.
    if (alpha != 1.0f) fa_hvx_scale_f32(o, alpha, D);
    for (uint32_t j = j0; j < jN; ++j) {
      const float* vptr = v_base + (size_t)j * kv_stride_elems;
      fa_hvx_axpy_f32(o, vptr, s_buf[j - j0], D);
    }

    l = alpha * l + l_tile;
    m = m_new;
  }

  // Normalize.
  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  fa_hvx_scale_f32(o, inv_l, D);
}

static void fa_flashattention_f32(float* restrict Obuf, const float* restrict Qbuf, const float* restrict Kbuf,
                                  const float* restrict Vbuf, uint32_t Bq, uint32_t Sq, uint32_t Sk,
                                  uint32_t Hq, uint32_t Hkv, uint32_t Dq, uint32_t q_stride,
                                  uint32_t kv_stride, uint32_t group, float scale, uint32_t is_causal,
                                  uint32_t global_sq_start, uint32_t original_sq) {
  for (uint32_t b = 0; b < Bq; ++b) {
    const float* Qb = Qbuf + (size_t)b * Sq * q_stride;
    const float* Kb = Kbuf + (size_t)b * Sk * kv_stride;
    const float* Vb = Vbuf + (size_t)b * Sk * kv_stride;
    float* Ob = Obuf + (size_t)b * Sq * q_stride;
    for (uint32_t s = 0; s < Sq; ++s) {
      // For decode (original_sq == 1) S_kv_lim == Sk; for sliced causal
      // prefill use the original row position, not the local slice offset.
      uint32_t kv_lim = Sk;
      if (is_causal && original_sq > 1) kv_lim = (Sk - original_sq) + global_sq_start + s + 1;
      for (uint32_t hq = 0; hq < Hq; ++hq) {
        uint32_t hkv = hq / group;
        const float* qrow = Qb + (size_t)s * q_stride + hq * Dq;
        const float* krow = Kb + hkv * Dq;
        const float* vrow = Vb + hkv * Dq;
        float* orow = Ob + (size_t)s * q_stride + hq * Dq;
        fa_one_row_f32(qrow, krow, vrow, kv_stride, orow, kv_lim, Dq, scale);
      }
    }
  }
}

#endif  // !REFERENCE_OP

// Reference / fallback path — also compiled for aarch64-android with -DREFERENCE_OP.
static inline void fa_one_row_ref_f32(const float* q, const float* k_base, const float* v_base, uint32_t kv_stride_elems,
                                      float* o, uint32_t S_kv_lim, uint32_t D, float scale) {
  for (uint32_t d = 0; d < D; ++d) o[d] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j0 = 0; j0 < S_kv_lim; j0 += FA_BC) {
    uint32_t jN = j0 + FA_BC;
    if (jN > S_kv_lim) jN = S_kv_lim;

    float s_buf[FA_BC];
    float m_tile = -INFINITY;
    for (uint32_t j = j0; j < jN; ++j) {
      const float* kptr = k_base + (size_t)j * kv_stride_elems;
      float dot = 0.0f;
      for (uint32_t d = 0; d < D; ++d) dot += q[d] * kptr[d];
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
      for (uint32_t d = 0; d < D; ++d) o[d] *= alpha;
    }
    for (uint32_t j = j0; j < jN; ++j) {
      const float* vptr = v_base + (size_t)j * kv_stride_elems;
      float p = s_buf[j - j0];
      for (uint32_t d = 0; d < D; ++d) o[d] += p * vptr[d];
    }

    l = alpha * l + l_tile;
    m = m_new;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (uint32_t d = 0; d < D; ++d) o[d] *= inv_l;
}

static inline void fa_one_row_ref_f16(const __fp16* q, const __fp16* k_base, const __fp16* v_base, uint32_t kv_stride_elems,
                                      __fp16* o, uint32_t S_kv_lim, uint32_t D, float scale) {
  // Accumulate the recurrence in fp32 for stability, write back fp16.
  float o_tmp[2048];  // D <= 2048 (head_dim) is plenty for current LLMs.
  for (uint32_t d = 0; d < D; ++d) o_tmp[d] = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  for (uint32_t j0 = 0; j0 < S_kv_lim; j0 += FA_BC) {
    uint32_t jN = j0 + FA_BC;
    if (jN > S_kv_lim) jN = S_kv_lim;

    float s_buf[FA_BC];
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
GraphStatus flashattentionImpl(TensorType& out_0, const TensorType& q_in, const TensorType& k_in, const TensorType& v_in,
                               const PlainFloatTensor& softmax_scale, const Tensor& causal,
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
  // HVX vectorized path requires D % 32 (32 fp32 lanes per HVX vector).
  if ((Dq & 31) != 0) return GraphStatus::ErrorDimensions;

  const uint32_t slice_count = slice_spec.num_slices();
  const uint32_t slice_idx = slice_spec.slice_idx();
  const uint32_t original_sq = (slice_count > 1 && Sq == FA_SQ_TILE) ? slice_count : (uint32_t)Sq;
  const uint32_t global_sq_start = (slice_count > 1 && Sq == FA_SQ_TILE) ? slice_idx : 0;
  if (Sk < original_sq) return GraphStatus::ErrorDimensions;

  const uint32_t group = Hq / Hkv;          // GQA fan-out
  const uint32_t kv_stride = Hkv * Dq;      // bytes are dtype-dependent; this is element count
  const uint32_t q_stride = Hq * Dq;        // per S_q row
  const float scale = softmax_scale(0, 0, 0, 0);
  const uint32_t is_causal = (uint32_t)causal(0, 0, 0, 0);

  if (q_in.get_dtype() == DType::Float32 && out_0.get_dtype() == DType::Float32) {
    const float* Qbuf = (const float*)q_in.raw_data_const();
    const float* Kbuf = (const float*)k_in.raw_data_const();
    const float* Vbuf = (const float*)v_in.raw_data_const();
    float* Obuf = (float*)out_0.raw_data();

#ifndef REFERENCE_OP
    fa_flashattention_f32(Obuf, Qbuf, Kbuf, Vbuf, (uint32_t)Bq, (uint32_t)Sq, (uint32_t)Sk, (uint32_t)Hq,
                          (uint32_t)Hkv, (uint32_t)Dq, q_stride, kv_stride, group, scale, is_causal,
                          global_sq_start, original_sq);
#else
    for (uint32_t b = 0; b < (uint32_t)Bq; ++b) {
      const float* Qb = Qbuf + (size_t)b * Sq * q_stride;
      const float* Kb = Kbuf + (size_t)b * Sk * kv_stride;
      const float* Vb = Vbuf + (size_t)b * Sk * kv_stride;
      float* Ob = Obuf + (size_t)b * Sq * q_stride;
      for (uint32_t s = 0; s < (uint32_t)Sq; ++s) {
        // For decode (original_sq == 1) S_kv_lim == Sk; for sliced causal
        // prefill use the original row position, not the local slice offset.
        uint32_t kv_lim = (uint32_t)Sk;
        if (is_causal && original_sq > 1) kv_lim = (Sk - original_sq) + global_sq_start + s + 1;
        for (uint32_t hq = 0; hq < (uint32_t)Hq; ++hq) {
          uint32_t hkv = hq / group;
          const float* qrow = Qb + (size_t)s * q_stride + hq * Dq;
          const float* krow = Kb + hkv * Dq;
          const float* vrow = Vb + hkv * Dq;
          float* orow = Ob + (size_t)s * q_stride + hq * Dq;
          fa_one_row_ref_f32(qrow, krow, vrow, kv_stride, orow, kv_lim, Dq, scale);
        }
      }
    }
#endif
    return GraphStatus::Success;
  }

  if (q_in.get_dtype() == DType::Float16 && out_0.get_dtype() == DType::Float16) {
    const __fp16* Qbuf = (const __fp16*)q_in.raw_data_const();
    const __fp16* Kbuf = (const __fp16*)k_in.raw_data_const();
    const __fp16* Vbuf = (const __fp16*)v_in.raw_data_const();
    __fp16* Obuf = (__fp16*)out_0.raw_data();

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
          fa_one_row_ref_f16(qrow, krow, vrow, kv_stride, orow, kv_lim, Dq, scale);
        }
      }
    }
    return GraphStatus::Success;
  }

  return GraphStatus::ErrorUnsupported;
}

__attribute__((unused)) static float flashattentionCostFunc(const Op* op) {
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_FlashAttention);
