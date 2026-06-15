// Two-pass GEMM-class FlashAttention prefill kernels (Adreno 830, fp16 storage,
// fp32 accumulate). Reuses the validated variant-C 8x4 image-A / packed-B GEMM
// (examples/fa_opencl_bench/gemm_vc_ref.cpp) for both QK^T and P*V, with a
// query-major fp16 score scratch and an in-pass softmax+transpose+pack so the
// two GEMMs' opposite score orientations reconcile without an extra transpose.
//
// All kernels batch heads via get_global_id(2) = h with per-head pointer bases.
//
// Layouts (fp16 unless noted; element-index formulas, per head):
//   Q  [Sq,128]      Q[q*128 + d]
//   K  [Skv,128]     K[k*128 + d]
//   V  [Skv,128]     V[k*128 + d]
//   Qp [32,Sq,4]     Qp[(g*Sq+q)*4+kk] = Q[q*128 + g*4+kk]      (pack_q;  QK B)
//   Kt [128,Skv]     Kt[d*Skv + k]     = K[k*128 + d]           (trans_k; QK A-image)
//   S  [Sq,Skv]      S[q*Skv + k]                               (qk_gemm out, query-major)
//   Pp [Skv/4,Sq,4]  Pp[(g*Sq+q)*4+kk] = P[q*Skv + g*4+kk]      (softmax_pack; PV B)
//   O  [Sq,128]      O[q*128 + d]                               (pv_gemm out)
//   V is its own PV A-image (natural [Skv,128]).

R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FA_D
#define FA_D 128
#endif
#ifndef QK_CLAMP
#define QK_CLAMP 60000.0f
#endif

// ---- pack_q: Q[Sq,128] -> Qp[32, Sq, 4].  global=(Sq, 32, H)
__kernel void pack_q(__global const half* Q, __global half* Qp,
                     const int Sq, const int H) {
  const int q = get_global_id(0);
  const int g = get_global_id(1);   // 0..31  (d-group)
  const int h = get_global_id(2);
  if (q >= Sq || g >= FA_D / 4) return;
  __global const half* Qh = Q + (long)h * Sq * FA_D;
  __global half* Qph = Qp + (long)h * (FA_D / 4) * Sq * 4;
  const int sd = g * 4;
  #pragma unroll
  for (int kk = 0; kk < 4; ++kk)
    Qph[((long)g * Sq + q) * 4 + kk] = Qh[(long)q * FA_D + sd + kk];
}

// ---- trans_k: K[Skv,128] -> Kt[128, Skv].  global=(Skv, 128, H)
__kernel void trans_k(__global const half* K, __global half* Kt,
                      const int Skv, const int H) {
  const int k = get_global_id(0);
  const int d = get_global_id(1);
  const int h = get_global_id(2);
  if (k >= Skv || d >= FA_D) return;
  __global const half* Kh = K + (long)h * Skv * FA_D;
  __global half* Kth = Kt + (long)h * FA_D * Skv;
  Kth[(long)d * Skv + k] = Kh[(long)k * FA_D + d];
}

// ---- qk_gemm: S[q,k] = scale * (Q[q,:].K[k,:]) with causal mask.
// A = Kt image [R=d, M=k=Skv]; B = Qp packed [d/4, q=N=Sq, 4]; reduction d=128.
// dst[n*M+m] = S[q*Skv + k].  global=(Skv/8, Sq/4, H), local tunable.
__kernel void qk_gemm(__read_only image1d_buffer_t Kt_img,
                      __global const half* Qp, __global half* S,
                      const int Sq, const int Skv, const int H,
                      const float scale, const int causal) {
  const int gy = get_global_id(0);   // M-tile: 8 keys
  const int gx = get_global_id(1);   // N-tile: 4 queries
  const int h  = get_global_id(2);
  const int M  = Skv, N = Sq;
  if (gy * 8 >= M || gx * 4 >= N) return;
  const int gx_4 = gx << 2;
  const int gy_8 = gy << 3;
  const int M_4  = M >> 2;

  // per-head bases
  const int kt_tex_base = h * FA_D * (M >> 2);              // half4 texels of Kt[h]
  __global const half* Qph = Qp + (long)h * (FA_D / 4) * N * 4;
  __global half* Sh = S + (long)h * (long)Sq * Skv;

  // Causal block-skip: if this 8-key x 4-query tile is ENTIRELY above the
  // diagonal (min key gy_8 > max q_pos over the 4 queries), skip the 128-deep
  // matmul and just store -INF so softmax/pv read exact zeros.
  if (causal && gy_8 > (Skv - Sq) + gx_4 + 3) {
    #define EMITINF(NN) { const int q = gx_4 + (NN); if (q < N) vstore8((half8)(-INFINITY), 0, Sh + (long)q * Skv + gy_8); }
    EMITINF(0); EMITINF(1); EMITINF(2); EMITINF(3);
    #undef EMITINF
    return;
  }

  float8 c0 = (float8)0, c1 = (float8)0, c2 = (float8)0, c3 = (float8)0;
  half8 B0, B1, B2, B3;
  for (int i = 0; i < FA_D; i += 4) {
    const int t = kt_tex_base + gy * 2 + i * M_4;
    B0.s0123 = read_imageh(Kt_img, t);       B0.s4567 = read_imageh(Kt_img, t + 1);
    const int t1 = kt_tex_base + gy * 2 + (i + 1) * M_4;
    B1.s0123 = read_imageh(Kt_img, t1);      B1.s4567 = read_imageh(Kt_img, t1 + 1);
    const int t2 = kt_tex_base + gy * 2 + (i + 2) * M_4;
    B2.s0123 = read_imageh(Kt_img, t2);      B2.s4567 = read_imageh(Kt_img, t2 + 1);
    const int t3 = kt_tex_base + gy * 2 + (i + 3) * M_4;
    B3.s0123 = read_imageh(Kt_img, t3);      B3.s4567 = read_imageh(Kt_img, t3 + 1);
    half16 w = vload16(0, Qph + ((long)(i >> 2) * N + gx_4) * 4);
    c0 += convert_float8(B0) * w.s0; c0 += convert_float8(B1) * w.s1; c0 += convert_float8(B2) * w.s2; c0 += convert_float8(B3) * w.s3;
    c1 += convert_float8(B0) * w.s4; c1 += convert_float8(B1) * w.s5; c1 += convert_float8(B2) * w.s6; c1 += convert_float8(B3) * w.s7;
    c2 += convert_float8(B0) * w.s8; c2 += convert_float8(B1) * w.s9; c2 += convert_float8(B2) * w.sa; c2 += convert_float8(B3) * w.sb;
    c3 += convert_float8(B0) * w.sc; c3 += convert_float8(B1) * w.sd; c3 += convert_float8(B2) * w.se; c3 += convert_float8(B3) * w.sf;
  }

  // epilogue: scale + causal mask + clamp, store query-major S[q,k].
  // n = query (gx_4+nn), m = key (gy_8+mm). q_pos = (Skv-Sq)+q.
  #define EMIT(NN, CV) {                                                   \
    const int q = gx_4 + (NN);                                            \
    if (q < N) {                                                          \
      const int q_pos = (Skv - Sq) + q;                                  \
      float8 v = (CV) * scale;                                           \
      half8 hv;                                                          \
      hv.s0 = (half)( (causal && (gy_8+0) > q_pos) ? -INFINITY : clamp(v.s0,-QK_CLAMP,QK_CLAMP) ); \
      hv.s1 = (half)( (causal && (gy_8+1) > q_pos) ? -INFINITY : clamp(v.s1,-QK_CLAMP,QK_CLAMP) ); \
      hv.s2 = (half)( (causal && (gy_8+2) > q_pos) ? -INFINITY : clamp(v.s2,-QK_CLAMP,QK_CLAMP) ); \
      hv.s3 = (half)( (causal && (gy_8+3) > q_pos) ? -INFINITY : clamp(v.s3,-QK_CLAMP,QK_CLAMP) ); \
      hv.s4 = (half)( (causal && (gy_8+4) > q_pos) ? -INFINITY : clamp(v.s4,-QK_CLAMP,QK_CLAMP) ); \
      hv.s5 = (half)( (causal && (gy_8+5) > q_pos) ? -INFINITY : clamp(v.s5,-QK_CLAMP,QK_CLAMP) ); \
      hv.s6 = (half)( (causal && (gy_8+6) > q_pos) ? -INFINITY : clamp(v.s6,-QK_CLAMP,QK_CLAMP) ); \
      hv.s7 = (half)( (causal && (gy_8+7) > q_pos) ? -INFINITY : clamp(v.s7,-QK_CLAMP,QK_CLAMP) ); \
      vstore8(hv, 0, Sh + (long)q * Skv + gy_8);                         \
    }                                                                     \
  }
  EMIT(0, c0); EMIT(1, c1); EMIT(2, c2); EMIT(3, c3);
  #undef EMIT
}

// ---- softmax_norm: per (h,q) row of S, normalize IN PLACE -> P query-major.
// One WORKGROUP of SM_LW lanes per (q,h) row: lanes stride over k (coalesced
// reads), LDS-reduce row max then sum-of-exp, write P back into S[q,k] in place
// (same query-major layout -> pv_gemm reads it directly, no transpose/pack).
// global=(SM_LW, Sq, H), local=(SM_LW,1,1).
#ifndef SM_LW
#define SM_LW 64
#endif
// Last causal key index for query row q (= q_pos), clamped to Skv-1.
inline int causalRow(int Sq, int Skv, int q) { return min(Skv - 1, (Skv - Sq) + q); }
__kernel void softmax_norm(__global half* S, const int Sq, const int Skv, const int H) {
  const int t = get_local_id(0);
  const int q = get_global_id(1);
  const int h = get_global_id(2);
  if (q >= Sq) return;
  __global half* Sh = S + (long)h * Sq * Skv + (long)q * Skv;

  __local float red[SM_LW];

  // Causal upper bound: keys k > q_pos are masked (-inf), so passes 1-2 only
  // scan k <= q_pos (the lower triangle). Pass 3 still writes the FULL row
  // (zeros above q_pos) because pv_gemm reads P up to its tile's max q_pos.
  const int kcap = causalRow(Sq, Skv, q);

  // pass 1: row max over the causal keys.
  float m = -INFINITY;
  for (int k = t; k <= kcap; k += SM_LW) { float s = (float)Sh[k]; if (!isinf(s)) m = fmax(m, s); }
  red[t] = m; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = SM_LW >> 1; o > 0; o >>= 1) { if (t < o) red[t] = fmax(red[t], red[t + o]); barrier(CLK_LOCAL_MEM_FENCE); }
  m = red[0]; barrier(CLK_LOCAL_MEM_FENCE);

  // pass 2: sum of exp(s - m) over the causal keys.
  float l = 0.0f;
  if (!isinf(m)) for (int k = t; k <= kcap; k += SM_LW) { float s = (float)Sh[k]; l += isinf(s) ? 0.0f : native_exp(s - m); }
  red[t] = l; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = SM_LW >> 1; o > 0; o >>= 1) { if (t < o) red[t] = red[t] + red[t + o]; barrier(CLK_LOCAL_MEM_FENCE); }
  l = red[0];
  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;

  // pass 3: write P = exp(s - m) * inv back into S[q,k] (in place, coalesced).
  for (int k = t; k < Skv; k += SM_LW) {
    float s = (float)Sh[k];
    float p = (isinf(m) || isinf(s)) ? 0.0f : native_exp(s - m) * inv;
    Sh[k] = (half)p;
  }
}

// ---- pv_gemm: O[q,d] = sum_k P[q,k] V[k,d].
// A = V image [R=k, M=d=128]; B = P query-major [Sq, Skv] read DIRECTLY (no
// transpose/pack): for the 4 queries of this n-tile, 4x vload4 over contiguous
// keys i..i+3. reduction k=Skv. dst[n*M+m] = O[q*128 + d].
// global=(128/8, Sq/4, H).
__kernel void pv_gemm(__read_only image1d_buffer_t V_img,
                      __global const half* P, __global half* O,
                      const int Sq, const int Skv, const int H, const int causal) {
  const int gy = get_global_id(0);   // M-tile: 8 d
  const int gx = get_global_id(1);   // N-tile: 4 queries
  const int h  = get_global_id(2);
  const int M  = FA_D, N = Sq;
  if (gy * 8 >= M || gx * 4 >= N) return;
  const int gx_4 = gx << 2;
  const int gy_8 = gy << 3;
  const int M_4  = M >> 2;            // 32

  // Causal reduction bound: P[q,k]=0 for k > q_pos (softmax wrote full-width
  // zeros), so only sum keys up to the tile's max q_pos (rounded up to 4).
  int kmax = Skv;
  if (causal) { const int qpm = (Skv - Sq) + gx_4 + 3; kmax = min(Skv, (qpm + 4) & ~3); }

  const int v_tex_base = h * Skv * (FA_D >> 2);            // half4 texels of V[h]
  __global const half* Ph = P + (long)h * (long)Sq * Skv;  // query-major [Sq,Skv]
  __global half* Oh = O + (long)h * (long)Sq * FA_D;
  __global const half* p0r = Ph + (long)(gx_4 + 0) * Skv;
  __global const half* p1r = Ph + (long)(gx_4 + 1) * Skv;
  __global const half* p2r = Ph + (long)(gx_4 + 2) * Skv;
  __global const half* p3r = Ph + (long)(gx_4 + 3) * Skv;

  float8 c0 = (float8)0, c1 = (float8)0, c2 = (float8)0, c3 = (float8)0;
  half8 B0, B1, B2, B3;
  for (int i = 0; i < kmax; i += 4) {
    const int t = v_tex_base + gy * 2 + i * M_4;
    B0.s0123 = read_imageh(V_img, t);        B0.s4567 = read_imageh(V_img, t + 1);
    const int t1 = v_tex_base + gy * 2 + (i + 1) * M_4;
    B1.s0123 = read_imageh(V_img, t1);       B1.s4567 = read_imageh(V_img, t1 + 1);
    const int t2 = v_tex_base + gy * 2 + (i + 2) * M_4;
    B2.s0123 = read_imageh(V_img, t2);       B2.s4567 = read_imageh(V_img, t2 + 1);
    const int t3 = v_tex_base + gy * 2 + (i + 3) * M_4;
    B3.s0123 = read_imageh(V_img, t3);       B3.s4567 = read_imageh(V_img, t3 + 1);
    // 4 queries x 4 keys (i..i+3), one vload4 per query row.
    half4 p0 = vload4(0, p0r + i), p1 = vload4(0, p1r + i),
          p2 = vload4(0, p2r + i), p3 = vload4(0, p3r + i);
    const float8 b0 = convert_float8(B0), b1 = convert_float8(B1),
                 b2 = convert_float8(B2), b3 = convert_float8(B3);
    c0 += b0 * p0.s0; c0 += b1 * p0.s1; c0 += b2 * p0.s2; c0 += b3 * p0.s3;
    c1 += b0 * p1.s0; c1 += b1 * p1.s1; c1 += b2 * p1.s2; c1 += b3 * p1.s3;
    c2 += b0 * p2.s0; c2 += b1 * p2.s1; c2 += b2 * p2.s2; c2 += b3 * p2.s3;
    c3 += b0 * p3.s0; c3 += b1 * p3.s1; c3 += b2 * p3.s2; c3 += b3 * p3.s3;
  }

  #define EMITO(NN, CV) {                                                  \
    const int q = gx_4 + (NN);                                            \
    if (q < N) vstore8(convert_half8(CV), 0, Oh + (long)q * FA_D + gy_8); \
  }
  EMITO(0, c0); EMITO(1, c1); EMITO(2, c2); EMITO(3, c3);
  #undef EMITO
}
)CL"
