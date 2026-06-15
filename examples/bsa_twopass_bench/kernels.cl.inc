// Block-sparse two-pass GEMM-class prefill kernels (Adreno 830, fp16/fp32-acc).
// Combines two ideas:
//   - index-driven (no gather): K/V are transposed/copied to FULL contiguous
//     images ONCE (dense pack_q/trans_k/copy_v, reused verbatim); the QK and PV
//     GEMMs read those images at the SELECTED block offsets via block_idx, so
//     there is no per-query-block gather replication.
//   - image-A GEMM: the validated variant-C 8x4 image-A / packed-B GEMM, so the
//     selected-key matmul runs at GEMM-class throughput (~1 TF dense lever).
//
// Selection: block_idx[(bh*num_qb + qb)*top_k + slot] = key-block id (units of
// BK keys) selected for query block qb of head bh, or < 0 for a padding slot.
// Must include the diagonal block per query (>=1 causal key) and hold DISTINCT
// ids. Causal/padding masking is per-element in the QK epilogue (k_abs > q_pos
// or blk_id < 0 -> -inf), so correctness mirrors the dense kernel.
//
// Layouts (fp16): Q[bh,q,d]; Kt[bh,d,Skv] (image); Vc[bh,Skv,d] (image);
// Qp[bh,D/4,Sq,4]; S[bhqb, BQ, sel] (sel = top_k*BK); O[bh,Sq,D].
// Compile-time: FA_D (128). BQ (query block), BK (key block), top_k passed in.

R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FA_D
#define FA_D 128
#endif
#ifndef BSA_CLAMP
#define BSA_CLAMP 60000.0f
#endif
#ifndef BSA_SM_LW
#define BSA_SM_LW 64
#endif

// ---- pack_q: strided Q[b,h,q,:] -> contiguous Qp[bh][D/4, Sq, 4]. (same as dense)
__kernel void pack_q(__global const half* Q, __global half* Qp,
                     const int Sq, const int H,
                     const int Qhs, const int Qss) {
  const int q = get_global_id(0);
  const int g = get_global_id(1);
  const int bh = get_global_id(2);
  if (q >= Sq || g >= FA_D / 4 || bh >= H) return;
  const int h = bh;
  __global const half* src = Q + (long)h * Qhs + (long)q * Qss;
  __global half* dst = Qp + (long)bh * (FA_D / 4) * Sq * 4;
  const int sd = g * 4;
  #pragma unroll
  for (int kk = 0; kk < 4; ++kk) dst[((long)g * Sq + q) * 4 + kk] = src[sd + kk];
}

// ---- trans_k: K[h,k,:] -> Kt[bh][FA_D, Skv]. (same as dense)
__kernel void trans_k(__global const half* K, __global half* Kt,
                      const int Skv, const int H, const int Khs, const int Kss) {
  const int k = get_global_id(0);
  const int d = get_global_id(1);
  const int bh = get_global_id(2);
  if (k >= Skv || d >= FA_D || bh >= H) return;
  Kt[(long)bh * FA_D * Skv + (long)d * Skv + k] = K[(long)bh * Khs + (long)k * Kss + d];
}

// ---- copy_v: V[h,k,:] -> Vc[bh][Skv, FA_D]. Vectorized half8 over the
// contiguous d (D-stride 1), coalesced. global=(FA_D/8, Skv, H).
__kernel void copy_v(__global const half* V, __global half* Vc,
                     const int Skv, const int H, const int Vhs, const int Vss) {
  const int dg = get_global_id(0);   // d-group of 8
  const int k = get_global_id(1);
  const int bh = get_global_id(2);
  if (dg >= FA_D / 8 || k >= Skv || bh >= H) return;
  half8 v = vload8(0, V + (long)bh * Vhs + (long)k * Vss + dg * 8);
  vstore8(v, 0, Vc + (long)bh * Skv * FA_D + (long)k * FA_D + dg * 8);
}

// ---- bs_qk_gemm: S[q_local, slot] = scale * Q[q].K[selected] + causal mask.
// A = Kt image read at SELECTED block offsets; B = Qp packed. M = sel (8 slots/
// tile), N = BQ (4 queries/tile), reduction d=128. global=(sel/8, BQ/4, BH*num_qb).
__kernel void bs_qk_gemm(__read_only image1d_buffer_t Kt_img,
                         __global const half* Qp, __global half* S,
                         __global const int* block_idx,
                         const int Sq, const int Skv, const int H,
                         const int num_qb, const int top_k, const int BK, const int BQ,
                         const float scale, const int causal) {
  const int gy = get_global_id(0);   // M-tile: 8 slots
  const int gx = get_global_id(1);   // N-tile: 4 queries (local in block)
  const int bhqb = get_global_id(2);
  const int sel = top_k * BK;
  if (gy * 8 >= sel || gx * 4 >= BQ || bhqb >= H * num_qb) return;
  const int bh = bhqb / num_qb, qb = bhqb - bh * num_qb;
  const int gy_8 = gy << 3, gx_4 = gx << 2;
  const int M_4kv = Skv >> 2;

  // selected block for this M-tile (8 contiguous slots within one BK block).
  const int blk = gy_8 / BK;
  const int key0 = gy_8 - blk * BK;
  const int blk_id = block_idx[bhqb * top_k + blk];
  const int kg0 = (blk_id < 0) ? 0 : (blk_id * BK + key0);   // k_global base
  const int kt_base = bh * FA_D * M_4kv + (kg0 >> 2);         // texel base, d=0

  __global const half* Qph = Qp + (long)bh * (FA_D / 4) * Sq * 4;
  __global half* Sh = S + (long)bhqb * BQ * sel;

  float8 c0 = (float8)0, c1 = (float8)0, c2 = (float8)0, c3 = (float8)0;
  half8 B0, B1, B2, B3;
  for (int i = 0; i < FA_D; i += 4) {
    const int t = kt_base + i * M_4kv;
    B0.s0123 = read_imageh(Kt_img, t);       B0.s4567 = read_imageh(Kt_img, t + 1);
    const int t1 = kt_base + (i + 1) * M_4kv;
    B1.s0123 = read_imageh(Kt_img, t1);      B1.s4567 = read_imageh(Kt_img, t1 + 1);
    const int t2 = kt_base + (i + 2) * M_4kv;
    B2.s0123 = read_imageh(Kt_img, t2);      B2.s4567 = read_imageh(Kt_img, t2 + 1);
    const int t3 = kt_base + (i + 3) * M_4kv;
    B3.s0123 = read_imageh(Kt_img, t3);      B3.s4567 = read_imageh(Kt_img, t3 + 1);
    half16 w = vload16(0, Qph + ((long)(i >> 2) * Sq + qb * BQ + gx_4) * 4);
    const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
    c0 += b0*w.s0; c0 += b1*w.s1; c0 += b2*w.s2; c0 += b3*w.s3;
    c1 += b0*w.s4; c1 += b1*w.s5; c1 += b2*w.s6; c1 += b3*w.s7;
    c2 += b0*w.s8; c2 += b1*w.s9; c2 += b2*w.sa; c2 += b3*w.sb;
    c3 += b0*w.sc; c3 += b1*w.sd; c3 += b2*w.se; c3 += b3*w.sf;
  }
  // epilogue: slot = gy_8+m -> k_global = blk_id*BK + key0 + m; mask if padding
  // or k_global > q_pos. store S[q_local, slot].
  #define EMIT(NN, CV) {                                                        \
    const int ql = gx_4 + (NN);                                               \
    if (ql < BQ) {                                                            \
      const int qg = qb * BQ + ql;                                           \
      const int q_pos = (Skv - Sq) + qg;                                     \
      float8 v = (CV) * scale; half8 hv;                                     \
      hv.s0 = (half)((blk_id<0 || (causal && (kg0+0) > q_pos)) ? -INFINITY : clamp(v.s0,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s1 = (half)((blk_id<0 || (causal && (kg0+1) > q_pos)) ? -INFINITY : clamp(v.s1,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s2 = (half)((blk_id<0 || (causal && (kg0+2) > q_pos)) ? -INFINITY : clamp(v.s2,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s3 = (half)((blk_id<0 || (causal && (kg0+3) > q_pos)) ? -INFINITY : clamp(v.s3,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s4 = (half)((blk_id<0 || (causal && (kg0+4) > q_pos)) ? -INFINITY : clamp(v.s4,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s5 = (half)((blk_id<0 || (causal && (kg0+5) > q_pos)) ? -INFINITY : clamp(v.s5,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s6 = (half)((blk_id<0 || (causal && (kg0+6) > q_pos)) ? -INFINITY : clamp(v.s6,-BSA_CLAMP,BSA_CLAMP)); \
      hv.s7 = (half)((blk_id<0 || (causal && (kg0+7) > q_pos)) ? -INFINITY : clamp(v.s7,-BSA_CLAMP,BSA_CLAMP)); \
      vstore8(hv, 0, Sh + (long)ql * sel + gy_8);                            \
    } }
  EMIT(0, c0); EMIT(1, c1); EMIT(2, c2); EMIT(3, c3);
  #undef EMIT
}

// ---- bs_softmax: per (bhqb, q_local) over sel slots, normalize in place.
// global=(BSA_SM_LW, BQ, BH*num_qb), local=(BSA_SM_LW,1,1).
__kernel void bs_softmax(__global half* S, const int H, const int num_qb,
                         const int top_k, const int BK, const int BQ) {
  const int t = get_local_id(0);
  const int ql = get_global_id(1);
  const int bhqb = get_global_id(2);
  const int sel = top_k * BK;
  if (ql >= BQ || bhqb >= H * num_qb) return;
  __global half* Sh = S + (long)bhqb * BQ * sel + (long)ql * sel;
  __local float redm[BSA_SM_LW];
  __local float redl[BSA_SM_LW];

  float m = -INFINITY, l = 0.0f;
  for (int k = t; k < sel; k += BSA_SM_LW) {
    float s = (float)Sh[k];
    if (!isinf(s)) { float mn = fmax(m, s); l = l * native_exp(m - mn) + native_exp(s - mn); m = mn; }
  }
  redm[t] = m; redl[t] = l; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = BSA_SM_LW >> 1; o > 0; o >>= 1) {
    if (t < o) {
      float ma = redm[t], mb = redm[t + o], la = redl[t], lb = redl[t + o];
      float mn = fmax(ma, mb);
      redl[t] = (isinf(mn) ? 0.0f : la * native_exp(ma - mn) + lb * native_exp(mb - mn));
      redm[t] = mn;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  m = redm[0]; l = redl[0];
  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int k = t; k < sel; k += BSA_SM_LW) {
    float s = (float)Sh[k];
    Sh[k] = (half)((isinf(m) || isinf(s)) ? 0.0f : native_exp(s - m) * inv);
  }
}

// ---- bs_pv_gemm: O[q,d] = sum_slot P[q,slot] V[selected]. A = Vc image read at
// selected offsets; B = P slot-major direct (4x vload4). M=d=128, N=BQ, reduction
// = sel. global=(FA_D/8, BQ/4, BH*num_qb). O[bh, Sq, D].
__kernel void bs_pv_gemm(__read_only image1d_buffer_t V_img,
                         __global const half* P, __global half* O,
                         __global const int* block_idx,
                         const int Sq, const int Skv, const int H,
                         const int num_qb, const int top_k, const int BK, const int BQ) {
  const int gy = get_global_id(0);   // M-tile: 8 d
  const int gx = get_global_id(1);   // N-tile: 4 queries
  const int bhqb = get_global_id(2);
  const int sel = top_k * BK;
  if (gy * 8 >= FA_D || gx * 4 >= BQ || bhqb >= H * num_qb) return;
  const int bh = bhqb / num_qb, qb = bhqb - bh * num_qb;
  const int gy_8 = gy << 3, gx_4 = gx << 2;
  const int M_4 = FA_D >> 2;  // 32

  const int v_head = bh * Skv * (FA_D >> 2);   // texel base of Vc[bh]
  __global const half* Ph = P + (long)bhqb * BQ * sel;
  __global half* Oh = O + (long)bh * (long)Sq * FA_D;
  __global const half* p0r = Ph + (long)(gx_4 + 0) * sel;
  __global const half* p1r = Ph + (long)(gx_4 + 1) * sel;
  __global const half* p2r = Ph + (long)(gx_4 + 2) * sel;
  __global const half* p3r = Ph + (long)(gx_4 + 3) * sel;

  float8 c0 = (float8)0, c1 = (float8)0, c2 = (float8)0, c3 = (float8)0;
  half8 B0, B1, B2, B3;
  // Outer loop over SELECTED blocks (block_idx read once/block, padding skipped:
  // its P is 0 anyway), inner over 4-key sub-tiles within the BK block.
  for (int blk = 0; blk < top_k; ++blk) {
    const int blk_id = block_idx[bhqb * top_k + blk];
    if (blk_id < 0) continue;
    const int kbase = blk_id * BK;
    const int sbase = blk * BK;
    for (int sub = 0; sub < BK; sub += 4) {
      const int s = sbase + sub;
      const int kg = kbase + sub;
      const int vt = v_head + kg * M_4 + gy * 2;
      B0.s0123 = read_imageh(V_img, vt);          B0.s4567 = read_imageh(V_img, vt + 1);
      const int vt1 = v_head + (kg + 1) * M_4 + gy * 2;
      B1.s0123 = read_imageh(V_img, vt1);         B1.s4567 = read_imageh(V_img, vt1 + 1);
      const int vt2 = v_head + (kg + 2) * M_4 + gy * 2;
      B2.s0123 = read_imageh(V_img, vt2);         B2.s4567 = read_imageh(V_img, vt2 + 1);
      const int vt3 = v_head + (kg + 3) * M_4 + gy * 2;
      B3.s0123 = read_imageh(V_img, vt3);         B3.s4567 = read_imageh(V_img, vt3 + 1);
      half4 p0 = vload4(0, p0r + s), p1 = vload4(0, p1r + s), p2 = vload4(0, p2r + s), p3 = vload4(0, p3r + s);
      const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
      c0 += b0*p0.s0; c0 += b1*p0.s1; c0 += b2*p0.s2; c0 += b3*p0.s3;
      c1 += b0*p1.s0; c1 += b1*p1.s1; c1 += b2*p1.s2; c1 += b3*p1.s3;
      c2 += b0*p2.s0; c2 += b1*p2.s1; c2 += b2*p2.s2; c2 += b3*p2.s3;
      c3 += b0*p3.s0; c3 += b1*p3.s1; c3 += b2*p3.s2; c3 += b3*p3.s3;
    }
  }
  #define EMITO(NN, CV) { const int ql = gx_4 + (NN); if (ql < BQ) { const int qg = qb * BQ + ql; vstore8(convert_half8(CV), 0, Oh + (long)qg * FA_D + gy_8); } }
  EMITO(0, c0); EMITO(1, c1); EMITO(2, c2); EMITO(3, c3);
  #undef EMITO
}
)CL"
