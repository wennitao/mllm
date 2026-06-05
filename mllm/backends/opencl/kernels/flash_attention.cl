#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// FlashAttention v1 (Dao et al., 2022), forward pass for inference.
// Same paper algorithm (K/V tiled in __local, online softmax merge with running
// (m_i, l_i, O_i) per Q row), but with a redesigned thread mapping for OpenCL:
//
//   local_size = FA_D       (= head_dim, e.g. 128 for Qwen3)
//   threads 0..(FA_BR*FA_BC - 1)   → compute S[i][c] = Q_i . K_c serially
//                                    (one thread per S element, NO tree reduce)
//   threads 0..(FA_D - 1)           → compute pv = sum_c P[i][c] * V_local[c][d]
//                                    (one thread per d-position; serial loop in c)
//
// This kills the per-S-element tree reduction barriers that dominated the
// previous implementation: ~3 barriers per j-iter total (load / S / end-of-j)
// instead of ~580.
//
// Layout: BHSD (all input tensors may be non-contiguous; strides passed in).
// Output O is contiguous BHSD.

#ifndef FA_BR
#define FA_BR 4
#endif
#ifndef FA_BC
#define FA_BC 16
#endif
#ifndef FA_D
#define FA_D 128
#endif

#define FA_S_THREADS (FA_BR * FA_BC) // = 64 active threads during S compute

// ---------------------------------------------------------------- fp32

__kernel void flash_attention_fp32(
    __global const float *Q, __global const float *K, __global const float *V,
    __global float *O,
    const int B, const int H, const int S_q, const int S_kv, const int D,
    const int Q_b_stride, const int Q_h_stride, const int Q_s_stride,
    const int K_b_stride, const int K_h_stride, const int K_s_stride,
    const int V_b_stride, const int V_h_stride, const int V_s_stride,
    const float scale, const int causal_mask) {
  const int t = get_local_id(0);          // 0..FA_D-1
  const int q_block = get_global_id(1);
  const int bh = get_global_id(2);
  if (bh >= B * H) return;
  const int q_row_start = q_block * FA_BR;
  if (q_row_start >= S_q) return;

  const int b = bh / H;
  const int h = bh - b * H;
  const int q_h_off = b * Q_b_stride + h * Q_h_stride;
  const int k_h_off = b * K_b_stride + h * K_h_stride;
  const int v_h_off = b * V_b_stride + h * V_h_stride;
  const int o_h_off = bh * S_q * D;

  __local float Q_local[FA_BR * FA_D];
  __local float K_local[FA_BC * FA_D];
  __local float V_local[FA_BC * FA_D];
  __local float S_local[FA_BR * FA_BC];
  __local float m_il[FA_BR];
  __local float l_il[FA_BR];

  // Load Q tile (B_r × D) — each thread (= d-position) loads B_r elements.
  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    Q_local[i * FA_D + t] =
        (q_row < S_q) ? Q[q_h_off + q_row * Q_s_stride + t] : 0.0f;
  }
  // Init per-Q-row running stats (only first FA_BR threads write).
  if (t < FA_BR) {
    m_il[t] = -INFINITY;
    l_il[t] = 0.0f;
  }
  // O accumulator in private registers: one float per Q row at this thread's d.
  float o_priv[FA_BR];
  for (int i = 0; i < FA_BR; ++i) o_priv[i] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  // Outer loop over K/V blocks (paper Algorithm 1, line 5).
  for (int j_start = 0; j_start < S_kv; j_start += FA_BC) {
    // Load K_j, V_j tiles cooperatively (each thread loads B_c elements).
    for (int c = 0; c < FA_BC; ++c) {
      const int k_row = j_start + c;
      if (k_row < S_kv) {
        K_local[c * FA_D + t] = K[k_h_off + k_row * K_s_stride + t];
        V_local[c * FA_D + t] = V[v_h_off + k_row * V_s_stride + t];
      } else {
        K_local[c * FA_D + t] = 0.0f;
        V_local[c * FA_D + t] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // S_ij = Q_i K_j^T. Each S element computed by ONE thread doing a serial
    // 128-element dot product in registers. NO tree reduction, NO per-element
    // barriers. Threads with t >= FA_S_THREADS idle this phase.
    if (t < FA_S_THREADS) {
      const int i = t / FA_BC;
      const int c = t - i * FA_BC;
      float acc = 0.0f;
      for (int d = 0; d < FA_D; ++d) {
        acc += Q_local[i * FA_D + d] * K_local[c * FA_D + d];
      }
      const int q_row = q_row_start + i;
      const int k_row = j_start + c;
      const int q_pos = S_kv - S_q + q_row;
      if (q_row >= S_q || k_row >= S_kv ||
          (causal_mask && k_row > q_pos)) {
        acc = -INFINITY;
      } else {
        acc *= scale;
      }
      S_local[i * FA_BC + c] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax merge per Q row. All threads compute the same scalars
    // (m_tilde, l_tilde, m_new, ...); pv is the only per-d quantity.
    for (int i = 0; i < FA_BR; ++i) {
      float m_tilde = -INFINITY;
      for (int c = 0; c < FA_BC; ++c) {
        m_tilde = fmax(m_tilde, S_local[i * FA_BC + c]);
      }
      float l_tilde = 0.0f;
      float pv = 0.0f;
      for (int c = 0; c < FA_BC; ++c) {
        const float s = S_local[i * FA_BC + c];
        const float p = (s == -INFINITY) ? 0.0f : exp(s - m_tilde);
        l_tilde += p;
        pv += p * V_local[c * FA_D + t];
      }
      const float m_old = m_il[i];
      const float l_old = l_il[i];
      const float m_new = fmax(m_old, m_tilde);
      const float a = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
      const float bb = (m_tilde == -INFINITY) ? 0.0f : exp(m_tilde - m_new);
      const float l_new = a * l_old + bb * l_tilde;
      o_priv[i] =
          (l_new > 0.0f) ? (l_old * a * o_priv[i] + bb * pv) / l_new : 0.0f;
      if (t == 0) {
        m_il[i] = m_new;
        l_il[i] = l_new;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write O (one float per (q_row, d) for this thread).
  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    if (q_row < S_q) { O[o_h_off + q_row * D + t] = o_priv[i]; }
  }
}

// ---------------------------------------------------------------- fp16 (optimized)
//
// Optimized vs the fp32 reference above, applying lessons from the tuned OpenCL
// GEMM/GEMV kernels. The fp32 kernel is left untouched as the numeric reference.
//
//   FA_BR_H q-rows per workgroup (default 8 — the host must launch q_blocks =
//   ceil(S_q/FA_BR_H) for fp16); FA_BC_H = FA_D/4 keys per K/V tile. With
//   FA_BR_H=8, D=128 → FA_BC_H=32 and FA_BR_H*FA_BC_H = 256 = 2*FA_D, so each
//   of the 128 lanes computes TWO S elements in QK^T (two Q rows i0 and i0+4
//   sharing column c → the K[c] row is loaded once and reused → cross-q reuse,
//   and the per-j-iter load+barrier overhead is amortized over 2x the rows).
//
//   Other techniques: native-half LDS storage; transposed-V LDS tile for
//   contiguous vectorized P·V; hoisted producer-lane softmax (native_exp);
//   half8-vectorized QK^T dot; causal block-skip of fully-masked K/V blocks.
//
// PRECISION INVARIANT: storage + multiply operands are half; ALL accumulators
// (QK^T dot, P·V sum, o_priv across the whole S_kv loop) and softmax stats
// (m, l, a, bb, l_new) stay fp32 — they feed exp(). Requires FA_D % 4 == 0 and
// FA_BR_H*FA_BC_H == 2*FA_D (true for FA_BR_H=8, any D divisible by 4).

#ifndef FA_BR_H
#define FA_BR_H 8
#endif
#define FA_BC_H (FA_D / 4)                  // 32 for D=128
#define FA_ROWSTEP (FA_D / FA_BC_H)         // 4: i1 = i0 + FA_ROWSTEP

__kernel void flash_attention_fp16(
    __global const half *Q, __global const half *K, __global const half *V,
    __global half *O,
    const int B, const int H, const int S_q, const int S_kv, const int D,
    const int Q_b_stride, const int Q_h_stride, const int Q_s_stride,
    const int K_b_stride, const int K_h_stride, const int K_s_stride,
    const int V_b_stride, const int V_h_stride, const int V_s_stride,
    const float scale, const int causal_mask) {
  const int t = get_local_id(0);            // 0..FA_D-1
  const int q_block = get_global_id(1);
  const int bh = get_global_id(2);
  if (bh >= B * H) return;
  const int q_row_start = q_block * FA_BR_H;
  if (q_row_start >= S_q) return;
  // Valid rows in this workgroup (< FA_BR_H only for the last block / decode).
  // A uniform loop bound (not a per-iter `continue`) keeps the compiler's loop
  // optimization for full blocks while letting decode (S_q=1) skip the waste.
  const int nrows = min(FA_BR_H, S_q - q_row_start);

  const int b = bh / H;
  const int h = bh - b * H;
  const int q_h_off = b * Q_b_stride + h * Q_h_stride;
  const int k_h_off = b * K_b_stride + h * K_h_stride;
  const int v_h_off = b * V_b_stride + h * V_h_stride;
  const int o_h_off = bh * S_q * D;

  __local half  Q_local[FA_BR_H * FA_D];
  __local half  K_local[FA_BC_H * FA_D];     // c-major [c*D + d] for the QK^T dot
  __local half  V_localT[FA_D * FA_BC_H];    // TRANSPOSED [d*FA_BC_H + c] for P·V
  __local float S_local[FA_BR_H * FA_BC_H];  // scores (fp32)
  __local float P_local[FA_BR_H * FA_BC_H];  // exp probs (fp32)
  __local float m_il[FA_BR_H];
  __local float l_il[FA_BR_H];
  __local float sh_a[FA_BR_H];
  __local float sh_bb[FA_BR_H];
  __local float sh_lold[FA_BR_H];
  __local float sh_lnew[FA_BR_H];

  for (int i = 0; i < FA_BR_H; ++i) {
    const int q_row = q_row_start + i;
    Q_local[i * FA_D + t] =
        (q_row < S_q) ? Q[q_h_off + q_row * Q_s_stride + t] : (half)0;
  }
  if (t < FA_BR_H) {
    m_il[t] = -INFINITY;
    l_il[t] = 0.0f;
  }
  float o_priv[FA_BR_H];
  for (int i = 0; i < FA_BR_H; ++i) o_priv[i] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  // Causal block-skip: cap the j-loop at this q_block's last diagonal block.
  int j_max = S_kv;
  if (causal_mask) {
    const int last_q_pos = S_kv - S_q + (q_row_start + FA_BR_H - 1);
    j_max = min(S_kv, last_q_pos + 1);
  }

  for (int j_start = 0; j_start < j_max; j_start += FA_BC_H) {
    for (int c = 0; c < FA_BC_H; ++c) {
      const int k_row = j_start + c;
      if (k_row < S_kv) {
        K_local[c * FA_D + t] = K[k_h_off + k_row * K_s_stride + t];
        V_localT[t * FA_BC_H + c] = V[v_h_off + k_row * V_s_stride + t];
      } else {
        K_local[c * FA_D + t] = (half)0;
        V_localT[t * FA_BC_H + c] = (half)0;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // QK^T — each of the 128 lanes computes TWO S elements: rows i0 and
    // i0+FA_ROWSTEP at the same column c, sharing the K[c] row load. Invalid
    // (masked / out-of-range) rows are gated out so partial blocks (e.g. decode
    // S_q=1 under FA_BR_H=8) don't pay for the wasted rows.
    {
      const int c = t % FA_BC_H;
      const int i0 = t / FA_BC_H;
      const int i1 = i0 + FA_ROWSTEP;
      __local const half* krow = K_local + c * FA_D;
      __local const half* q0 = Q_local + i0 * FA_D;
      __local const half* q1 = Q_local + i1 * FA_D;
      float8 a0 = (float8)(0.0f);
      float8 a1 = (float8)(0.0f);
      for (int d8 = 0; d8 < FA_D / 8; ++d8) {
        float8 kf = convert_float8(vload8(d8, krow));
        a0 += convert_float8(vload8(d8, q0)) * kf;
        a1 += convert_float8(vload8(d8, q1)) * kf;
      }
      float acc0 = a0.s0 + a0.s1 + a0.s2 + a0.s3 + a0.s4 + a0.s5 + a0.s6 + a0.s7;
      float acc1 = a1.s0 + a1.s1 + a1.s2 + a1.s3 + a1.s4 + a1.s5 + a1.s6 + a1.s7;
      const int k_row = j_start + c;
      const int qr0 = q_row_start + i0;
      const int qr1 = q_row_start + i1;
      const int qp0 = S_kv - S_q + qr0;
      const int qp1 = S_kv - S_q + qr1;
      S_local[i0 * FA_BC_H + c] =
          (qr0 >= S_q || k_row >= S_kv || (causal_mask && k_row > qp0)) ? -INFINITY : acc0 * scale;
      S_local[i1 * FA_BC_H + c] =
          (qr1 >= S_q || k_row >= S_kv || (causal_mask && k_row > qp1)) ? -INFINITY : acc1 * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax — hoisted: one producer lane per valid Q row.
    if (t < nrows) {
      const int i = t;
      float m_tilde = -INFINITY;
      for (int c = 0; c < FA_BC_H; ++c) {
        m_tilde = fmax(m_tilde, S_local[i * FA_BC_H + c]);
      }
      float l_tilde = 0.0f;
      for (int c = 0; c < FA_BC_H; ++c) {
        const float s = S_local[i * FA_BC_H + c];
        const float p = (s == -INFINITY) ? 0.0f : native_exp(s - m_tilde);
        P_local[i * FA_BC_H + c] = p;
        l_tilde += p;
      }
      const float m_old = m_il[i];
      const float l_old = l_il[i];
      const float m_new = fmax(m_old, m_tilde);
      const float a = (m_old == -INFINITY) ? 0.0f : native_exp(m_old - m_new);
      const float bb = (m_tilde == -INFINITY) ? 0.0f : native_exp(m_tilde - m_new);
      const float l_new = a * l_old + bb * l_tilde;
      sh_a[i] = a;
      sh_bb[i] = bb;
      sh_lold[i] = l_old;
      sh_lnew[i] = l_new;
      m_il[i] = m_new;
      l_il[i] = l_new;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // P·V + O rescale — ALL lanes: lane t owns d=t; FA_BR_H rows, vectorized in c.
    {
      __local const half* vrow = V_localT + t * FA_BC_H;
      for (int i = 0; i < nrows; ++i) {
        __local const float* prow = P_local + i * FA_BC_H;
        float8 pv8 = (float8)(0.0f);
        for (int c8 = 0; c8 < FA_BC_H / 8; ++c8) {
          pv8 += vload8(c8, prow) * convert_float8(vload8(c8, vrow));
        }
        const float pv = pv8.s0 + pv8.s1 + pv8.s2 + pv8.s3 +
                         pv8.s4 + pv8.s5 + pv8.s6 + pv8.s7;
        const float a = sh_a[i], bb = sh_bb[i];
        const float l_old = sh_lold[i], l_new = sh_lnew[i];
        o_priv[i] =
            (l_new > 0.0f) ? (l_old * a * o_priv[i] + bb * pv) / l_new : 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i = 0; i < nrows; ++i) {
    O[o_h_off + (q_row_start + i) * D + t] = (half)o_priv[i];
  }
}
