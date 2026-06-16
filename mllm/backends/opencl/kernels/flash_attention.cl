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
#define FA_ROWSTEP (FA_D / FA_BC_H)         // 4: rows handled by a lane are i0 + s*ROWSTEP
#define FA_NSPL (FA_BR_H * FA_BC_H / FA_D)  // S elements per lane (2 for BR=8, 4 for BR=16)

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
      const int k_row = j_start + c;
      __local const half* krow = K_local + c * FA_D;
      float8 acc[FA_NSPL];
      #pragma unroll
      for (int s = 0; s < FA_NSPL; ++s) acc[s] = (float8)(0.0f);
      for (int d8 = 0; d8 < FA_D / 8; ++d8) {
        float8 kf = convert_float8(vload8(d8, krow));   // shared K row across rows
        #pragma unroll
        for (int s = 0; s < FA_NSPL; ++s) {
          acc[s] += convert_float8(vload8(d8, Q_local + (i0 + s * FA_ROWSTEP) * FA_D)) * kf;
        }
      }
      #pragma unroll
      for (int s = 0; s < FA_NSPL; ++s) {
        const int i = i0 + s * FA_ROWSTEP;
        const int qr = q_row_start + i;
        const int qp = S_kv - S_q + qr;
        const float8 a = acc[s];
        const float v = a.s0 + a.s1 + a.s2 + a.s3 + a.s4 + a.s5 + a.s6 + a.s7;
        S_local[i * FA_BC_H + c] =
            (qr >= S_q || k_row >= S_kv || (causal_mask && k_row > qp)) ? -INFINITY : v * scale;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Online softmax — hoisted: one producer lane per valid Q row. Vectorized
    // max / exp / sum over c (native_exp on float8). Masked entries are -INF;
    // select() forces their prob to 0 (don't trust native_exp(-INF)).
    if (t < nrows) {
      const int i = t;
      __local float* prow = S_local + i * FA_BC_H;  // aliased: scores in, probs out
      float8 m8 = (float8)(-INFINITY);
      for (int c8 = 0; c8 < FA_BC_H / 8; ++c8) m8 = fmax(m8, vload8(c8, prow));
      float4 m4 = fmax(m8.lo, m8.hi);
      float2 m2 = fmax(m4.lo, m4.hi);
      float m_tilde = fmax(m2.s0, m2.s1);
      float8 l8 = (float8)(0.0f);
      for (int c8 = 0; c8 < FA_BC_H / 8; ++c8) {
        const float8 s8 = vload8(c8, prow);
        const float8 p8 = select(native_exp(s8 - m_tilde), (float8)(0.0f), isinf(s8));
        vstore8(p8, c8, prow);
        l8 += p8;
      }
      float4 l4 = l8.lo + l8.hi;
      float2 l2 = l4.lo + l4.hi;
      float l_tilde = l2.s0 + l2.s1;
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
        __local const float* prow = S_local + i * FA_BC_H;
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

// ---------------------------------------------------------------- fp16 decode (S_q == 1)
//
// Dedicated decode kernel: at S_q=1 each K/V element is used exactly ONCE, so
// the prefill kernel's global->LDS K/V staging is pure overhead and its
// (q-rows x keys) lane mapping wastes 3/4 of the QK lanes. This kernel streams
// K/V straight from global with no K/V LDS tiles:
//
//   score phase: lane t owns key row (bs + t)  -> full D-dot in registers, no
//                cross-lane reduction. Across a wave the row base addresses
//                stride by D*2 bytes, but each lane walks its row sequentially
//                (vload8), so every touched cache line is fully consumed --
//                the L1/UCHE absorbs the interleave (verified GEMV-style).
//   softmax:     one producer lane per block, float8-vectorized, online (m,l)
//                with the rescale factor a broadcast via LDS.
//   P*V phase:   lane t owns output dim d=t; V rows read straight from global,
//                coalesced across lanes; probs broadcast from LDS.
//
// 3 barriers per FA_D-key block (vs 4 per FA_BC_H=32 keys in the prefill
// kernel = ~5x fewer per key), zero wasted dot lanes.
//
// Split-K (flash-decoding): grid dim1 = nsplit partitions over S_kv. Each
// workgroup writes an UNNORMALIZED partial (o_acc scaled to its running max,
// plus m, l) to `partial` [B*H, nsplit, FA_D+2] (float); the merge kernel
// below combines partitions. With nsplit == 1 the kernel normalizes and
// writes O directly (no scratch, no merge launch).
//
// S_q==1 makes the causal mask a no-op (every key index <= S_kv-1 = q_pos),
// so no mask test is needed. Accumulators and softmax stats stay fp32.

__kernel void flash_attention_fp16_decode(
    __global const half *Q, __global const half *K, __global const half *V,
    __global half *O, __global float *partial,
    const int B, const int H, const int S_kv, const int D,
    const int Q_b_stride, const int Q_h_stride,
    const int K_b_stride, const int K_h_stride, const int K_s_stride,
    const int V_b_stride, const int V_h_stride, const int V_s_stride,
    const float scale, const int nsplit, const int span) {
  const int t = get_local_id(0);   // 0..FA_D-1
  const int split = get_global_id(1);
  const int bh = get_global_id(2);
  if (bh >= B * H) return;

  const int b = bh / H;
  const int h = bh - b * H;
  const int q_off = b * Q_b_stride + h * Q_h_stride;
  const int k_h_off = b * K_b_stride + h * K_h_stride;
  const int v_h_off = b * V_b_stride + h * V_h_stride;

  const int kv_begin = split * span;
  const int kv_end = min(S_kv, kv_begin + span);
  const int pbase = (bh * nsplit + split) * (FA_D + 2);

  if (kv_begin >= kv_end) {  // empty partition: still write a neutral partial
    if (nsplit > 1) {
      partial[pbase + t] = 0.0f;
      if (t == 0) {
        partial[pbase + FA_D] = -INFINITY;
        partial[pbase + FA_D + 1] = 0.0f;
      }
    }
    return;
  }

  __local half  Q_l[FA_D];
  __local float S_l[FA_D];     // one block of scores -> probs (in place)
  __local float sh_a;          // per-block rescale factor exp(m_old - m_new)

  Q_l[t] = Q[q_off + t];
  float o_acc = 0.0f;          // lane t's UNNORMALIZED output dim d=t
  float m_run = -INFINITY;     // tracked by lane 0 (authoritative)
  float l_run = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int bs = kv_begin; bs < kv_end; bs += FA_D) {
    const int blk_n = min(FA_D, kv_end - bs);

    // ---- scores: lane t -> key row bs+t (full-depth dot, fp32 accumulate)
    {
      float s = -INFINITY;
      if (t < blk_n) {
        __global const half *krow = K + k_h_off + (long)(bs + t) * K_s_stride;
        float8 acc = (float8)(0.0f);
        #pragma unroll
        for (int d8 = 0; d8 < FA_D / 8; ++d8) {
          acc += convert_float8(vload8(d8, Q_l)) * convert_float8(vload8(d8, krow));
        }
        s = (acc.s0 + acc.s1 + acc.s2 + acc.s3 +
             acc.s4 + acc.s5 + acc.s6 + acc.s7) * scale;
      }
      S_l[t] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- online softmax: producer lane 0, float8-vectorized over the block
    // (FA_DEC_PROF ablation: 1 = scores only, 2 = +softmax, 3/unset = full)
#if defined(FA_DEC_PROF) && FA_DEC_PROF < 2
    if (t == 0) { sh_a = S_l[t]; m_run = fmax(m_run, S_l[0]); l_run += 1.0f; }
#else
    if (t == 0) {
      float8 m8 = (float8)(-INFINITY);
      for (int c8 = 0; c8 < FA_D / 8; ++c8) m8 = fmax(m8, vload8(c8, S_l));
      float4 m4 = fmax(m8.lo, m8.hi);
      float2 m2 = fmax(m4.lo, m4.hi);
      const float m_tilde = fmax(m2.s0, m2.s1);
      const float m_new = fmax(m_run, m_tilde);
      const float a = (m_run == -INFINITY) ? 0.0f : native_exp(m_run - m_new);
      float8 l8 = (float8)(0.0f);
      for (int c8 = 0; c8 < FA_D / 8; ++c8) {
        const float8 s8 = vload8(c8, S_l);
        const float8 p8 = select(native_exp(s8 - m_new), (float8)(0.0f), isinf(s8));
        vstore8(p8, c8, S_l);
        l8 += p8;
      }
      float4 l4 = l8.lo + l8.hi;
      float2 l2 = l4.lo + l4.hi;
      l_run = a * l_run + (l2.s0 + l2.s1);
      m_run = m_new;
      sh_a = a;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- P*V: lane t = output dim t; V coalesced across lanes
#if defined(FA_DEC_PROF) && FA_DEC_PROF < 3
    o_acc += sh_a + S_l[t & (FA_D - 1)];  // keep prior phases live, skip PV
#else
    {
      const float a = sh_a;
      float pv = 0.0f;
      __global const half *vbase = V + v_h_off + (long)bs * V_s_stride + t;
      #pragma unroll 8
      for (int c = 0; c < blk_n; ++c) {
        pv += S_l[c] * (float)vbase[(long)c * V_s_stride];
      }
      o_acc = o_acc * a + pv;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);  // S_l reused next block
  }

  if (nsplit == 1) {
    // Single partition: normalize and write O directly (skip scratch+merge).
    // l_run is authoritative on lane 0 only -> broadcast via LDS (reuse sh_a).
    if (t == 0) sh_a = l_run;
    barrier(CLK_LOCAL_MEM_FENCE);
    const float l_all = sh_a;
    O[bh * FA_D + t] = (half)((l_all > 0.0f) ? o_acc / l_all : 0.0f);
  } else {
    partial[pbase + t] = o_acc;
    if (t == 0) {
      partial[pbase + FA_D] = m_run;
      partial[pbase + FA_D + 1] = l_run;
    }
  }
}

// Merge nsplit partials -> O. Dispatch: global (FA_D, 1, B*H), local (FA_D,1,1).
// partial layout: [B*H, nsplit, FA_D+2] float; slot FA_D = m, FA_D+1 = l.
// Standard flash-decoding merge: O = sum_s w_s*o_s / sum_s w_s*l_s with
// w_s = exp(m_s - max_s m_s); empty partitions (m = -inf) contribute 0.
__kernel void flash_attention_fp16_decode_merge(
    __global const float *partial, __global half *O,
    const int B, const int H, const int nsplit) {
  const int t = get_local_id(0);
  const int bh = get_global_id(2);
  if (bh >= B * H) return;
  const int base = bh * nsplit * (FA_D + 2);

  float m_max = -INFINITY;
  for (int s = 0; s < nsplit; ++s) {
    m_max = fmax(m_max, partial[base + s * (FA_D + 2) + FA_D]);
  }
  float num = 0.0f, den = 0.0f;
  for (int s = 0; s < nsplit; ++s) {
    const float m_s = partial[base + s * (FA_D + 2) + FA_D];
    if (m_s == -INFINITY) continue;
    const float w = native_exp(m_s - m_max);
    num += w * partial[base + s * (FA_D + 2) + t];
    den += w * partial[base + s * (FA_D + 2) + FA_D + 1];
  }
  O[bh * FA_D + t] = (half)((den > 0.0f) ? num / den : 0.0f);
}

// ============================================================================
// Two-pass GEMM-class PREFILL (S_q large, causal). Reuses the validated
// variant-C 8x4 image-A / packed-B GEMM for QK^T and P*V with a query-major
// fp16 score scratch + in-place softmax (no score-matrix transpose). Three
// stride-aware pre-passes compact the real (strided) Q/K/V into contiguous
// intermediates so the compute kernels run on dense data. All batch B*H via
// grid dim 2 (bh). Requires FA_D % 4 == 0; D == FA_D (128 typical).
// See docs/opencl_backend/flash_attention_optimization.md (Stage 2).
// ============================================================================

// pack_q: strided Q[b,h,q,:] -> contiguous Qp[bh][32, Sq, 4] (QK B-operand).
// global=(Sq, FA_D/4, B*H).
__kernel void tp_pack_q(__global const half* Q, __global half* Qp,
                        const int B, const int H, const int Sq,
                        const int Qbs, const int Qhs, const int Qss) {
  const int q = get_global_id(0);
  const int g = get_global_id(1);
  const int bh = get_global_id(2);
  if (q >= Sq || g >= FA_D / 4 || bh >= B * H) return;
  const int b = bh / H, h = bh - b * H;
  __global const half* src = Q + (long)b * Qbs + (long)h * Qhs + (long)q * Qss;
  __global half* dst = Qp + (long)bh * (FA_D / 4) * Sq * 4;
  const int sd = g * 4;
  #pragma unroll
  for (int kk = 0; kk < 4; ++kk) dst[((long)g * Sq + q) * 4 + kk] = src[sd + kk];
}

// trans_k: strided K[b,h,k,:] -> contiguous Kt[bh][FA_D, Skv] (QK A-image).
// global=(Skv, FA_D, B*H).
__kernel void tp_trans_k(__global const half* K, __global half* Kt,
                         const int B, const int H, const int Skv,
                         const int Kbs, const int Khs, const int Kss) {
  const int k = get_global_id(0);
  const int d = get_global_id(1);
  const int bh = get_global_id(2);
  if (k >= Skv || d >= FA_D || bh >= B * H) return;
  const int b = bh / H, h = bh - b * H;
  Kt[(long)bh * FA_D * Skv + (long)d * Skv + k] =
      K[(long)b * Kbs + (long)h * Khs + (long)k * Kss + d];
}

// copy_v: strided V[b,h,k,:] -> contiguous Vc[bh][Skv, FA_D] (PV A-image, natural).
// global=(Skv, FA_D, B*H).
// Vectorized half8 over the contiguous d (D-stride 1), coalesced.
// global=(FA_D/8, Skv, B*H).
__kernel void tp_copy_v(__global const half* V, __global half* Vc,
                        const int B, const int H, const int Skv,
                        const int Vbs, const int Vhs, const int Vss) {
  const int dg = get_global_id(0);   // d-group of 8
  const int k = get_global_id(1);
  const int bh = get_global_id(2);
  if (dg >= FA_D / 8 || k >= Skv || bh >= B * H) return;
  const int b = bh / H, h = bh - b * H;
  half8 v = vload8(0, V + (long)b * Vbs + (long)h * Vhs + (long)k * Vss + dg * 8);
  vstore8(v, 0, Vc + (long)bh * Skv * FA_D + (long)k * FA_D + dg * 8);
}

// tp_qk_gemm: S[q,k] = scale * (Q[q,:].K[k,:]) + causal mask + clamp, query-major.
// A = Kt image [R=d, M=k=Skv]; B = Qp packed [d/4, q=N=Sq, 4]. global=(Skv/8, Sq/4, B*H).
#ifndef TP_CLAMP
#define TP_CLAMP 60000.0f
#endif
__kernel void tp_qk_gemm(__read_only image1d_buffer_t Kt_img,
                         __global const half* Qp, __global half* S,
                         const int Sq, const int Skv, const int BH,
                         const float scale, const int causal) {
  const int gy = get_global_id(0);
  const int gx = get_global_id(1);
  const int bh = get_global_id(2);
  const int M = Skv, N = Sq;
  if (gy * 8 >= M || gx * 4 >= N || bh >= BH) return;
  const int gx_4 = gx << 2, gy_8 = gy << 3, M_4 = M >> 2;

  const int kt_tex_base = bh * FA_D * (M >> 2);
  __global const half* Qph = Qp + (long)bh * (FA_D / 4) * N * 4;
  __global half* Sh = S + (long)bh * (long)Sq * Skv;

  if (causal && gy_8 > (Skv - Sq) + gx_4 + 3) {
    #define TPINF(NN) { const int q = gx_4 + (NN); if (q < N) vstore8((half8)(-INFINITY), 0, Sh + (long)q * Skv + gy_8); }
    TPINF(0); TPINF(1); TPINF(2); TPINF(3);
    #undef TPINF
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
    const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
    c0 += b0 * w.s0; c0 += b1 * w.s1; c0 += b2 * w.s2; c0 += b3 * w.s3;
    c1 += b0 * w.s4; c1 += b1 * w.s5; c1 += b2 * w.s6; c1 += b3 * w.s7;
    c2 += b0 * w.s8; c2 += b1 * w.s9; c2 += b2 * w.sa; c2 += b3 * w.sb;
    c3 += b0 * w.sc; c3 += b1 * w.sd; c3 += b2 * w.se; c3 += b3 * w.sf;
  }
  #define TPEMIT(NN, CV) {                                                  \
    const int q = gx_4 + (NN);                                            \
    if (q < N) {                                                          \
      const int q_pos = (Skv - Sq) + q;                                  \
      float8 v = (CV) * scale; half8 hv;                                 \
      hv.s0 = (half)((causal && (gy_8+0) > q_pos) ? -INFINITY : clamp(v.s0,-TP_CLAMP,TP_CLAMP)); \
      hv.s1 = (half)((causal && (gy_8+1) > q_pos) ? -INFINITY : clamp(v.s1,-TP_CLAMP,TP_CLAMP)); \
      hv.s2 = (half)((causal && (gy_8+2) > q_pos) ? -INFINITY : clamp(v.s2,-TP_CLAMP,TP_CLAMP)); \
      hv.s3 = (half)((causal && (gy_8+3) > q_pos) ? -INFINITY : clamp(v.s3,-TP_CLAMP,TP_CLAMP)); \
      hv.s4 = (half)((causal && (gy_8+4) > q_pos) ? -INFINITY : clamp(v.s4,-TP_CLAMP,TP_CLAMP)); \
      hv.s5 = (half)((causal && (gy_8+5) > q_pos) ? -INFINITY : clamp(v.s5,-TP_CLAMP,TP_CLAMP)); \
      hv.s6 = (half)((causal && (gy_8+6) > q_pos) ? -INFINITY : clamp(v.s6,-TP_CLAMP,TP_CLAMP)); \
      hv.s7 = (half)((causal && (gy_8+7) > q_pos) ? -INFINITY : clamp(v.s7,-TP_CLAMP,TP_CLAMP)); \
      vstore8(hv, 0, Sh + (long)q * Skv + gy_8);                         \
    } }
  TPEMIT(0, c0); TPEMIT(1, c1); TPEMIT(2, c2); TPEMIT(3, c3);
  #undef TPEMIT
}

// tp_softmax_norm: per (bh,q) row of S, normalize IN PLACE -> P query-major.
// global=(TP_SM_LW, Sq, B*H), local=(TP_SM_LW,1,1).
#ifndef TP_SM_LW
#define TP_SM_LW 64
#endif
inline int tp_causal_row(int Sq, int Skv, int q) { return min(Skv - 1, (Skv - Sq) + q); }
__kernel void tp_softmax_norm(__global half* S, const int Sq, const int Skv, const int BH) {
  const int t = get_local_id(0);
  const int q = get_global_id(1);
  const int bh = get_global_id(2);
  if (q >= Sq || bh >= BH) return;
  __global half* Sh = S + (long)bh * Sq * Skv + (long)q * Skv;
  __local float redm[TP_SM_LW];
  __local float redl[TP_SM_LW];
  const int kcap = tp_causal_row(Sq, Skv, q);

  // pass 1: ONLINE (m, l) in one causal scan (running-max rescale) -> 2 total
  // S-touches instead of 3 (max-pass + sum-pass fused).
  float m = -INFINITY, l = 0.0f;
  for (int k = t; k <= kcap; k += TP_SM_LW) {
    float s = (float)Sh[k];
    if (!isinf(s)) { float mn = fmax(m, s); l = l * native_exp(m - mn) + native_exp(s - mn); m = mn; }
  }
  redm[t] = m; redl[t] = l; barrier(CLK_LOCAL_MEM_FENCE);
  for (int o = TP_SM_LW >> 1; o > 0; o >>= 1) {
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

  // pass 2: write P in place, only up to pv_gemm's per-n-tile read bound (wcap);
  // masked keys in (q_pos, wcap) -> 0; keys >= wcap are never read by pv.
  const int gx4 = q & ~3;
  const int wcap = min(Skv, (((Skv - Sq) + gx4 + 3) + 4) & ~3);
  for (int k = t; k < wcap; k += TP_SM_LW) {
    float s = (float)Sh[k];
    Sh[k] = (half)((isinf(m) || isinf(s)) ? 0.0f : native_exp(s - m) * inv);
  }
}

// tp_pv_gemm: O[q,d] = sum_k P[q,k] V[k,d]. A = Vc image [R=k, M=d=FA_D];
// B = P query-major (in place in S) read direct via 4x vload4. causal reduction
// bound per n-tile. global=(FA_D/8, Sq/4, B*H). O contiguous [B*H, Sq, FA_D].
__kernel void tp_pv_gemm(__read_only image1d_buffer_t V_img,
                         __global const half* P, __global half* O,
                         const int Sq, const int Skv, const int BH, const int causal) {
  const int gy = get_global_id(0);
  const int gx = get_global_id(1);
  const int bh = get_global_id(2);
  const int M = FA_D, N = Sq;
  if (gy * 8 >= M || gx * 4 >= N || bh >= BH) return;
  const int gx_4 = gx << 2, gy_8 = gy << 3, M_4 = M >> 2;

  int kmax = Skv;
  if (causal) { const int qpm = (Skv - Sq) + gx_4 + 3; kmax = min(Skv, (qpm + 4) & ~3); }

  const int v_tex_base = bh * Skv * (FA_D >> 2);
  __global const half* Ph = P + (long)bh * (long)Sq * Skv;
  __global half* Oh = O + (long)bh * (long)Sq * FA_D;
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
    half4 p0 = vload4(0, p0r + i), p1 = vload4(0, p1r + i), p2 = vload4(0, p2r + i), p3 = vload4(0, p3r + i);
    const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
    c0 += b0 * p0.s0; c0 += b1 * p0.s1; c0 += b2 * p0.s2; c0 += b3 * p0.s3;
    c1 += b0 * p1.s0; c1 += b1 * p1.s1; c1 += b2 * p1.s2; c1 += b3 * p1.s3;
    c2 += b0 * p2.s0; c2 += b1 * p2.s1; c2 += b2 * p2.s2; c2 += b3 * p2.s3;
    c3 += b0 * p3.s0; c3 += b1 * p3.s1; c3 += b2 * p3.s2; c3 += b3 * p3.s3;
  }
  #define TPEMITO(NN, CV) { const int q = gx_4 + (NN); if (q < N) vstore8(convert_half8(CV), 0, Oh + (long)q * FA_D + gy_8); }
  TPEMITO(0, c0); TPEMITO(1, c1); TPEMITO(2, c2); TPEMITO(3, c3);
  #undef TPEMITO
}

// ============================================================================
// Block-sparse two-pass GEMM prefill. Reuses the dense pre-passes (tp_pack_q /
// tp_trans_k / tp_copy_v -> contiguous Qp/Kt/Vc images); the QK/PV GEMMs read
// those images at the SELECTED block offsets from block_idx (index-driven, no
// gather). Selection: block_idx[(bh*num_qb + qb)*top_k + slot] = key-block id
// (units of BK keys) or < 0 padding. sel = top_k*BK. S scratch [B*H, num_qb*BQ,
// sel] query-major (BQ = Sq/num_qb). Correctness mirrors the dense path; the
// per-element causal/padding test handles the diagonal block's triangle.
// See docs/opencl_backend/block_sparse_two_pass.md.
// ============================================================================

#ifndef BSA_SM_LW
#define BSA_SM_LW 64
#endif

// bs_qk_gemm: S[q_local, slot] = scale*Q[q].K[selected] + causal/padding mask.
// A=Kt image @ selected offsets; B=Qp packed. global=(sel/8, BQ/4, B*H*num_qb).
__kernel void bs_qk_gemm(__read_only image1d_buffer_t Kt_img,
                         __global const half* Qp, __global half* S,
                         __global const int* block_idx,
                         const int Sq, const int Skv, const int BH,
                         const int num_qb, const int top_k, const int BK, const int BQ,
                         const float scale, const int causal) {
  const int gy = get_global_id(0);
  const int gx = get_global_id(1);
  const int bhqb = get_global_id(2);
  const int sel = top_k * BK;
  if (gy * 8 >= sel || gx * 4 >= BQ || bhqb >= BH * num_qb) return;
  const int bh = bhqb / num_qb, qb = bhqb - bh * num_qb;
  const int gy_8 = gy << 3, gx_4 = gx << 2, M_4kv = Skv >> 2;

  const int blk = gy_8 / BK;
  const int key0 = gy_8 - blk * BK;
  const int blk_id = block_idx[bhqb * top_k + blk];
  const int kg0 = (blk_id < 0) ? 0 : (blk_id * BK + key0);
  const int kt_base = bh * FA_D * M_4kv + (kg0 >> 2);

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
  #define BSEMIT(NN, CV) {                                                      \
    const int ql = gx_4 + (NN);                                              \
    if (ql < BQ) {                                                           \
      const int q_pos = (Skv - Sq) + qb * BQ + ql;                          \
      float8 v = (CV) * scale; half8 hv;                                    \
      hv.s0 = (half)((blk_id<0 || (causal && (kg0+0) > q_pos)) ? -INFINITY : clamp(v.s0,-TP_CLAMP,TP_CLAMP)); \
      hv.s1 = (half)((blk_id<0 || (causal && (kg0+1) > q_pos)) ? -INFINITY : clamp(v.s1,-TP_CLAMP,TP_CLAMP)); \
      hv.s2 = (half)((blk_id<0 || (causal && (kg0+2) > q_pos)) ? -INFINITY : clamp(v.s2,-TP_CLAMP,TP_CLAMP)); \
      hv.s3 = (half)((blk_id<0 || (causal && (kg0+3) > q_pos)) ? -INFINITY : clamp(v.s3,-TP_CLAMP,TP_CLAMP)); \
      hv.s4 = (half)((blk_id<0 || (causal && (kg0+4) > q_pos)) ? -INFINITY : clamp(v.s4,-TP_CLAMP,TP_CLAMP)); \
      hv.s5 = (half)((blk_id<0 || (causal && (kg0+5) > q_pos)) ? -INFINITY : clamp(v.s5,-TP_CLAMP,TP_CLAMP)); \
      hv.s6 = (half)((blk_id<0 || (causal && (kg0+6) > q_pos)) ? -INFINITY : clamp(v.s6,-TP_CLAMP,TP_CLAMP)); \
      hv.s7 = (half)((blk_id<0 || (causal && (kg0+7) > q_pos)) ? -INFINITY : clamp(v.s7,-TP_CLAMP,TP_CLAMP)); \
      vstore8(hv, 0, Sh + (long)ql * sel + gy_8);                           \
    } }
  BSEMIT(0, c0); BSEMIT(1, c1); BSEMIT(2, c2); BSEMIT(3, c3);
  #undef BSEMIT
}

// bs_softmax: per (bhqb, q_local) over sel slots, online (m,l) + write in place.
// global=(BSA_SM_LW, BQ, B*H*num_qb), local=(BSA_SM_LW,1,1).
__kernel void bs_softmax(__global half* S, const int BH, const int num_qb,
                         const int top_k, const int BK, const int BQ) {
  const int t = get_local_id(0);
  const int ql = get_global_id(1);
  const int bhqb = get_global_id(2);
  const int sel = top_k * BK;
  if (ql >= BQ || bhqb >= BH * num_qb) return;
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

// bs_pv_gemm: O[q,d] = sum_slot P[q,slot] V[selected]. A=Vc image @ selected
// offsets; B=P slot-major direct. Outer loop over selected blocks (block_idx
// once/block, padding skipped). global=(FA_D/8, BQ/4, B*H*num_qb). O[bh,Sq,D].
__kernel void bs_pv_gemm(__read_only image1d_buffer_t V_img,
                         __global const half* P, __global half* O,
                         __global const int* block_idx,
                         const int Sq, const int Skv, const int BH,
                         const int num_qb, const int top_k, const int BK, const int BQ) {
  const int gy = get_global_id(0);
  const int gx = get_global_id(1);
  const int bhqb = get_global_id(2);
  const int sel = top_k * BK;
  if (gy * 8 >= FA_D || gx * 4 >= BQ || bhqb >= BH * num_qb) return;
  const int bh = bhqb / num_qb, qb = bhqb - bh * num_qb;
  const int gy_8 = gy << 3, gx_4 = gx << 2, M_4 = FA_D >> 2;

  const int v_head = bh * Skv * (FA_D >> 2);
  __global const half* Ph = P + (long)bhqb * BQ * sel;
  __global half* Oh = O + (long)bh * (long)Sq * FA_D;
  __global const half* p0r = Ph + (long)(gx_4 + 0) * sel;
  __global const half* p1r = Ph + (long)(gx_4 + 1) * sel;
  __global const half* p2r = Ph + (long)(gx_4 + 2) * sel;
  __global const half* p3r = Ph + (long)(gx_4 + 3) * sel;

  float8 c0 = (float8)0, c1 = (float8)0, c2 = (float8)0, c3 = (float8)0;
  half8 B0, B1, B2, B3;
  for (int blk = 0; blk < top_k; ++blk) {
    const int blk_id = block_idx[bhqb * top_k + blk];
    if (blk_id < 0) continue;
    const int kbase = blk_id * BK, sbase = blk * BK;
    for (int sub = 0; sub < BK; sub += 4) {
      const int s = sbase + sub, kg = kbase + sub;
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
  #define BSEMITO(NN, CV) { const int ql = gx_4 + (NN); if (ql < BQ) { const int qg = qb * BQ + ql; vstore8(convert_half8(CV), 0, Oh + (long)qg * FA_D + gy_8); } }
  BSEMITO(0, c0); BSEMITO(1, c1); BSEMITO(2, c2); BSEMITO(3, c3);
  #undef BSEMITO
}
