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
