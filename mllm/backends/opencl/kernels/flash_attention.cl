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

// ---------------------------------------------------------------- fp16

__kernel void flash_attention_fp16(
    __global const half *Q, __global const half *K, __global const half *V,
    __global half *O,
    const int B, const int H, const int S_q, const int S_kv, const int D,
    const int Q_b_stride, const int Q_h_stride, const int Q_s_stride,
    const int K_b_stride, const int K_h_stride, const int K_s_stride,
    const int V_b_stride, const int V_h_stride, const int V_s_stride,
    const float scale, const int causal_mask) {
  const int t = get_local_id(0);
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

  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    Q_local[i * FA_D + t] =
        (q_row < S_q) ? (float)Q[q_h_off + q_row * Q_s_stride + t] : 0.0f;
  }
  if (t < FA_BR) {
    m_il[t] = -INFINITY;
    l_il[t] = 0.0f;
  }
  float o_priv[FA_BR];
  for (int i = 0; i < FA_BR; ++i) o_priv[i] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int j_start = 0; j_start < S_kv; j_start += FA_BC) {
    for (int c = 0; c < FA_BC; ++c) {
      const int k_row = j_start + c;
      if (k_row < S_kv) {
        K_local[c * FA_D + t] = (float)K[k_h_off + k_row * K_s_stride + t];
        V_local[c * FA_D + t] = (float)V[v_h_off + k_row * V_s_stride + t];
      } else {
        K_local[c * FA_D + t] = 0.0f;
        V_local[c * FA_D + t] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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

  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    if (q_row < S_q) { O[o_h_off + q_row * D + t] = (half)o_priv[i]; }
  }
}
