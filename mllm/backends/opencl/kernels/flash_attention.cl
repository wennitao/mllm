#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// FlashAttention v1 (Dao et al., 2022), forward pass for inference.
// Faithful to Algorithm 1 in the paper:
//   - K/V tiles loaded into SRAM (__local) once per outer j iteration and
//     reused across all B_r Q rows in this work-group.
//   - Online softmax merge with running (m_i, l_i, O_i) per Q row.
//
// Layout: BHSD, all input tensors may be non-contiguous (strides passed in).
//   Q: [B, H, S_q,  D]
//   K: [B, H, S_kv, D]   (already GQA-expanded to H = q_heads)
//   V: [B, H, S_kv, D]
//   O: [B, H, S_q,  D]   (always contiguous, allocated by reshape())
//
// Parallelization: one work-group per (batch, head, q_block), where each
// q_block covers B_r Q rows. local_size MUST equal FA_D (head dim, must be
// power-of-two for the tree reduction). Each thread = one d-position.

#ifndef FA_BR
#define FA_BR 4   // Q rows per work-group
#endif

#ifndef FA_BC
#define FA_BC 16  // K/V cols per inner tile
#endif

#ifndef FA_D
#define FA_D 128  // head dim; pass via -DFA_D=… at build time
#endif

// ---------------------------------------------------------------- fp32

__kernel void flash_attention_fp32(
    __global const float *Q, __global const float *K, __global const float *V,
    __global float *O,
    const int B, const int H, const int S_q, const int S_kv, const int D,
    const int Q_b_stride, const int Q_h_stride, const int Q_s_stride,
    const int K_b_stride, const int K_h_stride, const int K_s_stride,
    const int V_b_stride, const int V_h_stride, const int V_s_stride,
    const float scale, const int causal_mask) {
  const int d = get_local_id(0);
  const int q_block = get_global_id(1);
  const int bh = get_global_id(2);
  if (bh >= B * H) return;
  const int q_row_start = q_block * FA_BR;
  if (q_row_start >= S_q) return;

  const int local_size = get_local_size(0);
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
  __local float reduce_buf[FA_D];
  __local float m_il[FA_BR];
  __local float l_il[FA_BR];

  // Load Q tile for this q_block (B_r × D), zero-pad rows past S_q.
  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    Q_local[i * FA_D + d] =
        (q_row < S_q) ? Q[q_h_off + q_row * Q_s_stride + d] : 0.0f;
  }
  // Init running stats per i (only first FA_BR threads write).
  if (d < FA_BR) {
    m_il[d] = -INFINITY;
    l_il[d] = 0.0f;
  }
  // O accumulator in registers, one per Q row, this thread's d-position.
  float o_priv[FA_BR];
  for (int i = 0; i < FA_BR; ++i) o_priv[i] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  // Outer loop over K/V blocks (paper Algorithm 1, line 5).
  for (int j_start = 0; j_start < S_kv; j_start += FA_BC) {
    // Load K_j and V_j tiles cooperatively.
    for (int c = 0; c < FA_BC; ++c) {
      const int k_row = j_start + c;
      if (k_row < S_kv) {
        K_local[c * FA_D + d] = K[k_h_off + k_row * K_s_stride + d];
        V_local[c * FA_D + d] = V[v_h_off + k_row * V_s_stride + d];
      } else {
        K_local[c * FA_D + d] = 0.0f;
        V_local[c * FA_D + d] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // S_ij = Q_i K_j^T (B_r × B_c), one tree-reduction per element.
    for (int i = 0; i < FA_BR; ++i) {
      for (int c = 0; c < FA_BC; ++c) {
        reduce_buf[d] = Q_local[i * FA_D + d] * K_local[c * FA_D + d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = local_size / 2; off > 0; off >>= 1) {
          if (d < off) reduce_buf[d] += reduce_buf[d + off];
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (d == 0) {
          float s = reduce_buf[0] * scale;
          const int q_row = q_row_start + i;
          const int k_row = j_start + c;
          const int q_pos = S_kv - S_q + q_row;
          if (q_row >= S_q || k_row >= S_kv ||
              (causal_mask && k_row > q_pos)) {
            s = -INFINITY;
          }
          S_local[i * FA_BC + c] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    // Online softmax merge per Q row. All threads compute the same scalars
    // from S_local; PV is the only per-d quantity.
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
        pv += p * V_local[c * FA_D + d];
      }

      const float m_old = m_il[i];
      const float l_old = l_il[i];
      const float m_new = fmax(m_old, m_tilde);
      const float a = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
      const float bb = (m_tilde == -INFINITY) ? 0.0f : exp(m_tilde - m_new);
      const float l_new = a * l_old + bb * l_tilde;

      o_priv[i] =
          (l_new > 0.0f) ? (l_old * a * o_priv[i] + bb * pv) / l_new : 0.0f;

      if (d == 0) {
        m_il[i] = m_new;
        l_il[i] = l_new;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write O.
  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    if (q_row < S_q) { O[o_h_off + q_row * D + d] = o_priv[i]; }
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
  const int d = get_local_id(0);
  const int q_block = get_global_id(1);
  const int bh = get_global_id(2);
  if (bh >= B * H) return;
  const int q_row_start = q_block * FA_BR;
  if (q_row_start >= S_q) return;

  const int local_size = get_local_size(0);
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
  __local float reduce_buf[FA_D];
  __local float m_il[FA_BR];
  __local float l_il[FA_BR];

  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    Q_local[i * FA_D + d] =
        (q_row < S_q) ? (float)Q[q_h_off + q_row * Q_s_stride + d] : 0.0f;
  }
  if (d < FA_BR) {
    m_il[d] = -INFINITY;
    l_il[d] = 0.0f;
  }
  float o_priv[FA_BR];
  for (int i = 0; i < FA_BR; ++i) o_priv[i] = 0.0f;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int j_start = 0; j_start < S_kv; j_start += FA_BC) {
    for (int c = 0; c < FA_BC; ++c) {
      const int k_row = j_start + c;
      if (k_row < S_kv) {
        K_local[c * FA_D + d] = (float)K[k_h_off + k_row * K_s_stride + d];
        V_local[c * FA_D + d] = (float)V[v_h_off + k_row * V_s_stride + d];
      } else {
        K_local[c * FA_D + d] = 0.0f;
        V_local[c * FA_D + d] = 0.0f;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < FA_BR; ++i) {
      for (int c = 0; c < FA_BC; ++c) {
        reduce_buf[d] = Q_local[i * FA_D + d] * K_local[c * FA_D + d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = local_size / 2; off > 0; off >>= 1) {
          if (d < off) reduce_buf[d] += reduce_buf[d + off];
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (d == 0) {
          float s = reduce_buf[0] * scale;
          const int q_row = q_row_start + i;
          const int k_row = j_start + c;
          const int q_pos = S_kv - S_q + q_row;
          if (q_row >= S_q || k_row >= S_kv ||
              (causal_mask && k_row > q_pos)) {
            s = -INFINITY;
          }
          S_local[i * FA_BC + c] = s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

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
        pv += p * V_local[c * FA_D + d];
      }

      const float m_old = m_il[i];
      const float l_old = l_il[i];
      const float m_new = fmax(m_old, m_tilde);
      const float a = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
      const float bb = (m_tilde == -INFINITY) ? 0.0f : exp(m_tilde - m_new);
      const float l_new = a * l_old + bb * l_tilde;

      o_priv[i] =
          (l_new > 0.0f) ? (l_old * a * o_priv[i] + bb * pv) / l_new : 0.0f;

      if (d == 0) {
        m_il[i] = m_new;
        l_il[i] = l_new;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i = 0; i < FA_BR; ++i) {
    const int q_row = q_row_start + i;
    if (q_row < S_q) { O[o_h_off + q_row * D + d] = (half)o_priv[i]; }
  }
}
