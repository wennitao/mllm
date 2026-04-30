// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/FlashAttn2Op.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"

#include <cmath>

namespace mllm::opencl {

// Must match the -D defines used when building the kernel.
static constexpr int BLOCK_M     = 32;
static constexpr int BLOCK_N     = 16;
static constexpr int Q1_WG_SIZE  = 64;  // fixed in kernel source

OpenCLFlashAttention2Op::OpenCLFlashAttention2Op(const aops::FlashAttention2OpOptions& options)
    : aops::FlashAttention2Op(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  std::set<std::string> build_opts = {
      "-DDK="      + std::to_string(options.D),
      "-DDV="      + std::to_string(options.D),
      "-DBLOCK_M=" + std::to_string(BLOCK_M),
      "-DBLOCK_N=" + std::to_string(BLOCK_N),
  };

  kernel_prefill_ = runtime->buildKernel("flash_attn", "flash_attn_f32",    build_opts);
  kernel_decode_  = runtime->buildKernel("flash_attn", "flash_attn_f32_q1", build_opts);

  MLLM_RT_ASSERT(kernel_prefill_);
  MLLM_RT_ASSERT(kernel_decode_);
}

void OpenCLFlashAttention2Op::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Q: [B, S_q, H_q, D_qk]   V: [B, S_kv, H_kv, D_v]
  const auto& Q = inputs[0];
  const auto& V = inputs[2];
  outputs.emplace_back(Tensor::empty({Q.shape()[0], Q.shape()[1], Q.shape()[2], V.shape()[3]}, Q.dtype(), Q.device()));
}

void OpenCLFlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // All tensors are BSHD (batch, seq, heads, dim) and on OpenCL.
  const auto& Q = inputs[0];  // [B, S_q,  H_q,  D]
  const auto& K = inputs[1];  // [B, S_kv, H_kv, D]
  const auto& V = inputs[2];  // [B, S_kv, H_kv, D]
  auto&       O = outputs[0]; // [B, S_q,  H_q,  D]

  const int B        = Q.shape()[0];
  const int n_q      = Q.shape()[1];
  const int n_head   = Q.shape()[2];
  const int D        = Q.shape()[3];
  const int n_head_kv = K.shape()[2];
  const int n_kv     = K.shape()[1];

  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  // Byte strides for BSHD tensors.
  // Kernel convention for Q/K/V inputs:
  //   q_row_offset = batch * q_nb3 + head * q_nb2 + seq * q_nb1
  // So nb1 = S-dim stride, nb2 = H-dim stride, nb3 = B-dim stride.
  //
  // Kernel convention for O output:
  //   o_row_offset = batch * o_nb3 + seq * o_nb2 + head * o_nb1
  // So nb1 = H-dim stride, nb2 = S-dim stride, nb3 = B-dim stride.
  const cl_ulong elem_bytes = static_cast<cl_ulong>(bytesOfType(Q.dtype()));
  const cl_ulong lanes      = static_cast<cl_ulong>(lanesOfType(Q.dtype()));

  auto byte_stride = [&](const Tensor& t, int dim) -> cl_ulong {
    return static_cast<cl_ulong>(t.stride()[dim] / lanes) * elem_bytes;
  };
  auto byte_offset = [&](const Tensor& t) -> cl_ulong {
    return static_cast<cl_ulong>(t.impl()->storageOffset() / lanes) * elem_bytes;
  };

  cl_mem q_mem = (cl_mem)Q.impl()->storage()->ptr_;
  cl_mem k_mem = (cl_mem)K.impl()->storage()->ptr_;
  cl_mem v_mem = (cl_mem)V.impl()->storage()->ptr_;
  cl_mem o_mem = (cl_mem)O.impl()->storage()->ptr_;
  cl_mem null_mem = nullptr;

  cl_ulong q_off = byte_offset(Q);
  cl_ulong q_nb1 = byte_stride(Q, 1);  // S-dim
  cl_ulong q_nb2 = byte_stride(Q, 2);  // H-dim
  cl_ulong q_nb3 = byte_stride(Q, 0);  // B-dim

  cl_ulong k_off = byte_offset(K);
  cl_ulong k_nb1 = byte_stride(K, 1);
  cl_ulong k_nb2 = byte_stride(K, 2);
  cl_ulong k_nb3 = byte_stride(K, 0);

  cl_ulong v_off = byte_offset(V);
  cl_ulong v_nb1 = byte_stride(V, 1);
  cl_ulong v_nb2 = byte_stride(V, 2);
  cl_ulong v_nb3 = byte_stride(V, 0);

  cl_ulong o_off = byte_offset(O);
  cl_ulong o_nb1 = byte_stride(O, 2);  // H-dim (kernel writes seq*nb2 + head*nb1)
  cl_ulong o_nb2 = byte_stride(O, 1);  // S-dim
  cl_ulong o_nb3 = byte_stride(O, 0);  // B-dim

  // ALiBi disabled (max_bias=0 → slope=1 for all heads).
  float max_bias      = 0.0f;
  float m0            = 0.0f;
  float m1            = 0.0f;
  int   n_head_log2_i = 0;
  float logit_softcap = 0.0f;
  int   is_causal     = options_.causal_mask ? 1 : 0;
  cl_ulong zero_ul    = 0;
  int      zero_i     = 0;

  const bool is_decode = (n_q == 1);
  auto& kernel_wrap = is_decode ? kernel_decode_ : kernel_prefill_;
  auto& k = kernel_wrap->get();

  int arg = 0;
  cl_int ret = CL_SUCCESS;
  ret |= k.setArg(arg++, sizeof(cl_mem),   &q_mem);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &q_off);
  ret |= k.setArg(arg++, sizeof(cl_mem),   &k_mem);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &k_off);
  ret |= k.setArg(arg++, sizeof(cl_mem),   &v_mem);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &v_off);
  ret |= k.setArg(arg++, sizeof(cl_mem),   &o_mem);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &o_off);
  ret |= k.setArg(arg++, sizeof(float),    &scale);
  ret |= k.setArg(arg++, sizeof(int),      &n_q);
  ret |= k.setArg(arg++, sizeof(int),      &n_kv);
  ret |= k.setArg(arg++, sizeof(int),      &is_causal);
  ret |= k.setArg(arg++, sizeof(int),      &n_head);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &q_nb1);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &q_nb2);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &q_nb3);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &k_nb1);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &k_nb2);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &k_nb3);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &v_nb1);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &v_nb2);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &v_nb3);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &o_nb1);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &o_nb2);
  ret |= k.setArg(arg++, sizeof(cl_ulong), &o_nb3);
  ret |= k.setArg(arg++, sizeof(float),    &max_bias);
  ret |= k.setArg(arg++, sizeof(float),    &m0);
  ret |= k.setArg(arg++, sizeof(float),    &m1);
  ret |= k.setArg(arg++, sizeof(int),      &n_head_log2_i);
  ret |= k.setArg(arg++, sizeof(float),    &logit_softcap);
  ret |= k.setArg(arg++, sizeof(int),      &n_head_kv);
  ret |= k.setArg(arg++, sizeof(cl_mem),   &null_mem);   // mask_void = NULL
  ret |= k.setArg(arg++, sizeof(cl_ulong), &zero_ul);    // mask_offset
  ret |= k.setArg(arg++, sizeof(cl_ulong), &zero_ul);    // mask_nb1
  ret |= k.setArg(arg++, sizeof(cl_ulong), &zero_ul);    // mask_nb2
  ret |= k.setArg(arg++, sizeof(cl_ulong), &zero_ul);    // mask_nb3
  ret |= k.setArg(arg++, sizeof(int),      &zero_i);     // mask_ne2
  ret |= k.setArg(arg++, sizeof(int),      &zero_i);     // mask_ne3
  ret |= k.setArg(arg++, sizeof(cl_mem),   &null_mem);   // sinks_void = NULL
  ret |= k.setArg(arg++, sizeof(cl_ulong), &zero_ul);    // sinks_offset

  if (ret != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLFlashAttention2Op setArg failed: {}", ret);
  }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  cl_int error;

  if (is_decode) {
    // Single query row: one workgroup of Q1_WG_SIZE threads per (batch, head).
    cl::NDRange global(Q1_WG_SIZE, B * n_head);
    cl::NDRange local(Q1_WG_SIZE, 1);
    error = runtime->commandQueue().enqueueNDRangeKernel(k, cl::NullRange, global, local);
  } else {
    // Prefill: BLOCK_M threads per tile of query rows.
    int groups_q = (n_q + BLOCK_M - 1) / BLOCK_M;
    cl::NDRange global(groups_q * BLOCK_M, B * n_head);
    cl::NDRange local(BLOCK_M, 1);
    error = runtime->commandQueue().enqueueNDRangeKernel(k, cl::NullRange, global, local);
  }

  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLFlashAttention2Op kernel dispatch failed: {}", error);
  }
}

}  // namespace mllm::opencl
