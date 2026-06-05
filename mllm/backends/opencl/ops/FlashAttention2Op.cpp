// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// NOTE on naming: the *op class* is OpenCLFlashAttention2Op because that's the
// registered framework slot (OpTypes::kFlashAttention2 / aops::FlashAttention2Op).
// The actual kernel here implements **FlashAttention v1** (Dao et al., 2022,
// Algorithm 1) -- the simpler version that normalizes the output every block.

#include <cmath>

#include "mllm/backends/opencl/ops/FlashAttention2Op.hpp"
#include "CL/cl.h"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLFlashAttention2Op::OpenCLFlashAttention2Op(const aops::FlashAttention2OpOptions& options)
    : aops::FlashAttention2Op(options) {
  // Kernels are built lazily on first forward() because FA_D (head_dim) must
  // be baked in at compile time to size __local arrays.

  // One-shot constructor confirmation (first instantiation only). Remove later.
  static bool s_ctor_logged = false;
  if (!s_ctor_logged) {
    s_ctor_logged = true;
    MLLM_INFO("[OpenCL-FA1] OpenCLFlashAttention2Op constructed (factory dispatched)");
  }
}

void OpenCLFlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& O = outputs[0];

  // Expect BHSD: [B, H, S, D].
  const auto& q_shape = Q.shape();
  const auto& k_shape = K.shape();
  MLLM_RT_ASSERT_EQ(q_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(k_shape.size(), 4);

  const int B = q_shape[0];
  const int H = q_shape[1];
  const int S_q = q_shape[2];
  const int D = q_shape[3];
  const int S_kv = k_shape[2];

  // GQA expected to be pre-applied (StaticCache eager-mode does this), so K's
  // head dim should match Q's.
  MLLM_RT_ASSERT_EQ(k_shape[1], H);
  MLLM_RT_ASSERT_EQ(V.shape()[1], H);
  MLLM_RT_ASSERT_EQ(V.shape()[2], S_kv);

  // Kernel uses local_size = D + tree reduction; require D power of two.
  MLLM_RT_ASSERT(D > 0 && (D & (D - 1)) == 0);
  MLLM_RT_ASSERT(D <= 256);

  // Lazy build per head_dim — FA_D macro sizes __local arrays.
  if (built_for_d_ != D) {
    auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
    // fp16 kernel processes kBrFp16 q-rows per workgroup (cross-q reuse); fp32
    // reference uses kBr. The host dispatch below matches each kernel's row count.
    std::set<std::string> opts;
    opts.insert(std::string("-DFA_D=") + std::to_string(D));
    opts.insert(std::string("-DFA_BR=") + std::to_string(kBr));
    opts.insert(std::string("-DFA_BR_H=") + std::to_string(kBrFp16));
    kernel_fp32_ = runtime->buildKernel("flash_attention", "flash_attention_fp32", opts);
    MLLM_RT_ASSERT(kernel_fp32_);
    kernel_fp16_ = runtime->buildKernel("flash_attention", "flash_attention_fp16", opts);
    MLLM_RT_ASSERT(kernel_fp16_);
    built_for_d_ = D;
  }

  // Strides (in elements). Q is non-contig after .transpose(1,2); K/V are
  // non-contig prefix slices of the static cache. Innermost D-stride must be 1
  // for our simple element-wise kernel access pattern.
  const auto& qs = Q.stride();
  const auto& ks = K.stride();
  const auto& vs = V.stride();
  MLLM_RT_ASSERT_EQ(qs[3], 1);
  MLLM_RT_ASSERT_EQ(ks[3], 1);
  MLLM_RT_ASSERT_EQ(vs[3], 1);
  const int Q_b_stride = qs[0], Q_h_stride = qs[1], Q_s_stride = qs[2];
  const int K_b_stride = ks[0], K_h_stride = ks[1], K_s_stride = ks[2];
  const int V_b_stride = vs[0], V_h_stride = vs[1], V_s_stride = vs[2];

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int causal = options_.causal_mask ? 1 : 0;

  std::shared_ptr<KernelWrap> kernel = nullptr;
  if (Q.dtype() == mllm::kFloat32) {
    kernel = kernel_fp32_;
  } else if (Q.dtype() == mllm::kFloat16) {
    kernel = kernel_fp16_;
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLFlashAttention2Op supports only FP32 and FP16, got dtype={}",
                    nameOfType(Q.dtype()));
  }

  auto q_buf = (cl_mem)Q.impl()->ptr<void>();
  auto k_buf = (cl_mem)K.impl()->ptr<void>();
  auto v_buf = (cl_mem)V.impl()->ptr<void>();
  auto o_buf = (cl_mem)O.impl()->ptr<void>();

  cl_int err = CL_SUCCESS;
  err |= kernel->get().setArg(0, sizeof(cl_mem), &q_buf);
  err |= kernel->get().setArg(1, sizeof(cl_mem), &k_buf);
  err |= kernel->get().setArg(2, sizeof(cl_mem), &v_buf);
  err |= kernel->get().setArg(3, sizeof(cl_mem), &o_buf);
  err |= kernel->get().setArg(4, sizeof(int), &B);
  err |= kernel->get().setArg(5, sizeof(int), &H);
  err |= kernel->get().setArg(6, sizeof(int), &S_q);
  err |= kernel->get().setArg(7, sizeof(int), &S_kv);
  err |= kernel->get().setArg(8, sizeof(int), &D);
  err |= kernel->get().setArg(9, sizeof(int), &Q_b_stride);
  err |= kernel->get().setArg(10, sizeof(int), &Q_h_stride);
  err |= kernel->get().setArg(11, sizeof(int), &Q_s_stride);
  err |= kernel->get().setArg(12, sizeof(int), &K_b_stride);
  err |= kernel->get().setArg(13, sizeof(int), &K_h_stride);
  err |= kernel->get().setArg(14, sizeof(int), &K_s_stride);
  err |= kernel->get().setArg(15, sizeof(int), &V_b_stride);
  err |= kernel->get().setArg(16, sizeof(int), &V_h_stride);
  err |= kernel->get().setArg(17, sizeof(int), &V_s_stride);
  err |= kernel->get().setArg(18, sizeof(float), &scale);
  err |= kernel->get().setArg(19, sizeof(int), &causal);
  if (err != CL_SUCCESS) { MLLM_ERROR("OpenCLFlashAttention2Op setArg failed: {}", err); }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  // One-shot dispatch confirmation. Remove once verified on device.
  static bool s_logged_once = false;
  if (!s_logged_once) {
    s_logged_once = true;
    MLLM_INFO("[OpenCL-FA1] dispatched: B={} H={} S_q={} S_kv={} D={} dtype={} causal={}", B, H, S_q, S_kv, D,
              nameOfType(Q.dtype()), causal);
  }

  // One work-group per (q_block, batch*head). fp16 covers kBrFp16 rows/wg (the
  // cross-q-reuse kernel), fp32 covers kBr. local_size = D (one lane per d).
  const int rows_per_wg = (Q.dtype() == mllm::kFloat16) ? kBrFp16 : kBr;
  const int q_blocks = (S_q + rows_per_wg - 1) / rows_per_wg;
  cl::NDRange global(D, q_blocks, B * H);
  cl::NDRange local(D, 1, 1);
  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, global, local);
  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to enqueue OpenCL FlashAttention kernel, error code: {}", error);
  }
}

}  // namespace mllm::opencl
