// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::opencl {

class OpenCLLinearOp final : public aops::LinearOp {
 public:
  explicit OpenCLLinearOp(const aops::LinearOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  // ---- LPBQ (w4a16, two-scale int4) path ----------------------------------
  // Additive & non-breaking: only active after setLPBQ() is called. Weights
  // must be host-prepacked into the ushort [K/4,N/4,4] + combined-scale
  // [num_blocks,N/4,4] layout (mllm::cpu::lpbq_prepack_weights_USHORT4 /
  // lpbq_prepack_combined_scales). The cl_mem buffers are owned by the caller.
  void setLPBQ(cl_mem w_ushort, cl_mem combined_scales, int K, int N, int Bs);
  bool lpbqEnabled() const { return lpbq_enabled_; }
  // Execute the LPBQ matmul on fp16 buffers: input [M,K] -> output [M,N].
  // Used by forward() and by the op-level correctness test.
  void runLPBQ(cl_mem input_fp16, cl_mem output_fp16, int M);

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_q4_0_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp32_q4_0_transb_bias_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_gemv_fp32_q4_0_transb_bias_ = nullptr;         // GEMV
  std::shared_ptr<KernelWrap> kernel_gemv_fp16_q4_0_transb_bias_ = nullptr;         // GEMV
  std::shared_ptr<KernelWrap> kernel_gemv_fp16_q4_0_transb_bias_half16_ = nullptr;  // GEMV for K%16==0

  // LPBQ tuned kernels (built only when the device supports fp16).
  std::shared_ptr<KernelWrap> kernel_lpbq_gemm_v5_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_lpbq_gemv_v5_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_lpbq_transpose_ = nullptr;
  bool lpbq_enabled_ = false;
  cl_mem lpbq_w_ushort_ = nullptr;
  cl_mem lpbq_scales_ = nullptr;
  int lpbq_K_ = 0, lpbq_N_ = 0, lpbq_Bs_ = 16;
};

class OpenCLLinearOpFactory : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<OpenCLLinearOp>(options);
  }
};

}  // namespace mllm::opencl
