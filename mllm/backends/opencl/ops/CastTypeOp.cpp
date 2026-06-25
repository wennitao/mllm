// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "CastTypeOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLCastTypeOp::OpenCLCastTypeOp(const aops::CastTypeOpOptions& options) : aops::CastTypeOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  kernel_f32_to_f16_ = runtime->buildKernel("cast_type", "cast_f32_to_f16", {});
  kernel_f16_to_f32_ = runtime->buildKernel("cast_type", "cast_f16_to_f32", {});
}

void OpenCLCastTypeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_in = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_out = (cl_mem)output.impl()->storage()->ptr_;
  const size_t global_size = input.numel();

  std::shared_ptr<KernelWrap> kernel_wrapper = nullptr;
  if (input.dtype() == MLLM_TYPE_F32 && output.dtype() == MLLM_TYPE_F16) {
    kernel_wrapper = kernel_f32_to_f16_;
  } else if (input.dtype() == MLLM_TYPE_F16 && output.dtype() == MLLM_TYPE_F32) {
    kernel_wrapper = kernel_f16_to_f32_;
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLCastTypeOp unsupported cast {} -> {}", input.dtype(), output.dtype());
  }

  cl_int ret = CL_SUCCESS;
  ret |= kernel_wrapper->get().setArg(0, sizeof(cl_mem), &cl_buffer_in);
  ret |= kernel_wrapper->get().setArg(1, sizeof(cl_mem), &cl_buffer_out);
  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLCastTypeOp setArg failed: {}", ret); }

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_wrapper->get(), cl::NullRange,
                                                            cl::NDRange(global_size), cl::NullRange);
  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute cast_type kernel, error code: {}", error);
  }
}

}  // namespace mllm::opencl
