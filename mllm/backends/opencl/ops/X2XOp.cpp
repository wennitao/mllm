// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "X2XOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

OpenCLX2XOp::OpenCLX2XOp(const aops::X2XOpOptions& options) : aops::X2XOp(options) {}

void OpenCLX2XOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  return BaseOp::setup(inputs, outputs);
}

void OpenCLX2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& input = inputs[0];
  auto& output = outputs[0];

  if (input.device() == kOpenCL && output.device() == kOpenCL) {
    // If both input and output are on OpenCL device
    // Retain the buffer to ensure proper reference counting
    cl_mem src_buffer = static_cast<cl_mem>(input.impl()->storage()->ptr_);
    OpenCLLoader::instance().clRetainMemObject(src_buffer);
    output.impl()->storage()->ptr_ = src_buffer;
  } else if (input.device() == kOpenCL && output.device() == kCPU) {
    // Compute byte offset and size accounting for sliced tensors
    size_t src_offset = (size_t)(input.impl()->storageOffset() / lanesOfType(input.dtype())) * bytesOfType(input.dtype());
    size_t data_size = (size_t)(input.numel() / lanesOfType(input.dtype())) * bytesOfType(input.dtype());

    // Get OpenCL runtime
    auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

    // Get the OpenCL buffer from input tensor's storage
    auto cl_buffer = cl::Buffer(static_cast<cl_mem>(input.impl()->storage()->ptr_), true);

    // Get output data pointer
    void* dst_data = output.ptr<void>();

    // Read data from OpenCL buffer to host (CPU) memory, respecting storage offset
    auto error = runtime->commandQueue().enqueueReadBuffer(cl_buffer,
                                                           CL_TRUE,      // blocking read
                                                           src_offset,   // byte offset into OpenCL buffer
                                                           data_size,    // bytes to read (slice size)
                                                           dst_data      // pointer to host memory
    );

    if (error != CL_SUCCESS) { MLLM_ERROR("Failed to read data from OpenCL buffer, error code: {}", error); }

    // Wait for the command to finish
    runtime->commandQueue().finish();
    return;
  } else if (input.device() == kCPU && output.device() == kOpenCL) {
    size_t dst_offset = (size_t)(output.impl()->storageOffset() / lanesOfType(output.dtype())) * bytesOfType(output.dtype());
    size_t data_size = (size_t)(input.numel() / lanesOfType(input.dtype())) * bytesOfType(input.dtype());
    auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
    void* src_data = input.ptr<void>();
    cl_mem cl_buffer = (cl_mem)output.impl()->storage()->ptr_;

    cl_int error = OpenCLLoader::instance().clEnqueueWriteBuffer(runtime->commandQueue()(), cl_buffer, CL_TRUE, dst_offset,
                                                                 data_size, src_data, 0, nullptr, nullptr);

    if (error != CL_SUCCESS) { MLLM_ERROR("Failed to write data to OpenCL buffer, error code: {}", error); }
    return;
  }

  MLLM_ERROR("OpenCLX2XOp only supports transform between CPU and OpenCL.\n");
}

}  // namespace mllm::opencl