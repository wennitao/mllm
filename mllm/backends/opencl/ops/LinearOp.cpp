// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/ops/LinearOp.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/mllm.hpp"

namespace mllm::opencl {

OpenCLLinearOp::OpenCLLinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {
  auto runtime = std::static_pointer_cast<OpenCLBackend>(Context::instance().getBackend(kOpenCL))->runtime();
  std::string program_name = "matmul_transb_bias";

  const bool fp16 = runtime->isSupportedFP16();
  std::set<std::string> buildOptions;
  if (fp16) { buildOptions.insert("-DSUPPORTS_FP16"); }

  // Build kernels
  kernel_fp32_transb_bias_ = runtime->buildKernel(program_name, "gemm_fp32_transb_bias", buildOptions);
  MLLM_RT_ASSERT(kernel_fp32_transb_bias_);

  kernel_fp32_q4_0_transb_bias_ = runtime->buildKernel(program_name, "gemm_fp32_q4_0_transb_bias", buildOptions);
  MLLM_RT_ASSERT(kernel_fp32_q4_0_transb_bias_);

  kernel_gemv_fp32_q4_0_transb_bias_ = runtime->buildKernel(program_name, "gemv_fp32_q4_0_transb_bias", buildOptions);
  MLLM_RT_ASSERT(kernel_gemv_fp32_q4_0_transb_bias_);

  // LPBQ tuned kernels live inside the SUPPORTS_FP16 block of the program.
  if (fp16) {
    kernel_lpbq_gemm_v5_ = runtime->buildKernel(program_name, "lpbq_gemm_fp16_v5", buildOptions);
    MLLM_RT_ASSERT(kernel_lpbq_gemm_v5_);
    kernel_lpbq_gemv_v5_ = runtime->buildKernel(program_name, "lpbq_gemv_fp16_v5", buildOptions);
    MLLM_RT_ASSERT(kernel_lpbq_gemv_v5_);
    kernel_lpbq_transpose_ = runtime->buildKernel(program_name, "lpbq_transpose_mk_to_km_f16", buildOptions);
    MLLM_RT_ASSERT(kernel_lpbq_transpose_);
  }
}

void OpenCLLinearOp::setLPBQ(cl_mem w_ushort, cl_mem combined_scales, int K, int N, int Bs) {
  MLLM_RT_ASSERT(kernel_lpbq_gemm_v5_);  // requires fp16 support
  lpbq_w_ushort_ = w_ushort;
  lpbq_scales_ = combined_scales;
  lpbq_K_ = K;
  lpbq_N_ = N;
  lpbq_Bs_ = Bs;
  lpbq_enabled_ = true;
}

void OpenCLLinearOp::runLPBQ(cl_mem input_fp16, cl_mem output_fp16, int M) {
  MLLM_RT_ASSERT(lpbq_enabled_);
  auto runtime = std::static_pointer_cast<OpenCLBackend>(Context::instance().getBackend(kOpenCL))->runtime();
  const int K = lpbq_K_, N = lpbq_N_;
  cl_int ret = CL_SUCCESS;

  if (M == 1) {
    // Decode GEMV: activation is the plain [K] fp16 buffer (no transpose).
    auto k = kernel_lpbq_gemv_v5_->get();
    cl_uint a = 0;
    ret |= k.setArg(a++, sizeof(cl_mem), &input_fp16);
    ret |= k.setArg(a++, sizeof(cl_mem), &lpbq_w_ushort_);
    ret |= k.setArg(a++, sizeof(cl_mem), &lpbq_scales_);
    ret |= k.setArg(a++, sizeof(cl_mem), &output_fp16);
    ret |= k.setArg(a++, sizeof(int), &N);
    ret |= k.setArg(a++, sizeof(int), &K);
    if (ret != CL_SUCCESS) { MLLM_ERROR("LPBQ gemv setArg failed: {}", ret); }
    const int lws = 128;
    auto err = runtime->commandQueue().enqueueNDRangeKernel(
        k, cl::NullRange, cl::NDRange((size_t)N * lws), cl::NDRange(lws));
    if (err != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "LPBQ gemv failed: {}", err); }
    return;
  }

  // Prefill GEMM. (1) transpose acts [M,K] -> [K,M] half scratch buffer;
  // (2) wrap it as a half image1d_buffer; (3) dispatch the coalesced v5 GEMM.
  cl_context ctx = runtime->context()();
  cl_int err2 = CL_SUCCESS;
  cl_mem km = OpenCLLoader::instance().clCreateBuffer(
      ctx, CL_MEM_READ_WRITE, (size_t)M * K * sizeof(uint16_t), nullptr, &err2);
  if (err2 != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "LPBQ km alloc failed: {}", err2); }

  {
    auto kt = kernel_lpbq_transpose_->get();
    cl_uint a = 0;
    ret |= kt.setArg(a++, sizeof(cl_mem), &input_fp16);
    ret |= kt.setArg(a++, sizeof(cl_mem), &km);
    ret |= kt.setArg(a++, sizeof(int), &M);
    ret |= kt.setArg(a++, sizeof(int), &K);
    if (ret != CL_SUCCESS) { MLLM_ERROR("LPBQ transpose setArg failed: {}", ret); }
    auto e = runtime->commandQueue().enqueueNDRangeKernel(
        kt, cl::NullRange, cl::NDRange((size_t)M, (size_t)K), cl::NullRange);
    if (e != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "LPBQ transpose failed: {}", e); }
  }

  cl_image_format img_fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc img_desc = {};
  img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc.image_width = (size_t)M * K / 4;
  img_desc.buffer = km;
  cl_mem acts_img = OpenCLLoader::instance().clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &img_desc, nullptr, &err2);
  if (err2 != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "LPBQ acts image failed: {}", err2); }

  {
    auto kg = kernel_lpbq_gemm_v5_->get();
    cl_uint a = 0;
    ret |= kg.setArg(a++, sizeof(cl_mem), &acts_img);
    ret |= kg.setArg(a++, sizeof(cl_mem), &lpbq_w_ushort_);
    ret |= kg.setArg(a++, sizeof(cl_mem), &lpbq_scales_);
    ret |= kg.setArg(a++, sizeof(cl_mem), &output_fp16);
    ret |= kg.setArg(a++, sizeof(int), &M);
    ret |= kg.setArg(a++, sizeof(int), &N);
    ret |= kg.setArg(a++, sizeof(int), &K);
    if (ret != CL_SUCCESS) { MLLM_ERROR("LPBQ gemm setArg failed: {}", ret); }
    cl::NDRange global((size_t)M / 8, (size_t)N / 4);
    cl::NDRange local(8, 16);
    auto e = runtime->commandQueue().enqueueNDRangeKernel(kg, cl::NullRange, global, local);
    if (e != CL_SUCCESS) { MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "LPBQ gemm failed: {}", e); }
  }

  runtime->commandQueue().finish();
  OpenCLLoader::instance().clReleaseMemObject(acts_img);
  OpenCLLoader::instance().clReleaseMemObject(km);
}

void OpenCLLinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);

  // In Linear
  // inputs is always: [..., S, in_channels]
  // outputs is always: [..., S, out_channels]
  int M = input_shape[input_shape.size() - 2];
  int K = input_shape[input_shape.size() - 1];
  int N = options_.out_channels;
  MLLM_RT_ASSERT_EQ(K, options_.in_channels);

  int batch_count = 1;
  for (size_t i = 0; i < input_shape.size() - 2; ++i) { batch_count *= input_shape[i]; }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  auto cl_buffer_input = (cl_mem)input.impl()->storage()->ptr_;
  auto cl_buffer_weight = (cl_mem)weight_.impl()->storage()->ptr_;
  auto cl_buffer_output = (cl_mem)output.impl()->storage()->ptr_;
  cl_mem cl_buffer_bias = cl_buffer_input;  // dummy init
  int has_bias = 0;
  if (options_.bias) {
    cl_buffer_bias = (cl_mem)bias_.impl()->storage()->ptr_;
    has_bias = 1;
  }

  // LPBQ (w4a16) tuned path — only when setLPBQ() has been called. Expects
  // fp16 activations [M,K] and writes fp16 [M,N]. Non-breaking otherwise.
  if (lpbq_enabled_) {
    MLLM_RT_ASSERT(batch_count == 1);
    runLPBQ(cl_buffer_input, cl_buffer_output, M);
    return;
  }

  cl_int ret = CL_SUCCESS;
  std::shared_ptr<KernelWrap> kernel_wrapper;
  cl::NDRange global_size;
  cl::NDRange local_size = cl::NullRange;
  cl_uint index = 0;

  if (weight_.dtype() == DataTypes::kFloat32) {
    kernel_wrapper = kernel_fp32_transb_bias_;

    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_input);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_weight);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_bias);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_output);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &M);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &K);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &N);
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &has_bias);
    int offset_a = input.impl()->storageOffset();
    ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &offset_a);

    // TILE_SIZE = 16
    int tile_size = 16;
    int gws_0 = (N + tile_size - 1) / tile_size * tile_size;
    int gws_1 = (M + tile_size - 1) / tile_size * tile_size;
    int gws_2 = batch_count;

    global_size = cl::NDRange(gws_0, gws_1, gws_2);
    local_size = cl::NDRange(tile_size, tile_size, 1);

  } else if (weight_.dtype() == DataTypes::kGGUF_Q4_0) {
    if (M == 1) {
      kernel_wrapper = kernel_gemv_fp32_q4_0_transb_bias_;

      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_input);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_weight);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_bias);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_output);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &K);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &N);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &has_bias);
      int offset_a = input.impl()->storageOffset();
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &offset_a);

      int local_size_0 = 128;  // Must be <= 256
      int gws_0 = N * local_size_0;
      int gws_1 = batch_count;

      global_size = cl::NDRange(gws_0, gws_1);
      local_size = cl::NDRange(local_size_0, 1);

    } else {
      kernel_wrapper = kernel_fp32_q4_0_transb_bias_;

      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_input);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_weight);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_bias);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(cl_mem), &cl_buffer_output);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &M);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &K);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &N);
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &has_bias);
      int offset_a = input.impl()->storageOffset();
      ret |= kernel_wrapper->get().setArg(index++, sizeof(int), &offset_a);

      int tile_size = 16;
      int gws_0 = (N + tile_size - 1) / tile_size * tile_size;
      int gws_1 = (M + tile_size - 1) / tile_size * tile_size;
      int gws_2 = batch_count;

      global_size = cl::NDRange(gws_0, gws_1, gws_2);
      local_size = cl::NDRange(tile_size, tile_size, 1);
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Unsupported weight data type in LinearOp: {}", weight_.dtype());
  }

  if (ret != CL_SUCCESS) { MLLM_ERROR("OpenCLLinearOp setArg failed: {}", ret); }

  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel_wrapper->get(), cl::NullRange, global_size, local_size);

  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to execute linear kernel, error code: {}", error);
  }
}

void OpenCLLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (options_.isRedirect()) {
    outputs.emplace_back(inputs[1]);
    return;
  }

  MLLM_RT_ASSERT(options_.impl_type == aops::LinearImplTypes::kDefault || options_.impl_type == aops::LinearImplTypes::kGGUF);

  const auto& input = inputs[0];
  auto input_shape = input.shape();
  MLLM_RT_ASSERT(input_shape.size() >= 2);
  MLLM_RT_ASSERT_EQ(input_shape.back(), options_.in_channels);

  auto output_shape = input_shape;
  output_shape.back() = options_.out_channels;

  outputs.emplace_back(Tensor::empty(output_shape, input.dtype(), input.device()));
}

}  // namespace mllm::opencl
