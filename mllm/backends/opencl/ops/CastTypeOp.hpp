// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/CastTypeOp.hpp"
#include <vector>

namespace mllm::opencl {

class OpenCLCastTypeOp final : public aops::CastTypeOp {
 public:
  explicit OpenCLCastTypeOp(const aops::CastTypeOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_f32_to_f16_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_f16_to_f32_ = nullptr;
};

class OpenCLCastTypeOpFactory : public TypedOpFactory<OpTypes::kCastType, aops::CastTypeOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::CastTypeOpOptions& options) override {
    return std::make_shared<OpenCLCastTypeOp>(options);
  }
};

}  // namespace mllm::opencl
