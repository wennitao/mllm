// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

namespace mllm::opencl {

class OpenCLFlashAttention2Op final : public aops::FlashAttention2Op {
 public:
  explicit OpenCLFlashAttention2Op(const aops::FlashAttention2OpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_prefill_;
  std::shared_ptr<KernelWrap> kernel_decode_;
};

class OpenCLFlashAttention2OpFactory : public TypedOpFactory<OpTypes::kFlashAttention2, aops::FlashAttention2OpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FlashAttention2OpOptions& options) override {
    return std::make_shared<OpenCLFlashAttention2Op>(options);
  }
};

}  // namespace mllm::opencl
