// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/TopKMaskOp.hpp"

namespace mllm::cpu {

class CPUTopKMaskOp final : public aops::TopKMaskOp {
 public:
  explicit CPUTopKMaskOp(const aops::TopKMaskOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUTopKMaskOpFactory : public TypedOpFactory<OpTypes::kTopKMask, aops::TopKMaskOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::TopKMaskOpOptions& options) override {
    return std::make_shared<CPUTopKMaskOp>(options);
  }
};

}  // namespace mllm::cpu
