// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct TopKMaskOpOptions : public BaseOpOptions<TopKMaskOpOptions> {
  int32_t top_k = 64;

  TopKMaskOpOptions() = default;
  explicit TopKMaskOpOptions(int32_t k) : top_k(k) {}
};

class TopKMaskOp : public BaseOp {
 public:
  explicit TopKMaskOp(const TopKMaskOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const TopKMaskOpOptions& options() const { return options_; }

 protected:
  TopKMaskOpOptions options_;
};

}  // namespace mllm::aops
