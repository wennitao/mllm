// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/TopKMaskOp.hpp"
#include "mllm/nn/layers/TopKMask.hpp"

namespace mllm::nn {

TopKMask::TopKMask() : Layer(OpTypes::kTopKMask, aops::TopKMaskOpOptions()) {}

TopKMask::TopKMask(int32_t top_k) : Layer(OpTypes::kTopKMask, aops::TopKMaskOpOptions(top_k)) {}

}  // namespace mllm::nn
