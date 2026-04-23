// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/TopKMaskOp.hpp"

namespace mllm::nn {

class TopKMask : public Layer {
 public:
  TopKMask();

  explicit TopKMask(int32_t top_k);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
