// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/MatMulOp.hpp"

namespace mllm::nn {

// Thin wrapper that turns the kMatMul op into a registerable Layer, so calls
// to it pass through Layer::__main and can be timed by ModuleProfiler.
// Functional equivalent: nn::functional::matmul(A, B, transpose_a, transpose_b).
class MatMul : public Layer {
 public:
  MatMul();

  explicit MatMul(bool transpose_a, bool transpose_b = false,
                  aops::MatMulOpType matmul_type = aops::MatMulOpType::kDefault);

  explicit MatMul(const aops::MatMulOpOptions& options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
