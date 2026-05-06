// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/layers/MatMul.hpp"

namespace mllm::nn {

MatMul::MatMul() : Layer(OpTypes::kMatMul, aops::MatMulOpOptions{}) {}

MatMul::MatMul(bool transpose_a, bool transpose_b, aops::MatMulOpType matmul_type)
    : Layer(OpTypes::kMatMul,
            aops::MatMulOpOptions{
                .transpose_a = transpose_a,
                .transpose_b = transpose_b,
                .matmul_type = matmul_type,
            }) {}

MatMul::MatMul(const aops::MatMulOpOptions& options) : Layer(OpTypes::kMatMul, options) {}

}  // namespace mllm::nn
