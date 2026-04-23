// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/TopKMaskOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

TopKMaskOp::TopKMaskOp(const TopKMaskOpOptions& options) : BaseOp(OpTypes::kTopKMask), options_(options) {}

void TopKMaskOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void TopKMaskOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::TopKMaskOp>(shared_from_this(), i_irs, o_irs);
}

void TopKMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("TopKMaskOp::forward not implemented in aops base.");
}

void TopKMaskOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  outputs.emplace_back(Tensor::empty(i.shape(), i.dtype(), i.device()));
}

void TopKMaskOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
