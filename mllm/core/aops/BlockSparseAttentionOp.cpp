// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/BlockSparseAttentionOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

BlockSparseAttentionOp::BlockSparseAttentionOp(const BlockSparseAttentionOpOptions& options)
    : BaseOp(OpTypes::kBlockSparseAttention), options_(options) {}

void BlockSparseAttentionOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void BlockSparseAttentionOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                                   std::vector<Tensor>& outputs) {
  // No IR/linalg lowering yet — eager-only op (invoked via the OpenCL backend
  // op directly). Tracing into a compiled graph is future work.
  NYI("BlockSparseAttentionOp::trace not implemented (eager-only op).");
}

void BlockSparseAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("BlockSparseAttentionOp::forward not implemented in aops base.");
}

void BlockSparseAttentionOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // O has the same shape as Q: [B, H, S_q, D].
  const auto& q = inputs[0];
  outputs.emplace_back(Tensor::empty(q.shape(), q.dtype(), q.device()));
}

void BlockSparseAttentionOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
