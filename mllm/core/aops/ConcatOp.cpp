// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ConcatOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

ConcatOp::ConcatOp(const ConcatOpOptions& options) : BaseOp(OpTypes::kConcat), options_(options) {}

void ConcatOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ConcatOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ConcatOp>(shared_from_this(), i_irs, o_irs);
}

void ConcatOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("ConcatOp::forward not implemented in aops base.");
}

void ConcatOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto at_dim = options_.dim;
  if (inputs.empty()) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "ConcatOp: no inputs");
    return;
  }
  const int n_dims = inputs[0].shape().size();
  if (at_dim < 0) { at_dim += n_dims; }
  if (at_dim >= n_dims) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "ConcatOp: dim {} out of range [0, {})", at_dim, n_dims);
    return;
  }

  // Rank check FIRST. The original code only iterated `n_dims = inputs[0].shape().size()`
  // and compared positionally — if inputs had different ranks, the loop would
  // silently compare unrelated dims (or worse, accept a mismatch coincidence,
  // e.g., when seq_len happens to equal head_dim). Catch it here with shape
  // dump for debugging.
  auto dump_shapes = [&]() {
    std::string s;
    for (size_t j = 0; j < inputs.size(); ++j) {
      s += "[" + std::to_string(j) + "]=(";
      for (int dd = 0; dd < (int)inputs[j].shape().size(); ++dd) {
        s += std::to_string(inputs[j].shape()[dd]);
        if (dd + 1 < (int)inputs[j].shape().size()) s += ",";
      }
      s += ") ";
    }
    return s;
  };
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].shape().size() != inputs[0].shape().size()) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError,
                      "ConcatOp: rank mismatch at input[{}] ({}D vs {}D) shapes={}",
                      i, inputs[i].shape().size(), inputs[0].shape().size(), dump_shapes());
      return;
    }
  }

  // Check non-concat dim sizes.
  for (int d = 0; d < n_dims; ++d) {
    if (d == at_dim) continue;
    const int ref = inputs[0].shape()[d];
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i].shape()[d] != ref) {
        MLLM_ERROR_EXIT(ExitCode::kCoreError,
                        "ConcatOp: at_dim={} non-concat dim {} mismatch ({} vs {}) shapes={}",
                        at_dim, d, ref, inputs[i].shape()[d], dump_shapes());
        return;
      }
    }
  }

  std::vector<int> new_shape = inputs[0].shape();
  new_shape[at_dim] = 0;
  for (const auto& t : inputs) { new_shape[at_dim] += t.shape()[at_dim]; }

  outputs.emplace_back(Tensor::empty(new_shape, inputs[0].dtype(), inputs[0].device()));
}

void ConcatOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops
