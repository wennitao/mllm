// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Clip.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

bool QnnAOTClipPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::ClipOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTClipPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto clip_op = op->cast_<mllm::ir::linalg::ClipOp>();
  if (!clip_op) {
    MLLM_ERROR("Failed to cast to linalg::ClipOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto base_op = clip_op->getAOp();
  auto aop = dynamic_cast<mllm::aops::ClipOp*>(base_op);
  if (!aop) {
    MLLM_ERROR("Failed to cast base op to aops::ClipOp");
    return false;
  }
  const float min_v = aop->options().min_val;
  const float max_v = aop->options().max_val;

  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // QNN built-in clamp: ReluMinMax(x) = min(max(x, min_value), max_value) (qti.aisw).
  auto qnn_op_node = QnnAOTNodeOperation::create("ReluMinMax");
  qnn_op_node->setPackageName("qti.aisw");
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("min_value", min_v));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("max_value", max_v));
  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(clip_op->getAOp()->getName());

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);
  return true;
}

}  // namespace mllm::qnn::aot
