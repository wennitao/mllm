// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/visitor/Softplus.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

// QNN ElementWiseNeuron operation selector for softplus.
static constexpr uint32_t kQnnNeuronOpSoftplus = 7;  // QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS

bool QnnAOTSoftplusPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  return op->isa_<mllm::ir::linalg::SoftplusOp>() && (op->getAttr("using_qnn") != nullptr);
}

bool QnnAOTSoftplusPattern::rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  auto sp_op = op->cast_<mllm::ir::linalg::SoftplusOp>();
  if (!sp_op) {
    MLLM_ERROR("Failed to cast to linalg::SoftplusOp");
    return false;
  }

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto i_0 = op->inputs().front()->cast_<ir::tensor::TensorValue>();
  auto o_0 = op->outputs().front()->cast_<ir::tensor::TensorValue>();

  // softplus(x) = log(1 + exp(x)) via the parameterized QNN ElementWiseNeuron.
  auto qnn_op_node = QnnAOTNodeOperation::create("ElementWiseNeuron");
  qnn_op_node->setPackageName("qti.aisw");
  qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, i_0));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("operation", kQnnNeuronOpSoftplus));
  qnn_op_node->emplaceParamScalar(QNNParamScalarWrapper::create("beta", 1.0f));
  qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, o_0))
      ->setName(sp_op->getAOp()->getName());

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);
  return true;
}

}  // namespace mllm::qnn::aot
