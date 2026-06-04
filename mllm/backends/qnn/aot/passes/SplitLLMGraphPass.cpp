// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/SplitLLMGraphPass.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

namespace {

void recursiveAttachGraphNameAndContextName(const ir::IRContext::ptr_t& ctx, const std::string& qnn_context_name,
                                            const std::string& qnn_graph_name, ir::graph::SubGraphOp::ptr_t& g) {
  auto _ = ir::IRWriter(ctx, g->getTopRegion());
  _.walk<ir::Op>([&](ir::IRWriter& w /*writer*/, const ir::Op::ptr_t& owo) -> ir::IRWriter::WalkResult {
    if (owo->isa_<ir::graph::CallGraphOp>()) {
      auto p_owo_g =
          ctx->lookupSymbolTable(owo->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
      recursiveAttachGraphNameAndContextName(ctx, qnn_context_name, qnn_graph_name, p_owo_g);
    }
    if (owo->isa_<ir::linalg::LinalgIROp>()) {
      owo->setAttr("qnn_context_name", ctx->create<ir::StrAttr>(qnn_context_name));
      owo->setAttr("qnn_graph_name", ctx->create<ir::StrAttr>(qnn_graph_name));
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
}

void recursiveRemoveOpsIntoNewGraph(const ir::IRContext::ptr_t& ctx, ir::graph::SubGraphOp::ptr_t& g) {
  auto _ = ir::IRWriter(ctx, g->getTopRegion());
  _.walk<ir::Op>([&](ir::IRWriter& w /*writer*/, const ir::Op::ptr_t& owo) -> ir::IRWriter::WalkResult {
    if (owo->isa_<ir::graph::CallGraphOp>()) {
      auto p_owo_g =
          ctx->lookupSymbolTable(owo->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str())->cast_<ir::graph::SubGraphOp>();
      recursiveRemoveOpsIntoNewGraph(ctx, p_owo_g);
    }
    if (owo->isa_<ir::linalg::LinalgIROp>()) {
      auto _g_name = owo->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
      auto p_owo_g = ctx->lookupSymbolTable(_g_name)->cast_<ir::graph::SubGraphOp>();
      auto temp_w = ir::IRWriter(ctx, p_owo_g->getTopRegion());
      w.removeOp(owo);
      temp_w.insertOpAtLast(owo);
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });
}

}  // namespace

uint8_t SplitLLMGraphPass::run(const ir::node_ptr_t& op) {
  // Split-prefill path: the user has already produced one independent subgraph
  // per chunk. There's nothing to SPLIT — but the downstream lowering pass +
  // op visitors require:
  //   - the chunk's SubGraphOp to carry the "use_qnn" attribute
  //     (LLM2QnnLoweringPass:174 reads belongsTo()->getAttr("use_qnn"))
  //   - every linalg op inside the chunk to carry "qnn_graph_name" +
  //     "qnn_context_name" StrAttrs (every visitor's rewrite reads these to
  //     route Qnn AOT capture calls — see Embedding.cpp:28 etc.)
  // The standard split branch sets all of these at lines ~117 + ~177-178 on
  // the freshly-created sub-graphs; we just set them on our pre-split chunk.
  {
    auto& cfg = AOTCompileContext::getInstance().getConfig();
    auto chunk_name = cfg.value("chunk_graph_name", std::string{});
    // The multi-context split routes each chunk to context.<k> (default
    // context.0). This MUST match the context LLM2QnnLoweringPass captures the
    // chunk's graph into — the visitors read this attr to dispatch their QNN
    // capture calls, so a mismatch lands tensors in the wrong/missing context.
    const auto chunk_ctx = cfg.value("chunk_context_name", std::string{"context.0"});
    if (!chunk_name.empty()) {
      MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());
      auto chunk_subgraph_sym = getCtx()->lookupSymbolTable(chunk_name);
      if (chunk_subgraph_sym != nullptr) {
        auto chunk_subgraph = chunk_subgraph_sym->cast_<ir::graph::SubGraphOp>();
        if (chunk_subgraph != nullptr) {
          if (!chunk_subgraph->getAttr("use_qnn")) {
            chunk_subgraph->setAttr("use_qnn", getCtx()->create<ir::BoolAttr>(true));
          }
          // Tag every linalg op in the chunk with qnn_graph_name + qnn_context_name.
          // Visitors at rewrite time read these to dispatch QNN capture calls.
          recursiveAttachGraphNameAndContextName(getCtx(), /*qnn_context_name=*/chunk_ctx,
                                                 /*qnn_graph_name=*/chunk_name, chunk_subgraph);
        }
      }
      return ir::PASS_RET_SUCCESS;
    }
  }

  // The top op should be modelOp
  MLLM_RT_ASSERT(op->isa_<ir::ModuleOp>());

  auto model_op = op->cast_<ir::ModuleOp>();
  auto top_model_writer = ir::IRWriter(getCtx(), model_op->getTopRegion());

  // Check only has 1 call graph op in model_op
  ir::graph::CallGraphOp::ptr_t call_graph_op = nullptr;
  top_model_writer.walk<ir::graph::CallGraphOp>(
      [&](ir::IRWriter& /*writer*/, const ir::graph::CallGraphOp::ptr_t& call_op) -> ir::IRWriter::WalkResult {
        MLLM_RT_ASSERT(call_graph_op == nullptr);  // Should only have one CallGraphOp
        call_graph_op = call_op;
        return ir::IRWriter::WalkResult::WALK_CONTINUE;
      });

  if (call_graph_op == nullptr) {
    MLLM_ERROR("LLM2QnnLoweringPass: No CallGraphOp found in ModuleOp");
    return ir::PASS_RET_FAILURE;
  }

  // Check call graph op point to a subgraph named "model"
  auto symbol_attr = call_graph_op->getSymbolAttr();
  if (symbol_attr == nullptr || symbol_attr->str() != "model") {
    MLLM_ERROR("LLM2QnnLoweringPass: CallGraphOp should point to a subgraph named 'model'");
    return ir::PASS_RET_FAILURE;
  }

  // Get the "model" subgraph
  auto model_subgraph = getCtx()->lookupSymbolTable("model")->cast_<ir::graph::SubGraphOp>();
  if (model_subgraph == nullptr) {
    MLLM_ERROR("LLM2QnnLoweringPass: Cannot find 'model' subgraph in symbol table");
    return ir::PASS_RET_FAILURE;
  }

  // Split all layers, and fuse some op into one layer.
  // e.g.
  // op0
  // op1
  // op2
  // layer.0
  // layer.1
  // op3
  // When we split graphs. op0, op1 and op2 will be merged into layer.0. And op3 will be merged into op3

  // Count how many layers we have.
  auto cfg = AOTCompileContext::getInstance().getConfig();
  int32_t __global_total_layers = cfg["quant_recipe"]["layers"];
  int32_t __global_split_graphs = cfg["split_graph"];

  // Check seq length
  // FIXME: We suppose the first input to LLM is tokend_ids! Whose shape is [Batch, Sequence]
  int32_t __global_seq_len =
      model_subgraph->getTopRegion()->inputs().front()->cast_<ir::tensor::TensorValue>()->tensor_.size(1);

  // Create merged graph first!
  for (int i = 0; i < __global_split_graphs; ++i) {
    auto op = top_model_writer.create<ir::graph::SubGraphOp>(
        top_model_writer.create<ir::SymbolAttr>("model." + std::to_string(i) + ".s" + std::to_string(__global_seq_len)));
    op->setAttr("use_qnn", top_model_writer.create<ir::BoolAttr>(true));
  }

  // Solve op's scopes. Attach qnn_context_name and qnn_graph_name on them.
  // Suppose all layer's name is xxx.xxx.number
  {
    int graph_counter = 0;
    auto model_graph_writer = ir::IRWriter(getCtx(), model_subgraph->getTopRegion());
    model_graph_writer.walk<ir::Op>([&](ir::IRWriter& w /*writer*/, const ir::Op::ptr_t& one_op) -> ir::IRWriter::WalkResult {
      if (one_op->isa_<ir::graph::CallGraphOp>()) {
        auto g_w_g = getCtx()
                         ->lookupSymbolTable(one_op->cast_<ir::graph::CallGraphOp>()->getSymbolAttr()->str())
                         ->cast_<ir::graph::SubGraphOp>();

        auto g_w_g_name = g_w_g->getSymbolAttr()->str();

        // Extract layer number from g_w_g_name (format: xxx.xxx.number)
        // The layer number is the last part after the final dot
        size_t last_dot_pos = g_w_g_name.find_last_of('.');
        int layer_num = 0;
        if (last_dot_pos != std::string::npos && last_dot_pos + 1 < g_w_g_name.length()) {
          layer_num = std::stoi(g_w_g_name.substr(last_dot_pos + 1));
        }

        // Calculate which graph counter to use based on layer number
        // Each graph will contain approximately __global_total_layers / __global_split_graphs layers
        if (__global_split_graphs > 0) {
          int layers_per_graph = __global_total_layers / __global_split_graphs;
          graph_counter = layer_num / layers_per_graph;
          // Ensure graph_counter doesn't exceed the number of split graphs
          if (graph_counter >= __global_split_graphs) { graph_counter = __global_split_graphs - 1; }
        }

        recursiveAttachGraphNameAndContextName(
            getCtx(), "context." + std::to_string(graph_counter),
            "model." + std::to_string(graph_counter) + ".s" + std::to_string(__global_seq_len), g_w_g);
      }

      if (one_op->isa_<ir::linalg::LinalgIROp>()) {
        one_op->setAttr("qnn_context_name", getCtx()->create<ir::StrAttr>("context." + std::to_string(graph_counter)));
        one_op->setAttr("qnn_graph_name", getCtx()->create<ir::StrAttr>("model." + std::to_string(graph_counter) + ".s"
                                                                        + std::to_string(__global_seq_len)));
      }
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });
  }

  // Loop all ops in model_subgraph recursively. Using qnn_graph_name to merge those ops in on exists graph.
  recursiveRemoveOpsIntoNewGraph(getCtx(), model_subgraph);

  // Solve the inputs and output of splitted graph
  {
    std::vector<ir::Val::ptr_t> original_inputs;
    std::vector<ir::Val::ptr_t> original_outputs;
    for (auto item : model_subgraph->inputs()) { original_inputs.emplace_back(item->cast_<ir::Val>()); }
    for (auto item : model_subgraph->outputs()) { original_outputs.emplace_back(item->cast_<ir::Val>()); }

    // FIXME: currently only support one graph!
    MLLM_RT_ASSERT_EQ(__global_split_graphs, 1);
    auto one_graph =
        getCtx()->lookupSymbolTable("model.0.s" + std::to_string(__global_seq_len))->cast_<ir::graph::SubGraphOp>();
    for (auto item : original_inputs) {
      one_graph->inputs().emplace_back(item);
      one_graph->getTopRegion()->inputs().emplace_back(item);
    }
    for (auto item : original_outputs) {
      one_graph->outputs().emplace_back(item);
      one_graph->getTopRegion()->outputs().emplace_back(item);
    }
    auto wwww = ir::IRWriter(getCtx(), one_graph->getTopRegion());
    auto return_op = wwww.create<ir::cf::ReturnOp>(original_outputs);
  }

  // Remove old graphs
  {
    top_model_writer.walk<ir::graph::SubGraphOp>(
        [&](ir::IRWriter& wvw, const ir::graph::SubGraphOp::ptr_t& sub_g_op) -> ir::IRWriter::WalkResult {
          auto name = sub_g_op->getSymbolAttr()->str();
          bool matched = false;
          for (int i = 0; i < __global_split_graphs; ++i) {
            if (name == "model." + std::to_string(i) + ".s" + std::to_string(__global_seq_len)) { matched = true; }
            if (name == "model") { matched = true; }
          }
          if (!matched && name != "init") { wvw.removeOp(sub_g_op); }
          return ir::IRWriter::WalkResult::WALK_CONTINUE;
        });
  }

  // Insert call graph ops into top model subgraph.
  {
    // 1. remove all call graph ops in model_subgraph
    auto model_graph_writer = ir::IRWriter(getCtx(), model_subgraph->getTopRegion());
    model_graph_writer.walk<ir::Op>([&](ir::IRWriter& wvw /*writer*/, const ir::Op::ptr_t& one_op) -> ir::IRWriter::WalkResult {
      if (one_op->isa_<ir::graph::CallGraphOp>()) { wvw.removeOp(one_op); }
      if (one_op->isa_<ir::cf::ControlFlowIROp>()) { wvw.removeOp(one_op); }
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });

    // 2. Insert new call ops.
    // FIXME: currently only support one graph!
    MLLM_RT_ASSERT_EQ(__global_split_graphs, 1);
    auto call_op = model_graph_writer.create<ir::graph::CallGraphOp>(
        model_graph_writer.create<ir::SymbolAttr>("model.0.s" + std::to_string(__global_seq_len)));
    std::vector<ir::Val::ptr_t> original_inputs;
    std::vector<ir::Val::ptr_t> original_outputs;
    for (auto item : model_subgraph->inputs()) { (*item)-- > call_op; }
    for (auto item : model_subgraph->outputs()) {
      (*call_op)-- > item->cast_<ir::Val>();
      original_outputs.emplace_back(item->cast_<ir::Val>());
    }
    auto return_op = model_graph_writer.create<ir::cf::ReturnOp>(original_outputs);
  }

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createSplitLLMGraphPass() { return std::make_shared<SplitLLMGraphPass>(); }

}  // namespace mllm::qnn::aot
