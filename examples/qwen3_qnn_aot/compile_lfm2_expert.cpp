// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile ONE LFM2 MoE expert SwiGLU MLP (W4A16 LPBQ) as a DISTINCTLY-NAMED graph so
// many experts can be co-loaded (initQnnBackendGroup) and host-dispatched per token.
//
// An LFM2 expert == Qwen3MLP (gate/up/down LPBQ + closing QDQ). The only differences vs
// compile_mlp_decode are: (1) the module is named `--name` (e.g. "expert8") so the traced
// subgraph + emitted QNN graph take that name via the SplitLLMGraphPass chunk path
// (cfg["chunk_graph_name"]); (2) the bundle (exported with the "model." prefix) is re-keyed
// to that name at load. inter defaults to moe_intermediate_size=1792.
//
//   ./mllm-lfm2-aot-expert-c -aot_cfg qnn_aot_cfg_mlp_lpbq_microbench.json \
//       --name expert8 --params expert8-lpbq.mllm --inter 1792 --out expert8.bin
//   -> graph "expert8"  (run: QnnAOTModule("expert8"))
//
#include <cstdlib>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/passes/AOTCompileContext.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"  // Qwen3MLP + CONV2D_PROPERTY + ptq::QDQ

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
}  // namespace

namespace mllm::models::qwen3::sha {
// Full LFM2 expert MLP (gate/up/silu/mul/down, all LPBQ) + closing QDQ on the down output.
class Lfm2ExpertDecode final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_ = 2048, seq_ = 1;

 public:
  Lfm2ExpertDecode() = default;
  Lfm2ExpertDecode(const std::string& name, const Qwen3Config& cfg, int seq = 1)
      : nn::Module(name), hidden_(cfg.hidden_size), seq_(seq) {
    mlp_ = reg<Qwen3MLP>("", cfg);  // convs at <name>.gate_proj etc.
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = mlp_.forward(inputs, args)[0];
    return {ptq::QDQ(this, o, "down_proj_output_qdq").to(kFloat16).view({seq_, hidden_}, true)};
  }
};
}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& name_arg = Argparse::add<std::string>("--name").help("graph/module name, e.g. expert8").def("expert0");
  auto& params_arg = Argparse::add<std::string>("--params").help("expert LPBQ .mllm (model.* keys)").required(true);
  auto& out_arg = Argparse::add<std::string>("--out").help("output .bin name").def("lfm2-expert.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& inter_arg = Argparse::add<int>("--inter").def(1792);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int hidden = hidden_arg.get(), inter = inter_arg.get();
  const std::string name = name_arg.get();
  mllm::models::qwen3::Qwen3Config cfg;
  cfg.hidden_size = hidden;
  cfg.intermediate_size = inter;

  // Load the bundle (exported with "model." prefix) and re-key to "<name>." so the
  // module's param lookups (<name>.gate_proj.weight, <name>.up_proj_input_qdq, ...) resolve.
  auto src = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  for (const auto& kv : src->dict()) {
    const std::string& key = kv.first;
    std::string nk = (key.rfind("model.", 0) == 0) ? (name + "." + key.substr(6)) : key;
    auto t = kv.second;
    params->push(nk, t.setMemType(mllm::kParamsNormal).setName(nk));
  }

  std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x")};

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::Lfm2ExpertDecode m(name, cfg, 1);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);

  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  // createQnnAOTLoweringPipeline re-reads the config FILE each call, so override the
  // in-memory config AFTER the pipeline is built but BEFORE pm.run() reads it:
  //  - graph_on_qnn -> [<name>] so MarkQnnGraphPass marks our graph (cfg file says "model").
  //  - chunk_graph_name -> <name> so SplitLLMGraph/MergeLLMHead/LLM2QnnLowering operate on the
  //    pre-traced subgraph <name> and keep that distinct QNN graph name (not "model.0.s<N>").
  {
    auto& c = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
    c["graph_on_qnn"] = nlohmann::json::array({name});
    c["chunk_graph_name"] = name;
    c["chunk_context_name"] = "context.0";
  }
  pm.run();
  qnn_aot_env.saveContext("context.0", out_arg.get());
  mllm::print(fmt::format("LFM2 expert '{}' (LPBQ) hidden={} inter={} -> {} (graph {})", name, hidden, inter,
                          out_arg.get(), name));
});
