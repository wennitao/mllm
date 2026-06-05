// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile a whole LFM2 MoE block: all N experts as DISTINCTLY-NAMED graphs (expert0..expertN-1)
// in ONE QNN context, so the runtime loads one .bin and host-dispatches the top-k selected experts.
//
// Reads the per-expert W4A16-LPBQ bundles produced by lfm2_moe/export_moe_block.py
// (<dir>/expert{e}-lpbq.mllm, "model." keys). Each expert == Qwen3MLP (gate/up/down LPBQ + QDQ);
// we re-key its bundle to "expert{e}." and route it through the SplitLLMGraph chunk path so its
// QNN graph keeps the name "expert{e}".
//
//   ./mllm-lfm2-aot-moe-block-c -aot_cfg qnn_aot_cfg_mlp_lpbq_microbench.json \
//       --bundles_dir wm_moe_l2 --num_experts 32 --inter 1792 --out moe_block_l2.bin
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
// One LFM2 expert MLP (gate/up/silu/mul/down LPBQ) + closing QDQ on the down output.
class Lfm2ExpertGraph final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_ = 2048, seq_ = 1;

 public:
  Lfm2ExpertGraph() = default;
  Lfm2ExpertGraph(const std::string& name, const Qwen3Config& cfg, int seq = 1)
      : nn::Module(name), hidden_(cfg.hidden_size), seq_(seq) {
    mlp_ = reg<Qwen3MLP>("", cfg);
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
  auto& dir_arg = Argparse::add<std::string>("--bundles_dir").help("dir with expert{e}-lpbq.mllm").required(true);
  auto& ne_arg = Argparse::add<int>("--num_experts").def(32);
  auto& out_arg = Argparse::add<std::string>("--out").help("output .bin name").def("lfm2-moe-block.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& inter_arg = Argparse::add<int>("--inter").def(1792);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int hidden = hidden_arg.get(), inter = inter_arg.get(), NE = ne_arg.get();
  mllm::models::qwen3::Qwen3Config cfg;
  cfg.hidden_size = hidden;
  cfg.intermediate_size = inter;

  // One env accumulates all expert graphs into context.0.
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));

  for (int e = 0; e < NE; ++e) {
    const std::string name = "expert" + std::to_string(e);
    const std::string bundle = dir_arg.get() + "/expert" + std::to_string(e) + "-lpbq.mllm";

    auto src = mllm::load(bundle, mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
    for (const auto& kv : src->dict()) {
      const std::string& key = kv.first;
      std::string nk = (key.rfind("model.", 0) == 0) ? (name + "." + key.substr(6)) : key;
      auto t = kv.second;
      params->push(nk, t.setMemType(mllm::kParamsNormal).setName(nk));
    }

    sha::Lfm2ExpertGraph m(name, cfg, 1);
    m.load(params);
    std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x")};
    auto ir = mllm::ir::trace_(m, ti);

    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
    {  // override AFTER pipeline build, BEFORE run (pipeline re-reads the cfg file each call)
      auto& c = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
      c["graph_on_qnn"] = nlohmann::json::array({name});
      c["chunk_graph_name"] = name;
      c["chunk_context_name"] = "context.0";
    }
    pm.run();
    mllm::print(fmt::format("  [{}/{}] graph {} lowered", e + 1, NE, name));
  }

  qnn_aot_env.saveContext("context.0", out_arg.get());
  mllm::print(fmt::format("LFM2 MoE block: {} experts (hidden={} inter={}) -> {} (graphs expert0..expert{})",
                          NE, hidden, inter, out_arg.get(), NE - 1));
});
