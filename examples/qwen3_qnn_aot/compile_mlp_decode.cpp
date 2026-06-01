// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED MLP decode-step graph (Sq=1) for the Qwen3.5 whole-model pipeline.
// Wraps the production Qwen3MLP (gate/up/down LPBQ Conv2D + SiLU) and adds the
// closing down_proj_output_qdq. Consumes the per-layer bundle from
// export_whole_model.py (model.{gate,up,down}_proj.* + the 7 QDQ scales).
//
//   inputs : x[1,hidden]   outputs: y[1,hidden]
//   ./mllm-qwen3-aot-mlp-decode-c -aot_cfg qnn_aot_cfg_lmhead.json --params layer0_mlp-lpbq.mllm --out l0_mlp.bin
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

void pushLPBQConv(const mllm::ParameterFile::ptr_t& params, const std::string& prefix, int In, int Out, int G) {
  const int n_blk = In / G;
  std::vector<int8_t> w((size_t)In * Out);
  for (size_t i = 0; i < w.size(); ++i) w[i] = (int8_t)(i % 15);
  std::vector<uint8_t> s1((size_t)Out * n_blk);
  for (size_t i = 0; i < s1.size(); ++i) s1[i] = (uint8_t)(1 + i % 15);
  std::vector<float> s2((size_t)Out);
  for (size_t o = 0; o < s2.size(); ++o) s2[o] = 0.005f + 0.0001f * (o % 17);
  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(prefix + ".weight", Tensor::fromVector(w, {1, 1, In, Out}, mllm::kInt8));
  push(prefix + ".scale1", Tensor::fromVector(s1, {(int)s1.size()}, mllm::kUInt8));
  push(prefix + ".scale2", Tensor::fromVector(s2, {(int)s2.size()}, mllm::kFloat32));
}
void pushQDQ(const mllm::ParameterFile::ptr_t& params, const std::string& qdq) {
  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(qdq + ".fake_quant.scale", Tensor::fromVector(std::vector<float>{1.0f / 256.0f}, {1}, mllm::kFloat32));
  push(qdq + ".fake_quant.zero_point", Tensor::fromVector(std::vector<int32_t>{0}, {1}, mllm::kInt32));
}
}  // namespace

namespace mllm::models::qwen3::sha {
// Full Qwen3 MLP (gate/up/silu/mul/down, all LPBQ) + closing QDQ on the down output.
class FullMLPDecode final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_ = 2048;

 public:
  FullMLPDecode() = default;
  FullMLPDecode(const std::string& name, const Qwen3Config& cfg) : nn::Module(name), hidden_(cfg.hidden_size) {
    mlp_ = reg<Qwen3MLP>("", cfg);  // convs at model.gate_proj etc.
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = mlp_.forward(inputs, args)[0];
    return {ptq::QDQ(this, o, "down_proj_output_qdq").to(kFloat16).view({1, hidden_}, true)};
  }
};
}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& params_arg = Argparse::add<std::string>("--params").help("real MLP LPBQ .mllm; else synthetic").def("");
  auto& out_arg = Argparse::add<std::string>("--out").help("output .bin name").def("qwen3-mlp-decode.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& inter_arg = Argparse::add<int>("--inter").def(6144);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int hidden = hidden_arg.get(), inter = inter_arg.get(), G = 16;
  mllm::models::qwen3::Qwen3Config cfg;
  cfg.hidden_size = hidden;
  cfg.intermediate_size = inter;

  std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x")};

  auto compile = [&](const mllm::ParameterFile::ptr_t& params) {
    auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
        qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
    sha::FullMLPDecode m("model", cfg);
    m.load(params);
    auto ir = mllm::ir::trace_(m, ti);
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
    pm.run();
    qnn_aot_env.saveContext("context.0", out_arg.get());
    mllm::print(fmt::format("MLP decode (LPBQ) hidden={} inter={} -> {} (graph model.0.s{})", hidden, inter,
                            out_arg.get(), hidden));
  };

  if (!params_arg.get().empty()) {
    auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    compile(real);
    return 0;
  }
  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  pushLPBQConv(params, "model.gate_proj", hidden, inter, G);
  pushLPBQConv(params, "model.up_proj", hidden, inter, G);
  pushLPBQConv(params, "model.down_proj", inter, hidden, G);
  for (auto q : {"up_proj_input_qdq", "up_proj_output_qdq", "gate_proj_output_qdq", "sigmoid_output_qdq",
                 "act_output_qdq", "down_proj_input_qdq", "down_proj_output_qdq"})
    pushQDQ(params, std::string("model.") + q);
  compile(params);
});
