// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Minimal AOT compile for the new elementwise QNN visitors (Exp / Log), used to
// validate that they lower to a finalizable QNN graph and run on the HTP device.
//
// Graph: x -> QDQ -> {Exp | Log} -> QDQ -> y   (single "model.0.s<Seq>" graph)
// Synthetic uint16-asym activation scales (values arbitrary; only shapes/encodings
// must satisfy the compiler + QNN finalize). This is the elementwise analogue of
// compile_mlp_lpbq_microbench.cpp.
//
// Usage:
//   ./mllm-qwen3-aot-exp-microbench-c -aot_cfg qnn_aot_cfg_exp_microbench.json [--op exp] [--seq 64] [--width 128]
//
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/passes/AOTCompileContext.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"  // brings ptq::QDQ + Qwen3Config

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) { return std::string(qairt_root) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
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
// x -> QDQ -> {exp|log} -> QDQ. The unary op shares its input quant spec to its
// output; the closing QDQ gives the output a standalone solved scale.
class ExpUnaryTest final : public nn::Module {
  std::string op_;
 public:
  ExpUnaryTest() = default;
  ExpUnaryTest(const std::string& name, const std::string& op) : nn::Module(name), op_(op) {}
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = ptq::QDQ(this, inputs[0], "in_qdq");
    Tensor y = (op_ == "log")     ? nn::functional::log(x)
               : (op_ == "rsqrt") ? nn::functional::rsqrt(x)
                                   : nn::functional::exp(x);
    return {ptq::QDQ(this, y, "out_qdq")};
  }
};
}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath()).help("QNN AOT env path.");
  auto& op_arg = Argparse::add<std::string>("--op").help("exp | log").def("exp");
  auto& seq_arg = Argparse::add<int>("--seq").help("Sequence length (M)").def(64);
  auto& width_arg = Argparse::add<int>("--width").help("feature width").def(128);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!qnn_aot_cfg_files.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const std::string op = op_arg.get();
  const int Seq = seq_arg.get(), W = width_arg.get();

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  pushQDQ(params, "model.in_qdq");
  pushQDQ(params, "model.out_qdq");

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));
  const std::string bin = "qwen3-" + op + "-microbench.bin";

  sha::ExpUnaryTest m("model", op);
  m.load(params);
  auto x = mllm::Tensor::zeros({1, Seq, W}, mllm::kFloat16).setName("x0");
  auto ir = mllm::ir::trace_(m, {x});
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
  pm.run();
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print("Elementwise {} microbench Seq={} W={} -> {} (graph model.0.s{})", op, Seq, W, bin, Seq);
});
