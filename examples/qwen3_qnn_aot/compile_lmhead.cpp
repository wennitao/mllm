// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED head graph for the Qwen3.5 decode loop: final RMSNorm + lm_head, the
// last stage of the whole-model-on-NPU pipeline. lm_head is the tied embed_tokens
// matrix (hidden=2048 -> vocab=248320) as an LPBQ Conv2D (int4 weight, uint16 act);
// the final RMSNorm is hand-rolled in fp32 (RMSNormOp requires a uint16 weight) with
// the norm weight passed as a graph input (add_unit_offset 1+w baked at export).
//
//   inputs : x[1,hidden], norm_w[1,1,hidden], eps[1,1,1]
//   outputs: logits[1,vocab]
//
//   ./mllm-qwen3-aot-lmhead-c -aot_cfg qnn_aot_cfg_lmhead.json --params lmhead-lpbq.mllm
//
#include <cstdint>
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

#include "modeling_qwen_qnn_aot_sha.hpp"  // CONV2D_PROPERTY + ptq::QDQ helpers

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

class LmHeadLPBQ final : public nn::Module {
  int hidden_ = 2048, vocab_ = 248320;
  nn::Conv2D lm_head_;

 public:
  LmHeadLPBQ() = default;
  LmHeadLPBQ(const std::string& name, int hidden, int vocab) : nn::Module(name), hidden_(hidden), vocab_(vocab) {
    lm_head_ = reg<nn::Conv2D>("lm_head", hidden, vocab, CONV2D_PROPERTY);
  }

  // inputs: x[1,hidden], norm_w[1,1,hidden], eps[1,1,1]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0];
    auto norm_w = in[1];
    auto eps = in[2];
    // hand-rolled final RMSNorm over hidden, fp32, weight = 1+w (baked).
    auto xf = x.to(kFloat32);
    auto inv = F::rsqrt(F::mean(xf * xf, -1, true) + eps.to(kFloat32));
    auto xn = ((xf * inv) * norm_w.to(kFloat32)).to(kFloat16);
    // LPBQ lm_head: hidden -> vocab.
    auto xq = ptq::QDQ(this, xn, "lmhead_input_qdq").view({1, 1, -1, hidden_}, true);
    auto logits = ptq::QDQ(this, lm_head_(xq), "lmhead_output_qdq").to(kFloat16).view({1, vocab_}, true);
    return {logits};
  }
};

}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& params_arg = Argparse::add<std::string>("--params").help("real LPBQ .mllm (from export_lmhead.py); else synthetic").def("");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& vocab_arg = Argparse::add<int>("--vocab").def(248320);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int hidden = hidden_arg.get(), vocab = vocab_arg.get(), G = 16;

  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"));
  ti.push_back(Tensor::zeros({1, 1, hidden}, mllm::kFloat16).setName("norm_w"));
  ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("eps"));

  auto compile = [&](const mllm::ParameterFile::ptr_t& params) {
    auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
        qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
    sha::LmHeadLPBQ m("model", hidden, vocab);
    m.load(params);
    auto ir = mllm::ir::trace_(m, ti);
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
    pm.run();
    const std::string bin = "qwen3-lmhead.bin";
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print(fmt::format("LmHead (LPBQ) hidden={} vocab={} -> {} (graph model.0.s{})", hidden, vocab, bin, hidden));
  };

  if (!params_arg.get().empty()) {
    auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    compile(real);
    return 0;
  }

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  pushLPBQConv(params, "model.lm_head", hidden, vocab, G);
  pushQDQ(params, "model.lmhead_input_qdq");
  pushQDQ(params, "model.lmhead_output_qdq");
  compile(params);
});
