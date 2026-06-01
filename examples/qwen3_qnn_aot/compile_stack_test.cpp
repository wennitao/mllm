// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// MONOLITHIC FINALIZE FEASIBILITY TEST. Chains N LPBQ MLP blocks into ONE graph /
// ONE context (synthetic weights) to answer the question that decides the whole-
// model-on-NPU architecture: does a single QNN graph holding ~1 GB of LPBQ weights
// FINALIZE on V79/SM8750 at decode (Sq=1)?
//
// MLPs are the dominant weight (~18 MB int4 each, ~450 MB for all 24) and are
// single-in/single-out so trivial to chain; each mixer (deltanet 12 MB / attn 8.5
// MB) and the 291 MB head already finalize alone. So if N=24 MLP blocks finalize
// as one graph, the full ~1 GB monolithic decode graph is memory-feasible and the
// monolithic path is viable (no 49-context name-collision problem).
//
//   ./mllm-qwen3-aot-stack-test-c -aot_cfg qnn_aot_cfg_lmhead.json --layers 24
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

#include "modeling_qwen_qnn_aot_sha.hpp"  // Qwen3MLP + CONV2D_PROPERTY + ptq::QDQ

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
void pushLPBQConv(const mllm::ParameterFile::ptr_t& p, const std::string& prefix, int In, int Out, int G, int seed = 0) {
  const int n_blk = In / G;
  // Vary by `seed` (block index) so blocks have DISTINCT weights — else QNN dedups
  // identical static tensors and the context size no longer reflects real memory.
  std::vector<int8_t> w((size_t)In * Out);
  for (size_t i = 0; i < w.size(); ++i) w[i] = (int8_t)((i + 7 * seed) % 15);
  std::vector<uint8_t> s1((size_t)Out * n_blk);
  for (size_t i = 0; i < s1.size(); ++i) s1[i] = (uint8_t)(1 + (i + 3 * seed) % 15);
  std::vector<float> s2((size_t)Out);
  for (size_t o = 0; o < s2.size(); ++o) s2[o] = 0.005f + 0.0001f * ((o + seed) % 17);
  auto push = [&](const std::string& k, Tensor t) { p->push(k, t.contiguous().setMemType(mllm::kParamsNormal).setName(k)); };
  push(prefix + ".weight", Tensor::fromVector(w, {1, 1, In, Out}, mllm::kInt8));
  push(prefix + ".scale1", Tensor::fromVector(s1, {(int)s1.size()}, mllm::kUInt8));
  push(prefix + ".scale2", Tensor::fromVector(s2, {(int)s2.size()}, mllm::kFloat32));
}
void pushQDQ(const mllm::ParameterFile::ptr_t& p, const std::string& qdq) {
  auto push = [&](const std::string& k, Tensor t) { p->push(k, t.contiguous().setMemType(mllm::kParamsNormal).setName(k)); };
  push(qdq + ".fake_quant.scale", Tensor::fromVector(std::vector<float>{1.0f / 256.0f}, {1}, mllm::kFloat32));
  push(qdq + ".fake_quant.zero_point", Tensor::fromVector(std::vector<int32_t>{0}, {1}, mllm::kInt32));
}
}  // namespace

namespace mllm::models::qwen3::sha {
// N chained LPBQ MLP blocks + a residual add between them (realistic op mix at the
// block boundary). Each block: x -> Qwen3MLP -> +x. All in one graph.
class StackMLP final : public nn::Module {
  std::vector<Qwen3MLP> mlps_;
  int N_ = 24, hidden_ = 2048;

 public:
  StackMLP() = default;
  StackMLP(const std::string& name, const Qwen3Config& cfg, int N) : nn::Module(name), N_(N), hidden_(cfg.hidden_size) {
    for (int i = 0; i < N; ++i) mlps_.emplace_back(reg<Qwen3MLP>("blk." + std::to_string(i), cfg));
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>& args) override {
    namespace F = nn::functional;
    auto x = in[0];  // [1,hidden]
    for (int i = 0; i < N_; ++i) {
      // Each block matches the validated FullMLPDecode: Qwen3MLP + closing QDQ on the
      // down output (PTQPass needs every boundary quantized; a raw residual Add aborts).
      auto o = mlps_[i].forward({x}, args)[0];  // raw down_proj out [1,S,hidden]
      x = ptq::QDQ(this, o, "blk." + std::to_string(i) + ".down_proj_output_qdq").to(kFloat16).view({1, hidden_}, true);
    }
    return {x};
  }
};
}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& layers_arg = Argparse::add<int>("--layers").help("number of MLP blocks to chain").def(24);
  auto& wm_dir = Argparse::add<std::string>("--wm_dir").help("dir with real layer<i>_mlp-lpbq.mllm bundles (else synthetic, which compresses away)").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int hidden = 2048, inter = 6144, G = 16, N = layers_arg.get();
  mllm::models::qwen3::Qwen3Config cfg;
  cfg.hidden_size = hidden;
  cfg.intermediate_size = inter;

  std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x")};

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  for (int i = 0; i < N; ++i) {
    std::string b = "model.blk." + std::to_string(i) + ".";
    if (!wm_dir.get().empty()) {
      // Load REAL per-layer MLP weights (distinct, incompressible) and remap
      // model.<x> -> model.blk.<i>.<x>. Synthetic weights fold away (4 MB for 4
      // blocks vs 21 MB real each) so they cannot test the monolithic memory.
      std::string bf = wm_dir.get() + "/layer" + std::to_string(i) + "_mlp-lpbq.mllm";
      auto r = mllm::load(bf, mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
      const std::string src = "model.";
      for (auto& [k, t] : r->dict()) {
        if (k.rfind(src, 0) != 0) continue;
        std::string dst = b + k.substr(src.size());
        params->push(dst, t.setName(dst));
      }
      // bundle already includes down_proj_output_qdq (exporter wrote it) — no extra push.
    } else {
      pushLPBQConv(params, b + "gate_proj", hidden, inter, G, 3 * i + 1);
      pushLPBQConv(params, b + "up_proj", hidden, inter, G, 3 * i + 2);
      pushLPBQConv(params, b + "down_proj", inter, hidden, G, 3 * i + 3);
      for (auto q : {"up_proj_input_qdq", "up_proj_output_qdq", "gate_proj_output_qdq", "sigmoid_output_qdq",
                     "act_output_qdq", "down_proj_input_qdq", "down_proj_output_qdq"})
        pushQDQ(params, b + q);
    }
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::StackMLP m("model", cfg, N);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  const std::string bin = "qwen3-stack-" + std::to_string(N) + ".bin";
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print(fmt::format("StackMLP N={} -> {} (graph model.0.s{})  [~{} MB int4 weights]", N, bin, hidden, N * 18));
});
