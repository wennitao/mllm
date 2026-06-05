// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED LFM2 short-conv decode-step graph (Sq=1): in_proj/out_proj are LPBQ Conv2D
// (w4a16); the double-gating + depthwise causal conv (kernel = L_cache) runs fp16. cw/cs
// are graph inputs; the updated conv state is an output. Consumes the bundle from
// lfm2_moe/export_short_conv.py (model.{in,out}_proj.* + 4 *_qdq scales), or synthetic.
//
//   ./mllm-lfm2-aot-conv-decode-c -aot_cfg qnn_aot_cfg_mlp_lpbq_microbench.json \
//       --params layer0_conv-lpbq.mllm --out lfm2-conv.bin
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

#include "lfm2_decode_layers.hpp"

using mllm::Argparse;
using mllm::Tensor;
namespace lfm2 = mllm::models::lfm2::sha;

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
  auto push = [&](const std::string& key, Tensor t) { params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key)); };
  push(prefix + ".weight", Tensor::fromVector(w, {1, 1, In, Out}, mllm::kInt8));
  push(prefix + ".scale1", Tensor::fromVector(s1, {(int)s1.size()}, mllm::kUInt8));
  push(prefix + ".scale2", Tensor::fromVector(s2, {(int)s2.size()}, mllm::kFloat32));
}
void pushQDQ(const mllm::ParameterFile::ptr_t& params, const std::string& qdq) {
  auto push = [&](const std::string& key, Tensor t) { params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key)); };
  push(qdq + ".fake_quant.scale", Tensor::fromVector(std::vector<float>{1.0f / 256.0f}, {1}, mllm::kFloat32));
  push(qdq + ".fake_quant.zero_point", Tensor::fromVector(std::vector<int32_t>{0}, {1}, mllm::kInt32));
}
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& params_arg = Argparse::add<std::string>("--params").help("real conv LPBQ .mllm; else synthetic").def("");
  auto& out_arg = Argparse::add<std::string>("--out").def("lfm2-conv-decode.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& k_arg = Argparse::add<int>("--lcache").def(3);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int hidden = hidden_arg.get(), K = k_arg.get(), G = 16;

  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"));
  ti.push_back(Tensor::zeros({1, K, hidden}, mllm::kFloat16).setName("cw"));
  ti.push_back(Tensor::zeros({1, K - 1, hidden}, mllm::kFloat16).setName("cs"));

  mllm::ParameterFile::ptr_t params;
  if (!params_arg.get().empty()) {
    params = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  } else {
    params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
    pushLPBQConv(params, "model.in_proj", hidden, 3 * hidden, G);
    pushLPBQConv(params, "model.out_proj", hidden, hidden, G);
    for (auto q : {"in_proj_input_qdq", "in_proj_output_qdq", "out_proj_input_qdq", "out_proj_output_qdq"})
      pushQDQ(params, std::string("model.") + q);
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  lfm2::Lfm2ShortConvDecodeLPBQ m("model", hidden, K);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  qnn_aot_env.saveContext("context.0", out_arg.get());
  mllm::print(fmt::format("LFM2 short-conv decode (LPBQ) hidden={} K={} -> {} (graph model.0.s{})", hidden, K,
                          out_arg.get(), hidden));
});
