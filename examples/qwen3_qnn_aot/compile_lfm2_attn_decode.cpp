// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED LFM2 GQA full-attention decode-step graph (Sq=1): per-head LPBQ q/k/v/o
// (w4a16), plain RMSNorm q/k-norm + FULL RoPE + KV-cache concat + fp32 softmax. NO output
// gate. Consumes the bundle from lfm2_moe/export_attn_decode.py, or synthetic weights.
//
//   ./mllm-lfm2-aot-attn-decode-c -aot_cfg qnn_aot_cfg_attn_decode.json \
//       --params attn-l2-lpbq.mllm --ctx 256 --out lfm2-attn.bin
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
  auto& params_arg = Argparse::add<std::string>("--params").help("real attn LPBQ .mllm; else synthetic").def("");
  auto& out_arg = Argparse::add<std::string>("--out").def("lfm2-attn-decode.bin");
  auto& H_arg = Argparse::add<int>("--heads").def(32);
  auto& kv_arg = Argparse::add<int>("--kv").def(8);
  auto& d_arg = Argparse::add<int>("--dim").def(64);
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& ctx_arg = Argparse::add<int>("--ctx").def(256);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int H = H_arg.get(), KV = kv_arg.get(), D = d_arg.get(), hidden = hidden_arg.get(), ctx = ctx_arg.get();
  const int rot = D, G = 16, P = ctx - 1;

  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"));
  ti.push_back(Tensor::zeros({1, 1, rot}, mllm::kFloat16).setName("sin"));
  ti.push_back(Tensor::zeros({1, 1, rot}, mllm::kFloat16).setName("cos"));
  ti.push_back(Tensor::zeros({1, 1, D}, mllm::kFloat16).setName("q_norm_w"));
  ti.push_back(Tensor::zeros({1, 1, D}, mllm::kFloat16).setName("k_norm_w"));
  ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("eps"));
  ti.push_back(Tensor::zeros({1, KV, D, P}, mllm::kFloat16).setName("past_k"));
  ti.push_back(Tensor::zeros({1, KV, P, D}, mllm::kFloat16).setName("past_v"));
  ti.push_back(Tensor::zeros({1, 1, 1, ctx}, mllm::kFloat16).setName("mask"));

  mllm::ParameterFile::ptr_t params;
  if (!params_arg.get().empty()) {
    params = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  } else {
    params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
    pushQDQ(params, "model.qkv_input_qdq");
    for (int h = 0; h < H; ++h) {
      pushLPBQConv(params, "model.q_proj." + std::to_string(h), hidden, D, G);
      pushQDQ(params, "model.q_out_qdq_h" + std::to_string(h));
    }
    for (int kv = 0; kv < KV; ++kv) {
      pushLPBQConv(params, "model.k_proj." + std::to_string(kv), hidden, D, G);
      pushLPBQConv(params, "model.v_proj." + std::to_string(kv), hidden, D, G);
      pushQDQ(params, "model.k_out_qdq_h" + std::to_string(kv));
      pushQDQ(params, "model.v_out_qdq_h" + std::to_string(kv));
    }
    pushLPBQConv(params, "model.o_proj", H * D, hidden, G);
    pushQDQ(params, "model.o_proj_input_qdq");
    pushQDQ(params, "model.o_proj_output_qdq");
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  lfm2::Lfm2AttnDecodeLPBQ m("model", H, KV, D, hidden, rot, ctx);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  qnn_aot_env.saveContext("context.0", out_arg.get());
  mllm::print(fmt::format("LFM2 attn decode (LPBQ) H={} KV={} D={} ctx={} -> {} (graph model.0.s{})", H, KV, D, ctx,
                          out_arg.get(), hidden));
});
