// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile the per-layer decode graphs for an N-layer LFM2 stack into ONE context, each as a
// DISTINCTLY-NAMED graph so the host orchestrator (run_lfm2_decode) can dispatch them by name:
//   conv_l{i} | attn_l{i}  (mixer)  ;  ffn_l{i} (dense)  |  moe_l{i}_e{e} x32 (MoE experts)
// Reads the bundles from lfm2_moe/export_decode_stack.py (<dir>/l{i}_mixer-lpbq.mllm,
// l{i}_ffn-lpbq.mllm, l{i}_moe/expert{e}-lpbq.mllm), re-keys "model." -> "<graph>." and routes
// each through the SplitLLMGraph chunk path (chunk_graph_name) to keep the distinct name.
//
//   ./mllm-lfm2-aot-decode-stack-c -aot_cfg qnn_aot_cfg_mlp_lpbq_microbench.json \
//       --dir wm_dec --layers 2 --out stack.bin
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

#include "lfm2_decode_layers.hpp"     // Lfm2ShortConvDecodeLPBQ, Lfm2AttnDecodeLPBQ
#include "qwen3_5_decode_layers.hpp"  // sha::FullMLPDecode (Qwen3MLP + closing QDQ)

using mllm::Argparse;
using mllm::Tensor;
namespace lfm2 = mllm::models::lfm2::sha;
namespace sha = mllm::models::qwen3::sha;

// LFM2.5-8B-A1B layer disposition.
static bool isAttn(int i) { return i == 2 || i == 6 || i == 10 || i == 14 || i == 18 || i == 21; }
static bool isDense(int i) { return i < 2; }

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
// Load a "model."-prefixed bundle and re-key to "<name>." (also set tensor names — the symbol
// table keys weights by tensor.name(), see compile_lfm2_expert).
mllm::ParameterFile::ptr_t rekey(const std::string& bundle, const std::string& name) {
  auto src = mllm::load(bundle, mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  auto p = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  for (const auto& kv : src->dict()) {
    const std::string& key = kv.first;
    std::string nk = (key.rfind("model.", 0) == 0) ? (name + "." + key.substr(6)) : key;
    auto t = kv.second;
    p->push(nk, t.setMemType(mllm::kParamsNormal).setName(nk));
  }
  return p;
}
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& dir_arg = Argparse::add<std::string>("--dir").help("export_decode_stack out dir").required(true);
  auto& nl_arg = Argparse::add<int>("--layers").def(2);
  auto& out_arg = Argparse::add<std::string>("--out").def("lfm2-decode-stack.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& ctx_arg = Argparse::add<int>("--ctx").def(256);
  auto& nc_arg = Argparse::add<int>("--num_contexts").def(1);
  auto& vocab_arg = Argparse::add<int>("--vocab").def(128000);
  auto& head_arg = Argparse::add<bool>("--lm_head").help("also compile the lm_head graph").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int NL = nl_arg.get(), hidden = hidden_arg.get(), ctx = ctx_arg.get();
  const int NC = std::max(1, nc_arg.get()), vocab = vocab_arg.get();
  const int H = 32, KV = 8, D = 64, rot = 64, K = 3, E = 32;
  const int inter_dense = 7168, inter_moe = 1792, P = ctx - 1;
  const std::string dir = dir_arg.get();
  // assign layer i's graphs to context floor(i*NC/NL); lm_head -> last context.
  auto ctx_of = [&](int i) { return NC <= 1 ? 0 : std::min(NC - 1, i * NC / NL); };

  auto env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));

  // Compile one already-built module (params already re-keyed to `name`) as graph `name` into context k.
  auto lower = [&](mllm::nn::Module& m, const std::vector<Tensor>& ti, const mllm::ParameterFile::ptr_t& params,
                   const std::string& name, int k) {
    auto ir = mllm::ir::trace_(m, ti);
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&env, aot_cfg.get(), params));
    auto& c = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
    c["graph_on_qnn"] = nlohmann::json::array({name});
    c["chunk_graph_name"] = name;
    c["chunk_context_name"] = "context." + std::to_string(k);
    pm.run();
  };

  mllm::models::qwen3::Qwen3Config mlpcfg;  // for Qwen3MLP dims
  mlpcfg.hidden_size = hidden;

  for (int i = 0; i < NL; ++i) {
    // ---- mixer ----
    if (isAttn(i)) {
      const std::string name = "attn_l" + std::to_string(i);
      auto params = rekey(dir + "/l" + std::to_string(i) + "_mixer-lpbq.mllm", name);
      std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"),
                             Tensor::zeros({1, 1, rot}, mllm::kFloat16).setName("sin"),
                             Tensor::zeros({1, 1, rot}, mllm::kFloat16).setName("cos"),
                             Tensor::zeros({1, 1, D}, mllm::kFloat16).setName("q_norm_w"),
                             Tensor::zeros({1, 1, D}, mllm::kFloat16).setName("k_norm_w"),
                             Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("eps"),
                             Tensor::zeros({1, KV, D, P}, mllm::kFloat16).setName("past_k"),
                             Tensor::zeros({1, KV, P, D}, mllm::kFloat16).setName("past_v"),
                             Tensor::zeros({1, 1, 1, ctx}, mllm::kFloat16).setName("mask")};
      lfm2::Lfm2AttnDecodeLPBQ m(name, H, KV, D, hidden, rot, ctx);
      m.load(params);
      lower(m, ti, params, name, ctx_of(i));
    } else {
      const std::string name = "conv_l" + std::to_string(i);
      auto params = rekey(dir + "/l" + std::to_string(i) + "_mixer-lpbq.mllm", name);
      std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"),
                             Tensor::zeros({1, K, hidden}, mllm::kFloat16).setName("cw"),
                             Tensor::zeros({1, K - 1, hidden}, mllm::kFloat16).setName("cs")};
      lfm2::Lfm2ShortConvDecodeLPBQ m(name, hidden, K);
      m.load(params);
      lower(m, ti, params, name, ctx_of(i));
    }

    // ---- ffn ----
    if (isDense(i)) {
      const std::string name = "ffn_l" + std::to_string(i);
      auto params = rekey(dir + "/l" + std::to_string(i) + "_ffn-lpbq.mllm", name);
      mlpcfg.intermediate_size = inter_dense;
      sha::FullMLPDecode m(name, mlpcfg);
      m.load(params);
      std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x")};
      lower(m, ti, params, name, ctx_of(i));
    } else {
      mlpcfg.intermediate_size = inter_moe;
      const int GS = 8, NG = E / GS;  // 8 experts/group -> 4 grouped graphs/MoE-layer
      for (int g = 0; g < NG; ++g) {
        const std::string name = "moe_l" + std::to_string(i) + "_g" + std::to_string(g);
        // merge GS expert bundles, re-keying expert (g*GS+j)'s "model.X" -> "<name>.e{j}.X"
        auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
        for (int j = 0; j < GS; ++j) {
          int e = g * GS + j;
          auto src = mllm::load(dir + "/l" + std::to_string(i) + "_moe/expert" + std::to_string(e) + "-lpbq.mllm",
                                mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
          std::string pre = name + ".e" + std::to_string(j) + ".";
          for (const auto& kv : src->dict()) {
            const std::string& key = kv.first;
            std::string nk = (key.rfind("model.", 0) == 0) ? (pre + key.substr(6)) : key;
            auto t = kv.second;
            params->push(nk, t.setMemType(mllm::kParamsNormal).setName(nk));
          }
        }
        lfm2::Lfm2MoeGroupLPBQ m(name, mlpcfg, GS);
        m.load(params);
        std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x")};
        lower(m, ti, params, name, ctx_of(i));
      }
    }
    mllm::print(fmt::format("[{}/{}] layer {} graphs lowered (ctx {})", i + 1, NL, i, ctx_of(i)));
  }

  // ---- lm_head (final RMSNorm + LPBQ proj) into the last context ----
  if (head_arg.isSet()) {
    const std::string name = "lm_head";
    auto params = rekey(dir + "/lmhead-lpbq.mllm", name);
    lfm2::Lfm2LmHeadLPBQ m(name, hidden, vocab);
    m.load(params);
    std::vector<Tensor> ti{Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"),
                           Tensor::zeros({1, 1, hidden}, mllm::kFloat16).setName("norm_w"),
                           Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("eps")};
    lower(m, ti, params, name, NC - 1);
    mllm::print("lm_head graph lowered (ctx " + std::to_string(NC - 1) + ")");
  }

  // ---- save one .bin per context (NC=1 -> --out; else --out base + -c{k}of{NC}.bin) ----
  for (int k = 0; k < NC; ++k) {
    std::string bin = out_arg.get();
    if (NC > 1) {
      auto dot = bin.rfind(".bin");
      std::string base = (dot == std::string::npos) ? bin : bin.substr(0, dot);
      bin = base + "-c" + std::to_string(k) + "of" + std::to_string(NC) + ".bin";
    }
    env.saveContext("context." + std::to_string(k), bin);
    mllm::print(fmt::format("  saved context.{} -> {}", k, bin));
  }
  mllm::print(fmt::format("LFM2 decode stack: {} layers ({} contexts){} -> {}", NL, NC,
                          head_arg.isSet() ? " + lm_head" : "", out_arg.get()));
});
