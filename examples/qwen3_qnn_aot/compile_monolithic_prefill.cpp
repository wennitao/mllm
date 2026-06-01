// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// MONOLITHIC Qwen3.5-2B BLOCK-PREFILL graph (Sq=B): all `--layers N` decoder layers
// (deltanet-chunk or attn-causal by interval-4) + per-layer RMSNorm(1+w) + 2 residuals
// + MLP, chained into ONE QNN graph. The prefill counterpart of compile_monolithic.cpp
// (the Sq=1 decode stack). Each per-layer forward is the validated prefill mixer from
// qwen3_5_prefill_layers.hpp. Prefill from ZERO state (S0=0, conv=0, no past KV).
//
// Weights (LPBQ convs) load from the wm/ bundles (export_whole_model.py), remapped
// model.X -> model.layers.<i>.{mixer,mlp}.X — SAME bundles as the decode monolithic.
// Norm weights / gates (Wa/Wb/A_log/dt_bias/cw) / qscale are graph INPUTS (from consts).
//
//   ./mllm-qwen3-aot-monolithic-prefill-c -aot_cfg qnn_aot_cfg_lmhead.json --layers 4 \
//        --wm_dir wm --seq 128 --chunk 32
//
#include <cmath>
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

#include "qwen3_5_decode_layers.hpp"   // FullMLPDecode's Qwen3MLP/Qwen3Config (via modeling header)
#include "qwen3_5_prefill_layers.hpp"

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
bool isDeltanet(int i) { return ((i + 1) % 4) != 0; }

constexpr int kHidden = 2048, kLH = 16, kDk = 128, kDv = 128;     // deltanet
constexpr int kH = 8, kKV = 2, kAttD = 256, kRot = 64;           // attn
constexpr int kInter = 6144, kVocab = 248320;
}  // namespace

namespace mllm::models::qwen3::sha {

class Qwen3_5PrefillStack final : public nn::Module {
  int N_ = 4, B_ = 128, C_ = 32;
  bool head_ = false;
  std::vector<DeltaNetPrefillLPBQ> dn_;
  std::vector<AttnPrefillLPBQ> at_;
  std::vector<FullMLPPrefill> mlp_;
  nn::Conv2D head_proj_;
  std::vector<int> dn_idx_, at_idx_;

 public:
  Qwen3_5PrefillStack() = default;
  Qwen3_5PrefillStack(const std::string& name, int N, int B, int C, bool head, const Qwen3Config& mlpcfg)
      : nn::Module(name), N_(N), B_(B), C_(C), head_(head) {
    dn_idx_.assign(N, -1);
    at_idx_.assign(N, -1);
    for (int i = 0; i < N; ++i) {
      std::string p = "layers." + std::to_string(i) + ".";
      if (isDeltanet(i)) {
        dn_idx_[i] = (int)dn_.size();
        dn_.emplace_back(reg<DeltaNetPrefillLPBQ>(p + "mixer", kLH, kDk, kDv, kHidden, B, C));
      } else {
        at_idx_[i] = (int)at_.size();
        at_.emplace_back(reg<AttnPrefillLPBQ>(p + "mixer", kH, kKV, kAttD, kHidden, kRot, B));
      }
      mlp_.emplace_back(reg<FullMLPPrefill>(p + "mlp", mlpcfg, B));
    }
    if (head) head_proj_ = reg<nn::Conv2D>("lm_head", kHidden, kVocab, CONV2D_PROPERTY);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    int p = 0;
    auto x = in[p++];          // [B,hidden]
    auto sin = in[p++];        // [1,B,rot]
    auto cos = in[p++];        // [1,B,rot]
    auto cmask = in[p++];      // [1,1,B,B]
    auto eps = in[p++];        // [1,1,1]
    auto Ltri = in[p++];       // [1,1,C,C]
    auto strict = in[p++];
    auto eye = in[p++];
    auto S0z = in[p++];        // [H,Dk,Dv] zeros (shared)
    auto csqz = in[p++], cskz = in[p++], csvz = in[p++];  // [1,3,*] zeros (shared)
    auto eps_f = eps.to(kFloat32);

    // hand-rolled RMSNorm over hidden, per token, fp32, weight 1+w -> fp16.
    auto rmsnorm = [&](Tensor t, Tensor w) {
      auto tf = t.view({1, 1, B_, kHidden}, true).to(kFloat32);
      auto inv = F::rsqrt(F::mean(tf * tf, -1, true) + eps_f.view({1, 1, 1, 1}, true));
      return ((tf * inv) * w.to(kFloat32).view({1, 1, 1, kHidden}, true)).to(kFloat16).view({B_, kHidden}, true);
    };

    std::vector<Tensor> states;
    for (int i = 0; i < N_; ++i) {
      auto in_w = in[p++], post_w = in[p++];
      auto xn = rmsnorm(x, in_w);
      Tensor out;
      if (isDeltanet(i)) {
        auto gated_w = in[p++], qscale = in[p++], Wa = in[p++], Wb = in[p++];
        auto A_log = in[p++], dt_bias = in[p++], cwq = in[p++], cwk = in[p++], cwv = in[p++];
        std::vector<Tensor> di{xn, eps, qscale, gated_w, A_log, dt_bias, Wa, Wb,
                               cwq, cwk, cwv, csqz, cskz, csvz, S0z, Ltri, strict, eye};
        auto o = dn_[dn_idx_[i]].forward(di, {});   // [y, Sp]
        out = o[0].view({B_, kHidden}, true);
        states.push_back(o[1]);                      // Sp
      } else {
        auto qn = in[p++], kn = in[p++];
        std::vector<Tensor> ai{xn, sin, cos, qn, kn, eps, cmask};
        auto o = at_[at_idx_[i]].forward(ai, {});    // [y, k_all, v_all]
        out = o[0].view({B_, kHidden}, true);
        states.push_back(o[1]); states.push_back(o[2]);
      }
      x = x + out;                                   // residual 1
      auto xn2 = rmsnorm(x, post_w);
      auto m = mlp_[i].forward({xn2}, {})[0].view({B_, kHidden}, true);
      x = x + m;                                     // residual 2
    }

    std::vector<Tensor> outs;
    if (head_) {
      auto fnorm_w = in[p++];
      auto xn = rmsnorm(x, fnorm_w);
      auto xq = ptq::QDQ(this, xn, "lmhead_input_qdq").view({1, 1, -1, kHidden}, true);
      auto logits = ptq::QDQ(this, head_proj_(xq), "lmhead_output_qdq").to(kFloat16).view({B_, kVocab}, true);
      outs.push_back(logits);
    } else {
      outs.push_back(x);   // final hidden [B,hidden] (pre-final-norm) for validation
    }
    for (auto& s : states) outs.push_back(s);
    return outs;
  }
};

}  // namespace mllm::models::qwen3::sha

static std::vector<Tensor> buildInputs(int N, int B, int C, bool head) {
  const int kd = kLH * kDk, vd = kLH * kDv;
  std::vector<Tensor> ti;
  auto f16 = [&](std::vector<int> s, const std::string& n) { ti.push_back(Tensor::zeros(s, mllm::kFloat16).setName(n)); };
  auto f32 = [&](std::vector<int> s, const std::string& n) { ti.push_back(Tensor::zeros(s, mllm::kFloat32).setName(n)); };
  f16({B, kHidden}, "x");
  f16({1, B, kRot}, "sin");
  f16({1, B, kRot}, "cos");
  f16({1, 1, B, B}, "cmask");
  f32({1, 1, 1}, "eps");
  f32({1, 1, C, C}, "Ltri");
  f32({1, 1, C, C}, "strict");
  f32({1, 1, C, C}, "eye");
  f32({kLH, kDk, kDv}, "S0z");
  f32({1, 3, kd}, "csqz"); f32({1, 3, kd}, "cskz"); f32({1, 3, vd}, "csvz");
  for (int i = 0; i < N; ++i) {
    std::string s = "l" + std::to_string(i) + "_";
    f32({1, 1, kHidden}, s + "in_norm");
    f32({1, 1, kHidden}, s + "post_norm");
    if (isDeltanet(i)) {
      f32({1, 1, kDv}, s + "gated_norm");
      f32({1, 1, 1}, s + "qscale");
      f32({1, kHidden, kLH}, s + "Wa"); f32({1, kHidden, kLH}, s + "Wb");
      f32({1, 1, kLH}, s + "A_log"); f32({1, 1, kLH}, s + "dt_bias");
      f32({1, 4, kd}, s + "cw_q"); f32({1, 4, kd}, s + "cw_k"); f32({1, 4, vd}, s + "cw_v");
    } else {
      f32({1, 1, kAttD}, s + "q_norm"); f32({1, 1, kAttD}, s + "k_norm");
    }
  }
  if (head) f32({1, 1, kHidden}, "final_norm");
  return ti;
}

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& layers_arg = Argparse::add<int>("--layers").def(4);
  auto& wm_dir = Argparse::add<std::string>("--wm_dir").def("wm");
  auto& seq_arg = Argparse::add<int>("--seq").def(128);
  auto& chunk_arg = Argparse::add<int>("--chunk").def(32);
  auto& head_arg = Argparse::add<bool>("--head").def(false);
  auto& out_arg = Argparse::add<std::string>("--out").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config"); return -1; }

  const int N = layers_arg.get(), B = seq_arg.get(), C = chunk_arg.get();
  const bool head = head_arg.get();
  if (B % C != 0) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--seq must be a multiple of --chunk"); return -1; }
  mllm::models::qwen3::Qwen3Config mlpcfg;
  mlpcfg.hidden_size = kHidden;
  mlpcfg.intermediate_size = kInter;

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  auto merge = [&](const std::string& file, const std::string& dstpfx) {
    auto r = mllm::load(file, mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    const std::string src = "model.";
    for (auto& [k, t] : r->dict()) {
      if (k.rfind(src, 0) != 0) continue;
      std::string dst = dstpfx + k.substr(src.size());
      params->push(dst, t.setName(dst));
    }
  };
  for (int i = 0; i < N; ++i) {
    std::string d = wm_dir.get();
    merge(d + "/layer" + std::to_string(i) + "_mixer-lpbq.mllm", "model.layers." + std::to_string(i) + ".mixer.");
    merge(d + "/layer" + std::to_string(i) + "_mlp-lpbq.mllm", "model.layers." + std::to_string(i) + ".mlp.");
  }
  if (head) merge(wm_dir.get() + "/lmhead-lpbq.mllm", "model.");

  auto ti = buildInputs(N, B, C, head);
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::Qwen3_5PrefillStack m("model", N, B, C, head, mlpcfg);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  const std::string bin = out_arg.get().empty() ? ("qwen3-mono-prefill-" + std::to_string(N) + ".bin") : out_arg.get();
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print(fmt::format("Monolithic PREFILL N={} B={} C={} head={} -> {} (graph model.0.s{})", N, B, C, head, bin, kHidden));
});
