// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// MONOLITHIC Qwen3.5-2B decode graph (Sq=1): all `--layers N` decoder layers (deltanet
// or attn by interval-4) + per-layer RMSNorm(1+w) + residuals + MLP, chained into ONE
// QNN graph / ONE context. Proven feasible by compile_stack_test (24 real MLP blocks =
// 493 MB single graph loads on V79). Reuses the validated layer forwards from
// qwen3_5_decode_layers.hpp.
//
// v1 (structure): gt/beta (deltanet gates) and sin/cos (RoPE) are fed as INPUTS — the
// host can supply them. This validates the full chained structure compiles + loads (the
// PTQPass-residual / IO-count / ~1 GB finalize risks). v2 adds the in-graph gate math
// (gt/beta from in_proj_a/b + A_log/dt_bias, which depend on intermediate xn).
//
// Layer weights (LPBQ convs) load from the wm/ bundles (export_whole_model.py), remapped
// model.X -> model.layers.<i>.{mixer,mlp}.X. Norm weights / gates / states are inputs.
//
//   ./mllm-qwen3-aot-monolithic-c -aot_cfg qnn_aot_cfg_lmhead.json --layers 4 --wm_dir wm [--head]
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

#include "qwen3_5_decode_layers.hpp"

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
bool isDeltanet(int i) { return ((i + 1) % 4) != 0; }  // interval 4: 3,7,.. are attn

// Qwen3.5-2B dims.
constexpr int kHidden = 2048, kLH = 16, kDk = 128, kDv = 128;     // deltanet
constexpr int kH = 8, kKV = 2, kAttD = 256, kRot = 64;            // attn
constexpr int kInter = 6144, kVocab = 248320;
}  // namespace

namespace mllm::models::qwen3::sha {

// Monolithic decode stack. Holds N mixer sub-modules + N MLP sub-modules (+ optional
// head). forward() consumes a flat input vector in the SAME per-layer order the driver
// builds it (see buildInputs()): for each layer — input_norm_w, post_norm_w, then the
// mixer's dynamic inputs, then nothing extra for MLP; shared sin/cos/mask at the front.
class Qwen3_5DecodeStack final : public nn::Module {
  int N_ = 4;
  bool head_ = false;
  std::vector<DeltaNetDecodeLPBQ> dn_;     // indexed by layer (only deltanet layers used)
  std::vector<AttnDecodeLPBQ> at_;
  std::vector<FullMLPDecode> mlp_;
  nn::Conv2D head_proj_;
  std::vector<int> dn_idx_, at_idx_;       // map layer -> slot in dn_/at_

 public:
  Qwen3_5DecodeStack() = default;
  Qwen3_5DecodeStack(const std::string& name, int N, bool head, const Qwen3Config& mlpcfg)
      : nn::Module(name), N_(N), head_(head) {
    dn_idx_.assign(N, -1);
    at_idx_.assign(N, -1);
    for (int i = 0; i < N; ++i) {
      std::string p = "layers." + std::to_string(i) + ".";
      if (isDeltanet(i)) {
        dn_idx_[i] = (int)dn_.size();
        dn_.emplace_back(reg<DeltaNetDecodeLPBQ>(p + "mixer", kLH, kDk, kDv, kHidden, /*full=*/true, /*conv1d=*/true));
      } else {
        at_idx_[i] = (int)at_.size();
        at_.emplace_back(reg<AttnDecodeLPBQ>(p + "mixer", kH, kKV, kAttD, kHidden, kRot, /*ctx=*/256, 1e-6f));
      }
      mlp_.emplace_back(reg<FullMLPDecode>(p + "mlp", mlpcfg));
    }
    if (head) head_proj_ = reg<nn::Conv2D>("lm_head", kHidden, kVocab, CONV2D_PROPERTY);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    int p = 0;  // input cursor — MUST match the driver's buildInputs() order
    auto x = in[p++];                 // [1,hidden]
    auto sin = in[p++];               // [1,1,rot] (shared across attn layers)
    auto cos = in[p++];               // [1,1,rot]
    auto mask = in[p++];              // [1,1,1,ctx]
    auto eps = in[p++];               // [1,1,1] shared (input, NOT a constant — a bare
                                      // constant operand of an elementwise add gets no
                                      // quant_recipe and the recipe pass aborts).
    auto eps_f = eps.to(kFloat32);

    auto rmsnorm = [&](Tensor t, Tensor w) {
      auto tf = t.to(kFloat32);
      auto inv = F::rsqrt(F::mean(tf * tf, -1, true) + eps_f);
      return ((tf * inv) * w.to(kFloat32)).to(kFloat16);
    };

    std::vector<Tensor> states;  // updated states, appended in layer order
    for (int i = 0; i < N_; ++i) {
      auto in_w = in[p++];            // input_norm_w [1,1,hidden]
      auto post_w = in[p++];          // post_norm_w  [1,1,hidden]
      auto xn = rmsnorm(x, in_w);
      Tensor out;
      if (isDeltanet(i)) {
        // deltanet inputs: gated_norm_w, qscale, cw_q,cw_k,cw_v, then per-head S(16),
        // gt(16), beta(16), cs_q,cs_k,cs_v. Assemble the layer's input vector in the
        // order DeltaNetDecodeLPBQ::forward expects:
        //   [x, S_h0..15, gt_h0..15, beta_h0..15, eps, qscale, norm_w, cw_q,cw_k,cw_v, cs_q,cs_k,cs_v]
        auto gated_w = in[p++];       // [1,1,Dv]
        auto qscale = in[p++];        // [1,1,1]
        auto Wa = in[p++], Wb = in[p++];   // [1,LH,hidden]  in_proj_a/b
        auto A_log = in[p++], dt_bias = in[p++];  // [1,1,LH]
        auto cwq = in[p++], cwk = in[p++], cwv = in[p++];
        std::vector<Tensor> Sh(kLH);
        for (int h = 0; h < kLH; ++h) Sh[h] = in[p++];
        auto csq = in[p++], csk = in[p++], csv = in[p++];
        // in-graph gates (fp32, like HF): gt = exp(-exp(A_log)*softplus(a@xn+dt_bias)); beta = sigmoid(b@xn).
        // 4D matmul (HTP MatMul rejects 3D — error 3110): [1,1,1,hidden] x [1,1,LH,hidden] transB -> [1,1,1,LH].
        // HTP MatMul rejects the transB flag (err 3110); feed Wa/Wb PRE-TRANSPOSED as
        // [hidden,LH] (host transposes in_proj_a/b) and matmul without transpose.
        auto xn4 = xn.view({1, 1, 1, kHidden}, true).to(kFloat32);
        auto Wa4 = Wa.view({1, 1, kHidden, kLH}, true).to(kFloat32);
        auto Wb4 = Wb.view({1, 1, kHidden, kLH}, true).to(kFloat32);
        auto a = F::matmul(xn4, Wa4);                                   // [1,1,1,LH]
        auto b = F::matmul(xn4, Wb4);                                   // [1,1,1,LH]
        auto Al = A_log.view({1, 1, 1, kLH}, true).to(kFloat32);
        auto db = dt_bias.view({1, 1, 1, kLH}, true).to(kFloat32);
        auto gt = F::exp(-(F::exp(Al) * F::softplus(a + db)));          // [1,1,1,LH]
        auto beta = F::sigmoid(b);                                      // [1,1,1,LH] fp32
        std::vector<Tensor> gth(kLH), beh(kLH);
        for (int h = 0; h < kLH; ++h) {
          gth[h] = gt.slice({kAll, kAll, kAll, {h, h + 1}}, true).view({1, 1, 1}, true).to(kFloat16);
          beh[h] = beta.slice({kAll, kAll, kAll, {h, h + 1}}, true).view({1, 1, 1}, true).to(kFloat16);
        }
        std::vector<Tensor> di{xn};
        for (auto& t : Sh) di.push_back(t);
        for (auto& t : gth) di.push_back(t);
        for (auto& t : beh) di.push_back(t);
        di.push_back(eps); di.push_back(qscale); di.push_back(gated_w);
        di.push_back(cwq); di.push_back(cwk); di.push_back(cwv);
        di.push_back(csq); di.push_back(csk); di.push_back(csv);
        auto o = dn_[dn_idx_[i]].forward(di, {});
        // outs: Sp_h0..15, y(idx LH), new_cs_q,new_cs_k,new_cs_v
        out = o[kLH];
        for (int h = 0; h < kLH; ++h) states.push_back(o[h]);     // Sp
        states.push_back(o[kLH + 1]); states.push_back(o[kLH + 2]); states.push_back(o[kLH + 3]);  // new_cs
      } else {
        auto qn = in[p++], kn = in[p++];   // q_norm_w, k_norm_w [1,1,256]
        auto pk = in[p++], pv = in[p++];   // past_k [1,KV,D,P], past_v [1,KV,P,D]
        std::vector<Tensor> ai{xn, sin, cos, qn, kn, eps, pk, pv, mask};
        auto o = at_[at_idx_[i]].forward(ai, {});  // outs: y, k_new, v_new
        out = o[0];
        states.push_back(o[1]); states.push_back(o[2]);
      }
      x = x + out;                                   // residual 1
      auto xn2 = rmsnorm(x, post_w);
      auto m = mlp_[i].forward({xn2}, {})[0].view({1, kHidden}, true);
      x = x + m;                                     // residual 2
    }

    std::vector<Tensor> outs;
    if (head_) {
      auto fnorm_w = in[p++];
      auto xn = rmsnorm(x, fnorm_w);
      auto xq = ptq::QDQ(this, xn, "lmhead_input_qdq").view({1, 1, -1, kHidden}, true);
      auto logits = ptq::QDQ(this, head_proj_(xq), "lmhead_output_qdq").to(kFloat16).view({1, kVocab}, true);
      outs.push_back(logits);
    } else {
      outs.push_back(x);  // final hidden (for partial-stack validation)
    }
    for (auto& s : states) outs.push_back(s);
    return outs;
  }
};

}  // namespace mllm::models::qwen3::sha

// Build the flat input tensor list in the EXACT order the forward consumes it.
static std::vector<Tensor> buildInputs(int N, bool head) {
  std::vector<Tensor> ti;
  auto add = [&](std::vector<int> shp, const std::string& nm) {
    ti.push_back(Tensor::zeros(shp, mllm::kFloat16).setName(nm));
  };
  add({1, kHidden}, "x");
  add({1, 1, kRot}, "sin");
  add({1, 1, kRot}, "cos");
  add({1, 1, 1, 256}, "mask");
  add({1, 1, 1}, "eps");
  for (int i = 0; i < N; ++i) {
    std::string s = "l" + std::to_string(i) + "_";
    add({1, 1, kHidden}, s + "in_norm");
    add({1, 1, kHidden}, s + "post_norm");
    if (isDeltanet(i)) {
      add({1, 1, kDv}, s + "gated_norm");
      add({1, 1, 1}, s + "qscale");
      add({1, kHidden, kLH}, s + "Wa"); add({1, kHidden, kLH}, s + "Wb");
      add({1, 1, kLH}, s + "A_log"); add({1, 1, kLH}, s + "dt_bias");
      add({1, 4, kLH * kDk}, s + "cw_q"); add({1, 4, kLH * kDk}, s + "cw_k"); add({1, 4, kLH * kDv}, s + "cw_v");
      for (int h = 0; h < kLH; ++h) add({1, kDk, kDv}, s + "S" + std::to_string(h));
      add({1, 3, kLH * kDk}, s + "cs_q"); add({1, 3, kLH * kDk}, s + "cs_k"); add({1, 3, kLH * kDv}, s + "cs_v");
    } else {
      add({1, 1, kAttD}, s + "q_norm"); add({1, 1, kAttD}, s + "k_norm");
      add({1, kKV, kAttD, 255}, s + "past_k"); add({1, kKV, 255, kAttD}, s + "past_v");
    }
  }
  if (head) add({1, 1, kHidden}, "final_norm");
  return ti;
}

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& layers_arg = Argparse::add<int>("--layers").def(4);
  auto& wm_dir = Argparse::add<std::string>("--wm_dir").def("wm");
  auto& head_arg = Argparse::add<bool>("--head").def(false);
  auto& out_arg = Argparse::add<std::string>("--out").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config"); return -1; }

  const int N = layers_arg.get();
  const bool head = head_arg.get();
  mllm::models::qwen3::Qwen3Config mlpcfg;
  mlpcfg.hidden_size = kHidden;
  mlpcfg.intermediate_size = kInter;

  // ---- assemble params: remap each layer's wm bundle into layers.<i>.{mixer,mlp}.* ----
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
  if (head) merge(wm_dir.get() + "/lmhead-lpbq.mllm", "model.");  // keys: model.lm_head.{weight,scale1,scale2} + model.lmhead_{in,out}put_qdq.*

  auto ti = buildInputs(N, head);
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::Qwen3_5DecodeStack m("model", N, head, mlpcfg);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  const std::string bin = out_arg.get().empty() ? ("qwen3-mono-" + std::to_string(N) + ".bin") : out_arg.get();
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print(fmt::format("Monolithic decode N={} head={} -> {} (graph model.0.s{})", N, head, bin, kHidden));
});
