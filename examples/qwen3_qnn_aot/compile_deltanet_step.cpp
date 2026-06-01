// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// GatedDeltaNet *decode-step core* graph (Sq=1) — the linear-attention math of a
// Qwen3.5 deltanet layer running on the NPU via decomposition (no custom HVX op),
// minus the in/out projection Linears and the causal conv1d.
//
// Takes the post-projection per-head tensors and computes, all on the HTP:
//   gates:   gt = exp(-exp(A_log) * softplus(a + dt_bias));  beta = sigmoid(b)
//   l2norm:  q = (q_raw * rsqrt(sum(q_raw^2)+eps)) * qscale;  k = k_raw * rsqrt(...)
//   recur:   Ss=S*gt; kv=k@Ss; delta=(v-kv)*beta; S_present=Ss + k^T@delta; out=q@S_present
//   gated:   RMSNorm(out)*silu(z)
// Outputs {S_present, gated}. S is carried in/out (the recurrent-state cache).
//
// fp16 throughout (genSimpleQuantizationSpecAttr -> raw fp16, no QDQ).
//
// NOTE: the input projections (nn::Linear, x -> q_raw/k_raw/v/z) are a WIP: fp16
// FullyConnected now lowers + QNN-validates (LLMQuantRecipeLinearPattern FP16
// branch + PTQ raw passthrough), but a reshape of the FC output from [1,H*D] to
// [H,1,D] (heads -> batch) trips QNN's "no valid splitting rule" when consumed by
// an elementwise op. The fix is a 4D [1,H,1,D] layout (batch stays 1). Until then
// this graph takes the per-head tensors directly.
//
//   ./mllm-qwen3-aot-deltanet-step-c -aot_cfg qnn_aot_cfg_deltanet_step.json
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

using mllm::Argparse;
using mllm::Tensor;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}

class DeltaNetStep final : public mllm::nn::Module {
  int H_ = 0, Dk_ = 0, Dv_ = 0, hidden_ = 0, kd_ = 0, vd_ = 0;
  bool proj_ = false;
  mllm::nn::Linear in_q_, in_k_, in_v_, in_z_, out_proj_;

 public:
  DeltaNetStep() = default;
  explicit DeltaNetStep(const std::string& name) : mllm::nn::Module(name) {}
  DeltaNetStep(const std::string& name, int H, int Dk, int Dv, int hidden) : mllm::nn::Module(name) {
    H_ = H; Dk_ = Dk; Dv_ = Dv; hidden_ = hidden; kd_ = H * Dk; vd_ = H * Dv; proj_ = true;
    in_q_ = reg<mllm::nn::Linear>("in_proj_q", hidden_, kd_, false);
    in_k_ = reg<mllm::nn::Linear>("in_proj_k", hidden_, kd_, false);
    in_v_ = reg<mllm::nn::Linear>("in_proj_v", hidden_, vd_, false);
    in_z_ = reg<mllm::nn::Linear>("in_proj_z", hidden_, vd_, false);
    out_proj_ = reg<mllm::nn::Linear>("out_proj", vd_, hidden_, false);
  }

  // Core path (in = per-head tensors). With projections, `in` is
  // [x[1,hidden], S_past, a, b, A_log, dt_bias, z?, eps, qscale, norm_w].
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<mllm::AnyValue>&) override {
    namespace F = mllm::nn::functional;
    Tensor S, q_raw, k_raw, v, a, b, A_log, dt_bias, z, eps, qscale, norm_w;
    if (proj_) {
      auto x = in[0];
      S = in[1]; a = in[2]; b = in[3]; A_log = in[4]; dt_bias = in[5];
      eps = in[6]; qscale = in[7], norm_w = in[8];
      // Project x -> per-head. Reshape splits only the LAST dim [1,H*D]->[1,H,D]
      // (QNN can tile this, like attention's [B,S,H*D]->[B,S,H,D]); heads are then
      // moved to the batch position with a TRANSPOSE (not a flat-split Reshape).
      auto perhead = [&](Tensor t, int D) { return t.view({1, H_, D}).transpose(0, 1); };  // [H,1,D]
      q_raw = perhead(in_q_(x), Dk_);
      k_raw = perhead(in_k_(x), Dk_);
      v = perhead(in_v_(x), Dv_);
      z = perhead(in_z_(x), Dv_);
    } else {
      S = in[0]; q_raw = in[1]; k_raw = in[2]; v = in[3];
      a = in[4]; b = in[5]; A_log = in[6]; dt_bias = in[7];
      z = in[8]; eps = in[9]; qscale = in[10]; norm_w = in[11];
    }

    // Gate math.
    auto gt = F::exp(F::neg(F::exp(A_log) * F::softplus(a + dt_bias)));  // [H,1,1]
    auto beta = F::sigmoid(b);                                          // [H,1,1]

    // Per-head L2-norm of q,k over Dk (eps input), scale q by 1/sqrt(Dk).
    auto q = (q_raw * F::rsqrt(F::sum(q_raw * q_raw, -1, true) + eps)) * qscale;  // [H,1,Dk]
    auto k = k_raw * F::rsqrt(F::sum(k_raw * k_raw, -1, true) + eps);             // [H,1,Dk]

    // Recurrence (state carried as S_past -> S_present).
    auto Ss = S * gt;
    auto kv = F::matmul(k, Ss);
    auto delta = (v - kv) * beta;
    auto outer = F::matmul(k.transpose(1, 2), delta);
    auto Sp = Ss + outer;
    auto out = F::matmul(q, Sp);                            // [H,1,Dv]

    // Gated output: RMSNorm(out) over Dv (eps input) * silu(z).
    auto normed = (out * F::rsqrt(F::mean(out * out, -1, true) + eps)) * norm_w;  // [H,1,Dv]
    auto gated = normed * (z * F::sigmoid(z));                                    // [H,1,Dv]
    if (!proj_) return {Sp, gated};
    // out_proj: [H,1,Dv] -transpose-> [1,H,Dv] -reshape(combine last)-> [1,H*Dv] -> [1,hidden].
    auto y = out_proj_(gated.transpose(0, 1).view({1, vd_}));                     // [1,hidden]
    return {Sp, y};
  }
};

// Deterministic synthetic fp16 weight, shared bit-for-bit by the host reference.
inline float dnW(int o, int i, int seed) {
  uint32_t h = (uint32_t)(o * 4099 + i * 131 + seed * 7919);
  h ^= h >> 13; h *= 2654435761u; h ^= h >> 15;
  return 0.04f * ((float)(h % 257) / 257.0f - 0.5f);  // ~[-0.02, 0.02]
}
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& H_arg = Argparse::add<int>("--heads").def(16);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  auto& proj_arg = Argparse::add<bool>("--proj").help("include in/out projection Linears (x -> y)").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get();
  const int hidden = H * Dv, kd = H * Dk, vd = H * Dv;
  const bool proj = proj_arg.get();

  std::vector<Tensor> ti;
  if (proj) {
    ti = {
        Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"),
        Tensor::zeros({H, Dk, Dv}, mllm::kFloat16).setName("S_past"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("a"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("b"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("A_log"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("dt_bias"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("eps"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("qscale"),
        Tensor::zeros({H, 1, Dv}, mllm::kFloat16).setName("norm_w"),
    };
  } else {
    ti = {
        Tensor::zeros({H, Dk, Dv}, mllm::kFloat16).setName("S_past"),
        Tensor::zeros({H, 1, Dk}, mllm::kFloat16).setName("q"),
        Tensor::zeros({H, 1, Dk}, mllm::kFloat16).setName("k"),
        Tensor::zeros({H, 1, Dv}, mllm::kFloat16).setName("v"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("a"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("b"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("A_log"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("dt_bias"),
        Tensor::zeros({H, 1, Dv}, mllm::kFloat16).setName("z"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("eps"),
        Tensor::zeros({H, 1, 1}, mllm::kFloat16).setName("qscale"),
        Tensor::zeros({H, 1, Dv}, mllm::kFloat16).setName("norm_w"),
    };
  }

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  if (proj) {
    auto pushW = [&](const std::string& name, int out, int in, int seed) {
      auto t = Tensor::empty({out, in}, mllm::kFloat16).alloc();
      auto* p = t.ptr<mllm::mllm_fp16_t>();
      for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i) p[(size_t)o * in + i] = (mllm::mllm_fp16_t)dnW(o, i, seed);
      params->push(name, t.contiguous().setMemType(mllm::kParamsNormal).setName(name));
    };
    pushW("model.in_proj_q.weight", kd, hidden, 1);
    pushW("model.in_proj_k.weight", kd, hidden, 2);
    pushW("model.in_proj_v.weight", vd, hidden, 3);
    pushW("model.in_proj_z.weight", vd, hidden, 4);
    pushW("model.out_proj.weight", hidden, vd, 7);
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));

  DeltaNetStep m = proj ? DeltaNetStep("model", H, Dk, Dv, hidden) : DeltaNetStep("model");
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  qnn_aot_env.saveContext("context.0", "qwen3-deltanet-step.bin");
  mllm::print("DeltaNet decode-step core H={} Dk={} Dv={} -> qwen3-deltanet-step.bin", H, Dk, Dv);
});
