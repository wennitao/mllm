// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Minimal test of the per-head-projection layout for the DeltaNet decode step.
// Following the proven SHA attention pattern (modeling_qwen_qnn_aot_sha.hpp): use
// H separate per-head projections Linear(hidden->head_dim) so each head's q/k/v
// is already [1,1,D] (no flat-split-into-heads reshape, which aborts the HTP),
// and run the recurrence per-head with simple batch-1 matmuls (no rank-4 batch).
//
// Per head h:  q_h = proj_q_h(x)[1,1,Dk]; k_h, v_h similarly; with gates gt_h/beta_h:
//   Ss = S_h * gt_h; kv = k_h@Ss; delta=(v_h-kv)*beta_h; Sp = Ss + k_h^T@delta; out_h = q_h@Sp
// Outputs per-head {Sp_h, out_h}. Confirms the projection layout finalizes for V79.
//
//   ./mllm-qwen3-aot-deltanet-perhead-c -aot_cfg qnn_aot_cfg_deltanet_step.json --heads 4
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
inline float dnW(int o, int i, int seed) {
  uint32_t h = (uint32_t)(o * 4099 + i * 131 + seed * 7919);
  h ^= h >> 13; h *= 2654435761u; h ^= h >> 15;
  return 0.04f * ((float)(h % 257) / 257.0f - 0.5f);
}

class DeltaNetPerHead final : public mllm::nn::Module {
  int H_, Dk_, Dv_, hidden_;
  std::vector<mllm::nn::Linear> pq_, pk_, pv_;

 public:
  DeltaNetPerHead() = default;
  DeltaNetPerHead(const std::string& name, int H, int Dk, int Dv, int hidden) : mllm::nn::Module(name) {
    H_ = H; Dk_ = Dk; Dv_ = Dv; hidden_ = hidden;
    for (int h = 0; h < H; ++h) {
      pq_.push_back(reg<mllm::nn::Linear>("pq_" + std::to_string(h), hidden, Dk, false));
      pk_.push_back(reg<mllm::nn::Linear>("pk_" + std::to_string(h), hidden, Dk, false));
      pv_.push_back(reg<mllm::nn::Linear>("pv_" + std::to_string(h), hidden, Dv, false));
    }
  }

  // in = [x[1,hidden], S_h0..S_h{H-1} [1,Dk,Dv], gt_h0..[1,1,1], beta_h0..[1,1,1]]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<mllm::AnyValue>&) override {
    namespace F = mllm::nn::functional;
    auto x = in[0];
    std::vector<Tensor> outs;
    for (int h = 0; h < H_; ++h) {
      auto S = in[1 + h];
      auto gt = in[1 + H_ + h];
      auto beta = in[1 + 2 * H_ + h];
      auto q = pq_[h](x).view({1, 1, Dk_});   // per-head projection -> [1,1,Dk] (no head-split reshape)
      auto k = pk_[h](x).view({1, 1, Dk_});
      auto v = pv_[h](x).view({1, 1, Dv_});
      auto Ss = S * gt;                        // [1,Dk,Dv]
      auto kv = F::matmul(k, Ss);              // [1,1,Dv]
      auto delta = (v - kv) * beta;            // [1,1,Dv]
      auto outer = F::matmul(k.transpose(1, 2), delta);  // [1,Dk,Dv]
      auto Sp = Ss + outer;
      auto out = F::matmul(q, Sp);             // [1,1,Dv]
      outs.push_back(Sp);
      outs.push_back(out);
    }
    return outs;
  }
};
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& H_arg = Argparse::add<int>("--heads").def(4);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }
  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get(), hidden = H * Dv;

  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"));
  for (int h = 0; h < H; ++h) ti.push_back(Tensor::zeros({1, Dk, Dv}, mllm::kFloat16).setName("S_h" + std::to_string(h)));
  for (int h = 0; h < H; ++h) ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("gt_h" + std::to_string(h)));
  for (int h = 0; h < H; ++h) ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("beta_h" + std::to_string(h)));

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  auto pushW = [&](const std::string& name, int out, int in, int seed) {
    auto t = Tensor::empty({out, in}, mllm::kFloat16).alloc();
    auto* p = t.ptr<mllm::mllm_fp16_t>();
    for (int o = 0; o < out; ++o)
      for (int i = 0; i < in; ++i) p[(size_t)o * in + i] = (mllm::mllm_fp16_t)dnW(o, i, seed);
    params->push(name, t.contiguous().setMemType(mllm::kParamsNormal).setName(name));
  };
  for (int h = 0; h < H; ++h) {
    pushW("model.pq_" + std::to_string(h) + ".weight", Dk, hidden, 100 + h);
    pushW("model.pk_" + std::to_string(h) + ".weight", Dk, hidden, 200 + h);
    pushW("model.pv_" + std::to_string(h) + ".weight", Dv, hidden, 300 + h);
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  DeltaNetPerHead m("model", H, Dk, Dv, hidden);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  qnn_aot_env.saveContext("context.0", "qwen3-deltanet-perhead.bin");
  mllm::print("DeltaNet per-head H={} Dk={} Dv={} -> qwen3-deltanet-perhead.bin", H, Dk, Dv);
});
