// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED full-attention BLOCK-PREFILL graph (Sq=B causal) for the Qwen3.5 hybrid —
// the prefill counterpart of compile_attn_decode.cpp (Sq=1). Processes a whole block of
// B tokens with dense causal attention (no KV cache; the block attends to itself). The
// 6 full-attention layers (interval 4) interleave with the 18 GatedDeltaNet layers.
//
// Reuses the decode layer's pieces (LPBQ per-head q/k/v/o, partial RoPE, q/k-norm 1+w,
// output gate) but with PER-TOKEN RoPE (sin/cos [1,B,rot]) and a causal Sq×Sq score
// matmul + additive causal mask + softmax. Attention math in fp32 (HTP fp32 matmul).
//
// I/O contract (first block; no past KV):
//   inputs : x[B,hidden], sin[1,B,rot], cos[1,B,rot], q_norm_w[1,1,D], k_norm_w[1,1,D],
//            eps[1,1,1], cmask[1,1,B,B]  (additive fp16: 0 if j<=i else -50000)
//   outputs: y[B,hidden], k_all[1,KV,D,B] (post-norm/rope K), v_all[1,KV,B,D]
//
//   ./mllm-qwen3-aot-attn-prefill-c -aot_cfg qnn_aot_cfg_attn_decode.json --seq 128 \
//        --params attn-prefill-l3-lpbq.mllm
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

#include "modeling_qwen_qnn_aot_sha.hpp"  // CONV2D_PROPERTY + ptq::QDQ helpers

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
}  // namespace

namespace mllm::models::qwen3::sha {

// Quantized full-attention block prefill (Sq=B) for Qwen3.5. LPBQ q/k/v/o + fp32 causal
// attention. Per-head projection layout (avoids the flat head-split reshape that aborts HTP).
class AttnPrefillLPBQ final : public nn::Module {
  int H_ = 8, KV_ = 2, D_ = 256, hidden_ = 2048, rot_ = 64, B_ = 128;
  int grp_ = 4;  // H_/KV_
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_;  // q_proj: hidden->2D (query|gate)
  nn::Conv2D o_proj_;

 public:
  AttnPrefillLPBQ() = default;
  AttnPrefillLPBQ(const std::string& name, int H, int KV, int D, int hidden, int rot, int B)
      : nn::Module(name), H_(H), KV_(KV), D_(D), hidden_(hidden), rot_(rot), B_(B), grp_(H / KV) {
    for (int h = 0; h < H; ++h) q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden, 2 * D, CONV2D_PROPERTY));
    for (int kv = 0; kv < KV; ++kv) {
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(kv), hidden, D, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(kv), hidden, D, CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", H * D, hidden, CONV2D_PROPERTY);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0];          // [B,hidden]
    auto sin = in[1];        // [1,B,rot]
    auto cos = in[2];        // [1,B,rot]
    auto q_norm_w = in[3];   // [1,1,D] (1+w baked)
    auto k_norm_w = in[4];   // [1,1,D]
    auto eps = in[5];        // [1,1,1]
    auto cmask = in[6];      // [1,1,B,B] additive
    const float scale = 1.0f / std::sqrt((float)D_);
    const int half = rot_ / 2;
    // eps as rank-4 [1,1,1,1]: a rank-3 [1,1,1] operand added to the rank-4 mean
    // mis-broadcasts on HTP, so padded/zero tokens skip eps -> rsqrt(0)=inf -> nan, which
    // then poisons real tokens' scores through the matmul over (masked) padded keys.
    auto eps_f = eps.to(kFloat32).view({1, 1, 1, 1}, true);
    auto sin4 = sin.view({1, 1, B_, rot_}, true);
    auto cos4 = cos.view({1, 1, B_, rot_}, true);

    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);  // [1,1,B,hidden]

    // hand-rolled RMSNorm over D (last dim), per token, fp32, weight 1+w -> fp16.
    auto rmsnorm = [&](Tensor t, Tensor w) {
      auto tf = t.to(kFloat32);
      // FLOOR the denominator: the norm runs fp16 on HTP, so eps=1e-6 underflows the fp16
      // subnormal floor -> for padded/zero tokens (mean=0) rsqrt(0)=inf -> nan, which then
      // poisons real tokens through the score matmul over (masked) padded keys. Real tokens
      // have mean(q^2)~10s, far above the 1e-4 floor, so they're unaffected.
      auto denom = F::clip(F::mean(tf * tf, -1, true) + eps_f, 1e-4f, 1e30f);
      auto inv = F::rsqrt(denom);
      return ((tf * inv) * w.to(kFloat32).view({1, 1, 1, D_}, true)).to(kFloat16);
    };
    // partial RoPE on the first rot_ channels of [1,1,B,D] (per-token, split-half conv).
    auto rope = [&](Tensor t) {
      auto rotp = t.slice({kAll, kAll, kAll, {0, rot_}}, true);    // [1,1,B,rot]
      auto pass = t.slice({kAll, kAll, kAll, {rot_, D_}}, true);   // [1,1,B,D-rot]
      auto x1 = rotp.slice({kAll, kAll, kAll, {0, half}}, true);
      auto x2 = rotp.slice({kAll, kAll, kAll, {half, rot_}}, true);
      auto rh = F::concat({-x2, x1}, -1);                          // [1,1,B,rot]
      auto roped = rotp * cos4 + rh * sin4;
      return F::concat({roped, pass}, -1);                         // [1,1,B,D]
    };
    auto proj = [&](std::vector<nn::Conv2D>& pr, int h, int outD, const std::string& tag) {
      return ptq::QDQ(this, pr[h](xq), tag + std::to_string(h)).to(kFloat16).view({1, 1, B_, outD}, true);
    };

    // ---- K/V projections + norm/rope (per kv head) ----
    std::vector<Tensor> Kf, Vf, k_all, v_all;
    for (int kv = 0; kv < KV_; ++kv) {
      auto k_n = rope(rmsnorm(proj(k_proj_, kv, D_, "k_out_qdq_h"), k_norm_w));  // [1,1,B,D]
      auto v_r = proj(v_proj_, kv, D_, "v_out_qdq_h");                           // [1,1,B,D]
      Kf.push_back(k_n);
      Vf.push_back(v_r);
      k_all.push_back(k_n.transpose(2, 3));   // [1,1,D,B]
      v_all.push_back(v_r);                   // [1,1,B,D]
    }

    // ---- per q-head causal attention ----
    std::vector<Tensor> head_outs, gates;
    for (int h = 0; h < H_; ++h) {
      auto qg = proj(q_proj_, h, 2 * D_, "q_out_qdq_h");           // [1,1,B,2D]
      auto q_raw = qg.slice({kAll, kAll, kAll, {0, D_}}, true);    // [1,1,B,D]
      gates.push_back(qg.slice({kAll, kAll, kAll, {D_, 2 * D_}}, true));
      auto q_n = rope(rmsnorm(q_raw, q_norm_w)).to(kFloat32);      // [1,1,B,D] fp32
      int kv = h / grp_;
      auto kT = Kf[kv].transpose(2, 3).to(kFloat32);              // [1,1,D,B]
      auto sc = F::matmul(q_n, kT);                               // [1,1,B,B]
      sc = sc.mulConstant(Tensor::constant(scale, kFloat32)) + cmask.to(kFloat32);
      sc = F::softmax(sc, -1);
      auto out_h = F::matmul(sc, Vf[kv].to(kFloat32)).to(kFloat16);  // [1,1,B,D]
      head_outs.push_back(out_h);
    }

    // concat heads -> [1,1,B,H*D], output gate, o_proj (LPBQ)
    auto y = F::concat(head_outs, -1);                            // [1,1,B,H*D]
    auto g = F::concat(gates, -1);
    y = y * F::sigmoid(g);
    auto cq = ptq::QDQ(this, y, "o_proj_input_qdq").view({1, 1, -1, H_ * D_}, true);
    auto o = ptq::QDQ(this, o_proj_(cq), "o_proj_output_qdq").to(kFloat16).view({B_, hidden_}, true);

    std::vector<Tensor> outs;
    outs.push_back(o);                            // y (index 0)
    outs.push_back(F::concat(k_all, 1));          // k_all [1,KV,D,B] (index 1)
    outs.push_back(F::concat(v_all, 1));          // v_all [1,KV,B,D] (index 2)
    return outs;
  }
};

}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& params_arg = Argparse::add<std::string>("--params").help("real LPBQ .mllm (from export_attn_prefill.py)").def("");
  auto& seq_arg = Argparse::add<int>("--seq").help("block length B").def(128);
  auto& H_arg = Argparse::add<int>("--heads").def(8);
  auto& kv_arg = Argparse::add<int>("--kv").def(2);
  auto& d_arg = Argparse::add<int>("--dim").def(256);
  auto& rot_arg = Argparse::add<int>("--rot").def(64);
  auto& out_arg = Argparse::add<std::string>("--out").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config"); return -1; }
  if (params_arg.get().empty()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--params required"); return -1; }

  const int H = H_arg.get(), KV = kv_arg.get(), D = d_arg.get(), rot = rot_arg.get(), B = seq_arg.get();
  const int hidden = 2048;

  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({B, hidden}, mllm::kFloat16).setName("x"));
  ti.push_back(Tensor::zeros({1, B, rot}, mllm::kFloat16).setName("sin"));
  ti.push_back(Tensor::zeros({1, B, rot}, mllm::kFloat16).setName("cos"));
  ti.push_back(Tensor::zeros({1, 1, D}, mllm::kFloat32).setName("q_norm_w"));
  ti.push_back(Tensor::zeros({1, 1, D}, mllm::kFloat32).setName("k_norm_w"));
  ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat32).setName("eps"));
  ti.push_back(Tensor::zeros({1, 1, B, B}, mllm::kFloat16).setName("cmask"));

  auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::AttnPrefillLPBQ m("model", H, KV, D, hidden, rot, B);
  m.load(real);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), real));
  pm.run();
  const std::string bin = out_arg.get().empty() ? ("qwen3-attn-prefill-s" + std::to_string(B) + ".bin") : out_arg.get();
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print(fmt::format("Attn block-prefill (LPBQ) B={} H={} KV={} D={} -> {} (graph model.0.s{})", B, H, KV, D, bin,
                          hidden));
});
