// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED full-attention decode-step graph (Sq=1) for the Qwen3.5 hybrid model —
// the attention counterpart of compile_deltanet_decode.cpp. The 6 full-attention
// layers (interval 4) interleave with the 18 GatedDeltaNet layers; this builds ONE
// such layer as a per-layer QNN .bin for the host-orchestrated decode loop.
//
// Precision strategy (same lesson as deltanet): the weight-heavy projections
// (q/k/v/o) are LPBQ Conv2D (int4 weights, uint16 activations, block 16); the
// attention math (q/k-norm, partial RoPE, QK^T, masked softmax, attn@V, output
// gate) runs in fp16 — dynamic-range-sensitive and cheap at Sq=1, no static QDQ
// scale fits. KV cache is fp16, host-managed as a ring buffer.
//
// Qwen3.5-2B attention specifics:
//   H=8 query heads, KV=2 kv heads (GQA group 4), head_dim D=256, hidden=2048,
//   q_proj is DOUBLE-width per head: [query D | gate D] (attn_output_gate),
//   partial RoPE on the first rotary_dim=64 channels, q/k RMSNorm with
//   add_unit_offset (baked into the exported weight as 1+w), out = o_proj(out *
//   sigmoid(gate)).
//
// I/O contract (matches aot_rt/KVCacheManager ring buffer):
//   inputs : x[1,hidden], sin[1,1,rot], cos[1,1,rot],
//            past_k[1,KV,D,P], past_v[1,KV,P,D], mask[1,1,1,ctx]   (P = ctx-1)
//   outputs: y[1,hidden], k_new[1,KV,D,1], v_new[1,KV,1,D]
//   mask is ADDITIVE fp16 (0 for valid positions, -50000 for empty/future).
//
//   ./mllm-qwen3-aot-attn-decode-c -aot_cfg qnn_aot_cfg_attn_decode.json --ctx 256 --params attn-l3-lpbq.mllm
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

// Synthetic LPBQ-w4a16 conv weight + 2-level scales (same scaffolding as the
// deltanet/MLP microbenches). For latency/finalize smoke only; accuracy needs --params.
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

// Quantized full-attention decode step (Sq=1) for Qwen3.5. LPBQ q/k/v/o + fp16
// attention math. Per-head projection layout (avoids the flat head-split reshape
// that aborts the HTP), exactly like the deltanet decode graph.
class AttnDecodeLPBQ final : public nn::Module {
  int H_ = 8, KV_ = 2, D_ = 256, hidden_ = 2048, rot_ = 64, ctx_ = 256;
  int grp_ = 4;  // H_/KV_
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_;  // q_proj: hidden->2D (query|gate)
  nn::Conv2D o_proj_;

 public:
  AttnDecodeLPBQ() = default;
  AttnDecodeLPBQ(const std::string& name, int H, int KV, int D, int hidden, int rot, int ctx, float eps)
      : nn::Module(name), H_(H), KV_(KV), D_(D), hidden_(hidden), rot_(rot), ctx_(ctx), grp_(H / KV) {
    for (int h = 0; h < H; ++h) {
      auto hs = std::to_string(h);
      q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + hs, hidden, 2 * D, CONV2D_PROPERTY));  // query|gate
    }
    for (int h = 0; h < KV; ++h) {
      auto hs = std::to_string(h);
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + hs, hidden, D, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + hs, hidden, D, CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", H * D, hidden, CONV2D_PROPERTY);
  }

  // inputs: x[1,hidden], sin[1,1,rot], cos[1,1,rot], q_norm_w[1,1,D], k_norm_w[1,1,D],
  //         eps[1,1,1], past_k[1,KV,D,P], past_v[1,KV,P,D], mask[1,1,1,ctx]
  // q/k-norm is computed by hand in fp32 (RMSNormOp requires a uint16 weight; the
  // by-hand form keeps it fp32 and accepts the norm weight as a graph input with
  // add_unit_offset (1+w) already baked in at export).
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0];
    auto sin = in[1];       // [1,1,rot]
    auto cos = in[2];       // [1,1,rot]
    auto q_norm_w = in[3];  // [1,1,D]  (1+w baked)
    auto k_norm_w = in[4];  // [1,1,D]
    auto eps = in[5];       // [1,1,1]
    auto past_k = in[6];    // [1,KV,D,P]
    auto past_v = in[7];    // [1,KV,P,D]
    auto mask = in[8];      // [1,1,1,ctx] additive fp16
    const float scale = 1.0f / std::sqrt((float)D_);
    const int half = rot_ / 2;
    auto eps_f = eps.to(kFloat32);

    // Shared QDQ'd input to every projection ([1,1,1,hidden] uint16).
    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);

    // hand-rolled RMSNorm over the last dim (D), fp32, weight = 1+w (baked). -> fp16.
    auto rmsnorm = [&](Tensor t, Tensor w) {
      auto tf = t.to(kFloat32);
      auto inv = F::rsqrt(F::mean(tf * tf, -1, true) + eps_f);
      return ((tf * inv) * w.to(kFloat32)).to(kFloat16);
    };

    // partial RoPE on the first rot_ channels of a [1,1,D] head (split-half conv).
    auto rope = [&](Tensor t) {
      auto rotp = t.slice({kAll, kAll, {0, rot_}}, true);     // [1,1,rot]
      auto pass = t.slice({kAll, kAll, {rot_, D_}}, true);    // [1,1,D-rot]
      auto x1 = rotp.slice({kAll, kAll, {0, half}}, true);    // [1,1,half]
      auto x2 = rotp.slice({kAll, kAll, {half, rot_}}, true); // [1,1,half]
      auto rh = F::concat({-x2, x1}, -1);                     // rotate_half [1,1,rot]
      auto roped = rotp * cos + rh * sin;                     // [1,1,rot]
      return F::concat({roped, pass}, -1);                    // [1,1,D]
    };

    // ---- K/V projections + norm + rope + cache concat (per kv head) ----
    std::vector<Tensor> Kfull, Vfull, k_new_heads, v_new_heads;
    for (int kv = 0; kv < KV_; ++kv) {
      auto ks = std::to_string(kv);
      auto k_raw = ptq::QDQ(this, k_proj_[kv](xq), "k_out_qdq_h" + ks).to(kFloat16).view({1, 1, D_}, true);
      auto v_raw = ptq::QDQ(this, v_proj_[kv](xq), "v_out_qdq_h" + ks).to(kFloat16).view({1, 1, D_}, true);
      auto k_n = rmsnorm(k_raw, k_norm_w);   // RMSNorm over D
      k_n = rope(k_n);                       // [1,1,D]
      // new k for cache: [1,1,D,1] ; new v: [1,1,1,D]
      auto k_t = k_n.view({1, 1, D_, 1}, true);
      auto v_t = v_raw.view({1, 1, 1, D_}, true);
      k_new_heads.push_back(k_t);
      v_new_heads.push_back(v_t);
      // K_full = concat(past_k[kv] [1,1,D,P], k_t [1,1,D,1]) -> [1,1,D,ctx]
      auto pk = past_k.slice({kAll, {kv, kv + 1}, kAll, kAll}, true).view({1, 1, D_, ctx_ - 1}, true);
      auto pv = past_v.slice({kAll, {kv, kv + 1}, kAll, kAll}, true).view({1, 1, ctx_ - 1, D_}, true);
      Kfull.push_back(F::concat({pk, k_t}, -1));   // [1,1,D,ctx]
      Vfull.push_back(F::concat({pv, v_t}, 2));    // [1,1,ctx,D]
    }

    // ---- per q-head attention ----
    std::vector<Tensor> head_outs, gates;
    for (int h = 0; h < H_; ++h) {
      auto hs = std::to_string(h);
      auto qg = ptq::QDQ(this, q_proj_[h](xq), "q_out_qdq_h" + hs).to(kFloat16).view({1, 1, 2 * D_}, true);
      auto q_raw = qg.slice({kAll, kAll, {0, D_}}, true);       // query [1,1,D]
      auto gate = qg.slice({kAll, kAll, {D_, 2 * D_}}, true);   // gate  [1,1,D]
      gates.push_back(gate);
      auto q_n = rmsnorm(q_raw, q_norm_w);
      q_n = rope(q_n);                                          // [1,1,D]
      int kv = h / grp_;
      // attn = q @ K_full in fp32 (scores/softmax precision; HTP supports fp32 matmul).
      auto q4 = q_n.view({1, 1, 1, D_}, true).to(kFloat32);     // [1,1,1,D]
      auto attn = F::matmul(q4, Kfull[kv].to(kFloat32));        // [1,1,1,ctx]
      attn = attn.mulConstant(Tensor::constant(scale, kFloat32)) + mask.to(kFloat32);  // [1,1,1,ctx]
      attn = F::softmax(attn, -1);
      auto out_h = F::matmul(attn, Vfull[kv].to(kFloat32)).to(kFloat16);  // [1,1,1,D]
      head_outs.push_back(out_h.view({1, 1, D_}, true));
    }

    // concat heads -> [1,1,H*D], apply output gate sigmoid, o_proj (LPBQ)
    auto y = F::concat(head_outs, -1);                          // [1,1,H*D]
    auto g = F::concat(gates, -1);                              // [1,1,H*D]
    y = y * F::sigmoid(g);
    auto cq = ptq::QDQ(this, y, "o_proj_input_qdq").view({1, 1, -1, H_ * D_}, true);
    auto o = ptq::QDQ(this, o_proj_(cq), "o_proj_output_qdq").to(kFloat16).view({1, hidden_}, true);

    std::vector<Tensor> outs;
    outs.push_back(o);                                  // y  (index 0)
    outs.push_back(F::concat(k_new_heads, 1));          // k_new [1,KV,D,1] (index 1)
    outs.push_back(F::concat(v_new_heads, 1));          // v_new [1,KV,1,D] (index 2)
    return outs;
  }
};

}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& params_arg = Argparse::add<std::string>("--params").help("real LPBQ .mllm (from export_attn_decode.py); else synthetic").def("");
  auto& ctx_arg = Argparse::add<int>("--ctx").help("KV cache context length").def(256);
  auto& H_arg = Argparse::add<int>("--heads").def(8);
  auto& kv_arg = Argparse::add<int>("--kv").def(2);
  auto& d_arg = Argparse::add<int>("--dim").def(256);
  auto& rot_arg = Argparse::add<int>("--rot").def(64);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int H = H_arg.get(), KV = kv_arg.get(), D = d_arg.get(), rot = rot_arg.get(), ctx = ctx_arg.get();
  const int hidden = 2048;
  const float eps = 1e-6f;
  const int G = 16;

  // ---- inputs (names match the runtime binding order) ----
  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"));
  ti.push_back(Tensor::zeros({1, 1, rot}, mllm::kFloat16).setName("sin"));
  ti.push_back(Tensor::zeros({1, 1, rot}, mllm::kFloat16).setName("cos"));
  ti.push_back(Tensor::zeros({1, 1, D}, mllm::kFloat16).setName("q_norm_w"));
  ti.push_back(Tensor::zeros({1, 1, D}, mllm::kFloat16).setName("k_norm_w"));
  ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("eps"));
  ti.push_back(Tensor::zeros({1, KV, D, ctx - 1}, mllm::kFloat16).setName("past_k"));
  ti.push_back(Tensor::zeros({1, KV, ctx - 1, D}, mllm::kFloat16).setName("past_v"));
  ti.push_back(Tensor::zeros({1, 1, 1, ctx}, mllm::kFloat16).setName("mask"));

  auto compile = [&](const mllm::ParameterFile::ptr_t& params) {
    auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
        qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
    sha::AttnDecodeLPBQ m("model", H, KV, D, hidden, rot, ctx, eps);
    m.load(params);
    auto ir = mllm::ir::trace_(m, ti);
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
    pm.run();
    const std::string bin = "qwen3-attn-decode.bin";
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print(fmt::format("Attn decode (LPBQ) H={} KV={} D={} ctx={} -> {} (graph model.0.s{})", H, KV, D, ctx, bin,
                            hidden));
  };

  if (!params_arg.get().empty()) {
    auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    compile(real);
    return 0;
  }

  // ---- synthetic params ----
  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  pushQDQ(params, "model.qkv_input_qdq");
  for (int h = 0; h < H; ++h) {
    auto hs = std::to_string(h);
    pushLPBQConv(params, "model.q_proj." + hs, hidden, 2 * D, G);
    pushQDQ(params, "model.q_out_qdq_h" + hs);
  }
  for (int kv = 0; kv < KV; ++kv) {
    auto ks = std::to_string(kv);
    pushLPBQConv(params, "model.k_proj." + ks, hidden, D, G);
    pushLPBQConv(params, "model.v_proj." + ks, hidden, D, G);
    pushQDQ(params, "model.k_out_qdq_h" + ks);
    pushQDQ(params, "model.v_out_qdq_h" + ks);
  }
  pushLPBQConv(params, "model.o_proj", H * D, hidden, G);
  pushQDQ(params, "model.o_proj_input_qdq");
  pushQDQ(params, "model.o_proj_output_qdq");

  compile(params);
});
