// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED GatedDeltaNet decode-step graph (Sq=1) — the w4a16-LPBQ counterpart
// of compile_deltanet_step.cpp. The weight-heavy projections (q/k/v/z + out_proj)
// are LPBQ Conv2D (int4 weights, uint16 activations, block size 16), exactly like
// the MLP/attn LPBQ microbenches; the recurrent core (gate math, l2norm, the
// state matmuls, gated RMSNorm·silu) stays fp16 because the recurrent state S has
// a dynamic, growing magnitude that no static QDQ scale can capture.
//
// Layout: the PER-HEAD projection loop proven in compile_deltanet_perhead.cpp —
// H separate Conv2D(hidden->head_dim) so each head's q/k/v/z is already [1,1,D]
// (no flat-split-into-heads reshape, which aborts the HTP). Each conv's uint16
// output is dequantized (.to(fp16)) into the fp16 recurrence; the gated output is
// re-quantized (uint16) before the LPBQ out_proj.
//
// Modes:
//   recur : per-head LPBQ q/k/v + fp16 recurrence only (gt/beta as inputs).
//           Smallest new thing — validates the LPBQ-conv -> fp16-recurrence
//           bridge and the per-head layout under the LPBQ recipe.
//   full  : complete decode layer — gate math (a/b/A_log/dt_bias), q/k l2norm,
//           z projection, recurrence, gated RMSNorm·silu, concat, LPBQ out_proj.
//
// Synthetic weights/scales by default (clears compiler + QNN finalize + latency,
// the microbench purpose). Real-weight accuracy needs the fused in_proj_qkv ->
// per-head slice (TODO, phase 2).
//
//   ./mllm-qwen3-aot-deltanet-decode-c -aot_cfg qnn_aot_cfg_deltanet_decode.json --mode recur --heads 16
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

#include "modeling_qwen_qnn_aot_sha.hpp"  // CONV2D_PROPERTY + ptq::QDQ helpers

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}

// Synthetic LPBQ-w4a16 conv weight + 2-level scales — identical scaffolding to the
// MLP/attn microbenches. PTQPass pulls {prefix}.weight (int4-in-int8), .scale1
// (uint4-in-uint8 per-block, [Out, In/G]), .scale2 (fp32 per-output-channel).
void pushLPBQConv(const mllm::ParameterFile::ptr_t& params, const std::string& prefix, int In, int Out, int G) {
  const int n_blk = In / G;
  // NON-degenerate synthetic values: all-identical weights trigger a constant-fold
  // fast path that under-reports latency (see attn microbench --params note). Vary
  // the int4 codes / block scales so the conv does real work and the latency is
  // representative (values are still meaningless for accuracy — that needs real PTQ).
  std::vector<int8_t> w((size_t)In * Out);
  for (size_t i = 0; i < w.size(); ++i) w[i] = (int8_t)(i % 15);          // int4 codes 0..14
  std::vector<uint8_t> s1((size_t)Out * n_blk);
  for (size_t i = 0; i < s1.size(); ++i) s1[i] = (uint8_t)(1 + i % 15);   // uint4 block scales 1..15
  std::vector<float> s2((size_t)Out);
  for (size_t o = 0; o < s2.size(); ++o) s2[o] = 0.005f + 0.0001f * (o % 17);
  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(prefix + ".weight", Tensor::fromVector(w, {1, 1, In, Out}, mllm::kInt8));
  push(prefix + ".scale1", Tensor::fromVector(s1, {(int)s1.size()}, mllm::kUInt8));
  push(prefix + ".scale2", Tensor::fromVector(s2, {(int)s2.size()}, mllm::kFloat32));
}

// Synthetic per-tensor uint16-asym QDQ scale/zero_point.
void pushQDQ(const mllm::ParameterFile::ptr_t& params, const std::string& qdq) {
  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(qdq + ".fake_quant.scale", Tensor::fromVector(std::vector<float>{1.0f / 256.0f}, {1}, mllm::kFloat32));
  push(qdq + ".fake_quant.zero_point", Tensor::fromVector(std::vector<int32_t>{0}, {1}, mllm::kInt32));
}
}  // namespace

namespace mllm::models::qwen3::sha {

// Quantized per-head deltanet decode step. `full` toggles the complete layer
// (gate math + l2norm + z + gated norm + out_proj) vs the recurrence-only core.
class DeltaNetDecodeLPBQ final : public nn::Module {
  int H_ = 0, Dk_ = 0, Dv_ = 0, hidden_ = 0;
  bool full_ = false, conv1d_ = false;
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_, z_proj_;
  nn::Conv2D out_proj_;

 public:
  DeltaNetDecodeLPBQ() = default;
  DeltaNetDecodeLPBQ(const std::string& name, int H, int Dk, int Dv, int hidden, bool full, bool conv1d)
      : nn::Module(name), H_(H), Dk_(Dk), Dv_(Dv), hidden_(hidden), full_(full), conv1d_(conv1d && full) {
    for (int h = 0; h < H; ++h) {
      auto hs = std::to_string(h);
      q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
      if (full) z_proj_.emplace_back(reg<nn::Conv2D>("z_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
    }
    if (full) out_proj_ = reg<nn::Conv2D>("out_proj", H * Dv, hidden, CONV2D_PROPERTY);
  }

  // recur inputs: [x[1,hidden], S_h0..[1,Dk,Dv], gt_h0..[1,1,1], beta_h0..[1,1,1]]
  // full  inputs: + eps[1,1,1], qscale[1,1,1], norm_w[1,1,Dv]
  // full+conv1d inputs: + cw_q[1,4,H*Dk], cw_k[1,4,H*Dk], cw_v[1,4,H*Dv],
  //                       cs_q[1,3,H*Dk], cs_k[1,3,H*Dk], cs_v[1,3,H*Dv]
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    auto x = in[0];
    Tensor eps, qscale, norm_w, cw_q, cw_k, cw_v, cs_q, cs_k, cs_v;
    if (full_) {
      eps = in[1 + 3 * H_];
      qscale = in[2 + 3 * H_];
      norm_w = in[3 + 3 * H_];
    }
    if (conv1d_) {
      cw_q = in[4 + 3 * H_]; cw_k = in[5 + 3 * H_]; cw_v = in[6 + 3 * H_];
      cs_q = in[7 + 3 * H_]; cs_k = in[8 + 3 * H_]; cs_v = in[9 + 3 * H_];
    }

    // Shared QDQ'd input to every per-head projection ([1,1,1,hidden] uint16).
    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);

    auto proj = [&](std::vector<nn::Conv2D>& p, int h, int D, const std::string& tag) {
      // LPBQ conv (uint16 out) -> QDQ (attach output scale) -> dequant to fp16.
      auto y = ptq::QDQ(this, p[h](xq), tag + std::to_string(h));
      return y.to(kFloat16).view({1, 1, D}, true);  // [1,1,D] fp16 for the recurrence
    };

    // Depthwise causal conv1d (kernel 4) + silu on a head's raw projection, BEFORE
    // l2norm — matches HF causal_conv1d_update: out = silu(sum_j w_j * window_j),
    // window = [s0,s1,s2,x_t]; new state = [s1,s2,x_t]. cw/cs are stacked over heads
    // ([1,4|3,H*D]); slice this head's D channels. new_cs collected for the output.
    std::vector<Tensor> new_cs_q, new_cs_k, new_cs_v;
    auto conv_silu = [&](Tensor proj_raw, Tensor cw_full, Tensor cs_full, int h, int D, std::vector<Tensor>& ncs) {
      auto cw = cw_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true);  // [1,4,D]
      auto cs = cs_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true);  // [1,3,D]
      auto win = F::concat({cs, proj_raw}, 1);                            // [1,4,D] = [s0,s1,s2,x_t]
      auto c = F::sum(win * cw, 1, true);                                 // [1,1,D] depthwise tap-sum
      c = c * F::sigmoid(c);                                              // silu
      ncs.push_back(win.slice({kAll, {1, 4}, kAll}, true));               // [1,3,D]
      return c;
    };

    std::vector<Tensor> gated_heads;
    std::vector<Tensor> outs;  // recur: {Sp_h, out_h}* ; full: {Sp_h}* then [new_cs_qkv] y
    for (int h = 0; h < H_; ++h) {
      Tensor S_h = in[1 + h];            // [1,Dk,Dv] recurrent state
      Tensor gt_h = in[1 + H_ + h];      // [1,1,1]
      Tensor beta_h = in[1 + 2 * H_ + h];// [1,1,1]
      auto q_raw = proj(q_proj_, h, Dk_, "q_out_qdq_h");
      auto k_raw = proj(k_proj_, h, Dk_, "k_out_qdq_h");
      auto v = proj(v_proj_, h, Dv_, "v_out_qdq_h");

      if (conv1d_) {  // conv1d+silu on q/k/v raw projections (z is NOT convolved)
        q_raw = conv_silu(q_raw, cw_q, cs_q, h, Dk_, new_cs_q);
        k_raw = conv_silu(k_raw, cw_k, cs_k, h, Dk_, new_cs_k);
        v = conv_silu(v, cw_v, cs_v, h, Dv_, new_cs_v);
      }

      // The recurrence runs in FP32 — HF's torch_recurrent_gated_delta_rule upcasts
      // q/k/v/g/beta/state to fp32. out=q@Sp is a near-cancellation (~1e-4 from terms
      // ~1e-2) whose direction fp16 can't hold (inputs are fp16-rounded); the gated
      // RMSNorm then amplifies that error ~1e3x. State is carried fp32 in/out too.
      auto S_f = S_h.to(kFloat32), gt_f = gt_h.to(kFloat32), beta_f = beta_h.to(kFloat32);
      auto eps_f = eps.to(kFloat32);
      Tensor q = q_raw.to(kFloat32), k = k_raw.to(kFloat32), v_f = v.to(kFloat32);
      if (full_) {
        // q/k L2-norm over Dk (fp32) + qscale on q.
        q = (q * F::rsqrt(F::sum(q * q, -1, true) + eps_f)) * qscale.to(kFloat32);
        k = k * F::rsqrt(F::sum(k * k, -1, true) + eps_f);
      }
      auto Ss = S_f * gt_f;                              // [1,Dk,Dv] fp32
      auto kv = F::matmul(k, Ss);                        // [1,1,Dv]
      auto delta = (v_f - kv) * beta_f;                  // [1,1,Dv]
      auto outer = F::matmul(k.transpose(1, 2), delta);  // [1,Dk,Dv]
      auto Sp = Ss + outer;                              // fp32 state out
      auto out = F::matmul(q, Sp);                       // [1,1,Dv] fp32
      outs.push_back(Sp);

      if (!full_) {
        outs.push_back(out.to(kFloat16));
        continue;
      }
      // Gated RMSNorm(out) over Dv * silu(z); gated -> fp16 for the LPBQ out_proj.
      // out=q@Sp is a near-cancellation and the elementwise ops run in fp16, so mean(out^2)
      // underflows to 0 and rsqrt(0+eps) -> inf (eps=1e-6 is an fp16 subnormal). RMSNorm is
      // scale-invariant, so scale out up by K and eps by K^2 to keep out^2 in fp16's normal
      // range while preserving the HF eps: normed = (out*K)*rsqrt(mean((out*K)^2)+eps*K^2).
      // K=200 (NOT 256: K^2=65536 > fp16 max 65504 -> inf); K^2=40000 is fp16-representable.
      const float Kc = 200.0f;
      auto K = Tensor::constant(Kc, kFloat32);
      auto K2 = Tensor::constant(Kc * Kc, kFloat32);
      auto os = out.mulConstant(K);                                          // out*K [1,1,Dv]
      auto inv = F::rsqrt(F::mean(os * os, -1, true) + eps_f.mulConstant(K2));
      auto z = proj(z_proj_, h, Dv_, "z_out_qdq_h").to(kFloat32);
      auto normed = (os * inv) * norm_w.to(kFloat32);
      auto gated = normed * (z * F::sigmoid(z));
      gated_heads.push_back(gated.to(kFloat16));
    }

    if (!full_) return outs;

    // Concat per-head gated outputs -> [1,1,H*Dv], LPBQ out_proj -> [1,hidden].
    auto cat = F::concat(gated_heads, -1);                                  // [1,1,H*Dv]
    auto cq = ptq::QDQ(this, cat, "out_proj_input_qdq").view({1, 1, -1, H_ * Dv_}, true);
    auto y = ptq::QDQ(this, out_proj_(cq), "out_proj_output_qdq").to(kFloat16).view({1, hidden_}, true);
    outs.push_back(y);  // y right after the H Sp states (index H)

    // Multi-token decode needs the UPDATED conv state carried to the next step.
    // Concat per-head new_cs ([1,3,D] each) over heads -> [1,3,H*D], appended after y
    // (indices H+1,H+2,H+3). Single-step validators just ignore these extra outputs.
    if (conv1d_) {
      outs.push_back(F::concat(new_cs_q, -1));   // new_cs_q [1,3,H*Dk]  (index H+1)
      outs.push_back(F::concat(new_cs_k, -1));   // new_cs_k [1,3,H*Dk]  (index H+2)
      outs.push_back(F::concat(new_cs_v, -1));   // new_cs_v [1,3,H*Dv]  (index H+3)
    }
    return outs;
  }
};

}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& mode_arg = Argparse::add<std::string>("--mode").help("recur | full").def("recur");
  auto& params_arg = Argparse::add<std::string>("--params").help("real per-head LPBQ .mllm (from export_deltanet_decode.py); else synthetic").def("");
  auto& conv_arg = Argparse::add<bool>("--conv1d").help("full mode: include depthwise causal conv1d (kernel 4) + silu on q/k/v").def(false);
  auto& H_arg = Argparse::add<int>("--heads").def(16);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get(), hidden = H * Dv;
  const std::string mode = mode_arg.get();
  const bool full = (mode == "full");
  if (mode != "recur" && mode != "full") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Unknown --mode: {} (expected recur|full)", mode);
    return -1;
  }
  const int G = 16;
  const bool conv1d = full && conv_arg.get();
  const int kd = H * Dk, vd = H * Dv;  // fused key/value widths

  // ---- inputs ----
  std::vector<Tensor> ti;
  ti.push_back(Tensor::zeros({1, hidden}, mllm::kFloat16).setName("x"));
  for (int h = 0; h < H; ++h) ti.push_back(Tensor::zeros({1, Dk, Dv}, mllm::kFloat32).setName("S_h" + std::to_string(h)));  // fp32 recurrent state
  for (int h = 0; h < H; ++h) ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("gt_h" + std::to_string(h)));
  for (int h = 0; h < H; ++h) ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("beta_h" + std::to_string(h)));
  if (full) {
    ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("eps"));
    ti.push_back(Tensor::zeros({1, 1, 1}, mllm::kFloat16).setName("qscale"));
    ti.push_back(Tensor::zeros({1, 1, Dv}, mllm::kFloat16).setName("norm_w"));
  }
  if (conv1d) {
    ti.push_back(Tensor::zeros({1, 4, kd}, mllm::kFloat16).setName("cw_q"));
    ti.push_back(Tensor::zeros({1, 4, kd}, mllm::kFloat16).setName("cw_k"));
    ti.push_back(Tensor::zeros({1, 4, vd}, mllm::kFloat16).setName("cw_v"));
    ti.push_back(Tensor::zeros({1, 3, kd}, mllm::kFloat16).setName("cs_q"));
    ti.push_back(Tensor::zeros({1, 3, kd}, mllm::kFloat16).setName("cs_k"));
    ti.push_back(Tensor::zeros({1, 3, vd}, mllm::kFloat16).setName("cs_v"));
  }

  // ---- params: real per-head LPBQ checkpoint, or synthetic ----
  if (!params_arg.get().empty()) {
    // The .mllm from export_deltanet_decode.py already carries every name this
    // module loads (model.{q,k,v,z}_proj.<h>.{weight,scale1,scale2}, model.out_proj.*,
    // and all model.*_qdq.fake_quant.{scale,zero_point}). Load it verbatim.
    auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
        qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
    sha::DeltaNetDecodeLPBQ m("model", H, Dk, Dv, hidden, full, conv1d);
    m.load(real);
    auto ir = mllm::ir::trace_(m, ti);
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), real));
    pm.run();
    const std::string bin = "qwen3-deltanet-decode-" + mode + ".bin";
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print(fmt::format("DeltaNet decode (LPBQ, REAL weights) mode={} H={} -> {} (graph model.0.s{})", mode, H, bin,
                            hidden));
    return 0;
  }

  // ---- synthetic params ----
  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  pushQDQ(params, "model.qkv_input_qdq");
  for (int h = 0; h < H; ++h) {
    auto hs = std::to_string(h);
    pushLPBQConv(params, "model.q_proj." + hs, hidden, Dk, G);
    pushLPBQConv(params, "model.k_proj." + hs, hidden, Dk, G);
    pushLPBQConv(params, "model.v_proj." + hs, hidden, Dv, G);
    pushQDQ(params, "model.q_out_qdq_h" + hs);
    pushQDQ(params, "model.k_out_qdq_h" + hs);
    pushQDQ(params, "model.v_out_qdq_h" + hs);
    if (full) {
      pushLPBQConv(params, "model.z_proj." + hs, hidden, Dv, G);
      pushQDQ(params, "model.z_out_qdq_h" + hs);
    }
  }
  if (full) {
    pushLPBQConv(params, "model.out_proj", H * Dv, hidden, G);
    pushQDQ(params, "model.out_proj_input_qdq");
    pushQDQ(params, "model.out_proj_output_qdq");
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::DeltaNetDecodeLPBQ m("model", H, Dk, Dv, hidden, full, conv1d);
  m.load(params);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
  pm.run();
  const std::string bin = "qwen3-deltanet-decode-" + mode + ".bin";
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print(fmt::format("DeltaNet decode (LPBQ) mode={} H={} Dk={} Dv={} -> {} (graph model.0.s{})", mode, H, Dk, Dv,
                          bin, hidden));
});
