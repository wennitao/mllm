// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Standalone fake-scale w4a16-LPBQ attention-projection microbench compiler.
// Mirrors compile_mlp_lpbq_microbench.cpp but for Qwen3 attention linears
// (q_proj / k_proj / v_proj / o_proj). Same LPBQ Conv2D scaffolding + uint16
// QDQ wrappers around inputs/outputs; no checkpoint required (synthetic
// weights and scales — only shapes/encoding need to satisfy the compiler).
//
// Purpose: anchor het-pipeline simulator's attention-block latency model
// (tools/het_sim/qwen3_1p7b_block.py) with measured LPBQ projection costs
// — the production block uses w4a16 LPBQ for these matmuls, but only the
// MLP matmuls have a faithful microbench so far. Without this, the block
// simulator either (a) uses fp16 gemm_latency numbers — pessimistic by 1.6–5×
// for memory-bound projections, or (b) extrapolates — which the user
// explicitly rejected.
//
// Usage:
//   ./compile_attn_lpbq_microbench -aot_cfg qnn_aot_cfg_attn_lpbq_microbench.json --mode q  --sq 1024
//   ./compile_attn_lpbq_microbench -aot_cfg qnn_aot_cfg_attn_lpbq_microbench.json --mode kv --sq 1024
//   ./compile_attn_lpbq_microbench -aot_cfg qnn_aot_cfg_attn_lpbq_microbench.json --mode o  --sq 1024
//   ./compile_attn_lpbq_microbench -aot_cfg qnn_aot_cfg_attn_lpbq_microbench.json --mode qkv --sq 1024
//
// Modes:
//   q   : single Conv2D q_proj  [Sq, H=2048]    -> [Sq, Hq*D=2048]
//   kv  : single Conv2D for k OR v (same shape) [Sq, H=2048] -> [Sq, Hkv*D=1024]
//   o   : single Conv2D o_proj  [Sq, Hq*D=2048] -> [Sq, H=2048]
//   qkv : three Conv2Ds (q, k, v) sharing the QDQ'd input — one dispatch,
//         outputs (q, k, v). Compares against q+kv+kv summed.
//
// Output bin: qwen3-attn-lpbq-<mode>.bin, graph "model.0.s<Sq>" (single graph).

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
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
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) { return std::string(qairt_root) + "/lib/x86_64-linux-clang/"; }
  return "/opt/qcom/aistack/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}

// Synthetic LPBQ-w4a16 conv weight + 2-level scales (same scaffolding as the
// MLP microbench — PTQPass pulls {prefix}.weight (int4-in-int8), .scale1
// (uint4-in-uint8 per-block), .scale2 (fp32 per-output-channel)).
void pushLPBQConv(const mllm::ParameterFile::ptr_t& params, const std::string& prefix, int In, int Out, int G) {
  const int n_blk = In / G;
  std::vector<int8_t> w((size_t)In * Out, /*int4 mid*/ 8);
  std::vector<uint8_t> s1((size_t)Out * n_blk, /*uint4 mid*/ 8);
  std::vector<float> s2((size_t)Out, 0.01f);
  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(prefix + ".weight", Tensor::fromVector(w, {1, 1, In, Out}, mllm::kInt8));
  // Match production: pymllm qlinear.py flattens scale1/scale2 before saving
  // (rank-1). setupComplexTensorQuantization asserts both are rank 1.
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

// One LPBQ Conv2D projection wrapped in input/output QDQ. Works for q_proj,
// k_proj, v_proj, and o_proj (the only shape-dependent thing is in/out dim).
class OneProjLPBQ final : public nn::Module {
  nn::Conv2D proj_;
  int in_, out_;
 public:
  OneProjLPBQ() = default;
  OneProjLPBQ(const std::string& name, int in_dim, int out_dim) : nn::Module(name), in_(in_dim), out_(out_dim) {
    proj_ = reg<nn::Conv2D>("proj", in_dim, out_dim, CONV2D_PROPERTY);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = ptq::QDQ(this, inputs[0], "proj_input_qdq").view({1, 1, -1, in_}, true);
    auto y = ptq::QDQ(this, proj_(x), "proj_output_qdq").view({1, -1, out_}, true);
    return {y};
  }
};

// Three LPBQ Conv2Ds (q, k, v) sharing the QDQ'd input. One graph, one
// dispatch, three outputs — checks how close fused-qkv-dispatch is to
// summing q + 2·kv (any savings come from shared input QDQ and one-shot
// QNN dispatch overhead).
class QKVLPBQ final : public nn::Module {
  nn::Conv2D q_proj_, k_proj_, v_proj_;
  int hidden_, q_out_, kv_out_;
 public:
  QKVLPBQ() = default;
  QKVLPBQ(const std::string& name, int hidden, int q_out, int kv_out)
      : nn::Module(name), hidden_(hidden), q_out_(q_out), kv_out_(kv_out) {
    q_proj_ = reg<nn::Conv2D>("q_proj", hidden, q_out, CONV2D_PROPERTY);
    k_proj_ = reg<nn::Conv2D>("k_proj", hidden, kv_out, CONV2D_PROPERTY);
    v_proj_ = reg<nn::Conv2D>("v_proj", hidden, kv_out, CONV2D_PROPERTY);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = ptq::QDQ(this, inputs[0], "qkv_input_qdq").view({1, 1, -1, hidden_}, true);
    auto q = ptq::QDQ(this, q_proj_(x), "q_output_qdq").view({1, -1, q_out_}, true);
    auto k = ptq::QDQ(this, k_proj_(x), "k_output_qdq").view({1, -1, kv_out_}, true);
    auto v = ptq::QDQ(this, v_proj_(x), "v_output_qdq").view({1, -1, kv_out_}, true);
    return {q, k, v};
  }
};

// Full Qwen3 MLP (gate/up/silu/mul/down, all LPBQ) + a closing QDQ — mirror
// of compile_mlp_lpbq_microbench's FullMLPLPBQ, included here so the `block`
// mode can emit qkv + o + mlp into ONE bin (one initQnnBackend call).
class FullMLPLPBQBlock final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_size_;
 public:
  FullMLPLPBQBlock() = default;
  FullMLPLPBQBlock(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    mlp_ = reg<Qwen3MLP>("", cfg);
    hidden_size_ = cfg.hidden_size;
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = mlp_.forward(inputs, args)[0];
    return {ptq::QDQ(this, o, "down_proj_output_qdq")};
  }
};

// Pre-attention all-NPU graph: in_norm + qkv + q/k_norm + q/k rope-proxy.
// Output: residual (= input x), qr, kr, v. The GQA core (Q·K^T → softmax → @V)
// runs in a separate CPU/NPU-pipelined path and is OUT OF SCOPE for this bench;
// post-attention picks up from attn_output downstream.
//
// rope proxy: SiLU. Matches the bandwidth profile the het_block driver uses
// for its GPU rope path, so the all-NPU vs het comparison is apples-to-apples
// (both pay the same kernel-cost for the rope-equivalent op).
class PreAttnLPBQNoGqa final : public nn::Module {
  nn::RMSNorm in_norm_;
  nn::Conv2D q_proj_, k_proj_, v_proj_;
  int hidden_, q_out_, kv_out_, Hq_, Hkv_, D_;
 public:
  PreAttnLPBQNoGqa() = default;
  PreAttnLPBQNoGqa(const std::string& name, const Qwen3Config& cfg)
      : nn::Module(name) {
    hidden_ = cfg.hidden_size;
    Hq_     = cfg.num_attention_heads;
    Hkv_    = cfg.num_key_value_heads;
    D_      = cfg.head_dim;
    q_out_  = Hq_  * D_;
    kv_out_ = Hkv_ * D_;
    in_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    q_proj_  = reg<nn::Conv2D>("q_proj", hidden_, q_out_,  CONV2D_PROPERTY);
    k_proj_  = reg<nn::Conv2D>("k_proj", hidden_, kv_out_, CONV2D_PROPERTY);
    v_proj_  = reg<nn::Conv2D>("v_proj", hidden_, kv_out_, CONV2D_PROPERTY);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    // SCOPE: input_layernorm + qkv only. The per-head q/k_norm and RoPE are
    // OUT OF SCOPE here — production uses per-head Conv2D form (16+8 separate)
    // which would balloon the bench. RoPE has no qti.aisw:SiLU and would need
    // the 7-node decomposition. Both pieces are well-characterized:
    //   - q/k_norm at Sq=1024: ~0.27 ms (NPU bandwidth-bound)
    //   - q/k_rope at Sq=1024: ~0.85 ms (NPU 9.9 GB/s, per gemm_latency.md)
    // So the measured pre_attn here UNDER-COUNTS the production all-NPU
    // pre-attention work by ~1.1 ms. Add this back in the final comparison.
    auto x = ptq::QDQ(this, inputs[0], "in_input_qdq");
    auto residual = x;
    auto xv = x.view({1, 1, -1, hidden_}, true);
    auto xn = in_norm_(xv);
    auto xn_q = ptq::QDQ(this, xn, "qkv_input_qdq");
    auto q = ptq::QDQ(this, q_proj_(xn_q), "q_proj_out_qdq").view({1, -1, q_out_},  true);
    auto k = ptq::QDQ(this, k_proj_(xn_q), "k_proj_out_qdq").view({1, -1, kv_out_}, true);
    auto v = ptq::QDQ(this, v_proj_(xn_q), "v_proj_out_qdq").view({1, -1, kv_out_}, true);
    return {residual.view({1, -1, hidden_}, true), q, k, v};
  }
};

// Post-attention all-NPU graph: o_proj + res1 + post_norm + MLP + res2.
// Inputs: residual (from pre_attn, fp16-equivalent), attn_output (= GQA result,
// shape [1, Sq, Hq*D]). Output: y = next-layer input.
class PostAttnLPBQNoGqa final : public nn::Module {
  nn::Conv2D o_proj_;
  nn::RMSNorm post_norm_;
  Qwen3MLP mlp_;
  int hidden_, q_out_;
 public:
  PostAttnLPBQNoGqa() = default;
  PostAttnLPBQNoGqa(const std::string& name, const Qwen3Config& cfg)
      : nn::Module(name) {
    hidden_ = cfg.hidden_size;
    q_out_  = cfg.num_attention_heads * cfg.head_dim;
    o_proj_    = reg<nn::Conv2D>("o_proj", q_out_, hidden_, CONV2D_PROPERTY);
    post_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
    mlp_       = reg<Qwen3MLP>("mlp", cfg);
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    // inputs[0] = residual [1,Sq,H], inputs[1] = attn_out [1,Sq,q_out]
    auto residual = ptq::QDQ(this, inputs[0], "res_input_qdq");
    auto attn = ptq::QDQ(this, inputs[1], "o_input_qdq").view({1, 1, -1, q_out_}, true);
    auto o = ptq::QDQ(this, o_proj_(attn), "o_out_qdq").view({1, -1, hidden_}, true);
    auto x1 = ptq::QDQ(this, residual + o, "res1_qdq");
    auto x1v = x1.view({1, 1, -1, hidden_}, true);
    auto x1n = post_norm_(x1v).view({1, -1, hidden_}, true);
    auto mlp_out = ptq::QDQ(this, mlp_.forward({x1n}, {})[0], "mlp_out_qdq");
    auto y = ptq::QDQ(this, x1 + mlp_out, "res2_qdq");
    return {y};
  }
};

}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& aot_cfg_arg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_arg = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath()).help("QNN AOT env path.");
  auto& sq_arg = Argparse::add<int>("--sq").help("Compiled Sq (sequence length)").def(1024);
  auto& mode_arg = Argparse::add<std::string>("--mode").help("q | kv | o | qkv | block | split").def("q");
  auto& params_arg = Argparse::add<std::string>("--params").help("Real ptq_lpbq .mllm — use real weights instead of synthetic (synthetic weights trigger degenerate fast path)").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg_arg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  // Qwen3-1.7B geometry (matches qwen3_1p7b_block.py).
  const int hidden = 2048;
  const int n_q_heads = 16, n_kv_heads = 8, head_dim = 128;
  const int q_out = n_q_heads * head_dim;     // 2048
  const int kv_out = n_kv_heads * head_dim;   // 1024
  const int G = 16;                           // LPBQ block size

  const int Seq = sq_arg.get();
  const std::string mode = mode_arg.get();

  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  const std::string params_path = params_arg.get();
  const bool use_real_params = !params_path.empty();

  // Helper: copy a single conv's weight/scale1/scale2 from real .mllm.
  // Real .mllm has params at model.layers.0.{self_attn,mlp}.{name}.{weight,scale1,scale2}.
  auto copy_real_conv = [&](const std::string& src_name, const std::string& dst_name) {
    auto r = mllm::load(params_path, mllm::ModelFileVersion::kV2, mllm::kCPU, false);
    int copied = 0;
    for (auto& [k, t] : r->dict()) {
      if (k.rfind(src_name + ".", 0) != 0) continue;
      std::string suffix = k.substr(src_name.size());
      std::string dst = dst_name + suffix;
      params->push(dst, t.setName(dst));
      ++copied;
    }
    return copied;
  };

  if (mode == "q") {
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.q_proj", "model.proj");
    } else {
      pushLPBQConv(params, "model.proj", hidden, q_out, G);
    }
    pushQDQ(params, "model.proj_input_qdq");
    pushQDQ(params, "model.proj_output_qdq");
  } else if (mode == "kv") {
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.k_proj", "model.proj");
    } else {
      pushLPBQConv(params, "model.proj", hidden, kv_out, G);
    }
    pushQDQ(params, "model.proj_input_qdq");
    pushQDQ(params, "model.proj_output_qdq");
  } else if (mode == "o") {
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.o_proj", "model.proj");
    } else {
      pushLPBQConv(params, "model.proj", q_out, hidden, G);
    }
    pushQDQ(params, "model.proj_input_qdq");
    pushQDQ(params, "model.proj_output_qdq");
  } else if (mode == "qkv") {
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.q_proj", "model.q_proj");
      copy_real_conv("model.layers.0.self_attn.k_proj", "model.k_proj");
      copy_real_conv("model.layers.0.self_attn.v_proj", "model.v_proj");
    } else {
      pushLPBQConv(params, "model.q_proj", hidden, q_out, G);
      pushLPBQConv(params, "model.k_proj", hidden, kv_out, G);
      pushLPBQConv(params, "model.v_proj", hidden, kv_out, G);
    }
    pushQDQ(params, "model.qkv_input_qdq");
    pushQDQ(params, "model.q_output_qdq");
    pushQDQ(params, "model.k_output_qdq");
    pushQDQ(params, "model.v_output_qdq");
  } else if (mode == "block") {
    // Three graphs in one bin: qkv (3 convs), o (1 conv), mlp (gate/up/silu/mul/down).
    // Module prefixes are distinct: qkv.*, o_proj.*, mlp.*  → so the same param
    // names below don't collide.
    const int intermediate = 6144;
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.q_proj", "qkv.q_proj");
      copy_real_conv("model.layers.0.self_attn.k_proj", "qkv.k_proj");
      copy_real_conv("model.layers.0.self_attn.v_proj", "qkv.v_proj");
      copy_real_conv("model.layers.0.self_attn.o_proj", "o_proj.proj");
      copy_real_conv("model.layers.0.mlp.gate_proj",    "mlp.gate_proj");
      copy_real_conv("model.layers.0.mlp.up_proj",      "mlp.up_proj");
      copy_real_conv("model.layers.0.mlp.down_proj",    "mlp.down_proj");
    } else {
      pushLPBQConv(params, "qkv.q_proj", hidden, q_out, G);
      pushLPBQConv(params, "qkv.k_proj", hidden, kv_out, G);
      pushLPBQConv(params, "qkv.v_proj", hidden, kv_out, G);
      pushLPBQConv(params, "o_proj.proj", q_out, hidden, G);
      pushLPBQConv(params, "mlp.gate_proj", hidden, intermediate, G);
      pushLPBQConv(params, "mlp.up_proj", hidden, intermediate, G);
      pushLPBQConv(params, "mlp.down_proj", intermediate, hidden, G);
    }
    pushQDQ(params, "qkv.qkv_input_qdq");
    pushQDQ(params, "qkv.q_output_qdq");
    pushQDQ(params, "qkv.k_output_qdq");
    pushQDQ(params, "qkv.v_output_qdq");
    pushQDQ(params, "o_proj.proj_input_qdq");
    pushQDQ(params, "o_proj.proj_output_qdq");
    pushQDQ(params, "mlp.up_proj_input_qdq");
    pushQDQ(params, "mlp.up_proj_output_qdq");
    pushQDQ(params, "mlp.gate_proj_output_qdq");
    pushQDQ(params, "mlp.sigmoid_output_qdq");
    pushQDQ(params, "mlp.act_output_qdq");
    pushQDQ(params, "mlp.down_proj_input_qdq");
    pushQDQ(params, "mlp.down_proj_output_qdq");
  } else if (mode == "split") {
    // pre_attn + post_attn two-graph bin — the all-NPU baseline split that
    // mirrors production split-prefill (GQA core runs separately in a
    // CPU-NPU pipeline; this bin captures everything ELSE on NPU).
    const int intermediate = 6144;
    mllm::models::qwen3::Qwen3Config full_cfg;
    full_cfg.hidden_size = hidden;
    full_cfg.intermediate_size = intermediate;
    full_cfg.num_attention_heads = n_q_heads;
    full_cfg.num_key_value_heads = n_kv_heads;
    full_cfg.head_dim = head_dim;
    full_cfg.rms_norm_eps = 1e-6;
    // PRE-ATTN params under "pre.*"
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.q_proj",       "pre.q_proj");
      copy_real_conv("model.layers.0.self_attn.k_proj",       "pre.k_proj");
      copy_real_conv("model.layers.0.self_attn.v_proj",       "pre.v_proj");
      copy_real_conv("model.layers.0.input_layernorm",        "pre.input_layernorm");
    } else {
      pushLPBQConv(params, "pre.q_proj", hidden, q_out, G);
      pushLPBQConv(params, "pre.k_proj", hidden, kv_out, G);
      pushLPBQConv(params, "pre.v_proj", hidden, kv_out, G);
    }
    for (auto* q : {"pre.in_input_qdq", "pre.qkv_input_qdq",
                    "pre.q_proj_out_qdq", "pre.k_proj_out_qdq", "pre.v_proj_out_qdq"}) pushQDQ(params, q);
    // POST-ATTN params under "post.*"
    if (use_real_params) {
      copy_real_conv("model.layers.0.self_attn.o_proj",          "post.o_proj");
      copy_real_conv("model.layers.0.post_attention_layernorm",  "post.post_attention_layernorm");
      copy_real_conv("model.layers.0.mlp.gate_proj",             "post.mlp.gate_proj");
      copy_real_conv("model.layers.0.mlp.up_proj",               "post.mlp.up_proj");
      copy_real_conv("model.layers.0.mlp.down_proj",             "post.mlp.down_proj");
    } else {
      pushLPBQConv(params, "post.o_proj", q_out, hidden, G);
      pushLPBQConv(params, "post.mlp.gate_proj", hidden, intermediate, G);
      pushLPBQConv(params, "post.mlp.up_proj",   hidden, intermediate, G);
      pushLPBQConv(params, "post.mlp.down_proj", intermediate, hidden, G);
    }
    for (auto* q : {"post.res_input_qdq", "post.o_input_qdq", "post.o_out_qdq",
                    "post.res1_qdq", "post.mlp_out_qdq", "post.res2_qdq",
                    "post.mlp.up_proj_input_qdq", "post.mlp.up_proj_output_qdq",
                    "post.mlp.gate_proj_output_qdq", "post.mlp.sigmoid_output_qdq",
                    "post.mlp.act_output_qdq", "post.mlp.down_proj_input_qdq",
                    "post.mlp.down_proj_output_qdq"}) pushQDQ(params, q);
  } else {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Unknown --mode: {} (expected q|kv|o|qkv|block|split)", mode);
    return -1;
  }

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_arg.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg_arg.get()));
  const std::string bin = "qwen3-attn-lpbq-" + mode + ".bin";

  auto compile = [&](mllm::nn::Module& m, int in_w) {
    auto x = mllm::Tensor::zeros({1, Seq, in_w}, mllm::kFloat16).setName("x0");
    m.load(params);
    auto ir = mllm::ir::trace_(m, {x});
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg_arg.get(), params));
    pm.run();
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print("LPBQ attn mode={} Seq={} -> {} (graph model.0.s{})", mode, Seq, bin, Seq);
  };

  if (mode == "q")   { sha::OneProjLPBQ m("model", hidden, q_out);  compile(m, hidden); }
  else if (mode == "kv") { sha::OneProjLPBQ m("model", hidden, kv_out); compile(m, hidden); }
  else if (mode == "o")  { sha::OneProjLPBQ m("model", q_out, hidden);  compile(m, q_out); }
  else if (mode == "qkv") { sha::QKVLPBQ m("model", hidden, q_out, kv_out); compile(m, hidden); }
  else if (mode == "block") {
    // Three independently-dispatchable graphs in one bin (qkv, o_proj, mlp).
    // Same multi-graph recipe as the MLP microbench's "gatedown" mode.
    const int intermediate = 6144;
    mllm::models::qwen3::Qwen3Config mlp_cfg;
    mlp_cfg.hidden_size = hidden;
    mlp_cfg.intermediate_size = intermediate;
    sha::QKVLPBQ qkv_m("qkv", hidden, q_out, kv_out);
    sha::OneProjLPBQ o_m("o_proj", q_out, hidden);
    sha::FullMLPLPBQBlock mlp_m("mlp", mlp_cfg);
    qkv_m.load(params);
    o_m.load(params);
    mlp_m.load(params);

    auto lower = [&](mllm::nn::Module& m, std::vector<mllm::Tensor> xs, const std::string& gname) {
      auto ir = mllm::ir::trace_(m, xs);
      mllm::ir::PassManager pm(ir);
      pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg_arg.get(), params));
      auto& gc = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
      gc["graph_on_qnn"] = nlohmann::json::array({gname});
      gc["chunk_graph_name"] = gname;
      pm.run();
    };
    lower(qkv_m, {mllm::Tensor::zeros({1, Seq, hidden}, mllm::kFloat16).setName("x0")}, "qkv");
    lower(o_m,   {mllm::Tensor::zeros({1, Seq, q_out},  mllm::kFloat16).setName("x1")}, "o_proj");
    lower(mlp_m, {mllm::Tensor::zeros({1, Seq, hidden}, mllm::kFloat16).setName("x2")}, "mlp");
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print("LPBQ block mode Seq={} -> {} (graphs qkv.0.s{}, o_proj.0.s{}, mlp.0.s{})",
                Seq, bin, Seq, Seq, Seq);
  }
  else if (mode == "split") {
    // Two-graph bin: pre_attn (in_norm + qkv + q/k_norm + rope_proxy) +
    // post_attn (o_proj + res1 + post_norm + MLP + res2). Mirrors production
    // split-prefill chunk_i = post(layer_{i-1}) + pre(layer_i) split. GQA
    // core is OUT OF SCOPE — runs on a separate CPU-NPU-pipelined path.
    // Serial dispatch (pre, then post) gives the all-NPU full-block time
    // minus GQA — the apples-to-apples baseline for mllm-het-block.
    mllm::models::qwen3::Qwen3Config full_cfg;
    full_cfg.hidden_size = hidden;
    full_cfg.intermediate_size = 6144;
    full_cfg.num_attention_heads = n_q_heads;
    full_cfg.num_key_value_heads = n_kv_heads;
    full_cfg.head_dim = head_dim;
    full_cfg.rms_norm_eps = 1e-6;
    sha::PreAttnLPBQNoGqa pre_m("pre", full_cfg);
    sha::PostAttnLPBQNoGqa post_m("post", full_cfg);
    pre_m.load(params);
    post_m.load(params);

    auto lower = [&](mllm::nn::Module& m, std::vector<mllm::Tensor> xs, const std::string& gname) {
      auto ir = mllm::ir::trace_(m, xs);
      mllm::ir::PassManager pm(ir);
      pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg_arg.get(), params));
      auto& gc = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
      gc["graph_on_qnn"] = nlohmann::json::array({gname});
      gc["chunk_graph_name"] = gname;
      pm.run();
    };
    lower(pre_m,  {mllm::Tensor::zeros({1, Seq, hidden}, mllm::kFloat16).setName("x0")}, "pre");
    lower(post_m, {mllm::Tensor::zeros({1, Seq, hidden}, mllm::kFloat16).setName("residual"),
                   mllm::Tensor::zeros({1, Seq, q_out},  mllm::kFloat16).setName("attn_out")}, "post");
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print("LPBQ split mode Seq={} -> {} (graphs pre.0.s{}, post.0.s{})",
                Seq, bin, Seq, Seq);
  }
});
