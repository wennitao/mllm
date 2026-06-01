// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Standalone fake-scale w4a16-LPBQ MLP microbench compiler. Traces the existing
// (production-correct) Qwen3MLP module — gate/up/down as LPBQ-w4a16 Conv2D with
// the surrounding uint16 QDQ scaffolding — through the real AOT lowering pipeline,
// but feeds it SYNTHETIC weights and quantization scales (no PTQ checkpoint).
//
// Purpose: get a faithful w4a16-LPBQ conv onto the device to measure its latency
// and tiling behaviour (the precondition for the swapMN-orientation question:
// is the quantized conv path free-dimension/VTCM-tiling sensitive, or — like the
// uint16 matmul proxy — orientation-insensitive?). Scale VALUES are arbitrary;
// only the shapes/encoding need to satisfy the compiler and QNN finalize.
//
// Usage:
//   ./compile_mlp_lpbq_microbench -aot_cfg qnn_aot_cfg_mlp_lpbq_microbench.json
//
// Full Qwen3MLP: gate/up (2048->6144) + SiLU + mul + down (6144->2048), all LPBQ.
// --sq sets the compiled M (slice the chunk to study weight-re-read cost).

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/passes/AOTCompileContext.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"  // brings Qwen3MLP and the ptq QDQ scaffolding

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) { return std::string(qairt_root) + "/lib/x86_64-linux-clang/"; }
  return "/opt/qcom/aistack/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}

// Push a synthetic LPBQ-w4a16 conv weight + its two-level scales. PTQPass pulls
// {prefix}.weight (int4-in-int8, values 0..15), {prefix}.scale1 (uint4-in-uint8,
// the level-0 per-block scales, values 0..16) and {prefix}.scale2 (level-1
// per-output-channel fp32). Filter layout is [1,1,In,Out]; blocks tile the In
// (contraction) axis with block_size G. First-cut shapes — the host compiler is
// the authority on the exact scale1 layout.
void pushLPBQConv(const mllm::ParameterFile::ptr_t& params, const std::string& prefix, int In, int Out, int G) {
  const int n_blk = In / G;  // blocks per output channel along the contraction axis

  std::vector<int8_t> w((size_t)In * Out, /*int4 mid*/ 8);    // [1,1,In,Out]
  std::vector<uint8_t> s1((size_t)Out * n_blk, /*uint4 mid*/ 8);  // level-0 block scales, [Out, In/G]
  std::vector<float> s2((size_t)Out, 0.01f);                  // level-1 per-channel fp32 (positive)

  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(prefix + ".weight", Tensor::fromVector(w, {1, 1, In, Out}, mllm::kInt8));
  // Match production: pymllm qlinear.py flattens scale1/scale2 before saving
  // (rank-1). setupComplexTensorQuantization asserts both are rank 1.
  push(prefix + ".scale1", Tensor::fromVector(s1, {(int)s1.size()}, mllm::kUInt8));
  push(prefix + ".scale2", Tensor::fromVector(s2, {(int)s2.size()}, mllm::kFloat32));
}

// Push a per-tensor uint16-asym QDQ node's synthetic scale/zero_point. QDQ pulls
// {qdq}.fake_quant.scale (fp32, rank-1, one element) and .zero_point (int32 rank-1).
void pushQDQ(const mllm::ParameterFile::ptr_t& params, const std::string& qdq) {
  auto push = [&](const std::string& key, Tensor t) {
    params->push(key, t.contiguous().setMemType(mllm::kParamsNormal).setName(key));
  };
  push(qdq + ".fake_quant.scale", Tensor::fromVector(std::vector<float>{1.0f / 256.0f}, {1}, mllm::kFloat32));
  push(qdq + ".fake_quant.zero_point", Tensor::fromVector(std::vector<int32_t>{0}, {1}, mllm::kInt32));
}
}  // namespace

// Full Qwen3 MLP (gate/up/silu/mul/down, all LPBQ) + a closing QDQ so the down
// output has a solved scale standalone (production relies on the next layer for it).
namespace mllm::models::qwen3::sha {
class FullMLPLPBQ final : public nn::Module {
  Qwen3MLP mlp_;
  int hidden_size_;
 public:
  FullMLPLPBQ() = default;
  FullMLPLPBQ(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    mlp_ = reg<Qwen3MLP>("", cfg);  // convs stay at model.gate_proj etc.
    hidden_size_ = cfg.hidden_size;
  }
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = mlp_.forward(inputs, args)[0];
    return {ptq::QDQ(this, o, "down_proj_output_qdq")};
  }
};
// tiled: full Qwen3MLP but MLP run as 2 token-halves (M/2) concatenated — one
// graph/dispatch, but each gate/up/down panel fits VTCM (≤8 MB). Tests whether
// the 2×512 MLP win survives WITHOUT extra dispatches (attn would stay M=Sq).
class TiledMLPLPBQ final : public nn::Module {
  Qwen3MLP mlp0_, mlp1_;  // two halves, distinct QDQ keys (model.0.*, model.1.*)
 public:
  TiledMLPLPBQ() = default;
  TiledMLPLPBQ(const std::string& n, const Qwen3Config& c): nn::Module(n){ mlp0_=reg<Qwen3MLP>("0",c); mlp1_=reg<Qwen3MLP>("1",c); }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>& a) override {
    auto o0=ptq::QDQ(this,mlp0_.forward({in[0]},a)[0],"down_proj_output_qdq");
    auto o1=ptq::QDQ(this,mlp1_.forward({in[1]},a)[0],"down_proj_output_qdq");
    return {o0, o1}; }  // 2 halves, 1 dispatch
};
// gateup: gate/up convs + SiLU, output silu(gate) and up — CPU does the final mul.
class GateUpLPBQ final : public nn::Module {
  nn::Conv2D gate_proj_, up_proj_; nn::SiLU silu_; int hs_, is_;
 public:
  GateUpLPBQ() = default;
  GateUpLPBQ(const std::string& n, const Qwen3Config& c) : nn::Module(n) {
    gate_proj_=reg<nn::Conv2D>("gate_proj",c.hidden_size,c.intermediate_size,CONV2D_PROPERTY);
    silu_=reg<nn::SiLU>("act"); up_proj_=reg<nn::Conv2D>("up_proj",c.hidden_size,c.intermediate_size,CONV2D_PROPERTY);
    hs_=c.hidden_size; is_=c.intermediate_size; }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    auto x=ptq::QDQ(this,in[0],"up_proj_input_qdq").view({1,1,-1,hs_},true);
    auto u=ptq::QDQ(this,up_proj_(x),"up_proj_output_qdq").view({1,-1,is_},true);
    auto g=ptq::QDQ(this,gate_proj_(x),"gate_proj_output_qdq").view({1,-1,is_},true);
    auto sg=ptq::QDQ(this,g*ptq::QDQ(this,nn::functional::sigmoid(g),"sigmoid_output_qdq"),"act_output_qdq");
    return {sg,u}; }
};
// gateraw: gate/up convs ONLY (no SiLU), output raw gate + up — the CPU does the
// whole silu(gate)*up. Tests offloading the full activation (sigmoid + both muls)
// to CPU. Same 1-in/2-out [.,.,I] I/O as gateup, so the bench runs it with --mode gateup.
class GateUpRawLPBQ final : public nn::Module {
  nn::Conv2D gate_proj_, up_proj_; int hs_, is_;
 public:
  GateUpRawLPBQ() = default;
  GateUpRawLPBQ(const std::string& n, const Qwen3Config& c) : nn::Module(n) {
    gate_proj_=reg<nn::Conv2D>("gate_proj",c.hidden_size,c.intermediate_size,CONV2D_PROPERTY);
    up_proj_=reg<nn::Conv2D>("up_proj",c.hidden_size,c.intermediate_size,CONV2D_PROPERTY);
    hs_=c.hidden_size; is_=c.intermediate_size; }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    auto x=ptq::QDQ(this,in[0],"up_proj_input_qdq").view({1,1,-1,hs_},true);
    auto u=ptq::QDQ(this,up_proj_(x),"up_proj_output_qdq").view({1,-1,is_},true);
    auto g=ptq::QDQ(this,gate_proj_(x),"gate_proj_output_qdq").view({1,-1,is_},true);
    return {g,u}; }
};
// down: just the down conv, input is the CPU mul result.
class DownLPBQ final : public nn::Module {
  nn::Conv2D down_proj_; int hs_, is_;
 public:
  DownLPBQ() = default;
  DownLPBQ(const std::string& n, const Qwen3Config& c): nn::Module(n){
    down_proj_=reg<nn::Conv2D>("down_proj",c.intermediate_size,c.hidden_size,CONV2D_PROPERTY); hs_=c.hidden_size; is_=c.intermediate_size; }
  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    auto o=ptq::QDQ(this,in[0],"down_proj_input_qdq").view({1,1,-1,is_},true);
    // Closing QDQ so the down output's scale is solved standalone — mirrors
    // FullMLPLPBQ; without it PTQPass::recursiveCheckUnsolved aborts.
    return {ptq::QDQ(this,down_proj_(o).view({1,-1,hs_},true),"down_proj_output_qdq")}; }
};
// (gatedown packs gateraw + down as two separate graphs in one bin — built by two
// independent traces in the compile main below, not a combined module.)
}  // namespace mllm::models::qwen3::sha

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath()).help("QNN AOT env path.");
  auto& sq_arg = Argparse::add<int>("--sq").help("Sequence (M) the MLP graph is compiled for — slice M to study re-read cost").def(1024);
  auto& params_arg = Argparse::add<std::string>("--params").help("Real ptq_lpbq .mllm — copy layer-0 MLP scales (else synthetic)").def("");
  auto& mode_arg = Argparse::add<std::string>("--mode").help("full | gateup | down (split for CPU-offload bench)").def("full");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!qnn_aot_cfg_files.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config provided"); return -1; }

  // Full Qwen3 MLP block; Seq = the prefill chunk (sliceable via --sq).
  const int hidden = 2048, intermediate = 6144, G = 16;
  const int Seq = sq_arg.get();
  mllm::models::qwen3::Qwen3Config cfg;  // defaults; override the two dims we need.
  cfg.hidden_size = hidden;
  cfg.intermediate_size = intermediate;

  // In-memory synthetic parameter file — no checkpoint. Full gate/up/down LPBQ
  // convs + the six uint16 QDQ boundaries the production Qwen3MLP::forward emits.
  auto params = mllm::ParameterFile::create(mllm::ModelFileVersion::kV2);
  if (params_arg.get().empty()) {
    pushLPBQConv(params, "model.gate_proj", hidden, intermediate, G);
    pushLPBQConv(params, "model.up_proj", hidden, intermediate, G);
    pushLPBQConv(params, "model.down_proj", intermediate, hidden, G);
    pushQDQ(params, "model.up_proj_input_qdq");
    pushQDQ(params, "model.up_proj_output_qdq");
    pushQDQ(params, "model.gate_proj_output_qdq");
    pushQDQ(params, "model.sigmoid_output_qdq");
    pushQDQ(params, "model.act_output_qdq");
    pushQDQ(params, "model.down_proj_input_qdq");
  } else {
    // Copy layer-0 MLP weights + calibrated scales, remapping the prefix:
    // model.layers.0.mlp.<x> -> model.<x>, so the synthetic module names match.
    auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    const std::string src = "model.layers.0.mlp.";
    // tiled: two halves are submodules 0/1. gatedown: two SEPARATE graphs traced
    // independently from modules "gr" (gateraw) and "dn" (down), so copy under those
    // prefixes. Over-copy (gr.down_proj, dn.gate_proj) is orphaned and harmless.
    std::vector<std::string> pre;
    if (mode_arg.get() == "tiled") pre = {"model.0.", "model.1."};
    else if (mode_arg.get() == "gatedown") pre = {"gr.", "dn."};
    else pre = {"model."};
    int n = 0;
    for (auto& p : pre) {  // fresh load per prefix → distinct tensor objects, no rename collisions
      auto r = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, false);
      for (auto& [k, t] : r->dict()) { if (k.rfind(src,0)!=0) continue; std::string dst=p+k.substr(src.size()); params->push(dst, t.setName(dst)); ++n; }
    }
    mllm::print("Copied {} layer-0 MLP params from {}", n, params_arg.get());
  }

  // Closing down QDQ is synthetic (production relies on the next layer). Push it under
  // the prefix the down conv lives at: dn.* for gatedown (down module "dn"), else model.*.
  pushQDQ(params, mode_arg.get() == "gatedown" ? "dn.down_proj_output_qdq" : "model.down_proj_output_qdq");
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));
  const std::string mode = mode_arg.get(), bin = "qwen3-mlp-lpbq-" + mode + ".bin";
  auto compileXs = [&](mllm::nn::Module& m, std::vector<mllm::Tensor> xs) {
    m.load(params);
    auto ir = mllm::ir::trace_(m, xs);
    mllm::ir::PassManager pm(ir);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print("LPBQ MLP mode={} Seq={} -> {}", mode, Seq, bin);
  };
  auto compile = [&](mllm::nn::Module& m, int in_w, int nin = 1) {
    std::vector<mllm::Tensor> xs;
    for (int i = 0; i < nin; ++i) xs.push_back(mllm::Tensor::zeros({1, Seq / nin, in_w}, mllm::kFloat16).setName("x" + std::to_string(i)));
    compileXs(m, xs);
  };
  if (mode == "gateup") { sha::GateUpLPBQ m("model", cfg); compile(m, hidden); }
  else if (mode == "gateraw") { sha::GateUpRawLPBQ m("model", cfg); compile(m, hidden); }
  else if (mode == "down") { sha::DownLPBQ m("model", cfg); compile(m, intermediate); }
  else if (mode == "tiled") { sha::TiledMLPLPBQ m("model", cfg); compile(m, hidden, 2); }
  else if (mode == "gatedown") {
    // TWO independently-dispatchable graphs in ONE bin: gateraw (module "gr",
    // graph gr.0.s<Seq>, 1in[H]/2out[I]) and down (module "dn", graph dn.0.s<Seq>,
    // 1in[I]/1out[H]). Each is traced + lowered separately into the same env, then
    // one saveContext emits both — the split-prefill multi-graph pattern. The runtime
    // dispatches gr → (GPU sigmoid + CPU muls) → dn, with the handoff between graphs.
    sha::GateUpRawLPBQ gr("gr", cfg);
    sha::DownLPBQ dn("dn", cfg);
    gr.load(params);
    dn.load(params);
    // Each trace's top graph is named after its module ("gr"/"dn"); the QNN passes
    // lower whatever graph_on_qnn names, so override it per trace (the split-prefill
    // multi-graph recipe — see compile_sha_blocksparse_causal_split.cpp).
    auto lower = [&](mllm::nn::Module& m, std::vector<mllm::Tensor> xs, const std::string& gname) {
      auto ir = mllm::ir::trace_(m, xs);
      mllm::ir::PassManager pm(ir);
      pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
      auto& cfg = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
      cfg["graph_on_qnn"] = nlohmann::json::array({gname});
      cfg["chunk_graph_name"] = gname;  // tell Split/Merge/Lowering passes to use this graph, not "model"
      pm.run();
    };
    lower(gr, {mllm::Tensor::zeros({1, Seq, hidden}, mllm::kFloat16).setName("x0")}, "gr");
    lower(dn, {mllm::Tensor::zeros({1, Seq, intermediate}, mllm::kFloat16).setName("x1")}, "dn");
    qnn_aot_env.saveContext("context.0", bin);
    mllm::print("LPBQ MLP mode=gatedown Seq={} -> {} (graphs gr.0.s{}, dn.0.s{})", Seq, bin, Seq, Seq);
  } else { sha::FullMLPLPBQ m("model", cfg); compile(m, hidden); }
});
