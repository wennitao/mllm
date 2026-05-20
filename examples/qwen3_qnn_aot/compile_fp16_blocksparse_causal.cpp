// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile the fp16 PER-QB causal block-sparse Qwen3 model defined in
// modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp into a QNN-AOT context.
//
// Per-qb dispatch: the compiled graph processes BQ=32 tokens per call. The
// runner re-binds Q/K/V/padding_mask and calls graphExecute num_qb times to
// cover a prefill chunk of N tokens (N = num_qb * BQ).
//
// Per-layer inputs:
//   past_key_i      [1, num_kv_heads, head_dim, CL - BQ]
//   past_value_i    [1, num_kv_heads, CL - BQ, head_dim]
//   k_arranged_i    [num_attention_heads, 1, (top_k-1)*BK, head_dim]   (HISTORICAL slots only)
//   v_arranged_i    [num_attention_heads, 1, (top_k-1)*BK, head_dim]
// Plus a single shared input:
//   padding_mask    [1, top_k*BK]   (qb-dependent, APP_WRITE per dispatch)
//
// The diagonal slot's K/V is concat'd onto K_arranged inside the layer
// (computed from this dispatch's own hidden_states), so the runner only
// gathers the top_k-1 historical slots.
//
// Usage:
//   ./compile_fp16_blocksparse_causal -m /path/to/model.mllm \
//                                     -c /path/to/config.json \
//                                     -aot_cfg /path/to/qnn_aot_cfg.json

#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp"

using mllm::Argparse;
namespace bsc = mllm::models::qwen3::sha_fp16_blocksparse_causal;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) {
    return std::string(qairt_root) + "/lib/x86_64-linux-clang/";
  }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

// Same as compile_fp16_blocksparse — bake the rotary sin/cos LUTs.
void bakeRotaryEmbeddings(const mllm::ParameterFile::ptr_t& params, const mllm::models::qwen3::Qwen3Config& cfg) {
  const int head_dim = cfg.head_dim;
  const int max_pos = cfg.max_cache_length;
  const float theta = cfg.rope_theta;
  const int half = head_dim / 2;

  std::vector<float> inv_freq(half);
  for (int k = 0; k < half; ++k) {
    inv_freq[k] = 1.0f / std::pow(theta, (float)(2 * k) / (float)head_dim);
  }

  std::vector<float> sin_buf((size_t)max_pos * head_dim);
  std::vector<float> cos_buf((size_t)max_pos * head_dim);
  for (int p = 0; p < max_pos; ++p) {
    for (int k = 0; k < half; ++k) {
      float ang = (float)p * inv_freq[k];
      float s = std::sin(ang);
      float c = std::cos(ang);
      sin_buf[(size_t)p * head_dim + k] = s;
      sin_buf[(size_t)p * head_dim + k + half] = s;
      cos_buf[(size_t)p * head_dim + k] = c;
      cos_buf[(size_t)p * head_dim + k + half] = c;
    }
  }

  auto sin_t = mllm::Tensor::fromVector(sin_buf, {1, max_pos, head_dim}, mllm::kFloat32).to(mllm::kFloat16);
  auto cos_t = mllm::Tensor::fromVector(cos_buf, {1, max_pos, head_dim}, mllm::kFloat32).to(mllm::kFloat16);

  params->push("model.mllm_max_sin_embedding",
               sin_t.contiguous().setMemType(mllm::kParamsNormal).setName("model.mllm_max_sin_embedding"));
  params->push("model.mllm_max_cos_embedding",
               cos_t.contiguous().setMemType(mllm::kParamsNormal).setName("model.mllm_max_cos_embedding"));
}
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Path to .mllm params").required(true);
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Path to model config json").required(true);
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--qnn_aot_cfg").help("Path to QNN AOT config json");
  auto& qnn_env_path = Argparse::add<std::string>("-e|--qnn_env_path").help("QNN env / driver path").def(defaultQnnEnvPath());

  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  // Compile-time shape constants. Per-qb dispatch: graph processes one qb
  // (= BQ rows) per graphExecute call.
  const int BQ = bsc::kBQ;           // 32
  const int BK = bsc::kBK;           // 32
  const int top_k = bsc::kTopK;      // 8
  const int top_k_BK = bsc::kTopKBK; // 256 — full attention K width (after in-layer concat)
  const int hist_k_BK = bsc::kHistKBK;  // 224 — historical K width supplied by runner

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());
  const int CL = model_cfg.max_cache_length;  // max KV-cache length, driven by config

  // Load original (MHA) parameters, slice per-head for SHA layout.
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::print("Slicing MHA weights into SHA layout (fp16)...");
  bsc::prepareParametersForSHA_FP16(params, model_cfg);
  mllm::print("Baking rotary sin/cos LUTs (max_pos={}, head_dim={})...", model_cfg.max_cache_length, model_cfg.head_dim);
  bakeRotaryEmbeddings(params, model_cfg);
  mllm::print("Baking static causal triangle mask (BQ={}, top_k_BK={})...", BQ, top_k_BK);
  bsc::bakeCausalTriangleMask(params);
  mllm::print("Done.");

  // Build the model and load its parameters.
  bsc::Qwen3ForCausalLM_CausalBlockSparse model(model_cfg);
  model.load(params);

  // QNN AOT environment.
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  // -------------------------------------------------------------------------
  // Trace the per-qb graph. Per dispatch: one BQ-row chunk.
  // Inputs:
  //   sequence            [1, BQ]
  //   position_ids        [1, BQ]
  //   past_key_i × L      [1, Hkv, head_dim, CL - BQ]
  //   past_value_i × L    [1, Hkv, CL - BQ, head_dim]
  //   k_arranged_i × L    [Hq, top_k*BK, head_dim]
  //   v_arranged_i × L    [Hq, top_k*BK, head_dim]
  //   padding_mask        [1, top_k*BK]
  // -------------------------------------------------------------------------
  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["sequence"]     = mllm::Tensor::zeros({1, BQ}, mllm::kInt32);
  trace_inputs["position_ids"] = mllm::Tensor::zeros({1, BQ}, mllm::kInt32);

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    trace_inputs["past_key_" + std::to_string(i)] =
        mllm::Tensor::empty({1, model_cfg.num_key_value_heads, model_cfg.head_dim, CL - BQ}, mllm::kFloat16);
    trace_inputs["past_value_" + std::to_string(i)] =
        mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - BQ, model_cfg.head_dim}, mllm::kFloat16);
    trace_inputs["k_arranged_" + std::to_string(i)] =
        mllm::Tensor::empty({model_cfg.num_attention_heads, 1, hist_k_BK, model_cfg.head_dim}, mllm::kFloat16);
    trace_inputs["v_arranged_" + std::to_string(i)] =
        mllm::Tensor::empty({model_cfg.num_attention_heads, 1, hist_k_BK, model_cfg.head_dim}, mllm::kFloat16);
  }
  trace_inputs["padding_mask"] = mllm::Tensor::empty({1, top_k_BK}, mllm::kFloat16);

  mllm::print("Tracing fp16 per-qb causal block-sparse model (BQ={}, top_k={}, BK={}, top_k_BK={}, hist_k_BK={})...", BQ,
              top_k, BK, top_k_BK, hist_k_BK);
  auto ir = model.trace(trace_inputs, {});
  mllm::print("Trace complete.");

  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
  pm.run();

  mllm::redirect("qwen3_qnn_aot_fp16_blocksparse_causal_BQ32.mir", [&]() { mllm::print(ir["model"]); });
  qnn_aot_env.saveContext("context.0", "qwen3-fp16-blocksparse-causal.bin");

  mllm::print("FP16 per-qb causal block-sparse compilation complete.");
  mllm::print("Output files:");
  mllm::print("  - qwen3_qnn_aot_fp16_blocksparse_causal_BQ32.mir");
  mllm::print("  - qwen3-fp16-blocksparse-causal.bin");
});
