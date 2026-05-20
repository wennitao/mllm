// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile the fp16 block-sparse Qwen3 model defined in
// modeling_qwen_qnn_aot_fp16_blocksparse.hpp into a QNN-AOT context.
//
// Differences vs compile_sha.cpp:
//   * fp16 throughout — no causal_mask / KV-quant scale/zp attachments.
//   * Two extra inputs per layer: K_arranged and V_arranged
//     (shape [Hq, num_q_blocks, top_k·BK, head_dim]).
//   * Compiles only the chunk_size=128 prefill graph (decode graph would be
//     equivalent to the regular dense fp16 model — left as future work).
//
// Usage:
//   ./compile_fp16_blocksparse -m /path/to/model.mllm -c /path/to/config.json -aot_cfg /path/to/qnn_aot_cfg.json

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

#include "modeling_qwen_qnn_aot_fp16_blocksparse.hpp"

using mllm::Argparse;
namespace bs = mllm::models::qwen3::sha_fp16_blocksparse;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) { return std::string(qairt_root) + "/lib/x86_64-linux-clang/"; }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

// Bake the LLaMA-style rotary sin/cos LUTs and inject them into `params` as
// model.mllm_max_{sin,cos}_embedding so the compile graph's gather node can
// resolve them. Shape: [1, max_positions, head_dim] (fp16). Layout matches
// HuggingFace Qwen3 rotary: emb = concat(theta_p, theta_p) where theta_p[k] =
// p / rope_theta^(2k/head_dim), k=0..head_dim/2-1.
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
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path (fp16 weights).");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def(defaultQnnEnvPath())
                           .help("QNN AOT Environment path.");

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

  // Compile-time shape constants. Match the constants in the model header.
  const int N = 128;                 // chunk_size for prefill (must be ≥ kBQ and a multiple of kBQ)
  const int CL = 1024;               // max KV-cache length
  const int BK = bs::kBK;            // 32
  const int top_k_BK = bs::kTopKBK;  // 256
  const int num_q_blocks = N / bs::kBQ;  // 4

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());

  // Load original (MHA) parameters and slice per-head for SHA layout.
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::print("Slicing MHA weights into SHA layout (fp16)...");
  bs::prepareParametersForSHA_FP16(params, model_cfg);
  mllm::print("Baking rotary sin/cos LUTs (max_pos={}, head_dim={})...", model_cfg.max_cache_length, model_cfg.head_dim);
  bakeRotaryEmbeddings(params, model_cfg);
  mllm::print("Done.");

  // Build the model and load its parameters.
  bs::Qwen3ForCausalLM_BlockSparse model(model_cfg, /*chunk_size=*/N);
  model.load(params);

  // QNN AOT environment.
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  // ---------------------------------------------------------------------------
  // Trace the prefill graph (chunk_size=128).
  // Inputs: sequence, position_ids, past_key_i × L, past_value_i × L,
  //         k_arranged_i × L, v_arranged_i × L.
  // ---------------------------------------------------------------------------
  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["sequence"] = mllm::Tensor::zeros({1, N}, mllm::kInt32);
  trace_inputs["position_ids"] = mllm::Tensor::zeros({1, N}, mllm::kInt32);

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    trace_inputs["past_key_" + std::to_string(i)] =
        mllm::Tensor::empty({1, model_cfg.num_key_value_heads, model_cfg.head_dim, CL - N}, mllm::kFloat16);
    trace_inputs["past_value_" + std::to_string(i)] =
        mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim}, mllm::kFloat16);
    trace_inputs["k_arranged_" + std::to_string(i)] =
        mllm::Tensor::empty({model_cfg.num_attention_heads, num_q_blocks, top_k_BK, model_cfg.head_dim}, mllm::kFloat16);
    trace_inputs["v_arranged_" + std::to_string(i)] =
        mllm::Tensor::empty({model_cfg.num_attention_heads, num_q_blocks, top_k_BK, model_cfg.head_dim}, mllm::kFloat16);
  }

  mllm::print("Tracing fp16 block-sparse model (chunk_size={}, top_k={}, BK={})...", N, bs::kTopK, BK);
  auto ir = model.trace(trace_inputs, {});
  mllm::print("Trace complete.");

  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
  pm.run();

  mllm::redirect("qwen3_qnn_aot_fp16_blocksparse_128.mir", [&]() { mllm::print(ir["model"]); });
  qnn_aot_env.saveContext("context.0", "qwen3-fp16-blocksparse.bin");

  mllm::print("FP16 block-sparse compilation complete.");
  mllm::print("Output files:");
  mllm::print("  - qwen3_qnn_aot_fp16_blocksparse_128.mir");
  mllm::print("  - qwen3-fp16-blocksparse.bin");
});
