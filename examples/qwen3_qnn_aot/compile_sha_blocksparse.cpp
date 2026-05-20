// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile the block-sparse SHA Qwen3 model defined in
// modeling_qwen_qnn_aot_sha_blocksparse.hpp into a QNN-AOT context.
//
// Differences vs compile_sha.cpp:
//   * Adds two extra inputs per layer for the trace — `k_arranged_i` and
//     `v_arranged_i`, fp16, shape [Hq, num_q_blocks, kTopKBK, head_dim].
//   * Compiles only the prefill graph (chunk_size=128). The decode (Sq=1)
//     graph reuses the regular dense SHA model — block-sparse offers no
//     benefit at Sq=1.
//
// Usage:
//   ./compile_sha_blocksparse -m /path/to/model.mllm -c /path/to/config.json -aot_cfg /path/to/qnn_aot_cfg.json

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha_blocksparse.hpp"
#include "modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;
namespace bs = mllm::models::qwen3::sha_blocksparse;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) { return std::string(qairt_root) + "/lib/x86_64-linux-clang/"; }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path (LPBQ-quantized .mllm).");
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
  const int N = 128;                     // chunk_size for prefill
  const int CL = 1024;                   // max KV-cache length
  const int BK = bs::kBK;                // 32
  const int top_k_BK = bs::kTopKBK;      // 256
  const int num_q_blocks = N / bs::kBQ;  // 4

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());

  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  mllm::print("Preparing SHA parameters (slicing MHA weights)...");
  mllm::models::qwen3::sha::prepareParametersForSHA(params, model_cfg);
  mllm::print("SHA parameters prepared.");

  bs::Qwen3ForCausalLM_SHABlockSparse model(model_cfg, /*chunk_size=*/N);
  model.load(params);

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  // ---------------------------------------------------------------------------
  // Build trace inputs.
  // ---------------------------------------------------------------------------
  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["sequence"] = mllm::Tensor::zeros({1, N}, mllm::kInt32);

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    auto kar_name = "k_arranged_" + std::to_string(i);
    auto var_name = "v_arranged_" + std::to_string(i);

    // K_arranged / V_arranged are fp16, raw spec — no scale/zp attachments.
    trace_inputs[kar_name] =
        mllm::Tensor::empty({model_cfg.num_attention_heads, num_q_blocks, top_k_BK, model_cfg.head_dim}, mllm::kFloat16);
    trace_inputs[var_name] =
        mllm::Tensor::empty({model_cfg.num_attention_heads, num_q_blocks, top_k_BK, model_cfg.head_dim}, mllm::kFloat16);
  }

  mllm::print("Tracing block-sparse SHA model (chunk_size={}, top_k={}, BK={})...", N, bs::kTopK, BK);
  auto ir = model.trace(trace_inputs, {});
  mllm::print("Trace complete.");

  mllm::redirect("qwen3_qnn_aot_sha_blocksparse_128_pre.mir", [&]() { mllm::print(ir["model"]); });

  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
  pm.run();

  mllm::redirect("qwen3_qnn_aot_sha_blocksparse_128.mir", [&]() { mllm::print(ir["model"]); });

  // ---------------------------------------------------------------------------
  // Compile the dense Sq=1 decode graph in the same context.
  // Uses the regular SHA model (block-sparse adds no value at Sq=1).
  // ---------------------------------------------------------------------------
  mllm::print("Compiling dense Sq=1 decode graph...");

  // QDQ params for causal_mask and constant_zero (consumed by dense path).
  params->push("causal_mask.scale", mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
  params->push("causal_mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));

  mllm::models::qwen3::sha::Qwen3ForCausalLM_SHA dense_model(model_cfg);
  dense_model.load(params);

  {
    const int N1 = 1;
    auto sequence = mllm::Tensor::zeros({1, N1}, mllm::kInt32);
    auto causal_mask = mllm::Tensor::zeros({1, 1, N1, CL}, mllm::kUInt16);
    causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
    causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
    causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);

    std::unordered_map<std::string, mllm::Tensor> dense_inputs;
    dense_inputs["sequence"] = sequence;
    dense_inputs["causal_mask"] = causal_mask;

    for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
      auto pk = "past_key_" + std::to_string(i);
      auto pv = "past_value_" + std::to_string(i);
      dense_inputs[pk] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, model_cfg.head_dim, CL - N1}, mllm::kUInt8PerTensorSym);
      dense_inputs[pv] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N1, model_cfg.head_dim}, mllm::kUInt8PerTensorSym);
      dense_inputs[pk].attach(
          "scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(),
          true);
      dense_inputs[pk].attach(
          "zero_point",
          params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point").impl(),
          true);
      dense_inputs[pv].attach(
          "scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(),
          true);
      dense_inputs[pv].attach(
          "zero_point",
          params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point").impl(),
          true);
    }

    mllm::print("Tracing dense SHA model (Sq=1)...");
    auto ir1 = dense_model.trace(dense_inputs, {});
    mllm::print("Trace complete.");

    mllm::ir::PassManager pm1(ir1["model"]);
    pm1.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm1.run();

    mllm::redirect("qwen3_qnn_aot_sha_blocksparse_dense_1.mir", [&]() { mllm::print(ir1["model"]); });
  }

  qnn_aot_env.saveContext("context.0", "qwen3-1.7B-lpbq-sha-blocksparse.bin");

  mllm::print("Block-sparse + dense compilation complete.");
  mllm::print("Output files:");
  mllm::print("  - qwen3_qnn_aot_sha_blocksparse_128.mir (block-sparse prefill)");
  mllm::print("  - qwen3_qnn_aot_sha_blocksparse_dense_1.mir (dense decode)");
  mllm::print("  - qwen3-1.7B-lpbq-sha-blocksparse.bin");
});
