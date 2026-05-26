// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Benefits:
// 1. Reduces QNN AOT compilation time
// 2. Improves HTP runtime performance
// 3. Enables better memory locality per head
//
// Usage:
//   ./compile_sha -m /path/to/model.mllm -c /path/to/config.json -aot_cfg /path/to/qnn_aot_cfg.json

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;

namespace {

std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) { return std::string(qairt_root) + "/lib/x86_64-linux-clang/"; }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def(defaultQnnEnvPath())
                           .help("QNN AOT Environment path.");
  auto& ar_len_arg     = Argparse::add<int>("--ar_len").def(32).help("Prefill chunk size (N) baked into the graph.");
  auto& context_len_arg = Argparse::add<int>("--context_len").def(1024).help("Max KV cache length (CL) baked into the graph.");
  auto& output_path = Argparse::add<std::string>("-o|--output")
                          .def("qwen3-1.7B-lpbq-sha.bin")
                          .help("Output .bin path for the compiled QNN context.");

  Argparse::parse(argc, argv);

  int N  = ar_len_arg.get();
  int CL = context_len_arg.get();

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());

  // Load original parameters
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);

  // ============================================================================
  // Key Step: Prepare SHA parameters by slicing MHA weights
  // ============================================================================
  // This is the critical step that transforms MHA weights into SHA weights.
  // For each Q/K/V projection, we slice the weight matrix into per-head pieces.
  //
  // Original:  q_proj.weight [num_heads * head_dim, hidden_size, 1, 1]
  // SHA:       q_proj.{h}.weight [head_dim, hidden_size, 1, 1] for each head h
  //
  mllm::print("Preparing SHA parameters (slicing MHA weights)...");
  mllm::models::qwen3::sha::prepareParametersForSHA(params, model_cfg);
  mllm::print("SHA parameters prepared.");

  // Create SHA model
  auto model = mllm::models::qwen3::sha::Qwen3ForCausalLM_SHA(model_cfg);

  // Add params for causal mask
  {
    params->push("causal_mask.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
    params->push("causal_mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
    params->push("constant_zero.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
    params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  }
  model.load(params);

  // Create Qnn AOT Model
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  // Model length 32.

  {
    // Sequence: [B, N]
    // past_key_i: [B, H, D, CL-N] for each layer i
    // past_value_i: [B, H, CL-N, D] for each layer i
    // causal_mask: [B, 1, N, CL]
    auto sequence = mllm::Tensor::zeros({1, N}, mllm::kInt32);
    auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16);

    // NOTE: force set causal mask to UInt16Asy
    // NOTE: Attach scale and zero point to causal mask
    {
      causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
      causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
      causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
    }

    // Create KV cache inputs for all layers
    std::unordered_map<std::string, mllm::Tensor> trace_inputs;
    trace_inputs["sequence"] = sequence;
    trace_inputs["causal_mask"] = causal_mask;

    for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      auto past_value_name = "past_value_" + std::to_string(i);

      // clang-format off
    trace_inputs[past_key_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
        CL - N,
    }, mllm::kUInt8PerTensorSym);
    trace_inputs[past_value_name] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim}, mllm::kUInt8PerTensorSym);
    
    trace_inputs[past_key_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_key_name].attach("zero_point", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);

    trace_inputs[past_value_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_value_name].attach("zero_point", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      // clang-format on
    }

    mllm::print(fmt::format("Tracing SHA model (seq={})...", N));
    auto ir = model.trace(trace_inputs, {});
    mllm::print("SHA model traced successfully.");

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect(fmt::format("qwen3_qnn_aot_sha_{}.mir", N), [&]() { mllm::print(ir["model"]); });
  }

  // Model length 1.
  {
    N = 1;

    // Sequence: [B, N]
    // past_key_i: [B, H, D, CL-N] for each layer i
    // past_value_i: [B, H, CL-N, D] for each layer i
    // causal_mask: [B, 1, N, CL]
    auto sequence = mllm::Tensor::zeros({1, N}, mllm::kInt32);
    auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16);

    // NOTE: force set causal mask to UInt16Asy
    // NOTE: Attach scale and zero point to causal mask
    {
      causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
      causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
      causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
    }

    // Create KV cache inputs for all layers
    std::unordered_map<std::string, mllm::Tensor> trace_inputs;
    trace_inputs["sequence"] = sequence;
    trace_inputs["causal_mask"] = causal_mask;
    for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      auto past_value_name = "past_value_" + std::to_string(i);

      // clang-format off
    trace_inputs[past_key_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
        CL - N,
    }, mllm::kUInt8PerTensorSym);
    trace_inputs[past_value_name] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim}, mllm::kUInt8PerTensorSym);
    trace_inputs[past_key_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_value_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
      // clang-format on
    }

    mllm::print("Tracing SHA model (seq=1)...");
    auto ir = model.trace(trace_inputs, {});
    mllm::print("SHA model traced successfully.");

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect("qwen3_qnn_aot_sha_1.mir", [&]() { mllm::print(ir["model"]); });
  }

  qnn_aot_env.saveContext("context.0", output_path.get());

  mllm::print("SHA compilation completed successfully!");
  mllm::print(fmt::format("  ar_len (prefill chunk) = {}", ar_len_arg.get()));
  mllm::print(fmt::format("  context_len            = {}", CL));
  mllm::print("Output files:");
  mllm::print(fmt::format("  - qwen3_qnn_aot_sha_{}.mir (IR dump for prefill)", ar_len_arg.get()));
  mllm::print("  - qwen3_qnn_aot_sha_1.mir (IR dump for decode)");
  mllm::print(fmt::format("  - {} (QNN context)", output_path.get()));
});
