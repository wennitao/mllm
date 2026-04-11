// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Offline AOT compilation for Qwen3 prefill-only NPU graph.
// Run on x86 with QNN SDK to produce a .bin file, then push to device.
//
// Usage:
//   ./mllm-qwen3-npu-compile -m model.mllm -c config.json \
//       -aot_cfg qnn_aot_cfg.json [-qnn_env /path/to/qnn/libs] \
//       [--prefill_len 32] [-o qwen3_npu_prefill.bin]

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/models/qwen3/configuration_qwen3.hpp>
#include <mllm/utils/Log.hpp>

#include "../qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;

namespace {

std::string defaultQnnEnvPath() {
  if (const char* root = std::getenv("QAIRT_SDK_ROOT")) {
    return std::string(root) + "/lib/x86_64-linux-clang/";
  }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

}  // namespace

MLLM_MAIN({
  auto& help        = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path  = Argparse::add<std::string>("-m|--model_path").help("Model weights path (.mllm)").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Model config path (JSON)").required(true);
  auto& aot_cfg     = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT target config path (JSON)").required(true);
  auto& qnn_env     = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath()).help("QNN SDK lib dir");
  auto& prefill_len = Argparse::add<int>("--prefill_len").help("Prefill chunk size").def(32);
  auto& output_path = Argparse::add<std::string>("-o|--output").help("Output .bin path").def("qwen3_npu_prefill.bin");

  Argparse::parse(argc, argv);

  if (help.isSet()) { Argparse::printHelp(); return 0; }

  if (!aot_cfg.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot_config file path provided");
    return -1;
  }

  // int N  = prefill_len.get();
  // int CL = mllm::models::qwen3::Qwen3Config(config_path.get()).max_cache_length;
  int N = 32;
  int CL = 1024;

  auto model_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  // Load weights and push required AOT quantization constants
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  {
    params->push("causal_mask.scale",        mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
    params->push("causal_mask.zero_point",   mllm::Tensor::constant(65535, mllm::kInt32));
    params->push("constant_zero.scale",      mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
    params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  }

  // Slice MHA Q/K/V weights into per-head SHA weights (required for SHA attention)
  MLLM_INFO("Preparing SHA parameters (slicing MHA weights)...");
  mllm::models::qwen3::sha::prepareParametersForSHA(params, model_cfg);
  MLLM_INFO("SHA parameters prepared.");

  auto model = mllm::models::qwen3::sha::Qwen3ForCausalLM_SHA(model_cfg);
  model.load(params);

  // Create QNN AOT environment (same as compile_sha.cpp)
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));

  // Trace and compile only the prefill-length SHA graph. Keep the exact same
  // graph ABI as compile_sha.cpp / PromptProcessor:
  //   sequence    [1,N]
  //   causal_mask [1,1,N,CL]
  //   past_key_l  [1,Hkv,D,CL-N]
  //   past_val_l  [1,Hkv,CL-N,D]
  //
  // The first prefill chunk has zero valid past tokens at runtime, but the
  // graph still needs the past-cache inputs because PromptProcessor and the
  // smart mask are built around a fixed context-length ABI.
  {
    auto sequence    = mllm::Tensor::zeros({1, N}, mllm::kInt32);
    auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16);

    // Attach scale/zero_point to causal_mask
    causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
    causal_mask.attach("scale",      params->pull("causal_mask.scale"     ).impl(), true);
    causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);

    std::unordered_map<std::string, mllm::Tensor> trace_inputs;
    trace_inputs["sequence"]    = sequence;
    trace_inputs["causal_mask"] = causal_mask;

    for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      auto past_value_name = "past_value_" + std::to_string(i);

      trace_inputs[past_key_name] = mllm::Tensor::empty({
          1,
          model_cfg.num_key_value_heads,
          model_cfg.head_dim,
          CL - N,
      }, mllm::kUInt8PerTensorSym);
      trace_inputs[past_value_name] =
          mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim},
                              mllm::kUInt8PerTensorSym);

      trace_inputs[past_key_name].attach(
          "scale",
          params->pull("model.layers." + std::to_string(i)
                       + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(),
          true);
      trace_inputs[past_key_name].attach(
          "zero_point",
          params->pull("model.layers." + std::to_string(i)
                       + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point").impl(),
          true);

      trace_inputs[past_value_name].attach(
          "scale",
          params->pull("model.layers." + std::to_string(i)
                       + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(),
          true);
      trace_inputs[past_value_name].attach(
          "zero_point",
          params->pull("model.layers." + std::to_string(i)
                       + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point").impl(),
          true);
    }

    MLLM_INFO("Tracing prefill model (seq_len={})...", N);
    auto ir = model.trace(trace_inputs, {});
    MLLM_INFO("Trace done.");

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), params));
    pm.run();

    mllm::redirect("qwen3_npu_prefill_sha_" + std::to_string(N) + ".mir",
                   [&]() { mllm::print(ir["model"]); });
  }

  qnn_aot_env.saveContext("context.0", output_path.get());
  MLLM_INFO("Saved compiled prefill graph to: {}", output_path.get());

  return 0;
})
