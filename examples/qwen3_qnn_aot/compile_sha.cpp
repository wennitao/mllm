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

// Re-bake the rotary sin/cos LUTs (model.mllm_max_sin/cos_embedding) for the
// FULL max_cache_length. The ptq .mllm ships a LUT baked for only 1024
// positions; without this, dense prefill of prompts >1024 tokens gathers RoPE
// out of bounds → garbage output (exactly at the 1024 boundary). The split
// compiler already does this; the dense compiler was missing it. uint16-quantized
// to match the sin/cos_embedding_input_qdq the model applies.
void bakeRotaryEmbeddings(const mllm::ParameterFile::ptr_t& params, const mllm::models::qwen3::Qwen3Config& cfg) {
  const int head_dim = cfg.head_dim;
  const int max_pos = cfg.max_cache_length;
  const float theta = cfg.rope_theta;
  const int half = head_dim / 2;
  std::vector<float> inv_freq(half);
  for (int k = 0; k < half; ++k) inv_freq[k] = 1.0f / std::pow(theta, (float)(2 * k) / (float)head_dim);

  float sin_scale = 1.0f / 32768.0f, cos_scale = 1.0f / 32768.0f;
  int32_t sin_zp = 32768, cos_zp = 32768;
  if (params->has("model.sin_embedding_input_qdq.fake_quant.scale")) {
    sin_scale = params->pull("model.sin_embedding_input_qdq.fake_quant.scale").item<float>();
  }
  if (params->has("model.sin_embedding_input_qdq.fake_quant.zero_point")) {
    sin_zp = params->pull("model.sin_embedding_input_qdq.fake_quant.zero_point").item<int32_t>();
  }
  if (params->has("model.cos_embedding_input_qdq.fake_quant.scale")) {
    cos_scale = params->pull("model.cos_embedding_input_qdq.fake_quant.scale").item<float>();
  }
  if (params->has("model.cos_embedding_input_qdq.fake_quant.zero_point")) {
    cos_zp = params->pull("model.cos_embedding_input_qdq.fake_quant.zero_point").item<int32_t>();
  }

  auto quantize = [](float v, float scale, int32_t zp) -> uint16_t {
    long q = std::lround(v / scale) + zp;
    if (q < 0) q = 0;
    if (q > 65535) q = 65535;
    return (uint16_t)q;
  };

  std::vector<uint16_t> sin_buf((size_t)max_pos * head_dim);
  std::vector<uint16_t> cos_buf((size_t)max_pos * head_dim);
  for (int p = 0; p < max_pos; ++p) {
    for (int k = 0; k < half; ++k) {
      float ang = (float)p * inv_freq[k];
      float s = std::sin(ang), c = std::cos(ang);
      sin_buf[(size_t)p * head_dim + k] = quantize(s, sin_scale, sin_zp);
      sin_buf[(size_t)p * head_dim + k + half] = quantize(s, sin_scale, sin_zp);
      cos_buf[(size_t)p * head_dim + k] = quantize(c, cos_scale, cos_zp);
      cos_buf[(size_t)p * head_dim + k + half] = quantize(c, cos_scale, cos_zp);
    }
  }
  auto sin_t = mllm::Tensor::fromVector(sin_buf, {1, max_pos, head_dim}, mllm::kUInt16);
  auto cos_t = mllm::Tensor::fromVector(cos_buf, {1, max_pos, head_dim}, mllm::kUInt16);
  if (params->has("model.mllm_max_sin_embedding")) params->remove("model.mllm_max_sin_embedding");
  if (params->has("model.mllm_max_cos_embedding")) params->remove("model.mllm_max_cos_embedding");
  params->push("model.mllm_max_sin_embedding",
               sin_t.contiguous().setMemType(mllm::kParamsNormal).setName("model.mllm_max_sin_embedding"));
  params->push("model.mllm_max_cos_embedding",
               cos_t.contiguous().setMemType(mllm::kParamsNormal).setName("model.mllm_max_cos_embedding"));
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

  Argparse::parse(argc, argv);

  int N = 32;

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
  const int CL = model_cfg.max_cache_length;

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

  // Re-bake the rotary LUTs at CL (ptq .mllm ships a 1024-position LUT → dense
  // prefill >1024 tokens was garbage). See bakeRotaryEmbeddings above.
  bakeRotaryEmbeddings(params, model_cfg);
  mllm::print("Rotary sin/cos LUTs baked for {} positions.", CL);

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

    mllm::print("Tracing SHA model (seq=32)...");
    auto ir = model.trace(trace_inputs, {});
    mllm::print("SHA model traced successfully.");

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect("qwen3_qnn_aot_sha_32.mir", [&]() { mllm::print(ir["model"]); });
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

  qnn_aot_env.saveContext("context.0", "qwen3-1.7B-lpbq-sha.bin");

  mllm::print("SHA compilation completed successfully!");
  mllm::print("Output files:");
  mllm::print("  - qwen3_qnn_aot_sha_32.mir (IR dump for seq=32)");
  mllm::print("  - qwen3_qnn_aot_sha_1.mir (IR dump for seq=1)");
  mllm::print("  - qwen3-1.7B-lpbq-sha.bin (QNN context)");
});
