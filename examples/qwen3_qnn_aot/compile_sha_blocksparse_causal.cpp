// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile the LPBQ SHA per-qb CAUSAL block-sparse Qwen3 model. Uses
// ptq_lpbq.mllm (Int8 weights + scale1/scale2 + per-op QDQ params) and the
// existing SHA LPBQ infrastructure with the per-qb causal block-sparse
// attention path swapped in.
//
// Usage:
//   ./compile_sha_blocksparse_causal -m /path/to/qwen3_1.7b_ptq_lpbq.mllm \
//                                    -c /path/to/config.json \
//                                    -aot_cfg /path/to/qnn_aot_cfg.json

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

#include "modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp"

using mllm::Argparse;
namespace bsc = mllm::models::qwen3::sha_blocksparse_causal;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) {
    return std::string(qairt_root) + "/lib/x86_64-linux-clang/";
  }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

// Rebake the rotary sin/cos LUTs as uint16-quantized tensors that match the
// PTQ calibration of `{sin,cos}_embedding_input_qdq` (scale 1/32768, zp 32768
// in the shipped ptq_lpbq.mllm). The shipped LUT is sized at the training
// context (1024 positions) which is too small for our deployment context, so
// we must rebake at cfg.max_cache_length.
//
// IMPORTANT — must NOT push fp16 here. The downstream `QDQ_ROPE` helper does
// `__unsafeSetDType(kUInt16PerTensorAsy)` and attaches the calibrated
// scale/zp. If the underlying bytes are fp16, the reinterpretation produces
// values that have nothing to do with the real sin/cos table, which silently
// corrupts K after RoPE (V is fine since it doesn't go through RoPE). That
// was the 2026-05-18 mono blocksparse-causal incoherence bug.
void bakeRotaryEmbeddings(const mllm::ParameterFile::ptr_t& params, const mllm::models::qwen3::Qwen3Config& cfg) {
  const int head_dim = cfg.head_dim;
  const int max_pos = cfg.max_cache_length;
  const float theta = cfg.rope_theta;
  const int half = head_dim / 2;
  std::vector<float> inv_freq(half);
  for (int k = 0; k < half; ++k) inv_freq[k] = 1.0f / std::pow(theta, (float)(2 * k) / (float)head_dim);

  // Read the PTQ-calibrated scale/zp for sin/cos_embedding_input_qdq, fall
  // back to the standard 1/32768 + 32768 mapping if they're missing.
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
  fprintf(stderr, "  bake: sin scale=%g zp=%d, cos scale=%g zp=%d\n", sin_scale, sin_zp, cos_scale, cos_zp); fflush(stderr);

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

  fprintf(stderr, "  bake: building tensors...\n"); fflush(stderr);
  auto sin_t = mllm::Tensor::fromVector(sin_buf, {1, max_pos, head_dim}, mllm::kUInt16);
  auto cos_t = mllm::Tensor::fromVector(cos_buf, {1, max_pos, head_dim}, mllm::kUInt16);

  if (params->has("model.mllm_max_sin_embedding")) params->remove("model.mllm_max_sin_embedding");
  if (params->has("model.mllm_max_cos_embedding")) params->remove("model.mllm_max_cos_embedding");
  params->push("model.mllm_max_sin_embedding",
               sin_t.contiguous().setMemType(mllm::kParamsNormal).setName("model.mllm_max_sin_embedding"));
  params->push("model.mllm_max_cos_embedding",
               cos_t.contiguous().setMemType(mllm::kParamsNormal).setName("model.mllm_max_cos_embedding"));
  fprintf(stderr, "  bake: done.\n"); fflush(stderr);
}
}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Path to ptq_lpbq .mllm params").required(true);
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Path to model config json").required(true);
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--qnn_aot_cfg").help("Path to QNN AOT config json");
  auto& qnn_env_path = Argparse::add<std::string>("-e|--qnn_env_path").help("QNN env / driver path").def(defaultQnnEnvPath());

  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  const int BQ = bsc::kBQ;
  const int top_k_BK = bsc::kTopKBK;
  const int hist_k_BK = bsc::kHistKBK;

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());
  const int CL = model_cfg.max_cache_length;

  fprintf(stderr, "[STEP 1] Loading params...\n"); fflush(stderr);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  fprintf(stderr, "[STEP2] Preparing SHA parameters (slicing LPBQ MHA weights + QDQ params)...\n"); fflush(stderr);
  mllm::models::qwen3::sha::prepareParametersForSHA(params, model_cfg);
  fprintf(stderr, "[STEP3] Baking rotary sin/cos LUTs...\n"); fflush(stderr);
  bakeRotaryEmbeddings(params, model_cfg);
  fprintf(stderr, "[STEP4] Rotary baked.\n"); fflush(stderr);

  // QDQ params for the runtime `mask` graph input. Convention: real value = 0
  // means ACTIVE (attend), non-zero means MASKED. So zero_point = 65535 maps
  // quantized 65535 → real 0, and quantized 0 → real -scale*65535 (very
  // negative). scale is small so dequantized "masked" still becomes some
  // negative — the where() pattern only checks for == 0, so the magnitude
  // doesn't matter, just that it's non-zero.
  params->push("mask.scale", mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
  params->push("mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  // QDQ params for the `constant_zero` in the where pattern (used as the
  // comparison constant). Match `mask`'s scale/zp so equalConstant is
  // semantically dequantize(mask) == dequantize(0_quantized).
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));

  fprintf(stderr, "[STEP5] QDQ params for mask/constant_zero pushed.\n"); fflush(stderr);

  bsc::Qwen3ForCausalLM_SHABlockSparseCausal model(model_cfg);
  fprintf(stderr, "[STEP6] Model constructed.\n"); fflush(stderr);
  model.load(params);
  fprintf(stderr, "[STEP7] Model loaded.\n"); fflush(stderr);

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));
  fprintf(stderr, "[STEP8] QnnAOTEnv ready.\n"); fflush(stderr);

  // Build trace inputs.
  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["sequence"] = mllm::Tensor::zeros({1, BQ}, mllm::kInt32);
  trace_inputs["position_ids"] = mllm::Tensor::zeros({1, BQ}, mllm::kInt32);
  fprintf(stderr, "[STEP9] sequence + position_ids set.\n"); fflush(stderr);

  // mask: uint16 quantized, with scale/zp attached.
  {
    fprintf(stderr, "  mask: build...\n"); fflush(stderr);
    auto mask = mllm::Tensor::zeros({1, 1, BQ, top_k_BK}, mllm::kUInt16);
    mask = mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
    fprintf(stderr, "  mask: attach scale (has=%d)\n", (int)params->has("mask.scale")); fflush(stderr);
    mask.attach("scale", params->pull("mask.scale").impl(), true);
    fprintf(stderr, "  mask: attach zp (has=%d)\n", (int)params->has("mask.zero_point")); fflush(stderr);
    mask.attach("zero_point", params->pull("mask.zero_point").impl(), true);
    trace_inputs["mask"] = mask;
    fprintf(stderr, "  mask: done\n"); fflush(stderr);
  }

  // K_arranged / V_arranged: uint8, runner-gathered historical slots.
  // Same scale/zp as the corresponding layer's K/V cache.
  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    auto kn = "k_arranged_" + std::to_string(i);
    auto vn = "v_arranged_" + std::to_string(i);
    auto k_arr = mllm::Tensor::empty({model_cfg.num_attention_heads, 1, model_cfg.head_dim, hist_k_BK}, mllm::kUInt8PerTensorSym);
    auto v_arr = mllm::Tensor::empty({model_cfg.num_attention_heads, 1, hist_k_BK, model_cfg.head_dim}, mllm::kUInt8PerTensorSym);
    auto layer_pfx = "model.layers." + std::to_string(i) + ".self_attn.";
    k_arr.attach("scale", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    k_arr.attach("zero_point",
                 params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
    v_arr.attach("scale", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    v_arr.attach("zero_point",
                 params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
    trace_inputs[kn] = k_arr;
    trace_inputs[vn] = v_arr;
  }

  mllm::print("Tracing LPBQ SHA per-qb causal block-sparse model (BQ={}, top_k_BK={}, hist_k_BK={})...", BQ, top_k_BK,
              hist_k_BK);
  auto ir = model.trace(trace_inputs, {});
  mllm::print("Trace complete.");

  // Dump pre-lowering MIR for debugging if compile fails later.
  mllm::redirect("qwen3_qnn_aot_sha_blocksparse_causal_PRE.mir", [&]() { mllm::print(ir["model"]); });

  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
  pm.run();

  mllm::redirect("qwen3_qnn_aot_sha_blocksparse_causal_BQ32.mir", [&]() { mllm::print(ir["model"]); });
  qnn_aot_env.saveContext("context.0", "qwen3-lpbq-sha-blocksparse-causal.bin");

  mllm::print("LPBQ SHA per-qb causal block-sparse compilation complete.");
});
