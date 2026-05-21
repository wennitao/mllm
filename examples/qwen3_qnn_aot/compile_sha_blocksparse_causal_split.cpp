// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Compile driver for the SPLIT-PREFILL variant of the LPBQ SHA per-qb causal
// block-sparse Qwen3 model. Emits 2L+1 graphs into one QNN context:
//   chunk_0, attn_0, chunk_1, attn_1, ..., chunk_{L-1}, attn_{L-1}, chunk_L
//
// See modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp for the chunk
// structure. Compared to compile_sha_blocksparse_causal.cpp, this driver:
//   * Builds a much larger trace_inputs map covering every chunk-boundary
//     tensor (q/k_curr/v_curr per layer at full-Sq and at per-qb slices,
//     residual_pre_attn per layer, attn_output per layer, plus the existing
//     K/V_arranged + mask).
//   * Adds two new PTQ params for the explicit chunk-boundary QDQs:
//       embed_tokens_output_qdq           (reuses layers.1.input_layernorm_input_qdq)
//       model.layers.{i}.self_attn.attn_output_boundary_qdq
//                                          (reuses head 0's attn_value_matmul_output_qdq)
//   * Loops over the trace's 2L+1 IRs and runs the AOT lowering pipeline once
//     per chunk; saveContext at the end emits all of them in one binary.
//
// Usage:
//   ./compile_sha_blocksparse_causal_split -m /path/to/qwen3_1.7b_ptq_lpbq.mllm \
//                                          -c /path/to/config.json \
//                                          -aot_cfg /path/to/qnn_aot_cfg.json \
//                                          [--sq 1024]

#include <cmath>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/passes/AOTCompileContext.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp"

using mllm::Argparse;
namespace bsc = mllm::models::qwen3::sha_blocksparse_causal_split;

namespace {

std::string defaultQnnEnvPath() {
  if (const char* qairt_root = std::getenv("QAIRT_SDK_ROOT")) {
    return std::string(qairt_root) + "/lib/x86_64-linux-clang/";
  }
  return "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/";
}

// Rebake rotary LUTs as uint16-quantized using the PTQ-calibrated
// sin/cos_embedding_input_qdq scale + zp. See compile_sha_blocksparse_causal
// .cpp::bakeRotaryEmbeddings for the rationale — fp16 bytes here get silently
// reinterpreted as uint16 by QDQ_ROPE and break RoPE.
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

// Borrow a QDQ scale/zp from one PTQ key and re-publish it under another. Used
// to give the explicit chunk-boundary QDQs the calibration of their nearest
// monolithic-graph analogue (see header comment for which key feeds which).
void aliasQdqParam(const mllm::ParameterFile::ptr_t& params, const std::string& src, const std::string& dst) {
  if (!params->has(dst + ".fake_quant.scale")) {
    params->push(dst + ".fake_quant.scale", params->pull(src + ".fake_quant.scale"));
  }
  if (!params->has(dst + ".fake_quant.zero_point")) {
    params->push(dst + ".fake_quant.zero_point", params->pull(src + ".fake_quant.zero_point"));
  }
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Path to ptq_lpbq .mllm params").required(true);
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Path to model config json").required(true);
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--qnn_aot_cfg").help("Path to QNN AOT config json");
  auto& qnn_env_path = Argparse::add<std::string>("-e|--qnn_env_path").help("QNN env / driver path").def(defaultQnnEnvPath());
  auto& sq_arg = Argparse::add<int>("--sq").help("Sequence length for the full-Sq chunks").def(1024);

  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  const int Sq = sq_arg.get();
  const int BQ = bsc::kBQ;
  const int top_k_BK = bsc::kTopKBK;
  const int hist_k_BK = bsc::kHistKBK;

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());
  const int L = model_cfg.num_hidden_layers;
  const int Hq = model_cfg.num_attention_heads;
  const int Hkv = model_cfg.num_key_value_heads;
  const int D = model_cfg.head_dim;
  const int hidden = model_cfg.hidden_size;

  fprintf(stderr, "[STEP 1] Loading params...\n"); fflush(stderr);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  fprintf(stderr, "[STEP 2] Preparing SHA parameters (slicing LPBQ MHA weights + QDQ params)...\n"); fflush(stderr);
  mllm::models::qwen3::sha::prepareParametersForSHA(params, model_cfg);
  fprintf(stderr, "[STEP 3] Baking rotary sin/cos LUTs...\n"); fflush(stderr);
  bakeRotaryEmbeddings(params, model_cfg);

  // QDQ params for mask + the equalConstant comparison constant. Same as the
  // monolithic variant — convention "real==0 means ACTIVE; real!=0 means MASKED".
  params->push("mask.scale", mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
  params->push("mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001f / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));

  // Chunk-boundary tensors are now fp16 (see modeling header attn output +
  // chunk_0 embedding output for the cast). The previous approach of borrowing
  // scales from internal QDQs (head 0's attn_value_matmul, layer 1's
  // input_layernorm) saturated at uint16 max — see split_prefill.md
  // § "2026-05-18 update" for the investigation trail. No QDQ params needed
  // for the BOUNDARY tensors themselves.
  //
  // Layer 0 needs a model.layers.0.input_layernorm_input_qdq scale because the
  // chunk pre() now uniformly quantizes the boundary fp16 input back to uint16
  // for internal RMSNorm + projections. The original PTQ skipped this key
  // (layer 0's input was always the embedding output → had embed_tokens_output
  // _qdq instead). Alias from layer 1's calibrated value as a reasonable proxy.
  aliasQdqParam(params, "model.layers.1.input_layernorm_input_qdq", "model.layers.0.input_layernorm_input_qdq");

  fprintf(stderr, "[STEP 4] QDQ params pushed (mask, constant_zero, layer-0 input_layernorm alias). Boundaries are fp16, no boundary QDQ aliases needed.\n"); fflush(stderr);

  bsc::Qwen3ForCausalLM_SHABlockSparseCausalSplit model(model_cfg);
  model.load(params);
  fprintf(stderr, "[STEP 5] Model constructed and loaded.\n"); fflush(stderr);

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  // ===========================================================================
  // Build trace_inputs map. Every chunk-boundary tensor + every attention input
  // gets created here with the right shape/dtype/quant-params attached.
  //
  // The trace_inputs map + the IR map are deliberately scoped so they
  // destruct BEFORE qnn_aot_env.saveContext() + program exit. At Sq=1024
  // the destruction order otherwise tripped a use-after-free in QnnAOTNodeTensor
  // → QNNTensorWrapper teardown (the wrapper held raw pointers to Tensor data
  // freed by an earlier destructor). Matches the scoping pattern in
  // compile_sha.cpp.
  {
  std::unordered_map<std::string, mllm::Tensor> trace_inputs;

  // ----- top-level inputs (chunk_0) ------------------------------------------
  trace_inputs["input_ids"] = mllm::Tensor::zeros({1, Sq}, mllm::kInt32);
  trace_inputs["position_ids"] = mllm::Tensor::zeros({1, Sq}, mllm::kInt32);
  // Index of the last real prompt token; chunk_L gathers this single position
  // before lm_head so the head runs at M=1 instead of M=Sq.
  trace_inputs["last_token_index"] = mllm::Tensor::zeros({1, 1}, mllm::kInt32);

  // ----- mask (shared across every attn_i dispatch) --------------------------
  {
    auto mask = mllm::Tensor::zeros({1, 1, BQ, top_k_BK}, mllm::kUInt16);
    mask = mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
    mask.attach("scale", params->pull("mask.scale").impl(), true);
    mask.attach("zero_point", params->pull("mask.zero_point").impl(), true);
    trace_inputs["mask"] = mask;
  }

  // ----- K_arranged / V_arranged (per layer, runner-gathered) ---------------
  for (int i = 0; i < L; ++i) {
    auto layer_pfx = "model.layers." + std::to_string(i) + ".self_attn.";
    auto k_arr = mllm::Tensor::empty({Hq, 1, D, hist_k_BK}, mllm::kUInt8PerTensorSym);
    auto v_arr = mllm::Tensor::empty({Hq, 1, hist_k_BK, D}, mllm::kUInt8PerTensorSym);
    k_arr.attach("scale", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    k_arr.attach("zero_point", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
    v_arr.attach("scale", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    v_arr.attach("zero_point", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
    trace_inputs["K_arranged_" + std::to_string(i)] = k_arr;
    trace_inputs["V_arranged_" + std::to_string(i)] = v_arr;
  }

  // ----- chunk-boundary buffers (per layer) ----------------------------------
  // residual_pre_attn_i and attn_output_i are fp16 (was uint16+QDQ; see
  // split_prefill.md § "2026-05-18 update" for why). Per-op QDQs inside each
  // chunk are unchanged.
  // q_i / k_curr_i / v_curr_i (full Sq, output of pre_i): use the same QDQs as
  // their per-qb attention-input siblings (q_rope_add_0_output_qdq_h0,
  // k_cast_to_int8_qdq, v_cast_to_int8_qdq) — representative per-tensor scale.
  for (int i = 0; i < L; ++i) {
    auto si = std::to_string(i);
    auto layer_pfx = "model.layers." + si + ".self_attn.";

    // residual_pre_attn_i — [1, Sq, hidden] fp16.
    {
      auto t = mllm::Tensor::zeros({1, Sq, hidden}, mllm::kFloat16);
      trace_inputs["residual_pre_attn_" + si] = t;
    }

    // q_i — [1, Hq, Sq, D], uint16 per-tensor (head 0 scale as representative).
    {
      auto t = mllm::Tensor::zeros({1, Hq, Sq, D}, mllm::kUInt16);
      t = t.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
      t.attach("scale", params->pull(layer_pfx + "q_rope_add_0_output_qdq_h0.fake_quant.scale").impl(), true);
      t.attach("zero_point", params->pull(layer_pfx + "q_rope_add_0_output_qdq_h0.fake_quant.zero_point").impl(), true);
      trace_inputs["q_" + si] = t;
    }

    // k_curr_i — [1, Hkv, D, Sq], uint8 per-tensor symmetric.
    {
      auto t = mllm::Tensor::empty({1, Hkv, D, Sq}, mllm::kUInt8PerTensorSym);
      t.attach("scale", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
      t.attach("zero_point", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      trace_inputs["k_curr_" + si] = t;
    }

    // v_curr_i — [1, Hkv, Sq, D], uint8 per-tensor symmetric.
    {
      auto t = mllm::Tensor::empty({1, Hkv, Sq, D}, mllm::kUInt8PerTensorSym);
      t.attach("scale", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
      t.attach("zero_point", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      trace_inputs["v_curr_" + si] = t;
    }

    // attn_output_i — [1, Hq, Sq, D] fp16.
    {
      auto t = mllm::Tensor::zeros({1, Hq, Sq, D}, mllm::kFloat16);
      trace_inputs["attn_output_" + si] = t;
    }

    // q_i_qb / k_curr_i_qb / v_curr_i_qb — per-qb slices, same scales as the
    // full-Sq buffers they're sliced from.
    {
      auto t = mllm::Tensor::zeros({1, Hq, BQ, D}, mllm::kUInt16);
      t = t.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
      t.attach("scale", params->pull(layer_pfx + "q_rope_add_0_output_qdq_h0.fake_quant.scale").impl(), true);
      t.attach("zero_point", params->pull(layer_pfx + "q_rope_add_0_output_qdq_h0.fake_quant.zero_point").impl(), true);
      trace_inputs["q_" + si + "_qb"] = t;
    }
    {
      auto t = mllm::Tensor::empty({1, Hkv, D, BQ}, mllm::kUInt8PerTensorSym);
      t.attach("scale", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
      t.attach("zero_point", params->pull(layer_pfx + "k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      trace_inputs["k_curr_" + si + "_qb"] = t;
    }
    {
      auto t = mllm::Tensor::empty({1, Hkv, BQ, D}, mllm::kUInt8PerTensorSym);
      t.attach("scale", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
      t.attach("zero_point", params->pull(layer_pfx + "v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      trace_inputs["v_curr_" + si + "_qb"] = t;
    }
  }

  // ----- score graph inputs (NPU block-selection matmul, FIXED stride S=8) ---
  // One shared weightless "score" graph: logits = qr · kcᵀ. qr/kc are the
  // reduced antidiagonal-packed Q/K for a layer, fp16 [Hq, Lr, SD] with
  // Lr=Sq/S, SD=S·D. The runner fills them per layer and dispatches per layer;
  // it MUST use this same stride. (S=8 keeps the [Hq,Lr,Lr] readback small.)
  constexpr int kScoreStride = 8;
  {
    if (Sq % kScoreStride != 0) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Sq={} not divisible by score stride {}", Sq, kScoreStride);
    }
    const int Lr = Sq / kScoreStride;
    const int SD = kScoreStride * D;
    // The score graph processes Hq/2 heads per dispatch so its working set fits
    // the 8 MB VTCM (→ ~0 DDR spill-fill; a full-Hq graph spilled 43.8 MB and
    // blew the V79 PD cap). The runner dispatches it twice/layer. kScoreHeadSplit
    // in ShaBlockSparsePromptProcessorSplit must match this divisor (2).
    const int Hq_score = Hq / 2;
    // kc is pre-TRANSPOSED to [Hh, SD, Lr] so the matmul is non-transposed
    // (the AOT MatMul lowering ignores transpose flags). logits = qr · kc.
    trace_inputs["score_qr"] = mllm::Tensor::zeros({Hq_score, Lr, SD}, mllm::kFloat16);
    trace_inputs["score_kc"] = mllm::Tensor::zeros({Hq_score, SD, Lr}, mllm::kFloat16);
  }

  fprintf(stderr, "[STEP 6] trace_inputs built (%zu entries).\n", trace_inputs.size()); fflush(stderr);

  // ===========================================================================
  // Trace — returns 2L+1 IRs keyed by chunk_0, chunk_1, ..., chunk_L, attn_0,
  // ..., attn_{L-1}.
  // ===========================================================================
  mllm::print("Tracing SPLIT model (Sq={}, BQ={}, L={}, expected graphs={})...", Sq, BQ, L, 2 * L + 2);
  auto ir = model.trace(trace_inputs, {});
  mllm::print("Trace complete. {} IRs produced.", ir.size());

  // Dump pre-lowering MIR per chunk for debugging.
  for (auto& kv : ir) {
    const std::string fname = "qwen3_split_PRE_" + kv.first + ".mir";
    mllm::redirect(fname, [&]() { mllm::print(kv.second); });
  }

  // Lower each chunk independently. The shared qnn_aot_env accumulates them
  // all into one context binary.
  std::vector<std::string> chunk_order;
  chunk_order.reserve(2 * L + 2);
  chunk_order.push_back("chunk_0");
  for (int i = 0; i < L; ++i) {
    chunk_order.push_back("attn_" + std::to_string(i));
    chunk_order.push_back("chunk_" + std::to_string(i + 1));
  }
  chunk_order.push_back("score");  // NPU block-selection matmul (one shared graph)
  // Fast-feasibility: lower ONLY the score graph (validates fp16-matmul lowering
  // + saves a throwaway context quickly) when MLLM_SCORE_ONLY is set.
  if (std::getenv("MLLM_SCORE_ONLY")) { chunk_order = {"score"}; }
  for (const auto& name : chunk_order) {
    if (ir.find(name) == ir.end()) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Missing IR for {}", name);
    }
    mllm::print("Lowering {}...", name);
    mllm::ir::PassManager pm(ir[name]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    // The AOT config's "graph_on_qnn" whitelist names which graph the QNN
    // passes should lower. createQnnAOTLoweringPipeline (above) re-reads the
    // config file each call, so we override the in-memory list to point at
    // THIS chunk after the pipeline is built but before pm.run() reads it.
    // "op_on_qnn" stays ["lm_head"] for the final chunk and is harmlessly
    // never-matched for the others (no lm_head op exists there).
    auto& cfg = mllm::qnn::aot::AOTCompileContext::getInstance().getConfig();
    cfg["graph_on_qnn"] = nlohmann::json::array({name});
    // chunk_graph_name tells the patched SplitLLMGraph / MergeLLMHead /
    // LLM2QnnLowering passes to operate on this pre-split chunk instead of
    // the standard "model" graph (see those passes' early-return branches).
    cfg["chunk_graph_name"] = name;
    pm.run();
    mllm::redirect("qwen3_split_POST_" + name + ".mir", [&]() { mllm::print(ir[name]); });
  }
  }  // end trace_inputs + ir scope

  qnn_aot_env.saveContext("context.0", "qwen3-lpbq-sha-blocksparse-causal-split.bin");
  mllm::print("SPLIT-PREFILL compilation complete. {} graphs in one context (incl. 1 score).", 2 * L + 2);

  // FIXME(split-prefill): at Sq=1024 the program-exit destructor chain
  // crashes inside align_free → TensorStorage::~ → QNNTensorWrapper::~ →
  // QnnAOTNodeTensor::~. Root cause is shared data-pointer ownership between
  // mllm's CPU-allocated TensorStorage and QNN's clientBuf — the same
  // pointer ends up being freed twice across the two destructor chains.
  // The Sq=256 case lives because the smaller object volume happens not to
  // trip the bad pointer, but it's the same latent bug.
  //
  // The crash is purely cosmetic — saveContext above has already written the
  // valid context binary to disk. Skip the C++ runtime cleanup and let the
  // OS reclaim memory on process exit. Proper fix needs investigation in
  // QnnAOTNodeTensor / wrapTensors2TensorIR.
  std::fflush(nullptr);
  _exit(0);
});
