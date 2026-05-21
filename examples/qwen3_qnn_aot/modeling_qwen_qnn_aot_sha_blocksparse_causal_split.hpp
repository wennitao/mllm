// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// LPBQ SHA per-qb CAUSAL block-sparse Qwen3 model, SPLIT-PREFILL variant.
//
// Variant of modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp that breaks the
// monolithic per-qb graph into 2L+1 chunks so the non-attention compute
// (QKV/MLP/O-proj/Norms) runs at M=Sq instead of M=BQ=32, escaping the small-M
// HMX efficiency cliff. Compiled context has:
//
//   chunk_0          (M=Sq)    embedding + sin/cos gather + layer0_pre_attn
//   attn_0           (M=BQ)    layer 0 attention (per-qb dispatch, num_qb invocations)
//   chunk_i (1..L-1) (M=Sq)    layer_{i-1}_post_attn + layer_i_pre_attn
//   attn_i (1..L-1)  (M=BQ)    layer i attention
//   chunk_L          (M=Sq)    layer_{L-1}_post_attn + final_norm + lm_head
//
// Total: 2L+1 compiled QNN graphs, all in one context. Attention chunks are
// structurally identical across layers but each carries its layer's QDQ
// scale/zp constants — so we trace L separate attention graphs.
//
// Module hierarchy mirrors the monolithic variant so PTQ weights load
// unchanged: each Qwen3DecoderSplit lives at "layers.{i}" and owns
// "input_layernorm", "self_attn" (Qwen3AttnSplit), "post_attention_layernorm",
// "mlp". The forward() entry point of each module is unused; we expose
// pre()/attn()/post() member functions called directly from the top-level
// trace() so QDQ name resolution lines up with the existing param file.
//
// Chunk-boundary buffers are shared rpcmem (output of one chunk = APP_READ
// alias of the input of the next). The runner manages the buffer wiring +
// per-qb slicing of Q/K_curr/V_curr/attn_output between full-Sq pre/post and
// per-qb attention dispatches.
//
// KV cache:
//   * V: pre-attn V_curr aliases into the V cache buffer directly (V cache
//     layout [1, Hkv, max_kv_len, D] makes the first Sq rows a contiguous
//     prefix). Zero copy.
//   * K: K cache layout is [1, Hkv, D, max_kv_len] (transposed for the QK^T
//     dispatch), so the first Sq columns are NOT a contiguous prefix. The
//     pre-attn emits K_curr as a separate rpcmem buffer; the runner does a
//     strided memcpy into the cache after the dispatch. ~1 MB per layer at
//     Sq=1024 / ~28 MB per prefill / ~1 ms at 23 GB/s — small overhead, MVP
//     simplification. A later optimisation can flip K cache layout to
//     row-major and make this zero-copy too.

#pragma once

#include "mllm/core/TensorStorage.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

#include "modeling_qwen_qnn_aot_sha.hpp"  // sha::ptq, sha::Qwen3MLP, sha::rotateHalf, sha::prepareParametersForSHA

namespace mllm::models::qwen3::sha_blocksparse_causal_split {

namespace ptq = mllm::models::qwen3::sha::ptq;
using mllm::models::qwen3::sha::Qwen3MLP;
using mllm::models::qwen3::sha::rotateHalf;

using vi32 = std::vector<int32_t>;
#define BSC_CONV2D_PROPERTY vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G16

// Match the monolithic variant. Reused at trace time and by the runner for
// per-qb slicing.
constexpr int kBQ = 32;
constexpr int kBK = 32;
constexpr int kTopK = 8;
constexpr int kTopKBK = kTopK * kBK;        // 256
constexpr int kHistKBK = (kTopK - 1) * kBK; // 224

// ============================================================================
// Attention module: per-head Q/K/V proj, RMSNorm, RoPE, O-proj. No mask/softmax
// state held here — softmax is invoked per-qb inside attn(). Lives at
// "layers.{i}.self_attn"; QDQ keys resolve to the existing PTQ params unchanged.
//
// pre(h, sin, cos):
//   h is post-input_layernorm hidden_states at full Sq. Returns:
//     q       [1, Hq, Sq, head_dim]   uint16
//     k_curr  [1, Hkv, head_dim, Sq]  uint8 (transposed, cache row-stride match)
//     v_curr  [1, Hkv, Sq, head_dim]  uint8 (row-major)
//
// attn(q_qb, kc_qb, vc_qb, K_arr, V_arr, mask):
//   Per-qb. Inputs are per-qb slices of pre()'s outputs plus the runner-gathered
//   historical K/V and the per-qb causal+sparse mask. Returns:
//     y_qb    [1, Hq, BQ, head_dim]   uint16   (pre-O-proj attention values)
//
// post(attn_output):
//   Takes the full-Sq stitched attention output [1, Hq, Sq, head_dim] and runs
//   the O-projection. Returns:
//     y       [1, Sq, hidden_size]    uint16
// ============================================================================
class Qwen3AttnSplit final : public nn::Module {
 public:
  std::vector<nn::Conv2D> q_projs_;
  std::vector<nn::Conv2D> k_projs_;
  std::vector<nn::Conv2D> v_projs_;
  std::vector<nn::RMSNorm> rms_norm_q_;
  std::vector<nn::RMSNorm> rms_norm_k_;
  nn::Conv2D o_proj_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;

  Qwen3AttnSplit() = default;

  Qwen3AttnSplit(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    head_dim_ = cfg.head_dim;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = (1.f / sqrtf((float)head_dim_));

    for (int h = 0; h < num_attention_heads_; ++h) {
      q_projs_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden_size_, head_dim_, BSC_CONV2D_PROPERTY));
      rms_norm_q_.emplace_back(reg<nn::RMSNorm>("q_norm." + std::to_string(h), cfg.rms_norm_eps));
    }
    for (int h = 0; h < num_key_value_heads_; ++h) {
      k_projs_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(h), hidden_size_, head_dim_, BSC_CONV2D_PROPERTY));
      rms_norm_k_.emplace_back(reg<nn::RMSNorm>("k_norm." + std::to_string(h), cfg.rms_norm_eps));
      v_projs_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(h), hidden_size_, head_dim_, BSC_CONV2D_PROPERTY));
    }
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, BSC_CONV2D_PROPERTY);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>&, const std::vector<AnyValue>&) override {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Qwen3AttnSplit::forward unused — call pre/attn/post directly.");
    return {};
  }

  // ------- pre: Q/K/V proj + RoPE/RMSNorm + uint8 K/V quant ----------------
  std::vector<Tensor> pre(Tensor h, Tensor sin, Tensor cos) {
    h = ptq::QDQ(this, h, "q_proj_input_qdq");
    h = h.view({1, 1, -1, hidden_size_}, true);

    std::vector<Tensor> q_per_head; q_per_head.reserve(num_attention_heads_);
    for (int hd = 0; hd < num_attention_heads_; ++hd) {
      auto q_h = q_projs_[hd](h);
      q_h = q_h.view({1, 1, -1, head_dim_}, true);
      q_per_head.push_back(q_h);
    }
    std::vector<Tensor> k_per_head; k_per_head.reserve(num_key_value_heads_);
    std::vector<Tensor> v_per_head; v_per_head.reserve(num_key_value_heads_);
    for (int hd = 0; hd < num_key_value_heads_; ++hd) {
      auto k_h = k_projs_[hd](h);
      k_h = k_h.view({1, 1, -1, head_dim_}, true);
      k_per_head.push_back(k_h);
      auto v_h = v_projs_[hd](h);
      v_h = v_h.view({1, 1, -1, head_dim_}, true);
      v_per_head.push_back(v_h);
    }

    for (int hd = 0; hd < num_attention_heads_; ++hd) {
      std::string hs = std::to_string(hd);
      q_per_head[hd] = rms_norm_q_[hd](ptq::QDQ(this, q_per_head[hd], "q_norm_input_qdq_h" + hs));
      q_per_head[hd] = ptq::QDQ(this, q_per_head[hd], "q_norm_output_qdq_h" + hs);
      q_per_head[hd] = ptq::QDQ(this,
          ptq::QDQ(this, q_per_head[hd] * cos, "q_rope_mul_0_output_qdq_h" + hs)
              + ptq::QDQ(this, rotateHalf(q_per_head[hd], this, "q_rope_neg_half_qdq_h" + hs) * sin,
                         "q_rope_mul_1_output_qdq_h" + hs),
          "q_rope_add_0_output_qdq_h" + hs);
    }
    for (int hd = 0; hd < num_key_value_heads_; ++hd) {
      std::string hs = std::to_string(hd);
      k_per_head[hd] = rms_norm_k_[hd](ptq::QDQ(this, k_per_head[hd], "k_norm_input_qdq_h" + hs));
      k_per_head[hd] = ptq::QDQ(this, k_per_head[hd], "k_norm_output_qdq_h" + hs);
      k_per_head[hd] = ptq::QDQ(this,
          ptq::QDQ(this, k_per_head[hd] * cos, "k_rope_mul_0_output_qdq_h" + hs)
              + ptq::QDQ(this, rotateHalf(k_per_head[hd], this, "k_rope_neg_half_qdq_h" + hs) * sin,
                         "k_rope_mul_1_output_qdq_h" + hs),
          "k_rope_add_0_output_qdq_h" + hs);
    }

    std::vector<Tensor> k_u8; k_u8.reserve(num_key_value_heads_);
    std::vector<Tensor> v_u8; v_u8.reserve(num_key_value_heads_);
    for (int hd = 0; hd < num_key_value_heads_; ++hd) {
      std::string hs = std::to_string(hd);
      auto k_h = k_per_head[hd].to(kUInt8PerTensorSym);
      k_h = ptq::QDQ_KV(this, k_h, "k_cast_to_int8_qdq_h" + hs);
      k_h = k_h.transpose(2, 3);                                            // [1, 1, D, Sq]
      k_u8.push_back(k_h);
      auto v_h = ptq::QDQ(this, v_per_head[hd], "v_cast_to_int16_qdq_h" + hs);
      v_h = v_h.to(kUInt8PerTensorSym);
      v_h = ptq::QDQ_KV(this, v_h, "v_cast_to_int8_qdq_h" + hs);            // [1, 1, Sq, D]
      v_u8.push_back(v_h);
    }

    auto q_out      = nn::functional::concat(q_per_head, 1);  // [1, Hq, Sq, D]
    auto k_curr_out = nn::functional::concat(k_u8, 1);        // [1, Hkv, D, Sq]
    auto v_curr_out = nn::functional::concat(v_u8, 1);        // [1, Hkv, Sq, D]
    return {q_out, k_curr_out, v_curr_out};
  }

  // ------- attn: per-head QK/scale/mask/softmax/PV at M = BQ ----------------
  std::vector<Tensor> attn(Tensor q, Tensor k_curr, Tensor v_curr, Tensor K_arranged, Tensor V_arranged, Tensor mask) {
    std::vector<Tensor> attn_outputs; attn_outputs.reserve(num_attention_heads_);
    for (int hd = 0; hd < num_attention_heads_; ++hd) {
      std::string hs = std::to_string(hd);
      const int kv_head_idx = hd / num_key_value_groups_;

      auto q_h        = q.slice({kAll, {hd, hd + 1}, kAll, kAll}, true);                       // [1, 1, BQ, D]
      auto K_curr_h   = k_curr.slice({kAll, {kv_head_idx, kv_head_idx + 1}, kAll, kAll}, true); // [1, 1, D, BQ]
      auto V_curr_h   = v_curr.slice({kAll, {kv_head_idx, kv_head_idx + 1}, kAll, kAll}, true); // [1, 1, BQ, D]
      auto K_hist_h   = K_arranged.slice({{hd, hd + 1}, kAll, kAll, kAll}, true);              // [1, 1, D, kHistKBK]
      auto V_hist_h   = V_arranged.slice({{hd, hd + 1}, kAll, kAll, kAll}, true);              // [1, 1, kHistKBK, D]

      auto K_full_h = nn::functional::concat({K_hist_h, K_curr_h}, -1);  // [1, 1, D, kTopKBK]
      auto V_full_h = nn::functional::concat({V_hist_h, V_curr_h}, 2);   // [1, 1, kTopKBK, D]

      auto a = ptq::QDQ(this, nn::functional::matmul(q_h, K_full_h), "qk_matmul_output_qdq_h" + hs);

      auto sc = Tensor::constant(scale_, kFloat32);
      sc = ptq::QDQ(this, sc, "scaling_qdq_h" + hs);
      a = ptq::QDQ(this, a.mulConstant(sc), "mul_0_output_qdq_h" + hs);

      auto a_min = ptq::QDQ(this, a.min(-1, true), "reduce_min_output_qdq_h" + hs);
      auto mv = Tensor::constant(-20, kFloat32);
      mv = ptq::QDQ(this, mv, "neg_20_qdq_h" + hs);
      auto a_vv = ptq::QDQ(this, a_min.addConstant(mv), "minus_0_output_qdq_h" + hs);
      auto zero_c = Tensor::constant(0.f, kFloat32);
      zero_c = ptq::QDQ_CONSTANT(this, zero_c, "constant_zero");
      a = nn::functional::where(mask.equalConstant(zero_c), a, a_vv);
      a = ptq::QDQ(this, a, "where_attn_qdq_h" + hs);
      a = ptq::QDQ(this, nn::functional::softmax(a, -1), "softmax_output_qdq_h" + hs);

      auto y_h = ptq::QDQ(this, nn::functional::matmul(a, V_full_h), "attn_value_matmul_output_qdq_h" + hs);
      attn_outputs.push_back(y_h);
    }
    auto y = nn::functional::concat(attn_outputs, 1);  // [1, Hq, BQ, D]
    // Cross-chunk boundary: emit fp16 instead of uint16+QDQ. The earlier
    // unified-per-tensor-scale alias-from-head-0 approach saturated the boundary
    // at uint16 max (~63968) because head 0's calibrated scale didn't fit the
    // concat'd tensor's actual range — 4x/16x headroom tweaks didn't recover
    // signal because per-boundary requant noise accumulated across 2L+1 graph
    // crossings. fp16 dequantizes once at the chunk output, no per-boundary
    // requant. The next chunk's first per-op QDQ ("add_0_lhs_input_qdq") acts
    // as the entry quantize into uint16-land. See docs/qnn_backend/split_prefill.md
    // § "2026-05-18 update".
    y = y.to(kFloat16);
    return {y};
  }

  // ------- post: full-Sq O-projection --------------------------------------
  std::vector<Tensor> post(Tensor attn_output) {
    // attn_output: [1, Hq, Sq, D] fp16 (boundary; stitched from L per-qb attn
    // dispatches by the runtime). Quantize back to uint16 for o_proj — borrow
    // head 0's attn_value_matmul scale, matching what the mono SHA model
    // implicitly uses as o_proj's input scale. Tried passing fp16 directly
    // into o_proj — PTQPass fails to solve tensor quant spec without this.
    attn_output = ptq::QDQ(this, attn_output, "attn_value_matmul_output_qdq_h0");
    auto y = attn_output.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, true);  // [1, 1, Sq, Hq*D]
    y = o_proj_(y).view({1, -1, hidden_size_}, true);                                                // [1, Sq, hidden]
    return {y};
  }
};

// ============================================================================
// Per-layer decoder. Lives at "layers.{i}". Owns input_layernorm, self_attn
// (Qwen3AttnSplit), post_attention_layernorm, mlp. Exposes pre/attn/post
// instead of one forward.
//
// pre(hidden_states, sin, cos):
//   inputs: hidden_states  [1, Sq, hidden]
//   returns: [residual, q, k_curr, v_curr]  (residual = hidden_states pre-norm)
//
// attn(q_qb, kc_qb, vc_qb, K_arr, V_arr, mask):
//   inputs: per-qb tensors + runner-gathered K_arr/V_arr + per-qb mask
//   returns: [attn_output_qb]   [1, Hq, BQ, D]
//
// post(residual, attn_output):
//   inputs: residual [1, Sq, hidden],  attn_output [1, Hq, Sq, D]
//   returns: hidden_states [1, Sq, hidden]    (= next layer's residual input)
// ============================================================================
class Qwen3DecoderSplit final : public nn::Module {
 public:
  int layer_idx_ = 0;
  Qwen3AttnSplit self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layernorm_;
  nn::RMSNorm post_attention_layernorm_;

  Qwen3DecoderSplit() = default;

  Qwen3DecoderSplit(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3AttnSplit>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layernorm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layernorm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>&, const std::vector<AnyValue>&) override {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Qwen3DecoderSplit::forward unused — call pre/attn/post directly.");
    return {};
  }

  std::vector<Tensor> pre(Tensor hidden_states, Tensor sin, Tensor cos) {
    // hidden_states is the fp16 chunk-boundary input (chunk_0 dequantizes the
    // embedding; mid-chunks pass in the previous layer's uint16 post() output).
    // Quantize it once via the input_layernorm QDQ — this both (a) gives the
    // upstream Add a solved quant spec and (b) yields the uint16 tensor for the
    // internal RMSNorm/projection path. The residual that crosses to the next
    // chunk is the DEQUANTIZED (real fp16) form of that same quantized value.
    //
    // Critically, residual must NOT be the raw `hidden_states` passthrough:
    // for mid-chunks that tensor is a uint16 quantized layer output, and binding
    // it straight into the fp16 boundary buffer writes the integer codes (0..65535)
    // as fp16 — ~3000x inflated → fp16 overflow → inf → NaN from layer 2 on.
    auto h = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq");
    auto residual = h.to(kFloat16);
    h = input_layernorm_(h);
    auto qkv = self_attn_.pre(h, sin, cos);  // [q, k_curr, v_curr]
    return {residual, qkv[0], qkv[1], qkv[2]};
  }

  std::vector<Tensor> attn(Tensor q_qb, Tensor kc_qb, Tensor vc_qb, Tensor K_arr, Tensor V_arr, Tensor mask) {
    return self_attn_.attn(q_qb, kc_qb, vc_qb, K_arr, V_arr, mask);
  }

  std::vector<Tensor> post(Tensor residual, Tensor attn_output) {
    // residual arrives as fp16 (chunk boundary). Quantize to uint16 with
    // input_layernorm_input_qdq scale so the residual+attn_output add below
    // has matching dtypes. (Without this, QNN's ElementWiseAdd rejects mixed
    // fp16 + uint16 inputs with "mismatching datatypes 0x216 != 0x416".)
    residual = ptq::QDQ(this, residual, "input_layernorm_input_qdq");
    auto y = self_attn_.post(attn_output)[0];                                                       // [1, Sq, hidden]
    auto h = ptq::QDQ(this, residual + ptq::QDQ(this, y, "add_0_lhs_input_qdq"), "add_0_output_qdq");
    auto residual2 = h;
    h = post_attention_layernorm_(h);
    // Call mlp_.forward directly (not via operator()) so the MLP body is
    // emitted into THIS chunk's subgraph instead of a nested CallGraphOp.
    // Qwen3MLP is a composite Module; its operator() wraps the body in a
    // sub-graph that LLM2QnnLoweringPass's top-level walk does not capture,
    // leaving a dangling call that crashes QNN's HTP prep downstream.
    // Single-op Modules (RMSNorm, Conv2D) are inlined by the trace dispatcher
    // and don't need this workaround.
    h = mlp_.forward({h}, {})[0];
    h = residual2 + ptq::QDQ(this, h, "add_1_lhs_input_qdq");
    return {h};
  }
};

// ============================================================================
// Top-level text model. Owns embedding, RoPE LUTs, per-layer decoders, final
// norm. forward()/trace() are inherited from nn::Module; the real trace
// orchestration lives in Qwen3ForCausalLM_*::trace below.
// ============================================================================
class Qwen3TextSplit final : public nn::Module {
 public:
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  std::vector<Qwen3DecoderSplit> decoders_;
  nn::RMSNorm norm_;
  int num_hidden_layers_;
  int hidden_size_;

  Qwen3TextSplit() = default;

  Qwen3TextSplit(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
    decoders_.reserve(num_hidden_layers_);
    for (int i = 0; i < num_hidden_layers_; ++i) {
      auto& d = decoders_.emplace_back(reg<Qwen3DecoderSplit>("layers." + std::to_string(i), cfg));
      d.layer_idx_ = i;
    }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>&, const std::vector<AnyValue>&) override {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Qwen3TextSplit::forward unused — see Qwen3ForCausalLM_*::trace.");
    return {};
  }
};

// ============================================================================
// Chunk wrappers. Each chunk lives in its own QNN graph and must be wrapped in
// a CallGraphOp by the trace machinery — that wrapping happens inside
// Module::__trace, which is only invoked via Module::operator(). So we expose
// each chunk as a Module whose forward() calls the underlying decoder's
// pre/attn/post methods. The wrappers own no weights; QDQ resolution inside
// pre/attn/post uses the underlying decoder/attn module's `this` pointer, so
// PTQ key lookup is unaffected by the wrapper's module name.
//
// The wrappers are NOT registered as sub-modules of the CausalLM — they're
// constructed inline at trace time with explicit names, so they don't
// pollute the PTQ param namespace.
// ============================================================================
class Chunk0Module final : public nn::Module {
 public:
  Qwen3TextSplit* llm_ = nullptr;
  nn::Module* root_ = nullptr;  // CausalLM, used for the embed_tokens_output_qdq key (root-level)

  Chunk0Module() = default;
  Chunk0Module(const std::string& name, Qwen3TextSplit* llm, nn::Module* root) : nn::Module(name), llm_(llm), root_(root) {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto input_ids = inputs[0];
    auto position_ids = inputs[1];
    auto x = llm_->embedding_(input_ids);
    // Chunk-0 boundary output: emit fp16 to chunk_1 (see attn output note above).
    x = x.to(kFloat16);
    auto sin = nn::functional::gather(ptq::QDQ_ROPE(llm_, llm_->rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto cos = nn::functional::gather(ptq::QDQ_ROPE(llm_, llm_->rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);
    return llm_->decoders_[0].pre(x, sin, cos);  // {residual, q, k_curr, v_curr}
  }
};

class MidChunkModule final : public nn::Module {
 public:
  Qwen3TextSplit* llm_ = nullptr;
  int layer_i_post_ = 0;  // layer index whose post-attn runs
  // pre_i for layer (layer_i_post_+1) runs in the same chunk.

  MidChunkModule() = default;
  MidChunkModule(const std::string& name, Qwen3TextSplit* llm, int layer_i_post)
      : nn::Module(name), llm_(llm), layer_i_post_(layer_i_post) {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto residual = inputs[0];
    auto attn_output = inputs[1];
    auto position_ids = inputs[2];
    auto h = llm_->decoders_[layer_i_post_].post(residual, attn_output)[0];
    auto sin = nn::functional::gather(ptq::QDQ_ROPE(llm_, llm_->rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto cos = nn::functional::gather(ptq::QDQ_ROPE(llm_, llm_->rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);
    return llm_->decoders_[layer_i_post_ + 1].pre(h, sin, cos);  // {residual, q, k_curr, v_curr}
  }
};

class FinalChunkModule final : public nn::Module {
 public:
  Qwen3TextSplit* llm_ = nullptr;
  nn::Module* root_ = nullptr;
  nn::Conv2D* lm_head_ = nullptr;
  int hidden_size_ = 0;
  int last_layer_idx_ = 0;

  FinalChunkModule() = default;
  FinalChunkModule(const std::string& name, Qwen3TextSplit* llm, nn::Module* root, nn::Conv2D* lm_head, int hidden_size, int last_layer_idx)
      : nn::Module(name), llm_(llm), root_(root), lm_head_(lm_head), hidden_size_(hidden_size), last_layer_idx_(last_layer_idx) {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto residual = inputs[0];
    auto attn_output = inputs[1];
    auto last_token_index = inputs[2];  // [1, 1] int32 — position of the last real prompt token
    auto h = llm_->decoders_[last_layer_idx_].post(residual, attn_output)[0];  // [1, Sq, hidden]
    h = llm_->norm_(ptq::QDQ(llm_, h, "norm_input_qdq"));                      // [1, Sq, hidden]
    // Only the last real token's logits are ever sampled, so gather that one
    // position BEFORE lm_head. Running lm_head over all Sq positions wastes
    // ~Sq× the compute and (at Sq=1024) ~630 MB of IO+spill-fill on the
    // [1,1,Sq,vocab] output — the dominant term in the PD memory estimate.
    h = nn::functional::gather(h, 1, last_token_index);                        // [1, 1, hidden]
    h = h.view({1, 1, -1, hidden_size_}, true);                               // [1, 1, 1, hidden]
    auto logits = (*lm_head_)(ptq::QDQ(root_, h, "lm_head_input_qdq"));        // [1, 1, 1, vocab]
    logits = ptq::QDQ(root_, logits, "lm_head_output_qdq");
    return {logits};
  }
};

class AttnChunkModule final : public nn::Module {
 public:
  Qwen3TextSplit* llm_ = nullptr;
  int layer_i_ = 0;

  AttnChunkModule() = default;
  AttnChunkModule(const std::string& name, Qwen3TextSplit* llm, int layer_i) : nn::Module(name), llm_(llm), layer_i_(layer_i) {}

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    // q_qb, k_curr_qb, v_curr_qb, K_arranged, V_arranged, mask
    return llm_->decoders_[layer_i_].attn(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]);
  }
};

// ============================================================================
// XAttention SCORE graph (NPU-offloaded block-selection matmul). One shared,
// weightless graph reused for every layer: logits = qr · kcᵀ. Inputs are the
// reduced antidiagonal-packed Q and K for a layer (fp16, [Hq, Lr, S·D]); output
// is the raw logit grid [Hq, Lr, Lr]. The history-only block-causal softmax and
// the block-pool stay on CPU in the runner (the per-layer `temp` scale and the
// staircase causal mask are applied there), so this graph carries no per-layer
// constants and can be a single reused dispatch. Compiled at a FIXED stride
// (S=8 → SD=1024, Lr=Sq/8); the runner must build qr/kc at that same stride.
// ============================================================================
class ScoreModule final : public nn::Module {
 public:
  ScoreModule() = default;
  explicit ScoreModule(const std::string& name) : nn::Module(name) {}
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    // logits [Hq, Lr, Lr] = qr [Hq, Lr, SD] · kcT [Hq, SD, Lr]. NON-transposed:
    // the AOT MatMul lowering pattern (visitor/Matmul.cpp) ignores transpose
    // flags, so kcT must be supplied pre-transposed (the runner builds it that
    // way). This mirrors the attn QK matmul, which feeds K as [D, hist].
    return {nn::functional::matmul(inputs[0], inputs[1], /*transpose_a=*/false, /*transpose_b=*/false)};
  }
};

// ============================================================================
// CausalLM wrapper with multi-chunk trace driver. Returns 2L+1 IRs keyed by:
//   "chunk_0", "chunk_1", ..., "chunk_L", "attn_0", "attn_1", ..., "attn_{L-1}".
//
// trace_inputs map (built by the compile driver, used here via input.at):
//   input_ids                  [1, Sq] int32
//   position_ids               [1, Sq] int32                          — used by every chunk_i for RoPE recompute
//   mask                       [1, 1, kBQ, kTopKBK] uint16            — shared by every attn_i (runner refills per-qb)
//   residual_pre_attn_{i}      [1, Sq, hidden]                        — produced by chunk_{i}, consumed by chunk_{i+1}
//   attn_output_{i}            [1, Hq, Sq, head_dim]                  — produced by L attn_i dispatches (stitched), consumed by chunk_{i+1}
//   q_{i}_qb                   [1, Hq, kBQ, head_dim]                 — slice of pre_{i}'s q output; runner sets per qb
//   k_curr_{i}_qb              [1, Hkv, head_dim, kBQ]
//   v_curr_{i}_qb              [1, Hkv, kBQ, head_dim]
//   K_arranged_{i}             [Hq, 1, head_dim, kHistKBK]            — runner gather
//   V_arranged_{i}             [Hq, 1, kHistKBK, head_dim]            — runner gather
//
// All chunk-boundary tensors are uint16 (asymmetric per-tensor); K/V curr are
// uint8 (symmetric per-tensor). QDQ params attached by the compile driver
// from `params` before tracing.
// ============================================================================
class Qwen3ForCausalLM_SHABlockSparseCausalSplit : public ARGeneration, public nn::Module {
 public:
  Qwen3ForCausalLM_SHABlockSparseCausalSplit(const Qwen3Config& cfg) : cfg_(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;
    llm_ = reg<Qwen3TextSplit>("model", cfg);
    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, BSC_CONV2D_PROPERTY);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    IROutput result;
    const int L = cfg_.num_hidden_layers;

    auto input_ids = input.at("input_ids");
    auto position_ids = input.at("position_ids");
    auto mask = input.at("mask");

    // Each chunk is wrapped in its own Module so that Module::__trace creates
    // a CallGraphOp around it (required by MarkTensorIOPass to recognise
    // graph inputs/outputs at lowering time). Wrappers own no weights — the
    // ptq::QDQ calls inside pre/attn/post pass the underlying Decoder/Attn
    // pointer as `this`, so PTQ name resolution is unchanged.

    // -------- chunk_0 ---------------------------------------------------------
    {
      Chunk0Module chunk0("chunk_0", &llm_, this);
      ir::lowlevel::traceStart();
      (void)chunk0(input_ids, position_ids);
      result["chunk_0"] = ir::lowlevel::traceStop();
    }

    // -------- score (one shared, weightless block-selection matmul graph) -----
    // Traced once; the runner dispatches it per layer with that layer's qr/kc.
    // Only present when the compile driver supplies score_qr/score_kc inputs.
    if (input.count("score_qr") && input.count("score_kc")) {
      ScoreModule score("score");
      auto qr = input.at("score_qr");
      auto kc = input.at("score_kc");
      ir::lowlevel::traceStart();
      (void)score(qr, kc);
      result["score"] = ir::lowlevel::traceStop();
    }

    // -------- per-layer: attn_i then chunk_{i+1} -----------------------------
    for (int i = 0; i < L; ++i) {
      auto si = std::to_string(i);

      // attn_i (per-qb dispatch)
      {
        AttnChunkModule attn_mod("attn_" + si, &llm_, i);
        auto q_qb = input.at("q_" + si + "_qb");
        auto kc_qb = input.at("k_curr_" + si + "_qb");
        auto vc_qb = input.at("v_curr_" + si + "_qb");
        auto K_arr = input.at("K_arranged_" + si);
        auto V_arr = input.at("V_arranged_" + si);
        ir::lowlevel::traceStart();
        (void)attn_mod(q_qb, kc_qb, vc_qb, K_arr, V_arr, mask);
        result["attn_" + si] = ir::lowlevel::traceStop();
      }

      // chunk_{i+1}: post_i + (pre_{i+1} if not last, else final norm + lm_head)
      auto residual = input.at("residual_pre_attn_" + si);
      auto attn_output = input.at("attn_output_" + si);
      if (i + 1 < L) {
        MidChunkModule mid("chunk_" + std::to_string(i + 1), &llm_, i);
        ir::lowlevel::traceStart();
        (void)mid(residual, attn_output, position_ids);
        result["chunk_" + std::to_string(i + 1)] = ir::lowlevel::traceStop();
      } else {
        FinalChunkModule fin("chunk_" + std::to_string(i + 1), &llm_, this, &lm_head_, cfg_.hidden_size, i);
        auto last_token_index = input.at("last_token_index");
        ir::lowlevel::traceStart();
        (void)fin(residual, attn_output, last_token_index);
        result["chunk_" + std::to_string(i + 1)] = ir::lowlevel::traceStop();
      }
    }

    return result;
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

 private:
  const Qwen3Config& cfg_;
  Qwen3TextSplit llm_;
  nn::Conv2D lm_head_;
  bool tie_word_embeddings_;
};

#undef BSC_CONV2D_PROPERTY

}  // namespace mllm::models::qwen3::sha_blocksparse_causal_split
