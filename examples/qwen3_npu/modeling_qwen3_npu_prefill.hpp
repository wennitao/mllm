// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Prefill-only NPU model for Qwen3, using Single Head Attention (SHA).
// SHA splits Q/K/V projections per-head (like compile_sha.cpp), which is the
// only proven-correct approach on QNN HTP.
//
// Graph inputs:  sequence [1,N] int32, causal_mask [1,1,N,N] uint16
// Graph outputs: logits [1,1,N,vocab], key_l [1,Hkv,D,N], val_l [1,Hkv,N,D]
//                for l in [0, num_hidden_layers)

#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::models::qwen3_npu_prefill {

using namespace mllm;         // NOLINT
using namespace mllm::nn;     // NOLINT
using namespace mllm::models; // NOLINT
using mllm::models::qwen3::Qwen3Config;

// ---------------------------------------------------------------------------
// PTQ helpers (mirrors sha:: namespace)
// ---------------------------------------------------------------------------
namespace ptq {

inline Tensor QDQ_CONSTANT(nn::Module* m, Tensor in, const std::string& qdq_name) {
  auto scale = m->getTopParameterFile()->pull(qdq_name + ".scale");
  auto zp    = m->getTopParameterFile()->pull(qdq_name + ".zero_point");
  in.attach("scale", scale.impl(), true);
  in.attach("zero_point", zp.impl(), true);
  return in;
}

inline Tensor QDQ(nn::Module* m, Tensor in, const std::string& qdq_name) {
  std::string sn, zn;
  if (m->getModuleName().empty()) {
    sn = qdq_name + ".fake_quant.scale";
    zn = qdq_name + ".fake_quant.zero_point";
  } else {
    sn = m->getModuleName() + "." + qdq_name + ".fake_quant.scale";
    zn = m->getModuleName() + "." + qdq_name + ".fake_quant.zero_point";
  }
  in.attach("scale", m->getTopParameterFile()->pull(sn).impl(), true);
  in.attach("zero_point", m->getTopParameterFile()->pull(zn).impl(), true);
  return in;
}

inline Tensor QDQ_KV(nn::Module* m, Tensor in, const std::string& qdq_name) {
  auto sn = m->getModuleName() + "." + qdq_name + ".fake_quant.scale";
  auto zn = m->getModuleName() + "." + qdq_name + ".fake_quant.zero_point";
  auto scale = m->getTopParameterFile()->pull(sn);
  auto zp    = m->getTopParameterFile()->pull(zn);
  MLLM_RT_ASSERT_EQ(zp.item<mllm_int32_t>(), 128);
  auto new_zp = Tensor::constant(128, kInt32).setName(zn).setMemType(kParamsNormal);
  in.attach("scale", scale.impl(), true);
  in.attach("zero_point", new_zp.impl(), true);
  return in;
}

inline Tensor QDQ_ROPE(nn::Module* m, Tensor in, const std::string& qdq_name) {
  auto sn = m->getModuleName() + "." + qdq_name + ".fake_quant.scale";
  auto zn = m->getModuleName() + "." + qdq_name + ".fake_quant.zero_point";
  (void)in.__unsafeSetDType(kUInt16PerTensorAsy);
  in.attach("scale", m->getTopParameterFile()->pull(sn).impl(), true);
  in.attach("zero_point", m->getTopParameterFile()->pull(zn).impl(), true);
  return in;
}

}  // namespace ptq

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
using vi32 = std::vector<int32_t>;
#define CONV2D_P vi32{1,1}, vi32{1,1}, vi32{0,0}, vi32{1,1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G16

inline Tensor rotateHalf(Tensor x, nn::Module* m, const std::string& qdq_name) {
  auto D  = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, true);
  return nn::functional::concat({ptq::QDQ(m, -x2, qdq_name), x1}, -1);
}

// ---------------------------------------------------------------------------
// MLP (unchanged from MHA version)
// ---------------------------------------------------------------------------
class Qwen3MLPPrefill final : public nn::Module {
  nn::Conv2D gate_proj_, up_proj_, down_proj_;
  nn::SiLU silu_;
  int hidden_size_, intermediate_size_;

 public:
  Qwen3MLPPrefill() = default;
  Qwen3MLPPrefill(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Conv2D>("gate_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_P);
    silu_ = reg<nn::SiLU>("act");
    up_proj_   = reg<nn::Conv2D>("up_proj",   cfg.hidden_size, cfg.intermediate_size, CONV2D_P);
    down_proj_ = reg<nn::Conv2D>("down_proj", cfg.intermediate_size, cfg.hidden_size, CONV2D_P);
    hidden_size_       = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x = ptq::QDQ(this, inputs[0], "up_proj_input_qdq");
    x = x.view({1, 1, -1, hidden_size_}, true);
    auto up   = ptq::QDQ(this, up_proj_(x),  "up_proj_output_qdq"  ).view({1, -1, intermediate_size_}, true);
    auto gate = ptq::QDQ(this, gate_proj_(x), "gate_proj_output_qdq").view({1, -1, intermediate_size_}, true);
    gate = ptq::QDQ(this, gate * ptq::QDQ(this, nn::functional::sigmoid(gate), "sigmoid_output_qdq"), "act_output_qdq");
    auto o = ptq::QDQ(this, gate * up, "down_proj_input_qdq");
    o = o.view({1, 1, -1, intermediate_size_}, true);
    o = down_proj_(o).view({1, -1, hidden_size_}, true);
    return {o};
  }
};

// ---------------------------------------------------------------------------
// SHA Attention — prefill only (no past KV inputs, no KV concat)
//
// Per-head Q/K/V Conv2D projections, per-head RMSNorm+RoPE, per-head attention.
// causal_mask shape: [1,1,N,N] (square, matches prefill attention shape exactly).
//
// Outputs: [attn_output, key [B,Hkv,D,N], value [B,Hkv,N,D]]
// ---------------------------------------------------------------------------
class Qwen3AttentionSHAPrefill final : public nn::Module {
  std::vector<nn::Conv2D>  q_projs_;   // num_attention_heads
  std::vector<nn::Conv2D>  k_projs_;   // num_key_value_heads
  std::vector<nn::Conv2D>  v_projs_;   // num_key_value_heads
  nn::Conv2D               o_proj_;
  std::vector<nn::RMSNorm> rms_norm_q_;  // num_attention_heads
  std::vector<nn::RMSNorm> rms_norm_k_;  // num_key_value_heads

  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int   hidden_size_, head_dim_, num_attention_heads_, num_key_value_heads_, num_key_value_groups_;
  float scale_;

 public:
  int layer_idx_ = 0;

  Qwen3AttentionSHAPrefill() = default;
  Qwen3AttentionSHAPrefill(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_          = cfg.hidden_size;
    num_attention_heads_  = cfg.num_attention_heads;
    num_key_value_heads_  = cfg.num_key_value_heads;
    head_dim_             = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_                = 1.f / sqrtf(static_cast<float>(head_dim_));

    for (int h = 0; h < num_attention_heads_; ++h)
      q_projs_.emplace_back(reg<nn::Conv2D>("q_proj." + std::to_string(h), hidden_size_, head_dim_, CONV2D_P));
    for (int h = 0; h < num_key_value_heads_; ++h)
      k_projs_.emplace_back(reg<nn::Conv2D>("k_proj." + std::to_string(h), hidden_size_, head_dim_, CONV2D_P));
    for (int h = 0; h < num_key_value_heads_; ++h)
      v_projs_.emplace_back(reg<nn::Conv2D>("v_proj." + std::to_string(h), hidden_size_, head_dim_, CONV2D_P));
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, CONV2D_P);
    for (int h = 0; h < num_attention_heads_; ++h)
      rms_norm_q_.emplace_back(reg<nn::RMSNorm>("q_norm." + std::to_string(h), cfg.rms_norm_eps));
    for (int h = 0; h < num_key_value_heads_; ++h)
      rms_norm_k_.emplace_back(reg<nn::RMSNorm>("k_norm." + std::to_string(h), cfg.rms_norm_eps));
  
    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  // inputs: [hidden_states, sin, cos, causal_mask]
  // outputs: [attn_output, key [B,Hkv,D,N], value [B,Hkv,N,D]]
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto hidden_states = inputs[0];
    auto sin           = inputs[1];
    auto cos           = inputs[2];
    auto causal_mask   = inputs[3];

    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // Per-head Q/K/V projections: each output [B,1,N,head_dim]
    std::vector<Tensor> q_per_head, k_per_head, v_per_head;
    for (int h = 0; h < num_attention_heads_; ++h)
      q_per_head.push_back(q_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, true));
    for (int h = 0; h < num_key_value_heads_; ++h)
      k_per_head.push_back(k_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, true));
    for (int h = 0; h < num_key_value_heads_; ++h)
      v_per_head.push_back(v_projs_[h](hidden_states).view({1, 1, -1, head_dim_}, true));

    auto cos_u = cos.unsqueeze(1, true);  // [B,1,N,D]
    auto sin_u = sin.unsqueeze(1, true);  // [B,1,N,D]

    // Per-Q-head RMSNorm + RoPE
    for (int h = 0; h < num_attention_heads_; ++h) {
      const auto hs = std::to_string(h);
      q_per_head[h] = rms_norm_q_[h](ptq::QDQ(this, q_per_head[h], "q_norm_input_qdq_h" + hs));
      q_per_head[h] = ptq::QDQ(this, q_per_head[h], "q_norm_output_qdq_h" + hs);
      q_per_head[h] = ptq::QDQ(this,
          ptq::QDQ(this, q_per_head[h] * cos_u, "q_rope_mul_0_output_qdq_h" + hs)
          + ptq::QDQ(this, rotateHalf(q_per_head[h], this, "q_rope_neg_half_qdq_h" + hs) * sin_u,
                     "q_rope_mul_1_output_qdq_h" + hs),
          "q_rope_add_0_output_qdq_h" + hs);
    }

    // Per-KV-head RMSNorm + RoPE + quantize K/V for cache output
    std::vector<Tensor> new_key_per_head, new_val_per_head;
    for (int h = 0; h < num_key_value_heads_; ++h) {
      const auto hs = std::to_string(h);
      k_per_head[h] = rms_norm_k_[h](ptq::QDQ(this, k_per_head[h], "k_norm_input_qdq_h" + hs));
      k_per_head[h] = ptq::QDQ(this, k_per_head[h], "k_norm_output_qdq_h" + hs);
      k_per_head[h] = ptq::QDQ(this,
          ptq::QDQ(this, k_per_head[h] * cos_u, "k_rope_mul_0_output_qdq_h" + hs)
          + ptq::QDQ(this, rotateHalf(k_per_head[h], this, "k_rope_neg_half_qdq_h" + hs) * sin_u,
                     "k_rope_mul_1_output_qdq_h" + hs),
          "k_rope_add_0_output_qdq_h" + hs);

      // K → uint8 sym, transpose to [B,1,D,N] (matches past_key layout for CPU decode)
      auto k_int8 = k_per_head[h].to(kUInt8PerTensorSym);
      k_int8 = ptq::QDQ_KV(this, k_int8, "k_cast_to_int8_qdq_h" + hs);
      k_int8 = k_int8.transpose(2, 3);  // [B,1,D,N]
      new_key_per_head.push_back(k_int8);

      // V → uint8 sym, keep [B,1,N,D]
      auto v_int8 = ptq::QDQ(this, v_per_head[h], "v_cast_to_int16_qdq_h" + hs);
      v_int8 = v_int8.to(kUInt8PerTensorSym);
      v_int8 = ptq::QDQ_KV(this, v_int8, "v_cast_to_int8_qdq_h" + hs);
      new_val_per_head.push_back(v_int8);

      // Use quantized+transposed K for attention QK^T
      k_per_head[h] = k_int8;  // [B,1,D,N]
    }

    // Per-Q-head attention
    // q [B,1,N,D] @ k [B,1,D,N] → attn [B,1,N,N]
    // causal_mask [1,1,N,N] — matches attn exactly (no slice needed)
    // attn @ v [B,1,N,D] → y [B,1,N,D]
    std::vector<Tensor> attn_outputs;
    for (int h = 0; h < num_attention_heads_; ++h) {
      const auto hs   = std::to_string(h);
      const int  kv_h = h / num_key_value_groups_;

      auto attn = ptq::QDQ(this, nn::functional::matmul(q_per_head[h], k_per_head[kv_h]),
                           "qk_matmul_output_qdq_h" + hs);
      auto scale_t = Tensor::constant(scale_, kFloat32);
      scale_t = ptq::QDQ(this, scale_t, "scaling_qdq_h" + hs);
      attn = ptq::QDQ(this, attn.mulConstant(scale_t), "mul_0_output_qdq_h" + hs);

      auto attn_min  = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq_h" + hs);
      auto minus_val = Tensor::constant(-20, kFloat32);
      minus_val = ptq::QDQ(this, minus_val, "neg_20_qdq_h" + hs);
      auto attn_vv   = ptq::QDQ(this, attn_min.addConstant(minus_val), "minus_0_output_qdq_h" + hs);
      auto zero_c    = Tensor::constant(0.f, kFloat32);
      zero_c = ptq::QDQ_CONSTANT(this, zero_c, "constant_zero");
      attn = nn::functional::where(causal_mask.equalConstant(zero_c), attn, attn_vv);
      attn = ptq::QDQ(this, attn, "where_attn_qdq_h" + hs);
      attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq_h" + hs);

      auto y_h = ptq::QDQ(this, nn::functional::matmul(attn, new_val_per_head[kv_h]),
                          "attn_value_matmul_output_qdq_h" + hs);
      attn_outputs.push_back(y_h);
    }

    // Concat heads → O projection
    auto y = nn::functional::concat(attn_outputs, 1);  // [B, H, N, D]
    y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, true);
    y = o_proj_(y).view({1, -1, hidden_size_}, true);

    // Concat K/V across KV heads
    auto new_key = nn::functional::concat(new_key_per_head, 1);  // [B, Hkv, D, N]
    auto new_val = nn::functional::concat(new_val_per_head, 1);  // [B, Hkv, N, D]

    return {y, new_key, new_val};
  }
};

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------
class Qwen3DecoderSHAPrefill final : public nn::Module {
 public:
  int                       layer_idx_ = 0;
  Qwen3AttentionSHAPrefill  self_attn_;
  Qwen3MLPPrefill           mlp_;
  nn::RMSNorm               input_layer_norm_;
  nn::RMSNorm               post_attention_layer_norm_;

  Qwen3DecoderSHAPrefill() = default;
  // NOTE: layer_idx is last — ModuleListWithIdx appends it as the final arg.
  Qwen3DecoderSHAPrefill(const std::string& name, const Qwen3Config& cfg, int layer_idx)
      : nn::Module(name) {
    layer_idx_                 = layer_idx;
    self_attn_                 = reg<Qwen3AttentionSHAPrefill>("self_attn", cfg);
    self_attn_.layer_idx_      = layer_idx;
    mlp_                       = reg<Qwen3MLPPrefill>("mlp", cfg);
    input_layer_norm_          = reg<nn::RMSNorm>("input_layernorm",          cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  // inputs: [hidden_states, sin, cos, causal_mask]
  // outputs: [hidden_states, key, value]
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto sin         = inputs[1];
    auto cos         = inputs[2];
    auto causal_mask = inputs[3];

    auto hidden_states = inputs[0];
    if (layer_idx_ != 0) { hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq"); }
    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);

    std::vector<Tensor> attn_in = {hidden_states, sin, cos, causal_mask};
    auto attn_out = self_attn_(attn_in);

    hidden_states = ptq::QDQ(this, residual + ptq::QDQ(this, attn_out[0], "add_0_lhs_input_qdq"),
                             "add_0_output_qdq");
    residual      = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    std::vector<Tensor> mlp_in = {hidden_states};
    hidden_states = mlp_(mlp_in)[0];
    hidden_states = residual + ptq::QDQ(this, hidden_states, "add_1_lhs_input_qdq");

    return {hidden_states, attn_out[1], attn_out[2]};
  }
};

// ---------------------------------------------------------------------------
// Text model
// ---------------------------------------------------------------------------
class Qwen3TextSHAPrefill final : public nn::Module {
  nn::Embedding                              embedding_;
  nn::ModuleListWithIdx<Qwen3DecoderSHAPrefill> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Param   rope_sin_;
  nn::Param   rope_cos_;
  int32_t     num_hidden_layers_;
  int32_t     hidden_size_;

 public:
  Qwen3TextSHAPrefill() = default;
  Qwen3TextSHAPrefill(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_       = cfg.hidden_size;
    embedding_    = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen3DecoderSHAPrefill>>("layers", cfg.num_hidden_layers, cfg);
    norm_     = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");
  }

  // inputs: [token_ids, position_ids, causal_mask]
  // outputs: [hidden_states, k0..k_{L-1}, v0..v_{L-1}]
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>&) override {
    auto x                   = embedding_(inputs[0]);
    const auto& position_ids = inputs[1];
    auto causal_mask         = inputs[2];

    auto sin = nn::functional::gather(ptq::QDQ_ROPE(this, rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto cos = nn::functional::gather(ptq::QDQ_ROPE(this, rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);

    std::vector<Tensor> keys, values;
    for (auto [idx, block] : enumerate(decode_blocks_.list())) {
      std::vector<Tensor> block_in = {x, sin, cos, causal_mask};
      auto out = block(block_in);
      x = out[0];
      keys  .push_back(out[1]);
      values.push_back(out[2]);
    }

    x = norm_(ptq::QDQ(this, x, "norm_input_qdq"));
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto ret = std::vector<Tensor>{x};
    for (auto& k : keys)   ret.push_back(k);
    for (auto& v : values) ret.push_back(v);
    return ret;
  }

  nn::ModuleListWithIdx<Qwen3DecoderSHAPrefill>& decode_blocks() { return decode_blocks_; }
};

// ---------------------------------------------------------------------------
// Top-level prefill model
// ---------------------------------------------------------------------------
class Qwen3ForCausalLMPrefill : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLMPrefill(const Qwen3Config& cfg, int /*prefill_len*/ = 32) : cfg_(cfg) {
    eos_token_id_ = cfg_.end_of_text_token_id;
    llm_     = reg<Qwen3TextSHAPrefill>("model", cfg_);
    lm_head_ = reg<nn::Conv2D>("lm_head", cfg_.hidden_size, cfg_.vocab_size, CONV2D_P);
  }

  // position_ids = [0,1,...,N-1] is built as a kParamsNormal CPU tensor before
  // traceStart() so the QNN AOT lowering treats it as a static constant (not a
  // graph input). This folds gather(rope_sin/cos, position_ids) into static
  // weights baked into the .bin (same pattern as compile_sha.cpp).
  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs&) override {
    auto sequence    = input.at("sequence");
    auto causal_mask = input.at("causal_mask");

    int N = static_cast<int>(sequence.shape()[1]);
    // position_ids stays kNormal — it becomes a dynamic graph input, which is
    // correct: each prefill chunk needs positions [processed..processed+N-1].
    auto position_ids = Tensor::empty({N}, kInt32, kCPU).alloc();
    {
      auto* p = position_ids.ptr<int32_t>();
      for (int i = 0; i < N; ++i) p[i] = i;
    }

    ir::lowlevel::traceStart();

    std::vector<Tensor> llm_inputs = {sequence, position_ids, causal_mask};
    auto llm_out       = llm_(llm_inputs);
    auto hidden_states = llm_out[0];
    auto logits = lm_head_(ptq::QDQ(this, hidden_states, "lm_head_input_qdq"));
    logits = ptq::QDQ(this, logits, "lm_head_output_qdq");

    auto llm_ir = ir::lowlevel::traceStop();
    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast&, const ARGenerationArgs&) override { return {}; }

  Qwen3TextSHAPrefill llm_;

 private:
  const Qwen3Config& cfg_;
  nn::Conv2D lm_head_;
};

#undef CONV2D_P

}  // namespace mllm::models::qwen3_npu_prefill
