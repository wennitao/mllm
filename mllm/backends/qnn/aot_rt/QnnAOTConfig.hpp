// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"

namespace mllm::qnn::aot {

struct QnnAOTConfig {
  int num_layers = 28;
  int num_heads = 12;             // num_key_value_heads (the standard PromptProcessor used this for both Q and KV; kept for backwards compat)
  int num_attention_heads = 12;   // Hq for GQA — distinct from num_heads (Hkv) when Hq > Hkv. Defaults to num_heads (MHA).
  int head_dim = 128;
  int vocab_size = 151936;

  int context_len = 4096;
  int ar_len = 128;  // Chunk size for prefill
  int sliding_window = 0;

  // Derived/Computed
  int max_ar_len = 128;
  int max_cache_len = 4096;

  DataTypes kv_dtype = kUInt8;
  bool use_int64_token = true;
};

}  // namespace mllm::qnn::aot
