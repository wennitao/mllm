// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Runner for the LPBQ SHA per-qb CAUSAL block-sparse Qwen3 model.
// Mirrors aot_run_blocksparse_causal.cpp but with the SHA / uint8 KV cache
// pipeline (ShaBlockSparsePromptProcessor).

#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <string>
#include "mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("qwen3-lpbq-sha-blocksparse-causal.bin");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& max_new_tokens = Argparse::add<int>("--max_new_tokens").help("Max new tokens to generate").def(128);

  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int kMaxNew = max_new_tokens.get();

  mllm::initQnnBackend(model_path.get());

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  QnnAOTConfig config;
  config.num_layers          = qwen3_cfg.num_hidden_layers;
  config.num_heads           = qwen3_cfg.num_key_value_heads;
  config.num_attention_heads = qwen3_cfg.num_attention_heads;
  config.head_dim            = qwen3_cfg.head_dim;
  config.vocab_size          = qwen3_cfg.vocab_size;
  config.context_len         = qwen3_cfg.max_cache_length;
  config.ar_len              = ShaBlockSparsePromptProcessor::kBQ;
  config.kv_dtype            = mllm::kUInt8;

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);

  auto input_tensor = tokenizer.convertMessage({.prompt = prompt_text});
  auto& seq = input_tensor["sequence"];
  std::vector<int64_t> prompt_tokens;
  prompt_tokens.reserve(seq.shape()[1]);
  for (int i = 0; i < seq.shape()[1]; ++i) prompt_tokens.push_back(seq.ptr<int64_t>()[i]);

  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  KVCacheManager<uint8_t> kv(config);
  kv.initCache(backend->allocator().get(), config.ar_len);

  ShaBlockSparsePromptProcessor proc(&kv, config);
  proc.init_io();

  auto prefill_start = std::chrono::high_resolution_clock::now();
  int64_t first_token = proc.prefill(prompt_tokens, /*start_pos=*/0);
  auto prefill_end = std::chrono::high_resolution_clock::now();
  auto prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end - prefill_start).count();
  fmt::print("\nPrefill: {} tokens in {} µs ({:.2f} tokens/s)\n", prompt_tokens.size(), prefill_us,
             (double)prompt_tokens.size() / ((double)prefill_us / 1e6));

  // Build the running token history. Prefill produced `first_token` from
  // logits but didn't write it to the cache (it was just a sample). The decode
  // loop's first step will compute its KV and append it to the cache, then
  // sample the next token.
  std::vector<int64_t> all_tokens = prompt_tokens;
  all_tokens.push_back(first_token);

  // Print the first token immediately.
  fmt::print("\n=== Decode ===\n");
  {
    auto s = tokenizer.detokenize(first_token);
    std::wcout << s << std::flush;
  }

  auto decode_start = std::chrono::high_resolution_clock::now();
  int decoded = 0;
  proc.decode(all_tokens, kMaxNew, qwen3_cfg.end_of_text_token_id, [&](int64_t tok) {
    auto s = tokenizer.detokenize(tok);
    std::wcout << s << std::flush;
    decoded++;
  });
  auto decode_end = std::chrono::high_resolution_clock::now();
  auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();

  fmt::print("\n\nDecode: {} new tokens in {} µs ({:.2f} tokens/s)\n", decoded, decode_us,
             decoded > 0 ? (double)decoded / ((double)decode_us / 1e6) : 0.0);

  mllm::shutdownContext();
  return 0;
});
