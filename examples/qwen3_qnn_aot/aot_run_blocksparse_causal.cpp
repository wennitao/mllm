// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Runner for the fp16 PER-QB causal block-sparse Qwen3 model.
//
// Mirrors aot_run.cpp but uses BlockSparsePromptProcessor instead of the
// standard PromptProcessor:
//   * Per-dispatch shape = BQ = 32 tokens (one q-block per graphExecute)
//   * CPU-side per-qb gather of K_arranged/V_arranged
//   * Per-qb padding mask (qb-dependent)
//   * Static causal triangle mask baked into the compiled graph
//
// Usage:
//   ./mllm-qwen3-aot-blocksparse-runner -m qwen3-fp16-blocksparse-causal.bin \
//                                       -c config_1.7B_fp16_blocksparse_causal.json \
//                                       -t tokenizer.json
//
// NOTE: the per-qb gather logic in BlockSparsePromptProcessor is currently
// a STUB — graph executes but K_arranged stays zero. Functional correctness
// requires implementing the gather + diagonal-slot handling (see TODOs in
// BlockSparsePromptProcessor.cpp).

#include <cstdio>
#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <string>
#include "mllm/backends/qnn/aot_rt/BlockSparsePromptProcessor.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("qwen3-fp16-blocksparse-causal.bin");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(model_path.get());

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  QnnAOTConfig config;
  config.num_layers          = qwen3_cfg.num_hidden_layers;
  config.num_heads           = qwen3_cfg.num_key_value_heads;
  config.num_attention_heads = qwen3_cfg.num_attention_heads;
  config.head_dim            = qwen3_cfg.head_dim;
  config.vocab_size          = qwen3_cfg.vocab_size;
  config.context_len         = qwen3_cfg.max_cache_length;
  config.ar_len              = BlockSparsePromptProcessor::kBQ;   // one qb per dispatch
  config.kv_dtype            = mllm::kFloat16;                    // fp16 KV

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);

  auto input_tensor = tokenizer.convertMessage({.prompt = prompt_text});
  auto& seq = input_tensor["sequence"];

  // Qwen3Tokenizer::convertMessage emits an int64 sequence tensor.
  std::vector<int64_t> prompt_tokens;
  prompt_tokens.reserve(seq.shape()[1]);
  for (int i = 0; i < seq.shape()[1]; ++i) prompt_tokens.push_back(seq.ptr<int64_t>()[i]);

  // KV cache (fp16 stored as uint16 — bit-equivalent).
  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  KVCacheManager<uint16_t> kv(config);
  kv.initCache(backend->allocator().get(), config.ar_len);

  BlockSparsePromptProcessor proc(&kv, config);
  proc.init_io();

  auto prefill_start = std::chrono::high_resolution_clock::now();
  int64_t next_token = proc.prefill(prompt_tokens, /*start_pos=*/0);
  auto prefill_end = std::chrono::high_resolution_clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end - prefill_start).count();
  fmt::print("\nPrefill complete: {} tokens in {} µs ({:.2f} tokens/s)\n", prompt_tokens.size(), us,
             (double)prompt_tokens.size() / ((double)us / 1e6));
  fmt::print("Next token (greedy): {}\n", next_token);

  // (Decoding not implemented in this initial runner — would require a
  // separate per-token graph or a different ar_len. The per-qb prefill
  // graph processes BQ=32 tokens per dispatch, not a single token.)

  mllm::shutdownContext();
  return 0;
});
