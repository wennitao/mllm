// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Runner for the LPBQ SHA per-qb CAUSAL block-sparse Qwen3 model,
// SPLIT-PREFILL variant. Drives ShaBlockSparsePromptProcessorSplit against
// the 2L+1 chunked QNN context emitted by
// mllm-qwen3-aot-sha-blocksparse-causal-split-c.
//
// Decode is NOT implemented yet — this binary measures prefill latency
// against the saved split context and reports the first sampled token.
// To do full token generation, the runner falls back to no-op after prefill.
//
// Usage:
//   ./mllm-qwen3-aot-sha-blocksparse-causal-split-runner \
//       -m /path/to/qwen3-lpbq-sha-blocksparse-causal-split.bin \
//       -t /path/to/tokenizer.json \
//       -c /path/to/config_1.7B_w4a16_blocksparse_causal.json \
//       --sq 1024

#include <iostream>
#include <chrono>
#include <cstdio>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <string>
#include "mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

MLLM_MAIN({
  // Force unbuffered stdout/stderr so MLLM_INFO/MLLM_ERROR lines flush even on
  // abort() — Android adb shell pipes detect stdout as block-buffered by
  // default, swallowing the diagnostic prints right before a SIGABRT.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Path to split context .bin")
                         .def("qwen3-lpbq-sha-blocksparse-causal-split.bin");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Model config json").required(true);
  auto& sq_arg = Argparse::add<int>("--sq").help("Compiled Sq (must match the .bin's --sq)").def(1024);
  auto& gen_arg = Argparse::add<int>("--gen").help("Number of tokens to generate via decode-by-reprefill").def(0);
  auto& params_arg = Argparse::add<std::string>("--params")
                         .help("Path to ptq_lpbq .mllm — enables XAttention score-based block selection")
                         .def("");

  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  // Extract per-layer Q zero-points for XAttention score-based block selection
  // BEFORE initQnnBackend, so the 2.4 GB .mllm is freed before the QNN context
  // (~1.6 GB) + PD reservation load — avoids the memory spike.
  std::vector<int32_t> q_zp;
  std::vector<float> q_scale, k_scale;
  if (!params_arg.get().empty()) {
    auto params = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2);
    const int L = qwen3_cfg.num_hidden_layers;
    q_zp.reserve(L); q_scale.reserve(L); k_scale.reserve(L);
    for (int i = 0; i < L; ++i) {
      std::string p = "model.layers." + std::to_string(i) + ".self_attn.";
      std::string qzp = p + "q_rope_add_0_output_qdq.fake_quant.zero_point";
      std::string qsc = p + "q_rope_add_0_output_qdq.fake_quant.scale";
      std::string ksc = p + "k_cast_to_int8_qdq.fake_quant.scale";
      if (!params->has(qzp) || !params->has(qsc) || !params->has(ksc)) {
        fmt::print("⚠️  missing qdq params for layer {} — score-based selection NOT enabled\n", i);
        q_zp.clear(); q_scale.clear(); k_scale.clear();
        break;
      }
      q_zp.push_back(params->pull(qzp).item<int32_t>());
      q_scale.push_back(params->pull(qsc).item<float>());
      k_scale.push_back(params->pull(ksc).item<float>());
    }
  }  // `params` freed here

  mllm::initQnnBackend(model_path.get());

  QnnAOTConfig config;
  config.num_layers          = qwen3_cfg.num_hidden_layers;
  config.num_heads           = qwen3_cfg.num_key_value_heads;
  config.num_attention_heads = qwen3_cfg.num_attention_heads;
  config.head_dim            = qwen3_cfg.head_dim;
  config.vocab_size          = qwen3_cfg.vocab_size;
  config.context_len         = qwen3_cfg.max_cache_length;
  config.ar_len              = ShaBlockSparsePromptProcessorSplit::kBQ;
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

  const int Sq = sq_arg.get();
  if ((int64_t)prompt_tokens.size() > Sq) {
    fmt::print("Prompt has {} tokens but Sq is only {} — truncating.\n", prompt_tokens.size(), Sq);
    prompt_tokens.resize(Sq);
  }

  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  KVCacheManager<uint8_t> kv(config);
  kv.initCache(backend->allocator().get(), config.ar_len);

  ShaBlockSparsePromptProcessorSplit proc(&kv, config, Sq);
  proc.init_io();

  if (!q_zp.empty()) {
    proc.enableScoreBasedSelection(q_zp, q_scale, k_scale);
    fmt::print("✅ XAttention score-based block selection enabled ({} layers)\n", q_zp.size());
  }

  auto prefill_start = std::chrono::high_resolution_clock::now();
  int64_t first_token = proc.prefill(prompt_tokens, /*start_pos=*/0);
  auto prefill_end = std::chrono::high_resolution_clock::now();
  auto prefill_us = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end - prefill_start).count();
  fmt::print("\nPrefill (split, Sq={}): {} tokens in {} µs ({:.2f} tokens/s)\n", Sq, prompt_tokens.size(),
             prefill_us, (double)prompt_tokens.size() / ((double)prefill_us / 1e6));

  fmt::print("\n=== First sampled token ===\n");
  fmt::print("token_id={} -> ", first_token);
  {
    auto s = tokenizer.detokenize(first_token);
    std::wcout << s << std::flush;
  }
  fmt::print("\n");

  // ---- decode-by-reprefill -------------------------------------------------
  // The split context is compiled for a fixed Sq, so we don't have an ar_len=1
  // decode graph. As a validation tactic, append each sampled token to the
  // running prompt and re-run the full prefill; sample the next position.
  // This is ~Sq×slower than a proper decode but proves the model is
  // generating coherent prompt-dependent output.
  const int kMaxNew = gen_arg.get();
  if (kMaxNew > 0) {
    fmt::print("\n=== Decode (by reprefill, {} tokens) ===\n", kMaxNew);
    std::vector<int64_t> all_tokens = prompt_tokens;
    all_tokens.push_back(first_token);
    auto decode_start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < kMaxNew; ++step) {
      if ((int64_t)all_tokens.size() >= Sq) {
        fmt::print("\n[hit Sq={} limit at step {}]\n", Sq, step);
        break;
      }
      int64_t next_token = proc.prefill(all_tokens, /*start_pos=*/0);
      all_tokens.push_back(next_token);
      auto s = tokenizer.detokenize(next_token);
      fmt::print("[{}] ", next_token);
      std::wcout << s << std::flush;
    }
    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();
    fmt::print("\n\nDecode-by-reprefill: {} tokens in {} µs ({:.2f} ms/token)\n", kMaxNew, decode_us,
               (double)decode_us / 1000.0 / kMaxNew);
  }

  mllm::shutdownContext();
  return 0;
});
