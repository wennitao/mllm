// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3 NPU prefill + CPU decode.
// Prefill runs on NPU using the proven SHA .bin (from compile_sha.cpp).
// PromptProcessor handles all I/O setup, masking, and KV management exactly
// as the working aot_run.cpp does. KV outputs are then handed to CPU decode.

#include <chrono>
#include <fmt/core.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <mllm/mllm.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTModule.hpp>
#include <mllm/backends/qnn/aot_rt/KVCacheManager.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp>
#include <mllm/backends/qnn/aot_rt/PromptProcessor.hpp>
#include <mllm/models/qwen3/modeling_qwen3.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include <mllm/preprocessor/tokenizers/Unicode.hpp>
#include <mllm/utils/Log.hpp>

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

namespace {

auto parseModelVersion(const std::string& v) -> mllm::ModelFileVersion {
  if (v == "v1") return mllm::ModelFileVersion::kV1;
  if (v == "v2") return mllm::ModelFileVersion::kV2;
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Unsupported model_version: {}", v);
}

auto argmaxF32(mllm::Tensor logits) -> int64_t {
  auto t    = logits.to(mllm::kFloat32);
  auto* data = t.ptr<float>();
  int64_t vocab = t.shape().back();
  return std::distance(data, std::max_element(data, data + vocab));
}

auto tokenTensor(int64_t token_id) -> mllm::Tensor {
  auto token = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
  token.at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(token_id);
  return token;
}

}  // namespace

MLLM_MAIN({
  auto& help           = Argparse::add<bool>("-h|--help").help("Show help");
  auto& npu_bin_path   = Argparse::add<std::string>("--npu_bin").help("Pre-compiled NPU SHA .bin").required(true);
  auto& cpu_model_path = Argparse::add<std::string>("--cpu_model").help("CPU decode model weights (.mllm)").required(true);
  auto& config_path    = Argparse::add<std::string>("-c|--config").help("Model config JSON").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").required(true);
  auto& prefill_len    = Argparse::add<int>("--prefill_len").help("Prefill chunk size (must match SHA .bin)").def(32);
  auto& max_new_tokens = Argparse::add<int>("--max_new_tokens").help("Max decode steps").def(512);
  auto& model_version  = Argparse::add<std::string>("-mv|--model_version").def("v2");

  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  auto file_version = parseModelVersion(model_version.get());
  auto cfg          = mllm::models::qwen3::Qwen3Config(config_path.get());
  auto tokenizer    = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  // -----------------------------------------------------------------------
  // 1. Load NPU SHA .bin and initialize PromptProcessor
  // -----------------------------------------------------------------------
  mllm::initQnnBackend(npu_bin_path.get());

  QnnAOTConfig npu_cfg;
  npu_cfg.num_layers  = cfg.num_hidden_layers;
  npu_cfg.num_heads   = cfg.num_key_value_heads;
  npu_cfg.head_dim    = cfg.head_dim;
  npu_cfg.vocab_size  = cfg.vocab_size;
  npu_cfg.context_len = cfg.max_cache_length;
  npu_cfg.ar_len      = prefill_len.get();
  npu_cfg.kv_dtype    = mllm::kUInt8;

  // KVCacheManager owns the KV cache buffers (same as aot_run.cpp)
  auto backend    = mllm::Context::instance().getBackend(mllm::kQNN);
  auto kv_manager = std::make_unique<KVCacheManager<uint8_t>>(npu_cfg);
  kv_manager->initCache(backend->allocator().get(), npu_cfg.ar_len);

  auto prompt_processor = std::make_unique<PromptProcessor<uint8_t>>(kv_manager.get(), npu_cfg);
  prompt_processor->init_io();

  MLLM_INFO("NPU SHA prefill ready (chunk={}, ctx={}).", npu_cfg.ar_len, npu_cfg.context_len);

  // -----------------------------------------------------------------------
  // 2. Load CPU decode model
  // -----------------------------------------------------------------------
  auto cpu_params = mllm::load(cpu_model_path.get(), file_version);
  auto cpu_model  = mllm::models::qwen3::Qwen3ForCausalLM(cfg);
  cpu_model.load(cpu_params);
  MLLM_INFO("CPU decode model loaded.");

  // -----------------------------------------------------------------------
  // 3. Interactive loop
  // -----------------------------------------------------------------------
  fmt::print("\n{:*^60}\n", " Qwen3 NPU-prefill + CPU-decode ");
  fmt::print("Type 'exit' or 'quit' to quit.\n\n");

  while (true) {
    fmt::print("Prompt: ");
    std::string prompt_text;
    std::getline(std::cin, prompt_text);
    if (prompt_text == "exit" || prompt_text == "quit") break;
    if (prompt_text.empty()) continue;

    auto raw_tokens = tokenizer.convertMessage({.prompt = prompt_text})["sequence"];
    std::vector<int64_t> prompt_tokens;
    prompt_tokens.reserve(raw_tokens.shape()[1]);
    for (int i = 0; i < (int)raw_tokens.shape()[1]; ++i)
      prompt_tokens.push_back(raw_tokens.ptr<int64_t>()[i]);
    MLLM_INFO("Input tokens: {}", (int)prompt_tokens.size());

    // ------------------------------------------------------------------
    // 4. NPU prefill via PromptProcessor (exactly as aot_run.cpp)
    // ------------------------------------------------------------------
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    int64_t first_token_id = prompt_processor->prefill(prompt_tokens, /*start_pos=*/0);
    auto t_prefill_end = std::chrono::high_resolution_clock::now();

    double prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            t_prefill_end - t_prefill_start).count();
    MLLM_INFO("NPU prefill: {:.2f}s, first token={}", prefill_ms / 1000.0, first_token_id);

    std::string first_token_str = mllm::preprocessor::wideString2Utf8String(
        tokenizer.detokenize(first_token_id));
    MLLM_INFO("First token: {} ({})", first_token_str, first_token_id);
    std::cout << first_token_str << std::flush;

    // ------------------------------------------------------------------
    // 5. CPU prefill (baseline + KV cache population)
    //    Run the full prompt through the CPU model to:
    //      a) get the "golden" first token for NPU correctness verification
    //      b) populate the CPU KV cache so decode has full context
    // ------------------------------------------------------------------
    cpu_model.kvCache().clearCache();

    mllm::models::ARGenerationOutputPast cpu_prefill_past;
    cpu_prefill_past["sequence"] = raw_tokens;  // full prompt, int64 [1, seq_len]

    mllm::models::ARGenerationArgs decode_args;

    auto t_cpu_prefill_start = std::chrono::high_resolution_clock::now();
    auto cpu_prefill_out = cpu_model.forward(cpu_prefill_past, decode_args);
    auto t_cpu_prefill_end = std::chrono::high_resolution_clock::now();
    double cpu_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                t_cpu_prefill_end - t_cpu_prefill_start).count();

    int64_t cpu_first_token_id = argmaxF32(cpu_prefill_out.at("sequence"));
    std::string cpu_first_token_str = mllm::preprocessor::wideString2Utf8String(
        tokenizer.detokenize(cpu_first_token_id));

    MLLM_INFO("CPU prefill:  {:.2f}s, first token={} ({})", cpu_prefill_ms / 1000.0,
              cpu_first_token_id, cpu_first_token_str);
    MLLM_INFO("NPU prefill:  first token={} ({})", first_token_id, first_token_str);
    if (cpu_first_token_id == first_token_id) {
      MLLM_INFO("NPU first token MATCHES CPU. Prefill correct.");
    } else {
      MLLM_INFO("NPU first token MISMATCH. NPU={} CPU={}", first_token_id, cpu_first_token_id);
    }

    // Continue decode from CPU KV cache. The CPU prefill output contains
    // logits, so replace "sequence" with the sampled first token before the
    // first decode step.
    auto past = std::move(cpu_prefill_out);
    past["sequence"] = tokenTensor(first_token_id);

    double total_decode_ms = 0.0;
    int    decode_count    = 0;

    MLLM_INFO("Starting CPU decode from token={}...", first_token_id);
    for (int step = 1; step < max_new_tokens.get(); ++step) {
      auto t0    = std::chrono::high_resolution_clock::now();
      auto output = cpu_model.forward(past, decode_args);
      auto t1    = std::chrono::high_resolution_clock::now();
      total_decode_ms += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
      decode_count++;

      auto next_id = argmaxF32(output["sequence"]);
      std::cout << mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(next_id))
                << std::flush;

      past = std::move(output);
      past["sequence"] = tokenTensor(next_id);

      if (next_id == cfg.eos_token_id) break;
    }

    std::cout << "\n";
    if (decode_count > 0) {
      MLLM_INFO("Decode: {} tokens, avg {:.2f}ms/tok ({:.2f} tok/s)",
                decode_count, total_decode_ms / decode_count,
                decode_count / (total_decode_ms / 1000.0));
    }
  }  // end while(true)

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3_npu_prefill.perfetto");
#endif

  mllm::shutdownContext();
  return 0;
})
