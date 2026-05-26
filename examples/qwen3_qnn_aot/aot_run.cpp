#include <chrono>
#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <string>
#include "mllm/utils/Log.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("Model path").def("qwen3_qnn.mllm");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& ar_len = Argparse::add<int>("--ar_len").help("Autoregressive length (chunk size)").def(128);
  auto& context_len_arg = Argparse::add<int>("--context_len").help("Max KV cache length; MUST match the .bin's compile-time CL").def(1024);
  auto& seq_len_arg = Argparse::add<int>("--seq_len").help("If >0, use synthetic prefill of exactly N tokens (bypass tokenizer/template)").def(-1);
  auto& max_new_tokens = Argparse::add<int>("--max_new_tokens").help("Cap decode steps (<=0 = unlimited up to context_len)").def(-1);
  auto& quiet = Argparse::add<bool>("--quiet").help("Disable per-op profiling output (no CSV, no perfSummary)");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (quiet.isSet()) {
    mllm::Logger::level() = mllm::LogLevel::kWarn;
  }

  // mllm::initializeContext();
  mllm::initQnnBackend(model_path.get());

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());

  RunnerConfig config;
  config.num_layers = qwen3_cfg.num_hidden_layers;
  config.num_heads = qwen3_cfg.num_key_value_heads;
  config.head_dim = qwen3_cfg.head_dim;
  config.vocab_size = qwen3_cfg.vocab_size;
  config.context_len = context_len_arg.get();
  config.ar_len = ar_len.get();

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  std::string prompt_text;
  if (seq_len_arg.get() <= 0) {
    fmt::print("💬 Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);
  }

  #ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
  #endif

  mllm::models::ARGenerationOutputPast input_tensor;
  if (seq_len_arg.get() > 0) {
    int n = seq_len_arg.get();
    auto seq = mllm::Tensor::empty({1, n}, mllm::kInt64, mllm::kCPU).alloc();
    auto* p = seq.ptr<int64_t>();
    for (int i = 0; i < n; ++i) p[i] = 100;  // arbitrary safe text token
    input_tensor = {{"sequence", seq}};
    fmt::print("(synthetic input: {} tokens)\n", n);
  } else {
    input_tensor = tokenizer.convertMessage({.prompt = prompt_text});
  }

  Runner runner(config, &tokenizer);
  if (!runner.load()) {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  int decode_cap = (max_new_tokens.get() > 0) ? max_new_tokens.get() : config.context_len;
  auto t_e2e_start = std::chrono::high_resolution_clock::now();
  runner.generate(input_tensor["sequence"], decode_cap,
                  [](const std::string& token) { std::cout << token << std::flush; },
                  /*perf=*/true);  // always print prefill/decode summary
  auto t_e2e_end = std::chrono::high_resolution_clock::now();
  auto e2e_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_e2e_end - t_e2e_start).count();
  fmt::print("\n[E2E] generate() wall time = {} ms\n", e2e_ms);
  std::cout << "\n";

  #ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3.perfetto");
  #endif

  mllm::shutdownContext();

  return 0;
});
