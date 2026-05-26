#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/modeling_qwen3_fa2.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include <mllm/utils/AnyValue.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& perf_path = Argparse::add<std::string>("--perf_path").help("Perfetto trace output path").def("qwen3.perf");
  auto& max_new_tokens = Argparse::add<int>("--max_new_tokens").help("Cap decode steps (<=0 = unlimited)").def(-1);
  auto& seq_len = Argparse::add<int>("--seq_len").help("If >0, use synthetic prefill of exactly N tokens (bypass tokenizer/template)").def(-1);

  Argparse::parse(argc, argv);

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  } else {
    fmt::print("❌ Unsupported model_version: {} (expected v1 or v2)\n", model_version.get());
    mllm::shutdownContext();
    return 1;
  }

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  {
    auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());
    auto qwen3_tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());
    auto qwen3 = mllm::models::qwen3::Qwen3ForCausalLM(qwen3_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen3.load(param);

    fmt::print("\n{:*^60}\n", " Qwen3 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;

    fmt::print("💬 Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);

    try {
      fmt::print("🔄 Processing...\n");
      mllm::models::ARGenerationOutputPast inputs;
      int sl = seq_len.get();
      if (sl > 0) {
        // Synthetic prefill of exactly N tokens (avoid chat-template overhead).
        auto seq = mllm::Tensor::empty({1, sl}, mllm::kInt64, mllm::kCPU).alloc();
        auto* p = seq.ptr<int64_t>();
        for (int i = 0; i < sl; ++i) p[i] = 100;  // arbitrary safe text token
        inputs = {{"sequence", seq}};
        fmt::print("(synthetic input: {} tokens)\n", sl);
      } else {
        inputs = qwen3_tokenizer.convertMessage({.prompt = prompt_text});
      }

      fmt::print("\n🤖 Response: ");

      // Use for loop
      int decode_count = 0;
      int cap = max_new_tokens.get();
      for (auto& step : qwen3.chat(inputs)) {
        std::wcout << qwen3_tokenizer.detokenize(step.cur_token_id) << std::flush;
        if (cap > 0 && ++decode_count >= cap) {
          qwen3.decodeEventEndTimePoint();
          break;
        }
      }

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
    
    qwen3.perfSummary();
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport(perf_path.get());
#endif

  mllm::print("\n");
  mllm::memoryReport();
})
