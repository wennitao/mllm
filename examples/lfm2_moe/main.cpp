#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/lfm2_moe/configuration_lfm2_moe.hpp>
#include <mllm/models/lfm2_moe/modeling_lfm2_moe.hpp>
#include <mllm/models/lfm2_moe/tokenization_lfm2_moe.hpp>
#include <mllm/utils/AnyValue.hpp>

using mllm::Argparse;

// LFM2.5-8B-A1B (text) CPU/fp32 reference runner.
//   Convert weights first:  python convert_lfm2_moe.py convert --out lfm2.5-8b-a1b.mllm
//   Then:  mllm-lfm2_moe-runner -m lfm2.5-8b-a1b.mllm -t <dir>/tokenizer.json -c <dir>/config.json
MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Path to .mllm model").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("tokenizer.json path").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("config.json path").required(true);
  auto& check_tok = Argparse::add<bool>("--check-tokenizer").help("Tokenize a fixed prompt and print ids, then exit");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  {
    auto tokenizer = mllm::models::lfm2_moe::Lfm2MoeTokenizer(tokenizer_path.get());

    // Quick tokenizer sanity check (compare to HF `tok("The capital of France is")`).
    if (check_tok.isSet()) {
      auto ids = tokenizer.encodeRaw("The capital of France is");
      fmt::print("raw ids:");
      for (auto id : ids) { fmt::print(" {}", id); }
      fmt::print("\n(expected: 597 5205 302 3980 355)\n");
      mllm::shutdownContext();
      return 0;
    }

    auto cfg = mllm::models::lfm2_moe::Lfm2MoeConfig(config_path.get());
    auto model = mllm::models::lfm2_moe::Lfm2MoeForCausalLM(cfg);

    auto param = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
    model.load(param);

    fmt::print("\n{:*^60}\n", " LFM2.5-8B-A1B Interactive CLI ");
    std::string prompt_text;
    fmt::print("Prompt: ");
    std::getline(std::cin, prompt_text);

    try {
      fmt::print("Processing...\n\nResponse: ");
      auto inputs = tokenizer.convertMessage({.prompt = prompt_text});
      for (auto& step : model.chat(inputs)) {
        std::wcout << tokenizer.detokenize(step.cur_token_id) << std::flush;
      }
      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\nError: {}\n", e.what()); }

    model.perfSummary();
  }

  mllm::print("\n");
  mllm::memoryReport();
})
