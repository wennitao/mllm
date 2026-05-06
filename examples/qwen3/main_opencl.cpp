#include <iostream>

#include <fmt/core.h>
#include <mllm/core/DeviceTypes.hpp>
#include <mllm/engine/ModuleProfiler.hpp>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/modeling_qwen3_opencl.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include <mllm/utils/AnyValue.hpp>

using mllm::Argparse;

MLLM_MAIN({
  mllm::initOpenCLBackend();

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
  auto& perf_path = Argparse::add<std::string>("--perf_path").help("Perfetto trace output path").def("qwen3_opencl.perf");
  auto& module_profile_path =
      Argparse::add<std::string>("--module_profile_path")
          .help("If set, write per-Module wall-clock CSV (forces queue.finish() per module — adds overhead).")
          .def("");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  } else {
    fmt::print("Unsupported model_version: {} (expected v1 or v2)\n", model_version.get());
    mllm::shutdownContext();
    return 1;
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  {
    auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());
    auto qwen3_tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());
    auto qwen3 = mllm::models::qwen3::Qwen3ForCausalLMOpenCL(qwen3_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen3.load(param);
    qwen3.to(mllm::kOpenCL);

    if (!module_profile_path.get().empty()) { mllm::engine::ModuleProfiler::setEnabled(true); }

    fmt::print("\n{:*^60}\n", " Qwen3 OpenCL Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;
    fmt::print("Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);

    if (prompt_text == "exit" || prompt_text == "quit") {
      mllm::shutdownContext();
      return 0;
    }

    try {
      fmt::print("Processing...\n");
      auto inputs = qwen3_tokenizer.convertMessage({.prompt = prompt_text});

      fmt::print("\nResponse: ");
      for (auto& step : qwen3.chat(inputs)) { std::wcout << qwen3_tokenizer.detokenize(step.cur_token_id) << std::flush; }
      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) {
      fmt::print("\nError: {}\n{}\n", e.what(), std::string(60, '-'));
    }

    qwen3.perfSummary();

    if (!module_profile_path.get().empty()) {
      mllm::engine::ModuleProfiler::setEnabled(false);
      mllm::engine::ModuleProfiler::dumpCSV(module_profile_path.get());
      fmt::print("Wrote per-Module profile CSV to {}\n", module_profile_path.get());
    }
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport(perf_path.get());
#endif

  mllm::print("\n");
  mllm::memoryReport();
})