#include <iostream>
#include <string>

#include "mllm/mllm.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/modeling_qwen3_fa2.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
using mllm::Argparse;

MLLM_MAIN({
    auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
    auto& model_path =
        Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
    auto& model_version =
        Argparse::add<std::string>("-mv|--model_version")
            .help("Model file version: v1 or v2")
            .def("v1");
    auto& tokenizer_path =
        Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer path").required(true);
    auto& config_path =
        Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
    auto& prompt =
        Argparse::add<std::string>("-p|--prompt").help("Prompt text").def("Hello");
    auto& max_length =
        Argparse::add<int>("--max_length").help("Max decode steps").def(32);
    auto& trace_out =
        Argparse::add<std::string>("--trace_out")
            .help("Perfetto trace output path")
            .def("/data/local/tmp/perf_trace.perfetto");

    Argparse::parse(argc, argv);

    if (help.isSet()) {
        Argparse::printHelp();
        return 0;
    }

    mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
    if (model_version.get() == "v1") {
        file_version = mllm::ModelFileVersion::kV1;
    } else if (model_version.get() == "v2") {
        file_version = mllm::ModelFileVersion::kV2;
    } else {
        std::cerr << "Unknown --model_version: " << model_version.get()
                  << " (expected v1 or v2)" << std::endl;
        return 1;
    }

    mllm::initializeContext();

    try {
        auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());
        auto qwen3_tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());
        auto qwen3 = mllm::models::qwen3::Qwen3ForCausalLM(qwen3_cfg);

        auto params = mllm::load(model_path.get(), file_version);
        qwen3.load(params);

        auto inputs = qwen3_tokenizer.convertMessage({.prompt = prompt.get()});

        std::cout << "Prompt: " << prompt.get() << "\n\n";
        std::cout << "Response: ";

#ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::start();
#endif

        for (auto& step : qwen3.chat(inputs, {{"max_length", mllm::AnyValue(max_length.get())}})) {
            std::wcout << qwen3_tokenizer.detokenize(step.cur_token_id) << std::flush;
        }

#ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::stop();
        mllm::perf::saveReport(trace_out.get());
        std::cerr << "\n[Perfetto] trace saved to: " << trace_out.get() << std::endl;
#endif

        std::cout << std::endl;
        qwen3.perfSummary();
    } catch (const std::exception& e) {
#ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::stop();
#endif
        std::cerr << "Error: " << e.what() << std::endl;
        mllm::shutdownContext();
        return 1;
    }

    mllm::shutdownContext();
    return 0;
});