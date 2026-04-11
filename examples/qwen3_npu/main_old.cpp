#include <chrono>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <string>

#include <mllm/backends/qnn/passes/QNNGraphBuildPass.hpp>
#include <mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp>
#include <mllm/backends/qnn/passes/QNNOpNamingPass.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/modeling_qwen3.hpp>
#include <mllm/models/qwen3/modeling_qwen3_npu.hpp>
#include <mllm/models/qwen3/tokenization_qwen3.hpp>
#include <mllm/nn/lmcache/StaticCache.hpp>
#include <mllm/preprocessor/tokenizers/Unicode.hpp>
#include <mllm/utils/Log.hpp>

using mllm::Argparse;

namespace {

auto parseModelVersion(const std::string& model_version) -> mllm::ModelFileVersion {
  if (model_version == "v1") { return mllm::ModelFileVersion::kV1; }
  if (model_version == "v2") { return mllm::ModelFileVersion::kV2; }
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Unsupported model_version: {} (expected v1 or v2)", model_version);
}

auto argmaxTokenId(const mllm::Tensor& logits) -> int64_t {
  auto logits_f = logits;
  logits_f = logits_f.to(mllm::kFloat32);
  const auto& shape = logits_f.shape();
  MLLM_RT_ASSERT_EQ(shape.size(), 3);
  auto* data = logits_f.ptr<float>();
  int64_t vocab_size = shape[2];
  int64_t next_id = 0;
  float max_logit = data[0];
  for (int64_t i = 1; i < vocab_size; ++i) {
    if (data[i] > max_logit) {
      max_logit = data[i];
      next_id = i;
    }
  }
  return next_id;
}

void copyStaticCache(mllm::nn::StaticCache& src, mllm::nn::StaticCache& dst) {
  MLLM_RT_ASSERT_EQ(src.getLayerNums(), dst.getLayerNums());
  for (int32_t layer_idx = 0; layer_idx < src.getLayerNums(); ++layer_idx) {
    src.getKCacheBuffer(layer_idx).copy2(dst.getKCacheBuffer(layer_idx));
    src.getVCacheBuffer(layer_idx).copy2(dst.getVCacheBuffer(layer_idx));
  }
  dst.setCurrentSeqCnt(src.getCurrentSeqCnt(0));
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& npu_model_path =
      Argparse::add<std::string>("--npu_model_path").help("QNN-prefill model path").required(true);
  auto& cpu_model_path =
      Argparse::add<std::string>("--cpu_model_path").help("CPU-decode model path").required(true);
  auto& npu_config_path =
      Argparse::add<std::string>("--npu_config_path").help("QNN-prefill config path").required(true);
  auto& cpu_config_path =
      Argparse::add<std::string>("--cpu_config_path").help("CPU-decode config path").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer path").required(true);
  auto& model_version =
      Argparse::add<std::string>("-mv|--model_version").help("Model version").def("v2");
  auto& max_new_tokens =
      Argparse::add<int>("--max_new_tokens").help("Maximum decode steps").def(512);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  auto file_version = parseModelVersion(model_version.get());

  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());
  auto cpu_cfg = mllm::models::qwen3::Qwen3Config(cpu_config_path.get());
  auto npu_cfg = mllm::models::qwen3::Qwen3Config(npu_config_path.get());

  auto cpu_model = mllm::models::qwen3::Qwen3ForCausalLM(cpu_cfg);
  auto cpu_param = mllm::load(cpu_model_path.get(), file_version);
  cpu_model.load(cpu_param);
  MLLM_INFO("CPU decode model loaded from: {}", cpu_model_path.get());

  mllm::initQnnBackend();
  auto npu_model = mllm::models::qwen3_npu::Qwen3ForCausalLMPrefill("", npu_cfg);
  auto npu_param = mllm::load(npu_model_path.get(), file_version);
  npu_model.load(npu_param);
  MLLM_INFO("QNN prefill model loaded from: {}", npu_model_path.get());

  auto shared_kv_cache = std::make_unique<mllm::nn::StaticCache>(
      cpu_cfg.max_cache_length, cpu_cfg.num_hidden_layers, cpu_cfg.num_attention_heads, cpu_cfg.num_key_value_heads,
      cpu_cfg.head_dim, mllm::kFloat32, mllm::kFloat32, mllm::kCPU, false);
  for (int32_t layer_idx = 0; layer_idx < cpu_cfg.num_hidden_layers; ++layer_idx) {
    auto& kv_cache = npu_model.model.decode_blocks().list()[layer_idx].getKVCache();
    kv_cache.setStaticCache(shared_kv_cache.get());
    kv_cache.setLayerIndex(layer_idx);
  }

  fmt::print("\n{:*^60}\n", " Qwen3 Mixed CLI ");
  fmt::print("Prefill on QNN, decode on CPU\n");
  fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

  std::string prompt_text;
  fmt::print("💬 Prompt text (or 'exit/quit'): ");
  std::getline(std::cin, prompt_text);
  if (prompt_text == "exit" || prompt_text == "quit") {
    mllm::shutdownContext();
    return 0;
  }

  auto raw_input_tokens = tokenizer.convertMessage({.prompt = prompt_text})["sequence"];
  MLLM_INFO("Input tokens: {}", raw_input_tokens.shape()[1]);

  mllm::models::ARGenerationOutputPast past{{"sequence", raw_input_tokens}};

  MLLM_INFO("Building QNN graph for prefill...");
  auto irs = npu_model.trace(past, {});
  mllm::ir::PassManager rewrite_pm(irs["model"]);
  rewrite_pm.reg(mllm::qnn::createQNNGraphIOTensorPass());
  rewrite_pm.reg(mllm::qnn::createQNNOpNamingPass());
  rewrite_pm.run();

  mllm::ir::PassManager graph_build_pm(irs["model"]);
  graph_build_pm.reg(mllm::qnn::createQNNGraphBuildPass());
  graph_build_pm.run();

  npu_model.model.clearKVCache();

  MLLM_INFO("Starting QNN prefill...");
  auto prefill_start = std::chrono::high_resolution_clock::now();
  mllm::models::ARGenerationArgs prefill_args;
  prefill_args["seq_len"] = static_cast<int>(raw_input_tokens.shape()[1]);
  auto prefill_output = npu_model.forward(past, prefill_args);
  auto prefill_end = std::chrono::high_resolution_clock::now();
  auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();
  MLLM_INFO("QNN prefill completed in {:.2f}s", prefill_ms / 1000.0);

  copyStaticCache(*shared_kv_cache, cpu_model.kvCache());

  auto first_next_id = argmaxTokenId(prefill_output["sequence"]);

  past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
  past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(first_next_id);
  past["position_ids"] = prefill_output["position_ids"];

  std::cout << mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(first_next_id)) << std::flush;

  const auto eos_token_id = cpu_cfg.eos_token_id;
  mllm::models::ARGenerationArgs decode_args;
  decode_args["debug_layer_outputs"] = false;

  double total_decode_time_ms = 0.0;
  int decode_count = 0;

  MLLM_INFO("Starting CPU decode...");
  for (int step = 1; step < max_new_tokens.get(); ++step) {
    auto step_start = std::chrono::high_resolution_clock::now();
    auto output = cpu_model.forward(past, decode_args);
    auto step_end = std::chrono::high_resolution_clock::now();
    auto step_us = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count();
    auto step_time_ms = step_us / 1000.0;

    auto next_id = argmaxTokenId(output["sequence"]);
    std::cout << mllm::preprocessor::wideString2Utf8String(tokenizer.detokenize(next_id)) << std::flush;

    total_decode_time_ms += step_time_ms;
    decode_count++;

    past = std::move(output);
    past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
    past["sequence"].at<mllm::mllm_int64_t>({0, 0}) = static_cast<mllm::mllm_int64_t>(next_id);

    if (next_id == eos_token_id) { break; }
  }

  std::cout << "\n";
  if (decode_count > 0) {
    auto avg_time_ms = total_decode_time_ms / decode_count;
    MLLM_INFO("Decode completed: {} tokens, avg {:.2f}ms/token, throughput {:.2f} tok/s", decode_count, avg_time_ms,
              decode_count / (total_decode_time_ms / 1000.0));
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen3_mixed.perfetto");
#endif

  mllm::shutdownContext();
  return 0;
})
