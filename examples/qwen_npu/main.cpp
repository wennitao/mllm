#include <fmt/core.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/core/DataTypes.hpp>
#include <mllm/core/SlicePrimitives.hpp>
#include <mllm/utils/AnyValue.hpp>
#include <mllm/utils/Log.hpp>

#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/models/qwen_npu/modeling_qwen_npu_cpu.hpp"

int main(int argc, char** argv) {
#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif
  ::mllm::initializeContext();

  MLLM_INFO("Pure CPU inference mode: GGUF Q4_0 prefill + decode");

  const std::string config_path = "/data/local/tmp/qwen3_gguf/config_0.6B_gguf_q4_0.json";
  const std::string model_path = "/data/local/tmp/qwen3_gguf/qwen3-0.6B-gguf-q4_0.mllm";
  const std::string tokenizer_path = "/data/local/tmp/qwen3_gguf/tokenizer.json";

  MLLM_INFO("Creating tokenizer from: {}", tokenizer_path);
  auto qwen_tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path);
  MLLM_INFO("Tokenizer created");

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV2;
  const int max_new_tokens = 128;

  MLLM_INFO("Loading config from: {}", config_path);
  auto cfg = mllm::models::qwen_npu::QwenNPUConfig(config_path);
  MLLM_INFO("Config loaded: layers={}, hidden_size={}, kv_heads={}, head_dim={}, tie_word_embeddings={}",
            cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads, cfg.head_dim,
            cfg.tie_word_embeddings);

  auto raw_input_tokens = qwen_tokenizer.convertMessage(
      mllm::models::qwen3::Qwen3Message{
          .prompt =
              "提示:"
              "海洋世界里，鲸鱼是地球上体型最为庞大的哺乳动物，它们拥有流线型的身躯，主要通过头顶的喷水孔进行呼吸。与终生生活在水"
              "下并利用鱼鳃从水中提取溶解氧的鱼类有着本质区别。鲸鱼无法在水下直接呼吸氧气，因此它们需要耗费大量的体力，定时浮出水"
              "面完成一次快速而彻底的换气过程。令人惊奇的是，当它们处于睡眠状态时，为了确保不会因为忘记呼吸而发生危险，它们只会关"
              "闭大脑的一半来进行休息，另一半大脑则始终保持清醒和警觉，以便及时引导身体浮上水面。这种独特的生存机制是它们在深海中"
              "延续生命的关键。问题：鲸鱼与鱼类在呼吸方式上的根本区别是什么？它们在睡觉时会采取什么特殊的措施来保证安全和生存？"})
                              ["sequence"];
  MLLM_INFO("Input tokens: {} tokens", raw_input_tokens.shape()[1]);

  MLLM_INFO("Constructing CPU model...");
  auto model = std::make_unique<mllm::models::qwen_npu::QwenForCausalLMCPU>("", cfg);

  MLLM_INFO("Loading CPU parameter file from: {}", model_path);
  auto param = mllm::load(model_path, file_version);
  MLLM_INFO("Parameter file loaded, binding weights...");
  model->load(param);
  MLLM_INFO("CPU model loaded from: {}", model_path);

  const int64_t eos_token_id = cfg.eos_token_id;

  mllm::models::ARGenerationOutputPast past{{"sequence", raw_input_tokens}};
  mllm::models::ARGenerationArgs args;
  args["debug_layer_outputs"] = false;

  // Prefill
  MLLM_INFO("Starting CPU prefill...");
  auto prefill_start = std::chrono::high_resolution_clock::now();

  const int32_t prefill_seq_len = static_cast<int32_t>(raw_input_tokens.shape()[1]);
  mllm::models::ARGenerationArgs prefill_args;
  prefill_args["seq_len"] = prefill_seq_len;

  auto prefill_output = model->forward(past, prefill_args);

  auto prefill_end = std::chrono::high_resolution_clock::now();
  auto prefill_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start);
  MLLM_INFO("CPU prefill completed: seq_len={}, time={:.2f}s",
            prefill_seq_len, prefill_duration.count() / 1000.0);

  {
    auto prefill_logits = prefill_output["sequence"];
    auto prefill_logits_f = prefill_logits.to(mllm::kFloat32);
    const auto& prefill_shape = prefill_logits_f.shape();

    if (prefill_shape.size() == 3 && prefill_shape[1] > 0) {
      auto last_logits =
          prefill_logits_f[{mllm::kAll, {prefill_shape[1] - 1}, mllm::kAll}];
      last_logits = last_logits.view({1, 1, prefill_shape[2]});

      const int64_t vocab_size = prefill_shape[2];
      auto* logits_data = last_logits.ptr<float>();

      int64_t first_next_id = 0;
      float max_logit = logits_data[0];
      for (int64_t i = 1; i < vocab_size; ++i) {
        if (logits_data[i] > max_logit) {
          max_logit = logits_data[i];
          first_next_id = i;
        }
      }

      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) =
          static_cast<mllm::mllm_int64_t>(first_next_id);
      past["position_ids"] = prefill_output["position_ids"];

      auto first_token_str = qwen_tokenizer.detokenize(first_next_id);
MLLM_INFO("[Prefill next token] token_id: {}", first_next_id);
std::wcout << first_token_str << std::flush;
    } else {
      auto last_token_idx = raw_input_tokens.shape()[1] - 1;
      past["sequence"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["sequence"].at<mllm::mllm_int64_t>({0, 0}) =
          raw_input_tokens.at<mllm::mllm_int64_t>({0, last_token_idx});
      past["position_ids"] = mllm::Tensor::empty({1, 1}, mllm::kInt64, mllm::kCPU).alloc();
      past["position_ids"].at<mllm::mllm_int64_t>({0, 0}) =
          static_cast<mllm::mllm_int64_t>(prefill_seq_len);
    }
  }

  // Decode
  MLLM_INFO("Starting CPU decode...");
  double total_decode_time_ms = 0.0;
  int decode_count = 0;

  for (int step = 1; step < max_new_tokens; ++step) {
    auto step_start = std::chrono::high_resolution_clock::now();
    auto output = model->forward(past, args);
    auto step_end = std::chrono::high_resolution_clock::now();

    auto step_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start);
    double step_time_ms = step_duration.count() / 1000.0;

    total_decode_time_ms += step_time_ms;
    decode_count++;

    mllm::Tensor logits = output["sequence"];
    mllm::Tensor logits_f = logits.to(mllm::kFloat32);

    const auto& shape = logits_f.shape();
    if (shape.size() != 3) {
      MLLM_ERROR("Unexpected logits shape (expected [B,1,V])");
      break;
    }

    const int64_t vocab_size = shape[2];
    auto* data = logits_f.ptr<float>();

    int64_t next_id = 0;
    float max_logit = data[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
      if (data[i] > max_logit) {
        max_logit = data[i];
        next_id = i;
      }
    }

    auto next_token_str = qwen_tokenizer.detokenize(next_id);
    MLLM_INFO("[Decode step {:3d}] time: {:.2f}ms, token: ", step, step_time_ms);
    std::wcout << next_token_str << std::flush;

    past = std::move(output);
    past["sequence"] =
        mllm::Tensor::empty({1, 1}, mllm::kInt64, logits_f.device()).alloc();
    past["sequence"].at<mllm::mllm_int64_t>({0, 0}) =
        static_cast<mllm::mllm_int64_t>(next_id);

    if (next_id == eos_token_id) {
      MLLM_INFO("Hit EOS at step {}, stopping decode.", step);
      break;
    }
  }

  std::wcout << L"\n";
  if (decode_count > 0) {
    double avg_time_ms = total_decode_time_ms / decode_count;
    MLLM_INFO(
        "Decode completed: {} tokens, total time: {:.2f}ms ({:.2f}s), avg time: {:.2f}ms/token, throughput: {:.2f} tokens/s",
        decode_count, total_decode_time_ms, total_decode_time_ms / 1000.0,
        avg_time_ms, decode_count / (total_decode_time_ms / 1000.0));
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("qwen_cpu_gguf.perfetto");
#endif

  return 0;
}