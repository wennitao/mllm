#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include <algorithm>

namespace mllm::qnn::aot {

QnnAOTModule::QnnAOTModule(const std::string& graph_name) : mllm::nn::Module(graph_name), graph_name_(graph_name) {}

std::vector<mllm::Tensor> QnnAOTModule::forward(const std::vector<mllm::Tensor>& inputs,
                                                const std::vector<mllm::AnyValue>& args) {
  return output_tensors_;
}

int64_t QnnAOTModule::sampleGreedy(mllm::Tensor& logits) {
  int vocab_size = logits.shape().back();
  // Dispatch by dtype: quantized models store logits as uint16 with affine
  // quantization (and the bit pattern's natural ordering matches the value's
  // ordering for unsigned types), while fp16 models store logits as fp16 —
  // comparing their bits as uint16 would prefer the most-negative value
  // (sign bit set) which is wrong.
  if (logits.dtype() == kFloat16) {
    auto p = logits.ptr<mllm_fp16_t>();
    auto max_it = std::max_element(p, p + vocab_size,
                                   [](mllm_fp16_t a, mllm_fp16_t b) { return (float)a < (float)b; });
    return std::distance(p, max_it);
  }
  auto logits_data = logits.ptr<uint16_t>();
  auto max_it = std::max_element(logits_data, logits_data + vocab_size);
  return std::distance(logits_data, max_it);
}

}  // namespace mllm::qnn::aot
