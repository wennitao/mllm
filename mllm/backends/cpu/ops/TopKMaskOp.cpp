// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <vector>

#include "mllm/backends/cpu/ops/TopKMaskOp.hpp"

namespace mllm::cpu {

CPUTopKMaskOp::CPUTopKMaskOp(const aops::TopKMaskOpOptions& options) : aops::TopKMaskOp(options) {}

void CPUTopKMaskOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ins = inputs[0];
  auto ous = outputs[0];

  const auto& shape = ins.shape();
  const int B = shape[0];
  const int H = shape[1];
  const int S = shape[2];
  const int D = shape[3];
  const int k = options_.top_k;

  switch (ins.dtype()) {
    case kFloat32: {
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
          auto* i_ptr = ins.offsettedPtr<float>({b, h, 0, 0});
          auto* o_ptr = ous.offsettedPtr<float>({b, h, 0, 0});

          if (S == 1) {
            // Decode: keep top-k scores, mask the rest.
            if (k <= 0 || k >= D) {
              memcpy(o_ptr, i_ptr, D * sizeof(float));
            } else {
              std::vector<float> tmp(i_ptr, i_ptr + D);
              std::nth_element(tmp.begin(), tmp.begin() + (D - k), tmp.end());
              const float threshold = tmp[D - k];
              for (int d = 0; d < D; ++d) { o_ptr[d] = (i_ptr[d] >= threshold) ? i_ptr[d] : -1e10f; }
            }
          } else {
            // Prefill: apply causal (lower-triangular) mask.
            for (int r = 0; r < S; ++r) {
              const int copy_count = D - S + r + 1;
              const int fill_count = D - copy_count;
              memcpy(o_ptr + r * D, i_ptr + r * D, copy_count * sizeof(float));
              for (int d = copy_count; d < D; ++d) { o_ptr[r * D + d] = -1e10f; }
            }
          }
        }
      }
      break;
    }
    case kFloat16: {
      for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
          auto* i_ptr = ins.offsettedPtr<mllm_fp16_t>({b, h, 0, 0});
          auto* o_ptr = ous.offsettedPtr<mllm_fp16_t>({b, h, 0, 0});

          if (S == 1) {
            // Decode: keep top-k scores, mask the rest.
            if (k <= 0 || k >= D) {
              memcpy(o_ptr, i_ptr, D * sizeof(mllm_fp16_t));
            } else {
              std::vector<float> tmp(D);
              for (int d = 0; d < D; ++d) { tmp[d] = static_cast<float>(i_ptr[d]); }
              std::nth_element(tmp.begin(), tmp.begin() + (D - k), tmp.end());
              const float threshold = tmp[D - k];
              for (int d = 0; d < D; ++d) {
                o_ptr[d] = (static_cast<float>(i_ptr[d]) >= threshold) ? i_ptr[d] : static_cast<mllm_fp16_t>(-65500.f);
              }
            }
          } else {
            // Prefill: apply causal (lower-triangular) mask.
            for (int r = 0; r < S; ++r) {
              const int copy_count = D - S + r + 1;
              memcpy(o_ptr + r * D, i_ptr + r * D, copy_count * sizeof(mllm_fp16_t));
              for (int d = copy_count; d < D; ++d) { o_ptr[r * D + d] = static_cast<mllm_fp16_t>(-65500.f); }
            }
          }
        }
      }
      break;
    }
    default: NYI("TopKMaskOp::forward only supports fp32 and fp16");
  }
}

}  // namespace mllm::cpu
