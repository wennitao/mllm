// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/FlashAttention2Op.hpp"

namespace mllm::opencl {

// FlashAttention (Dao et al., 2022) for OpenCL.
//
// Inputs (BHSD layout, NOT BSHD as the CPU FA2 op expects):
//   inputs[0] Q : [B, H, S_q,  D]
//   inputs[1] K : [B, H, S_kv, D]   (already GQA-expanded to H = q_heads)
//   inputs[2] V : [B, H, S_kv, D]
// Output:
//   outputs[0] O : [B, H, S_q, D]
//
// The op assumes head_dim D is a power of two (Qwen3's D = 128).
class OpenCLFlashAttention2Op final : public aops::FlashAttention2Op {
 public:
  explicit OpenCLFlashAttention2Op(const aops::FlashAttention2OpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  // Kernels are lazily built per (head_dim) on first forward, since FA_D is a
  // compile-time constant in the kernel (sizes __local arrays).
  int built_for_d_ = 0;
  std::shared_ptr<KernelWrap> kernel_fp32_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_ = nullptr;        // prefill (big tile)
  std::shared_ptr<KernelWrap> kernel_fp16_small_ = nullptr;  // decode / tiny S_q

  // Tile sizes — must match the kernel macros (FA_BR_H).
  static constexpr int kBr = 4;             // fp32 reference: q-rows per workgroup
  static constexpr int kBrFp16 = 32;        // fp16 prefill: q-rows per wg (max cross-q reuse)
  static constexpr int kBrFp16Small = 8;    // fp16 decode/tiny S_q: q-rows per wg
  static constexpr int kSmallSqThreshold = 32;  // S_q < this -> use the small-tile kernel
};

class OpenCLFlashAttention2OpFactory : public TypedOpFactory<OpTypes::kFlashAttention2, aops::FlashAttention2OpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FlashAttention2OpOptions& options) override {
    return std::make_shared<OpenCLFlashAttention2Op>(options);
  }
};

}  // namespace mllm::opencl
