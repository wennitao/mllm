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
  ~OpenCLFlashAttention2Op();

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  // Kernels are lazily built per (head_dim) on first forward, since FA_D is a
  // compile-time constant in the kernel (sizes __local arrays).
  int built_for_d_ = 0;
  std::shared_ptr<KernelWrap> kernel_fp32_ = nullptr;
  std::shared_ptr<KernelWrap> kernel_fp16_ = nullptr;        // prefill (big tile)
  std::shared_ptr<KernelWrap> kernel_fp16_small_ = nullptr;  // tiny S_q (2..31)
  std::shared_ptr<KernelWrap> kernel_fp16_decode_ = nullptr;       // S_q == 1 (split-K stream)
  std::shared_ptr<KernelWrap> kernel_fp16_decode_merge_ = nullptr; // split-K partial merge

  // Two-pass GEMM-class prefill kernels (S_q >= kTwoPassThreshold, fp16, D==FA_D).
  std::shared_ptr<KernelWrap> tp_pack_q_ = nullptr;
  std::shared_ptr<KernelWrap> tp_trans_k_ = nullptr;
  std::shared_ptr<KernelWrap> tp_copy_v_ = nullptr;
  std::shared_ptr<KernelWrap> tp_qk_ = nullptr;
  std::shared_ptr<KernelWrap> tp_softmax_ = nullptr;
  std::shared_ptr<KernelWrap> tp_pv_ = nullptr;

  // Split-K scratch for the decode kernel: [B*H, nsplit, D+2] float partials.
  // Grow-only; persists across forwards.
  cl::Buffer decode_scratch_;
  size_t decode_scratch_bytes_ = 0;

  // Two-pass prefill scratch (grow-only, persist across forwards): contiguous
  // Qp[B*H,D/4,Sq,4], Kt[B*H,D,Skv], Vc[B*H,Skv,D], S[B*H,Sq,Skv] (all fp16).
  cl::Buffer tp_qp_, tp_kt_, tp_vc_, tp_s_;
  size_t tp_qp_b_ = 0, tp_kt_b_ = 0, tp_vc_b_ = 0, tp_s_b_ = 0;
  // image1d_buffer wrappers over Kt/Vc, recreated only when those buffers grow
  // (so no per-forward image recreate / clFinish). Released in the destructor.
  cl_mem tp_kt_img_ = nullptr, tp_vc_img_ = nullptr;

  bool tryForwardTwoPass(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                         int B, int H, int S_q, int S_kv, int D, float scale, int causal);

  static constexpr int kTwoPassThreshold = 512;  // S_q >= this -> two-pass prefill
  static constexpr int kTpSmLw = 64;             // must match TP_SM_LW in the kernel

  // Tile sizes — must match the kernel macros (FA_BR_H).
  static constexpr int kBr = 4;             // fp32 reference: q-rows per workgroup
  static constexpr int kBrFp16 = 32;        // fp16 prefill: q-rows per wg (max cross-q reuse)
  static constexpr int kBrFp16Small = 4;    // fp16 decode/tiny S_q: q-rows per wg
  static constexpr int kSmallSqThreshold = 32;  // S_q < this -> use the small-tile kernel
  static constexpr int kNsplitMax = 16;     // decode split-K partition cap
};

class OpenCLFlashAttention2OpFactory : public TypedOpFactory<OpTypes::kFlashAttention2, aops::FlashAttention2OpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::FlashAttention2OpOptions& options) override {
    return std::make_shared<OpenCLFlashAttention2Op>(options);
  }
};

}  // namespace mllm::opencl
