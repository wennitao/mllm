// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/BlockSparseAttentionOp.hpp"

namespace mllm::opencl {

// Block-sparse FlashAttention (two-pass GEMM-class, fp16) for OpenCL.
//
// Inputs (BHSD; K/V GQA-expanded to H = q_heads):
//   inputs[0] Q   : [B, H, S_q,  D]
//   inputs[1] K   : [B, H, S_kv, D]
//   inputs[2] V   : [B, H, S_kv, D]
//   inputs[3] Idx : [B*H, num_qb, top_k] int32 — selected key-block ids (units
//                   of BK keys) per query block, or < 0 for a padding slot.
// Output:
//   outputs[0] O  : [B, H, S_q, D]
//
// Reuses the dense two-pass pre-passes (tp_pack_q / tp_trans_k / tp_copy_v ->
// contiguous Qp/Kt/Vc images); the bs_* GEMMs read those images at the selected
// block offsets (index-driven, no gather). D must equal FA_D (128).
class OpenCLBlockSparseAttentionOp final : public aops::BlockSparseAttentionOp {
 public:
  explicit OpenCLBlockSparseAttentionOp(const aops::BlockSparseAttentionOpOptions& options);
  ~OpenCLBlockSparseAttentionOp();

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  int built_for_d_ = 0;
  std::shared_ptr<KernelWrap> tp_pack_q_ = nullptr;
  std::shared_ptr<KernelWrap> tp_trans_k_ = nullptr;
  std::shared_ptr<KernelWrap> tp_copy_v_ = nullptr;
  std::shared_ptr<KernelWrap> bs_qk_ = nullptr;
  std::shared_ptr<KernelWrap> bs_softmax_ = nullptr;
  std::shared_ptr<KernelWrap> bs_pv_ = nullptr;

  // Grow-only persistent scratch (fp16): Qp[BH,D/4,Sq,4], Kt[BH,D,Skv],
  // Vc[BH,Skv,D], S[BH,Sq,sel]. Kt/Vc images cached, rebuilt only on growth.
  cl::Buffer qp_, kt_, vc_, s_;
  size_t qp_b_ = 0, kt_b_ = 0, vc_b_ = 0, s_b_ = 0;
  cl_mem kt_img_ = nullptr, vc_img_ = nullptr;

  static constexpr int kSmLw = 64;  // must match BSA_SM_LW in the kernel
};

class OpenCLBlockSparseAttentionOpFactory
    : public TypedOpFactory<OpTypes::kBlockSparseAttention, aops::BlockSparseAttentionOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::BlockSparseAttentionOpOptions& options) override {
    return std::make_shared<OpenCLBlockSparseAttentionOp>(options);
  }
};

}  // namespace mllm::opencl
