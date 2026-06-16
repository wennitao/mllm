// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

// Block-sparse FlashAttention (two-pass GEMM-class, fp16).
//
// Same attention math as FlashAttention2, but each query block (BQ = S_q/num_qb
// rows) attends only to a host-selected subset of `top_k` key blocks (block size
// `BK` keys). The selection (which blocks) is decided upstream (XAttention
// scoring, sink+recent + top-k, etc.) and handed to the op as an int32
// block-index tensor inputs[3] of shape [B*H, num_qb, top_k]; this op only
// consumes the selection. See flash_attention.cl (bs_* kernels) and
// docs/opencl_backend/block_sparse_two_pass.md for the contract.
struct BlockSparseAttentionOpOptions : public BaseOpOptions<BlockSparseAttentionOpOptions> {
  int32_t B;
  int32_t q_head;
  int32_t kv_head;
  int32_t D;
  int32_t BK = 64;   // selection block size, in keys (must be a multiple of 8)
  bool causal_mask = true;
};

class BlockSparseAttentionOp : public BaseOp {
 public:
  explicit BlockSparseAttentionOp(const BlockSparseAttentionOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const BlockSparseAttentionOpOptions& options() const { return options_; }

 protected:
  BlockSparseAttentionOpOptions options_;
};

}  // namespace mllm::aops
