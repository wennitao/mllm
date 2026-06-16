// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>

#include "mllm/backends/opencl/ops/BlockSparseAttentionOp.hpp"
#include "CL/cl.h"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLBlockSparseAttentionOp::OpenCLBlockSparseAttentionOp(const aops::BlockSparseAttentionOpOptions& options)
    : aops::BlockSparseAttentionOp(options) {}

OpenCLBlockSparseAttentionOp::~OpenCLBlockSparseAttentionOp() {
  if (kt_img_) clReleaseMemObject(kt_img_);
  if (vc_img_) clReleaseMemObject(vc_img_);
}

void OpenCLBlockSparseAttentionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& Idx = inputs[3];
  auto& O = outputs[0];

  const auto& qs_shape = Q.shape();
  const int B = qs_shape[0], H = qs_shape[1], S_q = qs_shape[2], D = qs_shape[3];
  const int S_kv = K.shape()[2];
  const int BH = B * H;

  // Selection geometry from Idx[B*H, num_qb, top_k].
  const auto& ix = Idx.shape();
  MLLM_RT_ASSERT_EQ(ix.size(), 3);
  const int num_qb = ix[1], top_k = ix[2];
  const int BK = options_.BK;
  const int BQ = S_q / num_qb;
  const int sel = top_k * BK;

  MLLM_RT_ASSERT(D == 128);
  MLLM_RT_ASSERT(Q.dtype() == mllm::kFloat16);
  MLLM_RT_ASSERT((S_q % BQ) == 0 && (sel % 8) == 0 && (BQ % 4) == 0 && (BK % 8) == 0);

  auto rt = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  if (built_for_d_ != D) {
    std::set<std::string> base;
    base.insert(std::string("-DFA_D=") + std::to_string(D));
    tp_pack_q_ = rt->buildKernel("flash_attention", "tp_pack_q", base);
    tp_trans_k_ = rt->buildKernel("flash_attention", "tp_trans_k", base);
    tp_copy_v_ = rt->buildKernel("flash_attention", "tp_copy_v", base);
    bs_qk_ = rt->buildKernel("flash_attention", "bs_qk_gemm", base);
    bs_softmax_ = rt->buildKernel("flash_attention", "bs_softmax", base);
    bs_pv_ = rt->buildKernel("flash_attention", "bs_pv_gemm", base);
    MLLM_RT_ASSERT(tp_pack_q_ && tp_trans_k_ && tp_copy_v_ && bs_qk_ && bs_softmax_ && bs_pv_);
    built_for_d_ = D;
  }

  auto ctx = rt->context();
  const size_t hb = sizeof(uint16_t);
  auto ensure = [&](cl::Buffer& buf, size_t& cur, size_t need, bool& grew) {
    grew = false;
    if (cur < need) {
      cl_int e = CL_SUCCESS;
      buf = cl::Buffer(ctx, CL_MEM_READ_WRITE, need, nullptr, &e);
      MLLM_RT_ASSERT(e == CL_SUCCESS);
      cur = need;
      grew = true;
    }
  };
  bool g_qp = false, g_kt = false, g_vc = false, g_s = false;
  ensure(qp_, qp_b_, (size_t)BH * (D / 4) * S_q * 4 * hb, g_qp);
  ensure(kt_, kt_b_, (size_t)BH * D * S_kv * hb, g_kt);
  ensure(vc_, vc_b_, (size_t)BH * S_kv * D * hb, g_vc);
  ensure(s_, s_b_, (size_t)BH * S_q * sel * hb, g_s);
  (void)g_qp; (void)g_s;

  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  auto rebuild_img = [&](cl_mem& img, cl::Buffer& buf, size_t buf_bytes, bool grew) {
    if (img && grew) { clReleaseMemObject(img); img = nullptr; }
    if (!img) {
      cl_image_desc desc = {};
      desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
      desc.image_width = buf_bytes / 8;  // half4 texels over the whole buffer
      desc.buffer = buf();
      cl_int e = CL_SUCCESS;
      img = clCreateImage(ctx(), CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
      MLLM_RT_ASSERT(e == CL_SUCCESS);
    }
  };
  rebuild_img(kt_img_, kt_, kt_b_, g_kt);
  rebuild_img(vc_img_, vc_, vc_b_, g_vc);

  const auto& qst = Q.stride();
  const auto& kst = K.stride();
  const auto& vst = V.stride();
  const int Qbs = qst[0], Qhs = qst[1], Qss = qst[2];
  const int Kbs = kst[0], Khs = kst[1], Kss = kst[2];
  const int Vbs = vst[0], Vhs = vst[1], Vss = vst[2];

  auto q_buf = (cl_mem)Q.impl()->ptr<void>();
  auto k_buf = (cl_mem)K.impl()->ptr<void>();
  auto v_buf = (cl_mem)V.impl()->ptr<void>();
  auto o_buf = (cl_mem)O.impl()->ptr<void>();
  auto idx_buf = (cl_mem)Idx.impl()->ptr<void>();
  cl_mem qp = qp_(), kt = kt_(), vc = vc_(), s = s_();

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int causal = options_.causal_mask ? 1 : 0;

  auto& cq = rt->commandQueue();
  cl_int err = CL_SUCCESS;
  auto SI = [&](cl::Kernel& k, int i, int v) { err |= k.setArg(i, sizeof(int), &v); };
  auto SF = [&](cl::Kernel& k, int i, float v) { err |= k.setArg(i, sizeof(float), &v); };
  auto SM = [&](cl::Kernel& k, int i, cl_mem v) { err |= k.setArg(i, sizeof(cl_mem), &v); };

  // pack_q
  { auto& k = tp_pack_q_->get(); SM(k,0,q_buf); SM(k,1,qp); SI(k,2,B); SI(k,3,H); SI(k,4,S_q); SI(k,5,Qbs); SI(k,6,Qhs); SI(k,7,Qss);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(S_q, D / 4, BH), cl::NullRange); }
  // trans_k
  { auto& k = tp_trans_k_->get(); SM(k,0,k_buf); SM(k,1,kt); SI(k,2,B); SI(k,3,H); SI(k,4,S_kv); SI(k,5,Kbs); SI(k,6,Khs); SI(k,7,Kss);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(S_kv, D, BH), cl::NullRange); }
  // copy_v (vectorized half8)
  { auto& k = tp_copy_v_->get(); SM(k,0,v_buf); SM(k,1,vc); SI(k,2,B); SI(k,3,H); SI(k,4,S_kv); SI(k,5,Vbs); SI(k,6,Vhs); SI(k,7,Vss);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(D / 8, S_kv, BH), cl::NullRange); }
  // bs_qk_gemm
  { auto& k = bs_qk_->get(); SM(k,0,kt_img_); SM(k,1,qp); SM(k,2,s); SM(k,3,idx_buf);
    SI(k,4,S_q); SI(k,5,S_kv); SI(k,6,BH); SI(k,7,num_qb); SI(k,8,top_k); SI(k,9,BK); SI(k,10,BQ); SF(k,11,scale); SI(k,12,causal);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(sel / 8, BQ / 4, BH * num_qb), cl::NullRange); }
  // bs_softmax (in place on s)
  { auto& k = bs_softmax_->get(); SM(k,0,s); SI(k,1,BH); SI(k,2,num_qb); SI(k,3,top_k); SI(k,4,BK); SI(k,5,BQ);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(kSmLw, BQ, BH * num_qb), cl::NDRange(kSmLw, 1, 1)); }
  // bs_pv_gemm
  { auto& k = bs_pv_->get(); SM(k,0,vc_img_); SM(k,1,s); SM(k,2,o_buf); SM(k,3,idx_buf);
    SI(k,4,S_q); SI(k,5,S_kv); SI(k,6,BH); SI(k,7,num_qb); SI(k,8,top_k); SI(k,9,BK); SI(k,10,BQ);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(D / 8, BQ / 4, BH * num_qb), cl::NullRange); }

  if (err != CL_SUCCESS) { MLLM_ERROR("BlockSparseAttention setArg failed: {}", err); }
}

}  // namespace mllm::opencl
