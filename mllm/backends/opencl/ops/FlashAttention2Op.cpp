// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// NOTE on naming: the *op class* is OpenCLFlashAttention2Op because that's the
// registered framework slot (OpTypes::kFlashAttention2 / aops::FlashAttention2Op).
// The actual kernel here implements **FlashAttention v1** (Dao et al., 2022,
// Algorithm 1) -- the simpler version that normalizes the output every block.

#include <cmath>
#include <cstdlib>

#include "mllm/backends/opencl/ops/FlashAttention2Op.hpp"
#include "CL/cl.h"
#include "mllm/mllm.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::opencl {

OpenCLFlashAttention2Op::OpenCLFlashAttention2Op(const aops::FlashAttention2OpOptions& options)
    : aops::FlashAttention2Op(options) {
  // Kernels are built lazily on first forward() because FA_D (head_dim) must
  // be baked in at compile time to size __local arrays.

  // One-shot constructor confirmation (first instantiation only). Remove later.
  static bool s_ctor_logged = false;
  if (!s_ctor_logged) {
    s_ctor_logged = true;
    MLLM_INFO("[OpenCL-FA1] OpenCLFlashAttention2Op constructed (factory dispatched)");
  }
}

OpenCLFlashAttention2Op::~OpenCLFlashAttention2Op() {
  if (tp_kt_img_) clReleaseMemObject(tp_kt_img_);
  if (tp_vc_img_) clReleaseMemObject(tp_vc_img_);
}

void OpenCLFlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& O = outputs[0];

  // Expect BHSD: [B, H, S, D].
  const auto& q_shape = Q.shape();
  const auto& k_shape = K.shape();
  MLLM_RT_ASSERT_EQ(q_shape.size(), 4);
  MLLM_RT_ASSERT_EQ(k_shape.size(), 4);

  const int B = q_shape[0];
  const int H = q_shape[1];
  const int S_q = q_shape[2];
  const int D = q_shape[3];
  const int S_kv = k_shape[2];

  // GQA expected to be pre-applied (StaticCache eager-mode does this), so K's
  // head dim should match Q's.
  MLLM_RT_ASSERT_EQ(k_shape[1], H);
  MLLM_RT_ASSERT_EQ(V.shape()[1], H);
  MLLM_RT_ASSERT_EQ(V.shape()[2], S_kv);

  // Kernel uses local_size = D + tree reduction; require D power of two.
  MLLM_RT_ASSERT(D > 0 && (D & (D - 1)) == 0);
  MLLM_RT_ASSERT(D <= 256);

  // Lazy build per head_dim — FA_D macro sizes __local arrays.
  if (built_for_d_ != D) {
    auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
    // The fp16 kernel is compiled twice from the same parametric source: a
    // big-tile prefill build (FA_BR_H=kBrFp16, max cross-q reuse) and a small
    // build (FA_BR_H=kBrFp16Small) for decode / tiny S_q, where the big tile
    // would waste FA_NSPL QK dots per lane on out-of-range rows. The host picks
    // by S_q. fp32 reference uses kBr.
    std::set<std::string> base;
    base.insert(std::string("-DFA_D=") + std::to_string(D));
    base.insert(std::string("-DFA_BR=") + std::to_string(kBr));
    std::set<std::string> opts_pf = base;
    opts_pf.insert(std::string("-DFA_BR_H=") + std::to_string(kBrFp16));
    std::set<std::string> opts_sm = base;
    opts_sm.insert(std::string("-DFA_BR_H=") + std::to_string(kBrFp16Small));
    kernel_fp32_ = runtime->buildKernel("flash_attention", "flash_attention_fp32", opts_pf);
    MLLM_RT_ASSERT(kernel_fp32_);
    kernel_fp16_ = runtime->buildKernel("flash_attention", "flash_attention_fp16", opts_pf);
    MLLM_RT_ASSERT(kernel_fp16_);
    kernel_fp16_small_ = runtime->buildKernel("flash_attention", "flash_attention_fp16", opts_sm);
    MLLM_RT_ASSERT(kernel_fp16_small_);
    std::set<std::string> opts_dec = base;
    if (const char* p = std::getenv("FA_DEC_PROF")) {  // phase-ablation profiling
      opts_dec.insert(std::string("-DFA_DEC_PROF=") + p);
    }
    kernel_fp16_decode_ = runtime->buildKernel("flash_attention", "flash_attention_fp16_decode", opts_dec);
    MLLM_RT_ASSERT(kernel_fp16_decode_);
    kernel_fp16_decode_merge_ = runtime->buildKernel("flash_attention", "flash_attention_fp16_decode_merge", base);
    MLLM_RT_ASSERT(kernel_fp16_decode_merge_);
    // Two-pass GEMM-class prefill kernels (only viable for D == FA_D == 128).
    if (D == 128) {
      tp_pack_q_ = runtime->buildKernel("flash_attention", "tp_pack_q", base);
      tp_trans_k_ = runtime->buildKernel("flash_attention", "tp_trans_k", base);
      tp_copy_v_ = runtime->buildKernel("flash_attention", "tp_copy_v", base);
      tp_qk_ = runtime->buildKernel("flash_attention", "tp_qk_gemm", base);
      tp_softmax_ = runtime->buildKernel("flash_attention", "tp_softmax_norm", base);
      tp_pv_ = runtime->buildKernel("flash_attention", "tp_pv_gemm", base);
      MLLM_RT_ASSERT(tp_pack_q_ && tp_trans_k_ && tp_copy_v_ && tp_qk_ && tp_softmax_ && tp_pv_);
    }
    built_for_d_ = D;
  }

  // Strides (in elements). Q is non-contig after .transpose(1,2); K/V are
  // non-contig prefix slices of the static cache. Innermost D-stride must be 1
  // for our simple element-wise kernel access pattern.
  const auto& qs = Q.stride();
  const auto& ks = K.stride();
  const auto& vs = V.stride();
  MLLM_RT_ASSERT_EQ(qs[3], 1);
  MLLM_RT_ASSERT_EQ(ks[3], 1);
  MLLM_RT_ASSERT_EQ(vs[3], 1);
  const int Q_b_stride = qs[0], Q_h_stride = qs[1], Q_s_stride = qs[2];
  const int K_b_stride = ks[0], K_h_stride = ks[1], K_s_stride = ks[2];
  const int V_b_stride = vs[0], V_h_stride = vs[1], V_s_stride = vs[2];

  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int causal = options_.causal_mask ? 1 : 0;

  auto runtime_bk =
      std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  // ---- Dedicated decode path (S_q == 1, fp16): split-K streaming kernel.
  // At S_q=1 the causal mask is a no-op, K/V have zero reuse, and B*H
  // workgroups alone underfill the GPU; the decode kernel streams K/V from
  // global and splits S_kv across `nsplit` partitions merged by a second
  // tiny kernel (skipped when nsplit == 1).
  if (S_q == 1 && Q.dtype() == mllm::kFloat16) {
    // ~256-key partitions, capped; env FA_NSPLIT overrides for tuning.
    int nsplit = std::min((S_kv + 255) / 256, kNsplitMax);
    if (const char* e = std::getenv("FA_NSPLIT")) {
      const int v = std::atoi(e);
      if (v >= 1 && v <= kNsplitMax) nsplit = v;
    }
    // Round the span up to a multiple of D (the kernel's key-block size);
    // trailing empty partitions write neutral partials the merge ignores.
    int span = (S_kv + nsplit - 1) / nsplit;
    span = (span + D - 1) / D * D;

    const size_t need = static_cast<size_t>(B) * H * kNsplitMax * (D + 2) * sizeof(float);
    if (decode_scratch_bytes_ < need) {
      cl_int aerr = CL_SUCCESS;
      decode_scratch_ = cl::Buffer(runtime_bk->context(), CL_MEM_READ_WRITE, need, nullptr, &aerr);
      MLLM_RT_ASSERT(aerr == CL_SUCCESS);
      decode_scratch_bytes_ = need;
    }

    auto q_buf = (cl_mem)Q.impl()->ptr<void>();
    auto k_buf = (cl_mem)K.impl()->ptr<void>();
    auto v_buf = (cl_mem)V.impl()->ptr<void>();
    auto o_buf = (cl_mem)O.impl()->ptr<void>();
    auto p_buf = decode_scratch_();

    cl_int err = CL_SUCCESS;
    auto& kd = kernel_fp16_decode_->get();
    err |= kd.setArg(0, sizeof(cl_mem), &q_buf);
    err |= kd.setArg(1, sizeof(cl_mem), &k_buf);
    err |= kd.setArg(2, sizeof(cl_mem), &v_buf);
    err |= kd.setArg(3, sizeof(cl_mem), &o_buf);
    err |= kd.setArg(4, sizeof(cl_mem), &p_buf);
    err |= kd.setArg(5, sizeof(int), &B);
    err |= kd.setArg(6, sizeof(int), &H);
    err |= kd.setArg(7, sizeof(int), &S_kv);
    err |= kd.setArg(8, sizeof(int), &D);
    const int dq_b = qs[0], dq_h = qs[1];
    const int dk_b = ks[0], dk_h = ks[1], dk_s = ks[2];
    const int dv_b = vs[0], dv_h = vs[1], dv_s = vs[2];
    err |= kd.setArg(9, sizeof(int), &dq_b);
    err |= kd.setArg(10, sizeof(int), &dq_h);
    err |= kd.setArg(11, sizeof(int), &dk_b);
    err |= kd.setArg(12, sizeof(int), &dk_h);
    err |= kd.setArg(13, sizeof(int), &dk_s);
    err |= kd.setArg(14, sizeof(int), &dv_b);
    err |= kd.setArg(15, sizeof(int), &dv_h);
    err |= kd.setArg(16, sizeof(int), &dv_s);
    err |= kd.setArg(17, sizeof(float), &scale);
    err |= kd.setArg(18, sizeof(int), &nsplit);
    err |= kd.setArg(19, sizeof(int), &span);
    if (err != CL_SUCCESS) { MLLM_ERROR("FA decode setArg failed: {}", err); }

    cl::NDRange dglobal(D, nsplit, B * H);
    cl::NDRange dlocal(D, 1, 1);
    auto e1 = runtime_bk->commandQueue().enqueueNDRangeKernel(kd, cl::NullRange, dglobal, dlocal);
    if (e1 != CL_SUCCESS) {
      MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "FA decode kernel enqueue failed: {}", e1);
    }
    if (nsplit > 1) {
      cl_int merr = CL_SUCCESS;
      auto& km = kernel_fp16_decode_merge_->get();
      merr |= km.setArg(0, sizeof(cl_mem), &p_buf);
      merr |= km.setArg(1, sizeof(cl_mem), &o_buf);
      merr |= km.setArg(2, sizeof(int), &B);
      merr |= km.setArg(3, sizeof(int), &H);
      merr |= km.setArg(4, sizeof(int), &nsplit);
      if (merr != CL_SUCCESS) { MLLM_ERROR("FA decode merge setArg failed: {}", merr); }
      cl::NDRange mglobal(D, 1, B * H);
      auto e2 = runtime_bk->commandQueue().enqueueNDRangeKernel(km, cl::NullRange, mglobal, dlocal);
      if (e2 != CL_SUCCESS) {
        MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "FA decode merge enqueue failed: {}", e2);
      }
    }
    return;
  }

  // ---- Two-pass GEMM-class prefill (large causal S_q, fp16, D==128, aligned).
  // ~7x the fused kernel at S>=1024 and runs S=4096 (which the fused kernel
  // can't). Falls back to the fused kernel for unaligned / non-causal shapes.
  if (Q.dtype() == mllm::kFloat16 && D == 128 && options_.causal_mask &&
      S_q >= kTwoPassThreshold && (S_q & 7) == 0 && (S_kv & 7) == 0 && tp_qk_) {
    if (tryForwardTwoPass(Q, K, V, O, B, H, S_q, S_kv, D, scale, causal)) return;
  }

  std::shared_ptr<KernelWrap> kernel = nullptr;
  int rows_per_wg = kBr;
  if (Q.dtype() == mllm::kFloat32) {
    kernel = kernel_fp32_;
    rows_per_wg = kBr;
  } else if (Q.dtype() == mllm::kFloat16) {
    // Big-tile prefill kernel for S_q >= threshold; small-tile for decode/tiny.
    if (S_q >= kSmallSqThreshold) {
      kernel = kernel_fp16_;
      rows_per_wg = kBrFp16;
    } else {
      kernel = kernel_fp16_small_;
      rows_per_wg = kBrFp16Small;
    }
  } else {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "OpenCLFlashAttention2Op supports only FP32 and FP16, got dtype={}",
                    nameOfType(Q.dtype()));
  }

  auto q_buf = (cl_mem)Q.impl()->ptr<void>();
  auto k_buf = (cl_mem)K.impl()->ptr<void>();
  auto v_buf = (cl_mem)V.impl()->ptr<void>();
  auto o_buf = (cl_mem)O.impl()->ptr<void>();

  cl_int err = CL_SUCCESS;
  err |= kernel->get().setArg(0, sizeof(cl_mem), &q_buf);
  err |= kernel->get().setArg(1, sizeof(cl_mem), &k_buf);
  err |= kernel->get().setArg(2, sizeof(cl_mem), &v_buf);
  err |= kernel->get().setArg(3, sizeof(cl_mem), &o_buf);
  err |= kernel->get().setArg(4, sizeof(int), &B);
  err |= kernel->get().setArg(5, sizeof(int), &H);
  err |= kernel->get().setArg(6, sizeof(int), &S_q);
  err |= kernel->get().setArg(7, sizeof(int), &S_kv);
  err |= kernel->get().setArg(8, sizeof(int), &D);
  err |= kernel->get().setArg(9, sizeof(int), &Q_b_stride);
  err |= kernel->get().setArg(10, sizeof(int), &Q_h_stride);
  err |= kernel->get().setArg(11, sizeof(int), &Q_s_stride);
  err |= kernel->get().setArg(12, sizeof(int), &K_b_stride);
  err |= kernel->get().setArg(13, sizeof(int), &K_h_stride);
  err |= kernel->get().setArg(14, sizeof(int), &K_s_stride);
  err |= kernel->get().setArg(15, sizeof(int), &V_b_stride);
  err |= kernel->get().setArg(16, sizeof(int), &V_h_stride);
  err |= kernel->get().setArg(17, sizeof(int), &V_s_stride);
  err |= kernel->get().setArg(18, sizeof(float), &scale);
  err |= kernel->get().setArg(19, sizeof(int), &causal);
  if (err != CL_SUCCESS) { MLLM_ERROR("OpenCLFlashAttention2Op setArg failed: {}", err); }

  auto runtime = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();

  // One-shot dispatch confirmation. Remove once verified on device.
  static bool s_logged_once = false;
  if (!s_logged_once) {
    s_logged_once = true;
    MLLM_INFO("[OpenCL-FA1] dispatched: B={} H={} S_q={} S_kv={} D={} dtype={} causal={}", B, H, S_q, S_kv, D,
              nameOfType(Q.dtype()), causal);
  }

  // One work-group per (q_block, batch*head). rows_per_wg matches the selected
  // kernel's FA_BR_H (set above). local_size = D (one lane per d).
  const int q_blocks = (S_q + rows_per_wg - 1) / rows_per_wg;
  cl::NDRange global(D, q_blocks, B * H);
  cl::NDRange local(D, 1, 1);
  auto error = runtime->commandQueue().enqueueNDRangeKernel(kernel->get(), cl::NullRange, global, local);
  if (error != CL_SUCCESS) {
    MLLM_ERROR_EXIT(ExitCode::kOpenCLError, "Failed to enqueue OpenCL FlashAttention kernel, error code: {}", error);
  }
}

bool OpenCLFlashAttention2Op::tryForwardTwoPass(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                                int B, int H, int S_q, int S_kv, int D, float scale, int causal) {
  auto rt = std::static_pointer_cast<OpenCLBackend>(mllm::Context::instance().getBackend(kOpenCL))->runtime();
  auto ctx = rt->context();
  const int BH = B * H;

  // Grow-only scratch (fp16). Qp[BH,D/4,Sq,4], Kt[BH,D,Skv], Vc[BH,Skv,D], S[BH,Sq,Skv].
  // `grew` is set if the buffer was (re)allocated, so the dependent image is rebuilt.
  auto ensure = [&](cl::Buffer& buf, size_t& cur, size_t need, bool& grew) {
    grew = false;
    if (cur < need) {
      cl_int e = CL_SUCCESS;
      buf = cl::Buffer(ctx, CL_MEM_READ_WRITE, need, nullptr, &e);
      if (e != CL_SUCCESS) { MLLM_ERROR("FA two-pass scratch alloc failed: {}", e); return false; }
      cur = need;
      grew = true;
    }
    return true;
  };
  const size_t hb = sizeof(uint16_t);
  bool g_qp = false, g_kt = false, g_vc = false, g_s = false;
  if (!ensure(tp_qp_, tp_qp_b_, (size_t)BH * (D / 4) * S_q * 4 * hb, g_qp)) return false;
  if (!ensure(tp_kt_, tp_kt_b_, (size_t)BH * D * S_kv * hb, g_kt)) return false;
  if (!ensure(tp_vc_, tp_vc_b_, (size_t)BH * S_kv * D * hb, g_vc)) return false;
  if (!ensure(tp_s_, tp_s_b_, (size_t)BH * S_q * S_kv * hb, g_s)) return false;
  (void)g_qp; (void)g_s;

  // image1d_buffer (half4) over Kt/Vc — rebuilt ONLY when the buffer grows, AND
  // sized to the WHOLE buffer (not just this shape) so a smaller later shape
  // still has a valid image. texel width = buffer_bytes / 8 (half4 = 8 bytes).
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
      if (e != CL_SUCCESS) { MLLM_ERROR("FA two-pass image create failed: {}", e); img = nullptr; }
    }
    return img != nullptr;
  };
  if (!rebuild_img(tp_kt_img_, tp_kt_, tp_kt_b_, g_kt)) return false;
  if (!rebuild_img(tp_vc_img_, tp_vc_, tp_vc_b_, g_vc)) return false;
  cl_mem imgKt = tp_kt_img_, imgVc = tp_vc_img_;

  const auto& qs = Q.stride();
  const auto& ks = K.stride();
  const auto& vs = V.stride();
  auto q_buf = (cl_mem)Q.impl()->ptr<void>();
  auto k_buf = (cl_mem)K.impl()->ptr<void>();
  auto v_buf = (cl_mem)V.impl()->ptr<void>();
  auto o_buf = (cl_mem)O.impl()->ptr<void>();
  cl_mem qp = tp_qp_(), kt = tp_kt_(), vc = tp_vc_(), s = tp_s_();

  auto& cq = rt->commandQueue();
  cl_int err = CL_SUCCESS;
  auto SI = [&](cl::Kernel& k, int i, int v) { err |= k.setArg(i, sizeof(int), &v); };
  auto SF = [&](cl::Kernel& k, int i, float v) { err |= k.setArg(i, sizeof(float), &v); };
  auto SM = [&](cl::Kernel& k, int i, cl_mem v) { err |= k.setArg(i, sizeof(cl_mem), &v); };

  const int Qbs = qs[0], Qhs = qs[1], Qss = qs[2];
  const int Kbs = ks[0], Khs = ks[1], Kss = ks[2];
  const int Vbs = vs[0], Vhs = vs[1], Vss = vs[2];

  // pack_q
  { auto& k = tp_pack_q_->get(); SM(k,0,q_buf); SM(k,1,qp); SI(k,2,B); SI(k,3,H); SI(k,4,S_q); SI(k,5,Qbs); SI(k,6,Qhs); SI(k,7,Qss);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(S_q, D / 4, BH), cl::NullRange); }
  // trans_k
  { auto& k = tp_trans_k_->get(); SM(k,0,k_buf); SM(k,1,kt); SI(k,2,B); SI(k,3,H); SI(k,4,S_kv); SI(k,5,Kbs); SI(k,6,Khs); SI(k,7,Kss);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(S_kv, D, BH), cl::NullRange); }
  // copy_v
  { auto& k = tp_copy_v_->get(); SM(k,0,v_buf); SM(k,1,vc); SI(k,2,B); SI(k,3,H); SI(k,4,S_kv); SI(k,5,Vbs); SI(k,6,Vhs); SI(k,7,Vss);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(S_kv, D, BH), cl::NullRange); }
  // qk_gemm
  { auto& k = tp_qk_->get(); SM(k,0,imgKt); SM(k,1,qp); SM(k,2,s); SI(k,3,S_q); SI(k,4,S_kv); SI(k,5,BH); SF(k,6,scale); SI(k,7,causal);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(S_kv / 8, S_q / 4, BH), cl::NullRange); }
  // softmax_norm (in place on s)
  { auto& k = tp_softmax_->get(); SM(k,0,s); SI(k,1,S_q); SI(k,2,S_kv); SI(k,3,BH);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(kTpSmLw, S_q, BH), cl::NDRange(kTpSmLw, 1, 1)); }
  // pv_gemm
  { auto& k = tp_pv_->get(); SM(k,0,imgVc); SM(k,1,s); SM(k,2,o_buf); SI(k,3,S_q); SI(k,4,S_kv); SI(k,5,BH); SI(k,6,causal);
    cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(D / 8, S_q / 4, BH), cl::NullRange); }

  // No clFinish here: the persistent member images outlive these enqueues, so
  // the op pipelines with the caller's adjacent ops like the fused path.
  if (err != CL_SUCCESS) { MLLM_ERROR("FA two-pass setArg failed: {}", err); return false; }
  return true;
}

}  // namespace mllm::opencl
