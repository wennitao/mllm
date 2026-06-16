// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Block-sparse two-pass GEMM-class prefill microbench (Adreno 830). Applies the
// dense two-pass image-A GEMM (examples/fa_twopass_bench) to block-sparse
// attention WITHOUT a gather: K/V are transposed/copied to FULL images once
// (dense pre-passes), and the QK/PV GEMMs read those images at the SELECTED
// block offsets driven by block_idx. Validates against a CPU block-sparse
// reference, then times the pipeline and reports GF/s over the SELECTED work.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <vector>

#include <CL/cl.h>
#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/backends/opencl/ops/BlockSparseAttentionOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/core/aops/BlockSparseAttentionOp.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {

const char* kSrc =
#include "kernels.cl.inc"
;

#define CL_CHECK(e) do { cl_int _e=(e); if(_e!=CL_SUCCESS){ fmt::print(stderr,"CL err {} at {}:{}\n",_e,__FILE__,__LINE__); std::exit(1);} } while(0)

cl_mem up(cl_context c, cl_command_queue q, const void* h, size_t b, cl_mem_flags f){ cl_int e; cl_mem m=OpenCLLoader::instance().clCreateBuffer(c,f,b,nullptr,&e); CL_CHECK(e); if(h) CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q,m,CL_TRUE,0,b,h,0,nullptr,nullptr)); return m; }

cl_mem img(cl_context c, cl_mem buf, size_t texels){ cl_image_format f={CL_RGBA,CL_HALF_FLOAT}; cl_image_desc d={}; d.image_type=CL_MEM_OBJECT_IMAGE1D_BUFFER; d.image_width=texels; d.buffer=buf; cl_int e; cl_mem im=clCreateImage(c,CL_MEM_READ_ONLY,&f,&d,nullptr,&e); CL_CHECK(e); return im; }

// sink + recent selection: block 0 + the top_k-1 most recent causal blocks for
// each query block. Distinct, diagonal included, padded with -1.
std::vector<int> make_selection(int H, int num_qb, int top_k, int Sq, int Skv, int BQ, int BK) {
  std::vector<int> idx((size_t)H * num_qb * top_k, -1);
  for (int h = 0; h < H; ++h) {
    for (int qb = 0; qb < num_qb; ++qb) {
      const int qmax_pos = (Skv - Sq) + qb * BQ + (BQ - 1);
      const int diag_blk = qmax_pos / BK;               // last causal key block
      std::vector<int> sel;
      sel.push_back(0);                                 // sink
      for (int b = diag_blk; b >= 0 && (int)sel.size() < top_k; --b)
        if (std::find(sel.begin(), sel.end(), b) == sel.end()) sel.push_back(b);  // recent
      std::sort(sel.begin(), sel.end());
      for (size_t s = 0; s < sel.size() && (int)s < top_k; ++s)
        idx[((size_t)h * num_qb + qb) * top_k + s] = sel[s];
    }
  }
  return idx;
}

void cpu_ref(int h, int Sq, int Skv, int D, int num_qb, int top_k, int BK, int BQ,
             const __fp16* Q, const __fp16* K, const __fp16* V, const int* idx,
             float scale, std::vector<float>& O) {
  O.assign((size_t)Sq * D, 0.0f);
  for (int q = 0; q < Sq; ++q) {
    const int qb = q / BQ, q_pos = (Skv - Sq) + q;
    const int* sel = idx + ((size_t)h * num_qb + qb) * top_k;
    std::vector<std::pair<float,int>> keys;  // (score, k)
    float m = -INFINITY;
    for (int s = 0; s < top_k; ++s) {
      int blk = sel[s]; if (blk < 0) continue;
      for (int kk = 0; kk < BK; ++kk) {
        int k = blk * BK + kk;
        if (k > q_pos || k >= Skv) continue;
        double acc = 0; for (int d = 0; d < D; ++d) acc += (double)(float)Q[(long)q*D+d]*(double)(float)K[(long)k*D+d];
        float sc = (float)(acc * scale);
        keys.push_back({sc, k}); m = std::max(m, sc);
      }
    }
    float l = 0; for (auto& p : keys) l += std::exp(p.first - m);
    float inv = (l > 0) ? 1.f/l : 0;
    for (auto& p : keys) { float pr = std::exp(p.first - m) * inv; for (int d=0;d<D;++d) O[(long)q*D+d] += pr * (float)(float)V[(long)p.second*D+d]; }
  }
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help");
  auto& reps_a = Argparse::add<int>("--reps").def(20);
  auto& heads_a = Argparse::add<int>("--heads").def(16);
  auto& bq_a = Argparse::add<int>("--bq").def(64);
  auto& bk_a = Argparse::add<int>("--bk").def(64);
  auto& topk_a = Argparse::add<int>("--topk").def(8);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int reps=reps_a.get(), H=heads_a.get(), D=128, BQ=bq_a.get(), BK=bk_a.get(), top_k=topk_a.get();
  const float scale=1.f/std::sqrt((float)D);

  mllm::initOpenCLBackend();
  auto rt=std::static_pointer_cast<mllm::opencl::OpenCLBackend>(mllm::Context::instance().getBackend(mllm::kOpenCL))->runtime();
  cl_context ctx=rt->context()(); cl_command_queue q=rt->commandQueue()(); cl_device_id dev=rt->getDevices()[0]();
  cl_int err; size_t sl=std::strlen(kSrc);
  cl_program prog=OpenCLLoader::instance().clCreateProgramWithSource(ctx,1,&kSrc,&sl,&err); CL_CHECK(err);
  err=OpenCLLoader::instance().clBuildProgram(prog,0,nullptr,"-cl-std=CL2.0",nullptr,nullptr);
  if(err!=CL_SUCCESS){ char lg[32768]={0}; size_t n; OpenCLLoader::instance().clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof(lg),lg,&n); fmt::print(stderr,"build fail:\n{}\n",lg); return 1; }
  auto K_=[&](const char* n){ cl_int e; cl_kernel k=OpenCLLoader::instance().clCreateKernel(prog,n,&e); CL_CHECK(e); return k; };
  cl_kernel k_pack=K_("pack_q"), k_trans=K_("trans_k"), k_copy=K_("copy_v"), k_qk=K_("bs_qk_gemm"), k_sm=K_("bs_softmax"), k_pv=K_("bs_pv_gemm");
  constexpr int LW=64;

  struct Sh{int Sq,Skv;};
  std::vector<Sh> shapes={{1024,1024},{2048,2048},{4096,4096},{8192,8192}};
  fmt::print("\n=== block-sparse two-pass (H={} D={} BQ={} BK={} top_k={} -> sel={}) ===\n", H,D,BQ,BK,top_k,top_k*BK);
  std::mt19937 rng(0xB5);
  std::uniform_real_distribution<float> dist(-1,1);

  for (auto sh:shapes){
    const int Sq=sh.Sq, Skv=sh.Skv;
    if (Sq%BQ||Skv%BK){ fmt::print("skip {}x{}\n",Sq,Skv); continue; }
    const int num_qb=Sq/BQ, sel=top_k*BK;
    std::vector<__fp16> Q((size_t)H*Sq*D), Kk((size_t)H*Skv*D), V((size_t)H*Skv*D);
    for(auto&x:Q)x=(__fp16)(dist(rng)*0.5f); for(auto&x:Kk)x=(__fp16)(dist(rng)*0.5f); for(auto&x:V)x=(__fp16)(dist(rng)*0.5f);
    auto idx=make_selection(H,num_qb,top_k,Sq,Skv,BQ,BK);

    cl_mem dQ=up(ctx,q,Q.data(),Q.size()*2,CL_MEM_READ_ONLY), dK=up(ctx,q,Kk.data(),Kk.size()*2,CL_MEM_READ_ONLY), dV=up(ctx,q,V.data(),V.size()*2,CL_MEM_READ_ONLY);
    cl_mem dIdx=up(ctx,q,idx.data(),idx.size()*4,CL_MEM_READ_ONLY);
    cl_mem dQp=up(ctx,q,nullptr,(size_t)H*(D/4)*Sq*4*2,CL_MEM_READ_WRITE);
    cl_mem dKt=up(ctx,q,nullptr,(size_t)H*D*Skv*2,CL_MEM_READ_WRITE);
    cl_mem dVc=up(ctx,q,nullptr,(size_t)H*Skv*D*2,CL_MEM_READ_WRITE);
    cl_mem dS=up(ctx,q,nullptr,(size_t)H*Sq*sel*2,CL_MEM_READ_WRITE);
    cl_mem dO=up(ctx,q,nullptr,(size_t)H*Sq*D*2,CL_MEM_READ_WRITE);
    cl_mem imgKt=img(ctx,dKt,(size_t)H*D*Skv/4), imgVc=img(ctx,dVc,(size_t)H*Skv*D/4);

    const int causal=1, Qhs=Sq*D, Qss=D, Khs=Skv*D, Kss=D, Vhs=Skv*D, Vss=D;
    auto SI=[&](cl_kernel k,int i,int v){CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k,i,sizeof(int),&v));};
    auto SM=[&](cl_kernel k,int i,cl_mem v){CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k,i,sizeof(cl_mem),&v));};
    auto SF=[&](cl_kernel k,int i,float v){CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k,i,sizeof(float),&v));};

    auto run=[&](){
      SM(k_pack,0,dQ);SM(k_pack,1,dQp);SI(k_pack,2,Sq);SI(k_pack,3,H);SI(k_pack,4,Qhs);SI(k_pack,5,Qss);
      size_t gp[3]={(size_t)Sq,(size_t)D/4,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pack,3,nullptr,gp,nullptr,0,nullptr,nullptr));
      SM(k_trans,0,dK);SM(k_trans,1,dKt);SI(k_trans,2,Skv);SI(k_trans,3,H);SI(k_trans,4,Khs);SI(k_trans,5,Kss);
      size_t gt[3]={(size_t)Skv,(size_t)D,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_trans,3,nullptr,gt,nullptr,0,nullptr,nullptr));
      SM(k_copy,0,dV);SM(k_copy,1,dVc);SI(k_copy,2,Skv);SI(k_copy,3,H);SI(k_copy,4,Vhs);SI(k_copy,5,Vss);
      size_t gc[3]={(size_t)D/8,(size_t)Skv,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_copy,3,nullptr,gc,nullptr,0,nullptr,nullptr));
      SM(k_qk,0,imgKt);SM(k_qk,1,dQp);SM(k_qk,2,dS);SM(k_qk,3,dIdx);SI(k_qk,4,Sq);SI(k_qk,5,Skv);SI(k_qk,6,H);SI(k_qk,7,num_qb);SI(k_qk,8,top_k);SI(k_qk,9,BK);SI(k_qk,10,BQ);SF(k_qk,11,scale);SI(k_qk,12,causal);
      size_t gq[3]={(size_t)sel/8,(size_t)BQ/4,(size_t)H*num_qb}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_qk,3,nullptr,gq,nullptr,0,nullptr,nullptr));
      SM(k_sm,0,dS);SI(k_sm,1,H);SI(k_sm,2,num_qb);SI(k_sm,3,top_k);SI(k_sm,4,BK);SI(k_sm,5,BQ);
      size_t gs[3]={(size_t)LW,(size_t)BQ,(size_t)H*num_qb}; size_t ls[3]={(size_t)LW,1,1}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_sm,3,nullptr,gs,ls,0,nullptr,nullptr));
      SM(k_pv,0,imgVc);SM(k_pv,1,dS);SM(k_pv,2,dO);SM(k_pv,3,dIdx);SI(k_pv,4,Sq);SI(k_pv,5,Skv);SI(k_pv,6,H);SI(k_pv,7,num_qb);SI(k_pv,8,top_k);SI(k_pv,9,BK);SI(k_pv,10,BQ);
      size_t gv[3]={(size_t)D/8,(size_t)BQ/4,(size_t)H*num_qb}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pv,3,nullptr,gv,nullptr,0,nullptr,nullptr));
    };

    run(); CL_CHECK(OpenCLLoader::instance().clFinish(q));
    std::vector<__fp16> Oh((size_t)H*Sq*D); CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q,dO,CL_TRUE,0,Oh.size()*2,Oh.data(),0,nullptr,nullptr));

    if (const char* pe=std::getenv("BSA_PROF"); pe && pe[0]=='1') {
      auto timed=[&](const char* nm, auto enq){ for(int w=0;w<3;++w)enq(); CL_CHECK(OpenCLLoader::instance().clFinish(q)); auto a=std::chrono::high_resolution_clock::now(); for(int i=0;i<10;++i)enq(); CL_CHECK(OpenCLLoader::instance().clFinish(q)); auto b=std::chrono::high_resolution_clock::now(); fmt::print("   prof {:<10} {:7.3f} ms\n",nm,std::chrono::duration<double,std::milli>(b-a).count()/10); };
      timed("trans_k",[&]{ SM(k_trans,0,dK);SM(k_trans,1,dKt);SI(k_trans,2,Skv);SI(k_trans,3,H);SI(k_trans,4,Khs);SI(k_trans,5,Kss); size_t g[3]={(size_t)Skv,(size_t)D,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_trans,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
      timed("copy_v",[&]{ SM(k_copy,0,dV);SM(k_copy,1,dVc);SI(k_copy,2,Skv);SI(k_copy,3,H);SI(k_copy,4,Vhs);SI(k_copy,5,Vss); size_t g[3]={(size_t)D/8,(size_t)Skv,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_copy,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
      timed("qk",[&]{ SM(k_qk,0,imgKt);SM(k_qk,1,dQp);SM(k_qk,2,dS);SM(k_qk,3,dIdx);SI(k_qk,4,Sq);SI(k_qk,5,Skv);SI(k_qk,6,H);SI(k_qk,7,num_qb);SI(k_qk,8,top_k);SI(k_qk,9,BK);SI(k_qk,10,BQ);SF(k_qk,11,scale);SI(k_qk,12,causal); size_t g[3]={(size_t)sel/8,(size_t)BQ/4,(size_t)H*num_qb}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_qk,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
      timed("softmax",[&]{ SM(k_sm,0,dS);SI(k_sm,1,H);SI(k_sm,2,num_qb);SI(k_sm,3,top_k);SI(k_sm,4,BK);SI(k_sm,5,BQ); size_t g[3]={(size_t)LW,(size_t)BQ,(size_t)H*num_qb}; size_t l[3]={(size_t)LW,1,1}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_sm,3,nullptr,g,l,0,nullptr,nullptr)); });
      timed("pv",[&]{ SM(k_pv,0,imgVc);SM(k_pv,1,dS);SM(k_pv,2,dO);SM(k_pv,3,dIdx);SI(k_pv,4,Sq);SI(k_pv,5,Skv);SI(k_pv,6,H);SI(k_pv,7,num_qb);SI(k_pv,8,top_k);SI(k_pv,9,BK);SI(k_pv,10,BQ); size_t g[3]={(size_t)D/8,(size_t)BQ/4,(size_t)H*num_qb}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pv,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
    }
    std::vector<float> ref; cpu_ref(0,Sq,Skv,D,num_qb,top_k,BK,BQ,Q.data(),Kk.data(),V.data(),idx.data(),scale,ref);
    double mx=0,sm=0; int bad=0;
    for(long i=0;i<(long)Sq*D;++i){ float g=(float)Oh[i]; if(std::isnan(g)||std::isinf(g)){++bad;continue;} double d=std::fabs((double)g-(double)ref[i]); mx=std::max(mx,d); sm+=d; }

    // Op-level validation: build Tensors from the SAME data and drive
    // OpenCLBlockSparseAttentionOp; compare head 0 to the CPU reference.
    if (const char* oe=std::getenv("BSA_OP_TEST"); oe && oe[0]=='1') {
      using mllm::Tensor; using mllm::kFloat16; using mllm::kInt32; using mllm::kCPU; using mllm::kOpenCL;
      auto mkT=[&](const std::vector<__fp16>& src, int S){ return Tensor::fromVector<__fp16>(src,{1,H,S,D},kFloat16,kCPU).to(kOpenCL); };
      Tensor tQ=mkT(Q,Sq), tK=mkT(Kk,Skv), tV=mkT(V,Skv);
      Tensor tId=Tensor::fromVector<int>(idx,{H,num_qb,top_k},kInt32,kCPU).to(kOpenCL);
      mllm::aops::BlockSparseAttentionOpOptions o{}; o.B=1; o.q_head=H; o.kv_head=H; o.D=D; o.BK=BK; o.causal_mask=true;
      mllm::opencl::OpenCLBlockSparseAttentionOp op(o);
      std::vector<Tensor> in{tQ,tK,tV,tId}, out; op.reshape(in,out); op.setup(in,out); op.forward(in,out);
      CL_CHECK(OpenCLLoader::instance().clFinish(q));
      Tensor oc=out[0].to(kCPU); const __fp16* op_o=oc.ptr<__fp16>();
      double omx=0; int obad=0;
      for(long i=0;i<(long)Sq*D;++i){ float g=(float)op_o[i]; if(std::isnan(g)||std::isinf(g)){++obad;continue;} omx=std::max(omx,std::fabs((double)g-(double)ref[i])); }
      fmt::print("   [op] Sq={:5d} max_abs={:.3e} nan={}\n",Sq,omx,obad);
    }

    auto t0=std::chrono::high_resolution_clock::now(); for(int i=0;i<reps;++i) run(); CL_CHECK(OpenCLLoader::instance().clFinish(q)); auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
    // FLOPs over SELECTED keys (causal-clamped average per query ~ min(sel, q_pos+1)).
    double sel_pairs=0; for(int qb=0;qb<num_qb;++qb){ int qmax=(Skv-Sq)+qb*BQ+BQ-1; sel_pairs += (double)BQ*std::min(sel,qmax+1); }
    double flops=4.0*H*D*sel_pairs;
    fmt::print("Sq={:5d} Skv={:5d}  {:8.3f} ms  {:8.1f} GF/s(sel)  max_abs={:.3e} mean={:.2e} nan={}\n", Sq,Skv,ms,flops/(ms*1e6),mx,sm/((double)Sq*D),bad);
    for(cl_mem m:{dQ,dK,dV,dIdx,dQp,dKt,dVc,dS,dO,imgKt,imgVc}) OpenCLLoader::instance().clReleaseMemObject(m);
  }
  for(cl_kernel k:{k_pack,k_trans,k_copy,k_qk,k_sm,k_pv}) OpenCLLoader::instance().clReleaseKernel(k);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
