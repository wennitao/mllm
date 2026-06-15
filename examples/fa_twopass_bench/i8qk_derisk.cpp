// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Stage-3 GO/NO-GO derisk: does an int8-DP4A QK^T GEMM beat the fp16-image QK
// GEMM (~710 GF/s, Stage 1) at attention's K=128 reduction? DP4A is ~4x the
// fp16 ALU but wants K d-contiguous (4 int8 packed per uint), NOT the d-major
// transpose the fp16 image path uses, and gives up the texture-cached A operand.
// This measures both on the same harness at the QK shape (M=Skv, N=Sq, K=128)
// and reports GF/s + accuracy vs an fp32 reference.
//
// Layout for DP4A: K natural [k, d] (d contiguous), Q natural [q, d]. Each lane
// computes an 8-key x 4-query tile, reducing d in groups of 4 (32 dot4/group,
// 32 groups). acc int32, dequant s = dot * (q_scale * k_scale) in the epilogue.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <CL/cl.h>
#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {

const char* kSrc = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

#ifndef D
#define D 128
#endif

// int8 DP4A QK: A = K [Skv, D] int8 (d contiguous), B = Q [Sq, D] int8.
// Each lane: 8 keys (m) x 4 queries (n). dst S[q*Skv + k] = dot * dq (fp16).
// global = (Skv/8, Sq/4); A,B read as packed uint (4 int8 / uint).
__kernel void i8_qk(__global const char* K, __global const char* Q,
                    __global half* S, const int Sq, const int Skv,
                    const float dq) {
  const int gy = get_global_id(0);   // 8 keys
  const int gx = get_global_id(1);   // 4 queries
  if (gy * 8 >= Skv || gx * 4 >= Sq) return;
  const int gy_8 = gy << 3, gx_4 = gx << 2;
  // packed-uint views: row stride D/4 uints.
  __global const uint* Kp = (__global const uint*)K;
  __global const uint* Qp = (__global const uint*)Q;
  const int Du = D >> 2;

  int acc[4][8];
  #pragma unroll
  for (int n = 0; n < 4; ++n)
    #pragma unroll
    for (int m = 0; m < 8; ++m) acc[n][m] = 0;

  for (int g = 0; g < Du; ++g) {
    uint kk[8], qq[4];
    #pragma unroll
    for (int m = 0; m < 8; ++m) kk[m] = Kp[(long)(gy_8 + m) * Du + g];
    #pragma unroll
    for (int n = 0; n < 4; ++n) qq[n] = Qp[(long)(gx_4 + n) * Du + g];
    #pragma unroll
    for (int n = 0; n < 4; ++n)
      #pragma unroll
      for (int m = 0; m < 8; ++m) acc[n][m] = dot_acc_sat_4x8packed_ss_int(qq[n], kk[m], acc[n][m]);
  }
  #pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int q = gx_4 + n;
    if (q < Sq) {
      half8 hv;
      hv.s0 = (half)((float)acc[n][0] * dq); hv.s1 = (half)((float)acc[n][1] * dq);
      hv.s2 = (half)((float)acc[n][2] * dq); hv.s3 = (half)((float)acc[n][3] * dq);
      hv.s4 = (half)((float)acc[n][4] * dq); hv.s5 = (half)((float)acc[n][5] * dq);
      hv.s6 = (half)((float)acc[n][6] * dq); hv.s7 = (half)((float)acc[n][7] * dq);
      vstore8(hv, 0, S + (long)q * Skv + gy_8);
    }
  }
}

// fp16-image reference QK (the Stage-1/Stage-2 path) for same-harness A/B.
// A = Kt image [d, k], B = Qp packed [d/4, q, 4]. dst S[q*Skv+k].
__kernel void f16_qk(__read_only image1d_buffer_t Kt, __global const half* Qpk,
                     __global half* S, const int Sq, const int Skv, const float scale) {
  const int gy = get_global_id(0);
  const int gx = get_global_id(1);
  if (gy * 8 >= Skv || gx * 4 >= Sq) return;
  const int gx_4 = gx << 2, gy_8 = gy << 3, M_4 = Skv >> 2;
  float8 c0 = (float8)0, c1 = (float8)0, c2 = (float8)0, c3 = (float8)0;
  half8 B0, B1, B2, B3;
  for (int i = 0; i < D; i += 4) {
    const int t = gy * 2 + i * M_4;
    B0.s0123 = read_imageh(Kt, t);      B0.s4567 = read_imageh(Kt, t + 1);
    const int t1 = gy * 2 + (i + 1) * M_4;
    B1.s0123 = read_imageh(Kt, t1);     B1.s4567 = read_imageh(Kt, t1 + 1);
    const int t2 = gy * 2 + (i + 2) * M_4;
    B2.s0123 = read_imageh(Kt, t2);     B2.s4567 = read_imageh(Kt, t2 + 1);
    const int t3 = gy * 2 + (i + 3) * M_4;
    B3.s0123 = read_imageh(Kt, t3);     B3.s4567 = read_imageh(Kt, t3 + 1);
    half16 w = vload16(0, Qpk + ((long)(i >> 2) * Sq + gx_4) * 4);
    const float8 b0 = convert_float8(B0), b1 = convert_float8(B1), b2 = convert_float8(B2), b3 = convert_float8(B3);
    c0 += b0*w.s0; c0 += b1*w.s1; c0 += b2*w.s2; c0 += b3*w.s3;
    c1 += b0*w.s4; c1 += b1*w.s5; c1 += b2*w.s6; c1 += b3*w.s7;
    c2 += b0*w.s8; c2 += b1*w.s9; c2 += b2*w.sa; c2 += b3*w.sb;
    c3 += b0*w.sc; c3 += b1*w.sd; c3 += b2*w.se; c3 += b3*w.sf;
  }
  #define EM(NN,CV) { const int q=gx_4+(NN); if(q<Sq) vstore8(convert_half8((CV)*scale),0,S+(long)q*Skv+gy_8); }
  EM(0,c0); EM(1,c1); EM(2,c2); EM(3,c3);
  #undef EM
}
)CL";

#define CL_CHECK(e) do { cl_int _e=(e); if(_e!=CL_SUCCESS){ fmt::print(stderr,"CL err {} at {}:{}\n",_e,__FILE__,__LINE__); std::exit(1);} } while(0)

cl_mem up(cl_context c, cl_command_queue q, const void* h, size_t b, cl_mem_flags f){ cl_int e; cl_mem m=OpenCLLoader::instance().clCreateBuffer(c,f,b,nullptr,&e); CL_CHECK(e); CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q,m,CL_TRUE,0,b,h,0,nullptr,nullptr)); return m; }

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("h");
  auto& reps_a = Argparse::add<int>("--reps").def(30);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int reps = reps_a.get(), D = 128;

  mllm::initOpenCLBackend();
  auto rt = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(mllm::Context::instance().getBackend(mllm::kOpenCL))->runtime();
  cl_context ctx = rt->context()(); cl_command_queue q = rt->commandQueue()(); cl_device_id dev = rt->getDevices()[0]();

  cl_int err; size_t sl = std::strlen(kSrc);
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(ctx, 1, &kSrc, &sl, &err); CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) { char lg[16384]={0}; size_t n; OpenCLLoader::instance().clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof(lg),lg,&n); fmt::print(stderr,"build fail:\n{}\n",lg); return 1; }
  cl_kernel ki8 = OpenCLLoader::instance().clCreateKernel(prog,"i8_qk",&err); CL_CHECK(err);
  cl_kernel kf16 = OpenCLLoader::instance().clCreateKernel(prog,"f16_qk",&err); CL_CHECK(err);

  struct Sh { int Sq, Skv; };
  std::vector<Sh> shapes = {{1024,1024},{2048,2048},{4096,4096}};
  std::mt19937 rng(7); std::uniform_real_distribution<float> dist(-1.f,1.f);
  constexpr double kPeak = 3.0e12;

  fmt::print("\n=== int8-DP4A QK vs fp16-image QK (K=128, per-head; GF/s) ===\n");
  for (auto sh : shapes) {
    const int Sq=sh.Sq, Skv=sh.Skv;
    std::vector<__fp16> Q((size_t)Sq*D), K((size_t)Skv*D);
    for (auto& x:Q) x=(__fp16)(dist(rng)*0.5f);
    for (auto& x:K) x=(__fp16)(dist(rng)*0.5f);
    // per-tensor int8 quant
    float qmax=1e-6f, kmax=1e-6f;
    for (auto x:Q) qmax=std::max(qmax,std::fabs((float)x));
    for (auto x:K) kmax=std::max(kmax,std::fabs((float)x));
    const float qs=qmax/127.f, ks=kmax/127.f;
    std::vector<int8_t> Qi((size_t)Sq*D), Ki((size_t)Skv*D);
    for (size_t i=0;i<Q.size();++i) Qi[i]=(int8_t)std::max(-127,std::min(127,(int)std::lround((float)Q[i]/qs)));
    for (size_t i=0;i<K.size();++i) Ki[i]=(int8_t)std::max(-127,std::min(127,(int)std::lround((float)K[i]/ks)));
    const float attn_scale = 1.f/std::sqrt((float)D);
    const float dq = qs*ks*attn_scale;  // int dot -> scaled score

    // fp16 path needs Kt[d,k] transposed + Qp[d/4,q,4] packed.
    std::vector<__fp16> Kt((size_t)D*Skv), Qp((size_t)(D/4)*Sq*4);
    for (int k=0;k<Skv;++k) for (int d=0;d<D;++d) Kt[(long)d*Skv+k]=K[(long)k*D+d];
    for (int g=0;g<D/4;++g) for (int qq=0;qq<Sq;++qq) for (int kk=0;kk<4;++kk) Qp[((long)g*Sq+qq)*4+kk]=Q[(long)qq*D+g*4+kk];

    cl_mem dKi=up(ctx,q,Ki.data(),Ki.size(),CL_MEM_READ_ONLY);
    cl_mem dQi=up(ctx,q,Qi.data(),Qi.size(),CL_MEM_READ_ONLY);
    cl_mem dKt=up(ctx,q,Kt.data(),Kt.size()*2,CL_MEM_READ_ONLY);
    cl_mem dQp=up(ctx,q,Qp.data(),Qp.size()*2,CL_MEM_READ_ONLY);
    cl_mem dS=OpenCLLoader::instance().clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)Sq*Skv*2,nullptr,&err); CL_CHECK(err);
    cl_image_format fmt={CL_RGBA,CL_HALF_FLOAT}; cl_image_desc desc={}; desc.image_type=CL_MEM_OBJECT_IMAGE1D_BUFFER; desc.image_width=(size_t)D*Skv/4; desc.buffer=dKt;
    cl_mem imgKt=clCreateImage(ctx,CL_MEM_READ_ONLY,&fmt,&desc,nullptr,&err); CL_CHECK(err);

    auto run_i8=[&](){ CL_CHECK(OpenCLLoader::instance().clSetKernelArg(ki8,0,sizeof(cl_mem),&dKi)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(ki8,1,sizeof(cl_mem),&dQi)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(ki8,2,sizeof(cl_mem),&dS)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(ki8,3,sizeof(int),&Sq)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(ki8,4,sizeof(int),&Skv)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(ki8,5,sizeof(float),&dq)); size_t g[2]={(size_t)Skv/8,(size_t)Sq/4}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,ki8,2,nullptr,g,nullptr,0,nullptr,nullptr)); CL_CHECK(OpenCLLoader::instance().clFinish(q)); };
    auto run_f16=[&](){ CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kf16,0,sizeof(cl_mem),&imgKt)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kf16,1,sizeof(cl_mem),&dQp)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kf16,2,sizeof(cl_mem),&dS)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kf16,3,sizeof(int),&Sq)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kf16,4,sizeof(int),&Skv)); CL_CHECK(OpenCLLoader::instance().clSetKernelArg(kf16,5,sizeof(float),&attn_scale)); size_t g[2]={(size_t)Skv/8,(size_t)Sq/4}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,kf16,2,nullptr,g,nullptr,0,nullptr,nullptr)); CL_CHECK(OpenCLLoader::instance().clFinish(q)); };

    // accuracy: compare both vs fp32 ref for row q=Sq/2 (a few keys).
    auto check=[&](bool i8){
      std::vector<__fp16> S((size_t)Sq*Skv);
      CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q,dS,CL_TRUE,0,S.size()*2,S.data(),0,nullptr,nullptr));
      int qr=Sq/2; double mr=0;
      for (int k=0;k<std::min(Skv,256);++k){ double acc=0; for(int d=0;d<D;++d) acc+=(double)(float)Q[(long)qr*D+d]*(double)(float)K[(long)k*D+d]; double ref=acc*attn_scale; double got=(float)S[(long)qr*Skv+k]; mr=std::max(mr,std::fabs(got-ref)/(std::fabs(ref)+1e-3)); }
      return mr;
    };
    run_i8(); double mri8=check(true);
    run_f16(); double mrf16=check(false);

    auto bench=[&](auto fn){ for(int i=0;i<5;++i) fn(); auto t0=std::chrono::high_resolution_clock::now(); for(int i=0;i<reps;++i) fn(); auto t1=std::chrono::high_resolution_clock::now(); return std::chrono::duration<double,std::milli>(t1-t0).count()/reps; };
    double mi8=bench(run_i8), mf16=bench(run_f16);
    double flops=2.0*Sq*(double)Skv*D;
    fmt::print("Sq={:5d} Skv={:5d}  i8={:7.3f}ms {:7.1f}GF/s ({:.1f}%pk) mrel={:.2e}   f16={:7.3f}ms {:7.1f}GF/s mrel={:.2e}   i8/f16={:.2f}x\n",
               Sq,Skv, mi8, flops/(mi8*1e6), 100*flops/(mi8*1e-3)/kPeak, mri8, mf16, flops/(mf16*1e6), mrf16, mf16/mi8);

    for (cl_mem m:{dKi,dQi,dKt,dQp,dS,imgKt}) OpenCLLoader::instance().clReleaseMemObject(m);
  }
  OpenCLLoader::instance().clReleaseKernel(ki8); OpenCLLoader::instance().clReleaseKernel(kf16);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
