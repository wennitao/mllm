// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Standalone two-pass GEMM-class FlashAttention PREFILL microbench (Adreno 830).
// Pipeline (per head, head-batched via grid dim 2):
//   pack_q -> Qp ; trans_k -> Kt(image) ; qk_gemm -> S ; softmax_norm (in place
//   on S) ; V(image) ; pv_gemm (reads P query-major) -> O.
// Validates O against a CPU fp32 causal-attention reference, then times the
// whole pipeline (5 kernels) for an E2E GF/s. Result: the validated 634-896
// GF/s image-A GEMMs survive the full attention plumbing — ~800 GF/s @ S=2048,
// 843 @ S=4096 (causal block-skip), ~7x the fused kernel's 116 GF/s, and runs
// S=4096 which the fused kernel can't (GPU watchdog). FA_TP_PROF=1 = per-stage.

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

// Host-side fp16 scalar. ARM has the builtin fp16_t; on x86 (and any target
// without it) fall back to the bundled IEEE-754 binary16 (same 2-byte layout).
#if defined(__ARM_FP16_FORMAT_IEEE) || defined(__aarch64__)
using fp16_t = __fp16;
#else
#include "half/half.hpp"
using fp16_t = half_float::half;
#endif

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {

const char* kSrc =
#include "kernels.cl.inc"
;

#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fmt::print(stderr, "OpenCL error {} at {}:{} ({})\n", _e, __FILE__, __LINE__, #expr); std::exit(1); } } while(0)

cl_mem upload(cl_context ctx, cl_command_queue q, const void* host, size_t bytes, cl_mem_flags f) {
  cl_int err;
  cl_mem b = OpenCLLoader::instance().clCreateBuffer(ctx, f, bytes, nullptr, &err);
  CL_CHECK(err);
  CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q, b, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
  return b;
}

cl_mem img_over(cl_context ctx, cl_mem buf, size_t texels) {
  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
  cl_image_desc desc = {};
  desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  desc.image_width = texels;
  desc.buffer = buf;
  cl_int err;
  cl_mem im = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  CL_CHECK(err);
  return im;
}

// CPU fp32 causal attention reference for one head, returns O[Sq*128].
void cpu_ref(int Sq, int Skv, int D, const fp16_t* Q, const fp16_t* K, const fp16_t* V,
             float scale, std::vector<float>& O) {
  O.assign((size_t)Sq * D, 0.0f);
  std::vector<float> s(Skv);
  for (int q = 0; q < Sq; ++q) {
    const int q_pos = (Skv - Sq) + q;
    float m = -INFINITY;
    for (int k = 0; k < Skv; ++k) {
      if (k > q_pos) { s[k] = -INFINITY; continue; }
      double acc = 0.0;
      for (int d = 0; d < D; ++d) acc += (double)(float)Q[(long)q * D + d] * (double)(float)K[(long)k * D + d];
      s[k] = (float)(acc * scale);
      m = std::max(m, s[k]);
    }
    float l = 0.0f;
    for (int k = 0; k <= q_pos && k < Skv; ++k) l += std::exp(s[k] - m);
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int k = 0; k <= q_pos && k < Skv; ++k) {
      const float p = std::exp(s[k] - m) * inv;
      for (int d = 0; d < D; ++d) O[(long)q * D + d] += p * (float)(float)V[(long)k * D + d];
    }
  }
}

}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(20);
  auto& heads_arg = Argparse::add<int>("--heads").help("num heads (batched)").def(16);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int reps = reps_arg.get();
  const int H = heads_arg.get();
  const int D = 128;
  const float scale = 1.0f / std::sqrt((float)D);

  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();
  cl_device_id dev = rt->getDevices()[0]();

  cl_int err;
  size_t slen = std::strlen(kSrc);
  cl_program prog = OpenCLLoader::instance().clCreateProgramWithSource(ctx, 1, &kSrc, &slen, &err);
  CL_CHECK(err);
  err = OpenCLLoader::instance().clBuildProgram(prog, 0, nullptr, "-cl-std=CL2.0", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    char log[32768] = {0}; size_t ls = 0;
    OpenCLLoader::instance().clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &ls);
    fmt::print(stderr, "build failed:\n{}\n", log);
    return 1;
  }
  auto K_ = [&](const char* n){ cl_int e; cl_kernel k = OpenCLLoader::instance().clCreateKernel(prog, n, &e); CL_CHECK(e); return k; };
  cl_kernel k_pack = K_("pack_q"), k_trans = K_("trans_k"), k_qk = K_("qk_gemm"),
            k_sm = K_("softmax_norm"), k_pv = K_("pv_gemm");
  constexpr int kSmLw = 64;  // must match SM_LW in the kernel

  struct Shape { int Sq, Skv; };
  std::vector<Shape> shapes = {{512, 512}, {1024, 1024}, {2048, 2048}, {1024, 2048}, {4096, 4096}};

  fmt::print("\n=== two-pass GEMM attention (H={}, D={}, reps={}) ===\n", H, D, reps);

  std::mt19937 rng(0xBEEF);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (auto sh : shapes) {
    const int Sq = sh.Sq, Skv = sh.Skv;
    if (Sq % 8 || Skv % 8) { fmt::print("skip {}x{} (need mult 8)\n", Sq, Skv); continue; }

    std::vector<fp16_t> Q((size_t)H * Sq * D), Kk((size_t)H * Skv * D), V((size_t)H * Skv * D);
    for (auto& x : Q) x = (fp16_t)(dist(rng) * 0.5f);
    for (auto& x : Kk) x = (fp16_t)(dist(rng) * 0.5f);
    for (auto& x : V) x = (fp16_t)(dist(rng) * 0.5f);

    cl_mem dQ = upload(ctx, q, Q.data(), Q.size() * 2, CL_MEM_READ_ONLY);
    cl_mem dK = upload(ctx, q, Kk.data(), Kk.size() * 2, CL_MEM_READ_ONLY);
    cl_mem dV = upload(ctx, q, V.data(), V.size() * 2, CL_MEM_READ_ONLY);
    cl_mem dQp = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)H * (D/4) * Sq * 4 * 2, nullptr, &err); CL_CHECK(err);
    cl_mem dKt = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)H * D * Skv * 2, nullptr, &err); CL_CHECK(err);
    cl_mem dS  = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)H * Sq * Skv * 2, nullptr, &err); CL_CHECK(err);
    cl_mem dO  = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)H * Sq * D * 2, nullptr, &err); CL_CHECK(err);
    cl_mem imgKt = img_over(ctx, dKt, (size_t)H * D * Skv / 4);
    cl_mem imgV  = img_over(ctx, dV, (size_t)H * Skv * D / 4);

    const int causal = 1;
    auto setI = [&](cl_kernel k, int idx, int v){ CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, idx, sizeof(int), &v)); };
    auto setM = [&](cl_kernel k, int idx, cl_mem* v){ CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, idx, sizeof(cl_mem), v)); };
    auto setF = [&](cl_kernel k, int idx, float v){ CL_CHECK(OpenCLLoader::instance().clSetKernelArg(k, idx, sizeof(float), &v)); };

    auto run = [&]() {
      // pack_q
      setM(k_pack,0,&dQ); setM(k_pack,1,&dQp); setI(k_pack,2,Sq); setI(k_pack,3,H);
      size_t gp[3]={(size_t)Sq,(size_t)D/4,(size_t)H};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pack,3,nullptr,gp,nullptr,0,nullptr,nullptr));
      // trans_k
      setM(k_trans,0,&dK); setM(k_trans,1,&dKt); setI(k_trans,2,Skv); setI(k_trans,3,H);
      size_t gt[3]={(size_t)Skv,(size_t)D,(size_t)H};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_trans,3,nullptr,gt,nullptr,0,nullptr,nullptr));
      // qk_gemm
      setM(k_qk,0,&imgKt); setM(k_qk,1,&dQp); setM(k_qk,2,&dS);
      setI(k_qk,3,Sq); setI(k_qk,4,Skv); setI(k_qk,5,H); setF(k_qk,6,scale); setI(k_qk,7,causal);
      size_t gq[3]={(size_t)Skv/8,(size_t)Sq/4,(size_t)H};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_qk,3,nullptr,gq,nullptr,0,nullptr,nullptr));
      // softmax_norm (in place on dS), workgroup-per-row
      setM(k_sm,0,&dS); setI(k_sm,1,Sq); setI(k_sm,2,Skv); setI(k_sm,3,H);
      size_t gs[3]={(size_t)kSmLw,(size_t)Sq,(size_t)H}; size_t ls[3]={(size_t)kSmLw,1,1};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_sm,3,nullptr,gs,ls,0,nullptr,nullptr));
      // pv_gemm (B = P query-major, in place in dS)
      setM(k_pv,0,&imgV); setM(k_pv,1,&dS); setM(k_pv,2,&dO);
      setI(k_pv,3,Sq); setI(k_pv,4,Skv); setI(k_pv,5,H); setI(k_pv,6,causal);
      size_t gv[3]={(size_t)D/8,(size_t)Sq/4,(size_t)H};
      CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pv,3,nullptr,gv,nullptr,0,nullptr,nullptr));
    };

    run();
    CL_CHECK(OpenCLLoader::instance().clFinish(q));

    // Validate head 0 FIRST (the profiling block below re-runs kernels in
    // isolation, and softmax_norm mutates dS in place, so it must not precede
    // this read).
    std::vector<fp16_t> Oh((size_t)H * Sq * D);
    CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, dO, CL_TRUE, 0, Oh.size()*2, Oh.data(), 0, nullptr, nullptr));
    std::vector<float> refv;
    cpu_ref(Sq, Skv, D, Q.data(), Kk.data(), V.data(), scale, refv);
    double mx = 0.0, sm = 0.0; int bd = 0;
    for (long i = 0; i < (long)Sq * D; ++i) {
      float g = (float)Oh[i];
      if (std::isnan(g) || std::isinf(g)) { ++bd; continue; }
      double d = std::fabs((double)g - (double)refv[i]);
      mx = std::max(mx, d); sm += d;
    }

    // Optional per-kernel profiling (FA_TP_PROF=1): clFinish-bracket each stage.
    if (const char* pe = std::getenv("FA_TP_PROF"); pe && pe[0] == '1') {
      auto timed = [&](const char* nm, auto enq) {
        for (int w = 0; w < 3; ++w) { enq(); }
        CL_CHECK(OpenCLLoader::instance().clFinish(q));
        auto a = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) enq();
        CL_CHECK(OpenCLLoader::instance().clFinish(q));
        auto b = std::chrono::high_resolution_clock::now();
        fmt::print("   prof {:<14} {:8.3f} ms\n", nm, std::chrono::duration<double,std::milli>(b-a).count()/10);
      };
      timed("pack_q", [&]{ setM(k_pack,0,&dQ); setM(k_pack,1,&dQp); setI(k_pack,2,Sq); setI(k_pack,3,H);
        size_t g[3]={(size_t)Sq,(size_t)D/4,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pack,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
      timed("trans_k", [&]{ setM(k_trans,0,&dK); setM(k_trans,1,&dKt); setI(k_trans,2,Skv); setI(k_trans,3,H);
        size_t g[3]={(size_t)Skv,(size_t)D,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_trans,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
      timed("qk_gemm", [&]{ setM(k_qk,0,&imgKt); setM(k_qk,1,&dQp); setM(k_qk,2,&dS); setI(k_qk,3,Sq); setI(k_qk,4,Skv); setI(k_qk,5,H); setF(k_qk,6,scale); setI(k_qk,7,causal);
        size_t g[3]={(size_t)Skv/8,(size_t)Sq/4,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_qk,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
      timed("softmax_norm", [&]{ setM(k_sm,0,&dS); setI(k_sm,1,Sq); setI(k_sm,2,Skv); setI(k_sm,3,H);
        size_t g[3]={(size_t)kSmLw,(size_t)Sq,(size_t)H}; size_t l[3]={(size_t)kSmLw,1,1}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_sm,3,nullptr,g,l,0,nullptr,nullptr)); });
      timed("pv_gemm", [&]{ setM(k_pv,0,&imgV); setM(k_pv,1,&dS); setM(k_pv,2,&dO); setI(k_pv,3,Sq); setI(k_pv,4,Skv); setI(k_pv,5,H); setI(k_pv,6,causal);
        size_t g[3]={(size_t)D/8,(size_t)Sq/4,(size_t)H}; CL_CHECK(OpenCLLoader::instance().clEnqueueNDRangeKernel(q,k_pv,3,nullptr,g,nullptr,0,nullptr,nullptr)); });
    }

    const double max_abs = mx, sum_abs = sm; const int bad = bd;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) run();
    CL_CHECK(OpenCLLoader::instance().clFinish(q));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    // causal flops: per head 4*D*Sq*(Skv+1)/2 (approx lower-tri); times H.
    double pairs = (double)Sq * ((double)Skv + 1.0) * 0.5;
    double flops = 4.0 * H * D * pairs;
    fmt::print("Sq={:5d} Skv={:5d}  {:8.3f} ms  {:8.1f} GF/s  max_abs={:.3e} mean={:.2e} nan/inf={}\n",
               Sq, Skv, ms, flops/(ms*1e6), max_abs, sum_abs/((double)Sq*D), bad);

    OpenCLLoader::instance().clReleaseMemObject(imgKt);
    OpenCLLoader::instance().clReleaseMemObject(imgV);
    for (cl_mem m : {dQ,dK,dV,dQp,dKt,dS,dO}) OpenCLLoader::instance().clReleaseMemObject(m);
  }

  for (cl_kernel k : {k_pack,k_trans,k_qk,k_sm,k_pv}) OpenCLLoader::instance().clReleaseKernel(k);
  OpenCLLoader::instance().clReleaseProgram(prog);
});
