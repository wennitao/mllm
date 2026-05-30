// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Layer-level E2E test for the OpenCL LPBQ Linear path on REAL model weights.
// For each Qwen3 projection (q/k/v/o/gate/up/down) of one layer it:
//   - loads the real .mllm .weight/.scale1/.scale2,
//   - prepacks to the ushort + combined-scale layout (the validated bench path),
//   - runs FP32 activations through OpenCLLinearOp::setLPBQ + runLPBQ(io_fp32=true)
//     — exercising the fp32->fp16 (in) and fp16->fp32 (out) conversions, and
//   - compares the fp32 GPU output to the CPU reference (lpbq_matmul_fp16_packed).
// Validates the op + dtype conversion + real-weight prepack, without the full
// model generation loop.
//
// Usage: ./mllm-qwen3-opencl-linear-lpbq-layer --params qwen3_1.7b_ptq_lpbq.mllm [--layer 0]

#include <CL/cl.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <fmt/core.h>

#include <mllm/mllm.hpp>
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/ops/LinearOp.hpp"
#include "mllm/backends/opencl/runtime/OpenCLLoader.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/backends/cpu/kernels/common/lpbq_matmul.hpp"

using mllm::Argparse;
using mllm::opencl::OpenCLLoader;

namespace {
#define CL_CHECK(expr) do { cl_int _e = (expr); if (_e != CL_SUCCESS) { \
    fmt::print(stderr, "OpenCL error {} at {}:{}\n", _e, __FILE__, __LINE__); std::exit(1); } } while(0)

cl_mem upload(cl_context ctx, cl_command_queue q, const void* host, size_t bytes, cl_mem_flags f) {
  cl_int err; cl_mem b = OpenCLLoader::instance().clCreateBuffer(ctx, f, bytes, nullptr, &err); CL_CHECK(err);
  CL_CHECK(OpenCLLoader::instance().clEnqueueWriteBuffer(q, b, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
  return b;
}
}  // namespace

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& params_arg = Argparse::add<std::string>("--params").def("qwen3_1.7b_ptq_lpbq.mllm");
  auto& layer_arg = Argparse::add<int>("--layer").def(0);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int L = layer_arg.get();

  fmt::print("[lpbq-layer-test] loading {}\n", params_arg.get());
  auto params = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  fmt::print("[lpbq-layer-test] init OpenCL backend\n");
  mllm::initOpenCLBackend();
  auto clBE = std::static_pointer_cast<mllm::opencl::OpenCLBackend>(
      mllm::Context::instance().getBackend(mllm::kOpenCL));
  auto rt = clBE->runtime();
  cl_context ctx = rt->context()();
  cl_command_queue q = rt->commandQueue()();

  // Qwen3-1.7B geometry.
  const int H = 2048, Hq = 16 * 128, Hkv = 8 * 128, I = 6144;
  struct Spec { const char* tag; std::string prefix; int K, N; };
  std::string base = "model.layers." + std::to_string(L);
  std::vector<Spec> projs = {
    {"q", base + ".self_attn.q_proj", H, Hq},   {"k", base + ".self_attn.k_proj", H, Hkv},
    {"v", base + ".self_attn.v_proj", H, Hkv},  {"o", base + ".self_attn.o_proj", Hq, H},
    {"gate", base + ".mlp.gate_proj", H, I},    {"up", base + ".mlp.up_proj", H, I},
    {"down", base + ".mlp.down_proj", I, H},
  };

  std::mt19937 rng(0x77);
  std::uniform_real_distribution<float> ad(-0.1f, 0.1f);

  int failures = 0;
  fmt::print("\n=== OpenCL LPBQ LinearOp on REAL weights (fp32 IO) vs CPU, layer {} ===\n", L);
  fmt::print("  {:<5} {:>5} {:>5} {:>5}   {:>10}   {:>10}   {:>8}\n", "proj", "M", "K", "N", "max_abs", "max_ref", "result");

  for (auto& p : projs) {
    const int K = p.K, N = p.N;
    auto w_t = params->pull(p.prefix + ".weight");
    auto s1_t = params->pull(p.prefix + ".scale1");
    auto s2_t = params->pull(p.prefix + ".scale2");
    const int num_blocks = (int)((size_t)s1_t.numel() / (size_t)N);
    const int Bs = K / num_blocks;
    if (Bs != 16) { fmt::print("  {:<5} skip (Bs={})\n", p.tag, Bs); continue; }

    // Prepack (validated bench layouts).
    std::vector<uint16_t> w_ushort((size_t)K * N / 4);
    mllm::cpu::lpbq_prepack_weights_USHORT4(K, N, w_t.ptr<uint8_t>(), w_ushort.data());
    std::vector<uint16_t> comb((size_t)N * num_blocks);
    mllm::cpu::lpbq_prepack_combined_scales(N, num_blocks, s1_t.ptr<uint8_t>(), s2_t.ptr<float>(), comb.data());
    std::vector<uint8_t> w_packed((size_t)K * N);  // [N,K] for CPU ref
    mllm::cpu::lpbq_prepack_weights_HWOI(K, N, w_t.ptr<uint8_t>(), w_packed.data());

    cl_mem d_W = upload(ctx, q, w_ushort.data(), w_ushort.size() * 2, CL_MEM_READ_ONLY);
    cl_mem d_S = upload(ctx, q, comb.data(), comb.size() * 2, CL_MEM_READ_ONLY);

    mllm::aops::LinearOpOptions opt;
    opt.in_channels = K; opt.out_channels = N; opt.bias = false;
    opt.impl_type = mllm::aops::LinearImplTypes::kDefault;
    mllm::opencl::OpenCLLinearOp op(opt);
    op.setLPBQ(d_W, d_S, K, N, Bs);

    for (int M : {1, 512}) {
      std::vector<float> act_f32((size_t)M * K);
      for (auto& x : act_f32) x = ad(rng);
      std::vector<__fp16> act_f16(act_f32.size());
      for (size_t i = 0; i < act_f32.size(); ++i) act_f16[i] = (__fp16)act_f32[i];

      std::vector<__fp16> out_cpu((size_t)M * N);
      mllm::cpu::lpbq_matmul_fp16_packed(M, N, K, Bs, act_f16.data(), w_packed.data(),
                                         s1_t.ptr<uint8_t>(), s2_t.ptr<float>(), nullptr, out_cpu.data());

      cl_mem d_in = upload(ctx, q, act_f32.data(), act_f32.size() * 4, CL_MEM_READ_ONLY);
      cl_int err;
      cl_mem d_out = OpenCLLoader::instance().clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)M * N * 4, nullptr, &err);
      CL_CHECK(err);
      op.runLPBQ(d_in, d_out, M, /*io_fp32=*/true);
      std::vector<float> out_gpu((size_t)M * N);
      CL_CHECK(OpenCLLoader::instance().clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, out_gpu.size() * 4, out_gpu.data(), 0, nullptr, nullptr));

      double max_abs = 0.0, max_ref = 0.0;
      int N_check = std::min(64, N);
      for (int n = 0; n < N_check; ++n) {
        double g = out_gpu[n], r = (float)out_cpu[n];
        max_abs = std::max(max_abs, std::abs(g - r));
        max_ref = std::max(max_ref, std::abs(r));
      }
      bool ok = max_abs < 3e-2 * std::max(1e-3, max_ref) + 5e-3;
      if (!ok) ++failures;
      fmt::print("  {:<5} {:>5} {:>5} {:>5}   {:>.3e}   {:>.3e}   {:>8}\n",
                 p.tag, M, K, N, max_abs, max_ref, ok ? "PASS" : "FAIL");

      OpenCLLoader::instance().clReleaseMemObject(d_in);
      OpenCLLoader::instance().clReleaseMemObject(d_out);
    }
    OpenCLLoader::instance().clReleaseMemObject(d_W);
    OpenCLLoader::instance().clReleaseMemObject(d_S);
  }

  fmt::print("\n{}\n", failures == 0 ? "ALL PASS" : fmt::format("{} FAILURES", failures));
  return failures == 0 ? 0 : 1;
});
