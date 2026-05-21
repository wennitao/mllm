// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Microbenchmark for the XAttention antidiagonal block-selection SCORING pass
// (docs/qnn_backend/block_selection.md). Measures how expensive it is to score
// and top-k-select historical K/V blocks for a query block, across the three
// on-device compute units (ARM CPU / Hexagon NPU / Adreno GPU) via mllm's
// device-switchable Functional ops.
//
// The scored pass per head, with Q,K in [L, d], block B, stride S (S | B):
//   Kr = reshape(K, [L/S, S*d])                 # pack S consecutive keys
//   Qr = reshape(Q, [L/S, S*d])                 # ("inverse"/antidiagonal order
//                                               #  only reorders the feature
//                                               #  concat -> identical timing)
//   A  = softmax( (Qr @ Kr^T)/(sqrt(d)*S) + causal )   # [L/S, L/S]
//   P  = A.reshape([Nb, B/S, Nb, B/S]).sum(B/S dims)    # [Nb, Nb] block scores
//   sel = topk(P, k, dim=-1)                            # selected block ids
//
// QKᵀ MACs = Hq·L²·d / S  (a factor-S reduction vs dense-score Hq·L²·d).
//
// Usage:
//   mllm-bench-block-selection --device cpu  --L 256,512,1024,2048,4096
//   mllm-bench-block-selection --device opencl --stride 8 --topk 8

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/nn/Functional.hpp>

using mllm::Argparse;
using mllm::Tensor;
namespace F = mllm::nn::functional;

namespace {

std::vector<int> parseIntList(const std::string& s) {
  std::vector<int> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(std::atoi(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return out;
}

// SCORING CORE (device-resident, the dominant + parallelizable work):
//   A = softmax( Qr @ Kr^T )   over keys, [1,Hq,Lr,Lr]
// The 1/(sqrt(d)*S) scale is folded into Q at setup so no scalar-mul op is
// needed (OpenCL has no scalar mul). The causal mask is omitted from the timed
// core: it's an elementwise add that changes neither the matmul nor the softmax
// cost, and this phase measures raw scoring speed, not selection correctness.
// Returns A still on `dev`; caller forces completion.
Tensor scoreCore(const Tensor& Qr_scaled, const Tensor& Kr, mllm::aops::MatMulOpType mm) {
  Tensor A = F::matmul(Qr_scaled, Kr, /*transpose_A=*/false, /*transpose_B=*/true, mm);  // [1,Hq,Lr,Lr]
  A = F::softmax(A, -1);
  return A;
}

// POOL + SELECT (always on CPU; cheap O(Hq·Nb²)). `A` may live on a device — the
// .to(kCPU) readback is part of this cost. Pools the reduced [Lr,Lr] grid into
// [Nb,Nb] block scores and top-k-selects per query block.
Tensor poolSelect(Tensor A, int Hq, int Nb, int bs, int topk) {
  Tensor Ac = A.to(mllm::kCPU);
  Tensor P = Ac.view({1, Hq, Nb, bs, Nb, bs});
  P = F::sum(P, 5, /*keep_dim=*/false);  // [1,Hq,Nb,bs,Nb]
  P = F::sum(P, 3, /*keep_dim=*/false);  // [1,Hq,Nb,Nb]
  auto sel = F::topk(P, std::min(topk, Nb), -1, /*largest=*/true, /*sorted=*/true);
  return sel[1];
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& device_arg = Argparse::add<std::string>("--device").help("cpu | opencl").def("cpu");
  auto& heads_arg = Argparse::add<int>("--heads").help("num query heads Hq").def(16);
  auto& dim_arg = Argparse::add<int>("--dim").help("head_dim d").def(128);
  auto& block_arg = Argparse::add<int>("--block").help("block size B").def(32);
  auto& stride_arg = Argparse::add<int>("--stride").help("stride S (must divide B)").def(8);
  auto& topk_arg = Argparse::add<int>("--topk").help("blocks selected per query block").def(8);
  auto& L_arg = Argparse::add<std::string>("--L").help("comma list of seq lens").def("256,512,1024,2048,4096");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(50);
  auto& warmup_arg = Argparse::add<int>("--warmup").help("warmup reps").def(10);
  auto& mm_arg = Argparse::add<std::string>("--matmul").help("Default|BLAS|MllmBlas").def("MllmBlas");
  auto& fp16_arg = Argparse::add<bool>("--fp16").help("run core in fp16 (Adreno-friendly)").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  const std::string device = device_arg.get();
  mllm::DeviceTypes dev = mllm::kCPU;
  if (device == "opencl") {
#if BENCH_HAS_OPENCL
    mllm::initOpenCLBackend();
    dev = mllm::kOpenCL;
#else
    fmt::print("Built without OpenCL backend (set MLLM_BUILD_OPENCL_BACKEND=ON)\n");
    return 1;
#endif
  } else if (device != "cpu") {
    fmt::print("Unknown device {} (use cpu|opencl)\n", device);
    return 1;
  }

  const int Hq = heads_arg.get(), d = dim_arg.get(), B = block_arg.get(), S = stride_arg.get(), topk = topk_arg.get();
  const int reps = reps_arg.get(), warmup = warmup_arg.get();
  const auto mm = mllm::aops::str2MatMulOpType(mm_arg.get());
  if (B % S != 0) {
    fmt::print("stride S={} must divide block B={}\n", S, B);
    return 1;
  }

  const float scale = 1.0f / (std::sqrt((float)d) * (float)S);

  fmt::print("=== XAttention block-selection scoring benchmark ===\n");
  fmt::print("device={} Hq={} d={} B={} S={} topk={} matmul={} reps={} (warmup={})\n", device, Hq, d, B, S, topk,
             mm_arg.get(), reps, warmup);
  fmt::print("core = matmul+softmax on {}; pool+topk always on CPU (incl. readback)\n", device);
  fmt::print("{:>7} {:>5} {:>6} {:>11} {:>11} {:>11} {:>11} {:>9}\n", "L", "Nb", "Lr", "core_ms", "pool_ms", "total_ms",
             "core_GF/s", "vs_dense");

  auto bench_phase = [&](auto fn) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0 / reps;
  };

  for (int L : parseIntList(L_arg.get())) {
    if (L % B != 0) {
      fmt::print("{:>7}  (skipped: L not divisible by B)\n", L);
      continue;
    }
    const int Lr = L / S, Nb = L / B, bs = B / S;
    // Pre-scale Q (fold the softmax scale), reshape to [1,Hq,Lr,S*d], move to dev.
    Tensor Q = Tensor::random({1, Hq, L, d}, -1.f, 1.f, mllm::kFloat32, mllm::kCPU).mul(scale);
    Tensor K = Tensor::random({1, Hq, L, d}, -1.f, 1.f, mllm::kFloat32, mllm::kCPU);
    Tensor Qr = Q.view({1, Hq, Lr, S * d});
    Tensor Kr = K.view({1, Hq, Lr, S * d});
    if (fp16_arg.get()) {
      Qr = Qr.to(mllm::kFloat16);
      Kr = Kr.to(mllm::kFloat16);
    }
    if (dev != mllm::kCPU) {
      Qr = Qr.to(dev);
      Kr = Kr.to(dev);
    }

    double core_ms = bench_phase([&] { (void)scoreCore(Qr, Kr, mm).to(mllm::kCPU); });
    Tensor A = scoreCore(Qr, Kr, mm);  // realized A for the pool phase
    double pool_ms = bench_phase([&] { (void)poolSelect(A, Hq, Nb, bs, topk); });

    double macs = (double)Hq * Lr * Lr * (S * d);   // Qr@Kr^T MACs
    double core_gf = 2.0 * macs / (core_ms * 1e-3) / 1e9;
    fmt::print("{:>7} {:>5} {:>6} {:>11.4f} {:>11.4f} {:>11.4f} {:>11.1f} {:>8.1f}x\n", L, Nb, Lr, core_ms, pool_ms,
               core_ms + pool_ms, core_gf, (double)S);
  }
});
