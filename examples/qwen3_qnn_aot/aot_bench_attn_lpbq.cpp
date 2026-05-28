// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Driver for the fake-scale w4a16-LPBQ attention-projection microbench bins
// produced by compile_attn_lpbq_microbench.cpp. Dispatches the single
// "model.0.s<Sq>" graph and reports average latency — feeds the het-pipeline
// simulator's LPBQ q/k/v/o slots in tools/het_sim/qwen3_1p7b_block.py.
//
//   ./mllm-qwen3-aot-attn-lpbq-bench -m qwen3-attn-lpbq-q.bin   --mode q   --sq 1024
//   ./mllm-qwen3-aot-attn-lpbq-bench -m qwen3-attn-lpbq-kv.bin  --mode kv  --sq 1024
//   ./mllm-qwen3-aot-attn-lpbq-bench -m qwen3-attn-lpbq-o.bin   --mode o   --sq 1024
//   ./mllm-qwen3-aot-attn-lpbq-bench -m qwen3-attn-lpbq-qkv.bin --mode qkv --sq 1024
//
// --mode must match the bin's compile-time mode (defines I/O shapes).

#include <chrono>
#include <cstdio>
#include <vector>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::aot::QnnAOTModule;

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("attn LPBQ .bin").def("qwen3-attn-lpbq-q.bin");
  auto& sq_arg = Argparse::add<int>("--sq").help("compiled Sq").def(1024);
  auto& mode_arg = Argparse::add<std::string>("--mode").help("q | kv | o | qkv (must match compiled bin)").def("q");
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps after 1 warmup").def(50);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  // Qwen3-1.7B geometry.
  const int Sq = sq_arg.get();
  const int H = 2048;                    // hidden
  const int q_out = 16 * 128;            // Hq * D = 2048
  const int kv_out = 8 * 128;            // Hkv * D = 1024
  const std::string mode = mode_arg.get();
  const int reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());

  // I/O shapes per mode.
  auto mk = [&](int w) {
    auto x = Tensor::empty({1, Sq, w}, mllm::kFloat16, mllm::kQNN).alloc();
    for (int j = 0; j < Sq * w; ++j) x.ptr<__fp16>()[j] = (__fp16)0.1f;
    return x;
  };
  auto mkout = [&](int w) { return Tensor::empty({1, Sq, w}, mllm::kFloat16, mllm::kQNN).alloc(); };

  std::vector<Tensor> ins, outs;
  if (mode == "q") {
    ins = {mk(H)};
    outs = {mkout(q_out)};
  } else if (mode == "kv") {
    ins = {mk(H)};
    outs = {mkout(kv_out)};
  } else if (mode == "o") {
    ins = {mk(q_out)};
    outs = {mkout(H)};
  } else if (mode == "qkv") {
    ins = {mk(H)};
    outs = {mkout(q_out), mkout(kv_out), mkout(kv_out)};
  } else {
    fmt::print(stderr, "Unknown --mode: {} (expected q|kv|o|qkv)\n", mode);
    return 1;
  }

  QnnAOTModule proj("model.0.s" + std::to_string(Sq));
  proj.to(mllm::kQNN);
  auto dispatch = [&] { proj.setOutputTensors(outs); (void)proj(ins); };
  dispatch();   // warm
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[attn LPBQ] mode={} Sq={} avg = {:.4f} ms  (reps={})\n",
             mode, Sq,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps,
             reps);
});
