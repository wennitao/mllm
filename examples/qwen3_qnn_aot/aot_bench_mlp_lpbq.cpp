// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Driver for the fake-scale w4a16-LPBQ MLP microbench bin (compile_mlp_lpbq_microbench).
// Dispatches the single "model.0.s<Sq>" graph and reports avg latency — the faithful
// full-MLP (gate/up/silu/mul/down LPBQ + uint16 QDQ) latency to compare against the
// fp16 GemmLatency MlpBlock numbers and to drive MLP fusion/slicing experiments.
//
//   ./mllm-qwen3-aot-mlp-lpbq-bench -m qwen3-mlp-lpbq-microbench.bin --sq 1024
//
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
  auto& model_path = Argparse::add<std::string>("-m|--model").help("MLP LPBQ .bin").def("qwen3-mlp-lpbq-microbench.bin");
  auto& sq_arg = Argparse::add<int>("--sq").help("compiled Sq").def(1024);
  auto& tiled_arg = Argparse::add<bool>("--tiled").help("2-half tiled bin (2 in/out)").def(false);
  auto& mode_arg = Argparse::add<std::string>("--mode").help("full | gateup | down (match the compiled bin)").def("full");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int Sq = sq_arg.get(), H = 2048, I = 6144;  // hidden, intermediate
  const std::string mode = mode_arg.get();
  const int t = tiled_arg.get() ? 2 : 1, M = Sq / t;  // tiled: 2 inputs/outputs of M=Sq/2

  mllm::initQnnBackend(model_path.get());
  // I/O shapes per mode. gateup: 1 in [.,.,H] -> 2 out [.,.,I] (silu(gate), up);
  // the CPU does the gate*up mul off-graph. down: 1 in [.,.,I] -> 1 out [.,.,H].
  // full/tiled: in/out both [.,.,H].
  auto mk = [&](int w) { auto x = Tensor::empty({1, M, w}, mllm::kFloat16, mllm::kQNN).alloc();
                         for (int j = 0; j < M * w; ++j) x.ptr<__fp16>()[j] = (__fp16)0.1f; return x; };
  auto mkout = [&](int w) { return Tensor::empty({1, M, w}, mllm::kFloat16, mllm::kQNN).alloc(); };
  std::vector<Tensor> ins, outs;
  if (mode == "gateup") { ins = {mk(H)}; outs = {mkout(I), mkout(I)}; }
  else if (mode == "down") { ins = {mk(I)}; outs = {mkout(H)}; }
  else { for (int i = 0; i < t; ++i) { ins.push_back(mk(H)); outs.push_back(mkout(H)); } }  // full / tiled
  QnnAOTModule mlp("model.0.s" + std::to_string(M));
  mlp.to(mllm::kQNN);
  auto dispatch = [&] { mlp.setOutputTensors(outs); (void)mlp(ins); };
  dispatch();  // warm
  const int reps = 50;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[MLP LPBQ] Sq={} avg = {:.4f} ms\n", Sq,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps);
});
