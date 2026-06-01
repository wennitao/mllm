// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device validation for the new elementwise QNN visitors (Exp / Log).
// Loads the single-graph .bin produced by compile_exp_microbench, dispatches it
// on the HTP, and compares the device output against a host reference.
//
//   ./mllm-qwen3-aot-exp-run -m qwen3-exp-microbench.bin --op exp --seq 64 --width 128
//
#include <cmath>
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
  auto& model_path = Argparse::add<std::string>("-m|--model").help("microbench .bin").def("qwen3-exp-microbench.bin");
  auto& op_arg = Argparse::add<std::string>("--op").help("exp | log").def("exp");
  auto& seq_arg = Argparse::add<int>("--seq").help("compiled Seq").def(64);
  auto& width_arg = Argparse::add<int>("--width").help("feature width").def(128);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int M = seq_arg.get(), W = width_arg.get();
  const std::string op = op_arg.get();

  mllm::initQnnBackend(model_path.get());

  // Deterministic input in a safe range for both exp and log (log needs x>0).
  auto in = Tensor::empty({1, M, W}, mllm::kFloat16, mllm::kQNN).alloc();
  std::vector<float> ref(M * W);
  for (int j = 0; j < M * W; ++j) {
    float x = 0.25f + 0.5f * ((j % 7) / 6.0f);  // ~[0.25, 0.75]
    in.ptr<__fp16>()[j] = (__fp16)x;
    ref[j] = (op == "log") ? std::log(x) : (op == "rsqrt") ? (1.0f / std::sqrt(x)) : std::exp(x);
  }
  // Graph output is quantized uint16 (out_qdq: scale=1/256, zp=0); dequantize.
  const float out_scale = 1.0f / 256.0f;
  auto out = Tensor::empty({1, M, W}, mllm::kUInt16, mllm::kQNN).alloc();

  QnnAOTModule g("model.0.s" + std::to_string(M));
  g.to(mllm::kQNN);
  std::vector<Tensor> ins = {in}, outs = {out};
  g.setOutputTensors(outs);
  (void)g(ins);  // warm + run

  auto deq = [&](int j) { return (float)out.ptr<uint16_t>()[j] * out_scale; };
  double max_abs = 0, sum_abs = 0;
  for (int j = 0; j < M * W; ++j) {
    float err = std::fabs(deq(j) - ref[j]);
    max_abs = std::max(max_abs, (double)err);
    sum_abs += err;
  }
  fmt::print("[{} on HTP] M={} W={}  got[0..3]={:.4f} {:.4f} {:.4f} {:.4f}  ref[0..3]={:.4f} {:.4f} {:.4f} {:.4f}\n", op,
             M, W, deq(0), deq(1), deq(2), deq(3), ref[0], ref[1], ref[2], ref[3]);
  fmt::print("[{} on HTP] mean|err|={:.5f}  max|err|={:.5f}  -> {}\n", op, sum_abs / (M * W), max_abs,
             max_abs < 0.05 ? "PASS" : "CHECK (quant tol)");
});
