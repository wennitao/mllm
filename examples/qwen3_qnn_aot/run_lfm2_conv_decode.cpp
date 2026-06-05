// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device validator/latency driver for the LFM2 short-conv decode graph
// (compile_lfm2_conv_decode). Loads the ref (x, cw, cs, exp_y from export_short_conv.py),
// dispatches the graph, and reports max/rel error of the device output vs exp_y.
//
//   ./mllm-lfm2-aot-conv-decode-run -m lfm2-conv.bin --hidden 2048 --lcache 3 --ref lfm2_conv-ref.mllm
//
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("lfm2-conv-decode.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& k_arg = Argparse::add<int>("--lcache").def(3);
  auto& reps_arg = Argparse::add<int>("--reps").def(50);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("ref .mllm (x, cw, cs, exp_y)").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int H = hidden_arg.get(), K = k_arg.get(), reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());
  auto x = Tensor::empty({1, H}, mllm::kFloat16, mllm::kQNN).alloc();
  auto cw = Tensor::empty({1, K, H}, mllm::kFloat16, mllm::kQNN).alloc();
  auto cs = Tensor::empty({1, K - 1, H}, mllm::kFloat16, mllm::kQNN).alloc();
  for (int64_t i = 0; i < x.numel(); ++i) x.ptr<__fp16>()[i] = (__fp16)0.05f;
  for (int64_t i = 0; i < cw.numel(); ++i) cw.ptr<__fp16>()[i] = (__fp16)0.1f;
  for (int64_t i = 0; i < cs.numel(); ++i) cs.ptr<__fp16>()[i] = (__fp16)0.0f;
  auto y = Tensor::empty({1, H}, mllm::kFloat16, mllm::kQNN).alloc();
  auto new_cs = Tensor::empty({1, K - 1, H}, mllm::kFloat16, mllm::kQNN).alloc();
  std::vector<Tensor> ins{x, cw, cs}, outs{y, new_cs};

  const bool validate = !ref_arg.get().empty();
  Tensor exp_y;
  if (validate) {
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto setf16 = [&](Tensor& dst, const Tensor& src) {
      for (int64_t i = 0; i < dst.numel(); ++i) dst.ptr<__fp16>()[i] = (__fp16)src.ptr<float>()[i];
    };
    setf16(x, ref->pull("x"));
    setf16(cw, ref->pull("cw"));
    setf16(cs, ref->pull("cs"));
    exp_y = ref->pull("exp_y");
  }

  QnnAOTModule g("model.0.s" + std::to_string(H));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();

  if (validate) {
    double e = 0, range = 0;
    for (int i = 0; i < H; ++i) {
      float yi = (float)y.ptr<__fp16>()[i], ei = exp_y.ptr<float>()[i];
      e = std::max(e, (double)std::fabs(yi - ei));
      range = std::max(range, (double)std::fabs(ei));
    }
    double rel = 100.0 * e / (range + 1e-9);
    fmt::print("[conv VALIDATE] max|err|={:.5f} |y|max={:.4f} rel={:.2f}pct -> {}\n", e, range, rel,
               rel < 5.0 ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[conv LPBQ] H={} K={} avg = {:.4f} ms ({} reps)\n", H, K,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
  mllm::shutdownContext();
});
