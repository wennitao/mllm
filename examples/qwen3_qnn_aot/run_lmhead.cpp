// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device driver for the QUANTIZED (w4a16-LPBQ) head graph (final RMSNorm +
// lm_head) from compile_lmhead. Reports latency and (with --ref) validates the
// logits + argmax against the reference bundle from export_lmhead.py.
//
//   ./mllm-qwen3-aot-lmhead-run -m qwen3-lmhead.bin --ref lmhead-ref.mllm
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("qwen3-lmhead.bin");
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& vocab_arg = Argparse::add<int>("--vocab").def(248320);
  auto& reps_arg = Argparse::add<int>("--reps").def(50);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("reference .mllm -> validate accuracy").def("");
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int hidden = hidden_arg.get(), vocab = vocab_arg.get(), reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());

  auto mkf16 = [&](std::vector<int> shp, float v) {
    auto t = Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc();
    for (int64_t i = 0; i < t.numel(); ++i) t.ptr<__fp16>()[i] = (__fp16)v;
    return t;
  };

  std::vector<Tensor> ins;
  ins.push_back(mkf16({1, hidden}, 0.1f));      // x
  ins.push_back(mkf16({1, 1, hidden}, 1.0f));   // norm_w
  ins.push_back(mkf16({1, 1, 1}, 1e-6f));       // eps

  std::vector<Tensor> outs;
  outs.push_back(Tensor::empty({1, vocab}, mllm::kFloat16, mllm::kQNN).alloc());  // logits

  const bool validate = !ref_arg.get().empty();
  Tensor exp_logits;
  int exp_argmax = -1;
  if (validate) {
    auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
    auto f = [&](const std::string& n) { return ref->pull(n); };
    auto setf16 = [&](Tensor& dst, const float* src, int64_t n) {
      for (int64_t i = 0; i < n; ++i) dst.ptr<__fp16>()[i] = (__fp16)src[i];
    };
    setf16(ins[0], f("x").ptr<float>(), hidden);
    setf16(ins[1], f("norm_w").ptr<float>(), hidden);
    setf16(ins[2], f("eps").ptr<float>(), 1);
    exp_logits = f("exp_logits");
    double best = -1e30;
    for (int i = 0; i < vocab; ++i) {
      double v = exp_logits.ptr<float>()[i];
      if (v > best) { best = v; exp_argmax = i; }
    }
  }

  QnnAOTModule g("model.0.s" + std::to_string(hidden));
  g.to(mllm::kQNN);
  auto dispatch = [&] { g.setOutputTensors(outs); (void)g(ins); };
  dispatch();  // warm

  if (validate) {
    double e = 0, r = 0;
    int got_argmax = -1;
    double best = -1e30;
    for (int i = 0; i < vocab; ++i) {
      double gv = (double)outs[0].ptr<__fp16>()[i], ev = exp_logits.ptr<float>()[i];
      e = std::max(e, std::fabs(gv - ev));
      r = std::max(r, std::fabs(ev));
      if (gv > best) { best = gv; got_argmax = i; }
    }
    fmt::print("[lmhead VALIDATE] graph vs int4 ref:\n");
    fmt::print("  logits: max|err|={:.4f}  |logit|max={:.4f}  rel={:.2f}pct\n", e, r, 100.0 * e / (r + 1e-9));
    fmt::print("  argmax: graph={}  ref={}  -> {}\n", got_argmax, exp_argmax, got_argmax == exp_argmax ? "MATCH" : "MISMATCH");
    fmt::print("  -> {} (argmax match is what matters for greedy decode)\n", got_argmax == exp_argmax ? "PASS" : "CHECK");
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) dispatch();
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[lmhead LPBQ] hidden={} vocab={} avg = {:.4f} ms ({} reps)\n", hidden, vocab,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps, reps);
});
