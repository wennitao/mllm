// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device LFM2 MoE block runner (host router + per-expert-graph top-k dispatch).
// Loads ONE context with all N expert graphs (expert0..expertN-1, from compile_lfm2_moe_block),
// reads the router consts + a block golden, computes routing on the host, dispatches the top-k
// selected expert graphs on the HTP, weighted-sums their outputs, and validates vs the golden.
//
//   ./mllm-lfm2-aot-moe-block-run -m moe_block_l2.bin --router wm_moe_l2/router.mllm \
//       --ref wm_moe_l2/block-ref.mllm
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
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
  auto& model_path = Argparse::add<std::string>("-m|--model").def("lfm2-moe-block.bin");
  auto& router_arg = Argparse::add<std::string>("--router").help("router.mllm (gate_weight, expert_bias, meta, scaling)").required(true);
  auto& ref_arg = Argparse::add<std::string>("--ref").help("block-ref.mllm (x, sel_ids, sel_w, exp_y)").required(true);
  auto& reps_arg = Argparse::add<int>("--reps").def(20);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  // ---- host: router consts + golden ----
  auto router = mllm::load(router_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  auto ref = mllm::load(ref_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  auto gate_w = router->pull("gate_weight");     // [E, hidden] fp32
  auto bias = router->pull("expert_bias");       // [E] fp32
  auto meta = router->pull("meta");              // [L,E,top_k,hidden,inter,norm_topk] int32
  auto scaling_t = router->pull("scaling");      // [1] fp32
  const int32_t* mp = meta.ptr<int32_t>();
  const int E = mp[1], top_k = mp[2], hidden = mp[3];
  const bool norm_topk = mp[5] != 0;
  const float scaling = scaling_t.ptr<float>()[0];

  auto x_ref = ref->pull("x");                   // [1,hidden] fp32
  auto exp_y = ref->pull("exp_y");               // [1,hidden] fp32
  auto golden_ids = ref->pull("sel_ids");        // [top_k] int32

  // ---- host routing: sigmoid(x@gate^T)+bias -> top-k -> gather sigmoid -> /(sum+1e-6) -> *scaling ----
  const float* xp = x_ref.ptr<float>();
  const float* gw = gate_w.ptr<float>();
  const float* bp = bias.ptr<float>();
  std::vector<float> sig(E), score(E);
  for (int e = 0; e < E; ++e) {
    double acc = 0;
    const float* row = gw + (size_t)e * hidden;
    for (int i = 0; i < hidden; ++i) acc += (double)xp[i] * row[i];
    sig[e] = 1.f / (1.f + std::exp(-(float)acc));
    score[e] = sig[e] + bp[e];
  }
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                    [&](int a, int b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
  std::vector<int> sel(order.begin(), order.begin() + top_k);
  float wsum = 0;
  for (int j = 0; j < top_k; ++j) wsum += sig[sel[j]];
  float inv = norm_topk ? (1.f / (wsum + 1e-6f)) : 1.f;
  std::vector<float> w(top_k);
  for (int j = 0; j < top_k; ++j) w[j] = sig[sel[j]] * inv * scaling;

  fmt::print("[router] host sel={}", fmt::join(sel, ","));
  fmt::print("  golden=");
  for (int j = 0; j < top_k; ++j) fmt::print("{}{}", golden_ids.ptr<int32_t>()[j], j + 1 < top_k ? "," : "");
  fmt::print("  w={}\n", fmt::join(w, ","));

  // ---- device: dispatch the selected expert graphs, weighted-sum on host ----
  mllm::initQnnBackend(model_path.get());
  auto x = Tensor::empty({1, hidden}, mllm::kFloat16, mllm::kQNN).alloc();
  for (int i = 0; i < hidden; ++i) x.ptr<__fp16>()[i] = (__fp16)xp[i];
  std::vector<Tensor> ins{x};

  std::vector<double> y_block(hidden, 0.0);
  auto run_expert = [&](int e, float weight) {
    auto y = Tensor::empty({1, hidden}, mllm::kFloat16, mllm::kQNN).alloc();
    std::vector<Tensor> outs{y};
    QnnAOTModule g("expert" + std::to_string(e));
    g.to(mllm::kQNN);
    g.setOutputTensors(outs);
    (void)g(ins);
    for (int i = 0; i < hidden; ++i) y_block[i] += (double)weight * (float)y.ptr<__fp16>()[i];
  };
  for (int j = 0; j < top_k; ++j) run_expert(sel[j], w[j]);

  // ---- validate block output vs golden ----
  double e = 0, range = 0;
  const float* ey = exp_y.ptr<float>();
  for (int i = 0; i < hidden; ++i) {
    e = std::max(e, std::fabs(y_block[i] - ey[i]));
    range = std::max(range, (double)std::fabs(ey[i]));
  }
  double rel = 100.0 * e / (range + 1e-9);
  fmt::print("[moe-block VALIDATE] top_k={} max|err|={:.5f} |y|max={:.4f} rel={:.2f}pct -> {}\n", top_k, e, range, rel,
             rel < 5.0 ? "PASS" : "CHECK");

  // ---- latency: full block (router negligible; top_k expert dispatches) ----
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < reps_arg.get(); ++r) {
    for (int j = 0; j < top_k; ++j) {
      auto y = Tensor::empty({1, hidden}, mllm::kFloat16, mllm::kQNN).alloc();
      std::vector<Tensor> outs{y};
      QnnAOTModule g("expert" + std::to_string(sel[j]));
      g.to(mllm::kQNN);
      g.setOutputTensors(outs);
      (void)g(ins);
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  fmt::print("[moe-block] {} experts/token avg = {:.4f} ms ({} reps)\n", top_k,
             std::chrono::duration<double, std::milli>(t1 - t0).count() / reps_arg.get(), reps_arg.get());

  mllm::shutdownContext();
});
