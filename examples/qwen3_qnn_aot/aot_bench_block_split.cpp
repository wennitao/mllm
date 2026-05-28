// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Driver for the split-attention bin from
// `mllm-qwen3-aot-attn-lpbq-microbench-c --mode split`. Loads pre + post
// graphs and times them dispatched SERIALLY back-to-back. This is the
// all-NPU full-block latency MINUS the GQA core (which production runs in
// a separate CPU-NPU-pipelined path).
//
// Scope of the measurement:
//   pre  graph: input_layernorm + qkv (LPBQ)
//   post graph: o_proj + res1 + post_norm + MLP + res2
// NOT measured here (but documented as known additions): q/k_norm (~0.27 ms
// at Sq=1024) and q/k_rope (~0.85 ms at Sq=1024). Add ~1.1 ms to the result
// to get the production-equivalent all-NPU full-block time.
//
//   ./mllm-qwen3-aot-block-split-bench -m qwen3-attn-lpbq-split-real-sq1024.bin --sq 1024

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
  auto& model_path = Argparse::add<std::string>("-m|--model").help("split LPBQ .bin")
                         .def("qwen3-attn-lpbq-split.bin");
  auto& sq_arg = Argparse::add<int>("--sq").help("compiled Sq").def(1024);
  auto& reps_arg = Argparse::add<int>("--reps").help("timed reps").def(50);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  const int Sq = sq_arg.get();
  const int H = 2048, q_out = 16 * 128, kv_out = 8 * 128;
  const int reps = reps_arg.get();

  mllm::initQnnBackend(model_path.get());

  QnnAOTModule pre("pre");
  QnnAOTModule post("post");
  pre.to(mllm::kQNN);
  post.to(mllm::kQNN);

  auto mk = [&](int w) {
    auto x = Tensor::empty({1, Sq, w}, mllm::kFloat16, mllm::kQNN).alloc();
    for (int j = 0; j < Sq * w; ++j) x.ptr<__fp16>()[j] = (__fp16)0.05f;
    return x;
  };
  auto mkout = [&](int w) { return Tensor::empty({1, Sq, w}, mllm::kFloat16, mllm::kQNN).alloc(); };

  // pre: in[1,Sq,H] -> {residual[1,Sq,H], q[1,Sq,q_out], k[1,Sq,kv_out], v[1,Sq,kv_out]}
  auto pre_in = mk(H);
  auto pre_res = mkout(H), pre_q = mkout(q_out), pre_k = mkout(kv_out), pre_v = mkout(kv_out);
  std::vector<Tensor> pre_ins = {pre_in};
  std::vector<Tensor> pre_outs = {pre_res, pre_q, pre_k, pre_v};

  // post: (residual[1,Sq,H], attn_out[1,Sq,q_out]) -> y[1,Sq,H]
  auto post_residual = mk(H);
  auto post_attn_out = mk(q_out);
  auto post_y = mkout(H);
  std::vector<Tensor> post_ins = {post_residual, post_attn_out};
  std::vector<Tensor> post_outs = {post_y};

  auto run_pre  = [&] { pre.setOutputTensors(pre_outs);   (void)pre(pre_ins); };
  auto run_post = [&] { post.setOutputTensors(post_outs); (void)post(post_ins); };

  // Warm + isolated timings + serial combined.
  run_pre(); run_post();
  double t_pre = 0, t_post = 0, t_combined = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) run_pre();
  auto t1 = std::chrono::high_resolution_clock::now();
  t_pre = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) run_post();
  t1 = std::chrono::high_resolution_clock::now();
  t_post = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

  t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < reps; ++i) { run_pre(); run_post(); }
  t1 = std::chrono::high_resolution_clock::now();
  t_combined = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

  fmt::print("\n=== All-NPU block split (Sq={}, reps={}) ===\n", Sq, reps);
  fmt::print("  pre  (in_norm + qkv)                : {:.4f} ms\n", t_pre);
  fmt::print("  post (o_proj + res + post_norm + mlp): {:.4f} ms\n", t_post);
  fmt::print("  pre + post serial                    : {:.4f} ms\n", t_combined);
  fmt::print("  + q/k_norm  (estimated)              : ~0.27 ms\n");
  fmt::print("  + q/k_rope  (estimated)              : ~0.85 ms\n");
  fmt::print("  => Full all-NPU block-no-GQA         : {:.2f} ms\n", t_combined + 1.12);
});
