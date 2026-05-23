// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Attn-only AOT micro-benchmark: loads the REAL quantized per-qb block-sparse
// attention graphs (attn_0 / attn_1, compiled by
// `mllm-qwen3-aot-sha-blocksparse-causal-split-c MLLM_ATTN_ONLY=1` → tiny
// weightless qwen3-attn-only.bin) and measures two ways of driving them across
// a simulated num_layers × num_qb dispatch sequence with the REAL transposed-
// uint8 gather:
//
//   sync : gather(qb) → dispatch(qb)                 serially (V1-style baseline)
//   pipe : worker gathers(qb+1) ‖ main dispatches(qb)  (V2-style overlap)
//
// Two graphs (attn_0/attn_1) bind the two double-buffer slots, so the pipeline
// ping-pongs WITHOUT a per-dispatch rebind — this isolates the gather↔NPU-attn
// overlap effect. Verifies the finding that in this regime (short quantized
// dispatch + slow memory-bound gather) the pipeline DOES NOT beat sync: the
// gather can't hide behind the short dispatch (DRAM-bandwidth contention).
//
// Usage:
//   ./mllm-qwen3-aot-attn-bench -m qwen3-attn-only.bin --sq 1024 [--layers 28]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <sched.h>
#include <thread>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::aot::QnnAOTModule;

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("attn-only .bin").def("qwen3-attn-only.bin");
  auto& sq_arg = Argparse::add<int>("--sq").help("sequence length").def(1024);
  auto& layers_arg = Argparse::add<int>("--layers").help("simulated layers").def(28);
  auto& ctx_arg = Argparse::add<int>("--ctx").help("per-layer KV cache length (=real max_cache_length for matching footprint)").def(4096);
  auto& chunks_arg = Argparse::add<int>("--chunks").help("dispatch a real chunk graph between layers (needs full split bin)").def(0);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }

  // ----- Fixed Qwen3-1.7B block-sparse attention shape (matches the split). ---
  const int Hq = 16, Hkv = 8, D = 128, BQ = 32, BK = 32, topK = 8;
  const int group = Hq / Hkv;
  const int histKBK = (topK - 1) * BK;  // 224 historical rows (sink+recent+middle)
  const int topKBK = topK * BK;         // 256 score columns (mask width)
  const int Sq = sq_arg.get();
  const int num_qb = Sq / BQ;
  const int num_layers = layers_arg.get();
  // Cache length: enlarge so the gather can be made genuinely DRAM-bound (reads
  // scattered over a region >> L2). The integrated runner has 28 per-layer
  // caches + 1.6 GB weights resident, so its gather reads are cold DRAM; this
  // synthetic cache must be large+cold to reproduce that memory pressure.
  const int ctx = ctx_arg.get();
  const int max_blocks = ctx / BK;

  // Load the context binary. Use the tiny weightless qwen3-attn-only.bin to
  // measure the attention pipeline with NO memory pressure, OR the full 1.6 GB
  // split bin (qwen3-split-sq1024-fresh.bin) to keep the model weights resident
  // — same attn graphs (attn_0/attn_1 exist in both), but the full bin reproduces
  // the real runner's memory footprint. Combined with the per-layer KV caches
  // below, this isolates attention while matching real memory pressure.
  mllm::initQnnBackend(model_path.get());

  // ----- PER-LAYER quantized KV caches (uint8): matches the real runner's
  // footprint (num_layers separate caches, each [Hkv,D,ctx] / [Hkv,ctx,D]).
  // num_layers × ctx large → the gathers are cold DRAM, not L2-resident. -----
  std::mt19937 rng(0xA77E0001u);
  std::vector<Tensor> Kc(num_layers), Vc(num_layers);
  for (int L = 0; L < num_layers; ++L) {
    Kc[L] = Tensor::empty({Hkv, D, ctx}, mllm::kUInt8, mllm::kQNN).alloc();
    Vc[L] = Tensor::empty({Hkv, ctx, D}, mllm::kUInt8, mllm::kQNN).alloc();
    uint8_t* kp = Kc[L].ptr<uint8_t>();
    uint8_t* vp = Vc[L].ptr<uint8_t>();
    for (size_t i = 0; i < (size_t)Hkv * D * ctx; ++i) kp[i] = (uint8_t)(rng() & 0xFF);
    for (size_t i = 0; i < (size_t)Hkv * ctx * D; ++i) vp[i] = (uint8_t)(rng() & 0xFF);
  }
  const double cache_mb = (double)num_layers * Hkv * D * ctx * 2 / (1024 * 1024);  // K+V, all layers
  // Full-Sq current-token buffers the per-qb staging slices from.
  auto q_full = Tensor::empty({1, Hq, Sq, D}, mllm::kUInt16, mllm::kQNN).alloc();
  auto kcur_full = Tensor::empty({1, Hkv, D, Sq}, mllm::kUInt8, mllm::kQNN).alloc();
  auto vcur_full = Tensor::empty({1, Hkv, Sq, D}, mllm::kUInt8, mllm::kQNN).alloc();

  // ----- Per-qb staging buffers, double-buffered (slot 0/1). -----
  std::array<Tensor, 2> q_qb, kc_qb, vc_qb, K_arr, V_arr, mask, O;
  for (int b = 0; b < 2; ++b) {
    q_qb[b]  = Tensor::empty({1, Hq, BQ, D}, mllm::kUInt16, mllm::kQNN).alloc();
    kc_qb[b] = Tensor::empty({1, Hkv, D, BQ}, mllm::kUInt8, mllm::kQNN).alloc();
    vc_qb[b] = Tensor::empty({1, Hkv, BQ, D}, mllm::kUInt8, mllm::kQNN).alloc();
    K_arr[b] = Tensor::empty({Hq, 1, D, histKBK}, mllm::kUInt8, mllm::kQNN).alloc();
    V_arr[b] = Tensor::empty({Hq, 1, histKBK, D}, mllm::kUInt8, mllm::kQNN).alloc();
    mask[b]  = Tensor::empty({1, 1, BQ, topKBK}, mllm::kUInt16, mllm::kQNN).alloc();
    O[b]     = Tensor::empty({1, Hq, BQ, D}, mllm::kFloat16, mllm::kQNN).alloc();
  }

  // ----- Per-(layer,qb) random block selection over [0, active_blocks). The
  // scatter range sets how much of the cache the gather touches: small =
  // L2-resident reads (no DRAM pressure), large = cold DRAM reads. -----
  std::vector<std::vector<int>> sel(num_layers);
  std::vector<int> shuf(max_blocks);
  auto gen_sel = [&](int active_blocks) {
    active_blocks = std::min(active_blocks, max_blocks);
    for (int L = 0; L < num_layers; ++L) {
      sel[L].resize((size_t)Hq * num_qb * (topK - 1));
      for (int h = 0; h < Hq; ++h)
        for (int q = 0; q < num_qb; ++q) {
          for (int j = 0; j < active_blocks; ++j) shuf[j] = j;
          std::shuffle(shuf.begin(), shuf.begin() + active_blocks, rng);
          for (int s = 0; s < topK - 1; ++s)
            sel[L][((size_t)h * num_qb + q) * (topK - 1) + s] = shuf[s % active_blocks];
        }
    }
  };
  gen_sel(Sq / BK);  // default: scatter over this prefill's own tokens (matches integrated Sq=1024)

  // ----- The REAL prep: transposed-uint8 gather (slow) + per-qb staging. -----
  auto prep = [&](int L, int qb, int slot) {
    // gather K/V_arranged from the cache (K is D-major → D strided BK-byte copies).
    const int* selL = sel[L].data();
    uint8_t* kdst = K_arr[slot].ptr<uint8_t>();
    uint8_t* vdst = V_arr[slot].ptr<uint8_t>();
    const uint8_t* kc = Kc[L].ptr<uint8_t>();  // per-layer cache → cold DRAM reads
    const uint8_t* vc = Vc[L].ptr<uint8_t>();
    for (int h = 0; h < Hq; ++h) {
      const int kv = h / group;
      const int* hs = selL + ((size_t)h * num_qb + qb) * (topK - 1);
      const uint8_t* kc_h = kc + (size_t)kv * D * ctx;
      const uint8_t* vc_h = vc + (size_t)kv * ctx * D;
      uint8_t* kdst_h = kdst + (size_t)h * D * histKBK;
      uint8_t* vdst_h = vdst + (size_t)h * histKBK * D;
      for (int s = 0; s < topK - 1; ++s) {
        const int koff = hs[s] * BK;
        std::memcpy(vdst_h + (size_t)s * BK * D, vc_h + (size_t)koff * D, (size_t)BK * D);
        for (int d = 0; d < D; ++d)
          std::memcpy(kdst_h + (size_t)d * histKBK + (size_t)s * BK, kc_h + (size_t)d * ctx + koff, (size_t)BK);
      }
    }
    // stage q_qb / kc_qb / vc_qb (current-token slices for this qb).
    const uint16_t* qsrc = q_full.ptr<uint16_t>();
    uint16_t* qd = q_qb[slot].ptr<uint16_t>();
    for (int h = 0; h < Hq; ++h)
      std::memcpy(qd + (size_t)h * BQ * D, qsrc + ((size_t)h * Sq + (size_t)qb * BQ) * D, (size_t)BQ * D * sizeof(uint16_t));
    const uint8_t* ksrc = kcur_full.ptr<uint8_t>();
    uint8_t* kd = kc_qb[slot].ptr<uint8_t>();
    for (int h = 0; h < Hkv; ++h)
      for (int c = 0; c < D; ++c)
        std::memcpy(kd + ((size_t)h * D + c) * BQ, ksrc + ((size_t)h * D + c) * Sq + (size_t)qb * BQ, (size_t)BQ);
    const uint8_t* vsrc = vcur_full.ptr<uint8_t>();
    uint8_t* vd = vc_qb[slot].ptr<uint8_t>();
    for (int h = 0; h < Hkv; ++h)
      std::memcpy(vd + (size_t)h * BQ * D, vsrc + ((size_t)h * Sq + (size_t)qb * BQ) * D, (size_t)BQ * D);
  };

  // ----- Two AOT attn graphs (attn_0/attn_1) bound to slot 0/1. -----
  std::array<std::unique_ptr<QnnAOTModule>, 2> attn;
  attn[0] = std::make_unique<QnnAOTModule>("attn_0");
  attn[1] = std::make_unique<QnnAOTModule>("attn_1");
  attn[0]->to(mllm::kQNN);
  attn[1]->to(mllm::kQNN);
  auto dispatch = [&](int slot) {
    std::vector<Tensor> ins = {q_qb[slot], kc_qb[slot], vc_qb[slot], K_arr[slot], V_arr[slot], mask[slot]};
    attn[slot]->setOutputTensors({O[slot]});
    (void)(*attn[slot])(ins);
  };

  // ----- Optional: dispatch a REAL mid-chunk graph (chunk_1) between layers, to
  // reproduce the integrated runner's interleaved heavy projection+MLP NPU work
  // (and its thermal/contention effect). Needs the full split bin. Inputs:
  // [residual [1,Sq,H] fp16, attn_output [1,Hq,Sq,D] fp16, position_ids [1,Sq] i32]. ---
  const int hidden = 2048;
  const bool have_chunk = chunks_arg.get() != 0;  // requires the full split bin (has chunk_1)
  Tensor ch_resid, ch_attnout, ch_pos, ch_resid_o, ch_q_o, ch_kc_o, ch_vc_o;
  std::unique_ptr<QnnAOTModule> chunk;
  if (have_chunk) {
    ch_resid   = Tensor::empty({1, Sq, hidden}, mllm::kFloat16, mllm::kQNN).alloc();
    ch_attnout = Tensor::empty({1, Hq, Sq, D}, mllm::kFloat16, mllm::kQNN).alloc();
    ch_pos     = Tensor::empty({1, Sq}, mllm::kInt32, mllm::kQNN).alloc();
    for (int i = 0; i < Sq; ++i) ch_pos.ptr<int32_t>()[i] = i;
    ch_resid_o = Tensor::empty({1, Sq, hidden}, mllm::kFloat16, mllm::kQNN).alloc();
    ch_q_o     = Tensor::empty({1, Hq, Sq, D}, mllm::kUInt16, mllm::kQNN).alloc();
    ch_kc_o    = Tensor::empty({1, Hkv, D, Sq}, mllm::kUInt8, mllm::kQNN).alloc();
    ch_vc_o    = Tensor::empty({1, Hkv, Sq, D}, mllm::kUInt8, mllm::kQNN).alloc();
    chunk = std::make_unique<QnnAOTModule>("chunk_1");
    chunk->to(mllm::kQNN);
  }
  auto dispatch_chunk = [&](int n) {  // dispatch the chunk graph n times (surrounding graph work per layer)
    for (int i = 0; i < n && have_chunk; ++i) {
      std::vector<Tensor> ins = {ch_resid, ch_attnout, ch_pos};
      chunk->setOutputTensors({ch_resid_o, ch_q_o, ch_kc_o, ch_vc_o});
      (void)(*chunk)(ins);
    }
  };
  // Warm both graphs/slots (+ chunk if present).
  prep(0, 0, 0); prep(0, 0, 1);
  dispatch(0); dispatch(1);
  dispatch_chunk(have_chunk ? 1 : 0);

  // Clean isolated latency of the real LPBQ "chunk_1" graph (the genuine w4a16-LPBQ
  // non-attention compute per layer: O-proj + MLP gate/up/down + next-layer norm/QKV).
  // Faithful conventional-layout latency for the real quantized conv path — just wall time.
  if (have_chunk) {
    const int crit = 30;
    auto c0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < crit; ++i) dispatch_chunk(1);
    auto c1 = std::chrono::high_resolution_clock::now();
    const double ch_ms = std::chrono::duration<double, std::milli>(c1 - c0).count() / crit;
    fmt::print("[chunk-only] real LPBQ chunk_1 dispatch  Sq={}  avg = {:.4f} ms\n", Sq, ch_ms);
  }

  const int reps = 3;
  // ----- Persistent prep worker for the pipeline. -----
  std::mutex mtx; std::condition_variable cv;
  bool has_job = false, done = true, stop = false;
  int jl = 0, jq = 0, js = 0;
  std::thread worker([&] {
    {
      cpu_set_t set; CPU_ZERO(&set); bool any = false;
      if (const char* e = std::getenv("MLLM_PIPELINE_WORKER_CPUS")) {
        for (const char* p = e; *p;) { CPU_SET(std::atoi(p), &set); any = true; while (*p && *p != ',') ++p; while (*p == ',') ++p; }
      } else { CPU_SET(6, &set); CPU_SET(7, &set); any = true; }
      if (any) sched_setaffinity(0, sizeof(set), &set);
    }
    for (;;) {
      int l, q, s;
      { std::unique_lock<std::mutex> lk(mtx); cv.wait(lk, [&] { return has_job || stop; });
        if (stop) return; l = jl; q = jq; s = js; has_job = false; }
      prep(l, q, s);
      { std::lock_guard<std::mutex> lk(mtx); done = true; } cv.notify_all();
    }
  });
  auto submit = [&](int l, int q, int s) {
    std::unique_lock<std::mutex> lk(mtx); cv.wait(lk, [&] { return done; });
    jl = l; jq = q; js = s; has_job = true; done = false; cv.notify_all();
  };
  auto wait = [&] { std::unique_lock<std::mutex> lk(mtx); cv.wait(lk, [&] { return done; }); };

  // run_sync / run_pipe with `nch` chunk dispatches interleaved per layer.
  auto run_sync = [&](int nch) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      for (int L = 0; L < num_layers; ++L) {
        dispatch_chunk(nch);
        for (int qb = 0; qb < num_qb; ++qb) { prep(L, qb, 0); dispatch(0); }
      }
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
  };
  auto run_pipe = [&](int nch) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
      prep(0, 0, 0);
      auto t0 = std::chrono::steady_clock::now();
      for (int L = 0; L < num_layers; ++L) {
        dispatch_chunk(nch);
        if (L != 0) prep(L, 0, 0);
        for (int qb = 0; qb < num_qb; ++qb) {
          const int cur = qb & 1;
          if (qb + 1 < num_qb) submit(L, qb + 1, (qb + 1) & 1);
          dispatch(cur);
          if (qb + 1 < num_qb) wait();
        }
      }
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
  };

  // ===== CHUNK SWEEP: vary surrounding graph work (chunks/layer) 0 → N, watch
  // whether the pipe SAVING (sync - pipe) stays constant (pure dilution) or
  // shrinks (the surrounding graphs contend with / degrade the attn pipeline). =====
  struct Pt { int nch; double sync; double pipe; };
  std::vector<Pt> sweep;
  std::vector<int> nch_list = have_chunk ? std::vector<int>{0, 1, 2, 4} : std::vector<int>{0};
  for (int nch : nch_list) sweep.push_back({nch, run_sync(nch), run_pipe(nch)});

  { std::lock_guard<std::mutex> lk(mtx); stop = true; } cv.notify_all();
  worker.join();

  fmt::print("\n=== Attn pipeline + surrounding-graph (chunk) sweep (Sq={}, layers={}, num_qb={}, top_k={}) ===\n",
             Sq, num_layers, num_qb, topK);
  fmt::print("  bin={}  ctx={}  per-layer KV cache={:.0f} MB\n", model_path.get(), ctx, cache_mb);
  fmt::print("  Vary chunks/layer (heavy graphs dispatched serially before each attn layer).\n");
  fmt::print("  If the pipe SAVING (sync-pipe) stays ~constant → pure Amdahl dilution (no contention).\n");
  fmt::print("  If it SHRINKS as chunks grow → surrounding graphs degrade the attn pipeline.\n\n");
  fmt::print("    {:>11}  {:>10}  {:>10}  {:>10}  {:>9}\n", "chunks/layer", "sync ms", "pipe ms", "saving ms", "ratio");
  for (auto& p : sweep)
    fmt::print("    {:>11}  {:>10.2f}  {:>10.2f}  {:>10.2f}  {:>8.3f}x\n", p.nch, p.sync, p.pipe, p.sync - p.pipe,
               p.pipe / p.sync);
  if (sweep.size() > 1) {
    const double s0 = sweep.front().sync - sweep.front().pipe;
    const double sN = sweep.back().sync - sweep.back().pipe;
    fmt::print("\n  pipe saving: {:.2f} ms (0 chunks) -> {:.2f} ms ({} chunks/layer)  =>  {}\n", s0, sN,
               sweep.back().nch,
               std::abs(sN - s0) < 0.25 * std::max(s0, 1.0)
                   ? "CONSTANT — pure dilution; surrounding graphs do NOT degrade the attn pipeline ✅"
                   : "SHRINKS — surrounding graphs erode the attn pipeline saving");
  } else {
    fmt::print("\n  (run with --chunks 1 and the FULL split bin to sweep surrounding graphs)\n");
  }
  return 0;
})
