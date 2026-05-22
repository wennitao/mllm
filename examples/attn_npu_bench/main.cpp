// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Microbenchmark for the "normal" (decomposed) attention block
//
//   attn = QK^T * scale
//   attn = mask(attn)
//   attn = softmax(attn)
//   out  = attn V
//
// running on the QNN HTP NPU through the online graph-build path.
//
// The block is the same shape as modeling_qwen3.hpp:187-203. We build it
// from qti.aisw built-in ops (MatMul + ElementWiseMultiply +
// ElementWiseAdd + Softmax + MatMul) via QNNBackend::graphAddNode directly,
// because the online QNN op-wrapper layer (mllm/backends/qnn/op/) has no
// MatMul / Softmax / CausalMask classes -- those live only in the AOT path.
//
// Output format mirrors examples/fa_opencl_bench and examples/fa_cpu_bench
// so the three numbers compare apples-to-apples (modulo dtype: this bench
// is fp16, the others are fp32).
//
// Build:
//   cmake --build build-android-arm64-v8a-qnn --target mllm-attn-npu-bench
//
// Push + run (per the user's adb -P/-s rule):
//   adb -P $PORT -s $DEVICE push \
//       build-android-arm64-v8a-qnn/bin/mllm-attn-npu-bench /data/local/tmp/
//   # Also push the matching libQnnHtp*Stub.so and hexagon-v*/.../libQnnHtp*Skel.so
//   adb -P $PORT -s $DEVICE shell \
//       "cd /data/local/tmp && export LD_LIBRARY_PATH=. && \
//        export ADSP_LIBRARY_PATH=.:/data/local/tmp && ./mllm-attn-npu-bench"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <unistd.h>

#include "QnnBackend.h"

#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"

using mllm::Context;
using mllm::kFloat16;
using mllm::kQNN;
using mllm::mllm_fp16_t;
using mllm::Tensor;
using mllm::qnn::QNNBackend;
using mllm::qnn::QNNParamScalarWrapper;

namespace {

struct Shape {
  int H;           // shared between Q and KV heads (no GQA in this bench)
  int S_q;
  int S_kv;
  int D;
  bool causal;
  const char* label;
};

struct Stats {
  double min_ms;
  double median_ms;
  double mean_ms;
};

Stats summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Stats s{};
  s.min_ms = samples.front();
  s.median_ms = samples[samples.size() / 2];
  s.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  return s;
}

// Same flop count the OpenCL/CPU benches use: lower-triangle for causal,
// full rectangle when non-causal. Two matmuls each contribute 2*D flops per
// (q, kv) entry, hence the factor of 4.
double gflops(const Shape& sh, double ms) {
  double pairs;
  if (sh.causal) {
    // For S_q <= S_kv (typical): only the last S_q rows of an S_kv-wide
    // triangle are realized. We approximate with the same formula the
    // OpenCL bench uses (S_q * (S_kv+1) / 2). For S_q > S_kv the bench
    // launches the work but the causal mask zeroes most rows; treat that
    // case as a full rectangle for flop-counting parity with OpenCL.
    if (sh.S_q <= sh.S_kv) {
      pairs = static_cast<double>(sh.S_q) * (static_cast<double>(sh.S_kv) + 1.0) * 0.5;
    } else {
      pairs = static_cast<double>(sh.S_q) * static_cast<double>(sh.S_kv);
    }
  } else {
    pairs = static_cast<double>(sh.S_q) * static_cast<double>(sh.S_kv);
  }
  const double flops = 4.0 * sh.H * sh.D * pairs;
  return (flops / (ms * 1e-3)) / 1e9;
}

// Lower-bound traffic (fp16 = 2 B / elem): Q read once, K and V read once,
// O written once. Real attention re-reads K/V from HBM/cache per row block,
// so true bandwidth is higher than this -- the number lets you compare
// against device peak BW.
double gbps_optimal(const Shape& sh, double ms) {
  const double bytes_per_elem = 2.0;  // fp16
  const double hd = static_cast<double>(sh.H) * sh.D;
  const double bytes = bytes_per_elem * (2.0 * hd * sh.S_q + 2.0 * hd * sh.S_kv);
  return (bytes / (ms * 1e-3)) / 1e9;
}

void announce(const Shape& sh) {
  std::printf(">> %-10s H=%2d D=%3d S_q=%5d S_kv=%5d causal=%d ...\n", sh.label, sh.H, sh.D, sh.S_q, sh.S_kv,
              (int)sh.causal);
  std::fflush(stdout);
}

// Fill a fp16 buffer with uniform random values in [-1, 1].
void fill_random_fp16(mllm_fp16_t* ptr, size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<mllm_fp16_t>(dist(rng));
}

// Build a static [1, S_q, S_kv] fp16 causal mask: 0 inside the allowed
// region, -1e4 outside. -1e4 (vs. -inf) keeps softmax NaN-free if a row
// happens to be fully masked.
Tensor build_causal_mask(int S_q, int S_kv) {
  auto m = Tensor::empty({1, S_q, S_kv}, kFloat16, kQNN).alloc();
  auto* mp = m.ptr<mllm_fp16_t>();
  for (int s = 0; s < S_q; ++s) {
    // Standard "new tokens align to end of K" convention. Clamped to
    // [0, S_kv] so S_q > S_kv (the OpenCL bench's odd prefill regime) does
    // not produce a negative or out-of-range limit.
    int kv_lim;
    if (S_kv >= S_q) {
      kv_lim = (S_kv - S_q) + s + 1;
    } else {
      kv_lim = S_kv;  // S_q > S_kv: nothing meaningful to mask off
    }
    if (kv_lim < 0) kv_lim = 0;
    if (kv_lim > S_kv) kv_lim = S_kv;
    for (int j = 0; j < S_kv; ++j) {
      mp[s * S_kv + j] = (j < kv_lim) ? static_cast<mllm_fp16_t>(0.f) : static_cast<mllm_fp16_t>(-1.e4f);
    }
  }
  return m;
}

void bench_one(const std::shared_ptr<QNNBackend>& backend, const Shape& sh, int warmup, int iters, int shape_idx) {
  // 1. Allocate I/O tensors on the QNN allocator.
  auto Q = Tensor::empty({sh.H, sh.S_q, sh.D}, kFloat16, kQNN).alloc();
  auto K = Tensor::empty({sh.H, sh.S_kv, sh.D}, kFloat16, kQNN).alloc();
  auto V = Tensor::empty({sh.H, sh.S_kv, sh.D}, kFloat16, kQNN).alloc();
  auto O = Tensor::empty({sh.H, sh.S_q, sh.D}, kFloat16, kQNN).alloc();

  std::mt19937 rng(0xA77E0001u ^ static_cast<unsigned>(shape_idx));
  fill_random_fp16(Q.ptr<mllm_fp16_t>(), static_cast<size_t>(sh.H) * sh.S_q * sh.D, rng);
  fill_random_fp16(K.ptr<mllm_fp16_t>(), static_cast<size_t>(sh.H) * sh.S_kv * sh.D, rng);
  fill_random_fp16(V.ptr<mllm_fp16_t>(), static_cast<size_t>(sh.H) * sh.S_kv * sh.D, rng);

  // 2. Static scalars / mask.
  const float scale_value = 1.f / std::sqrt(static_cast<float>(sh.D));
  auto scale_t = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  scale_t.ptr<mllm_fp16_t>()[0] = static_cast<mllm_fp16_t>(scale_value);

  Tensor mask_t;
  if (sh.causal) mask_t = build_causal_mask(sh.S_q, sh.S_kv);

  // 3. Build the graph. Unique name per shape so we never collide.
  char buf[96];
  std::snprintf(buf, sizeof(buf), "attn_%s_sq%d_skv%d_%d", sh.label, sh.S_q, sh.S_kv, shape_idx);
  const std::string graph = buf;
  if (!backend->createQnnGraph(graph)) {
    std::fprintf(stderr, "createQnnGraph failed for %s\n", graph.c_str());
    return;
  }

  // I/O tensors
  backend->addTensor(graph, "Q", QNN_TENSOR_TYPE_APP_WRITE, Q);
  backend->addTensor(graph, "K", QNN_TENSOR_TYPE_APP_WRITE, K);
  backend->addTensor(graph, "V", QNN_TENSOR_TYPE_APP_WRITE, V);
  backend->addTensor(graph, "O", QNN_TENSOR_TYPE_APP_READ, O);

  // Statics
  backend->addStaticTensor(graph, "scale", scale_t);
  if (sh.causal) backend->addStaticTensor(graph, "mask", mask_t);

  // Intermediates (HTP-managed)
  auto QK_t = Tensor::empty({sh.H, sh.S_q, sh.S_kv}, kFloat16, kQNN);
  auto QK_scaled_t = Tensor::empty({sh.H, sh.S_q, sh.S_kv}, kFloat16, kQNN);
  auto P_t = Tensor::empty({sh.H, sh.S_q, sh.S_kv}, kFloat16, kQNN);
  backend->addTensor(graph, "QK", QNN_TENSOR_TYPE_NATIVE, QK_t);
  backend->addTensor(graph, "QK_scaled", QNN_TENSOR_TYPE_NATIVE, QK_scaled_t);
  backend->addTensor(graph, "P", QNN_TENSOR_TYPE_NATIVE, P_t);
  Tensor QK_masked_t;
  if (sh.causal) {
    QK_masked_t = Tensor::empty({sh.H, sh.S_q, sh.S_kv}, kFloat16, kQNN);
    backend->addTensor(graph, "QK_masked", QNN_TENSOR_TYPE_NATIVE, QK_masked_t);
  }

  // QK = MatMul(Q, K, transpose_in1=true)
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> mm_qk_params = {
      QNNParamScalarWrapper::create<bool>("transpose_in1", true),
  };
  backend->graphAddNode(graph, "matmul_qk", "MatMul", {"Q", "K"}, {"QK"}, {}, mm_qk_params, "qti.aisw");

  // QK_scaled = QK * scale  (broadcast [1,1,1])
  backend->graphAddNode(graph, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QK_scaled"}, {}, {}, "qti.aisw");

  // Optional mask: QK_masked = QK_scaled + mask  (broadcast over H)
  std::string softmax_in = "QK_scaled";
  if (sh.causal) {
    backend->graphAddNode(graph, "add_mask", "ElementWiseAdd", {"QK_scaled", "mask"}, {"QK_masked"}, {}, {}, "qti.aisw");
    softmax_in = "QK_masked";
  }

  // P = Softmax(_, axis=2)
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm_params = {
      QNNParamScalarWrapper::create<uint32_t>("axis", 2u),
      QNNParamScalarWrapper::create<float>("beta", 1.f),
  };
  backend->graphAddNode(graph, "softmax", "Softmax", {softmax_in}, {"P"}, {}, sm_params, "qti.aisw");

  // O = MatMul(P, V)
  backend->graphAddNode(graph, "matmul_av", "MatMul", {"P", "V"}, {"O"}, {}, {}, "qti.aisw");

  if (!backend->graphFinalize(graph)) {
    std::fprintf(stderr, "graphFinalize failed for %s\n", graph.c_str());
    return;
  }

  // 4. Warmup + timed loop. graphExecute is synchronous on HTP.
  std::vector<Tensor> ins = {Q, K, V};
  std::vector<Tensor> outs = {O};
  for (int i = 0; i < warmup; ++i) backend->graphExecute(graph, ins, outs);

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    backend->graphExecute(graph, ins, outs);
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  Stats st = summarize(std::move(samples));
  std::printf(
      "%-10s  H=%2d D=%3d  S_q=%5d S_kv=%5d  min=%8.3f ms  med=%8.3f ms  mean=%8.3f ms  %7.1f GF/s  %6.1f GB/s\n",
      sh.label, sh.H, sh.D, sh.S_q, sh.S_kv, st.min_ms, st.median_ms, st.mean_ms, gflops(sh, st.min_ms),
      gbps_optimal(sh, st.min_ms));
  std::fflush(stdout);
}

}  // namespace

MLLM_MAIN({
  // adb shell block-buffers stdout; force line buffering so each printf
  // hits the host before any HTP watchdog can SIGKILL us mid-launch.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  // ADSP_LIBRARY_PATH is how the HTP runtime locates libQnnHtpV*Skel.so on
  // device. The QAIRT SDK expects it; set it if the caller hasn't.
  if (const char* p = std::getenv("ADSP_LIBRARY_PATH"); !p || *p == '\0') {
    setenv("ADSP_LIBRARY_PATH", ".:/data/local/tmp", /*overwrite=*/1);
  }

  // Disable QNN's DETAILED profiling. Each graph would otherwise allocate a
  // profile buffer on the NSP; across a long shape sweep these accumulate
  // and starve later graphExecute calls (err 6001 / "Failed to map
  // profiling buffer on NSP").
  setenv("MLLM_QNN_PROFILE_OFF", "1", /*overwrite=*/1);

  // Initialize QNN online (no context file -- pass a guaranteed-absent
  // path so initQnnBackend takes the "fresh context" branch even if some
  // stale qnn_context.bin from an AOT run is sitting in CWD).
  mllm::initQnnBackend("/tmp/__mllm_attn_npu_bench_no_context__.bin");

  auto backend = std::static_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!backend) {
    std::fprintf(stderr, "QNN backend not available\n");
    return 1;
  }

  // Match the OpenCL/CPU benches' defaults: B=1 implicit, H=16, D=128.
  constexpr int kH = 16;
  constexpr int kD = 128;
  constexpr int kWarmup = 3;
  constexpr int kIters = 20;

  std::printf("[npu-attn-bench] fp16 / qti.aisw decomposed (MatMul + EWMul + EWAdd + Softmax + MatMul)\n");
  std::printf("[npu-attn-bench] H=%d D=%d  warmup=%d iters=%d\n\n", kH, kD, kWarmup, kIters);

  // ---- Prefill: square attention, S_q = S_kv = N. Causal mask on. ----
  std::printf("[mode=prefill]  S_q = S_kv = N\n");
  {
    const int n_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096};
    int idx = 0;
    for (int n : n_sizes) {
      Shape sh{kH, n, n, kD, /*causal=*/true, "prefill"};
      announce(sh);
      bench_one(backend, sh, kWarmup, kIters, idx++);
    }
  }

  // ---- Decode: S_q fixed = 1, S_kv varies. No causal mask (single
  // query attending over the whole cache). ----
  std::printf("\n[mode=decode]   S_q fixed = 1, S_kv varies\n");
  {
    constexpr int kS_q = 1;
    const int s_kv_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int idx = 1000;  // distinct from prefill so graph names never collide
    for (int s_kv : s_kv_sizes) {
      Shape sh{kH, kS_q, s_kv, kD, /*causal=*/false, "decode"};
      announce(sh);
      bench_one(backend, sh, kWarmup, kIters, idx++);
    }
  }
});
