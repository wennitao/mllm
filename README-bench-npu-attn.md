# Plan: NPU "Normal" Attention Microbenchmark (Online Graph Build)

Goal: measure the wall-clock time of one `qk_matmul → mask → softmax → av_matmul`
block — the code at [modeling_qwen3.hpp:187-203](mllm/models/qwen3/modeling_qwen3.hpp#L187-L203) —
running on the QNN HTP NPU. Mirror the layout of
[examples/fa_opencl_bench/](examples/fa_opencl_bench/) and
[examples/fa_cpu_bench/](examples/fa_cpu_bench/) so the three numbers compare
apples-to-apples.

Approach: **online graph build** — single binary on device, no `.bin`, no
AOT pipeline. Pattern lifted verbatim from
[examples/qwen_npu/main.cpp:164-180](examples/qwen_npu/main.cpp#L164-L180).

---

## 1. Why this path

| | AOT path | Online graph build (this plan) |
|---|---|---|
| Used by | `qwen3_qnn_aot`, `qwen3_npu` | `qwen_npu` |
| Binaries | x86 host (`compile`) + arm64 device (`run`) | one arm64 device binary |
| Per-shape iteration | Recompile `.bin`, push to device, ~minutes | Rebuild Module in-process, ~seconds |
| Graph optimizer | Full QNN offline optimizer (op fusion, layout sinks) | Just the three online passes |
| Setup overhead | High (QAIRT SDK on host, two build targets) | Low |
| What it measures (once warm) | HTP kernel time | HTP kernel time |

Both paths end at `QnnGraph_execute` on the HTP. For a microbench that
iterates on shapes, the online path saves all the AOT plumbing and gives the
same kernel number.

---

## 2. Files to add

```
examples/attn_npu_bench/
├── CMakeLists.txt
├── modeling_attn_only.hpp     # Attention-only nn::Module (Q@K → mask → softmax → @V)
├── main.cpp                   # initQnnBackend → trace → 3 passes → timed forward() loop
└── README.md                  # invocation notes
```

Wire it up by appending to [examples/CMakeLists.txt](examples/CMakeLists.txt):

```cmake
if(MLLM_BUILD_QNN_BACKEND)
  add_subdirectory(attn_npu_bench)
endif()
```

---

## 3. `modeling_attn_only.hpp` — the graph

A single `nn::Module` containing **only** the attention math. No projections,
no RoPE, no RMSNorm, no KV cache — those are model-level concerns we are
deliberately excluding so the bench number is just the attention cost.

```cpp
class AttnOnly final : public nn::Module {
  nn::CausalMask mask_;
  nn::Softmax    softmax_;
  nn::MatMul     qk_matmul_;   // transpose_b = true
  nn::MatMul     av_matmul_;   // transpose_b = false
  int head_dim_;

 public:
  AttnOnly() = default;
  AttnOnly(const std::string& name, int head_dim) : nn::Module(name), head_dim_(head_dim) {
    mask_       = reg<nn::CausalMask>("mask");
    softmax_    = reg<nn::Softmax>("softmax", -1);
    qk_matmul_  = reg<nn::MatMul>("qk_matmul", false, true);
    av_matmul_  = reg<nn::MatMul>("av_matmul", false, false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& in,
                              const std::vector<AnyValue>&) override {
    auto q = in[0], k = in[1], v = in[2];     // [B, H, S, D] each
    auto a = qk_matmul_(q, k) * (1.f / sqrtf(head_dim_));
    a     = mask_(a);
    a     = softmax_(a);
    return { av_matmul_(a, v) };
  }
};
```

**Inputs to the bench graph** (we feed Q/K/V as ready-made tensors; the
projections that produce them in the real model are not part of this bench):
- `q : [1, H, N, D]`
- `k : [1, H, N, D]`
- `v : [1, H, N, D]`

**Output:**
- `o : [1, H, N, D]`

dtype: start with `kFloat32`. If HTP only accepts the quantized op variants
for some of these patterns, fall back to whatever quant flavor `qwen_npu`'s
`modeling_qwen_npu.hpp` uses — copy the pattern, don't invent one.

---

## 4. `main.cpp` — the bench harness

Structure ports `fa_opencl_bench/main.cpp` 1:1, replacing OpenCL calls with
the three QNN passes.

```cpp
#include <chrono>
#include <cstdio>
#include <vector>
#include "mllm/mllm.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "modeling_attn_only.hpp"

void bench_one(int H, int D, int N, int warmup, int iters) {
  // 1. Fresh module per shape (graph is shape-fixed once built)
  auto attn = AttnOnly("attn", D);

  auto q = Tensor::random({1, H, N, D}, -1.f, 1.f, kFloat32, kCPU).to(kQNN);
  auto k = Tensor::random({1, H, N, D}, -1.f, 1.f, kFloat32, kCPU).to(kQNN);
  auto v = Tensor::random({1, H, N, D}, -1.f, 1.f, kFloat32, kCPU).to(kQNN);

  // 2. Trace the IR
  std::map<std::string, Tensor> inputs = {{"q", q}, {"k", k}, {"v", v}};
  auto irs = attn.trace(inputs, {});

  // 3. Build the QNN graph on-device (the three passes)
  mllm::ir::PassManager rewrite(irs["attn"]);
  rewrite.reg(mllm::qnn::createQNNGraphIOTensorPass());
  rewrite.reg(mllm::qnn::createQNNOpNamingPass());
  rewrite.run();

  mllm::ir::PassManager build(irs["attn"]);
  build.reg(mllm::qnn::createQNNGraphBuildPass());
  build.run();

  // 4. Warmup + timed iterations
  std::vector<Tensor> in_vec = {q, k, v};
  for (int i = 0; i < warmup; ++i) attn.forward(in_vec, {});

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    attn.forward(in_vec, {});
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  /* summarize + print: reuse Stats / summarize() / gflops_causal() /
     gbps_optimal() verbatim from fa_opencl_bench/main.cpp */
}

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  mllm::initQnnBackend();             // online init, no .bin

  constexpr int H = 16, D = 128;
  constexpr int kWarmup = 3, kIters = 20;
  const int N_set[] = {32, 64, 128, 256, 512, 1024};

  for (int N : N_set) {
    std::printf(">> H=%d D=%d N=%d ...\n", H, D, N);
    bench_one(H, D, N, kWarmup, kIters);
  }
});
```

Key invariants:

- `mllm::initQnnBackend()` with **no path argument** — that's the online
  variant. The AOT variant takes a `.bin` path.
- The three passes run **once per shape**, outside the timed loop.
- The timed loop contains **only** `forward()`. No `trace`, no pass
  registration, no I/O setup.
- QNN execute is synchronous; no `clFinish`-style flush needed.

---

## 5. `CMakeLists.txt`

```cmake
if(MLLM_BUILD_QNN_BACKEND)
  add_executable(mllm-attn-npu-bench main.cpp)
  target_link_libraries(mllm-attn-npu-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-attn-npu-bench PRIVATE ${MLLM_INCLUDE_DIR})
endif()
```

(Compare [examples/qwen_npu/CMakeLists.txt](examples/qwen_npu/CMakeLists.txt)
for the exact lib targets the online QNN path needs.)

---

## 6. Build + run

Build for android-arm64 (whatever cross-compile target this repo uses for the
phone — same one `qwen_npu` builds with):

```sh
cmake --build build-android-arm64-v8a --target mllm-attn-npu-bench
```

Push and run on device, per the `adb -P $PORT -s $DEVICE …` rule:

```sh
adb -P $PORT -s $DEVICE push \
    build-android-arm64-v8a/bin/mllm-attn-npu-bench /data/local/tmp/
adb -P $PORT -s $DEVICE shell \
    "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./mllm-attn-npu-bench"
```

Expected output format (matches `fa_opencl_bench`):

```
>> H=16 D=128 N=  32 ...
attn-npu  H=16 D=128 N=   32  min=  X.XXX ms  med=  X.XXX ms  mean=  X.XXX ms  XXX.X GF/s  XX.X GB/s
>> H=16 D=128 N=  64 ...
...
```

---

## 7. Sanity checks before trusting the numbers

1. **Variance < 5 % between two back-to-back runs.** If not, HTP is thermal-throttling — let the device cool down and re-run.
2. **Cross-check against the real Qwen3 NPU prefill.** Run `qwen3_npu` with `--prefill_len 32` and use its per-layer profiler. The bench's `N=32` number × `num_hidden_layers` should land within ~20 % of the real run's summed attention time. Larger gap → the bench Module is not running the same op variant the production graph runs, and you need to align dtypes / op fusion accordingly.
3. **Confirm the graph actually landed on HTP.** QNN logs which ops fall back to CPU on graph init. If you see CPU fallbacks for `MatMul` or `Softmax`, the bench is measuring CPU, not NPU, and the number is meaningless. Fix by swapping in whatever op variant `qwen_npu`'s Module uses for that op.

---

## 8. Out of scope on purpose

- **No Q/K/V projections.** They're linear-bound, not attention-bound. If you want a full-block number later, add a second bench (`mlp_npu_bench`) and sum.
- **No fp16/fp32 dual path.** The cast block at [modeling_qwen3.hpp:194-199](mllm/models/qwen3/modeling_qwen3.hpp#L194-L199) is a CPU/OpenCL accuracy workaround. NPU runs one fixed dtype regime; pick the one the QNN ops accept and stick with it.
- **No graph reuse across shapes.** QNN graphs are shape-fixed. Build a fresh Module per `N` (cheap in the online path) instead of trying to reshape.
- **No `.bin` serialization.** That's the AOT path. Out of scope.

---

## 9. Decisions still open

1. **Shape sweep:** is `N ∈ {32, 64, 128, 256, 512, 1024}` the right set, or do you want to match specific prompt lengths?
2. **Output:** stdout table only, or also a CSV side-by-side with the CPU/OpenCL bench results for plotting?
3. **Reporting metric:** the OpenCL bench prints GF/s using a causal flop count. Keep that, or also print a non-causal version so the NPU number is comparable to vendor datasheet TOPS?

Tell me how you want (1)–(3) resolved and I'll implement.
