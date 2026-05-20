# How OpenCL profiling is measured

Three independent measurement layers run side-by-side. They give different
numbers because they measure different things — pick the one that matches the
question you're asking.

| Layer | Scope | Sync? | Overhead | Output |
|---|---|---|---|---|
| `perfSummary()` | whole prefill / whole decode | implicit (first token forces sync) | none | stdout (TTFT, tokens/s, total μs) |
| Perfetto `mllm.kernel` slices | one slice per op | **`clFinish` per op** | high — only in `MLLM_PERFETTO_ENABLE` builds | `.perf` trace (binary) |
| `ModuleProfiler` CSV | one row per Module/Layer call | **`clFinish` per Module** | high — only when `--module_profile_path` is set | `phase,call,module_name,us` |

Underlying clock for layers 2 & 3 is host `std::chrono::high_resolution_clock`,
not OpenCL event timestamps. The command queue *is* created with
`CL_QUEUE_PROFILING_ENABLE`
([OpenCLRuntime.cpp:39](mllm/backends/opencl/runtime/OpenCLRuntime.cpp#L39)) —
that exposes per-event GPU timestamps via `clGetEventProfilingInfo`, but mllm
does not currently consume them. The flag is on so future tooling can.

---

## 1. `perfSummary()` — model-level wall-clock

Defined at [ARGeneration.cpp:269](mllm/models/ARGeneration.cpp#L269). Always on;
printed by the runner before exit.

What it measures:

- `llm_prefill_{start,end}_time_` are stamped around the prefill forward.
- `llm_decode_{start,end}_time_` are stamped around the decode loop.
- TTFT = `decode_start − prefill_start`.
- `tokens/s` is integer-token-count ÷ duration.

There is no explicit `clFinish` — but prefill must return the first token
(which materializes on the host), so the end timestamp is effectively bounded
by the GPU finishing. **This is the number to quote for end-to-end throughput.**

## 2. Perfetto `mllm.kernel` slices — per-op GPU wall-clock

Emitted in [OpenCLDispatcher.cpp:50-74](mllm/backends/opencl/OpenCLDispatcher.cpp#L50-L74).
For each `kExecuteOp`, the dispatcher wraps the work like this:

```
MLLM_PERF_TRACE_BEGIN("mllm.kernel", op_name, /* input shape annotations */)
op->reshape(...); op->setup(...); op->forward(...);   // enqueues async
commandQueue().finish();                              // drain the queue
MLLM_PERF_TRACE_END("mllm.kernel")
```

The slice **closes after `clFinish`**, so the duration is real GPU completion
time for that op, not the return of `clEnqueueNDRangeKernel`. Each slice carries
the op type as its name and the input tensor shapes as debug annotations.

Caveat: the per-op `clFinish` serializes the pipeline that the GPU would
otherwise run async, so absolute kernel durations are accurate **but the
end-to-end run is slower than a non-profiled build**. Use this trace to compare
kernels against each other, not to measure throughput.

Open the resulting `.perf` file at <https://ui.perfetto.dev/>. Other track
categories ([docs/quick_start/how_to_perf.rst](docs/quick_start/how_to_perf.rst)):
`mllm.ar_step`, `mllm.tensor_lifecycle`, `mllm.func_lifecycle`.

## 3. `ModuleProfiler` CSV — per-Module wall-clock

Defined in [ModuleProfiler.cpp](mllm/engine/ModuleProfiler.cpp). Driven from
two call sites:

- [Module.cpp:126-146](mllm/nn/Module.cpp#L126-L146) — wraps every `Module::__main`.
- [Layer.cpp:65-77](mllm/nn/Layer.cpp#L65-L77) — wraps every `Layer::__main`.

Both follow the same pattern:

```cpp
t0 = clock::now();
dispatcherManager->submit(device, task);              // async enqueue
dispatcherManager->syncWait(device);                  // → clFinish
t1 = clock::now();
ModuleProfiler::record(absolute_name, t0, t1);
```

`syncWait` on OpenCL is `commandQueue().finish()`
([OpenCLDispatcher.cpp:86-94](mllm/backends/opencl/OpenCLDispatcher.cpp#L86-L94)),
so the row reflects real GPU completion of that Module's enqueued kernels.

CSV schema:

```
phase,call,module_name,us
prefill,0,model.layers.0.self_attn.q_proj,612.5
prefill,0,model.layers.0.self_attn,8421.2
```

- `phase` is set by `ARGeneration` — `"prefill"` before the prefill forward
  ([ARGeneration.cpp:499](mllm/models/ARGeneration.cpp#L499)), `"decode"` before
  the decode loop ([ARGeneration.cpp:506](mllm/models/ARGeneration.cpp#L506)).
- `call` increments **only at top-level modules** — controlled by a depth
  counter (`g_module_depth`) so nested modules share their parent's `call` index
  ([Module.cpp:127-132](mllm/nn/Module.cpp#L127-L132)).
- `module_name` is the absolute dotted name (e.g. `model.layers.5.mlp.gate_proj`).
- Rows are **nested**: a parent module's row covers its children's rows. Don't
  sum naively across all rows — pick one level of the tree.

### Bucketing the CSV

`tools/parse_opencl_profile.py` groups leaves into attention / MLP buckets per
layer. The attention `matmul (est.)` bucket is computed as

```
self_attn − (q_proj + k_proj + v_proj + o_proj + mask + softmax + q_norm + k_norm + q_rope + k_rope)
```

i.e. the residual after subtracting named leaves from the parent — covers the
two functional matmuls (Q·Kᵀ and attn·V) plus scale/view/transpose overhead.

---

## Build / run

Both Perfetto slices and the CSV require `-DMLLM_PERFETTO_ENABLE=ON` for the
build (already in [tasks/build_android_opencl.yaml](tasks/build_android_opencl.yaml)
— the Perfetto guard wraps `perf::start()/stop()/saveReport()` in the runner;
`ModuleProfiler` itself compiles unconditionally but the runner only toggles
it when `--module_profile_path` is set).

Runner flags ([examples/qwen3/main_opencl.cpp:21-25](examples/qwen3/main_opencl.cpp#L21-L25)):

- `--perf_path <file>` — Perfetto trace path (default `qwen3_opencl.perf`).
- `--module_profile_path <file>` — when non-empty, enables `ModuleProfiler` and
  dumps the CSV at shutdown. Empty (default) means no per-Module syncs.

## What to use when

- **"How fast is the model?"** → `perfSummary()` tokens/s. The other two layers
  serialize the pipeline and underestimate throughput.
- **"Which kernel is slow?"** → Perfetto `mllm.kernel` track, sorted by duration.
- **"Which layer / which projection?"** → `ModuleProfiler` CSV +
  `parse_opencl_profile.py`.
