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

### Methodology change vs. older branches — do not compare `.perf` files across this boundary

Before this branch (`opencl-flash-attention`), the dispatcher emitted a single
Perfetto `TRACE_EVENT` per op and **did not** call `clFinish` afterwards. That
made each slice an instant marker fired at host-side enqueue return, while the
GPU continued running asynchronously. The slice width on those older traces is
the host enqueue cost (microseconds, roughly constant), not the kernel's GPU
time. The current branch wraps each op in `TRACE_BEGIN/END` with a `clFinish`
between, so slice width is real GPU wall-clock plus per-op sync overhead.

This means **two `.perf` files from different sides of this change are not
directly comparable per-op** — the same SiLU/Linear/etc. will appear an order
of magnitude or more "slower" on the new branch purely because the measurement
now includes GPU completion. The `perfSummary` tokens/s number is unaffected
(it is wall-clocked by `ARGeneration` independently of which dispatcher macro
is in use).

```
Same op, same GPU work (≈ 300 µs kernel):

Old branch (TRACE_EVENT, no clFinish)
  CPU ──┤E├──────────────────────────────
         │
         ▼ marker
  GPU      ░░░[══════ kernel ══════]░░░░░
  Slice  ▒ 15 µs   ← only the enqueue return

Current branch (TRACE_BEGIN/END + clFinish)
  CPU ──┤E├────── wait on clFinish ──────
         │                            │
         ▼ BEGIN                      ▼ END
  GPU      ░[══════ kernel ══════]░░░░░░
  Slice  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 320 µs
                                  ↑
                                  per-op clFinish destroys
                                  pipeline overlap → end-to-end
                                  runtime is inflated under
                                  this trace
```

What each trace can and cannot answer:

| Question                                | Old branch        | Current branch    |
|-----------------------------------------|-------------------|-------------------|
| Per-op GPU wall-clock                   | no                | yes               |
| Rank kernels by GPU cost                | no                | yes               |
| Realistic end-to-end runtime in trace   | yes               | no (inflated)     |
| Isolated host-side enqueue overhead     | yes               | mixed-in          |
| FA-vs-eager per-op A/B                  | only against another `TRACE_EVENT` run | only against another `BEGIN/END + clFinish` run |

To do a fair per-op A/B between two attention implementations, both runs must
use the same dispatcher path — cherry-pick the current dispatcher onto the
older branch and rebuild, do not compare a current-branch `.perf` against an
old-branch `.perf`.

The fully correct fix is to read OpenCL event timestamps via
`clGetEventProfilingInfo`, which gives true per-op GPU start/end times without
forcing pipeline serialization. The queue is already created with
`CL_QUEUE_PROFILING_ENABLE` ([OpenCLRuntime.cpp:39](mllm/backends/opencl/runtime/OpenCLRuntime.cpp#L39)),
so the events are available; mllm just does not consume them yet.

#### The exact code change

The dispatcher diff between `profile_cpu_gpu_npu` and `opencl-flash-attention`
is the only thing that produces the per-op timing differences. Two things
change at once.

**`profile_cpu_gpu_npu` (old)** — instant marker, GPU runs async:

```cpp
case TaskTypes::kExecuteOp: {
#ifdef MLLM_PERFETTO_ENABLE
  MLLM_PERF_TRACE_EVENT("mllm.kernel", op_name, /* annotations */);  // instant marker
#endif
  op->reshape(inputs, outputs);
  op->setup(inputs, outputs);
  op->forward(inputs, outputs);   // clEnqueueNDRangeKernel — returns async
  break;                          // loop moves on while GPU still running
}
```

**`opencl-flash-attention` (current)** — explicit slice with `clFinish` inside:

```cpp
case TaskTypes::kExecuteOp: {
#ifdef MLLM_PERFETTO_ENABLE
  MLLM_PERF_TRACE_BEGIN("mllm.kernel", op_name, /* annotations */);  // slice OPENS
#endif
  op->reshape(inputs, outputs);
  op->setup(inputs, outputs);
  op->forward(inputs, outputs);   // clEnqueueNDRangeKernel — still returns async...
#ifdef MLLM_PERFETTO_ENABLE
  backend->runtime()->commandQueue().finish();  // ...but BLOCK here until GPU drains
  MLLM_PERF_TRACE_END("mllm.kernel");            // slice CLOSES after GPU completion
#endif
  break;
}
```

The two changes:

1. **Macro shape.** `MLLM_PERF_TRACE_EVENT(...)` is a Perfetto `TRACE_EVENT` —
   a single RAII-scoped slice. Standing alone as a statement, it opens and
   closes back-to-back: effectively an instant event with no duration.
   `MLLM_PERF_TRACE_BEGIN/_END` are explicit boundaries: the slice's lifetime
   is exactly the code between them.

2. **`clFinish` added inside the slice.** This is the part that actually
   changes what is being timed.
   - `op->forward` calls `clEnqueueNDRangeKernel`, which only queues the
     kernel and returns immediately. The GPU starts working in parallel.
   - Without `clFinish`, the dispatcher returns from `process()` and the next
     `kExecuteOp` can reshape/enqueue while the previous kernel is still
     running on the GPU. Pipeline overlap is preserved, but no Perfetto slice
     captures GPU completion.
   - With `clFinish`, the host thread is blocked inside the slice until the
     command queue drains. The slice's closing timestamp is therefore
     "GPU finished kernel N", not "kernel N enqueued".

Switching the macro from `TRACE_EVENT` to `BEGIN/END` alone would still emit
a slice, but it would cover only host enqueue time (microseconds, roughly
constant). It is the **added `clFinish`** that makes the slice include real
GPU wall-clock.

Because this change is at the dispatcher level, it applies uniformly to every
op — SiLU, Linear, RMSNorm, RoPE, the kv-cache copies, everything. There is
no per-op code change. On the old branch, every op's slice was host enqueue
time (~µs). On the new branch, every op's slice is host enqueue + GPU
wall-clock + `clFinish` overhead (hundreds of µs to ms). That is the entire
mechanism behind the 30x SiLU, the inflated first three Linears, and every
other "regression" visible in a cross-branch trace comparison — same
kernels, same shapes, same inputs, different stopwatch.

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
