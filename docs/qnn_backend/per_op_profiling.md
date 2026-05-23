# Per-Operator Cycle Profiling on QNN HTP (in-process, no schematic)

How to answer "which operator inside a QNN graph eats the cycles?" — for any
graph that runs through `QNNBackend::graphExecute`, including **AOT-compiled
contexts loaded from a `.bin`**. This is the cheap, schematic-free path; it
gives per-QNN-op cycle counts but not the per-hardware-unit (HMX-vs-HVX) split
— see the boundary at the end.

## Enable it

Set the environment variable; nothing else changes in the build or the graph.

```bash
MLLM_QNN_PROFILE=DETAILED   # per-op NODE cycle events + coarse timing
MLLM_QNN_PROFILE=BASIC      # coarse graph-level timing only
# unset / anything else      → profiling off (the default)
```

The `QNNBackend` constructor reads this env var and sets the runtime profiling
level (`mllm/backends/qnn/QNNBackend.cpp`, the `profilingLevel_` initialisation).
`QNNRuntime::create` then builds a QNN profile handle at that level, every
`graphExecute` passes the handle, and after each dispatch
`extractBackendProfilingInfo` walks the event tree and **appends** rows to a
`qnn_profile.csv` in the process's working directory. Nothing is graph- or
model-specific: a runtime-built graph (e.g. the `GemmLatencyTest` micro-graphs)
and an AOT bin dispatched by the aot-runtime processors both go through the same
`graphExecute`, so both produce the same per-op csv. (Whatever runner you use,
profiling is purely opt-in via the env var — there is no separate flag.)

## The csv schema

```
depth,parent_event_id,event_id,type,unit,value,timestamp_us,identifier
```

- **Top-level coarse rows** (`type=UNKNOWN`) give the per-dispatch headline
  numbers — e.g. `"QNN accelerator (execute) time"` (DSP-side compute, µs),
  `"RPC (execute) time"` (host↔DSP transport + wait, µs — usually the bulk of
  wall time), `"Number of HVX threads used"`.
- **Per-op rows** are `type=NODE`, `unit=CYCLES`, with an identifier like
  `attn_0.Concat.0:OpId_46 (cycles)` — i.e. `<graph_name>.<Op>.<index>:OpId_N`.
  The `value` column is the op's cycle count.

Two things to keep in mind reading the cycle values:

1. **They are summed across the DSP's hardware units** (1 HMX + 6 HVX threads),
   so they are *active* cycles, not wall-clock. The reliable signal is the
   *relative share* between ops, not an absolute time. (To map cycles to
   wall-clock per unit you need the per-hardware-unit view — schematic-gated,
   below.)
2. **The csv accumulates across every dispatch in the run** (warmup, the graph
   of interest, and any others). Filter by the graph-name prefix and aggregate;
   drop the first dispatch as warmup.

## Aggregation recipe

Mean cycles per op-family for one graph, sorted by the heaviest — strip the
trailing `:OpId_N` so repeated dispatches of the "same" op fold together:

```bash
awk -F, '$4=="NODE" && $8 ~ /chunk_1\./ {
           gsub(/"/,"",$6); name=$8; sub(/:OpId.*/,"",name);
           sum[name]+=$6; cnt[name]++ }
     END { for (n in sum) printf "%12.0f  x%-4d  %s\n", sum[n]/cnt[n], cnt[n], n }' \
  qnn_profile.csv | sort -rn | head -20
```

Coarse op-class budget (Mul / CastType / Add / Gather / Reshape / other) for the
same graph follows the same shape — bucket the identifier and sum per class.

## Worked example: the real w4a16-LPBQ chunk graph

Profiling the production split-prefill `chunk_1` graph at Sq=1024 (the
non-attention compute of one layer: O-proj + MLP gate/up/down + the next layer's
input-norm and Q/K/V projections), the per-op-class budget came out (mean cycles
per dispatch, summed across HW units):

| Op class | share | what it is |
|---|---:|---|
| `Mul` (elementwise) | ~46% | the MLP wide-tensor products — SiLU's gate×sigmoid and gate×up over the 6144-wide intermediate, on HVX |
| `CastType` | ~23% | the fp16↔uint16 quantization-boundary requantizations |
| `Add` | ~14% | residual adds |
| `Gather` | ~6% | RoPE LUT gathers |
| *other* (incl. the conv GEMMs, Input/Output) | ~6% | — |
| `Reshape`/`View` | ~4% | layout conversions |

The headline this exposes: the conv GEMMs — the gate/up/down/projection
matrix-multiplies — are a *minor* slice of the budget (buried in the ~6%
"other"); they don't even surface by name among the top consumers. The chunk is
dominated by **HVX elementwise multiplies and quantization-boundary casts** over
the wide intermediate tensors — the same structural lesson as the attention
graph (HVX elementwise/glue dominates, the HMX matrix-multiply is tiny). The
coarse rows also showed DSP "accelerator (execute) time" ≈ 2.2 ms versus a ~10 ms
wall — most of the wall is RPC/host overhead, a separate lever.

This is the kind of conclusion the per-op view exists to deliver: it tells you
the lever that would actually move latency (here: fusing the SiLU/multiplies and
cutting cast traffic) rather than the one that looks attractive in a
single-GEMM micro-benchmark.

## Boundary: what this path does *not* give

Per-op *cycle counts* need no schematic. The per-*hardware-unit* breakdown (how
each op's cycles split across the HMX engine and the six HVX threads), and the
DRAM/VTCM bandwidth counters, require the offline `qnn-profile-viewer` + QHAS
toolchain fed a `<graph>_schematic.bin`. That schematic only emits on the
runtime graph-build path during `graphFinalize` with profiling on — **AOT-saved
context bins carry no schematic**, so the per-hardware-unit decomposition is not
reachable for AOT graphs. The full optrace/QHAS workflow (and example tables for
the dense/big-batch attention graphs) is documented in
[block_sparse_attention.md](block_sparse_attention.md) under "Per-hardware-unit
utilisation via qnn-profile-viewer" and "QHAS". Use that for runtime-built
graphs when you need the HMX/HVX split; use the in-process per-op csv described
here for everything else, including AOT bins.
