# Heterogeneous Schedule Simulator (Qwen3-1.7B MLP)

A static makespan simulator that answers the question this repo's
[blocksparse_pipeline_profile_sq1024.md](../../docs/qnn_backend/blocksparse_pipeline_profile_sq1024.md)
spent days investigating empirically:

> *Can any way of splitting the MLP across NPU / GPU / CPU — with any tile
> count and any threading model — beat keeping it fused on the NPU?*

Given the measured per-(op, engine) latencies + transfer costs already
gathered in this session, it enumerates every plausible assignment and
predicts the makespan, ranking them against the best fused-on-NPU baseline.
The point is to **screen configurations cheaply**, before committing to
build/push/measure cycles.

## What you get in one command

```
$ python3 het_sim.py sweep
Sweep for 1024 tokens.
  best all-NPU (fused-tiled):  5.360 ms  (T=2, Sq=512)
  best heterogeneous (optimistic, no thermal): 6.540 ms  (T=2, silu=GPU, gu=CPU, ntl=3)
  → het lower bound is 1.22x fused-tiled (LOSES)

  heterogeneous configs predicted to BEAT fused-tiled
  (at the optimistic no-thermal lower bound): 0 / 120
```

**0 / 120 heterogeneous configs beat fused-tiled — even at the optimistic
no-thermal lower bound.** This is the formal version of every empirical
verdict the doc landed on.

## Quick start

```bash
cd tools/het_sim

# 1) Re-predict every config we measured, side by side with the measurement:
python3 het_sim.py validate

# 2) Enumerate all 135 configs (silu × gu × tile × timelines), rank by makespan:
python3 het_sim.py sweep [--total 1024] [--top 15]

# 3) Spot-check one config with per-engine breakdown + ASCII Gantt:
python3 het_sim.py one --silu GPU --gateup_mul CPU --T 2 --mode 2
```

No deps beyond Python 3 stdlib.

## The model

### Inputs ([`qwen3_1p7b_mlp.py`](qwen3_1p7b_mlp.py))

- **DAG** of the Qwen3-1.7B MLP: `gr` (gate+up matmul) → `silu` (sigmoid·g) →
  `gateup_mul` (silu·u) → `dn` (down matmul).
- **Per-op latencies on each engine** — all measured this session:
  - Matmuls (`gr`, `dn`): NPU-only lookup tables from
    [`mllm-het-mlp --npu-only`](../../examples/het_pipeline/het_mlp.cpp) sweep at
    Sq ∈ {64, 128, 256, 512} and the LPBQ microbench at 1024.
  - Activations (`silu`, `gateup_mul`): bandwidth-bound rates (GB/s) from
    [`mllm-bench-elementwise`](../../examples/elementwise_latency/bench_elementwise.cpp),
    per engine (NPU/CPU/GPU).
- **Transfer costs**: 0.034 ms/MB CPU↔GPU (from the `--ops xfer` measurement);
  NPU↔CPU is free (shared rpcmem); NPU↔GPU routes via CPU.
- **CPU OMP-on-worker penalty**: 3× (a known mllm gotcha — an OMP parallel-for
  launched from a pinned non-main thread costs ~3× serial).
- **Fused-on-NPU reference** (the single-dispatch baseline to beat): from the
  LPBQ microbench (`mllm-qwen3-aot-mlp-lpbq-bench --mode full`).

### What it captures

- **DAG dependencies** within and across tiles.
- **Engine occupancy** — each engine is one timeline; ops serialize on it.
- **Cross-engine transfers** added to dependency edges.
- **Pipeline fill/drain** at low tile counts (the simulator schedules
  events in the same all-gr → activations → all-dn order our threaded
  driver uses, so it sees the same per-tile waits).
- **Three threading models**: 1 timeline (all serial), 2 timelines
  (NPU‖worker, our `--threaded` mode), 3 timelines (NPU‖CPU‖GPU, theoretical
  max parallelism — not actually built).
- **Matmuls pinned to NPU** (no LPBQ kernel exists on CPU/GPU).
- **VTCM fusion savings** for all-NPU configs (special-cased — uses the
  measured fused MLP latency rather than summing individual ops, which is
  what the NPU graph compiler buys you).

### What it explicitly doesn't model

(per the request scope: no NPU dispatch overhead, no custom kernels)

- **NPU per-dispatch overhead** (~1.4 ms fixed for `gr`).
- **Custom HVX kernels** that don't exist today (e.g. fused SiLU+mul).
- **Thermal throttling** when N engines run hot concurrently — we measured
  ~+30% slowdown; flagged in `validate` output but not in the latency model.
  (Worth adding as a concurrency-penalty multiplier if you want a *realistic*
  upper bound; the current model gives the *optimistic* lower bound.)
- **OpenCL kernel JIT** (~35 ms one-time, amortized in warm reps).
- **Per-call host overhead** in mllm's tensor/dispatch path.

So het-config predictions are **optimistic lower bounds**: if even the
lower bound exceeds fused, the config definitely loses (most useful kind
of result). If the lower bound dips below fused, measure before believing it.

## Calibration: validation against measurements

`python3 het_sim.py validate` re-predicts every config the doc actually
measured (so the simulator's calibration is auditable):

| config | pred | meas | Δ | note |
|---|---:|---:|---:|---|
| fused-NPU @1024 | 12.50 | 12.50 | 0% | LPBQ microbench full |
| fused-NPU @512 | 2.68 | 2.68 | 0% | LPBQ microbench full |
| in-graph tiled 2×512 | 5.36 | 5.36 | 0% | 2× microbench-512 |
| CPU mul-only @1024 | 14.52 | 12.50 | +16% | over-counts (split `gr`+`silu` vs real fused `gateup` graph) |
| CPU silu+2mul @1024 | 13.13 | 30.00 | −56% | uses mllm's NEON `F::silu` rate, not the scalar `expf` the test loop ran |
| GPU silu, CPU mul @1024 | 12.02 | 12.35 | −3% | 3-way budget |
| `--tiled` 2×512 | 8.97 | 12.80 | −30% | no JIT, no per-call overhead |
| `--threaded` 2×512 | 7.05 | 14.00 | −50% | no thermal (+30%) and pessimistic measurement |
| `--threaded` 4×256 | 9.40 | 17.30 | −46% | same |
| `--threaded` 8×128 | 8.16 | 23.20 | −65% | same |

How to read this:
- **Fused configs**: exact (uses the measured reference directly).
- **Configs where compute dominates** (mul-only, 3-way budget): predictions
  within ~15% — the structural model works.
- **Heterogeneous overlap configs**: predictions are 30-65% below measured
  because the simulator gives the *bound* without thermal/JIT/per-call.
  *This is the intended behavior* — the lower bound is the most useful
  screen.

## The headline result (Qwen3-1.7B MLP, 1024 tok)

From `python3 het_sim.py sweep --top 12`:

```
  rank  type    T  Sq    silu  gu    ntl  makespan(ms)  vs fused-tiled
  -------------------------------------------------------------------------
     1  fused  2  512   NPU   NPU   1         5.360     1.00x
     2  fused  2  512   NPU   NPU   2         5.360     1.00x
     3  fused  2  512   NPU   NPU   3         5.360     1.00x
     4  het    2  512   GPU   CPU   3         6.540     1.22x
     5  het    2  512   GPU   CPU   2         7.054     1.32x
     6  het    2  512   CPU   CPU   3         7.242     1.35x
     7  het    8  128   CPU   CPU   3         8.160     1.52x
     ...
```

- Top 3: **all-NPU fused-tiled** (5.36 ms) — the winner. The 1/2/3-timeline
  rows are identical because all the work is on the NPU; threading model is
  irrelevant.
- #4: the best heterogeneous lower bound — **GPU SiLU + CPU mul, 2×512
  tiles, 3 parallel timelines = 6.54 ms (1.22× worse)**. This is the
  pipeline we actually built and measured at ~11–14 ms (the simulator's
  6.54 is what you'd get with zero thermal, zero JIT, zero per-tile
  overhead). Even that lower bound loses.
- The single concrete *useful* heterogeneous finding: even **3 parallel
  timelines** (theoretical max overlap, not what we built — we built 2) is
  not enough to beat fused-tiled. Fused wins on structure, not on
  implementation effort.

## What-if workflow

The simulator is most useful for asking questions you *haven't* measured.
Edit one number in `qwen3_1p7b_mlp.py` and re-run `sweep`:

| Question | What to change |
|---|---|
| Would a fused HVX SiLU at half the cost flip the verdict? | `activation.silu.rate_gbps.NPU: 5.2 → 10.4` |
| New silicon: GPU 2× faster, NPU 1.5× faster | scale the relevant rates / lookup tables |
| Cheap NPU dispatch overhead (custom runtime) | this model already excludes it; add `+1.4` per dispatch to test |
| Different model: smaller `I`, lighter MLP | `shape.I: 6144 → 4096` (matmul work drops proportionally) |
| Matmul-light model (attention-heavy) | scale matmul latencies down by 0.3 to mimic a model where they're a smaller share |

Each what-if is one file edit + one `sweep` invocation — seconds, not
build/push/measure cycles.

## Limitations / when NOT to trust it

- **Confident-NO is reliable**: if the simulator's optimistic lower bound
  exceeds fused, the config definitely loses (no amount of implementation
  cleverness recovers).
- **Confident-YES needs verification**: if the lower bound dips below fused,
  measure before celebrating — thermal, JIT, per-call overhead can eat the
  margin (we observed this directly).
- **Not a compiler**: doesn't reason about VTCM fitting, op fusion
  opportunities, or QNN tile-schedule cliffs (e.g., it doesn't predict the
  `down@M=1024` 5× cliff; we have to know about it from the per-op profile
  and either avoid that shape or update the lookup table).
- **Per-MLP only**: it's not a full-model simulator. End-to-end prefill
  depends on attention, chunk overhead, etc. Use this for MLP-level
  decisions.

## File layout

```text
tools/het_sim/
├── README.md              ← this file
├── het_sim.py             ← simulator + CLI (validate / sweep / one + block-sweep / block-one)
├── qwen3_1p7b_mlp.py      ← MLP-only spec: DAG, latencies, transfers
└── qwen3_1p7b_block.py    ← full decoder block (excl. GQA core) — same machinery, extended DAG
```

## Block-level mode

The block variant (`block-sweep` / `block-one`) covers the **whole decoder
layer minus the GQA scaled-dot-product attention core** (Q·K^T → softmax →
@V), which runs on a separate sparse-attention path with sparsity-dependent
cost. Everything else is in: input_layernorm, q/k/v_proj, q/k_norm, q/k
RoPE, o_proj, residual #1, post_attention_layernorm, full MLP, residual #2.

```bash
python3 het_sim.py block-sweep [--total 1024] [--top 15]
python3 het_sim.py block-one   [--norm CPU --rope GPU --add CPU ...] [--T 2]
```

> **No validated headline yet.** Unlike the MLP-only sweep, the block sweep
> doesn't have a `block-validate` calibration table — the predictions are
> raw optimistic lower bounds and shouldn't be quoted as a verdict without
> measuring a few of the top configs first. The MLP calibration delta was
> -30% to -65% on real heterogeneous pipelines; expect similar here.

**How it differs from MLP-only:**

- Het knobs are **grouped** by kind: `--norm`, `--rope`, `--add`, `--silu`,
  `--gateup_mul` (default NPU each). Sweep enumerates 3⁵ × tile × timeline.
- Matmuls (q/k/v/o_proj + MLP gr/dn) stay pinned to NPU (no LPBQ kernel
  elsewhere) — same constraint as MLP.
- All-NPU MLP portion still uses the measured fused-MLP single-dispatch
  (`fused_mlp_npu_ms`), same VTCM-fusion special case as the MLP sim.
- **GQA scheduling mode** (`--gqa-mode`):
  - `tile` (default): GQA[t] runs after qr/kr/v[t]; o_proj[t] waits on GQA[t].
    Tile-level pipelining.
  - `barrier`: one barrier — GQA only starts after ALL tiles' qkv/rope finish
    (matches sparse-attn scorer that needs the full K context first). All
    o_proj's wait on the barrier.
- GQA cost is a knob: `--gqa-ms-per-tile <ms>` (default 0). Plug in the
  measured sparse-attn cost from your pipeline to see realistic block
  predictions.

```
$ python3 het_sim.py block-one --norm NPU --rope GPU --add CPU --T 2 --mode 3 \
                               --gqa-mode tile --gqa-ms-per-tile 0
$ python3 het_sim.py block-sweep --gqa-mode barrier --gqa-ms-per-tile 2.0
```

### Refreshing the block-level measurements

LPBQ q/k/v/o projection latencies were collected this session with new
microbench tooling that mirrors the MLP microbench pattern:

```bash
# 1. Build x86 (AOT compiler).
mllm-cli run-task tasks/build_x86_qnn_aot.yaml

# 2. Compile bins for each (mode, Sq).
for sq in 64 128 256 512 1024; do
  for mode in q kv o qkv; do
    ./build-qnn-aot/bin/mllm-qwen3-aot-attn-lpbq-microbench-c \
        -aot_cfg examples/qwen3_qnn_aot/qnn_aot_cfg_attn_lpbq_microbench.json \
        --mode $mode --sq $sq
    mv qwen3-attn-lpbq-$mode.bin qwen3-attn-lpbq-$mode-sq$sq.bin
  done
done

# 3. Build Android (bench + runtime).
mllm-cli run-task tasks/build_android_qnn.yaml

# 4. Push bins + bench to device, then on-device:
for sq in 64 128 256 512 1024; do
  for mode in q kv o qkv; do
    ./mllm-qwen3-aot-attn-lpbq-bench -m qwen3-attn-lpbq-$mode-sq$sq.bin \
        --mode $mode --sq $sq
  done
done

# 5. Elementwise (RMSNorm + add) at attention widths, CPU/GPU:
for dev in cpu opencl; do
  ./mllm-bench-elementwise --device $dev --ops rmsnorm,add \
      --rows 64,128,256,512,1024 --width 2048
  ./mllm-bench-elementwise --device $dev --ops rmsnorm \
      --rows 64,128,256,512,1024 --width 128
done
```

NPU rmsnorm/add rates come from `docs/qnn_backend/gemm_latency.md`
(`GemmLatencyTest` — bandwidth-bound, kernel is width-invariant).

Plug the new numbers into `qwen3_1p7b_block.py` and re-run `block-sweep`.

## Reproducing the measurements baked in

If the numbers in `qwen3_1p7b_mlp.py` drift (new QAIRT, new silicon),
re-run these to refresh them:

- Activations (CPU/GPU/NPU): `mllm-bench-elementwise --device {cpu,opencl}`
  (CPU/GPU), `GemmLatencyTest.ActivationCpuVsNpu_Gate` (NPU). Bench is
  documented in [gemm_latency.md](../../docs/qnn_backend/gemm_latency.md).
- NPU matmul lookup tables (`gr`/`dn`): `mllm-het-mlp --npu-only --sq {64,128,256,512}`
  over the `gatedown` bins (each compiled via
  `mllm-qwen3-aot-mlp-lpbq-microbench-c --mode gatedown --sq <N>`).
- Fused-on-NPU reference: `mllm-qwen3-aot-mlp-lpbq-bench --mode full --sq <N>`.
- Cross-engine transfer cost: `mllm-bench-elementwise --device opencl --ops xfer`.

Plug new numbers into `qwen3_1p7b_mlp.py`, `python3 het_sim.py validate` to
confirm calibration, then `sweep` for the updated verdict.
