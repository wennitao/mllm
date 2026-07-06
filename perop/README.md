# AI 100 per-op profiling — Qwen3-8B (prefill vs decode)

Per-operator hardware-cycle profiling of Qwen3-8B on the Qualcomm Cloud AI 100, at two
granularities and for both inference regimes, from a single command.

- **Regimes:** `prefill` (`seq_len=128`) and `decode` (`seq_len=1`), both at `ctx_len=256`.
- **Op level:** per *operator kind* (`aicconvolutiond32`, `blockdequantize_mxfp6`, …) — the native `qaic-opstats` summary.
- **Arch level:** per *projection* (`q/k/v/o_proj`, `gate/up/down_proj`, `lm_head`, attention core) — derived from the trace.

## How to run

```bash
bash /home/chihao/mllm/tools/perop_profile_qwen3.sh
```

That's the whole thing — it runs **both prefill and decode** and writes op-level + arch-level
results under `/home/chihao/mllm/perop/`.

Optional knobs (defaults shown):

```bash
CTX=256 PREFILL_SEQ=128 SAMPLES=4 CORES=16 DEV=0 \
  bash /home/chihao/mllm/tools/perop_profile_qwen3.sh
```

- First run compiles the QPCs (~10 min); re-runs reuse them and go straight to profiling.
- `DEV=0` = which AI 100 card; `CTX` / `PREFILL_SEQ` change the shape (forces a recompile).
- Run just one regime by commenting out the other `run_regime` line at the bottom of the script.

## Requirements

- Qualcomm AIC SDK at `/opt/qti-aic` (provides `qaic-compile`, `qaic-runner`, `qaic-opstats`).
- QEfficient venv at `/home/chihao/qeff-venv` (the script sources it; only Python 3 is really needed for aggregation).
- An exported ONNX + custom-IO for the model (defaults point at the Qwen3-8B QEfficient cache).
- At least one free AI 100 card (`DEV=0` by default).

## Usage & knobs

All knobs are environment variables:

```bash
SAMPLES=8 PREFILL_SEQ=256 CTX=512 CORES=16 DEV=0 \
  bash /home/chihao/mllm/tools/perop_profile_qwen3.sh
```

| Var | Default | Meaning |
|---|---|---|
| `CTX` | `256` | context length (KV-cache depth); fixed at compile time |
| `PREFILL_SEQ` | `128` | tokens processed per prefill step |
| `SAMPLES` | `4` | profiling iterations captured & decoded per regime |
| `CORES` | `16` | NSP cores (tensor-parallel shards) |
| `DEV` | `0` | card id (`-d`) |
| `ONNX` / `CUSTOM_IO` | Qwen3-8B cache | model graph + IO descriptor |
| `OUTROOT` | `/home/chihao/mllm/perop` | where outputs are written |

Changing `CTX`/`PREFILL_SEQ`/`CORES` forces a fresh compile (shape is baked into the QPC).

## Pipeline (per regime)

```
qaic-compile -stats-level=70        → instrumented QPC (programqpc.bin)
qaic-runner  --aic-profiling-type raw_device_stats  → per-op cycle buffers (*-aiccyclecounts_*.bin)
qaic-opstats --summary --trace      → OP-LEVEL summary.txt + trace.json
perop_by_projection.py              → ARCH-LEVEL per_projection_*.txt/.csv
```

**The input is random, not a real prompt.** `qaic-runner` synthesizes bounded-random tensors
because there is no `bindings.json` in the QPC dir. This is correct for *timing*: per-op cycle
counts are determined by tensor **shapes and the op graph**, not by values. To profile a
different workload, change the **shape** (`PREFILL_SEQ`/`CTX`), not the data.

## What it generates

```
perop/
├── prefill/                          (seq_len=128, ctx=256)
│   ├── op_level_prefill.summary.txt      ← OP LEVEL (copy of one sample's summary)
│   ├── per_projection_prefill.txt/.csv   ← ARCH LEVEL (per projection)
│   ├── spec.json  custom_io.yaml
│   ├── out/    *.qaic-opstats.summary.txt + *.trace.json   (per sample; large)
│   ├── stats/  *-aiccyclecounts_*.bin     (raw cycle buffers; large)
│   └── qpc/    programqpc.bin             (compiled model; ~7 GB)
├── decode/                           (seq_len=1, ctx=256)  — same layout
└── prefill_vs_decode_by_projection.csv   ← combined arch-level comparison
```

### Tracked in git vs. regenerable (git-ignored)

| Committed (small) | Ignored — regenerate with the script (large) |
|---|---|
| `tools/perop_profile_qwen3.sh`, `tools/perop_by_projection.py` | `perop/**/out/` (summaries **and** 100–460 MB traces) |
| `per_projection_*.txt/.csv`, `op_level_*.summary.txt` | `perop/**/stats/` (`.bin` buffers) |
| `spec.json`, `custom_io.yaml` | `perop/**/qpc` (7 GB QPC) |
| `prefill_vs_decode_by_projection.csv` | `perop/verify_run.log` |

## Reading the results

### Op level — `op_level_<regime>.summary.txt`
Per-operator-kind table, one block per NSP core (`c0`…`c15`). Columns: `ucycles`, `pcycles`,
`count`, `out KB`, `opdetails`. Percentages **don't sum to 100** — HVX / HMX / DMAIssue engines
run concurrently, so each `%` is against the same total. This file is a byte-for-byte copy of
the first sample in `out/`; the other samples in `out/` differ only by clock jitter (~1 %).

### Arch level — `per_projection_<regime>.txt` / `.csv`
Cycles aggregated per projection (summed over all 16 cores), with each projection's op-kind
split. Answers "which linear costs the most, and is it matmul or weight-dequant?"

> **Note on `o_proj` / `down_proj`:** these have no named MatMul node — the ONNX exporter names
> their matmul after the consuming residual add (`/layers.N/Add` → `o_proj`,
> `/layers.N/Add_1` → `down_proj`). The aggregator remaps them; a tiny residual-add cost rides along.

## Notes

- **Weight-dequant dominates.** In both regimes, `blockdequantize_mxfp6` (unpacking MXFP6 weights)
  is 55–70 % of each projection's cost — more than the matmul itself. Decode is dequant/DMA-bound;
  even prefill stays weight-bound per projection, though attention core (softmax, O(seq²)) grows.
- **Cost is summed across 16 cores** in the arch-level files (weights are tensor-parallel-sharded),
  so absolute pcycles ≈ 16× a single core; the **share %** is the portable number.
- `qaic-opstats` can throw while enumerating a missing sample slot; the script tolerates this and
  proceeds with the samples it decoded (needs ≥1 trace).
- Re-running is cheap: the QPC is reused, so only profile+decode+aggregate rerun.
