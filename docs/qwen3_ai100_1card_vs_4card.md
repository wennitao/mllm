# Qwen3 on Qualcomm AI 100 — 1 card vs 4 cards

Running **Qwen3** (1.7B / 8B) on `ur-ai100-ws-02` (4× Cloud AI 100 Ultra, SDK **1.19.8**), and comparing single-card vs 4-card tensor-parallel.

> **Why this needed work:** SDK 1.19.8's QEfficient (`qeff19-venv`) has **no Qwen3 support**. Only QEfficient **1.22** (`qeff-venv`) knows Qwen3 — but its exported artifacts don't load cleanly on the 1.19.8 compiler/runtime. Two workarounds bridge that gap (below). Result: **Qwen3 runs on this box today, no SDK upgrade.**

---

## TL;DR — run it

```bash
# Qwen3-8B, 1 card
sg qaic -c 'bash /home/chihao/models/run_qwen3_8b_npu.sh "what is the capital of china"'

# Qwen3-8B, 4 cards (tensor-parallel, ~3.3x faster)
sg qaic -c 'bash /home/chihao/models/run_qwen3_8b_4card.sh "what is the capital of china"'
```

Replace the quoted text with your own prompt (must be **non-empty and quoted** — an empty prompt triggers a QEfficient `UnboundLocalError`). Edit `--generation_len 40` inside a script for longer answers.

---

## Results — Qwen3-8B, prompt "what is the capital of china", `generation_len 40`

| Metric | 1 card `[0]` | 4 cards `[0,1,2,3]` | Speedup |
|---|---:|---:|---:|
| **Decode** | 15.36 tok/s | **50.59 tok/s** | **3.3×** |
| **TTFT (prefill)** | 0.08 s | **0.03 s** | 2.7× |
| **E2E** | 2.62 s | **0.80 s** | 3.3× |

Completion (both, correct): *"The capital of China is Beijing. It is a city that has a rich history …"*

**~3.3× decode on 4 cards ≈ near-linear.** Much better than the small Qwen2.5-1.5B (~1.7×), because 8B's per-card matmuls are big enough that compute dominates the per-layer p2p all-reduce sync.

For reference, **Qwen3-1.7B** single card ≈ **63 tok/s** (too small for 4-card to help).

---

## Context-length sweep — 1 card vs 4 cards

Qwen3-8B, **prompt-len = ctx-len − 32**, 32 tokens generated. Each (ctx-len, device-count) is a **separately compiled QPC** (from the repacked ONNX with a matching `specializations.json`). Prefill / decode / E2E are QEfficient's inline stats. *(updating live as the sweep runs)*

**1 card `[0]`**

| ctx-len | prompt-len | prefill (s) | decode tok/s | E2E (s) |
|---:|---:|---:|---:|---:|
| 128 | 96 | 0.08 | 15.39 | 2.10 |
| 256 | 224 | 0.18 | 15.39 | 2.20 |
| 512 | 480 | 0.48 | 15.15 | 2.53 |
| 1024 | 992 | 0.73 | 14.97 | 2.80 |
| 2048 | 2016 | 1.24 | 14.47 | 3.38 |

**4 cards `[0,1,2,3]` (tensor-parallel)**

| ctx-len | prompt-len | prefill (s) | decode tok/s | E2E (s) |
|---:|---:|---:|---:|---:|
| 128 | 96 | 0.05 | 49.74 | 0.67 |
| 256 | 224 | 0.08 | 50.31 | 0.70 |
| 512 | 480 | 0.29 | 51.42 | 0.90 |
| 1024 | 992 | 0.34 | 48.33 | 0.99 |
| 2048 | 2016 | ⏳ | ⏳ | ⏳ |

**4-card speedup vs 1 card**

| ctx-len | decode | prefill | E2E |
|---:|---:|---:|---:|
| 128 | 3.23× | 1.6× | 3.13× |
| 256 | 3.27× | 2.25× | 3.14× |
| 512 | 3.39× | 1.66× | 2.81× |
| 1024 | 3.23× | 2.15× | 2.83× |
| 2048 | ⏳ | ⏳ | ⏳ |

_Takeaways: to be filled once the sweep completes (expected: decode speedup strong at short/medium ctx, softening at long ctx as the per-layer p2p all-reduce grows; prefill speedup growing with ctx as the big matmuls split cleanly)._

---

## The key concept: tensor parallelism is a COMPILE-TIME decision

Passing `--device_group [0,1,2,3]` to a **single-card QPC does nothing** — the extra cards are ignored and you get identical 1-card numbers. The weight-slicing must be **baked into the QPC at compile time** with a Multi-Device Partition (MDP) config.

| | 1-card QPC | 4-card QPC |
|---|---|---|
| compile flag | `-aic-num-cores=16` | `+ -mdp-load-partition-config=mdp_ts_4.json` |
| partition | single | `Partition0` spanning devices 0–3, `"type":"p2p"` |
| binary | `programqpc.bin` (7.9 GB) | `programqpc.bin` (15 GB), 4 slice graphs |
| `--device_group` at runtime | `[0]` | `[0,1,2,3]` |

`--device_group` at run time only *places* an already-partitioned program; it never creates the partitioning.

---

## Two workarounds that make Qwen3 run on SDK 1.19.8

1. **ONNX weight repack (fixes the compile).** QEfficient 1.22 writes external weights with only `location` (no `offset`/`length`); the 1.19.8 compiler's mmap loader rejects that (`Failed to mmap … Invalid argument` at `qaicModuleFinalize`). Re-saving the ONNX as a single blob with explicit offset+length fixes it.
2. **Runtime shim (fixes the run).** The 1.22 runner sets `prog_properties.dataPathTimeoutMs`, absent in 1.19.8 `qaicrt`. Guarded with `hasattr(...)` in
   `qeff-venv/.../QEfficient/generation/cloud_infer.py` (backup: `cloud_infer.py.orig`).

> The `rc=134 / Aborted (core dumped)` printed at the end of a build is just the compiler's post-compile **benchmark** (random inputs) crashing — it happens *after* `programqpc.bin` is written, so the QPC is valid. `QPC_PRESENT: yes` is the thing to check.

---

## How the QPCs were built (reproduce)

All artifacts live under `/home/chihao/models/`. Uses `qeff-venv` (QEfficient 1.22) + on-box `qaic-compile` (1.19.8).

### Step 1 — export + repack (shared by both)
`build_qwen3_8b_npu.sh`:
1. `QEfficient.cloud.infer` exports Qwen3-8B → ONNX (its own compile step fails at the mmap bug — expected, artifacts are kept).
2. Repack the ONNX to a single blob with explicit offset+length (`onnx.save(..., all_tensors_to_one_file=True)`).
3. Compile **1-card** QPC → `qwen3_8b_repacked/qpc`.

### Step 2 — 4-card QPC
`build_qwen3_8b_4card.sh` (reuses the repacked ONNX — no re-download/export):
1. Generate the 4-device MDP config: `generate_mdp_partition_config(4, 16)` → `mdp_ts_4.json`.
2. Recompile with `-mdp-load-partition-config=mdp_ts_4.json` → `qwen3_8b_4card/qpc`.

### Step 3 — run
`QEfficient.cloud.execute --qpc_path <qpc> --device_group <ids> --prompt "…"` (via the run scripts).

---

## Files

| Purpose | Path |
|---|---|
| 1-card run script | `/home/chihao/models/run_qwen3_8b_npu.sh` |
| 4-card run script | `/home/chihao/models/run_qwen3_8b_4card.sh` |
| 1-card QPC | `/home/chihao/models/qwen3_8b_repacked/qpc/` (7.9 GB) |
| 4-card QPC | `/home/chihao/models/qwen3_8b_4card/qpc/` (15 GB) |
| 4-card MDP config | `/home/chihao/models/qwen3_8b_4card/mdp_ts_4.json` |
| Build: export+repack+1-card | `/home/chihao/models/build_qwen3_8b_npu.sh` |
| Build: 4-card | `/home/chihao/models/build_qwen3_8b_4card.sh` |
| Runtime shim | `…/qeff-venv/…/QEfficient/generation/cloud_infer.py` (+ `.orig`) |
| Qwen3-1.7B run (1 card) | `/home/chihao/models/run_qwen3_npu.sh` |

---

## Notes / gotchas
- **Non-empty prompt** — empty/blank prompt → `UnboundLocalError: local variable 'outputs'` (QEfficient has no empty-prompt guard).
- **`sg qaic` uses `sh`**, not bash — always call a `#!/bin/bash` script (`sg qaic -c "bash …"`), never inline `source`.
- **Greedy repetition** on trivial prompts ("Beijing. Beijing. …") is a decoding artifact (greedy, no repetition penalty), not a model fault; open-ended prompts don't loop.
- **When 4-card pays off:** big models (8B+) or long context (prefill scales best). For ≤1.7B the p2p sync dominates and 4-card barely helps.
- This is a **workaround stack**, not the vendor-blessed path. The clean long-term fix is upgrading `/opt/qti-aic` to SDK 1.20+ (to match the 1.22 QEfficient already installed).
