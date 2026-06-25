# Qwen on Qualcomm AI 100 — QEfficient/qaic runs

Machine `ur-ai100-ws-02`: 4× Qualcomm Cloud AI 100 (`/dev/accel/accel0-3`, 16 NSP + ~31 GB each), Platform/Apps SDK **1.19.8** at `/opt/qti-aic`.

---

## Setup (one-time)
- venv (SDK-matched): `/home/chihao/qeff19-venv` — QEfficient `release/v1.19` (transformers 4.45.2). Compiler = `/opt/qti-aic/exec/qaic-exec -aic-hw -aic-hw-version=2.0`.
- User in `qaic` group → run under `sg qaic -c "..."` (note: `sg` uses `sh`, so call a `#!/bin/bash` script, not inline `source`).
- Caches kept outside the repo: `QEFF_HOME=/home/chihao/models/qeff19_cache`, `HF_HOME=/home/chihao/models/hf`.

> Qwen3 is **not** usable here: it needs QEfficient `main` (SDK 1.20), and the 1.19.8 compiler rejects its ONNX format. So these runs use **Qwen2.5-1.5B-Instruct** on the matched stack.

---

## Model & how it's run
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, mxfp6 weights, fp16 compute.
- **Compile:** QEfficient exports HF → ONNX, then `qaic-exec` builds an AIC `.qpc` (cached per shape/device-count).
- **Single card:** `--device-group [0]`.
- **4 cards (tensor-parallel):** `--device-group [0,1,2,3]` → one model **partitioned (tensor-sliced)** across 4 SoCs (64 NSP cores), cards sync over p2p each layer. (Confirmed in `mdp_ts_config.json`: one Partition0 spanning devices 0-3, `"type":"p2p"`.) This is one inference on four cards — **not** four independent copies.

---

## How the 4-card partition works (tensor-slicing)

The `_ts_` (tensor-slice) MDP config is **Megatron-style intra-layer tensor parallelism**, *not* pipeline parallelism: every layer's weight matrices are cut into 4 pieces (one per card) and **all 4 cards work on the same token simultaneously**. Each card holds ~¼ of the weights (~0.38 B of 1.5 B params) and does ~¼ of the math per layer.

For Qwen2.5-1.5B (hidden 1536, FFN 8960, 12 query heads, 2 KV heads, 28 layers), per layer:

- **Attention — split by heads.** 12 query heads → **3 per card**; `q/k/v_proj` column-split, `o_proj` row-split. GQA subtlety: only 2 KV heads for 4 cards, so KV heads are **replicated** onto the cards that share them.
- **MLP/FFN — split by intermediate dim.** `gate_proj`/`up_proj` (1536→8960) column-parallel → **8960/4 = 2240** channels per card; `down_proj` (8960→1536) row-parallel → each card produces a **partial** output vector.
- **Combine.** After `o_proj` and after `down_proj` the cards **all-reduce (sum) over p2p** to rebuild the full vector → **~2 syncs × 28 layers = ~56 cross-card syncs per token**.
- Embedding / LM head (vocab 151936) is vocab-split across cards.

**Net:** ¼ weights + ¼ compute per card, at the cost of a per-layer all-reduce. For a small model the sync dominates (sub-linear scaling); for 7B/14B the per-card compute outweighs the sync (near-linear) and you can run models too big for one card's 31 GB.

---

## How to run

```bash
# 1 card
sg qaic -c "bash /home/chihao/models/run_qeff19.sh"
# 4 cards, tensor-parallel
sg qaic -c "bash /home/chihao/models/run_qeff19_4card.sh"
```
Change the question by editing the `--prompt "..."` line in the script. Switch `--model-name` to a bigger model (e.g. `Qwen/Qwen2.5-7B-Instruct`) to see near-linear 4-card scaling; the first run recompiles a `.qpc` (cached afterward). `sg` runs in `sh`, so always call a `#!/bin/bash` script (not inline `source`).

---

## Results — "What is the capital of France?"
Completion (both): *"The capital of France is Paris. It is located on the Seine River ..."* (correct)

| Config | Decode tok/s | Total tok/s | TTFT | Speedup (decode) |
|---|---:|---:|---:|---:|
| 1 card `[0]` | 66.8 | 64.6 | 0.02 s | 1.0× |
| 4 cards `[0,1,2,3]` | **115.7** | **111.3** | **0.01 s** | **1.73×** |

**1 vs 4 cards: only ~1.7× faster despite 4× the hardware** (ideal would be ~4× / ~260 tok/s). It is **sub-linear** because Qwen2.5-1.5B is small — each card's ¼ compute slice is tiny, so the per-layer all-reduce p2p sync (~56 syncs/token) dominates and eats most of the gain. Tensor-parallel only pays off (near-linear) on models too big for one card (7B/14B).

vs Intel iGPU (mllm OpenCL, Qwen3-1.7B): decode ~3.7 tok/s → AI 100 single card is ~18× faster.

---

## Results — UCSD instruction prompt (4 cards, `prompt-len 256 ctx-len 512`)
Prompt: a constrained "explain UCSD in ≤12 words, one sentence" instruction.

Completion: *"UCSD is a top-ranked university known for its strong computer science, engineering, and hardware research."* (then over-generated — 1.5B doesn't stop cleanly or fully honor the word-limit/format constraints).

Perf: TTFT 0.04 s · Decode **111.3 tok/s** · Total ~100 tok/s.

> Perf note: these TTFT / tok/s numbers are QEfficient's **built-in inline stats** (printed every run). They are **not** a Perfetto `.perf` trace (that was the mllm OpenCL path). Per-op AI 100 profiling would use `/opt/qti-aic/tools/` (opstats-profiling), separately.

---

## Commands
Run scripts (edit `--prompt "..."` inside to change the question):
```bash
# Single card
sg qaic -c "bash /home/chihao/models/run_qeff19.sh"
# 4 cards (tensor-parallel)
sg qaic -c "bash /home/chihao/models/run_qeff19_4card.sh"
# 4 cards, UCSD instruction prompt (prompt-len 256)
sg qaic -c "bash /home/chihao/models/run_qeff19_ucsd.sh"
```

Device status:
```bash
sg qaic -c "/opt/qti-aic/tools/qaic-util -q" | grep -E "QID|Status"
```

---

## Context-length sweep — 1 card vs 4 cards

Qwen2.5-1.5B on the AI 100 NPU, **prompt-len scaled to fill the context** (so prefill/decode actually run at the target length), 32 tokens generated for ctx≥64. `decode_time = gen_tokens / decode_tok/s`; `total = prefill + decode`.

**1 card `[0]`**

| ctx-len | prompt-len | prefill (s) | decode tok/s | decode (s) | total (s) |
|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 0.02 | 67.2 | 0.12 | 0.14 |
| 32 | 16 | 0.02 | 67.9 | 0.24 | 0.26 |
| 64 | 32 | 0.02 | 66.1 | 0.48 | 0.50 |
| 128 | 96 | 0.03 | 65.8 | 0.49 | 0.52 |
| 256 | 224 | 0.03 | 67.2 | 0.48 | 0.51 |
| 1024 | 992 | 0.13 | 60.9 | 0.53 | 0.66 |
| 2048 | 2016 | 0.61 | 59.7 | 0.54 | 1.15 |

**4 cards `[0,1,2,3]` (tensor-parallel)**

| ctx-len | prompt-len | prefill (s) | decode tok/s | decode (s) | total (s) |
|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 0.01 | 116.6 | 0.07 | 0.08 |
| 32 | 16 | 0.01 | 120.2 | 0.13 | 0.14 |
| 64 | 32 | 0.01 | 122.7 | 0.26 | 0.27 |
| 128 | 96 | 0.03 | 113.6 | 0.28 | 0.31 |
| 256 | 224 | 0.04 | 113.3 | 0.28 | 0.32 |
| 1024 | 992 | 0.11 | 98.9 | 0.32 | 0.43 |
| 2048 | 2016 | 0.26 | 75.6 | 0.42 | 0.68 |

**4-card speedup vs 1 card**

| ctx-len | decode tok/s | prefill time | total time |
|---:|---:|---:|---:|
| 64 | 1.86× | ~1× | 1.85× |
| 256 | 1.69× | ~1× | 1.6× |
| 1024 | 1.62× | 1.2× | 1.5× |
| 2048 | **1.27×** | **2.35×** | **1.68×** |

**Takeaways (the two phases scale oppositely with context):**
- **Decode (tok/s)** — 4 cards ≈ 1.6–1.9× at short/medium context, **fading to 1.27× at 2048** (sync-bound: each token does tiny per-card work but pays the ~56 p2p all-reduces/token).
- **Prefill (time)** — break-even when short, **growing to 2.35× at 2048** (compute-bound: the big matmuls split cleanly across cards).
- **Total time** — ~1.5–1.7× overall; the win is largest at long context (1.68× @ 2048), driven by prefill.

### Run the sweep
Script: [`/home/chihao/models/sweep_scaled.sh`](file:///home/chihao/models/sweep_scaled.sh) (loops ctx-len 16→2048, sets `prompt-len = ctx-len − gen`).
```bash
# 1 card
sg qaic -c "DEVGROUP='[0]'        bash /home/chihao/models/sweep_scaled.sh"
# 4 cards (tensor-parallel)
sg qaic -c "DEVGROUP='[0,1,2,3]'  bash /home/chihao/models/sweep_scaled.sh"
```
First run of each (ctx-len, device-count) compiles a `.qpc` (cached afterward, so re-runs are instant). The `qaic-exec` compiler prints prefill/decode/E2E stats per point.

---

## Next options
- Larger model for real 4-card scaling + better instruction-following: `Qwen/Qwen2.5-7B-Instruct` (or 14B), tensor-parallel `[0,1,2,3]`.
- Throughput mode: 4 single-card **replicas** (`[0]`,`[1]`,`[2]`,`[3]`) for ~4× aggregate on small models.
- Per-op AI 100 profile via the SDK opstats tools.



