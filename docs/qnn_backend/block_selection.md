# Block Selection: XAttention antidiagonal scoring for block-sparse Qwen3

This doc covers the **block-selection** stage of the block-sparse Qwen3 path:
deciding *which* historical K/V blocks each query block attends to. The
[split-prefill work](split_prefill.md) brought the block-sparse model up to
coherent, fast (~1.6–1.75× over dense) prefill, but with a **placeholder
selector** — `selectTopKBlocks` keeps the attention sink (block 0) + the
most-recent block and fills the rest with *random* middle blocks. That keeps
the model fluent but cannot retrieve facts from arbitrary distant blocks
(see [split_prefill.md § "random-middle selection can't do exact retrieval"](split_prefill.md#L162)).

The goal here is to replace random-middle with **attention-score-based**
selection, following the XAttention algorithm (Xu et al., ICML 2025,
*Block Sparse Attention with Antidiagonal Scoring*, arXiv:2503.16428,
[mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention)).

> **⚠️ CORRECTION (2026-05-20): the scorer matmul was a silent no-op until now.**
> The scorer called `functional::matmul(..., MatMulOpType::kBLAS)`, but the
> Android build is `MLLM_USE_BLAS=OFF`, so `kBLAS` compiles to a `NYI()` that
> only **prints** and computes nothing (device log c3.log had 10080 `[NYI]`
> lines). Every measurement below dated 2026-05-20 *before this banner's section*
> — the "scoring +680 ms / 1.74× slower", "scoring is FLOP-bound ~400 ms at
> S=1", "Lever-1 wrong", "NPU not the lever" findings — was taken with the
> matmul doing nothing, so the logits were garbage and those numbers/conclusions
> are **invalid**. Any retrieval that "worked" did so via the sink+recent anchors
> or the `MLLM_FORCE_BLOCKS` oracle, not via scoring. The fix is to use
> `MatMulOpType::kMllmBlas` (real ARM gemm). **See the section
> "[Corrected results: real scoring with kMllmBlas](#corrected-results)" for the
> true timings.**

> **Status (2026-05-19):** kernel-first. Before integrating selection into the
> runtime we are writing a standalone **selection kernel** and benchmarking it
> on the three on-device compute units (ARM **CPU** / Hexagon **NPU** / Adreno
> **GPU**) to learn how expensive the scoring pass actually is. Integration
> design is sketched below but deferred until the kernel cost is known.

---

## The algorithm (XAttention antidiagonal scoring)

The screenshot algorithm ("blocks scored with diagonal patterns, choose top-k")
is XAttention's block-importance estimator. Core idea from the paper: **the sum
of antidiagonal values (lower-left → upper-right) inside each B×B block of the
attention map is a strong proxy for that block's importance.** Instead of
materializing the full B×B block, a strided reshape computes those antidiagonal
sums with a single reduced matmul.

### Reshape mechanics (from the reference `xattn_estimate`, `select_mode="inverse"`)

Per head, with `Q, K ∈ R^{L×d}`, block size `B`, stride `S` (`S | B`):

```python
# K: row r packs the S consecutive keys [r·S : r·S+S) into the feature dim
reshaped_key   = cat([K[:, :, k::S, :] for k in range(S)], dim=-1)   # [L/S, S·d]
# Q: same S positions but in REVERSED order (the "inverse"/antidiagonal pairing)
reshaped_query = cat([Q[:, :, (S-1-q)::S, :] for q in range(S)], dim=-1)  # [L/S, S·d]

attn = (reshaped_query @ reshaped_key.T) / (sqrt(d) · S)   # [L/S, L/S]
attn = softmax(attn + causal_mask, dim=-1)
# pool the reduced grid into per-block-pair scores (reshaped_block_size = B/S):
attn_sum = attn.view(N_B, B/S, N_B, B/S).sum(dim=(1,3))     # [N_B, N_B]
mask = find_blocks(attn_sum, threshold τ)                   # [N_B, N_B] bool
```

Why the reversal gives antidiagonals: entry `(rq, rk)` of the reduced matmul is
`Σ_{q=0}^{S-1} Q[(S-1-q)+rq·S] · K[q+rk·S]` — the query offset `(S-1-q)` pairs
with key offset `q`, so the two offsets always sum to `S-1`. That is exactly the
antidiagonal of the S×S sub-tile, summed by the dot product over the packed
`S·d` feature dim.

### Cost (this is what the benchmark measures)

| Quantity | Full attention scores | XAttention scoring |
|---|---|---|
| QKᵀ MACs (per head) | `L²·d` | `(L/S)·(L/S)·(S·d) = L²·d / S` |
| reduction factor | 1× | **S×** cheaper |

So scoring is a factor-`S` cheaper matmul of shape `[L/S, S·d] × [S·d, L/S]`,
a softmax over `[L/S, L/S]`, a block-pool reduction to `[N_B, N_B]`, and a cheap
per-row sort/threshold in `find_blocks`. The paper reports up to 13.5× attention
speedup overall with this estimator + block-sparse compute.

### `find_blocks` (threshold τ)

Per query block (row of `attn_sum`): always keep **block 0 (sink)** and the
**diagonal (self)** block; sort the remaining blocks' scores descending and take
the smallest prefix whose cumulative mass reaches `τ · total` (default
`τ = 0.9`). Optional `keep_sink` / `keep_recent` force those slots. Output is a
variable-count boolean block mask.

---

## How it maps onto our split architecture

The split prefill loop ([`ShaBlockSparsePromptProcessorSplit::prefill`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp#L490))
already calls selection **per-layer, per-qb**, and at that exact point all the
inputs the scorer needs are sitting on the CPU:

- **Q** for layer `i`'s current query block: `q_full_[0]` `[1,Hq,Sq,D]` (uint16).
- **K** for all historical key blocks: committed to the uint8 KV cache by
  `copy_kv_to_cache(i, …)` just before the qb loop — `[Hkv, D, max_kv_len]`.

So the scorer is a **CPU (or offloaded) step** that replaces the random-middle
picker. The monolithic path shares one `sel` across all layers and can't do
per-layer Q·K scoring without restructuring — split already has the right shape.

### Quantization is friendlier than expected for *ranking*

- **K** is `kUInt8PerTensorSym` **per head** (one scale/head, `zp = 128`). Within
  a head the K scale is a uniform positive multiplier — irrelevant to ranking
  blocks against each other.
- **Q** is per-head per-tensor **asym** (`q_rope_add_0_output_qdq_h{h}`). Ranking
  historical blocks within a head only needs **Q's per-head zero-point** to
  center Q before the integer dot product; the Q/K scales cancel out of a
  within-head ranking.

So a faithful integer-domain scorer needs just the per-(layer, head) Q
zero-point. **The runtime currently has no access to any QDQ scale/zp** — every
dequant is baked inside the QNN graph and the runtime only shuffles raw
uint8/uint16 codes (verified: no scale/zp reference anywhere in `aot_rt/`).
Block selection would be the *first* place the runtime needs a baked constant.
Options when we get there: export a small sidecar of Q zero-points from the
compile driver, load the `.mllm` params in the runner, or score on the NPU/GPU
where dequant is natural.

### One hard constraint: the compiled graph has fixed slots

The attn graph is compiled with `kHistKBK = (kTopK-1)·kBK = 224` historical
columns and a fixed-shape mask. XAttention's **variable-count** threshold-τ
selection (line 15) does not fit without recompiling the graph shape. The
drop-in version is **fixed top-k by score**: keep sink (block 0) + most-recent
(block qb-1) anchors — the two slots that fixed the attention-sink bug — and
fill the remaining `kTopK-3` slots with the highest-scoring middle blocks
instead of random ones. (A future variable-count variant would need a graph
shape change.)

---

## Plan

1. **Kernel + benchmark (current).** Standalone XAttention scoring kernel,
   benchmarked on ARM CPU / Hexagon NPU / Adreno GPU at realistic dims
   (Hq=16, d=128, B=32, S∈{4,8,16}, L sweep). Decide where scoring should run.
2. **Validate selection quality** offline: does score-based top-k recover the
   needle-in-haystack blocks the random selector misses?
3. **Integrate** into `ShaBlockSparsePromptProcessorSplit::selectTopKBlocks`
   (per-layer, per-qb), wiring in the Q zero-points via the chosen scale source.
4. **Re-run retrieval tests** (NIAH1_1k, MK2_500) that currently fail.

### Benchmark dimensions

Qwen3-1.7B selection uses the **query heads**: `Hq=16`, `D=128`, `B=BK=32`.
Stride `S` must divide `B`; `S∈{4,8,16}` (reshaped_block_size `B/S ∈ {8,4,2}`).
`L` swept across the model's chunk sizes {256,512,1024} and beyond
{2048,4096,8192} to show scaling. Compare against dense-score FLOPs `Hq·L²·D`.

---

## Benchmark kernel + results (2026-05-20)

Standalone microbenchmark:
[`examples/block_selection/bench_block_selection.cpp`](../../examples/block_selection/bench_block_selection.cpp)
(`mllm-bench-block-selection`). It expresses the scoring core from
device-switchable mllm Functional ops so the same code runs on CPU / OpenCL:

- **score core** (device-resident, the dominant work): `softmax(Qr @ Krᵀ)` over
  `[1,Hq,Lr,Lr]`. The `1/(√d·S)` scale is folded into Q at setup. Causal mask is
  omitted from the timed core (an elementwise add; changes neither matmul nor
  softmax cost — this phase measures raw scoring speed, not selection
  correctness).
- **pool+select** (always CPU; `O(Hq·Nb²)`): block-pool the reduced grid into
  `[Nb,Nb]` scores + per-row top-k. Includes the device→CPU readback.

> The "inverse"/antidiagonal feature ordering and the causal mask are deferred
> to the correctness phase — they don't change the timed FLOP shape. The reshape
> is done with `view` (CPU `reshape` forward is NYI — metadata-only `view`
> works for contiguous tensors). OpenCL has **no `sum`/`topk` op**, which is why
> pool+select is fixed to CPU.

**Device: SM8750 (Snapdragon 8 Elite), Adreno GPU, Hexagon V79 NPU.** `Hq=16
d=128 B=32 S=8 topk=8`, 20 reps / warmup. CPU/GPU via the example above
(fp32; OpenCL also tried fp16, no change). NPU via the
[`GemmLatencyTest.BlockSelScore_*`](../../tests/qnn/GemmLatencyTest.cpp) cases —
a 2-node `MatMul(transpose)→Softmax` HTP graph, fp16, both inputs dynamic.
`GF/s` is from the reduced-matmul MAC count `Hq·Lr²·(S·d)`.

| L | Lr | CPU core ms | CPU GF/s | NPU core ms | NPU GF/s | GPU core ms | GPU GF/s | pool+sel ms (CPU) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 32 | 0.049 | 691 | 0.166 | 202 | 1.38 | 24 | 0.09 |
| 512 | 64 | 0.124 | 1079 | 0.183 | 734 | 3.47 | 39 | 0.27 |
| 1024 | 128 | 0.369 | 1453 | 0.323 | 1660 | 12.8 | 42 | 0.47 |
| 2048 | 256 | 0.616 | 3485 | 0.763 | 2813 | 48.4 | 44 | 1.48 |
| 4096 | 512 | 2.19 | 3920 | 2.25 | 3820 | 190 | 45 | 4.71 |

**Findings:**
1. **CPU (BLAS) and NPU (HMX) are on par** — both ~2.2 ms to score L=4096
   (~3.8 TFLOP/s), and the NPU is marginally *faster* at L=1024 (0.32 vs
   0.37 ms). The NPU pays a higher fixed dispatch cost, so the CPU wins at small
   L (256: 0.05 vs 0.17 ms). (`--matmul MllmBlas` on CPU is ~50× slower than
   `BLAS` — wrong path; always use `BLAS`.)
2. **Adreno OpenCL eager is unoptimized for this shape** — flat ~45 GFLOP/s
   (<5% of peak), and fp16 made no difference, so the bottleneck is the kernel,
   not dtype. The mllm eager OpenCL matmul doesn't batch the 16-head
   `[Lr,1024]×[Lr,1024]ᵀ` well. A competitive GPU number would need a custom
   fused scoring kernel, not eager ops.
3. **pool+select is cheap** (≤ a few ms on CPU even at L=4096), dominated by the
   readback — keeping it on CPU is fine.

**Takeaway for integration:** at the model's real chunk sizes (Sq ≤ 1024) the
score core is **≪ 1 ms per layer on both CPU and NPU**. CPU scoring is the
pragmatic first integration — the data (KV cache + a Q readback) is already
CPU-side and it needs no graph machinery. **NPU offload is the better long-term
home** because in the split graph Q is *already resident on HTP* (it's the
`q_{i}` chunk output) and K is the on-HTP cache — scoring there avoids the Q
readback entirely and runs at parity. The GPU is not competitive without kernel
work. Recommend: integrate CPU scoring first to validate retrieval, then move
the score core into the per-layer HTP graph if profiling warrants.

---

## CPU scorer integrated into the split runtime (2026-05-20)

The CPU scorer is wired into
[`ShaBlockSparsePromptProcessorSplit::selectTopKBlocks`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp)
(now takes `layer`), gated on `enableScoreBasedSelection(q_zp)`. When enabled,
the `qb_global > n_hist` branch keeps the **sink (block 0)** and **most-recent
(block qb-1)** anchors and fills the remaining `kTopK-3 = 5` slots with the
**highest-scoring** middle blocks (per head) instead of random ones. If not
enabled, it falls back to the original random-middle policy (so existing runs
are unchanged).

### Scoring formula (softmax attention mass — needs Q zp + Q/K scales)

`computeBlockScores(layer, qb_global)` does, per head:

```
dot   = Qc @ Kcᵀ                       # Qc=[1,Hq,BQ,D], Kc=[1,Hq,hist,D]
logit = dot * (q_scale·k_scale) / √D   # real-valued attention logit
prob  = softmax(logit, over all hist keys)        # per query row
score(block kb) = Σ_q  Σ_{k in kb}  prob[h,q,k]    # total attention mass
```

- `Qc = Q_code − Qzp[layer]` (uint16 Q from `q_full_`, centered); `Kc = K_code −
  128` (uint8 K, symmetric). The K cache is transposed `[Hkv,D,max_kv_len]`,
  de-transposed + GQA-expanded + centered into a per-layer `[Hq,Sq,D]` fp32
  buffer **once per layer** (`kc_layer_`), then sliced per qb.
- **softmax is essential, not optional.** A first cut used a *scale-free*
  `Σ_q maxₖ (centered Q·K)` (no softmax). On NIAH it gave **near-flat scores
  (~7 % spread)** and didn't reliably rank the needle block — because raw dot
  magnitude is dominated by token/key **norm**, not query relevance. Per-query
  softmax normalizes the norm away and surfaces the block the queries actually
  attend to. That requires the real `q_scale·k_scale/√D` temperature, hence the
  scale load. (Sum-of-mass over the block ≈ XAttention's post-softmax pooling.)
- The matmul uses the optimized `MatMulOpType::kBLAS`. This is the **exact**
  (non-strided) scorer; the stride-S antidiagonal reduction (8× cheaper,
  benchmarked above) is a later optimization.

### Where the scales come from

Each head's Q/K is QDQ'd with `..._qdq_h{h}`, but those per-head scales are
**copies of one MHA-level value** (`copyQDQParams` in
`prepareParametersForSHA`), and the `concat` into the boundary forces a single
QNN encoding — so one triple `(q_zp, q_scale, k_scale)` per layer describes all
of `q_full_` / the K cache. The split runner
([`aot_run_sha_blocksparse_causal_split.cpp`](../../examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal_split.cpp))
gained a `--params <ptq_lpbq.mllm>` flag: per layer it pulls
`q_rope_add_0_output_qdq.fake_quant.{zero_point,scale}` and
`k_cast_to_int8_qdq.fake_quant.scale`, frees the params **before**
`initQnnBackend` (memory), and calls `enableScoreBasedSelection`. Without
`--params`, scoring stays off. (This is the runtime's first use of baked QDQ
constants — everything else stays code-only.)

### Validation — softmax scoring fixes NIAH retrieval ✅ (2026-05-20, SM8750 a615391a)

End-to-end on the **Sq=1024** split context (recompiled; the on-host bin had
been a Sq=256 compile — running it at `--sq 1024` misaligns every graph I/O and
produces token soup, which masqueraded as a model bug. A bin must run at its
compiled Sq). `NIAH1_1k.txt`, 818 tok, needle `diligent-joke is: 8090293` (≈
block 9 after the chat-template offset), question ≈ block 24:

| Selection | Generated answer | needle? |
|---|---|---|
| **Random** middle | "…magic number … is **123**." | ❌ |
| **Oracle** (force blocks 7-10) | "…:\n\n**8090293**" | ✅ (ceiling) |
| **Softmax XAttention score** | "…magic number is **8090293**." | ✅ |

The progression that got here, all on the same prompt:

1. **Oracle test was decisive.** Force-selecting the needle blocks
   (`MLLM_FORCE_BLOCKS=7,8,9,10`) made the model emit `8090293` exactly — proving
   the **model can retrieve at 818 tok; the bottleneck is selection**, not
   quantization/length degradation.
2. **The scale-free `Σ_q maxₖ` scorer was too weak** — needle block ranked only
   mid-pack (flat scores), selected at some layers but not reliably; end-to-end
   it failed (rambled).
3. **Switching to softmax attention-mass scoring fixed it** — the per-query
   normalization surfaces the needle block, which is then selected at the
   retrieval layers, and the model outputs `8090293`. Random (no scoring) still
   answers `123`.

**Memory ordering:** the runner extracts the per-layer `(q_zp, q_scale, k_scale)`
and frees the 2.4 GB `.mllm` **before** `initQnnBackend`, so params don't coexist
with the 1.6 GB context + PD reservation.

**Caveats / next:** (1) validated on one needle prompt — should sweep
`MK2_500.txt` (multi-key) and varied needle depths; (2) the `MLLM_DUMP_SEL`
printout uses `%.0f`, too coarse for softmax masses (shows "1") — cosmetic.

### Verified against the official XAttention repo (2026-05-20)

[`scripts/verify_xattn.py`](../../scripts/verify_xattn.py) cross-checks our
scorer against the reference `xattn_estimate` (mit-han-lab/x-attention @
`e379887`, inverse/antidiagonal mode), on H100 with random fp16 Q/K
(H=4, D=128, B=32):

| S | replica vs `xattn_estimate` max\|Δ\| | ours-vs-official Spearman | top-5 overlap |
|---|---|---|---|
| **1** (= full Q·Kᵀ, our default) | 7.1e-3 MATCH | 0.990 | 0.995 |
| 2 | 3.5e-3 MATCH | 0.953 | 0.985 |
| 4 | 2.0e-3 MATCH | 0.989 | 0.985 |
| 8 | 5.4e-4 MATCH | 0.969 | 0.995 |

- **Scoring math matches.** A from-scratch reimplementation of the official
  inverse-mode formula (reshape pack `D→S·D`, scale `1/√D/S`, softmax,
  block-pool) reproduces `xattn_estimate()` to fp16 tolerance at every stride —
  i.e. our reshape/scale/softmax/pool *is* XAttention's.
- **Our runtime scorer ranks blocks ~identically to official** (Spearman
  ≥0.95, top-5 overlap ≥0.985). The small gap is exactly our intentional
  deviations: per-qb incremental scoring (per-block-equivalent to their full
  grid), **history-only softmax** (we exclude the diagonal/current block — it's
  handled by a separate mask slot — which renormalizes per query), and the C++
  using dequantized uint16 Q / uint8 K (same formula). The fixed-k slot policy
  (sink + recent + top-k vs their threshold-τ) is a graph-shape constraint on
  *selection*, not the scoring.

So the S=8 retrieval failure above is **not** a divergence from XAttention — our
S=8 scoring matches theirs; it's the method's inherent coarseness for
single-token needles in 32-key blocks.

### Prefill speed: block-sparse + XAttention selection vs dense (2026-05-20)

On-device (SM8750 a615391a), 818-token prompt, Sq=1024, **prefill only** (`--gen 0`),
dense via `mllm-qwen3-aot-runner --ar_len 32`:

| Method | Prefill | tok/s | vs dense |
|---|---|---|---|
| **Dense full attention** (ar_len=32) | 698 ms | 1172 | 1.00× |
| Block-sparse split, **no scoring** (random sel) | 537 ms | 1524 | **1.30× faster** |
| Block-sparse + XAttention **S=8** | 685 ms | 1194 | 1.02× |
| Block-sparse + XAttention **S=4** | 701 ms | 1167 | 0.99× |
| Block-sparse + XAttention **S=2** | 808 ms | 1012 | 0.86× |
| Block-sparse + XAttention **S=1** (full Q·Kᵀ, the only correct one) | 1215 ms | 673 | **0.57× (1.74× slower)** |

**The block-sparse attention is ~1.3× faster than dense, but the CPU-side
XAttention scoring overhead wipes that out.** At S=1 (the stride that retrieves)
scoring adds **+680 ms** → 1.74× *slower* than dense; S=8 only breaks even.
(At a chunk-filling ~1000-tok prompt the no-scoring split advantage grows toward
the doc's ~1.5×, since split always pays the full Sq=1024; here 818 underfills.)

**Why scoring costs +680 ms when the standalone kernel benchmarked at ~2 ms:**
the runtime scores **per-(layer, qb)** — ~28 layers × ~26 query blocks ≈ **730
tiny matmuls**, each paying fp32 tensor construction + K-dequant + a BLAS
dispatch on small matrices. That per-call overhead is **stride-independent and
dominant**, which is why stride helps only modestly (S=1→S=8: 1215→685 ms) and
plateaus past S=4. To make selection pay off, scoring must get **off the
critical path**: batch all qb of a layer into one matmul (or all layers), reuse
the dequantized K (already cached per layer) without rebuilding tensors, or
offload to the NPU where Q/K already live (benchmarked at parity with CPU and
avoiding the readback). As-is, the placeholder/random selector is the only
config that keeps the block-sparse prefill win.

### OpenMP-parallel scoring + the double-buffer pitfall (2026-05-20)

`computeBlockScores`'s per-head loops (K dequant, Q reshape, softmax+pool) are
`#pragma omp parallel for` over `Hq=16` (independent; `prob` is per-thread). This
required adding `-fopenmp` to the **QNN backend** target — it links MllmRT but
didn't inherit MllmRT's OpenMP compile flag, so the pragmas were silently no-ops
until [`mllm/backends/qnn/CMakeLists.txt`](../../mllm/backends/qnn/CMakeLists.txt)
applied the flag explicitly (mirroring the OpenCL backend).

Result (S=1, no pipeline, 818-tok prefill, scoring on the main thread):

| OMP threads | prefill | scoring overhead (− 537 ms attn baseline) |
|---|---|---|
| 1 | 1147 ms | 610 ms |
| **2** | **885 ms** | **~350 ms (1.7×)** |
| 4 | 908 ms | 371 ms |
| 8 | 918 ms | 381 ms |

OMP=2 is the sweet spot; more threads regress — the loops are memory-bound
(dequant/softmax) and there are ~500 small per-qb calls, so per-call barrier
overhead + bandwidth saturation cancel the parallelism. Correctness holds at all
thread counts (`8090293` ✅). **Still 1.27× slower than dense (698 ms)** — OMP
shrinks the scoring tax but doesn't eliminate it.

> **Pipeline regression found & worked around.** A parallel session double-buffered
> the QNN-facing per-qb buffers (2 slots, `qb&1`) to overlap CPU prep with the
> NPU. That **garbles any multi-qb prompt**: [`QNNBackend::graphExecute`](../../mllm/backends/qnn/QNNBackend.cpp#L813)
> binds a graph's I/O buffer only on its *first* dispatch (`!isAlloc()`), so the
> alternating slot pointer is ignored and `attn_i` reads a stale slot for qb>0.
> The old single-buffer-per-layer code worked because the pointer never changed
> (same buffer, refilled). Fix (for now): the serial path forces `slot=0`
> always (single buffer, bind-once-valid) — coherence restored. Re-enabling the
> pipeline needs a `graphExecute` that rebinds per call, **or** overlapping only
> the scoring (which produces just the `sel` int array, no QNN buffer) while
> gather/stage stay on the main thread into the single bound buffer. That
> overlap is what would finally hide the ~350 ms scoring behind NPU time and let
> selection beat dense — deferred.

### Option 2 — scoring-only pipeline overlap (2026-05-20)

To hide scoring behind the NPU without re-triggering the bind-once bug, the
pipeline overlaps **only** the scoring on the worker (`selectTopKBlocks` →
produces the `sel` int array, no QNN buffer); the cheap gather/stage + NPU
dispatch stay on the main thread into the single slot-0 buffers
(`MLLM_SPLIT_PIPELINE=1`, worker pinned via `MLLM_PIPELINE_WORKER_CPUS`). `sel`
is double-buffered so the worker scores qb+1 while main consumes qb. Correct
(`8090293` ✅).

Measured (818-tok, OMP=2; no-scoring baseline 475 ms here, dense ≈ 1.3× the
no-scoring split):

| S | serial | pipeline | timing breakdown (S=1) |
|---|---|---|---|
| 1 | 884 | **795** | prep(main)=83, dispatch=141, **pp_wait=249** |
| 2 | 659 | 663 | |
| 4 | 604 | 605 | |
| 8 | 592 | 630 | |

- **Helps only at S=1** (~90 ms) and is **neutral-to-worse at S≥2**. The cap is
  the `pp_wait`: scoring (≈350 ms at S=1) outruns main's overlap window
  (gather/stage 83 + dispatch 141 = 224 ms), and the ~896 per-qb condvar
  handoffs add ~130 ms of pure synchronization — which is why S=8 (tiny scoring)
  *regresses* (handoff cost > scoring saved). A coarser producer/consumer handoff
  (worker churns all qbs, main polls a ready-count) would cut that overhead.
- **Even with the pipeline, S=1 (795 ms) doesn't beat dense** — the scoring tax
  is too large to hide behind the per-qb attn alone (the big chunk dispatches
  can't overlap scoring due to the Q-from-chunk_i dependency). The block-sparse
  prefill only beats dense in the **approximate** S≥2 regime, where serial
  already suffices.

### Buffer reuse — attempted, reverted (engine memoizes matmul by tensor identity)

Reusing fixed scoring-scratch Tensors (`sc_Q_`, `sc_K_[qb]`) across the ~500
per-qb calls to avoid `Tensor::alloc` **broke retrieval deterministically**
(answered "7", not 8090293, even at OMP=1). Cause: mllm's engine
(`buildOpAndSubmitTask`) memoizes the matmul by input-tensor identity, so
feeding the *same* `sc_Q_`/`sc_K_` handles every call returns a **stale cached
result** → wrong scores → wrong selection. Reverted to per-call alloc (correct).
A correct reuse must wrap reused storage in **fresh Tensor handles** each call,
or bypass the engine with a direct gemm into a reused buffer (which would also
drop the per-call engine-dispatch overhead) — deferred.

### Lever 1 (full-grid per-layer matmul) — tried, REVERTED; scoring is FLOP-bound

Hypothesis: the integrated scorer (~346 ms) was ~175× the standalone kernel's
"~2 ms ideal", so consolidating the ~500 per-qb matmuls into **one big
`[Hq,Lr,Lr]` matmul per layer** should reclaim that gap. Implemented (history-only
softmax preserved, so selection is identical) and measured — it was **slower**:

| S | per-qb (kBLAS) | Lever-1 full-grid (GGUF) |
|---|---|---|
| 1 | 905 ms | **2147 ms** |
| 2 | 659 | 1211 |
| 4 | 604 | 840 |
| 8 | 592 | 660 |

Two findings killed it:
1. **`kBLAS` is buggy for the large batched shape.** `MatMulOpType::kBLAS` gave
   *flat/garbage* logits for `[16,1024,1024]` (uniform softmax → picks lowest
   blocks → answers "7"), while correct on the small per-qb shapes. `kGGUF`
   (llamafile) and `kMllmBlas` are correct there; GGUF is the faster correct one.
2. **The "~2 ms ideal" was a mirage** — the standalone benchmark used `kBLAS`,
   which at L=1024 was computing that *same wrong-but-fast* result. The correct
   full grid is **2× the FLOPs** (it computes the masked-out upper triangle) and,
   on a correct kernel, is slower than the per-qb causal slices.

**Conclusion: scoring is FLOP-bound (~400 ms of Q·K at S=1), not
per-call-overhead-bound.** The per-qb `kBLAS` scorer (correct on small causal
slices) is already the fast+correct CPU choice; consolidating matmuls only adds
FLOPs. Since correct (S=1) scoring (~400 ms) exceeds the block-sparse saving over
dense (~160 ms), **selection-by-scoring cannot beat dense on the CPU** — the real
lever is to run the score matmul on the **NPU (HMX)**, which has the FLOP
throughput to make it cheap (the standalone NPU benchmark was at CPU parity for a
*single* matmul and HMX scales far better). Reverted to the per-qb scorer.

### Where the scoring time actually goes (2026-05-20) — NPU-matmul is NOT the lever

Instrumented `computeBlockScores` (`MLLM_BLOCKSEL_TIMING=1`). At S=1, OMP=2, the
818-tok scoring is **410 ms total = matmul 2.7 ms (1%) + tail 408 ms (99%)**.
The matmul (kBLAS on the small per-qb causal slices) is already cheap; the cost
is the **CPU softmax+pool+dequant loops** — touching the full query×history logit
grid (max-pass, exp-pass, pool-pass) plus dequantizing Q/K. This **invalidates
the NPU-scoring plan**: offloading the matmul to HMX would save ~2.7 ms.

Two real (kept) micro-opts on the tail:
- **`fast_exp`** (degree-3 2^x approx, ~10× faster than libm `expf`; exact enough
  for monotone ranking): tail 408 → 326 ms. Retrieval unchanged (`8090293` ✅).
- **OMP scaling** of the tail is sub-linear / memory-bound: 530 / 333 / 284 /
  252 ms at OMP 1 / 2 / 4 / 8 (only 2.1× over 8 threads).

Best correct (S=1, OMP=8, fast_exp): **785 ms prefill** — still > dense (~617 ms).

**Fundamental conclusion.** S=1 XAttention scoring is **O(L²)** — it scores every
query block against all history, the *same order of work as dense attention
itself*. So on the CPU it cannot be made cheaper than the block-sparse saving it
buys (~140 ms), regardless of fast_exp / OMP / NPU-matmul. XAttention's actual
speedup *depends on* the stride-S reduction to make scoring sub-quadratic — but
S>1 drops the single-token needle here (§ below). Net: **for precise retrieval,
selection-by-scoring can't beat dense on this CPU.** Genuine paths left: (a) move
the *entire* scoring (matmul+softmax+block-reduce) onto the NPU emitting only the
tiny [Nb,Nb] block scores (no big-logit readback) — uncertain whether HMX beats
the memory-bound CPU loops, sizeable build; (b) larger blocks (B=64/128) so the
stride reduction is far less lossy — a model/graph change; (c) accept S≥2
approximate selection for non-retrieval (coherence) use, where it already beats
dense.

### S=8 (practical regime) bottleneck: the redundant per-qb K memcpy (2026-05-20)

With `MLLM_BLOCKSEL_TIMING=1`, the S=8 scoring (132 ms) splits as: kc
de-transpose 13 ms, **prep (Q-build + K-slice memcpy) 111 ms (84%)**, matmul
2.7 ms, softmax+pool 4.9 ms. So at S≥2 the bottleneck is **neither matmul nor
softmax** — it's the per-qb **K-slice memcpy**: each query block copies its
*growing* `[0,hist)` history out of `kc_layer_` into a fresh contiguous K
tensor, so block 0 is copied ~Nb×, block 1 ~(Nb-1)× … → **~2.2 GB of redundant
copy**, stride-independent (dominates once the softmax shrinks).

Removing it (point the matmul at the per-layer K instead of recopying) is
**blocked by mllm's CPU matmul backends**:
- full-K matmul (whole `kc_layer_`, softmax only `[0,histr)`): `kBLAS` returns
  **flat/garbage** for that shape — independent of N (S=8 N=128 also flat) and of
  handle reuse — correct only via `kGGUF`, which is ~50× slower per call;
- `kc_layer_.slice([0,histr))` view: `kBLAS` mishandles the strided batch → wrong.
- Correct+fast `kBLAS` works only on the freshly-built contiguous per-qb K — i.e.
  it *requires* the memcpy.

**Tried the direct-gemm fix (reverted).** Implemented an engine-bypassing batched
gemm (`mllm::cpu::arm::mllm_blas_batch_matmul_fp32`) that reads `kc_layer_`'s
per-head history prefix **in place** via `B_batch_stride = Lr·SD` with `N=histr`
— eliminating the memcpy (prep 92 → **4 ms**, confirmed). But mllm's own ARM gemm
is the wrong tool:
- **S=1: correct (`8090293` ✅) but the matmul ballooned to 711 ms** — mllm's arm
  gemm is far slower than the `kBLAS` path (which was ~2.6 ms). (This also
  resolves the earlier "matmul=1%" reading: that was the *fast* `kBLAS`; `kBLAS`
  is genuinely fast+correct on the contiguous per-qb K, despite `MLLM_USE_BLAS=OFF`.)
- **S=8: SIGABRT** — `M=BQr=4` hits the gemm's small-M/gemv specialization, which
  doesn't honor the non-contiguous `B_batch_stride` → out-of-bounds.

So **every way to skip the memcpy is blocked at the kernel level**: `kBLAS` is
fast+correct but only on a *contiguous* per-qb K (⇒ the memcpy); a strided/full K
is wrong (`kBLAS`), slow (`kGGUF` ~50×, arm gemm ~270×), or crashes (arm gemm
small-M). Reverted to the per-qb `kBLAS` + memcpy (correct, S=8 ≈ 597 ms).

**The genuine fix** needs a kernel change, not a wiring change: either (a) fix
mllm's gemm to correctly handle a strided/sub-matrix B (small-M + custom batch
stride), or (b) build with a real cblas (`MLLM_USE_BLAS=ON`) whose `sgemm` takes
`ldb` — then one `cblas_sgemm`-per-head reads the prefix in place, no copy. Kept:
`fast_exp` (degree-3 2^x; helps the S=1 softmax) and `MLLM_BLOCKSEL_TIMING`. The
matmul-is-cheap / NPU-is-not-the-lever conclusion stands.

> **Op note (device hygiene):** a crashed runner leaves a stale process holding
> HTP PD memory; on a memory-tight device (~2 GB free) the next run then fails to
> load the ~3.6 GB context (`err 5005`, "no available PD"). `pkill -f ...split-runner`
> between runs after any crash.

### Stride-S antidiagonal reduction: wired in, but degrades single-token retrieval

The scorer's matmul can be cut ~S× by the XAttention reshape (pack S consecutive
positions into the feature dim `D→S·D`, subsample sequence `len→len/S`; the dot
over the packed feature is the antidiagonal sum). Wired into `computeBlockScores`
and gated by `MLLM_BLOCKSEL_STRIDE` (S must divide BQ=BK=32; **default S=1 =
exact**). NIAH1_1k sweep (Sq=1024, full prefill incl. scoring):

| S | prefill | answer | scoring matmul |
|---|---|---|---|
| **1 (= full Q·Kᵀ, no reduction)** | 1.22 s | `8090293` ✅ | BQ·hist·D |
| 2 | 0.80 s | "7" ❌ | ÷2 |
| 4 | 0.70 s | "7" ❌ | ÷4 |
| 8 | 0.69 s | "7" ❌ | ÷8 |

Note S=1 is literally the **full `Q·Kᵀ`** (the stride code collapses to it:
`S·D=D`, no subsample, no reversal) — i.e. the only config that retrieves is the
one with **no speedup**. **The reduction misses the needle even at S=2**, so the default stays S=1.

**Root cause (and it is *not* quantization).** The reduction **pools S keys into
one reduced logit *before* the softmax**; the needle is a single key with a sharp
logit spike, and averaging it with S−1 filler-key logits pre-softmax blurs the
spike so the softmax never sees it and the needle block isn't ranked. The exact
scorer softmaxes each key's logit individually, so the spike survives.
- Confirmed independent of the diagonal chosen: a `MLLM_BLOCKSEL_SLASH=1` toggle
  (diagonal/`+s` packing vs antidiagonal/`S-1-s`) **also** fails at S≥2 — so it's
  the summing, not the antidiagonal direction. XAttention's antidiagonal is a
  *statistical* block-importance proxy, fine for diffuse/long-context signals but
  too coarse for **precise single-token retrieval in 32-key blocks**.
- Quantization is **not** the cause: the exact S=1 scorer uses the identical
  uint16 Q + uint8 K and retrieves correctly. uint8 K (256 levels) is only a
  *secondary amplifier* — it makes the needle spike smaller, so the pre-softmax
  pooling buries it more easily; in fp the reduction would degrade more gently
  but the pooling-blur mechanism remains.

The scoring overhead the reduction would save (~0.7 s → ~0.18 s per prefill at
Sq=1024) is real, so stride stays an opt-in knob for coherence-only / approximate
use; exact (S=1, = full `Q·Kᵀ`) is required for retrieval.

Future precision-preserving speedups instead of stride: score a subset of layers
and reuse the selection; score only the trailing query rows (the question lives
at the chunk end); or use the reduction as a coarse pre-filter then re-rank
candidates exactly.

---

## Corrected results: real scoring with kMllmBlas (2026-05-20) {#corrected-results}

After discovering the `kBLAS` no-op (banner at top), switched the scorer matmul
to `MatMulOpType::kMllmBlas`, which dispatches to `arm::mllm_blas_batch_matmul_fp32`
— actually compiled on this ARM build. For our shape (`transpose_b`, `M=BQr>1`)
it takes the gemm path (`__mllm_blas_sgemm_nt_t`, explicit ldb + batch stride);
the `M==1` gemv path — the one that crashed on a strided batch in earlier
attempts — is never hit for S≤16 (BQr≥2).

**Verification (device b84a556d, 818-tok NIAH, needle at ~35% depth = middle
block qb≈9, so retrieval *requires* real scoring; no `MLLM_FORCE_BLOCKS`):**

| stride | matmul | prep (Q+K build) | softmax+pool | scoring total | prefill | retrieval |
|--------|--------|------------------|--------------|---------------|---------|-----------|
| **S=1** (exact)  | 3015 ms | 135 ms | 229 ms | 3406 ms | 3.95 s (207 tok/s) | ✅ 8090293 |
| **S=8** (approx) | 554 ms  | 126 ms | 18 ms  | 710 ms  | 1.24 s (658 tok/s) | ✅ 8090293 |

`[NYI]` count = 0 in both runs (matmul really computes now).

Key corrections to earlier conclusions:
1. **The scoring matmul is the dominant cost, not the K memcpy.** matmul is
   78–88 % of scoring at every stride; the per-qb K memcpy ("prep") is only
   ~126 ms. So the no-copy stride-view of `kc_layer_` (which only trims `prep`)
   is now a *minor* optimization, not the lever it appeared to be.

   **Zero-copy K view applied anyway (Step B, done):** the per-qb K is now a
   `TensorViewImpl::create(offset, {1,Hq,histr,SD}, {Hq·Sq·D, Sq·D, SD, 1}, kc_layer_.storage())`
   — same buffer as `kc_layer_`, but the head/batch stride stays `Sq·D` so each
   head reads its contiguous first-`histr`-rows prefix in place. The batched
   kMllmBlas gemm path honors `rhs.stride()[-3]` as the per-head batch stride and
   uses `ldb=SD`, so it's correct with no copy (M=BQr>1 keeps off the gemv path).
   Result (device b84a556d, same NIAH, retrieval ✅ at both strides):

   | stride | prep (was) | prep (view) | scoring total | prefill |
   |--------|-----------|-------------|---------------|---------|
   | S=1 | 135 ms | **16 ms** | 3328 ms | 3.87 s (211 tok/s) |
   | S=8 | 126 ms | **11 ms** | 576 ms  | 1.09 s (751 tok/s) |

   Confirms the prediction: ~115–120 ms saved, matmul unchanged and still
   dominant. Step B is a clean correctness-preserving win but does not move the
   needle on the real cost.
2. **Real S=1 scoring is ~3.4 s, not the ~400 ms claimed earlier.** The old
   "~400 ms / FLOP-bound" number was a no-op print. The arm gemm is genuinely
   slow here: M=BQr is tiny (4 at S=8) and there are 728 small calls/prefill
   (26 scored qbs × 28 layers), so it runs at ~13 GFLOP/s (far below ARM peak).
3. **Approximate S=8 scoring retrieves the needle** even when it sits in a middle
   block — XAttention's stride subsampling is good enough for this NIAH, and
   cuts the matmul ~5.4× (3015→554 ms). S=8 prefill 1.24 s.

**Next lever = a faster scoring matmul** (the memcpy is no longer it). Candidates:
- One big-M matmul per layer against full `kc_layer_` (M=Lr=128, 28 calls
  instead of 728, efficient gemm, ~2× FLOPs but far less small-M/per-call waste).
  The earlier "Lever-1 reverted as wrong" verdict was based on the no-op kBLAS,
  so it is worth re-trying with kMllmBlas.
- GPU/OpenCL offload of the scoring matmul.
- A tuned small-M sgemm micro-kernel.

### Prefill-speed sweep: seq length × stride (2026-05-20, with kMllmBlas + zero-copy view)

Device b84a556d, 4k NIAH prompt truncated to fill each window, `--gen 0`
(prefill-only). "no-score" = no `--params` (random-middle selection, block-sparse
NPU compute only, scoring disabled).

**Sq=1024** (num_qb=32 → top-8, selection is real):

| config | prefill | tok/s | scoring total | matmul | overhead vs no-score |
|--------|---------|-------|---------------|--------|----------------------|
| no-score (block-sparse only) | **475 ms** | 2157 | — | — | baseline |
| XAttention S=1 | 3861 ms | 265 | 3326 ms | 3018 ms | +3386 ms (8.1× slower) |
| XAttention S=2 | 2321 ms | 441 | 1794 ms | 1741 ms | +1846 ms (4.9×) |
| XAttention S=4 | 1525 ms | 672 | 1007 ms | 967 ms  | +1050 ms (3.2×) |
| XAttention S=8 | 1097 ms | 934 | 579 ms  | 544 ms  | +622 ms (2.3×) |

**Sq=256** (num_qb=8 ≤ kTopK=8 → no selection): ~97 ms (≈2630 tok/s) for *every*
config — all blocks fit, so scoring is ~0 and block-sparse == dense here.

Findings:
- **CPU scoring dominates prefill at Sq=1024.** Even at S=8 the scoring pass
  (579 ms) exceeds the whole NPU block-sparse compute (475 ms); at S=1 it's 7×.
- **matmul is 94–97 % of scoring** at every stride — the gemm is the cost.
- Stride scales ≈ 1/S (matmul 3018→1741→967→544 for S=1/2/4/8), slightly worse
  than ideal because the per-layer K de-transpose + per-call overhead are fixed.
- **Sq=256 is degenerate** for this kTopK=8: only 8 blocks, nothing to select. A
  meaningful length axis needs Sq with num_qb > kTopK (Sq≥512 → num_qb≥16). Note
  also that runtime prompt length does NOT change cost — `prefill` always
  processes the full compiled Sq window (num_qb_=Sq/kBQ, padded), so the length
  axis must come from recompiling at different `--sq`, not from shorter prompts.

### Per-qb stage profiling: select vs gather vs NPU (2026-05-20) — pipeline is a dead end

`MLLM_QB_PROFILE=1` times each serial qb stage: sel (build_mask+selectTopKBlocks),
gather+stage (K/V/Q), npu (dispatch_attn). Per-stage totals across all 28 layers
(Sq=1024, 4k prompt):

| stage | S=1 | S=8 | per-qb steady (S=8) | scales with |
|-------|-----|-----|---------------------|-------------|
| select (mask+scoring) | 3246 ms | 585 ms | ~600 µs | qb (hist) & 1/S |
| gather+stage | 72 ms | 59 ms | ~45 µs | constant |
| NPU attn dispatch | 152 ms | 140 ms | ~125 µs | constant |
| qb-loop sum | 3470 ms | 784 ms | | |

Plus ~312 ms of chunk dispatches (stride-independent: 1096−784 at S=8, 3785−3470
at S=1). Full prefill share at **S=8: select 53%, chunks 28%, NPU-attn 13%,
gather 5%**; at **S=1: select 86%**.

**The per-qb pipeline cannot help.** The only NPU work overlappable with select
is the attn dispatch (~125 µs/qb) — 5× smaller than select at S=8, ~50× at S=1.
Even perfectly overlapping ALL NPU (attn 140 + chunks 312 = 452 ms ≈ the 475 ms
no-score baseline) against ALL select (585 ms) floors prefill at ~585 ms. So
scoring must get cheaper; overlapping it can't (matches the earlier pipeline
experiments that only helped marginally at S=1).

Two shaping findings:
- **One-time per-layer spike at qb8** (~5–6 ms, S-independent → NOT the matmul):
  `kc_layer_` 64 MB re-alloc + first-touch + de-transpose + first-op construction.
  ~150 ms across 28 layers ≈ 25% of S=8 select. Reusing the buffer instead of
  re-allocating per layer is a cheap orthogonal win.
- **NPU attn is only ~125 µs/qb**, so moving the scoring matmul to NPU (approach 2)
  pays only if batched to one dispatch/layer — 672 tiny per-qb dispatches would
  drown in dispatch overhead. This makes approaches 1 (big-M per layer) and 2
  (NPU) converge: consolidate to one matmul/layer first, then pick CPU vs NPU.

### Approach 1 (big-M per-layer matmul) — implemented, CORRECT, but a NET LOSS (2026-05-20)

Implemented behind `MLLM_BLOCKSEL_BIGM`: build the reduced full-sequence Q
(`qr_layer_` [1,Hq,Lr,SD], same antidiagonal reversal as the per-qb Q) once per
layer and do ONE matmul `qr_layer_·kc_layerᵀ → logits_full_ [1,Hq,Lr,Lr]`
(M=Lr=128); each qb then just slices+softmax+pools `logits_full_` (no matmul).
Verified correct: retrieves 8090293 at S=8, identical to the per-qb path.

But it is **slower**, not faster:

| S=8 | per-qb (M=4) | big-M (M=128) |
|-----|--------------|---------------|
| matmul | 544 ms | **1059 ms** |
| prefill (818 tok) | 1.10 s | 1.74 s |

**Why — the ARM fp32 gemm is throughput-bound at ~13–14 GFLOP/s regardless of M:**
- per-qb: ~6.8 GFLOP / 544 ms = 12.5 GFLOP/s (M=4)
- big-M:  ~15 GFLOP / 1059 ms = 14.2 GFLOP/s (M=128)

So the "small M is inefficient / per-call overhead dominates" hypothesis is
**wrong**. Bigger M does not raise throughput; big-M just computes 2.2× the FLOPs
(the full Lr×Lr square, including the masked upper triangle and the qb<8 rows we
discard) at the same rate → ~2× slower. The arm `__mllm_blas_sgemm_nt_t` fp32
kernel itself is the ceiling (likely bandwidth-bound / not using NEON+dotprod for
fp32). The big-M code stays env-gated; default remains the faster per-qb path.

**Implication:** CPU matmul consolidation can't win. The levers are now (a) move
this matmul to a faster compute unit — the **NPU/HTP** (approach 2; the big-M
single-matmul-per-layer structure is exactly the right granularity: one dispatch
per layer, not 672 tiny ones), or (b) a genuinely faster CPU fp32 kernel
(KleidiAI fp32 microkernel / hand-tuned NEON, vs the current ~14 GFLOP/s), or
(c) cut FLOPs (higher S, score fewer layers/qbs and reuse the selection).

### Approach 2 (NPU offload of the scoring matmul) — benchmarked BOTH granularities (2026-05-20)

The earlier "CPU and NPU on par at ~2.2 ms" verdict is void: that "CPU" was the
no-op kBLAS / x86-host BLAS, not the real 544 ms device cost. With the real CPU
numbers in hand, the HTP `GemmLatencyTest.BlockSelScore*` cases (MatMul+Softmax,
fp16, qti.aisw, inputs already on QNN) settle it. Per layer at Sq=1024 (Lr=128):

| scoring matmul / layer | per-qb | big-batch |
|------------------------|--------|-----------|
| CPU (arm fp32)         | 19.4 ms | 37.8 ms |
| **NPU (HTP fp16)**     | 8.5 ms (31 dispatches, 30 GFLOP/s) | **0.32 ms (1 dispatch, 1672 GFLOP/s)** |

NPU big-batch HTP sweep: L1024 0.32 ms (1672 GFLOP/s), L2048 0.76 ms (2823),
L4096 2.25 ms (3810). NPU per-qb: L1024 8.5 ms/layer, L2048 22.7 ms/layer.

Findings:
- **NPU big-batch is the lever.** 0.32 ms/layer → ~9 ms for all 28 layers vs
  544 ms on CPU (~60×). The 2.2× extra FLOPs of the full Lr×Lr grid are free at
  1672 GFLOP/s.
- **Granularity flips by device.** On CPU per-qb wins (fp32 kernel is
  throughput-bound at ~14 GFLOP/s, so fewer FLOPs win). On NPU big-batch wins by
  ~27× (per-qb is dispatch-bound at 30 GFLOP/s; one big launch is compute-bound).
- Even NPU per-qb (8.5 ms/layer × 28 = 239 ms) beats CPU (544 ms), but big-batch
  is far better.

**Integration design (next):** the 0.32 ms is matmul+softmax only, inputs already
fp16 on QNN. Real integration must (a) get `qr_layer_`/`kc_layer_` into the
[Hq,Lr,SD] fp16 reduced layout on HTP — K already lives in the on-HTP KV cache,
Q is a chunk output, so the reduce/cast could be an HTP graph rather than a CPU
build+upload; (b) handle the [Hq,Lr,Lr] logits — either pool on NPU (reshape+
reduce → tiny [Hq,Nb,Nb] scores, minimal readback) or read back fp16 (512 KB/
layer) and pool on CPU; (c) either add the scoring graph to the AOT context
(recompile) or run a separate runtime QNN graph between chunks. With the matmul
at ~9 ms, the new bottleneck becomes this data prep + readback + the CPU
softmax/pool (if not moved to NPU).

### Approach 2 — AOT-bake plan (chosen 2026-05-20)

Runtime separate-graph rejected: the binary context is immutable (need a 2nd
context) and QNNAllocator mem-handles are context-bound (QNN_TENSORMEMTYPE_MEMHANDLE,
memRegister(context_)), so a 2nd-context graph needs an allocator-rebind dance +
risks the V79 PD cap. AOT-bake sidesteps all of it — the scoring graph shares the
model context + allocator like the existing 2L+1 graphs.

Design (locked):
- **One shared "score" graph** (no per-layer weights), executed L times with
  per-layer qr/kc inputs. Compiled at FIXED stride S=8 → qr,kc fp16 [Hq,Lr,SD]
  with Lr=Sq/8 (128 at Sq=1024), SD=8·128=1024; output logits fp16 [Hq,Lr,Lr].
  (S=8 keeps readback at 512 KB/layer; S=1 would be 32 MB/layer.)
- **Matmul-only on NPU** (option A): graph = `matmul(qr, kc, transpose_b)`. CPU
  does the block-causal history-only softmax + block-pool (the per-layer `temp`
  scale stays on CPU, avoiding baking per-layer constants into the graph). NPU
  softmax+mask is a later optimization.
- Expected per-prefill cost: CPU qr/kc fp16 build (~36 ms) + NPU matmul (~9 ms)
  + readback (~14 MB) + CPU softmax+pool (~54 ms) ≈ ~105 ms vs 576 ms CPU (~5.5×);
  projected prefill ~620 ms (≈ the 475 ms no-score baseline + scoring).

Implementation steps:
1. modeling header: add `ScoreModule` (forward = matmul transpose_b) + trace one
   "score" graph with inputs score_qr/score_kc. (DONE/IN PROGRESS)
2. compile driver: add fp16 trace_inputs score_qr/score_kc [Hq,Lr,SD], add "score"
   to the lowered-graph list, bump expected graph count to 2L+2. Recompile at
   --sq 1024 (S=8 baked).
3. runtime (ShaBlockSparsePromptProcessorSplit, env MLLM_BLOCKSEL_NPU): build
   qr/kc as fp16 kQNN tensors, `graphExecute("score", {qr,kc}, {logits})`, read
   back, CPU softmax+pool. Reuse the existing logits_full_ softmax+pool indexing.
4. Risks to verify: (a) AOT pipeline lowers a bare fp16 matmul (no QDQ) — boundary
   tensors are already fp16 so likely OK; (b) PD memory with +1 graph; (c) fp16
   logit precision for ranking.

### Approach 2 — AOT-bake implemented; BLOCKED at load on a QNN PD size-calc error (2026-05-20)

Implemented end-to-end:
- modeling: `ScoreModule` (non-transposed `matmul(qr, kc)`; the AOT MatMul pattern
  ignores transpose flags, so kc is fed PRE-TRANSPOSED [Hq,SD,Lr]) + one shared
  "score" graph traced in the driver.
- compile driver: fp16 trace_inputs score_qr [Hq,Lr,SD] / score_kc [Hq,SD,Lr],
  "score" added to the lowered-graph list, FIXED stride S=8. **The fp16
  non-transposed matmul lowers cleanly** (validateOpConfig OK, precision FP16) —
  the fp16-lowering risk is cleared. Full 58-graph bin builds (1.62 GB).
- runtime (env MLLM_BLOCKSEL_NPU): allocates fp16 kQNN score_qr/score_kc/
  score_logits, builds them per layer (qr reduced+centered; kc reduced+centered
  +TRANSPOSED), dispatches the "score" QnnAOTModule, converts fp16 logits→fp32
  logits_full_, reuses the big-M CPU softmax+pool. Forces S=8.

**BLOCKER:** the 58-graph bin fails at `initQnnBackend` (context-from-binary):
```
Failed to find available PD ... context size estimate 3659319552 (3.66 GB)
Mismatch in the PD reserved space for context 1
Size Calculation encounter error! Doing Hard reset of reserved mem to 0.
err 1002 / 0x3ea
```
Identical estimate (3.66 GB) regardless of MLLM_QNN_SPILLFILL_MB (64/128/256/512)
or with sharing OFF. Refined diagnosis after A/B with the 57-graph bin: the PD
error is PARTLY TRANSIENT — the cDSP PD reservation lingers after a crashed run,
so `pkill -9 + sleep 8` is required between load attempts or PD-exhaustion masks
the real result (many false failures this session came from rapid crash-retry).
With PD freed, the 57-graph bin (estimate 3.64 GB) LOADS, but the 58-graph bin
(3.66 GB) still FAILS — so the score graph's ~20 MB tips an already-at-ceiling
estimate over the V79 PD cap AND adds a structural "Mismatch in PD reserved space
/ Size Calculation encounter error" not seen for the 57-graph bin. Likely: a
weightless all-fp16 graph mixed into a quantized-graph context records an
inconsistent PD/spill-fill reservation.

New bin preserved at qwen3-lpbq-sha-blocksparse-causal-split.score.bin.
Next-session candidates: (a) inspect the score graph's per-graph VTCM/spill-fill
in the compile prep log (vs the chunk graphs); (b) try compiling "score" inside
the shared-spill-fill group / matching the chunks' memory profile; (c) shrink the
score graph (higher S → smaller Lr, or split heads); (d) fall back to the runtime
2nd-context path (allocator-rebind) which keeps the score graph out of the model
context entirely.

### Approach 2 — PD blocker ROOT-CAUSED (2026-05-20, option-1 inspection)

Per-graph spill-fill from the compile-prep log (58 graphs, VTCM = 8 MB):
- 29 graphs (attn + small): 0 MB spill-fill
- 28 graphs (chunks, M=Sq): 27.6–27.8 MB
- **1 graph (score): 43.8 MB — the LARGEST in the whole context.**

The score matmul's working set (qr 4 MB + kc 4 MB + out 0.5 MB ≈ 8.5 MB) just
overflows the 8 MB VTCM → 43.8 MB DDR spill. The shared spill-fill buffer is
sized to the max graph, so the score graph raises it 27.8 → 43.8 MB (+16 MB);
that + its I/O tips the PD estimate 3.64 → 3.66 GB, over the V79 cap (and trips
the "Size Calculation" mismatch). Net delta vs the 57-graph bin (~20 MB) matches.

**Fix:** keep the score working set under 8 MB VTCM so it spills ~0 (like attn).
Cleanest = compile score at **Hq=8** (working set ≈ 4.25 MB < 8 MB), dispatch
twice/layer (heads 0–7, 8–15). Alternatives: higher S (smaller Lr) or split the
matmul. Then the score graph's PD contribution ≈ 0 and the bin should load.

### Approach 2 — Hq-split tested; PD blocker is the SATURATED model context, not the score graph (2026-05-20)

Correction to the previous section: the score graph has **0 MB spill-fill** in
BOTH the Hq=16 and Hq=8 compiles (it's the last entry in the per-graph spill-fill
summary = 0.0 MB; it always fit the 8 MB VTCM). The 43.8 MB spill-fill belongs to
a CHUNK (the lm_head/final chunk), not score. So the score graph's PD cost is its
**I/O buffers** (qr+kc+logits): ~8.5 MB at Hq=16, ~4.25 MB at Hq=8.

Hq=8 recompile + device test: PD estimate dropped only 3.659 → 3.655 GB (the
~4.5 MB I/O saving), and it STILL fails to load — same "Failed to find available
PD / Mismatch in PD reserved space / Size Calculation encounter error / err
1002". The 57-graph bin (3.640 GB) loads; the 58-graph bin (3.655 GB) doesn't.

**Real root cause: the split-prefill model context already saturates the V79 PD
(~3.64 GB used of ~3.65 GB available — essentially zero headroom).** Adding ANY
graph tips it over, and the weightless-fp16 score graph in a quantized context
also trips a structural PD size-calc error. Shrinking the score graph (Hq, S)
only nudges I/O by a few MB — not enough, and doesn't fix the structural error.

Conclusion: NPU score offload baked into the SAME context is **PD-blocked on this
device at Sq=1024**. Viable paths: (a) free model PD — e.g. the 43.8 MB lm_head
chunk spill-fill, or a smaller Sq; (b) runtime 2nd context for score (own PD
reservation, but total PD is tight so may also not fit); (c) keep the working CPU
kMllmBlas scorer (544 ms/prefill at S=8) — the NPU matmul is 60× faster in
isolation but undeployable here without PD headroom. NPU code stays behind
MLLM_BLOCKSEL_NPU; the Hq=8 score bin is preserved as ...split.score.bin.

### Approach 2 — SOLVED via runtime 2nd context + shared spill-fill group (2026-05-20) ✅

Instead of baking the score graph into the PD-saturated model context, build it at
RUNTIME in a SECOND HTP context that joins the model context's spill-fill group:
- `QNNBackend::beginAuxContext(mb)` creates a 2nd context with
  `QnnHtpContext_GroupRegistration_t{firstGroupHandle = model context_, maxSpillFillBuffer}`
  (REGISTER_MULTI_CONTEXTS) so it SHARES the model's spill-fill buffer (the
  documented multi-context pattern; ONNX Runtime QNN EP does the same). While
  open, context_/allocator point at the aux context so the normal
  createQnnGraph/addTensor/graphAddNode/graphFinalize + kQNN allocs land in (and
  memRegister against) it; `endAuxContext()` restores the model context.
- The score graph is built live like GemmLatencyTest: `MatMul(score_qr, score_kc)`
  with `transpose_in1=true` (runtime path honors transpose → kc fed NATURAL
  [Hh,Lr,SD], no pre-transpose). No AOT recompile — uses the existing 57-graph bin.
- **No PD blowup**: the model context loads at its usual 3.64 GB; the aux context
  adds only its tiny IO (shares the spill-fill group), so it fits.

**fp16 overflow fix (critical):** the raw logits qr·kc are ~1e7 (qr,kc ~O(100),
summed over SD=1024) — far over fp16's 65504 max → inf → garbage selection. Fix:
pre-scale BOTH qr and kc by sqrt(temp) in the fp16 build (temp = q_scale·k_scale/
(√D·S)), so the matmul output is the properly-scaled attention logit (O(1-10));
the CPU softmax then uses temp=1. Without this, retrieval emitted "** 2 **"; with
it, retrieval is correct.

**Result (device, SM8750/V79, 818-tok NIAH, S=8, MLLM_BLOCKSEL_NPU=1):**

| | CPU kMllmBlas | NPU 2nd-context |
|---|---|---|
| scoring matmul | 544 ms | **17 ms** |
| kc de-transpose (CPU build) | ~36 ms | ~33 ms |
| softmax+pool (CPU) | ~54 ms | ~20 ms |
| scoring total | 576 ms | **70 ms** (8×) |
| **prefill (818 tok)** | 1097 ms | **573 ms — 1427 tok/s (1.9×)** |

Retrieves 8090293 correctly. Prefill is now ~100 ms over the no-score baseline
(475 ms). The remaining scoring cost is the CPU kc/qr fp16 build (~33 ms) +
softmax+pool (~20 ms); the NPU matmul itself is only 17 ms. Score graph is built
at Hq/2=8 heads × 2 dispatches/layer (kept small; could be 1 dispatch since the
aux context shares spill-fill). Enabled by env MLLM_BLOCKSEL_NPU (needs
MLLM_QNN_SPILLFILL_MB set so the model context registers a group to join).

### Prefill benchmark: NPU-score vs CPU-score vs dense (2026-05-20, SM8750/V79, Sq=1024)

4k prompt filling the window, --gen 0. NPU scoring locked to S=8 (its graph is
built for that stride); stride sweep is the CPU scorer. (pkill + ~12 s wait
between runs — PD lingers; "available PD" failures are transient.)

| config | prefill | tok/s | scoring total | matmul | vs dense |
|--------|---------|-------|---------------|--------|----------|
| dense full-attn (ref, doc, ~1000 tok) | ~850 ms | ~1100 | — | — | 1.0× |
| block-sparse no-score | **478 ms** | 2142 | — | — | 1.8× faster |
| block-sparse NPU-score S=8 | **~570–760 ms** | 1350–1800 | 70–194 ms | **17–24 ms** | **~1.3× faster** |
| block-sparse CPU-score S=8 | 1084 ms | 944 | 571 ms | 534 ms | ~1.3× slower |
| block-sparse CPU-score S=4 | ~1525 ms | ~670 | ~1007 ms | ~967 ms | slower |
| block-sparse CPU-score S=2 | 2320 ms | 441 | 1794 ms | 1733 ms | slower |
| block-sparse CPU-score S=1 | 3895 ms | 263 | 3347 ms | 3114 ms | slower |

**Headline:** with NPU scoring, block-sparse + XAttention selection is ~1.3×
faster than dense WHILE retrieving correctly (8090293); with the CPU scorer it's
slower than dense at every stride. The NPU offload is what makes score-based
selection worthwhile. NPU matmul is a stable ~20 ms; the 570–760 ms variance is
the CPU-side qr/kc fp16 de-transpose (33–193 ms) + softmax/pool (20–83 ms) — the
next optimization target. Length axis = one point (cost is fixed per compiled Sq);
doc's split no-score points: Sq256≈147 ms, Sq512≈256 ms, Sq1024≈530 ms.

KNOWN cosmetic issue: aux context isn't freed at shutdown → "free device failed:
context still associated" at exit (harmless; fix = free aux_context_ in dtor).

### CPU-side scorer tuning: de-transpose & softmax (2026-05-20)

Looked into the two CPU costs in the NPU-score path (matmul itself is ~17 ms NPU).
- **kc de-transpose (~22 ms warm) is NOT the bottleneck.** An A/B (sequential-
  read loop order vs naive d-innermost) showed no difference — both ~22 ms warm,
  ~80 ms on the cold first prefill. The 33–193 ms "variance" seen earlier was
  cold-start / CPU-frequency ramp (first prefill after launch), not cache access.
  Kept the simple loop.
- **Softmax+pool: fused + skip-max → ~30 → ~20 ms warm.** Fused the exp and the
  block-pool into one pass over histr (no prob[] scratch: per kb-block, exp+sum
  straight into the block partial-sum + row total; then scores += bsum/total),
  and skipped the max-subtraction for the NPU path (pre-scaled logits are O(1-10)
  → no float-exp overflow). Retrieval unchanged (8090293).

Warm steady-state (single prefill, Sq=1024, S=8): kc ~22 ms, matmul ~17 ms,
softmax+pool ~20 ms → **scoring ~59 ms, prefill ~569 ms (1799 tok/s)** — ~90 ms
over the 478 ms no-score floor, ~1.5× faster than dense. (Decode-by-reprefill
inflates the first prefill via cold-start; warm prefills are ~569–600 ms.)
