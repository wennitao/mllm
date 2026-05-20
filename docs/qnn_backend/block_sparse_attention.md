# Block-Sparse Attention on QNN: Decomposition, Profiling, and Optimisation

This doc covers when and how to run attention as a **decomposed QNN graph**
(MatMul + Mul + Softmax + MatMul) rather than as a custom HVX op, and how
**block-sparse attention** extends that decomposed approach past the
O(Sq²) intermediate wall.

It is a deep dive on the second half of §2 of
[custom_hvx_op_skill.md](custom_hvx_op_skill.md) — see that doc for the
high-level "should I write a custom op?" framing and the custom-op
implementation recipe. Everything here assumes you've already concluded
that decomposition is on the table and you want to understand the
sparse path's measured behaviour, profiling breakdowns, and tuning
levers.

All numbers below are for Qwen3-style fp16 attention (Hq=16, Hkv=8,
D=128, BQ=BK=32, 1/4 sparsity) on V79 HTP, end-to-end measured.

---

## Extending decomposition with block-sparse attention

If the model tolerates approximate attention (top-k blocks per q-block),
block-sparse extends the decomposed-graph approach past the O(Sq²)
intermediate wall. The CPU selects top-k K/V blocks per q-block and
gathers them into a per-q-block tile; the NPU runs the same
`MatMul → Mul → Softmax → MatMul` decomposition on the gathered tile.
The materialised intermediate is now `[Hq, Sq, top_k·BK]` — fixed by the
sparsity ratio, not by Sq alone.

**Three ways to express the sparse computation, measured at Qwen3 fp16
attention (Hq=16, BQ=BK=32, D=128, 1/4 sparsity):**

All numbers ms/layer, end-to-end measured. For sparse rows the
breakdown is `gather (CPU) + NPU (DSP) = total`. For per-qb pipelined
the gather runs concurrently with the NPU dispatch so only the total
is meaningful; the row underneath shows the components that overlap.

| Sq | Dense<br>NPU | Rank-4 sparse<br>gather + NPU = **total** | Big-batch r3 sparse<br>gather + NPU = **total** | Per-qb pipelined r3<br>**total** (overlapped) |
|---:|---:|---:|---:|---:|
| 128 | **0.18** | 0.15 + 0.24 = **0.39** | 0.15 + 0.21 = **0.36** | **0.77** |
| 256 | **0.38** | 0.35 + 0.46 = **0.81** | 0.67 + 0.39 = **1.06** | **1.36** |
| 512 | **1.83** | 1.75 + 1.22 = **2.97** | 1.75 + 1.04 = **2.79** | **3.63** |
| 1024 | 10.61 | 4.84 + 7.06 = **11.90** | 4.86 + 5.57 = **10.43** | **7.58** |
| 2048 | 86.47 | 11.64 + 40.39 = **52.03 *(OOM-borderline)*** | 8.96 + 22.41 = **31.37** | **22.82** |

Rank-4 and big-batch rank-3 use the same `K_arr [Hq·num_qb, top_k·BK,
D]` bytes and OOM at the same point at Sq>2048; per-qb's per-dispatch
K_qb is 2 MB and doesn't.

**Sq < 1024: stay with dense.** Block-sparse loses to dense decomposed
because the gather cost and per-dispatch overhead aren't amortised across
enough compute. Sq=1024 is the crossover; per-qb pipelined sparse is
1.4× faster than dense there, and 3.8× faster at Sq=2048.

**The naive rank-4 sparse graph is slower than it should be.** Folding
the `num_q_blocks` axis into a leading batch dim means QNN's MatMul gives
you `[Hq, num_qb, BQ, D] × [Hq, num_qb, top_k·BK, D]ᵀ`. QNN appears to
schedule the leading batch dim serially, not in parallel with the inner
Hq batch. At the same total bytes and same FLOPs the rank-3 form
(`[Hq · num_qb, BQ, D]`) is 1.2–1.9× faster — purely from QNN's MatMul
scheduling, not from any data-size difference:

| Sq | Rank-3 (same data) | Rank-4 | Penalty |
|---:|---:|---:|---:|
| 128 | 0.20 | 0.24 | 1.21× |
| 512 | 0.90 | 1.22 | 1.36× |
| 1024 | 4.02 | 7.06 | 1.76× |
| 2048 | 21.6 | 40.4 | 1.87× |

**Two ways to recover the rank-4 loss:**

1. **Big-batch rank-3.** Flatten the leading batch dim: reshape inputs
   from `[Hq, num_qb, …]` to `[Hq · num_qb, …]`. Same bytes, same
   FLOPs, same single dispatch per layer — only the QNN shape header
   changes. QNN's MatMul scheduler treats the resulting `Hq · num_qb`
   (up to 1024 at Sq=2048) as one batch dim it can freely schedule on
   HMX, avoiding the rank-4 outer-batch serialisation. ~5 lines of
   change from the rank-4 graph; no worker thread or pipelining.
   Beats rank-4 everywhere, beats per-qb pipelined at Sq ≤ 512, loses
   to per-qb at Sq ≥ 1024 by a few ms/layer (gather can't be hidden).
   Same OOM ceiling as rank-4.
2. **Per-qb dispatched, pipelined.** num_qb separate rank-3 dispatches
   of `[Hq, BQ, D]` × `[Hq, top_k·BK, D]`, with CPU gather of q-block
   i+1 running concurrently with NPU compute of q-block i, via a
   persistent worker thread. Per-dispatch overhead is ~43 µs (so
   num_q_blocks × 43 µs of fixed cost), and `std::async`-per-iter adds
   ~80 µs/qb in thread-spawn — use a persistent worker, not async.
   Pays off at Sq ≥ 1024 when gather is meaningful; at Sq=2048 it
   recovers ~13 ms/layer of overlapped CPU work. Bonus: per-dispatch
   K_qb is only 2 MB (vs rank-4/big-batch K_arr's 128 MB at Sq=2048),
   so the OOM ceiling rises.

**Suggested tiering:**

```
Sq < 1024:   decomposed dense
Sq ≥ 1024:   per-qb pipelined sparse (1.4× faster than dense at Sq=1024,
              3.8× at Sq=2048; also dodges the rank-4/big-batch K_arr
              OOM ceiling at Sq>2048)
```

Big-batch rank-3 is always faster than rank-4 sparse (drop-in fix for
the rank-4 scheduling penalty) but never beats either dense (at small Sq)
or per-qb pipelined (at large Sq) in measured data. Consider it as a
stepping stone — ship big-batch first as the cheap fix to rank-4, then
add the per-qb pipeline if you need the further win at long context.

## Causal masking: intra-block triangular on the diagonal tile

The numbers above assume every q-block can attend to every k-block.
For prefill, q-block i can only attend to k-blocks 0..i, and within the
diagonal k-block (k-block i itself) only the lower-triangular part
(k ≤ q within the block) is valid. Two changes from the non-causal path:

1. **Causal selection.** For q-block i, the top_k slots must come from
   [0, i]. The diagonal k-block i is forced into a fixed slot (here:
   slot top_k-1) so the mask is structurally known. The other top_k-1
   slots draw from [0, i-1]. For early q-blocks where i+1 < top_k,
   slots without a valid historical block are padded with k-block 0 and
   masked out fully.

2. **Per-q-block mask** injected via `ElementWiseAdd(QKs, mask)` between
   scale and softmax. Three regions per (i, q, slot):
   - **Diagonal slot (last BK columns of the score tensor):** intra-block
     triangular — 0 if `col_within_blk ≤ q`, else -1e4.
   - **Historical slots:** 0 (no masking; selected block index < i).
   - **Padding slots (only for q-block i < top_k-1):** -1e4 across all BK
     columns of the slot.

`-1e4` is the -inf surrogate so a fully-masked row doesn't NaN through
softmax (matches `FlashAttentionDecomposedTest`'s convention).

**Mask shapes per method:**

| Method | Mask shape | Broadcast over Hq? | Bytes (Sq=1024, top_k=8) |
|---|---|---|---:|
| rank-4 | `[1, num_qb, BQ, top_k·BK]` | yes (leading 1) | 512 KB |
| big-batch r3 | `[Hq·num_qb, BQ, top_k·BK]` | no — tiled Hq times | 8 MB |
| per-qb pipelined r3 | `[1, BQ, top_k·BK]` (APP_WRITE) | n/a — one qb per dispatch | 16 KB |

Per-qb gets the smallest mask: each dispatch sees only its own q-block,
so the mask is just `[BQ, top_k·BK]` written by CPU prep before the
dispatch. The trade-off is that the mask is dynamic, not static — it
ships in the input bind.

Big-batch's mask is the worst: flattening Hq into the leading batch dim
removes the broadcast hook the rank-4 form gets for free. A
reshape-bracketed pattern (`[Hq·num_qb, ...] → reshape → [Hq, num_qb, ...]
+ mask[1, num_qb, ...] → reshape back`) keeps the mask small in theory,
but the QNN PD silently fails graphFinalize ~1/3 of runs with that
pattern on V79. The plain tiled mask is stable; the cost is 16× more
mask memory.

### Measured causal numbers

End-to-end ms/dispatch, Qwen3 fp16 (Hq=16, Hkv=8, D=128, BQ=BK=32),
selection seed fixed across methods. CPU gather + mask build single
thread (no OMP). NPU-side timing only — gather is comparable to the
non-causal numbers above.

| Sq | top_k | Dense causal NPU | Rank-4 causal | Big-batch r3 causal | Per-qb r3 (sync) |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 | **0.21** | 0.38 | 0.26 | 0.65 |
| 256 | 2 | **0.43** | 0.73 | 0.48 | 1.22 |
| 512 | 4 | **2.08** | 2.90 | 1.20 | 3.10 |
| 1024 | 8 | 13.72 | 12.15 | **6.44–7.39** | 9.08 |
| 2048 | 8 | OOM | 27.92 | 11.96–13.31 | **19.02** |

Two ways causal differs from non-causal:

1. **Sq=1024 is still the crossover.** Sparse beats dense by ~1.1×
   (rank-4) to ~1.9× (big-batch when it executes). The same Sq < 1024
   "stay with dense" rule holds; the causal mask doesn't shift the
   crossover.

2. **Dense causal hits the same O(Sq²) wall.** At Sq=2048 the dense
   graph materialises a 128 MB `[Hq=16, Sq=2048, Skv=2048]` fp16 score
   tensor ×4 native copies (QK, QKs, QKm, P) and silently OOMs the QNN
   PD at graphFinalize — same failure mode as non-causal dense at
   Sq>2048. The mask is one more 4 MB static tensor; it isn't the
   trigger. **Block-sparse remains the only viable path at Sq ≥ 2048
   regardless of causality.**

Rank-4 and big-batch causal also hit the same K_arr OOM ceiling as the
non-causal path. We dropped to top_k=8 (vs the doc's standard top_k=16
at Sq=2048) for the causal sweep to stay stable; at top_k=16 the
sparse paths flake ~2/3 of runs. Per-qb pipelined's 2 MB per-dispatch
K_qb is unaffected — same OOM advantage as in the non-causal table.

### Practical caveats observed on V79

- **Big-batch causal is intermittent.** Same code, same data, ~1/3 to
  2/3 of fresh-process runs silently fail graphFinalize and emit zeros
  at execute time (0.02 ms dispatch, err ≈ 1.0). When it executes,
  output matches the host reference exactly (miss=0). The flake
  correlates with graph complexity, not with Sq directly — Sq=128
  (1 MB tiled mask) and Sq=2048 (16 MB) both flake at comparable rates.
  Re-running typically gets a passing run. Worth confirming this
  reproduces on the target device before relying on big-batch causal
  in production.
- **Per-qb sync vs pipelined needs a multi-layer harness to compare
  fairly.** The single-sweep test shows pipelined ~2× *slower* than
  sync at every Sq — cv-worker startup overhead dominates when
  amortised over only `num_qb` dispatches. The doc's 1.4× pipelined
  speedup (non-causal) uses a 28-layer harness (28 × `num_qb`
  dispatches) where the cv overhead amortises. Treat the **sync**
  column above as the per-qb method's representative per-layer cost;
  in a real-model run (Qwen3 has 28 layers), pipelined should recover
  the published gains. A causal multi-layer harness isn't measured
  here yet.
- **The mask add itself is cheap.** ElementWiseAdd on the score tensor
  is the same shape as scale_qk and softmax, so it shrinks
  proportionally with top_k just like they do. Per-op cost is ~0.05–
  0.5 ms over non-causal at the same Sq.

Reference test: [`tests/qnn/BlockSparseAttentionCausalTest.cpp`](../../tests/qnn/BlockSparseAttentionCausalTest.cpp).
Build with `Mllm-Test-QNN-BlockSparseCausal`. Each method has a
TEST_F per Sq (e.g., `Sq1024_TopK8`, `BigBatch_Sq1024_TopK8`,
`PerQb_Sq1024_TopK8`); run each in a fresh process via
`--gtest_filter` to avoid the multi-graph state flake.

## Per-op profile: what actually dominates inside the graph

Set `MLLM_QNN_PROFILE=DETAILED` to dump a `qnn_profile.csv` with per-op
cycle counts. At Sq=512 on the steady-state warmed dispatch (totals
9.95 M cycles dense / 7.69 M cycles big-batch sparse):

| Op | Dense cycles | Dense % | Sparse cycles | Sparse % | Note |
|---|---:|---:|---:|---:|---|
| Input | 0 | 0% | 48,824 | 0.6% | metadata |
| `reshape_q` | — | — | **0** | **0%** | rank conversion is free at kernel level |
| `matmul_qk` | 2,797,104 | 28.1% | 3,612,240 | 47.0% | HMX |
| `scale_qk` (ElementWiseMul) | **2,276,280** | **22.9%** | 865,080 | 11.2% | HVX — streams full score tensor |
| `softmax` | **4,114,430** | **41.3%** | 1,497,126 | 19.5% | HVX |
| `matmul_av` | 576,014 | 5.8% | **1,259,297** | **16.4%** | HMX (per-MAC efficiency drops for small M) |
| `reshape_o` | — | — | **0** | **0%** | rank conversion is free at kernel level |
| Output | 191,011 | 1.9% | 412,542 | 5.4% | result writeback |
| **Total** | **9,954,839** | | **7,695,109** | | sparse 23% fewer cycles overall |

Three things this clarifies:

1. **Softmax + scale are ~65 % of dense cycles, not the matmuls.** Both
   ops run on HVX, which has ~30–50× less arithmetic throughput than HMX
   per FLOP. Even though the QK matmul has 250× more arithmetic, HMX
   chews through it faster than HVX gets through one elementwise pass
   over the same `[Hq, Sq, Skv]` tensor. **The sparsity win comes
   primarily from shrinking the score tensor that softmax + scale
   operate on, not from skipping matmul FLOPs.** With 1/4 sparsity the
   `[Hq, Sq, top_k·BK]` intermediate is 4× smaller and the HVX kernels
   shrink proportionally.

2. **Reshape ops cost 0 cycles on the DSP.** QNN folds them into
   surrounding ops at graph-prepare time. The rank-3 ↔ rank-4 conversion
   we keep talking about is purely metadata — every cycle of the rank-4
   vs rank-3 gap measured above is in QNN's matmul scheduler, not in
   any reshape kernel.

3. **`matmul_av` is *slower per MAC* in big-batch sparse than in dense.**
   At Sq=512: dense `matmul_av` does 540 M MACs in 0.58 M cycles
   (939 MAC/cycle); sparse does 135 M MACs in 1.26 M cycles
   (107 MAC/cycle) — an **8.8× drop in per-MAC efficiency**. The
   inner matmul shape is `[256 batches] × (BQ=32, top_k·BK=128) ·
   (top_k·BK=128, D=128)`. The `M=32` dim is the minimum HMX tile size,
   so each batch is one HMX tile worth of output with no cross-tile data
   reuse on the Q side. Dense's `M=Sq=512` lets HMX reuse Q-row loads
   across multiple output tiles. This is the deeper "why" of QNN's
   rank-4 vs rank-3 penalty showing up *per kernel* in the profile.

So the matmul cycles increase by a factor in sparse but the softmax /
scale cycles drop by a bigger factor. Net: sparse is faster, but not by
the 4× that the FLOPs ratio would predict. At Sq=512 the measured speedup
is 1.78× (1.78 ms → 1.0 ms) — consistent with most of the speedup coming
from the softmax/scale shrinkage.

## Optrace: what else the profile tells us

The same `MLLM_QNN_PROFILE=DETAILED` setting also enables QNN HTP
optrace, which exposes timing-side data beyond per-op cycles:

| Optrace event (dense Sq=512, warmed) | Value | What it means |
|---|---:|---|
| `Accelerator (execute) time` | 1816 µs | DSP-side wall-clock for graph execution |
| `Accelerator (execute excluding wait) time` | 1783 µs | DSP-busy time |
| **Idle within execute** (derived) | **33 µs / 1.8 %** | the DSP wait inside an execute is negligible once warmed |
| `RPC (execute) time` | 4837 µs | host↔DSP transport + wait — the dominant CPU-side cost |
| `Time for HVX + HMX power on and acquire` (first execute) | **5786 µs** | one-time engine warmup penalty |
| `Time for HVX + HMX power on and acquire` (warmed) | 36 µs | steady-state acquire — tiny |
| `Time for initial VTCM acquire` (warmed) | 199 µs | VTCM reservation per dispatch |
| `Number of HVX threads used` | 6 | aggregate parallelism on the DSP |
| `Num times yield occured` | 0 | no preemption — the kernels ran to completion |

Per-MAC HMX efficiency derived from per-op cycles is more variable than
you'd expect. For dense Sq=512:

| Matmul | MACs | Cycles | MAC/cycle (aggregate) | Why |
|---|---:|---:|---:|---|
| `matmul_qk` (Q·Kᵀ) | 540 M | 2.80 M | **193** | inner K = D = 128 (small) |
| `matmul_av` (P·V) | 540 M | 0.58 M | **939** | inner K = Skv = 512 (large), 4.9× more efficient |

Same MAC count, both on HMX, both same M=Sq dim — but `matmul_av` runs
**4.9× faster per MAC**, simply because its inner K dim is 4× larger
(Skv=512 vs D=128). HMX prefers large K because it amortises the
per-tile setup over more MAC accumulations. This shows up symmetrically
in sparse, just at lower absolute numbers (M=32 ceiling).

Two implications you can act on:

- **The CPU-side cost is bigger than the DSP-side cost.** RPC + setup
  is ~5 ms while the actual DSP work is ~1.8 ms. For latency-critical
  dispatch patterns (per-qb pipelined attention), the
  `num_q_blocks × per-dispatch-overhead` term is mostly RPC, not
  arithmetic.
- **HMX likes large inner K.** Sparse's small-M matmuls were obvious
  losers, but the QK matmul is also leaving 5× on the table even in
  dense, because Q·Kᵀ's inner dim is the head_dim (D=128, smallish).
  Fusing scale + softmax into a matmul-epilogue would be one knob;
  going wider (D=256 head_dim) would be another.

## Per-hardware-unit utilisation via qnn-profile-viewer

The event-API data above stops at "per QNN op, total cycles". To see how
those cycles are distributed *across HMX and the six HVX threads* the
DSP exposes, you need the full optrace toolchain. Workflow:

```bash
# 1) Save the QNN context binary AND the schematic
#    (schematic is auto-generated when graphFinalize runs with
#    MLLM_QNN_PROFILE=DETAILED; lands as <graph_name>_schematic.bin
#    in the cwd).
MLLM_QNN_PROFILE=DETAILED ./Mllm-Test-QNN-BlockSparse \
    --gtest_filter='*DumpDenseContext_Sq512'

# 2) On-device: run the saved context under qnn-net-run with optrace.
qnn-net-run --backend libQnnHtp.so \
            --retrieve_context dense_sq512.bin \
            --input_list dense_sq512_inputs.txt \
            --profiling_level detailed \
            --profiling_option optrace \
            --output_dir profile_out

# 3) Render the FlatBuffer log + schematic into a Chrome Trace JSON.
qnn-profile-viewer \
    --input_log profile_out/qnn-profiling-data_0.log \
    --schematic dump_dense_dense_sq512_schematic.bin \
    --reader libQnnHtpOptraceProfilingReader.so \
    --output chrometrace.json
```

The resulting `chrometrace.json` shows per-op cycle counts on each of
the seven hardware units in the V79 HTP — 1 HMX engine + 6 HVX threads.
For the dense Sq=512 graph:

| Op | HMX | HVX-512 | HVX-513 | HVX-514 | HVX-515 | HVX-516 | HVX-517 | Sum | % HMX | HVX max/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `matmul_qk` | 163,779 | 426,190 | 460,682 | 579,607 | 489,301 | 414,487 | 439,055 | 2,973,101 | **5.5%** | 1.40× |
| `scale_qk` | 1,533 | 574,019 | 241,808 | 336,739 | 291,664 | 480,942 | 404,970 | 2,331,675 | 0.1% | 2.37× |
| `softmax` | 1,119 | 669,921 | 549,488 | 652,069 | 633,304 | 712,065 | 914,187 | 4,132,153 | 0.0% | 1.66× |
| `matmul_av` | 133,760 | 164,291 | 25,930 | 44,031 | 45,547 | 60,999 | 79,776 | 554,334 | **24.1%** | 6.34× |
| `Output` | 192,269 | 0 | 0 | 0 | 0 | 0 | 0 | 192,269 | 100% | — |

(All cycles. Sum across all 7 HW units. HMX is the V79's matrix engine;
6 HVX are 1024-bit SIMD threads. Total accelerator cycles for this
execute: ~9.8 M.)

Three findings the per-op view (alone) couldn't show:

1. **HMX is barely used.** Total HMX cycles across all ops: 492 K out of
   9.78 M total accelerator cycles → **HMX is busy for only ~5 % of the
   accelerator's wall-clock time** at Sq=512. The fast engine sits idle
   most of the dispatch. The dispatch is HVX-bound, not HMX-bound — so
   "go wider on D" or "fuse more compute into matmul" only helps
   marginally; the real lever is reducing the HVX-side work.

2. **The matmuls themselves are mostly HVX work, not HMX work.**
   `matmul_qk` does 95% of its cycles on HVX (data loading, transpose
   from `transpose_in1=true`, layout shuffling) and only 5.5% on HMX
   (the actual matrix multiply). `matmul_av` is much more HMX-heavy
   (24%) because its inner K=Skv=512 lets HMX amortise the per-tile
   setup. **The `transpose_in1=true` in `matmul_qk` is expensive** —
   pre-transposing K offline would shift work off HVX.

3. **HVX thread load is imbalanced.** Per op, the busiest HVX thread
   does 1.4×–6.3× the cycles of the least-busy one. `matmul_av` has the
   worst imbalance (HVX-512 does 164 K, HVX-513 does 26 K — 6.3×) but
   it's also the lowest-cycle op, so the absolute waste is small.
   `softmax` is the most balanced of the heavy ops, at 1.66× imbalance.
   Wall-clock per op is set by the slowest thread, so even modest
   imbalance translates directly into idle cycles on the others.

The two ops the HMX engine is actually utilised on (`matmul_av` 24%,
`Output` 100%) account for less than 8 % of total dispatch time. So
chasing HMX MAC/cycle on `matmul_qk` (where we measured 193 MAC/cycle
earlier) is a red herring — it's already a small fraction of the HMX
budget. The 940 MAC/cycle figure we saw on `matmul_av` is closer to
what HMX can sustain when not throttled by HVX-side prep.

## QHAS: utilisation %, DRAM/VTCM bandwidth, dominant-path cycles

`qnn-profile-viewer` can also emit a QHAS (QNN HTP Analysis Summary)
JSON/HTML report — a more compact view of the same data with explicit
utilisation percentages and memory-bandwidth counters. Enable via a
config file:

```bash
# config.json: { "qhas_schema": true, "qhas_json": true }
qnn-profile-viewer \
    --config qhas_config.json \
    --input_log profile_out/qnn-profiling-data_0.log \
    --schematic dump_dense_dense_sq512_schematic.bin \
    --reader libQnnHtpOptraceProfilingReader.so \
    --output dense_chrometrace.json
```

Outputs (alongside the chrometrace):

- `dense_chrometrace_qnn_htp_analysis_summary.html` — viewable in a browser
- `dense_chrometrace_qnn_htp_analysis_summary.json` — same data programmatic
- `dense_chrometrace_htp.json` — HTP-level graph (Netron-viewable)

**Per-hardware-unit utilisation & memory bandwidth (dense Sq=512):**

| HW | timeline cycles | cycles used | **util %** | DRAM rd | DRAM wr | VTCM rd | VTCM wr |
|---|---:|---:|---:|---:|---:|---:|---:|
| HMX (tid 256) | 3,204,634 | 697,152 | **21.8%** | 4.20 MB | **27.26 MB** | 48.43 MB | 14.68 MB |
| HVX-512 | 3,132,302 | 1,791,639 | **55.9%** | 0.31 MB | 0 | 11.67 MB | 11.98 MB |
| HVX-513 | 3,030,567 | 1,331,092 | 41.5% | 0.39 MB | 0 | 4.59 MB | 4.98 MB |
| HVX-514 | 3,156,960 | 1,773,415 | **55.3%** | 0.36 MB | 0 | 12.75 MB | 13.11 MB |
| HVX-515 | 3,075,694 | 1,710,694 | 53.4% | 0.26 MB | 0 | 10.19 MB | 10.45 MB |
| HVX-516 | 3,094,504 | 1,639,189 | 51.2% | 0.33 MB | 0 | 8.95 MB | 9.27 MB |
| HVX-517 | 3,101,794 | 1,433,159 | 44.7% | 0.44 MB | 0 | 6.39 MB | 6.83 MB |

(Bytes are decoded from the raw counter values in the JSON.)

**Per-QNN-op breakdown:**

| QNN op | cycles | %active | **dominant-path** % | HTP sub-ops | DRAM rd | DRAM wr | VTCM rd | VTCM wr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 4.16 M | 40.1% | 35.4% | 190 | 0 | 0 | 16.78 MB | 16.78 MB |
| MatMul (qk + av) | 3.47 M | 33.4% | **42.1%** | 738 | 6.30 MB | 0 | 42.14 MB | 37.75 MB |
| ElementWiseMul | 2.36 M | 22.7% | 19.6% | 333 | 0 | 0 | 16.78 MB | 16.78 MB |
| Output | 0.38 M | 3.7% | 2.9% | 188 | 0 | **27.26 MB** | 27.26 MB | 0 |

Five things the QHAS report makes obvious that the bare chrometrace
didn't:

1. **HMX runs at 21.8 % utilisation, HVX threads at 41–56 %.** The
   "DSP busy excluding wait" headline says 98 % busy, but that's
   counting "at least one hardware unit active". When you decompose by
   unit, HMX sits idle 78 % of the time and the HVX threads idle
   ~45–60 %. There's a lot of headroom on each unit individually —
   the bottleneck is *cross-unit dependencies*, not any one unit
   saturating.

2. **HVX threads are noticeably load-imbalanced.** HVX-512 / 514 run
   at ~55 %, HVX-513 / 517 only ~41–45 %. The slowest thread is what
   the wall-clock waits for, so the 41 % HVX-513 is what's actually
   on the critical path while the others have spare cycles. QNN's
   scheduler doesn't perfectly balance.

3. **MatMul is on the dominant path more than Softmax is.** Active
   cycles say Softmax dominates (40 %) but dominant-path cycles say
   MatMul dominates (42 %). The gap is the parallelisable portion:
   Softmax has 27 % of its work on the critical path → 73 % is
   parallel; MatMul has 39 % critical → only 61 % parallel. **Reducing
   MatMul cycles pays out more in wall-clock than reducing Softmax
   cycles by the same amount**, because Softmax's parallelism already
   overlaps it with other work.

4. **The `Output` op is the biggest DRAM writer (27.3 MB).** This is
   the final result writeback `[Hq, Sq, D] = [16, 512, 128]` fp16 ≈
   2 MB × at least 13 channels of DMA. Note: it's running entirely on
   HMX (it's a DMA-flavoured op), and it's 100 % HMX in the
   chrometrace breakdown above. The DMA path for output writeback
   *cannot* overlap with the next dispatch's compute, so this is one
   place per-qb pipelining pays.

5. **A "MatMul" QNN op compiles into ~370 HTP-level ops.** The QHAS
   shows `num_htp_ops` per QNN op type — Softmax decomposes to 190
   sub-ops, both MatMuls together to 738, ElementWiseMul to 333.
   Each HMX-eligible inner matmul tile is one HTP op; each HVX
   data-movement micro-step is one HTP op. The "1 QNN op = many HTP
   ops" fan-out is why per-dispatch QNN graphs aren't infinitely
   faster than smaller graphs — there's a 1000+-op pipeline that runs
   inside one `graphExecute`.

## Inside the HTP ops: half the cycles are layout shuffling

The QHAS report also exposes the *HTP-level* op type for every internal
sub-op — and this is where it gets uncomfortable. The "MatMul" QNN op
in our graph is not just one HMX call; QNN compiles it into a sequence
of slicing, transposition, packing, the actual matrix-multiply tile,
and reassembly. Aggregating across the whole dense Sq=512 dispatch
(10.37 M total cycles):

| HTP op | cycles | % of total | Implements |
|---|---:|---:|---|
| `q::Softmax` | 3,273,278 | **31.5%** | the softmax kernel |
| `q::Reshape` | **2,941,394** | **28.3%** | layout conversions between ops |
| `q::Mul.fp16` | 1,478,202 | 14.2% | the elementwise multiply |
| `q::Transpose_impl` | 943,359 | 9.1% | `transpose_in1=true` on QK |
| `q::*OutputSlice` | 380,544 | 3.7% | output DMA tiles |
| `q::Concat` | 379,303 | 3.7% | reassembling matmul-tile outputs |
| `q::*InputSlicePad` | 338,359 | 3.3% | padding inputs to HMX tile size |
| `q::pack_fp16_pkweights_dynamic` | 276,340 | 2.7% | packing weights for HMX |
| `q::ConvLayer.fp16.s1.tcm` | **236,452** | **2.3%** | the actual HMX matrix multiply |
| `q::*InputSlice` | 68,594 | 0.7% | input DMA tiles |
| `q::ForceFormat_Crouton` | 48,953 | 0.5% | force Crouton tile layout |

**The actual matrix multiply on HMX (`q::ConvLayer.fp16.s1.tcm`)
accounts for 2.3 % of total cycles. Layout / reshape / transpose /
slice / pack ops total ~48 %.** The rest is the genuine HVX kernels
(Softmax 31.5%, Mul 14.2%) plus a sprinkle of DMA.

Where does the 28.3 % `q::Reshape` come from? Not from the `reshape_q`
and `reshape_o` in our graph (those compiled to 0 cycles, as we already
noted). These are reshapes QNN's optimiser inserts *between* user-level
ops to convert between layouts each op prefers:

| QNN op | total cycles | `q::Reshape` cycles | reshape % of that op |
|---|---:|---:|---:|
| `matmul_qk` | 2,888,881 | 879,617 | **30.4%** |
| `scale_qk` | 2,358,609 | 879,061 | **37.3%** |
| `softmax` | 4,163,259 | 888,957 | 21.4% |
| `matmul_av` | 581,933 | 293,759 | **50.5%** |

Every QNN op pulls in 250 K – 900 K cycles of layout-conversion work.
Three things drive this:

1. **HMX vs HVX prefer different layouts.** HMX wants Crouton-tiled
   (4D-tile-blocked storage). HVX wants flat row-major. Transitioning
   between a MatMul (HMX-flavoured) and a Softmax (HVX-flavoured) needs
   a reshape on the boundary.
2. **`transpose_in1=true` is expensive.** It's not a no-op; the 943 K
   `q::Transpose_impl` cycles in `matmul_qk` (9.1 % of total dispatch
   time) are literally just transposing K to K-transpose before the
   matmul. Pre-transposing K offline at chunk-prep time would save
   this every dispatch.
3. **HMX tile reassembly.** A `(M, K, N) = (512, 128, 512)` matmul is
   split into many `(32, 32, 32)` HMX tiles. Each tile is one
   `q::ConvLayer.fp16.s1.tcm` call. The output tiles are then
   `q::Concat`'d and the slicing/reassembly each cost cycles
   (`q::Concat` is 13.1 % of `matmul_qk` alone).

Internal cycle breakdown of `matmul_qk` makes this concrete:

| HTP sub-op | cycles | % of matmul_qk |
|---|---:|---:|
| `q::Transpose_impl` | 943,359 | 32.7% |
| `q::Reshape` | 879,617 | 30.4% |
| `q::Concat` | 379,303 | 13.1% |
| `q::*InputSlicePad` | 338,359 | 11.7% |
| `q::pack_fp16_pkweights_dynamic` | 140,666 | 4.9% |
| `q::ConvLayer.fp16.s1.tcm` (HMX matmul) | **129,000** | **4.5%** |
| `q::ForceFormat_Crouton` | 48,953 | 1.7% |
| `q::*InputSlice` | 24,734 | 0.9% |

The HMX engine does the actual multiplication in **4.5 %** of
`matmul_qk`'s wall-clock — the rest is layout work to get data in and
out of HMX's preferred format. This is the *real* explanation for why
HMX shows 21.8 % utilisation: it's mostly waiting for the surrounding
HVX layout-conversion code to feed it tiles.

## Implications

- ~~**Pre-transpose K at chunk-prep time.**~~ — *tried; it doesn't
  work.* We rebuilt the graph with K declared as `[Hq, D, Skv]` and
  `transpose_in1=false`, so the user transposes K once on the CPU and
  feeds it pre-transposed. Result: `q::Transpose_impl` did go to zero
  as predicted, but total cycles **almost doubled** (10.4 M → 19.9 M,
  NPU dispatch 1.83 → 4.78 ms). QNN's optimiser was already handling
  `transpose_in1=true` as part of HMX tile loading; removing it
  forced new `q::ForceFormat_Crouton` (+1.0 M cycles), `@Fill`
  (+3.3 M), `@Spill` (+2.9 M), and an explosion in `q::*OutputSlice`
  (+5.4 M) because the rest of the graph no longer agreed on a
  consistent internal layout. **The "wasted" 943 K Transpose_impl
  cycles weren't actually wasted — they were the optimal load
  pattern.**

  Lesson: in a heavily-optimised compiler pipeline like QNN's, what
  looks like removable overhead in the profile may be load-bearing.
  Verify by measuring before assuming.

- **Avoid HMX↔HVX op alternation when possible.** Our graph alternates
  MatMul (HMX) → Mul (HVX) → Softmax (HVX) → MatMul (HMX). Each
  transition pulls in ~880 K cycles of reshape. Fusing Mul + Softmax
  into one HVX kernel (or making it part of softmax's epilogue) would
  cut one transition. Practical only if QNN exposes the fused op.
- **Don't expect dense decomposed attention to scale much beyond
  current per-op throughput.** Even if HMX ran at 100 % MAC/cycle on
  `q::ConvLayer`, that's only ~2 % of the dispatch — so a 2× HMX
  speedup gives ~1 % total improvement. The lever is the HVX-side
  layout/softmax work, not HMX.

## CPU↔rpcmem bandwidth and the NPU-gather dead-end

Before picking gather-optimisation strategies, it's worth establishing
what the CPU and the NPU can actually do for data movement. The
`Mllm-Test-QNN-MemBandwidth` suite probes this directly.

**Pure memcpy bandwidth (rpcmem ↔ rpcmem):**

| Size | qnn → qnn | qnn → cpu | cpu → qnn | cpu → cpu |
|---:|---:|---:|---:|---:|
| 8 MB | 31.28 GB/s | 28.06 GB/s | 39.33 GB/s | 34.29 GB/s |
| 16 MB | 24.59 | 23.97 | 29.72 | 28.05 |
| 32 MB | 22.81 | 23.68 | 24.74 | 24.31 |
| 64 MB | **22.99** | **23.41** | **23.62** | **23.20** |

Above 32 MB the working set overflows cache and we hit the DRAM ceiling
at **~23 GB/s** for an rpcmem-to-rpcmem copy. Small (8 MB) sizes get
~30–40 GB/s because the source fits in L2.

**The actual block-gather pattern Path B uses** — selected `top_k`
blocks of `BK = 32` rows from `K [Hq, Skv, D]` into `K_arranged [Hq,
num_qb, top_k·BK, D]`, 8 KB chunks at random offsets:

| Shape | Chunks | dst size | Time | Write BW |
|---|---:|---:|---:|---:|
| Sq=2048, BK=32, top_k=16 | 16,384 × 8 KB | 128 MB | 4.41 ms | **30.4 GB/s** |
| Sq=4096, BK=32, top_k=32, Hq=8 | 32,768 × 8 KB | 256 MB | 8.70 ms | 30.9 GB/s |

Both tables report `useful_bytes / time` (one-sided): the memcpy table
counts the destination buffer size, the gather counts the bytes written
into `K_arranged`. The gather's ~30 GB/s sits in the same regime as the
23–40 GB/s pure-memcpy range — the K source (8 MB) fits in L2, so the
random reads are mostly cached and only the destination stream actually
crosses DRAM. **CPU memcpy-based gather is already operating at the DRAM
write-bandwidth wall.**

**QNN GatherNd (the NPU equivalent):** same pattern, but expressed as a
one-node QNN graph using `qti.aisw GatherNd` so the DSP does the
data-movement.

| Shape | Time (NPU GatherNd) | Time (CPU memcpy gather) | NPU vs CPU |
|---|---:|---:|---:|
| Sq=2048, BK=32, top_k=16 | **19.41 ms** | 4.41 ms | **4.4× slower** |
| Sq=4096, BK=32, top_k=32, Hq=8 | **40.23 ms** | 8.70 ms | **4.6× slower** |

NPU gather is *much* slower than CPU memcpy gather at the same workload.
**Moving the gather to the NPU is a regression, not an optimisation.**
The DSP has plenty of compute throughput, but its DMA path for irregular
small-tile reads is markedly worse than CPU memcpy's DRAM bandwidth.

**The bandwidth wall is only at long Sq.** At smaller Sq the working
set fits in cache and the regime is different:

| Sq | dst size | single-thread time | Write BW |
|---:|---:|---:|---:|
| 128 | 0.5 MB | 0.04 ms | 13.3 GB/s |
| 256 | 2 MB | 0.13 ms | 16.1 GB/s |
| 512 | 8 MB | 0.44 ms | 19.1 GB/s |
| 1024 | 32 MB | 1.69 ms | 19.8 GB/s |
| 2048 | 128 MB | 4.82 ms | 27.9 GB/s |

DRAM bandwidth only saturates around Sq≥1024. Below that the gather is
*not* throughput-limited — there's room to parallelise. **Use OpenMP,
not hand-rolled threads.** OpenMP's barrier is ~5–10 µs vs a
`condition_variable` based barrier at ~30–50 µs; at small Sq the
barrier dominates and the difference is dramatic:

| Sq | Single-thread | Hand-rolled cv (best T) | OpenMP (best T) | OMP speedup |
|---:|---:|---:|---:|---:|
| 128 | 0.040 | 0.040 (T=1; **MT hurts**) | 0.039 (T=8) | parity (no regression) |
| 256 | 0.130 | 0.088 (T=8) | **0.027** (T=4) | **3.3×** vs cv |
| 512 | 0.439 | 0.304 (T=8) | **0.070** (T=8) | **4.3×** vs cv |
| 1024 | 1.69 | 1.08 (T=8) | **0.85** (T=8) | **1.3×** vs cv (2.0× vs single) |
| 2048 | 4.824 | 4.564 (T=4) | 4.968 (T=4) | parity (memory-bound) |

Same work-distribution strategy in both (split `Hq=16` heads across
threads); only the runtime barrier differs.

Four regimes:

1. **Sq=128:** OpenMP catches up to single-thread; cv-based MT is still
   a regression because of cv overhead.
2. **Sq=256–512:** OpenMP gives **3–4× speedup** over single-thread,
   3–4× over hand-rolled MT. Working set fits per-thread cache;
   per-iter barrier overhead is small enough not to bury the gain.
3. **Sq=1024:** still parallel-friendly — OMP T=8 gives 2.0× over
   single-thread and 1.3× over cv. Per-iter barrier is now small
   relative to the gather work, so both runtimes scale; the 32 MB
   working set is large enough that the cv ~30 µs/iter overhead
   amortises but the gather hasn't hit the DRAM write ceiling yet.
4. **Sq=2048:** memory-bound; threading runtime doesn't matter, both
   are within noise of single-thread.

Swapping the BigBatchE2E gather to `cpuGatherOMP` (T=8) and re-running
end-to-end gives a real but smaller win than the bandwidth microbench
predicted:

| Sq | Dense NPU | Sparse big-batch: gather + NPU = total | sparse vs dense | vs no-OMP |
|---:|---:|---:|---:|---:|
| 128 | **0.18** | 0.22 + 0.20 = **0.43** | dense 2.4× faster | −19% (regression) |
| 256 | **0.38** | 0.40 + 0.39 = **0.79** | dense 2.1× faster | +26% |
| 512 | **1.83** | 0.86 + 1.04 = **1.90** | essentially tied (dense 1.04×) | +32% |
| 1024 | 10.61 | 3.00 + 5.53 = **8.54** | **sparse 1.24× faster** (was 1.02×) | +18% |
| 2048 | 86.47 | 10.69 + 22.26 = **32.95** | sparse 2.62× faster | −5% (noise) |

A 2× gap between the in-test gather and the bandwidth-test floor persists
— at Sq=512 the bandwidth test shows 0.07 ms for K-gather (T=8), so K+V
+Q-slice "should" be ~0.19 ms, but the end-to-end measurement gives
0.86 ms. The exact cause hasn't been isolated (cache state, system
contention from the longer test process, OpenMP-pool warmup) — but the
gap is reproducible.

Net effect:

- **Sq=512** sparse closes from 1.5× slower than dense to **parity** —
  worth using sparse here if any other consideration (e.g. KV-cache
  capacity, memory pressure) favours it.
- **Sq=1024** sparse goes from "barely beats dense" (1.02×) to a
  comfortable **1.24× win**.
- **Sq=128–256** OpenMP can't close the gap; dense still wins by 2×.
- **Sq=2048** is DRAM-bound and unchanged.

OpenMP at Sq=128 is a *regression* (the 4 q-blocks × tiny gather work
is smaller than even OpenMP's small barrier overhead). The production
code should fall back to serial `cpuGather` below Sq≈256.

**Implications for gather optimisation:**

- At Sq≈2048+ gather is at the memory-bandwidth ceiling and not
  shrinkable by implementation tweaks.
- At Sq=256–1024 gather is *not* fully bandwidth-limited; multi-threading
  (4–8 workers across heads) gives 1.4–2× — meaningful across the whole
  range where sparse is a candidate (dense competitive at 256–512, sparse
  winning at 1024).
- Below Sq=128, don't try to parallelise — the work is too small.
- Moving gather to QNN GatherNd is a 4-5× regression at any Sq, full stop.
- The only ways to *fundamentally* cut gather at long Sq are
  data-movement reductions: delta-gather across layers (only fetch
  newly-selected blocks), bigger blocks (fewer chunks), sliding-window
  selection (no gather at all, just a free strided view).

## Dense vs big-batch rank-3 sparse, same profile path

Running the same `qnn-net-run + qnn-profile-viewer` pipeline on the
big-batch rank-3 sparse graph at Sq=512 (top_k=4) lets us see *how*
sparse spends its 1.78× wall-clock speedup vs dense.

**Per-HW utilisation:**

| Unit | Dense util % | Sparse util % | Δ |
|---|---:|---:|---:|
| HMX (tid 256) | 21.8% | **46.4%** | **+24.6 pp** |
| HVX-512 | 55.9% | 70.6% | +14.7 |
| HVX-513 | 41.5% | 67.1% | +25.6 |
| HVX-514 | 55.3% | 73.5% | +18.2 |
| HVX-515 | 53.4% | 69.3% | +15.9 |
| HVX-516 | 51.2% | 67.6% | +16.4 |
| HVX-517 | 44.7% | 67.4% | +22.7 |

Every hardware unit is more utilised under sparse. HMX usage *doubles*,
HVX threads gain ~15–25 pp each. The sparse dispatch packs more
concurrent work per unit time — the units that previously sat idle
waiting for the next op are kept busy. That, plus fewer total cycles,
is how sparse wins.

**HTP op-type breakdown** (Sq=512 dense 10.38 M cycles total, sparse
7.66 M cycles total):

| HTP op | Dense | Dense % | Sparse | Sparse % | Why |
|---|---:|---:|---:|---:|---|
| `q::Softmax` | 3.27 M | 31.5% | **1.09 M** | **14.2%** | sparse score tensor is 4× smaller (`[256, 32, 128]` vs `[16, 512, 512]`) |
| `q::Reshape` | 2.94 M | 28.3% | 1.58 M | 20.6% | proportionally less inter-op layout work |
| `q::Mul.fp16` | 1.48 M | 14.2% | **0.48 M** | **6.3%** | scale op also touches the smaller score tensor |
| `q::Transpose_impl` | 0.94 M | 9.1% | **2.30 M** | **30.0%** | sparse has 16× more matmul batches, each needs its K-tile transposed |
| `q::pack_fp16_pkweights_dynamic` | 0.28 M | 2.7% | **1.20 M** | **15.6%** | many small batches → per-batch HMX weight-packing overhead |
| `q::ConvLayer` (HMX matmul) | 0.24 M | 2.3% | 0.15 M | 1.9% | still ~2% in both — HMX does little of the wall-clock |
| `q::*OutputSlice` | 0.38 M | 3.7% | 0.39 M | 5.1% | result-writeback unchanged |
| **TOTAL** | **10.38 M** | | **7.66 M** | | sparse 26% fewer cycles |

The shape of the workload changes how QNN compiles:

- **Sparse cuts HVX work on the score tensor.** Softmax + Mul together
  drop from 45.7% → 20.5% of total. The 4× smaller `[Hq*num_qb, BQ,
  top_k·BK]` intermediate is the direct driver.
- **Sparse grows HMX-side prep work.** Transpose_impl triples and
  weight-packing 4×s because the matmul shape changes from "16 batches
  of (512, 128, 512)" to "256 batches of (32, 128, 128)". Same total
  MACs, but the per-batch fixed costs (transpose tiles, pack weights)
  scale with batch count, not MAC count.
- **Net is still a win** because the HVX shrinkage (−3.18 M cycles on
  Softmax+Mul) beats the HMX-prep growth (+2.30 M cycles on Transpose +
  pack).

**Per-QNN-op cycles & dominant-path %** for sparse:

| QNN op | total cycles | % active | % dominant-path | Note |
|---|---:|---:|---:|---|
| `matmul_qk` | 3.62 M | 47.2% | **45.4%** | mostly on the critical path (Transpose-dominated internally) |
| `softmax` | 1.49 M | 19.4% | 16.2% | well-overlapped (3.2 pp drop vs active) |
| `matmul_av` | 1.25 M | 16.4% | **22.3%** | *more* critical than active → bottleneck on its side too |
| `scale_qk` | 0.88 M | 11.4% | 8.0% | well-overlapped |
| `Output` | 0.40 M | 5.2% | 6.0% | mostly DMA writeback |

`matmul_qk` is on the critical path 45% — same as it was for dense.
The 64% Transpose_impl inside it is therefore worth chasing — but
*not by removing it*, as the pre-transpose-K experiment above showed.
The bigger sparse-specific opportunity is the 47% of `matmul_av`'s
cycles spent on `q::pack_fp16_pkweights_dynamic` (587 K cycles, 7.7%
of total). If QNN exposed a way to cache packed V across dispatches
(V doesn't change within a layer, only across layers), this would
amortise to near-zero.

## Final summary: best result per method

End-to-end ms/layer, best-variant numbers (Qwen3 fp16 attention shape:
Hq=16, BQ=BK=32, D=128, 1/4 sparsity). For each sparse method the
sub-table-best gather strategy was picked (serial vs OpenMP). Per-qb
pipelined uses the cv-based worker (OMP-based was a regression — see
the worker-thread discussion).

| Sq | Dense<br>(NPU only) | Rank-4 sparse<br>(serial gather + rank-4 NPU) | Big-batch rank-3<br>(best gather + rank-3 NPU) | Per-qb pipelined rank-3<br>(cv worker, overlapped) | **Best**           |
|---:|---:|---:|---:|---:|---:|
| 128 | **0.18** | 0.39 | 0.36 (serial gather) | 0.59 | **dense (2.0× over best sparse)** |
| 256 | **0.38** | 0.81 | 0.79 (OMP gather) | 1.84 | **dense (2.1×)** |
| 512 | **1.83** | 2.97 | 1.90 (OMP gather) | 3.48 | **dense (1.04×, near-parity with big-batch)** |
| 1024 | 10.61 | 11.90 | 8.54 (OMP gather) | **7.90** | **per-qb pipelined (1.34× over dense)** |
| 2048 | 86.47 | 52.03 | 31.37 (serial gather) | **22.72** | **per-qb pipelined (3.81× over dense)** |

(For big-batch: serial-gather is slightly faster at Sq=128 because OMP's
~10-20 µs barrier overhead exceeds the ~40 µs work — at Sq=256+ OMP
wins by 1.3-1.5×. At Sq=2048 the difference is within noise: both
schemes hit DRAM bandwidth.)

**Speedup vs dense at each Sq:**

| Sq | Best sparse method | Best sparse total | Sparse vs Dense |
|---:|---|---:|---:|
| 128 | big-batch r3 (serial) | 0.36 | 0.50× (dense 2.0× faster) |
| 256 | big-batch r3 (OMP) | 0.79 | 0.48× |
| 512 | big-batch r3 (OMP) | 1.90 | 0.96× |
| 1024 | per-qb pipelined | 7.90 | **1.34× faster** |
| 2048 | per-qb pipelined | 22.72 | **3.81× faster** |

**Speedup vs naive rank-4 sparse** (showing the cumulative impact of
all our optimisations on the sparse path):

| Sq | Rank-4 naive | Best sparse | Improvement |
|---:|---:|---:|---:|
| 128 | 0.39 | 0.36 | 1.08× |
| 256 | 0.81 | 0.79 | 1.03× |
| 512 | 2.97 | 1.90 | **1.56×** |
| 1024 | 11.90 | 7.90 | **1.51×** |
| 2048 | 52.03 | 22.72 | **2.29×** |

**What changed each method's win:**

- **Rank-4 → big-batch rank-3**: 1.5× from QNN MatMul scheduling
  efficiency on rank-3 inputs (no leading-batch serialisation penalty)
- **Big-batch serial gather → OMP gather**: 1.4× at Sq=256-512 from
  parallelising gather across `Hq=16` heads on 4-8 CPU threads.
  Memory-bound at Sq≥1024 so no further gain there.
- **Big-batch → per-qb pipelined**: 1.4-1.5× *only* at Sq≥1024 from
  hiding CPU gather behind NPU compute. Below Sq=1024 it's a regression
  because cv worker-sync overhead (~30 µs/qb × num_qb) > gather savings.
