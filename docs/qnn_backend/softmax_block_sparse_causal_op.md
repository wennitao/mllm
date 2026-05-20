# Fused `SoftmaxBlockSparseCausal` Custom Op on QNN HTP

This doc covers the **fused** softmax+mask+scale custom op we wrote for
the causal block-sparse attention pipeline on V79 HTP. The op replaces

```
QKs = ElementWiseMul(QK, scale)            // HVX
QKm = ElementWiseAdd(QKs, mask_tensor)     // HVX, materialised mask tensor
P   = Softmax(QKm, axis=-1)                // HVX
```

with a single op:

```
P = SoftmaxBlockSparseCausal(QK, q_block_idx, softmax_scale, bk)
```

The mask is **structurally** computed inside the kernel — never
materialised as a tensor, never shipped over the bind path.

This is a deep dive on one custom op. For the surrounding causal
block-sparse attention pipeline (gather strategy, mask shapes per
method, decomposed measured numbers), see
[block_sparse_attention.md § "Causal masking"](block_sparse_attention.md).
For the general "should I write a custom op?" framing and the package
build recipe, see [custom_hvx_op_skill.md](custom_hvx_op_skill.md).

All numbers below are for Qwen3-shape attention (Hq=16, Hkv=8, D=128,
BQ=BK=32, top_k variable), fp16, on V79 HTP, end-to-end measured.

---

## Why fuse: the decomposed mask is doing real work

The decomposed per-qb causal pipeline ships a `[1, BQ=32, top_k·BK]`
fp16 mask per dispatch via `APP_WRITE`, computes
`Add(QKs, mask)` on HVX, then runs Softmax. Three observations
motivated fusing:

1. **The mask is structurally tiny.** Per (q, c) the mask is one of
   exactly three values: `0` (real history slot or below-diagonal
   triangle), `-1e4` (padding slot or above-diagonal triangle), or part
   of the diagonal triangle. The information needed to reconstruct it
   is `(q_block_idx, q_row, slot, col_in_slot, bk, top_k)` — all
   either op-static or trivially computable from indices.
2. **Materialising it costs DMA + an HVX pass.** The mask is rebuilt
   on CPU per qb (~20–40 µs/qb), DMA'd to rpcmem, and then read back
   by the `Add` op on the DSP. Skipping the materialisation skips the
   CPU build, the DRAM write, and the DRAM read.
3. **Softmax already streams the score tensor.** Adding the mask add
   *and* the scale mul as part of the softmax kernel's first pass
   over the data is essentially free per-element: no extra DRAM
   traffic vs the bare softmax.

The fuse therefore eliminates one materialised tensor (`mask`), one
QNN op (`Add`), and one inter-op layout transition (HVX→HVX is cheap,
but each QNN op still pulls in 250–900 K cycles of `q::Reshape` —
see [block_sparse_attention.md § "Inside the HTP ops"](block_sparse_attention.md#inside-the-htp-ops-half-the-cycles-are-layout-shuffling)).

## Op signature and the runtime `q_block_idx` input

```cpp
out[0] P : [B, H_q, BQ, top_k*BK]   // same dtype as in[0]
in [0] QK            : [B, H_q, BQ, top_k*BK]   // fp16 fast path / fp32 ref
in [1] q_block_idx   : uint32 scalar            // APP_WRITE input, NOT a param
param softmax_scale  : float scalar
param bk             : uint32 scalar            // top_k = W / bk
```

The notable design decision is `q_block_idx` as a **runtime input**, not
a compile-time parameter. The first cut had it as a `Param` — which
meant one compiled QNN graph per q-block (32 graphs/layer at Sq=1024,
BQ=32). Refactoring to a `TensorType1`-templated input (the same
trick `RoPE` and `IRoPE` use for `h_cnt`) lets one compiled graph
serve every q-block — the host writes the value to an `APP_WRITE`
tensor before each `graphExecute`.

Implementation note: the input is a scalar tensor of shape `{1,1,1,1}`
allocated as `kUInt32 / kQNN`. QNN backfills it just like any other
input. Reading it inside the kernel is a single load from the input
tensor's `block(0)` data.

```cpp
// Test-side allocation
auto qb_idx_t = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
*reinterpret_cast<uint32_t*>(qb_idx_t.ptr<void>()) = qb_idx;
// then bind as APP_WRITE input #1
```

## Structural mask: three vectors, never tensorised

For q-block `i`, the score columns `[0, top_k·BK)` are partitioned
into `top_k` slots of `BK` columns each. The selection is arranged so
that:

- **Slot `top_k - 1`** is always the diagonal k-block itself
  (intra-block triangular mask applies).
- **Slots `[0, n_real_hist)`** with `n_real_hist = min(top_k - 1, i)`
  are real historical k-blocks (no mask).
- **Slots `[n_real_hist, top_k - 1)`** are padding slots used only
  when `i < top_k - 1` — fully masked.

Per (q-row q, column c):

```
slot        = c / bk                  // bk == BK
col_in_slot = c % bk
diag_slot   = top_k - 1
n_real_hist = min(top_k - 1, q_block_idx)

if slot < n_real_hist                              -> 0       (no mask)
if slot in [n_real_hist, diag_slot)                -> -1e4    (padding)
if slot == diag_slot && col_in_slot >  q           -> -1e4    (above diag)
if slot == diag_slot && col_in_slot <= q           -> 0       (below diag)
```

Inside the HVX kernel this collapses to **three pre-built vectors per
`(q_row, slot)` decision**:

| vector | when used |
|---|---|
| `zero_v` | real-history slot, or below-diagonal triangle |
| `neg100_v` (`-1e4`) | padding slot |
| `triangle_mask` (built per q_row) | diagonal slot |

The triangle mask itself is built per q-row with one `Q6_Q_vsetq_R`
call:

```cpp
// triangle_mask: lanes [0..q] = 0, lanes [q+1..BK-1] = -1e4
HVX_VectorPred valid_pred = Q6_Q_vsetq_R((q_row + 1) * 4);  // 4 bytes/lane
HVX_Vector triangle_mask  = Q6_V_vmux_QVV(valid_pred, zero_v, neg100_v);
```

**Gotcha caught in debug**: `Q6_Q_vsetq_R(R)` only uses the low 7 bits
of `R`. At `q_row = 31` and `BK = 32`, `(q+1)*4 = 128` wraps to 0 and
the diagonal row gets fully masked. Special-cased:

```cpp
if (q_row + 1 >= bk) triangle_mask = zero_v;  // diagonal q=BK-1: nothing to mask
```

Pair classification at the slot level — bk=32 means two slots fit in
one fp16 64-lane HVX vector (the "pair"):

| Pair class | Action in pass 1 |
|---|---|
| `PK_REAL_REAL` | mul by scale, no mask add |
| `PK_HAS_MASK` | mul by scale, then add appropriate mask vector |
| `PK_SKIP` | both halves are padding → output zero, skip exp |

## HVX kernel architecture

The fp16 fast path is in
[`SoftmaxBlockSparseCausal.cpp`](../../mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/SoftmaxBlockSparseCausal.cpp).
A few design choices that matter:

- **Pair processing.** `bk = 32` and an HVX vector at fp16 holds 64
  lanes, so two slots fit in one vector. The pair walks the row in
  units of `2 * BK` columns; the source-order pair lookup uses
  `Q6_W_vshuff_VVR(-4)` / `Q6_W_vdeal_VVR(-4)` to interleave/deinterleave
  the two slot halves.
- **Three-phase per-row.** Pass 1: compute scaled+masked sf, find
  per-row max. Pass 2: subtract max, expf, sum. Pass 3: multiply by
  `1/sum`, narrow back to fp16, write out. Pass 1's sf vectors are
  cached in a register-resident array (`HVX_Vector sf_a[8], sf_b[8]`)
  so passes 2 and 3 don't redo the widen/scale/mask.
- **Cephes polynomial expf, inlined.** `expf` from libm returns 0 on
  the HTP runtime — a known gotcha. The Cephes polynomial body is
  inlined with `__attribute__((always_inline))`. With the polynomial
  in place expf is ~0.05 ms out of v5's 0.49 ms (small).
- **Widen / narrow primitives.** `Q6_Wqf32_vmpy_VhfVhf` (qf32 = hf×hf
  widen), `Q6_Vsf_equals_Vqf32` (qf32 → sf), `Q6_Vhf_equals_Wqf32`
  (paired qf32 → hf narrow back). All vectorised; no scalar-narrow
  fallback.
- **Scalar reference path.** Same .cpp keeps a scalar reference under
  `#ifdef REFERENCE_OP` for the aarch64 prepare-side build and as a
  fallback for non-fp16 dtypes. Used `__restrict__` not `restrict`
  (the latter is C99, not C++).

## AUTOSPLIT: getting all 6 HVX threads engaged

By default the kernel runs on a single HVX thread. QHAS profiling
showed `tid=513` doing ~1.13 M cycles, the other five idle. The
reference decomposed `Mul + Add + Softmax` distributes across all
six HVX threads at 50–80 % util each — that single-vs-six asymmetry
was the dominant ~4× wall-clock gap once the kernel was vectorised.

The fix is the QNN HTP `AUTOSPLIT` directive — but two prerequisites
took most of the debug effort:

### Prerequisite 1: `REGISTER_PACKAGE_OPTIMIZATIONS()` must run

Without `REGISTER_PACKAGE_OPTIMIZATIONS()` in `LLaMAPackageInit`,
`DEF_PACKAGE_OPTIMIZATION` rules compile but never reach the HTP
optimizer. The macro was missing entirely from `LLaMAPackageInit` —
added in
[`LLaMAPackageInterface.cpp`](../../mllm/backends/qnn/custom-op-package/LLaMAPackage/src/LLaMAPackageInterface.cpp).
Existing FA / RoPE optimisation rules in this package were
*also* no-ops up to that point. Worth verifying by inspecting
`<graph>_htp.json` after graph finalize: a fired AUTOSPLIT splits one
op into N copies with `_split_*` suffixes.

### Prerequisite 2: split on `DIM_HEIGHT` (dim 1), not `DIM_BATCHES` (dim 0)

The op input is rank-3 `[H_q=16, BQ=32, top_k_BK=256]`. QNN backfills
this to rank-4 NHWC `[B=1, H_q=16, BQ=32, top_k_BK=256]` internally.
That means `DIM_BATCHES("*") = 1` (not 16) and `DIM_HEIGHT("*") = 16`.
Splitting along H_q means dim **1**, not dim 0. Verified by
inspecting `fused_v*_htp.json` — the tensor's `dims` field shows
`[1, 16, 32, 256]`.

### Final directive

```cpp
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("SoftmaxBlockSparseCausal", "QK", "q_block_idx", "softmax_scale", "bk"),
    GT(DIM_HEIGHT("*"), 3),
    AUTOSPLIT(1, "I", 3, Op("SoftmaxBlockSparseCausal", TYPICAL_SLICE("QK", "I"),
                            "q_block_idx", "softmax_scale", "bk")))
```

Notes on the `Op(...)` pattern:

- Inputs come first (`QK`, `q_block_idx`), then params (`scale`, `bk`).
- `q_block_idx` is a **scalar input** — pass it through `TYPICAL_SLICE`-
  untouched (no per-slice modification).
- `GT(DIM_HEIGHT("*"), 3)` guards: only fire when there's enough work
  along H_q to be worth splitting.

### Chunk-size tuning

For Hq=16 on V79 HTP with 6 HVX threads:

| chunk | slices | HVX threads | dispatch (standalone) |
|---:|---:|---:|---:|
| 4 | 4 | 4 | 0.19 ms |
| **3** | **6** | **all 6** | **0.152 ms** ← chosen |
| 2 | 8 | rule never fires (htp_ops=1) | 0.49 ms |
| 8 | 2 | rule never fires (broke correctness on qb0_tk8) | — |
| `AUTOTHREAD_HVX` | — | rule never fires | — |

QHAS confirms HVX 512/513/514/515/517 at 85–91 % util on chunk 3,
the leftover-1 slice on one thread at 42 %. Why chunks 2 / 8 don't
fire isn't documented in the public headers; likely "too many slices
for the inner work" or a "splits must be even-divide" heuristic. **3
is the empirical sweet spot for H_q=16.**

### Why H_q only — slicing on BQ is unsafe

The diagonal-slot triangle mask depends on the **global q-row index
within BQ**. AUTOSPLIT on BQ gives each slice rows `q=0..(slice_size-1)`
without adjusting `q_block_idx`, so the mask is recomputed wrong for
slices past row 0. Confirmed: `AUTOSPLIT(2, "I", 4, ...)` produces
fused-vs-host `err = 0.96` with 8K-element divergence. Heads (H_q)
are independent — slicing there is safe.

## Performance progression: v1 → v7

Per-dispatch latency at Hq=16, BQ=32, top_k=8 on V79, **standalone
op test** (one-op graph, repeat-execute the same input):

| Version | Description | ms / dispatch | Speedup vs v1 |
|---:|---|---:|---:|
| v1 | scalar reference | 34.1 | 1.0× |
| v2 | HVX-vectorised, scalar narrow | 8.5 | 4.0× |
| v3 | vectorised qf32→hf narrow | 5.2 | 6.6× |
| v4 | pair widen + pair narrow | 0.51 | 67× |
| v5 | cache pass-1 sf in registers | 0.49 | 70× |
| v6 | AUTOSPLIT chunk 4 (4 HVX threads) | 0.19 | 180× |
| **v7** | **AUTOSPLIT chunk 3 (all 6 HVX threads)** | **0.152** | **226×** |

Reference decomposed `Mul + Add + Softmax` on the same shape: **0.114 ms**
(also HVX-only — not HMX, despite the `Mul` op name). v7 is within
**1.33× of decomposed standalone** — essentially as good as it gets
for an external custom-op-package op without manual QURT threading.

## End-to-end integration: `BlockSparseAttentionCausalFusedTest`

The fused op slots into the per-qb pipelined causal pipeline by
replacing the `Mul → Add → Softmax` triple with one node and
binding `q_block_idx` per-qb via APP_WRITE. Reference test:
[`BlockSparseAttentionCausalTest.cpp`](../../tests/qnn/BlockSparseAttentionCausalTest.cpp)
(fixture `BlockSparseAttentionCausalFusedTest`). Uses two ping-pong
graphs with one APP_WRITE `q_block_idx` per buffer set.

**Correctness verified across all Sq:**

| Test case | Sq | top_k | err vs host | miss |
|---|---:|---:|---:|---:|
| `FusedPerQb_Sq128_TopK2` | 128 | 2 | < 1e-3 | 0 |
| `FusedPerQb_Sq256_TopK2` | 256 | 2 | < 1e-3 | 0 |
| `FusedPerQb_Sq512_TopK4` | 512 | 4 | < 1e-3 | 0 |
| `FusedPerQb_Sq1024_TopK8` | 1024 | 8 | 0.0005 | 0 |
| `FusedPerQb_Sq2048_TopK8` | 2048 | 8 | < 1e-3 | 0 |

**Wall-clock per qb (NPU side, integration):**

| Sq | top_k | Decomp per-qb | Fused per-qb | Fused / Decomp |
|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.300 ms | **0.515–0.530 ms** | **1.72–1.77×** |

The fused integration is **1.75× slower** than the decomposed
integration at Sq=1024, even though the fused kernel itself is only
1.33× of decomposed standalone. The remaining gap is wiring
overhead — quantified in the next section.

## Profiling: where the integration overhead actually comes from

Set `MLLM_QNN_PROFILE=DETAILED` to dump per-op CSV and enable optrace.
The standalone-vs-integration discrepancy decomposes as follows.

### Per-op CSV: aggregate cycles, not wall-clock

| Path | Op | aggregate cycles / dispatch | Note |
|---|---|---:|---|
| **Decomp** | `scale_qk` (Mul) | 145 K | HVX |
| **Decomp** | `add_mask` (Add) | 60 K | HVX |
| **Decomp** | `softmax` | 304 K | HVX |
| **Decomp** | sum | **509 K** | |
| **Fused** | `softmaxblocksparsecausal` | **809 K** | HVX, AUTOSPLIT-replicated across 6 threads |

Important: the per-op CSV reports **aggregate cycles across all HW
units**, not wall-clock. So an AUTOSPLIT op spread across 6 HVX
threads shows ~6× the per-thread cycle count. Wall-clock-wise the
fused op runs in roughly `809 K / 6 = 135 K` cycles per HVX thread,
consistent with its 0.15 ms standalone time at the V79's HVX clock.

### AUTOSPLIT-removal experiment

Built fused with the `DEF_PACKAGE_OPTIMIZATION` directive removed
(single-thread fallback) and re-measured:

| Configuration | Standalone | Integration (Sq=1024) |
|---|---:|---:|
| Fused, AUTOSPLIT chunk 3 | 0.152 ms | 0.530 ms / qb |
| Fused, no AUTOSPLIT | 0.494 ms | 0.778 ms / qb |
| Decomp (reference) | 0.114 ms | 0.300 ms / qb |

Two takeaways:

- **AUTOSPLIT is essential**: removing it costs 3.2× standalone and
  1.5× integration. The 6-thread parallelism is real and matters.
- **AUTOSPLIT adds wiring overhead**: the integration penalty over
  standalone goes from `0.530 - 0.152 = 0.378 ms/qb` (with split) to
  `0.778 - 0.494 = 0.284 ms/qb` (without split). The "cost of having
  AUTOSPLIT in the graph" — sub-dispatch coordination on the HTP side —
  is roughly 100 µs/qb of extra fixed overhead.

### Chunk-size sweep within integration

| Chunk | Standalone | Integration (Sq=1024) | Δ vs chunk 3 |
|---:|---:|---:|---:|
| 3 | 0.152 ms | 0.515 ms | — |
| 4 | 0.190 ms | 0.545 ms | +30 µs |

Chunk 3 wins both standalone and integration; the gap in integration
is within noise. Chunk 3 retained as the default.

### Wiring-overhead diagnosis

Decompose the `0.530 - 0.300 = 0.230 ms/qb` integration gap:

| Component | Estimated contribution | Source |
|---|---:|---|
| Kernel difference | ~40 µs/qb | `0.152 - 0.114 = 38 µs` standalone gap |
| Custom-op-package + AUTOSPLIT wiring | **~190 µs/qb** | residual after kernel diff |

The ~190 µs/qb residual is **per-dispatch wiring overhead specific to
custom-op-package ops with AUTOSPLIT**, not anything in the kernel.
At Sq=1024 with `num_qb=32`, that's ~6 ms/layer of pure overhead the
custom-op route can't escape. Hypotheses:

- Custom-op-package dispatch has a longer setup path than QNN's
  built-in `Softmax` op (which is fused into the QNN core).
- AUTOSPLIT sub-dispatch coordination (6 HVX threads spinning up,
  per-slice tensor metadata setup) compounds with custom-op-package
  per-op fixed costs.

These are QNN-runtime costs we can't reduce from the kernel side.

### What couldn't be measured

`qnn-net-run` rejects the dumped context with "expected batch size = 2"
on the `q_block_idx` scalar input — currently a blocker for getting an
optrace `chrometrace.json` that would directly confirm the wiring
hypothesis. The standalone op test
(`SoftmaxBlockSparseCausalOpTest.cpp`) does dump a context successfully
and would give per-HW-unit breakdown if needed.

## When to use this op

- **Standalone correctness/perf testing**: yes, the kernel is solid
  (1.33× of decomp standalone is a fine result for an external custom op).
- **Per-qb pipelined causal integration**: **no current win**. The
  custom-op + AUTOSPLIT wiring eats more than the mask-elimination
  saves. Decomp Mul+Add+Softmax with the small `[1, BQ, top_k·BK]`
  APP_WRITE mask remains the integration baseline at Sq=1024.
- **Big-batch causal integration**: **also no win** — measured 3× slower
  than decomp big-batch at every Sq. See the next section for details.

## Big-batch variant: `SoftmaxBlockSparseCausalBigBatch`

A second op variant takes a per-row `q_block_idx` vector input instead
of a scalar, so it can replace `Mul+Add+Softmax` in the big-batch r3
graph (where `num_q_blocks` is folded into the leading matmul batch
dim, giving QK shape `[Hq·num_qb, BQ, top_k·BK]`):

```
in [0] QK            : [B, Hq·num_qb, BQ, top_k·bk]
in [1] q_block_idx   : [B, Hq·num_qb, 1, 1]    uint32 vector
                       (host fills [h*num_qb + qb] = qb at graph build)
```

The host fills the vector once at graph build (it's static for a given
Sq — the values just enumerate qb cyclically per head). AUTOSPLIT
slices both QK and `q_block_idx` along dim 1 via `TYPICAL_SLICE` on
both, so each slice's local row index reads the right qb_idx without
any explicit slice-offset math.

### Op signature evolution

Two designs tried for getting the per-row qb index into the kernel:

**Design A (initial)** — per-row `q_block_idx` vector input shape
`[big_batch, 1, 1]` (rank-3 → rank-4 backfill `[1, big_batch, 1, 1]`):

```
Op(QK [big_batch, BQ, K], q_block_idx [big_batch, 1, 1], scale, bk)
AUTOSPLIT(1, "I", 16,
    Op(TYPICAL_SLICE("QK", "I"), TYPICAL_SLICE("q_block_idx", "I"), ...))
```

Host fills `q_block_idx[h*num_qb + qb] = qb` once at graph build. Each
slice's kernel reads `q_block_idx_vec[h_local]`. Two `TYPICAL_SLICE`s on
two different inputs — non-standard pattern for QNN.

**Design B (`SPLIT_START` from optimization grammar)** — uses the grammar's
special cst_int `SPLIT_START("I")` (page 7 of the optimization-grammar
PDF) to synthesize a per-slice `Const` carrying the slice's global row
offset:

```
Op(QK [big_batch, BQ, K], slice_base uint32 scalar, scale, bk, num_qb)
AUTOSPLIT(1, "I", 16,
    Op(TYPICAL_SLICE("QK", "I"),
       gen_ConstScalar_i32(SPLIT_START("I")),  // synthesised per slice
       ...))
```

Host sets `slice_base = 0` (STATIC). For the un-split case
`slice_base + h_local = 0..big_batch-1`, modulo by `num_qb` gives the
global qb. For the split case the AUTOSPLIT replacement substitutes the
slice's actual start row, and the kernel reads it as a normal scalar
input. Single `TYPICAL_SLICE` on QK only, plus a synthesized `Const` —
the standard QNN pattern.

Also added (from the implementing-ops PDF, "Step 3"):

```cpp
DEF_TENSOR_PROPERTIES(
    Op("SoftmaxBlockSparseCausalBigBatch", "QK", "slice_base"),
    Flat("*", "QK"),         // HVX-only kernel — no Crouton conversion
    MainMemory("slice_base"),
    Tcm("QK"))               // pin score tensor in TCM if it fits
```

### Measured numbers

End-to-end ms/dispatch on V79 HTP, Qwen3 fp16 attention shape (Hq=16,
Hkv=8, D=128, BQ=BK=32). Single-shot timing per fresh process to dodge
the big-batch decomp graphFinalize flake (see
[block_sparse_attention.md § "Practical caveats"](block_sparse_attention.md#practical-caveats-observed-on-v79)).
All correctness `miss=0`.

| Sq | top_k | Dense causal | Decomp big-batch | **Fused (Design A)** | **Fused (Design B + Tcm)** | B / Decomp |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 0.22 | 0.28 | 0.89 | **0.71** | 2.54× |
| 256 | 2 | 0.41 | 0.47 | 1.43 | **1.48** | 3.15× |
| 512 | 4 | 2.07 | 1.19 | 3.75 | **2.74** | 2.30× |
| 1024 | 8 | 13.54 | 7.38 | 20.66 | **17.49** | 2.37× |
| 2048 | 8 | OOM | 13.20 | 42.22 | **34.64** | 2.62× |

Design B is **15–27% faster than Design A** at every Sq, but still
**~2.5× slower than decomp big-batch.** The grammar-aware redesign
helped, but didn't unlock parallelism.

The fused kernel takes ~30 M cycles for the softmax stage at Sq=1024
top_k=8, vs decomp's ~11.5 M (Mul+Add+Softmax combined). Per-op CSV:

| Op | Decomp big-batch | Fused big-batch |
|---|---:|---:|
| matmul_qk | 11.05 M | 11.20 M |
| `scale_qk` (Mul) | 3.08 M | — |
| `add_mask` (Add) | 3.05 M | — |
| `softmax` | 5.37 M | — |
| `fused_softmax` | — | **30.0 M** |
| matmul_av | 4.66 M | 5.74 M |
| reshape_o | 1.25 M | 1.21 M |
| Output | 1.29 M | 0.69 M |

(Aggregate cycles per dispatch, Sq=1024, top_k=8.)

### Why fused loses in big-batch — *updated diagnosis*

**Earlier conclusion (wrong)**: "AUTOSPLIT not firing for big-batch."
Based on wall-clock invariance with vs without the directive.

**Correct conclusion**: AUTOSPLIT *does* fire — confirmed by dumping
the QNN graph schematic. The rule produces 32 split-ops at Sq=1024,
each handling 16 rows. **They just run serially instead of in
parallel.**

How we found out: with `MLLM_QNN_PROFILE=DETAILED` set during the
dump test, QNN writes a `<graph>_schematic.bin` flatbuffer alongside
the context binary. Despite being called `.bin`, it's actually a
Python-pprint dump of the graph at every optimization stage. Useful
greps:

```bash
# Count split-ops produced by AUTOSPLIT
grep -c "splithist" <graph>_schematic.bin

# See per-split metadata: [split_anchor_id, chunk_size, ((dim, size, slice_idx),)]
grep "splithist" <graph>_schematic.bin | head
# →  splithist : [ 0x16, 3, ((1,16,0),)]
# →  splithist : [ 0x16, 3, ((1,16,1),)]
# →  ... 32 entries for big-batch fused
```

Per-qb fused has **6 splits** at chunk=3 over Hq=16 — runs in parallel
on 6 HVX threads (~0.15 ms wall = parallel-fan-out). Big-batch fused
has **32 splits** at chunk=16 over big_batch=512 — runs serially
(~17 ms wall ≈ 32 × 0.5 ms per split).

The 0.5 ms per-split number is exactly the **per-qb integration
per-dispatch cost** measured earlier. So big-batch with 32 splits is
effectively 32 sequential per-qb-equivalent dispatches.

**Why per-qb's splits parallelise but big-batch's don't** — best
hypothesis: TCM contention. Each split needs ~256 KB of QK in TCM
scratchpad. Per-qb's 6 splits' total working set fits comfortably in
the 8 MB TCM, so QNN's scheduler can co-resident them across HVX
threads. Big-batch's 32 splits' total exceeds TCM, forcing the
scheduler to serialize TCM allocations. (Cannot confirm definitively
without QNN-internal logging.)

### Things tried during diagnosis (none unlocked parallelism)

| Test | Result |
|---|---|
| Design A vs no-AUTOSPLIT (chunk=16) | Same wall-clock (~21 ms at Sq=1024) |
| Design B vs no-AUTOSPLIT (chunk=16) | Same wall-clock (~18 ms at Sq=1024) |
| AUTOSPLIT chunks 32, 64, 85 | All break graphFinalize — only chunk=16 produces a working graph for our shape |
| `AUTOTHREAD_HVX` (auto-pick chunk) | Same as chunk=16 at Sq=1024; breaks at other Sq |
| Priority `LATE+900` instead of `EARLY` | Same wall-clock |
| `DEF_TENSOR_PROPERTIES` + `Tcm("QK")` | 2–5% faster — minor |

### Schematic-inspection cookbook (for future work)

To diagnose AUTOSPLIT firing for any custom op, add a `DumpContext`
test that creates a single graph and calls `saveContext`. Then:

```bash
# 1. Dump with profiling so the schematic is generated
MLLM_QNN_PROFILE=DETAILED ./test_binary --gtest_filter='*DumpContext*'

# 2. Pull the schematic
adb pull /data/local/tmp/<graphname>_schematic.bin

# 3. Inspect (it's plain text after the .bin header)
grep -B 2 -A 12 "OpName" <graphname>_schematic.bin
grep "splithist" <graphname>_schematic.bin
```

`splithist` lines mean AUTOSPLIT fired. Their count is the number of
split-ops produced. Each split-op's `outputs.dims` shows the per-slice
shape — divide your input H by the chunk to predict how many splits
you should see.

`graph_before_optimization` shows the rule-level graph (post-AUTOSPLIT
but pre-lowering). `graph_after_optimization` shows the fully lowered
HTP graph (with `q::*InputSlice`, `q::ConvLayer.fp16.s1.tcm`,
`q::flat_to_vtcm`, etc).

### qnn-net-run + scalar-input blocker (still unresolved)

Trying to drive the dumped context through `qnn-net-run --profiling_option
optrace` to get a Chrome-trace JSON for per-HW-unit timing fails on
both per-qb and big-batch:

```
Current input tensor with name: q_block_idx with index 3
files populated = 1, batch size = 1 does not match with
expected files populated = 1, batch size = 2
```

The `[1,1,1,1]` uint32 scalar input gets reported as expected
batch=2 by qnn-net-run, even though the schematic confirms it's
`[1,1,1,1]` Int32 in both pre- and post-opt graphs. Tested
workarounds that didn't help: providing 8-byte input file, providing
2-line input list, `--batch_multiplier` variations. The schematic-dump
path (above) bypasses this and gives us all the structural info we
need without `qnn-net-run`.

**QNN built-in Softmax/Mul/Add do parallelize.** They have internal
Crouton tiling and scheduler integration our HVX-only custom op can't
reach. Our cycles (30 M for fused softmax) approximate single-thread
work for the score tensor; decomp's 11.5 M (sum of Mul+Add+Softmax)
reflects multi-thread parallelism the QNN scheduler arranges
automatically.

(Update: schematic dump confirmed AUTOSPLIT *does* fire for our op,
producing 32 splits — but those splits run serially. See updated
section above. Net effect is the same: our op runs single-thread.)

**Stability bonus, not a perf bonus.** Fused big-batch *does* dodge
the decomp big-batch's graphFinalize flake (the static 8 MB tiled
mask tensor is the suspected culprit, which fused eliminates). Out
of ~10 fresh-process runs the fused variant always passed; decomp
big-batch flaked 1–3 times before passing. So fused is a stability
win even though it's a perf loss.

### Conclusion across all integration paths

| Pipeline | Best variant at Sq=1024 |
|---|---|
| Per-qb causal (sync) | Decomp Mul+Add+Softmax (0.300 ms/qb) |
| Big-batch causal | Decomp Mul+Add+Softmax (7.38 ms/dispatch) |
| Either | **Fused op is a perf loss; per-qb fused is closer (1.7×) than big-batch fused (2.8×)** |

The kernel itself is fast (1.33× of decomp standalone), but the QNN
runtime's custom-op wiring overhead and AUTOSPLIT parallelism limits
prevent it from beating the decomposed path in production graph
contexts. Either:

- Stay with decomp Mul+Add+Softmax for both per-qb and big-batch.
- Or skip QNN entirely (raw QURT threads via `libcdsprpc`, llama.cpp
  style) to bypass the wiring overhead — much heavier lift.

## Future work / handoff notes

- **Resolve the `qnn-net-run` scalar-input quirk** so optrace can
  directly confirm the AUTOSPLIT wiring overhead hypothesis. This may
  require a custom input-list format or shipping `q_block_idx` as a
  rank-4 `[1,1,1,1]` tensor in the dump path.
- ~~**Try the fused op in big-batch.**~~ — *done; 3× slower than
  decomp. See "Big-batch variant" section above.* The kernel cost
  scales with row count, but the QNN built-in `Softmax`/`Mul`/`Add`
  scheduler is better than our AUTOSPLIT can match at big-batch scale.
  Worth coming back to if AUTOSPLIT firing for multi-input ops becomes
  more reliable in newer QNN, but as of QAIRT 2.43 it doesn't.
- **Look at llama.cpp's HVX scheduling.** Their custom kernels appear
  to bypass QNN entirely (raw QURT threads via `libcdsprpc`),
  side-stepping the custom-op-package wiring overhead. Heavier lift
  but likely the only path to close the remaining ~190 µs/dispatch
  gap.

## File index

- Kernels:
  - [`SoftmaxBlockSparseCausal.cpp`](../../mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/SoftmaxBlockSparseCausal.cpp)
    — per-qb variant (scalar `q_block_idx` input)
  - [`SoftmaxBlockSparseCausalBigBatch.cpp`](../../mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/SoftmaxBlockSparseCausalBigBatch.cpp)
    — big-batch variant (per-row `q_block_idx` vector input)
- Op registration: [`LLaMAPackageInterface.cpp`](../../mllm/backends/qnn/custom-op-package/LLaMAPackage/src/LLaMAPackageInterface.cpp)
- XML config: [`LLaMAOpPackageHtp.xml`](../../mllm/backends/qnn/custom-op-package/LLaMAPackage/config/LLaMAOpPackageHtp.xml)
- Standalone op test: [`SoftmaxBlockSparseCausalOpTest.cpp`](../../tests/qnn/SoftmaxBlockSparseCausalOpTest.cpp)
  — build target `Mllm-Test-QNN-SoftmaxBlockSparseCausal`
- Integration tests: [`BlockSparseAttentionCausalTest.cpp`](../../tests/qnn/BlockSparseAttentionCausalTest.cpp)
  fixture `BlockSparseAttentionCausalFusedTest` —
  build target `Mllm-Test-QNN-BlockSparseCausal`:
  - per-qb fused: filter `*FusedPerQb_*`
  - big-batch fused: filter `*BigBatchFused_*`
