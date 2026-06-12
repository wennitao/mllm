# OpenCL FlashAttention optimization (Adreno 830) — learning from the GEMM/GEMV kernels

The OpenCL FlashAttention kernel (`mllm/backends/opencl/kernels/flash_attention.cl`)
started at **~5.5 GF/s prefill / ~0.9 GF/s decode** on Adreno 830 (SM8750) —
≈0.2% of the device's ~3 TFLOP/s fp16 peak — while the tuned LPBQ matmul kernels
on the *same* silicon reach ~1080 GF/s (fp16 GEMM) / ~1520 GF/s (int4 GEMM) /
40-50 GB/s (decode GEMV). This doc records which techniques from those matmul
kernels transfer to attention, what landed, and the measured result.

Bench + raw numbers: [examples/fa_opencl_bench/](../../examples/fa_opencl_bench/)
(`FA_DTYPE=fp16|fp32`). Device roofline reference for the matmul side:
the OpenCL LPBQ work (separate branch).

> **Status (current):** prefill **~111–116 GF/s** @ S≥1024 (≈3.7–3.9% of the 3 TF
> peak; S=2048 = 148 ms), decode **~5.5 GB/s** (S_kv=4096 6.15 ms). That is
> **~20× over the fp16 v1 baseline** and ~7× over the session-1 result. Session 1
> (ranks 1–5) reached ~16 GF/s and concluded the rest needed a redesign;
> **session 2 disproved that** — causal block-skip + cross-q reuse + a vectorized
> softmax got there with no algorithm change. The prefill ceiling is now genuine:
> bank-conflict padding, manual V-register hoisting, and subgroup barriers were
> all tried and gave nothing (the Adreno wave is <128 lanes, so the 128-lane
> workgroup barriers are irreducible; double-buffering K/V to cut them overflows
> the 32 KB LDS at BR=32). See "Session 2" below.

## The key reframe: FA is not slow for the reason GEMM was

The famous GEMM/GEMV wins (texture cache, int4 weight packing, 8×4 register
tiles to ~1 TF/s) target kernels that are **bandwidth- or ALU-bound**. The FA v1
kernel is neither — it is **occupancy / issue / serialization-bound**:

1. **Only 64 of 128 lanes** ran QK^T (`FA_S_THREADS = FA_BR*FA_BC = 4*16 = 64`).
2. **All 128 lanes redundantly recomputed** every softmax scalar + `exp()`
   (~128× wasted transcendentals).
3. Both matmul dots were **scalar serial LDS reads** (no vectorization).
4. **~384 barriers** at S_kv=2048 from the tiny `FA_BC=16` tile.
5. The fp16 path **widened Q/K/V to `__local float`**, so fp16 == fp32.

So the transferable lessons are mostly the *structural* ones plus one FA-specific
fix (redundant `exp`), not the throughput tricks. **Realistic ceiling with this
scalar-FMA design is ~150 GF/s, not 1 TF/s** — the GEMM kernels reach 1 TF/s via
matmul-style register-tiled data reuse and texture-cached *static* weights, which
FA's online-softmax + per-row structure and *dynamic* K/V can't replicate without
a fundamentally different design (split-K / two-pass, or an HMX dot-product path).

## Precision invariant (non-negotiable)

**Storage and multiply operands → `half`; every accumulator stays `fp32`.** The
QK^T dot, the P·V sum, `o_priv` (lives across the whole S_kv loop), and all
online-softmax stats (`m`, `l`, `a`, `bb`, `l_new`) must be fp32 because they
feed `exp()`. Half-accumulating the D=128 dot measurably perturbs the scores
(~4e-2 max) and corrupts softmax. This is the llama.cpp "accumulate in float,
store in half" rule.

## Transferable techniques (verified) and status

| # | Technique | From | Status | Note |
|--:|---|---|---|---|
| 1 | **`FA_BC` 16→32** so `FA_BR*FA_BC == FA_D == local size` → all 128 lanes compute one S element in QK^T; K/V tile doubles → barriers halved | (FA-structural) | **DONE** | the dominant win |
| 2 | **Native-`half` LDS** for Q/K/V/P (was widened to `float`) — halves LDS traffic, lets `FA_BC=32` fit 32 KB, enables `half8` loads | GEMM fp16 storage | **DONE** | makes fp16 finally beat fp32 |
| 3 | **Hoisted softmax**: one producer lane/row computes `m`,`exp`,`l` once → LDS broadcast (was recomputed ×128) | GEMV subgroup-reduce | **DONE** | |
| 4 | **`half8`-vectorized QK^T dot**, `float8` accumulate, horizontal sum | fp16 GEMM `vload16`→FMA | **DONE** | |
| 5 | **Vectorize P·V**: transposed-V LDS tile (`V_localT[d*BC+c]`) so the P·V read is contiguous in `c` → `half8`/`float8` `vload8` + fp32 accumulate (kept the 1-d/lane mapping; no 8-d re-tile) | GEMM register-tile + contiguous reads | **DONE** | the dominant phase post-1-4; +1.4× |
| 6 | Vectorize global K/V tile loads (`vload8` over contiguous D) | GEMM wide loads | TODO (minor) | guard pointer alignment |
| 7 | `image1d_buffer` texture for re-read K/V | GEMM image acts | **deferred** | weak transfer: K/V dynamic, cross-WG reuse scheduling-dependent, FA not BW-bound |
| 8 | Coalesced vector O store | GEMM `vstore4` | **dropped** | FA's O store is already coalesced |

The fp32 kernel is left **untouched** as the numeric reference (`FA_DTYPE=fp32`).
Optimizations apply only to `flash_attention_fp16` (the half-storage path is what
lets `FA_BC=32` fit; fp32 float-LDS at BC=32 would overflow the 32 KB budget).

## Measured result — ranks 1-5 (on-device, Adreno 830)

Correctness preserved at every step: fp16 vs fp32 kernel max_abs ~4e-4, mean_abs
~1.5e-5, no NaN/Inf (unchanged from the v1 fp16 kernel; all float-accumulate).

Cumulative: **~2.9× prefill (5.5 → ~16 GF/s @S=1024), ~2.3× decode**; fp16 now
beats fp32 ~3× (it used to tie). The ranks-1-4 vs rank-5 columns below are a
clean same-device A/B (the two physical SM8750 units measured within ~0.5%).

**Prefill** (min latency, S_q=S_kv):

| S | fp16 v1 | ranks 1-4 | **ranks 1-5** | 1-5 GF/s | r5 vs 1-4 | total vs v1 |
|---:|---:|---:|---:|---:|---:|---:|
| 128  | 12.23 ms | 6.09 ms  | **4.31 ms**   | 15.7 | 1.41× | 2.84× |
| 256  | 48.03 ms | 23.82 ms | **16.70 ms**  | 16.1 | 1.43× | 2.88× |
| 1024 | 765 ms   | 378 ms   | **266 ms**    | 16.1 | 1.42× | 2.87× |
| 2048 | 3119 ms  | 1587 ms  | **1136 ms**   | 15.1 | 1.40× | 2.75× |

**Decode** (min latency, S_q=1):

| S_kv | fp16 v1 | ranks 1-4 | **ranks 1-5** | r5 vs 1-4 | total vs v1 |
|---:|---:|---:|---:|---:|---:|
| 512  | 2.40 ms  | 1.44 ms  | **1.12 ms**  | 1.29× | 2.14× |
| 1024 | 4.69 ms  | 2.74 ms  | **2.07 ms**  | 1.32× | 2.27× |
| 2048 | 9.20 ms  | 5.25 ms  | **3.93 ms**  | 1.34× | 2.34× |
| 4096 | 18.26 ms | 10.36 ms | **7.87 ms**  | 1.32× | 2.32× |

Per-step: **ranks 1-4 ≈ 2× (mostly the rank-1 lane recovery), rank 5 ≈ +1.4×**
(P·V was the dominant phase after 1-4, exactly as predicted). Decode now sustains
~4.3 GB/s (was ~1.8 effective).

### Honest read: still ~0.5% of peak
At ~16 GF/s prefill we are ~3× up but still **~0.5% of the ~3 TF fp16 peak** and
far from the ~150 GF/s aspiration. Ranks 1-5 fixed the *gross* waste (idle lanes,
redundant exp, scalar/strided dots). The remaining gap is structural and harder:
this kernel still has **one S-element-per-lane QK^T with a per-lane horizontal
reduction, no cross-q data reuse, and 4 barriers per j-iter** — none of which the
GEMM-style register-tiled reuse can be bolted onto without a different algorithm
(split-K / two-pass, or an HMX dot path). So ~16 GF/s is a reasonable plateau for
*this* design; the next big step is a redesign, not another incremental rank.

## Session 2 — block-skip + cross-q reuse (16 → 102 GF/s)

Each change measured on device (Adreno 830), committed individually, correctness
re-checked each step (fp16-vs-fp32 max_abs ≤4.1e-4 incl. partial-block shapes
S_q=130 / 100×250; no NaN).

| # | Change | Prefill effect | Commit |
|--:|---|---|---|
| 1 | **Causal block-skip** — cap the j-loop at the q-block's last diagonal block; the kernel was looping ALL S_kv blocks and masking the upper triangle (~half the work wasted) | 16.2 → 30.2 GF/s (**1.87×**) | `ebb84fd3` |
| 2 | **native_exp** in the softmax | 30.2 → 32.1 GF/s (1.06×) | `5927ee0e` |
| 3 | **Cross-q reuse, BR=8** — each workgroup does 8 q-rows (was 4), each lane computes 2 S-elements sharing the K[c] row load; `nrows` uniform bound (not a `continue`) for partial/decode blocks | 32 → 53 GF/s (1.65×) | `d4079673` |
| 4 | **BR=16** — generalized QK to FA_NSPL S-elements/lane | 53 → 80 GF/s (1.5×) | `79992062` |
| 5 | **BR=32 + S/P LDS alias** (alias the prob buffer into S_local; frees 4 KB → BR=32 fits the 32 KB ceiling, FA_NSPL=8) | 80 → 102 GF/s (1.28×) | `bb48e9f5` |
| 6 | **Dual-kernel** — same source compiled BR=32 (prefill) and BR=8 (decode/tiny S_q); op picks by S_q | decode back to baseline (BR=32 alone was −40% at S_q=1) | `0d9fd6d4` |

| 7 | **Vectorized softmax** (`native_exp` on float8 + `select` for the -inf guard) | 102 → 111 GF/s (1.09×) | `9c6a8a06` |
| 8 | **Decode kernel BR=4** (FA_NSPL=1, less wasted-row work at S_q=1) | decode ~9% | `d05d08a4` |

**Final (min latency):**

| S | rank-5 (start) | session-2 | speedup | GF/s |
|---:|---:|---:|---:|---:|
| prefill 128 | 6.11 ms | **1.06 ms** | 5.8× | 64 |
| prefill 256 | 16.7 ms | **3.04 ms** | 5.5× | 89 |
| prefill 1024 | 266 ms | **38.7 ms** | 6.9× | 111 |
| prefill 2048 | 1136 ms | **148 ms** | 7.7× | 116 |
| decode 2048 | 3.98 ms | **3.21 ms** | 1.24× | — |
| decode 4096 | 7.87 ms | **6.15 ms** | 1.28× | 5.5 GB/s |

**Explored, no gain (confirms the ceiling):** LDS bank-conflict padding of the
transposed-V tile (neutral — conflicts aren't the limiter); manual V-register
hoisting in P·V (neutral — the compiler already hoists loop-invariant loads);
`sub_group_barrier` instead of workgroup `barrier` (broke correctness → the
Adreno wave is <128 lanes, so the 128-lane barriers can't be downgraded).

**Corrected ceiling read:** session 1 called ~16 GF/s a near-plateau needing a
redesign — wrong. The dominant cost was *wasted work* (causal upper-triangle
blocks) and *under-amortized overhead* (load+barriers per j-iter over only 4
rows), both fixable incrementally. At ~102 GF/s (~3.5% of peak) the kernel is now
genuinely overhead/issue-bound: 4 barriers/j-iter, a per-lane horizontal QK
reduction, and the softmax phase running on only FA_BR_H of 128 lanes. The big
remaining levers do need real work (barrier reduction via K/V double-buffering —
blocked by the 32 KB LDS budget at BR=32; or an HMX/dot path absent in OpenCL).

## Session 3 — dedicated decode kernel (5.5 → 43.7 GB/s) + prefill-1TF derisk

### Decode kernel: DONE, 8.1× at S_kv=4096 (validated on device)

The from-scratch decode kernel (`flash_attention_fp16_decode` + a split-K merge
`flash_attention_fp16_decode_merge` in `flash_attention.cl`; dispatched by
`OpenCLFlashAttention2Op` whenever `S_q==1`, fp16) lands the predicted win:

| S_kv | session-2 (prefill kernel @ S_q=1) | **session-3 decode kernel** | speedup | GB/s | % of 60 GB/s |
|---:|---:|---:|---:|---:|---:|
| 512  | ~0.9 ms  | **0.244 ms** | 3.7× | 17.2 | 29% |
| 1024 | 1.68 ms  | **0.397 ms** | 4.2× | 21.2 | 35% |
| 2048 | 3.20 ms  | **0.543 ms** | 5.9× | 30.9 | 52% |
| 4096 | 6.19 ms  | **0.768 ms** | **8.1×** | **43.7** | **73%** |

Correctness vs the fp32 reference is unchanged (S_q=1, S_kv=512: max_abs
4.75e-05). Numbers on a fresh SM8750 unit (`eb49fb9d`); the 2a38935c unit gives
the same within ~5%.

**Design.** Three structural changes vs reusing the prefill kernel at S_q=1:
- **No K/V LDS staging.** At S_q=1 each K/V element is used exactly once — the
  prefill kernel's global→LDS tiling is pure overhead. The decode kernel streams
  K (score phase, lane `t` = key row, full-D dot in registers) and V (P·V phase,
  lane `t` = output dim `d`, V read straight from global, coalesced across lanes)
  directly from global.
- **Zero wasted dot lanes.** All 128 lanes compute a distinct score (vs the
  prefill kernel wasting ~3/4 of QK lanes on out-of-range rows at S_q=1).
- **Split-K (flash-decoding).** `nsplit` partitions over S_kv (default
  `ceil(S_kv/256)`, capped 16; env `FA_NSPLIT` to tune) fill the GPU when B·H=16
  workgroups alone underfill it. Each partition writes an unnormalized
  `(o, m, l)` partial to a persistent scratch `cl::Buffer` ([B·H, nsplit, D+2]
  float); the merge kernel combines them with the standard `w_s = exp(m_s −
  max m_s)` weighting (empty partitions carry `m=−inf` and contribute 0). At
  `nsplit==1` the kernel normalizes and writes O directly (no scratch, no merge
  launch). Measured: `nsplit` is a weak lever (16 marginally best at large S_kv,
  ±5%) — the kernel is already near-roofline; the split mostly matters for
  filling the GPU, which B·H=16 + the streaming layout already mostly achieve.

Phase ablation (env `FA_DEC_PROF` = 1 scores / 2 +softmax / 3 full): scores
alone is **48 GB/s @ S_kv=2048** (near roofline), +softmax 44, full 29 on the
2a38935c unit — i.e. the **P·V phase (strided global V reads) is the residual
bottleneck**, not K reads. A transposed-V or `[S_kv/4,D,4]`-packed V layout is
the lever for the last ~25% but needs a V repack the caller doesn't do today.

### Prefill → 1 TF/s: GEMM-class derisk (measured), full build still TODO

The fused kernel's ~116 GF/s is a property of its 1-score-per-lane,
4-barriers-per-tile design, **not** of the problem: at S≥1024 attention is
compute-bound (materializing the causal score matrix costs only ~1–2 ms/pass of
BW), and this same silicon runs the repo's fp16 GEMM at ~1080 GF/s. So a
two-pass GEMM-class architecture (QK^T GEMM → softmax → P·V GEMM, fp16
S-scratch) is the path to multi-hundred-GF/s.

The open risk was whether the ~1080 GF/s GEMM structure survives attention's
**shallow K=128** (QK^T) and **narrow N=128** (P·V). Measured directly with a
standalone variant-C GEMM microbench at FA shapes (`examples/fa_opencl_bench/`
`gemm_vc_ref.cpp`, target `mllm-fa-gemm-vc-bench`), sweeping {buffer-A, image-A}
× {fp16-acc, fp32-acc} (the image1d_buffer path needed `clCreateImage` added to
`OpenCLLoader`). **Stage-1 GO/NO-GO numbers, on device (SM8750), per head and
head-batched M=16384:**

| shape | M | K | N | **img/fp32-acc** | img/fp16-acc | buf/fp32-acc |
|---|---:|---:|---:|---:|---:|---:|
| QK S=2048 1head | 2048 | 128 | 2048 | **634** | 679 | 511 |
| QK S=2048 H=16  | 16384| 128 | 2048 | **710** | 800 | 445 |
| QK S=1024 H=16  | 16384| 128 | 1024 | **675** | 759 | 439 |
| PV S=2048 1head | 2048 | 2048| 128  | **725** | 795 | 336 |
| PV S=2048 H=16  | 16384| 2048| 128  | **896** | 1033| 413 |
| PV S=1024 H=16  | 16384| 1024| 128  | **865** | 982 | 451 |
| ref q_proj      | 1024 | 2048| 2048 | 936 | **1058** | 370 |

**Decisive GO — both passes clear the 450 GF/s gate by 1.5–2× at fp32-acc.**
Findings:
- **fp32 accumulate is mandatory and cheap.** Only ~12% slower than fp16-acc,
  but fp16-acc's `max_rel` is 0.1–9 (it mis-accumulates the long dot) vs fp32's
  **4e-4**. The QK dot feeds `exp()`, so fp32-acc is non-negotiable (F2); the
  small cost is affordable.
- **The image1d_buffer texture A-operand is the multiplier** (1.6–2.2× over
  buffer; the `ref q_proj` img reproduces the doc's ~1080). The win *grows* with
  M (more rows reuse the cached A texels) → **head-batched M=H·S dispatch is
  fastest**.
- **K=128 does NOT collapse the GEMM**, and the narrow-N P·V is actually *faster*
  than QK when head-batched (deep K amortizes). `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE`
  = 134M texels ≫ the A-image at any target S (S=4096 H=16 needs 524K), so F9 is
  a non-issue.

So a texture two-pass prefill should land the matmul-bound portion at ~650–900
GF/s; net E2E after softmax-reduce + repack overhead realistically **~500–650
GF/s @ S=2048 (≈4–5× the fused 116)**. **Stage 2 (the full three-kernel path:
QK-GEMM w/ scale+causal+partial-stats epilogue → softmax-reduce → P·V-GEMM w/
exp-normalize epilogue, fp16 score scratch, op dispatch S_q ≥ 512) is the next
chunk.** Stretch: int8-DP4A QK GEMM (the `block_sparse_attention.cl` recipe;
~1520 GF/s class) — the fp16 backend's int8-dot probe reports "No" on this branch
but `cl_khr_integer_dot_product` IS exposed (wrong ext-name probe; fixed on
`blocksparse-opencl`). Full corrected spec + staged plan + kill-criteria:
the design-workflow synthesis (run `wf_91ae0fe5-6b8`).

## Earlier next steps (superseded above for decode)
1. ~~**Dedicated decode kernel.**~~ DONE — see Session 3.
2. **Prefill is at its ceiling** for this design (~116 GF/s). Going further needs
   either barrier elimination (blocked: wave<128 so no subgroup barriers;
   double-buffer K/V overflows the 32 KB LDS at BR=32) or an HMX/dot-product path
   that OpenCL on Adreno doesn't expose. Both are large, uncertain efforts.
3. Per-phase profiling (QK^T vs P·V vs softmax vs barriers) to confirm where the
   residual prefill time goes before any further prefill attempt.

## Reproduce
```bash
cmake --build build-android-arm64-v8a --target mllm-fa-opencl-bench -j
# After editing flash_attention.cl, regenerate the embedded source:
( cd mllm/backends/opencl/kernels && python3 gen_opencl_kernel_cpp_wrap.py . )
#   then `git checkout` the unrelated churned *_cl.cpp + a_opencl_source_map.hpp
adb push build-android-arm64-v8a/bin/mllm-fa-opencl-bench \
         build-android-arm64-v8a/bin/libMllm{RT,CPUBackend,OpenCLBackend}.so \
         <ndk>/.../aarch64/libomp.so  /data/local/tmp/fa_ocl/
adb shell 'cd /data/local/tmp/fa_ocl && LD_LIBRARY_PATH=. FA_DTYPE=fp16 ./mllm-fa-opencl-bench'
adb shell 'cd /data/local/tmp/fa_ocl && LD_LIBRARY_PATH=. FA_DTYPE=fp32 ./mllm-fa-opencl-bench'  # reference
```
