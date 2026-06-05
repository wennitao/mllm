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

## Next steps (evidence-ranked)
1. **Per-phase profiling** to confirm the post-session-2 dominant phase (likely
   barriers + QK horizontal reduction).
2. **Barrier reduction** (software-pipelined K/V load / double-buffer) — needs
   an LDS budget that BR=32 doesn't leave; may trade BR down.
3. A **dedicated decode kernel** (true GEMV-style, lane=key split-K) — the BR=8
   reuse still wastes lanes at S_q=1; decode is memory-bound at ~4.3 GB/s
   (≈7% of the 60 GB/s roofline), so there's headroom.
4. Vectorized global K/V tile loads (`vload8` over contiguous D).

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
