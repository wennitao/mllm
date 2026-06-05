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

## Next steps (evidence-ranked)
1. **Per-phase profiling** (QK^T vs softmax vs P·V vs barriers) to find the new
   dominant phase post-rank-5 and isolate ranks 3/4 (A/B the softmax barrier and
   the half8 QK dot, which were masked by P·V before).
2. Rank 6 — vectorized global K/V tile loads (`vload8` over contiguous D), now
   that the LDS-side P·V is fixed.
3. A **redesign** for the structural ceiling (cross-q reuse / split-K), if the
   target is >>16 GF/s — incremental ranks won't get there.
4. Decode wants its own path eventually (S_q=1 is launch/barrier-floor bound).

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
