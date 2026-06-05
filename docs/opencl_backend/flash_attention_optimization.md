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
| 5 | **Vectorize P·V**: 8-d/lane re-tile + transposed-V LDS tile (`V_localT[d*BC+c]`) for contiguous `vload8`, free `vstore8` O | GEMM register-tile + coalesced store | **TODO** | now the dominant phase |
| 6 | Vectorize global K/V tile loads (`vload8` over contiguous D) | GEMM wide loads | TODO (minor) | guard pointer alignment |
| 7 | `image1d_buffer` texture for re-read K/V | GEMM image acts | **deferred** | weak transfer: K/V dynamic, cross-WG reuse scheduling-dependent, FA not BW-bound |
| 8 | Coalesced vector O store | GEMM `vstore4` | **dropped** | FA's O store is already coalesced |

The fp32 kernel is left **untouched** as the numeric reference (`FA_DTYPE=fp32`).
Optimizations apply only to `flash_attention_fp16` (the half-storage path is what
lets `FA_BC=32` fit; fp32 float-LDS at BC=32 would overflow the 32 KB budget).

## Measured result — ranks 1-4 (on-device, Adreno 830)

Correctness preserved: fp16 vs fp32 kernel max_abs ~4e-4, mean_abs ~1.5e-5, no
NaN/Inf (unchanged from the v1 fp16 kernel; both float-accumulate).

**Prefill** (min latency, S_q=S_kv):

| S | fp32 ref | fp16 v1 | **fp16 opt** | opt GF/s | opt vs v1 |
|---:|---:|---:|---:|---:|---:|
| 64   | 3.28 ms  | 3.28 ms  | **1.76 ms**  | 9.7  | 1.86× |
| 128  | 12.22 ms | 12.23 ms | **6.11 ms**  | 11.1 | 2.00× |
| 256  | 48.44 ms | 48.03 ms | **23.99 ms** | 11.2 | 2.00× |
| 1024 | 791 ms   | 765 ms   | **378 ms**   | 11.4 | 2.02× |
| 2048 | 3220 ms  | 3119 ms  | **1586 ms**  | 10.8 | 1.97× |

**Decode** (min latency, S_q=1):

| S_kv | fp16 v1 | **fp16 opt** | opt vs v1 |
|---:|---:|---:|---:|
| 512  | 2.40 ms  | **1.44 ms**  | 1.67× |
| 1024 | 4.69 ms  | **2.72 ms**  | 1.72× |
| 2048 | 9.20 ms  | **5.28 ms**  | 1.74× |
| 4096 | 18.26 ms | **10.43 ms** | 1.75× |

**~2× prefill, ~1.75× decode**, and fp16 now beats fp32 ~2× (it used to tie).
5.5 → ~11.4 GF/s prefill.

### Honest read of the gap to the projection
The multi-agent analysis projected ~20-40× from ranks 1-5; ranks 1-4 delivered
**~2×**. The measured 2× tracks the **lane-recovery from rank 1** almost exactly,
which means ranks 3-4 added little *observable* throughput on top — because the
now-dominant cost is the **still-scalar P·V phase** (rank 5, deferred): each lane
does `FA_BR*FA_BC = 128` scalar FMAs reading `V_local[c*D+t]` at **stride-128 in
LDS**. (Per-technique attribution was not isolated — ranks 1 and 2 are coupled,
since BC=32 needs half storage to fit.) This is exactly why we measure: the
biggest remaining lever is **rank 5**, and per-phase profiling (QK^T vs softmax
vs P·V) should come before further work.

## Next steps (evidence-ranked)
1. **Rank 5 — vectorize P·V** with a transposed-V LDS tile (the now-dominant
   phase). Highest expected remaining lever.
2. **Per-phase profiling** to confirm P·V dominance and quantify ranks 3/4 in
   isolation (A/B the hoisted-softmax barrier and the half8 QK dot).
3. Rank 6 (vectorized global K/V loads) once P·V is fixed.
4. Decode needs its own attention path eventually (S_q=1 is launch/barrier-floor
   bound, not compute-bound).

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
