# GEMM Latency on QNN HTP: Measured Reference for Qwen3-1.7B Shapes

A baseline of single-MatMul latency on the QNN HTP backend, covering every
GEMM shape that shows up in Qwen3-1.7B prefill and decode. Useful as
roofline reference when deciding whether a custom op is worth writing,
when budgeting a layer, or when checking that a graph isn't leaving HMX
on the table.

The benchmark is reproducible — see
[tests/qnn/GemmLatencyTest.cpp](../../tests/qnn/GemmLatencyTest.cpp).

---

## What was measured

A single-node QNN graph with one `qti.aisw` MatMul. Both A and B are
fp16 `APP_WRITE` inputs (dynamic — no constant-folding), and the output
is fp16 `APP_READ`. Each shape runs one untimed warmup followed by 10
timed `graphExecute` calls; the table reports the average wall time and
the effective rate (`2·M·N·K / latency`).

Dynamic-B is deliberately apples-to-apples with the Q·K^T attention case
(where neither side is a constant). A static-B run — closer to a real
`Linear` layer — would be ~10–20% faster than the numbers below at large
M because QNN can quant-prepare and pre-tile the weight at finalize.

**Test setup:**

- Hardware: Snapdragon 8 Elite (SM8750, Sun) — V79 HTP
- QAIRT: v2.43.0.260127150333_193827
- All operands fp16, layout NCHW-ish (QNN's default row-major)
- `MatMul` with `transpose_in1=true` only on Q·K^T

---

## LLM linear projections (Qwen3-1.7B)

Qwen3-1.7B layer geometry: hidden=2048, intermediate=6144, Hq=16, Hkv=8,
head_dim=128. Per-head widths: q/o_proj=2048, k/v_proj=1024, gate/up=6144,
down=2048.

### Prefill — M is sequence length

| M | q_proj<br>[M,2048]·[2048,2048] | kv_proj<br>[M,2048]·[2048,1024] | o_proj<br>[M,2048]·[2048,2048] | gate/up<br>[M,2048]·[2048,6144] | down<br>[M,6144]·[6144,2048] |
|---:|---:|---:|---:|---:|---:|
| 512  | 0.93 ms · 4.6 TF | 0.38 ms · 5.7 TF | 0.94 ms · 4.6 TF | 3.28 ms · 3.9 TF | 4.65 ms · 2.8 TF |
| 1024 | 1.38 ms · 6.2 TF | 0.60 ms · 7.2 TF | 1.38 ms · 6.2 TF | 9.92 ms · 2.6 TF | 7.97 ms · 3.2 TF |
| 2048 | 2.59 ms · 6.6 TF | 1.01 ms · 8.5 TF | 2.49 ms · 6.9 TF | 31.80 ms · 1.6 TF | 9.62 ms · 5.4 TF |
| 4096 | 4.41 ms · 7.8 TF | 1.82 ms · 9.4 TF | 4.35 ms · 7.9 TF | 26.84 ms · 3.8 TF | 17.63 ms · 5.8 TF |

**Notes:**

- **Peak rate seen: 9.4 TF/s** at M=4096 KV-proj. That's the de-facto fp16
  HMX upper bound under our dispatch path — useful as the denominator
  for a "what fraction of peak am I hitting?" sanity check.
- **Gate at M=2048 is an outlier** (1.6 TF/s — half the rate of M=1024 *or*
  M=4096 at the same op shape). The QNN compiler picks a different tile
  schedule for that exact (M, K, N) combo and lands on a worse one. If
  your prefill chunk lands at M=2048, consider either splitting it
  (2× M=1024 is ~20 ms vs 31.8 ms here) or padding it up to M=2560/3072
  where HMX is happier. Worth re-measuring after any QAIRT bump.
- **Down at M=512 is slower than q_proj at M=512** despite the same FLOP
  count, because K=6144 means more weight reads per output tile and the
  smaller M doesn't amortize them.

### Decode — M=1

Memory-bound by weight loads from DDR; FLOP rate is meaningless here,
read the latency column directly.

| op | shape | latency |
|---|---|---:|
| q_proj  | [1,2048]·[2048,2048] | 0.46 ms |
| kv_proj | [1,2048]·[2048,1024] | 0.17 ms |
| o_proj  | [1,2048]·[2048,2048] | 0.45 ms |
| gate/up | [1,2048]·[2048,6144] | 3.01 ms |
| down    | [1,6144]·[6144,2048] | 2.15 ms |

Per-layer linear-only decode budget: 1 q + 1 kv + 1 kv + 1 o + 2·gate +
1 down = ~9.7 ms. With 28 layers this comes to ~270 ms/token from
linears alone — already over the 100 ms/token comfort budget, which is
why **decode wants quantised weights** (Q4_0 / Q4_K) more than prefill
does. Dynamic-fp16 numbers here are an upper bound on what static-Q4
will improve over.

---

## Attention matmuls (Hq=16, D=128)

After GQA expansion to Hq=16 heads. Q·K^T uses `transpose_in1=true`.

### Prefill — Sq = Skv

| Sq | Q·K^T<br>[16,Sq,128]·[16,Skv,128]^T | A·V<br>[16,Sq,Skv]·[16,Skv,128] |
|---:|---:|---:|
| 512  | 0.52 ms · 2.1 TF | 0.26 ms · 4.1 TF |
| 1024 | 2.67 ms · 1.6 TF | 0.91 ms · 4.7 TF |
| 2048 | 30.33 ms · 0.57 TF | 3.52 ms · 4.9 TF |
| 4096 | 113.31 ms · 0.61 TF | — (unreachable, see below) |

**Q·K^T is the bad case.** K=128 is too narrow to amortize HMX tile
overhead, so it's running closer to HVX rates than HMX rates. By Sq=2048
it dominates the whole attention block (30 ms for Q·K^T vs 3.5 ms for
A·V).

This is why decomposed attention beats a fused single-op approach: A·V
gets the HMX fast path, and you only pay the Q·K^T tax on the part that
genuinely can't avoid it. Block-sparse attention sidesteps Q·K^T
entirely on the rows the heuristic prunes — see
[block_sparse_attention.md](block_sparse_attention.md).

**`Attn_Prefill4096_AV` is unreachable** in the current setup: the P
tensor [16, 4096, 4096] alone is 512 MB in fp16, which exceeds what the
DSP process domain can register in a single graph context (~4 GB PD
limit, and the same context also holds the inputs, outputs, and any
sibling graphs). To benchmark it you need to tear the QNN context down
between cases, or split per-head.

### Decode — Sq=1

| Skv | Q·K^T | A·V |
|---:|---:|---:|
| 512  | 0.25 ms | 0.09 ms |
| 2048 | 0.42 ms | 0.20 ms |
| 4096 | 0.76 ms | 0.38 ms |

Both scale roughly linearly with Skv — pure KV-cache read bandwidth.
Aggregate (QK + AV) at Skv=2048 is 0.62 ms, which is ~6% of a decode
layer's linear budget. Attention is not the bottleneck at decode for
Qwen3-class context lengths.

---

## Square roofline

Pure square (M=N=K) — independent reference for what the HMX path can
do when nothing else is in the way.

| size | latency | TFLOP/s |
|---:|---:|---:|
| 256  | 0.071 ms | 0.47 |
| 512  | 0.114 ms | 2.35 |
| 1024 | 0.305 ms | 7.03 |
| 2048 | 2.53 ms  | 6.80 |
| 4096 | 30.43 ms | 4.52 |

**1024³ is the sweet spot.** HMX tile utilisation peaks there
(7.0 TF/s ≈ 75% of the 9.4 TF/s ceiling seen in KV-proj at M=4096).
The 4096³ regression is the same gate-at-M=2048 phenomenon — the
compiler picks a worse schedule for the larger square than for the
rectangular version of the same workload.

---

## W4 weight (naive per-channel int4) × fp16 activation

Same Qwen3-1.7B shape sweep, but with the static weight quantized to 4-bit
using the simplest QNN encoding: `QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET`
with `bitwidth=4`, one fp32 scale + int32 offset per output channel. No
block scaling (LPBQ is a separate path — the production Qwen3-NPU pipeline
uses it for better accuracy, not for speed).

**Notes that matter for reading the numbers:**

- **Activation had to be fp16, not fp32.** HTP MatMul rejects fp32 × int4
  outright. Asked for "W4A32"; the kernel surface only offers W4A16.
- **HTP MatMul doesn't actually list `int4` in its supported I/O dtype set.**
  QNN prints a `validateOpConfig 3110` warning at graph build, then runs
  the op anyway. Output passes; the warning is benign.
- **Weights must be non-zero** in the test buffer. With all-zero nibbles
  QNN folds the matmul out (we measured 19–27 TF/s, which is well past
  HMX peak — that was the giveaway). The numbers below use random nibbles.
- **No native int4-mac on V79.** HMX dequantizes the W4 weight to fp16
  inline before the matmul. The win is weight-load bandwidth (half the
  bytes), not raw FLOP rate. At small M / decode this is decisive; at
  large compute-bound M the W4 path collapses back to roughly the fp16
  rate.

### Prefill (M = seq len)

| M | q_proj | kv_proj | o_proj | gate/up | down |
|---:|---:|---:|---:|---:|---:|
| 512  | 0.53 ms · 8.0 TF | 0.29 ms · 7.4 TF | 0.53 ms · 8.2 TF | 1.59 ms · 8.1 TF | 3.50 ms · 3.7 TF |
| 1024 | 1.10 ms · 7.8 TF | 0.43 ms · 10.0 TF | 1.11 ms · 7.8 TF | 8.24 ms · 3.1 TF | 7.02 ms · 3.7 TF |
| 2048 | 2.19 ms · 7.8 TF | 0.72 ms · 11.9 TF | 2.17 ms · 7.9 TF | 23.20 ms · 2.2 TF | 5.04 ms · 10.2 TF |
| 4096 | 3.97 ms · 8.7 TF | 1.76 ms · 9.8 TF | 4.06 ms · 8.5 TF | 28.82 ms · 3.6 TF | 16.37 ms · 6.3 TF |

### Decode (M=1) — where the W4 win actually shows

| op | shape | fp16 | W4 | speedup |
|---|---|---:|---:|---:|
| q_proj  | [1,2048]·[2048,2048] | 0.46 ms | 0.14 ms | **3.3×** |
| kv_proj | [1,2048]·[2048,1024] | 0.17 ms | 0.10 ms | 1.7× |
| o_proj  | [1,2048]·[2048,2048] | 0.45 ms | 0.14 ms | **3.3×** |
| gate/up | [1,2048]·[2048,6144] | 3.01 ms | 0.28 ms | **10.6×** |
| down    | [1,6144]·[6144,2048] | 2.15 ms | 0.61 ms | **3.5×** |

This is the headline result: per-layer linear decode budget drops from
~9.7 ms (fp16) to **~1.3 ms (W4)**. With 28 layers that's ~37 ms/token
from linears, comfortably under any reasonable interactivity bar. W4 is
basically mandatory for decode; the fp16 numbers were never a real target.

### Square roofline

| size | fp16 | W4 |
|---:|---:|---:|
| 256  | 0.071 ms · 0.47 TF | 0.066 ms · 0.51 TF |
| 512  | 0.114 ms · 2.35 TF | 0.100 ms · 2.69 TF |
| 1024 | 0.305 ms · 7.03 TF | 0.264 ms · 8.13 TF |
| 2048 | 2.53 ms  · 6.80 TF | 2.16 ms  · 7.95 TF |
| 4096 | 30.43 ms · 4.52 TF | 27.62 ms · 4.98 TF |

Modest improvement (10–15%) across the square sweep — exactly the
"compute-bound, bandwidth doesn't matter much" regime. The decode column
above is the regime that pays off; this column is the regime that doesn't.

### Caveats specific to the W4 path

- **The "gate at M=2048" cliff is worse here** (2.2 TF/s vs 1.6 TF/s in
  fp16). Same compiler tile-schedule pathology, amplified by the W4
  dequant path. Splitting that shape into two M=1024 calls is now even
  more profitable.
- **kv_proj at M=2048 hits 11.9 TF/s** — the W4 peak observed in this
  sweep, higher than the fp16 peak (9.4 TF/s) at the same op family.
  Likely because W4 reduces tile-prefetch pressure enough that HMX stays
  fed across the wider N=1024 output.
- **down at M=2048 → 10.2 TF/s; same shape at M=4096 → 6.3 TF/s.** Not a
  measurement error — confirmed across re-runs. Likely a tile-edge effect
  where the M=4096 schedule has uneven last-tile work. Treat as a
  reminder that the QNN compiler picks tile shapes per (M, N, K) and you
  shouldn't extrapolate.

---

## Cross-stack comparison — vs llama.cpp ggml-hexagon

Same SM8750/V79 silicon, different software stack:
`libggml-htp-v79.so` from llama.cpp dispatches MUL_MAT directly to the
DSP queue, bypassing QNN. Numbers in this section come from the F16 /
Q4_0 sweep added to llama.cpp's `tests/test-backend-ops.cpp` on commit
`4c1c3ac09`.

### Methodology asymmetry — read this first

The two harnesses are **not measuring the same thing**:

- **ggml-hexagon** replicates the op N times into one graph until total
  work ≥1 s, then divides wall time by N. Per-call dispatch overhead is
  amortized to ≈0.
- **QNN (this doc)** times 10 separate `graphExecute` calls. Per-call
  launch overhead is *in* each sample.

For sub-ms decode ops this is the dominant variable. Their F16 decode
plateaus at ~63 GF/s across every shape — that's ~60 GB/s effective DDR,
i.e. memory-bound on the silicon. Our fp16 decode q_proj reports ~18 GF/s,
which would imply ~18 GB/s effective DDR (well below realistic). The gap
is dispatch overhead, not silicon.

Treat their numbers as **"what the kernel could do"**, ours as
**"what an actual graphExecute call costs"**. Both useful; not directly
comparable for sub-ms ops without acknowledging this.

### Decode (M=1) — sub-ms regime, methodology dominates

| op | ggml-hex Q4_0 | QNN W4 | ggml F16 | QNN fp16 |
|---|---:|---:|---:|---:|
| q_proj  | 43 µs  | 140 µs | 134 µs | 460 µs |
| kv_proj | 24 µs  | 100 µs | 68 µs  | 170 µs |
| o_proj  | 43 µs  | 140 µs | 134 µs | 450 µs |
| gate/up | 120 µs | 280 µs | 395 µs | 3010 µs |
| down    | 120 µs | 610 µs | 400 µs | 2150 µs |

Their Q4_0 vs our W4: **2–5× faster on the wire**. Per-layer
linear-decode budget: **16 ms/token (theirs) vs 36 ms/token (ours)**.
Probably half that gap is dispatch amortization, half is real (W4
dequant in QNN adds latency that ggml-hexagon's REPACK fast path avoids).
Our fp16 `gate` at 3 ms is a separate outlier — QNN's M=1 schedule for
that shape is genuinely bad, not just a dispatch artefact.

### Prefill — compute-bound, more apples-to-apples

**M=512:**

| op | ggml-hex Q4_0 | QNN W4 | QNN fp16 |
|---|---:|---:|---:|
| q_proj  | 0.70 ms · 6.2 TF | **0.53 ms · 8.0 TF** | 0.93 ms · 4.6 TF |
| kv_proj | 0.44 ms · 4.9 TF | **0.29 ms · 7.4 TF** | 0.38 ms · 5.7 TF |
| o_proj  | 0.70 ms · 6.2 TF | **0.53 ms · 8.2 TF** | 0.94 ms · 4.6 TF |
| gate/up | 1.72 ms · 7.5 TF | **1.59 ms · 8.1 TF** | 3.28 ms · 3.9 TF |
| down    | **2.53 ms · 5.1 TF** | 3.50 ms · 3.7 TF | 4.65 ms · 2.8 TF |

**M=1024:**

| op | ggml-hex Q4_0 | QNN W4 | QNN fp16 |
|---|---:|---:|---:|
| q_proj  | 1.43 ms · 6.0 TF | **1.10 ms · 7.8 TF** | 1.38 ms · 6.2 TF |
| kv_proj | 0.91 ms · 4.7 TF | **0.43 ms · 10.0 TF** | 0.60 ms · 7.2 TF |
| o_proj  | 1.42 ms · 6.1 TF | **1.11 ms · 7.8 TF** | 1.38 ms · 6.2 TF |
| gate/up | **3.46 ms · 7.5 TF** | 8.24 ms · 3.1 TF | 9.92 ms · 2.6 TF |
| down    | **4.99 ms · 5.2 TF** | 7.02 ms · 3.7 TF | 7.97 ms · 3.2 TF |

The story flips by op:

- **q/kv/o_proj (compact 2048×2048-ish rectangles):** QNN W4 wins
  consistently. KV proj at M=1024 is the biggest gap (0.43 vs 0.91 ms —
  QNN is **2.1× faster**). QNN's HMX schedule for these shapes is
  genuinely tighter.
- **gate/up and down (wide K or wide N):** ggml-hexagon wins, sometimes
  by a lot. At M=1024, ggml is **2.4× faster on gate/up** (3.46 vs 8.24
  ms) — that's the QNN compiler cliff this doc already flagged, sidestepped
  by a different scheduler.
- **`nrows(src1) ≤ 1024` is a hard ggml-hex cap**, so M≥2048 single-call
  prefill (and Sq≥64 batched attention) have no ggml counterpart — those
  workloads either chunk on ggml or stay on QNN.

Aggregate per layer at M=1024 (Q4_0 vs W4): **16.6 ms (ggml) vs 26.6 ms
(QNN)**. ggml wins by 1.6×, **almost entirely because of the gate/down
delta**. Fix the QNN gate cliff (split it, pad to a friendlier M, or
swap op type) and the gap mostly disappears.

### Attention decode — QNN wins clearly

| op | ggml-hex F16 | QNN fp16 |
|---|---:|---:|
| Q·K^T Skv=512  | 242 µs | 250 µs |
| Q·K^T Skv=2048 | 1551 µs | **420 µs** |
| Q·K^T Skv=4096 | 3092 µs | **760 µs** |
| A·V   Skv=512  | 103 µs | 90 µs |
| A·V   Skv=2048 | 394 µs | **200 µs** |
| A·V   Skv=4096 | 734 µs | **380 µs** |

Parity at Skv=512, then QNN scales 4× better. The ggml note in their doc
flags batched-GEMV not tiling across 16 heads — that's the cost.
Aggregate at Skv=4096: 3.83 ms (ggml) vs 1.14 ms (QNN). Long-context
decode attention is a clear QNN advantage; the ggml stack also forbids
prefill attention with Sq≥64 entirely under its `nrows ≤ 1024` cap.

### Square roofline — pure HMX rate

| size | ggml-hex Q4_0 | QNN fp16 | QNN W4 |
|---|---:|---:|---:|
| 256³  | 0.73 TF | 0.47 TF | 0.51 TF |
| 512³  | 2.13 TF | 2.35 TF | 2.69 TF |
| 1024³ | 4.47 TF | **7.03 TF** | **8.13 TF** |

At 1024³ — purely compute-bound — QNN hits HMX peak and ggml leaves
~40% on the floor. ggml's own doc calls this out; HMX usage there may be
gated behind a flag, or the schedule is simply weaker for the square
workload.

### Where each stack wins

| regime | winner | why |
|---|---|---|
| Decode (M=1), linear projections | ggml-hexagon | dispatch amortization + tighter W4 path |
| Prefill q/kv/o projections | QNN | better HMX schedule on compact rectangles |
| Prefill gate/up + down | ggml-hexagon | QNN compiler-cliff territory |
| Prefill at M ≥ 2048 single-call | QNN | ggml `nrows ≤ 1024` cap |
| Decode attention (long context) | QNN | ggml batched-GEMV doesn't tile across heads |
| Prefill attention (Sq ≥ 64) | QNN | ggml `nrows ≤ 1024` cap blocks it |
| Pure 1024³ compute | QNN | HMX peak vs ggml's gap |

**Bottom line:** ggml-hexagon wins decode (where dispatch overhead
matters most) and a couple of specific prefill shapes where QNN's
compiler picks bad tiles. QNN wins almost everything else — attention at
long context, large prefill, peak HMX utilisation. For Qwen3-1.7B
end-to-end the per-layer M=1024 prefill budget is ~1.6× faster on ggml
today, almost entirely because of the QNN gate cliff. That's a fixable
problem on the QNN side, not a structural disadvantage.

---

## How to reproduce

```bash
# Build (one-time)
cmake --build build-android-arm64-v8a-qnn --target Mllm-Test-QNN-GemmLatency -j

# Push QNN runtime + test binary
adb push $QAIRT_SDK_ROOT/lib/aarch64-android/{libQnnHtp.so,libQnnHtpV79Stub.so,libQnnHtpPrepare.so,libQnnHtpV79CalculatorStub.so,libQnnSystem.so} /data/local/tmp/
adb push $QAIRT_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so /data/local/tmp/
adb push build-android-arm64-v8a-qnn/bin/libMllm{RT,QNNBackend,CPUBackend}.so /data/local/tmp/
adb push build-android-arm64-v8a-qnn/bin/Mllm-Test-QNN-GemmLatency /data/local/tmp/
adb push $ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/21/lib/linux/aarch64/libomp.so /data/local/tmp/

# Run a subset (full test sweep exhausts the DSP PD memory in one go —
# the test fixture keeps every graph alive in a single context). Split into
# filter groups, e.g.:
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=".;/data/local/tmp" \
  ./Mllm-Test-QNN-GemmLatency --gtest_filter="-*W4A16_*:*Attn_Prefill4096_AV*"'
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=".;/data/local/tmp" \
  ./Mllm-Test-QNN-GemmLatency --gtest_filter="*W4A16_*"'

# Bump iteration count for tighter numbers (default 10):
MLLM_QNN_GEMM_TIMING_RUNS=50 ./Mllm-Test-QNN-GemmLatency --gtest_filter="*Square_2048*"
```

---

## Caveats

- **Single graph, no scheduling overhead.** A real layer runs many
  matmuls in one graph and the QNN scheduler can hide some launch
  overhead. These numbers are an *upper bound* on per-op cost.
- **Dynamic B inflates by ~10–20%.** A real Linear layer with a static
  fp16 weight will be faster than the prefill numbers in this doc.
  (The W4 weight here *is* static, which is part of why the W4 numbers
  beat the fp16 numbers in the prefill table.)
- **No LPBQ here.** Production Qwen3-NPU uses block-wise int4 (LPBQ)
  for better accuracy at the same bit count. Speed-wise LPBQ is
  comparable to the naive per-channel int4 measured here; the doc
  focuses on the simpler encoding because it's easier to set up and
  the latency story is the same.
- **Numbers drift across QAIRT versions.** The "gate at M=2048" cliff
  is exactly the kind of thing that moves with a compiler bump. Re-run
  this whole sweep after every QAIRT upgrade.
