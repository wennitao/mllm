# Split-Prefill: 2L+1 chunked LPBQ SHA model on QNN HTP

This doc covers the **split-prefill** restructuring of the LPBQ SHA per-qb
causal block-sparse Qwen3 model: how the monolithic per-qb graph is broken
into one chunk per "attention boundary" so that the bulk of the transformer
compute (QKV / O / MLP / RMSNorm) runs at `M = Sq` instead of `M = BQ = 32`,
escaping the small-M HMX efficiency cliff documented in
[block_sparse_attention.md § "matmul_av is slower per MAC in big-batch sparse"](block_sparse_attention.md#per-op-profile-what-actually-dominates-inside-the-graph).

The architecture compiles, loads, and dispatches end-to-end on V79 HTP with
the full L=28 model.

> **2026-05-19 update — block-sparse correctness BUG FOUND AND FIXED.** The
> long-running "block-sparse-causal is itself the deeper bottleneck"
> investigation (see § "Mono blocksparse-causal is itself the deeper bottleneck"
> further down) is **resolved**. The bug was in
> [`compile_sha_blocksparse_causal.cpp::bakeRotaryEmbeddings`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal.cpp)
> (and the split twin):
> the rotary sin/cos LUTs were being pushed as **fp16** under the same param
> name (`model.mllm_max_sin_embedding` / `..._cos_embedding`) that the
> runtime `QDQ_ROPE` helper then `__unsafeSetDType(kUInt16PerTensorAsy)`'d and
> attached the PTQ-calibrated `sin_embedding_input_qdq` scale/zp to. The fp16
> bytes got silently reinterpreted as uint16 with scale ≈ 1/32768, zp = 32768,
> which is nonsense — RoPE-applied K is therefore garbage in mono and split.
> V is unaffected (no RoPE), which is what localised the bug. Fix: rebake to
> uint16 using the PTQ-calibrated scale/zp. After fix, mono L=28 produces
> `"The capital of France is Paris."` exactly like dense, and layer-0 K
> bytes are byte-identical between dense and mono on the same prompt. See
> § "Tap-point bisect (2026-05-19) — root cause" below for the full bisect.

All numbers below are for Qwen3-1.7B (L=28, Hq=16, Hkv=8, D=128, BQ=BK=32,
hidden=2048, intermediate=6144) on SM8650 V79 HTP unless noted.

---

## Tap-point bisect (2026-05-19) — root cause

Added env-gated K and V output-buffer dumps in both
[`PromptProcessor::prefill`](../../mllm/backends/qnn/aot_rt/PromptProcessor.cpp)
(dense) and
[`ShaBlockSparsePromptProcessor::prefill`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.cpp)
(mono blocksparse), gated on `MLLM_DUMP_K_LAYER` + `MLLM_DUMP_K_PATH` /
`MLLM_DUMP_V_PATH`. Layout dumped is the present_key/value output buffer for
that layer (shape `[Hkv, D, BQ]` for K, `[Hkv, BQ, D]` for V, uint8 LPBQ).

Ran both on the same 13-token prompt at start_pos=0 and byte-compared the
layer-0 dumps:

| Tensor | Diff bytes / total | Max abs diff | Mean abs diff |
| --- | --- | --- | --- |
| Layer-0 K (before fix) | 12697 / 32768 = 38.7% | 247 | 2.17 |
| Layer-0 V (before fix) | 0 / 32768 = **0.0%** | 0 | 0 |
| Layer-0 K (after fix)  | 0 / 32768 = **0.0%** | 0 | 0 |
| Layer-0 V (after fix)  | 0 / 32768 = **0.0%** | 0 | 0 |

V being identical and K differing localizes the bug to **RoPE** —
input_layernorm, v_proj, and the entire pre-attention pipeline are
correct. K goes through everything V goes through plus the rotary
multiplication, so the divergence is in the rotary tables.

Inspection of [ptq_lpbq.mllm](../../Qwen3-1.7b-mllm/qwen3_1.7b_ptq_lpbq.mllm):
- `model.mllm_max_sin_embedding` dtype = **kUInt16 (raw uint16 bytes)**, shape `[1, 1024, 128]`.
- `model.sin_embedding_input_qdq.fake_quant.scale` = 3.05e-5 (= 1/32768).
- `model.sin_embedding_input_qdq.fake_quant.zero_point` = 32768.
- First position (sin=0): raw values are all 32768, which dequantizes to (32768-32768)*3.05e-5 = 0.0 ✓.

The runtime `QDQ_ROPE` ([modeling_qwen_qnn_aot_sha.hpp:118](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp#L118))
unconditionally `__unsafeSetDType(kUInt16PerTensorAsy)`'s the rotary tensor
and attaches the calibrated scale/zp. So the param **must** be uint16-quantized
in the same scale/zp convention; raw fp16 bytes in the same slot are silently
reinterpreted as uint16 with scale 1/32768, zp 32768 — for value 0 the fp16
encoding is 0x0000, which dequantizes to (0-32768)*3.05e-5 = -1.0 ❌.

The blocksparse compile drivers' `bakeRotaryEmbeddings` was writing fp16:
```cpp
auto sin_t = mllm::Tensor::fromVector(sin_buf, {1, max_pos, head_dim}, kFloat32).to(kFloat16);
params->push("model.mllm_max_sin_embedding", sin_t.contiguous().setMemType(kParamsNormal)...);
```

Fix: quantize to uint16 using the calibrated scale/zp, with sensible defaults
if the PTQ keys are missing:
```cpp
auto quantize = [](float v, float scale, int32_t zp) -> uint16_t {
  long q = std::lround(v / scale) + zp;
  return (uint16_t)std::clamp<long>(q, 0, 65535);
};
std::vector<uint16_t> sin_buf((size_t)max_pos * head_dim);
// ... fill via quantize(sin(...), sin_scale, sin_zp)
auto sin_t = mllm::Tensor::fromVector(sin_buf, {1, max_pos, head_dim}, kUInt16);
```

After this fix in `compile_sha_blocksparse_causal.cpp` (mono) and
`compile_sha_blocksparse_causal_split.cpp` (split), layer-0 K bytes for the
mono path match dense byte-for-byte, and the model produces coherent text:

```
$ ./mllm-qwen3-aot-sha-blocksparse-runner -m qwen3-lpbq-sha-blocksparse-causal.bin ...
prompt: "Hi"
output: "Hello! I'm Ai, your a language model developed by Alibaba Group. How can I assist you today?"

prompt: "What is the capital of France? Please give a single short answer."
output: "The capital of France is Paris."
```

Split L=4 still outputs `/topics` (model is lobotomized to 4 of 28 layers,
so this is expected — not a bug). Split L=28 hits the V79 PD memory cap
(4.2 GB context vs ~3-4 GB cap even with HtpUnsignedPd); the prior
documented Sq=256 workaround should still apply.

---

## Multi-qb attention-sink bug (2026-05-19) — second root cause

After the rotary fix, mono blocksparse-causal was coherent on short prompts
(1-2 qbs) but a length sweep exposed a hard cliff:

| Prompt tokens | qbs | Last qb_global | Output |
| --- | --- | --- | --- |
| 13 | 1 | 0 | coherent |
| 58 | 2 | 1 | coherent |
| 207 | 7 | 6 | coherent |
| 238 | 8 | 7 | coherent in prefill, degrades in decode as pos crosses 256 |
| 266 | 9 | 8 | **garbage** (`Classe Classe…`) |
| 287 | 9 | 8 | **garbage** (`leş leş…`) |

The cliff is exactly at `qb_global = top_k = 8` — the first qb where the
sliding-window `selectTopKBlocks` stub stopped including **block 0**. Block 0
holds the BOS + first tokens, which act as the **attention sink**
(StreamingLLM, Xiao et al. 2023): the softmax dumps excess probability mass
onto them, and dropping them makes the attention distribution diverge →
token garbage. Both qb=7 and qb=8 select 7 real historical blocks (no padding
difference); the only change is qb=7 keeps block 0 and qb=8 drops it.

**Fix** ([`ShaBlockSparsePromptProcessor::selectTopKBlocks`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.cpp)):
when selection is needed (`qb_global > n_hist`), force-anchor two slots and
randomly sample the rest:
- slot 0 = block 0 (sink) — always.
- slot 1 = block `qb_global-1` (most-recent / sliding-window) — always.
- slots 2..n_hist-1 = `n_hist-2` distinct random blocks from the middle range
  `[1, qb_global-2]`, sampled per-head via a `(qb_global, head)`-seeded
  `mt19937` (reproducible per prompt, spreads coverage over middle context).

For `qb_global <= n_hist` all historical blocks fit, so they're all included
(unchanged). `build_mask` already marks `min(n_hist, qb_global)` slots active,
which matches: for `qb_global > n_hist` all 7 slots carry real blocks; block
order within slots is irrelevant because each cached K carries its own
absolute-position RoPE and the mask treats every active historical slot
identically.

**Validation after fix:**
- 266 & 287-token prompts (9 qbs): fully coherent.
- Needle-in-haystack (265 tok, needle "7492" in block 0): **correctly recalls
  7492** — direct proof the sink block is attended again. Pre-fix it produced
  garbage.
- 440-token prompt (14 qbs, real middle-block dropping): coherent on-topic
  summary.

This is the second of two independent block-sparse correctness bugs fixed on
2026-05-19 (the first was the fp16-vs-uint16 rotary mismatch above). Mono
blocksparse-causal is now coherent across the full prompt-length range tested.

**Known limitation — random-middle selection can't do exact retrieval.** The
sink + last + random-middle policy keeps the model *coherent* on long prompts,
but it's blind to *which* distant block holds a queried fact. Tested on two
needle-in-haystack prompts (run on mono, >256 tokens so middle blocks are
actually dropped):
- `NIAH1_1k.txt` (818 tok, needle `diligent-joke=8090293` at 35% depth) → model
  answered "5" (needle's block usually not sampled).
- `MK2_500.txt` (590 tok, multi-key) → fabricated a plausible-looking number.
Output stays fluent/correct-format — it just can't *retrieve*. Reliable
retrieval needs **top-k-by-attention-score** middle-block selection (score each
historical block by Q·K-summary, keep sink + highest-scoring blocks) instead of
random. That's the "real" block-sparse policy; deferred.

---

## Split-prefill fixed (2026-05-19) — residual-boundary dequant + speed result

After fixing mono, brought the split path up too. Three things were needed:

1. **PD memory cap** → recompile at `--sq 256` (the 57-graph L=28 context at
   Sq=1024 estimates ~4.2 GB, over the ~3-4 GB HtpUnsignedPd cap). At Sq=256 it
   loads. **Consequence:** split prefill currently handles prompts ≤ 256 tokens
   (`ShaBlockSparsePromptProcessorSplit::prefill` asserts `num_tokens ≤ Sq`;
   multi-chunk prefill is still unimplemented).
2. **Sink fix** → ported the sink + last + random-middle `selectTopKBlocks`
   policy from mono into `ShaBlockSparsePromptProcessorSplit` (it had the old
   sliding-window stub).
3. **Residual-boundary dequant bug (the split-specific root cause).** Even on a
   short prompt the split produced garbage. Tap-point bisect (dump layer-N
   K_curr from split, byte-compare vs dense) showed layers 0-1 correct but
   **layer 2 K jumping to 68% diff**. Dumping the fp16 boundary buffers found
   `residual_pre_attn_1` was **inf/NaN** while `attn_output_1` was clean.
   Root cause: `MidChunkModule` passed the previous layer's **uint16** `post()`
   output straight into the next layer's `pre()`, whose `residual =
   hidden_states` passthrough then bound a uint16 quantized tensor into the
   **fp16** boundary buffer — writing the integer codes (0..65535) as fp16,
   ~3000× inflated → fp16 overflow → inf → NaN from layer 2 on. `Chunk0Module`
   never hit this because it explicitly dequantizes (`x = x.to(kFloat16)`)
   before `pre()`.

   Fix in [`Qwen3DecoderSplit::pre`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp):
   quantize the boundary input once via the `input_layernorm_input_qdq` QDQ
   (this also gives the upstream Add a solved quant spec) and emit the residual
   as the **dequantized** form of that value:
   ```cpp
   auto h = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq");  // uint16, calibrated
   auto residual = h.to(kFloat16);   // real fp16 boundary (was: residual = hidden_states)
   h = input_layernorm_(h);
   ```

   **After fix:** `residual_1` is clean (NaN=0, range ±14.5, matching the
   layer-1 input scale 0.000386 → ±13). Layer-2 K diff drops 68% → 1.4%;
   deeper layers show only gradual quantization noise from the boundary
   round-trips (abs_max ≤ 17 at layer 27). Split argmax now matches mono
   (token 9707 on "Hi", logit corr 0.92), and split generates coherent text
   ("How can I assist you today? … Hello!").

### Prefill speed: split vs dense vs mono (Sq=256, L=28, ~247-token prompt)

| Path | Prefill time | tok/s | Notes |
| --- | --- | --- | --- |
| Dense (full attn) | 234,555 µs | 1053 | 8× M=32 dispatches, full causal mask |
| Mono block-sparse | 230,099 µs | 1073 | per-qb M=32 + CPU gather; ≈ dense |
| **Split block-sparse** | **216,118 µs** | **1143** | bulk QKV/O/MLP at M=256, attn per-qb M=32 |

Split is **~12% faster** than dense per 256-token chunk: running the bulk
matmuls once at M=256 amortizes the ~1.7 GB weight read that dense pays 8×
(once per M=32 chunk). The win is modest because Sq is capped at 256 by PD
memory; a larger Sq would amortize the weight read further. Mono block-sparse
gives **no** prefill speedup over dense (every 32-token qb re-reads all
weights), confirming the bulk-at-M=Sq restructuring is what matters.

**Remaining split limitations:** (1) prompts ≤ Sq=256 only — multi-chunk
prefill TBD; (2) Sq capped at 256 by the V79 PD memory cap, bounding the
speedup; (3) deeper-layer quantization noise from boundary round-trips (small,
not corrupting).

---

## PD memory optimization #1 (2026-05-19) — lm_head last-position gather

**Where the PD memory goes** (from the compile log RAM summary; `constSize` =
weights, `graphIOTensorSize` = activations, `spillFillBufferSizes` = HTP scratch;
all summed across the 57 graphs in the context):

| Component | Sq=256 | Sq=1024 | Scales with |
| --- | --- | --- | --- |
| Weights (const) | ~1.6 GB | ~1.6 GB | **fixed** (each layer's weights live in exactly one graph; not duplicated) |
| Activation IO | small | ~0.85 GB | Sq |
| Spill-fill scratch | ~0.08 GB | ~1.1 GB | Sq |
| **PD estimate (load-time)** | ~1.6 GB ✓ | **4.23 GB** ✗ | |

The on-disk bin is ~1.6 GB for *both* Sq (weights dominate, fixed). The
load-time PD estimate explodes because IO + spill-fill scale with Sq and are
**summed per-graph** (the 57 graphs run sequentially but QNN reserves
separately). The single biggest term was the **final/lm_head chunk**: at
Sq=1024 it was ~319 MB IO + ~311 MB spill-fill on the `[1,1,Sq,vocab]` output —
but we only ever sample the **last** token's logits.

**Fix:** gather the last real token's hidden state *before* lm_head so the head
runs at M=1 instead of M=Sq.
[`FinalChunkModule`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp)
now takes a `last_token_index` [1,1] int32 input and does
`h = gather(h, dim=1, last_token_index)` → `[1,1,hidden]` before the view +
lm_head → `[1,1,1,vocab]`. The runner sets `last_token_index = num_tokens-1`
and samples the single row. (Threaded through compile `trace_inputs`, the trace
driver, and `ShaBlockSparsePromptProcessorSplit` I/O + sampling.)

**Result:**
- PD estimate at Sq=1024: **4.23 GB → 3.64 GB** (−590 MB, exactly the lm_head term). Still over the cap, so Sq=1024 alone still doesn't load.
- **Sq=512 now loads** and is coherent ("The capital of France is Paris.").
- Correctness preserved at Sq=256 and Sq=512.

**Speed at the larger chunk** (split always processes the full Sq per chunk, so
its prefill cost is ~constant in actual prompt length):

| Path | ~492-token prefill | tok/s | vs dense |
| --- | --- | --- | --- |
| Dense (8/16× M=32) | 437,889 µs | 1124 | — |
| Split **Sq=256** | 216,118 µs (per 256-chunk) | 1143 | +12% |
| Split **Sq=512** | 293,963 µs (per 512-chunk) | **1674** | **+49%** |

The bigger Sq amortizes each layer's ~weight read over 16 qbs' worth of tokens
(one M=512 chunk-graph pass) instead of dense's 16× M=32 weight reloads — so the
block-sparse prefill win grows with Sq. Getting to Sq=1024 needs the next lever
(shared spill-fill buffer across the 57 graphs, ~1.1 GB summed → ~0.3 GB max).

---

## PD memory optimization #2 (2026-05-19) — shared spill-fill buffer

Memory composition of the Sq=1024 load-time PD estimate (summed over 57 graphs,
after the lm_head fix):

| Component | Summed | Note |
| --- | --- | --- |
| Weights (`constSize`) | 1528 MB | fixed |
| Activation IO (`graphIOTensorSize`) | 538 MB | scales with Sq; backed by runtime shared buffers |
| Spill-fill scratch (`spillFillBufferSizes`) | **821 MB** | per-graph, summed — but max single graph is only **46 MB** |
| (reported PD estimate w/ overhead) | 3471 MB | overflows the V79 PD cap |

The 57 graphs run **strictly sequentially**, so they don't need 57 separate
spill-fill buffers — they can share one. QNN exposes this via context group
registration: [`QNNBackend::loadContext`](../../mllm/backends/qnn/QNNBackend.cpp)
now passes a `QnnHtpContext_CustomConfig_t` with
`QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS`,
`groupRegistration = {firstGroupHandle=0, maxSpillFillBuffer=128 MB}`. All graphs
in the context then share one 128 MB spill-fill buffer (vs 821 MB summed).

**Result:** the Sq=1024 context now **loads** — PD drops to ~2923 MB in use
(was estimated 3471 MB; ~550 MB saved as predicted). Verified no regression:
mono and Sq≤512 split still load and produce "Paris". (128 MB comfortably
covers the 46 MB max single-graph need with headroom for larger Sq.)

**New blocker exposed (the next lever, #3):** Sq=1024 now gets *past* context
creation but fails at **runtime I/O buffer registration**:
```
Failed to register memHandles ... Current PD has ~2922.94 MB in use   (failing on a +4 MB buffer)
```
The runtime (`ShaBlockSparsePromptProcessorSplit`) registers **L=28 separate**
copies of each boundary buffer (`residual_full_`, `attn_output_full_`,
`q_full_`, `k_curr_full_`, `v_curr_full_`, `K/V_arranged_`) as QNN shared
buffers — ~430 MB at Sq=1024 — which pushes past the ~2.93 GB PD cap by a hair.
But the dataflow only needs a few live at once:
- `q/k_curr/v_curr_full_[i]` are produced by chunk_i and consumed by attn_i in
  the **same** layer iteration → **1× each** suffices (reuse across layers).
- `residual_full_` / `attn_output_full_` cross exactly one layer boundary
  (chunk_i → chunk_{i+1}) → **2× ping-pong** each.

Collapsing 28× → ~1-2× would free ~350-400 MB, comfortably fitting Sq=1024.
That's optimization #3 (runtime boundary-buffer reuse), deferred.

---

## PD memory optimization #3 (2026-05-19) — runtime boundary-buffer reuse

The runtime allocated **L=28 distinct copies** of each full-Sq boundary buffer,
but the strictly-sequential chunk dataflow only needs a handful live at once:

| Buffer | Old | New | Live range |
| --- | --- | --- | --- |
| `q_full_` / `k_curr_full_` / `v_curr_full_` | 28× | **1×** | produced by chunk_i, consumed by attn_i same iteration |
| `attn_output_full_` | 28× | **1×** | written by attn_i, read by chunk_{i+1}; never read+written at once |
| `residual_full_` | 28× | **2× ping-pong** | chunk_{i+1} reads R[i] and writes R[i+1] in one graphExecute (read-write hazard) → needs 2 |

Implemented in [`ShaBlockSparsePromptProcessorSplit::init_io`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp):
allocate the reduced set; wire `residual_full_[i%2]` for the ping-pong and
`[0]` for the singles in `chunk_in_/chunk_out_`; point the staging/copy/dump
helpers at the single buffers. Safe because (a) chunks run sequentially and
(b) [`QNNAllocator::registerQnnTensorToSharedBuffer`](../../mllm/backends/qnn/QNNAllocator.cpp#L84)
dedupes by pointer, so one buffer bound to many graph tensors = one rpcmem
registration. Frees ~390 MB at Sq=1024 (from ~410 MB → ~22 MB of boundary
buffers).

**Result: Sq=1024 now loads, runs, and is coherent** — "The capital of France
is Paris." End of the PD-memory road for L=28: weights (1.5 GB) + shared
spill-fill (≤128 MB) + reduced runtime buffers + graph IO now fit under the
~2.93 GB V79 PD cap.

### Final prefill-speed scaling (L=28, vs dense ~1159 t/s plateau)

Split's prefill cost is ~constant in actual prompt length (it always processes
the full Sq); throughput rises with Sq as each layer's weight read amortizes
over more tokens:

| Sq | Full-chunk prefill | effective tok/s | vs dense |
| --- | --- | --- | --- |
| 256 | ~216 ms | ~1185 | +2% |
| 512 | ~294 ms | ~1741 | +50% |
| 1024 | ~530 ms | ~1932 | **+67%** |

(A real prompt of N tokens uses the smallest Sq ≥ N; e.g. a 932-token prompt at
Sq=1024 prefills in 535 ms = 1741 t/s vs dense's ~804 ms — ~1.5× faster.)

Three PD-memory optimizations together (lm_head M=1 gather, shared spill-fill,
boundary-buffer reuse) lifted the usable chunk size 256 → 1024 (4×) and the
prefill speedup +12% → +67% over dense. Pushing Sq further would need
multi-chunk prefill (to also handle prompts > Sq) and/or splitting weights
across contexts.

---

## Comprehensive prefill benchmark — dense vs mono vs split (2026-05-19)

Full head-to-head on one device (SM8750 / V79, QAIRT 2.43, Qwen3-1.7B W4A16
LPBQ, L=28). Each runner measured at a sweep of prompt lengths; split measured
at each compiled Sq. All times are the **prefill** phase only.

> **Methodology note.** Dense uses `mllm-qwen3-aot-runner --ar_len 32` (ar_len=32
> chunks); mono uses `mllm-qwen3-aot-sha-blocksparse-runner` (per-qb BQ=32);
> split uses `mllm-qwen3-aot-sha-blocksparse-split-runner --sq {256,512,1024}`
> with `MLLM_QNN_SPILLFILL_MB=128` exported (required for Sq=1024 to load; see
> the regression note at the end). Filler prompts of increasing length;
> `tokens` is the tokenized prompt length. Split requires `tokens ≤ Sq`.

### Raw prefill time (µs) by actual prompt length

| Tokens | Dense | Mono sync | Split Sq256 | Split Sq512 | Split Sq1024 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30  | 54,367  | 50,644  | 146,654 | 255,513 | 526,455 |
| 127 | 124,162 | 120,603 | 146,819 | —       | —       |
| 247 | 241,078 | 234,894 | **147,673** | 257,790 | —   |
| 492 | 448,802 | 444,689 | n/a     | **255,791** | 530,205 |
| 733 | 620,713 | 653,821 | n/a     | n/a     | —       |
| 973 | 833,417 | 891,173 | n/a     | n/a     | **531,749** |

(n/a = prompt exceeds Sq.)

### Findings

1. **Dense ≈ mono sync** — within ~3% at every length (both ~1050–1180 tok/s,
   scaling linearly with prompt length). The mono per-qb block-sparse path gives
   **no** prefill speedup: every 32-token qb still re-reads all ~1.5 GB of
   weights, exactly like dense's ar_len=32 chunks.

2. **Split prefill cost is ~constant per Sq** (it always processes the full
   padded chunk regardless of actual prompt length):
   - Sq=256 ≈ **147 ms**, Sq=512 ≈ **256 ms**, Sq=1024 ≈ **530 ms**.

3. **Effective throughput** (full-chunk tokens ÷ constant time):

   | | Dense plateau | Split Sq256 | Split Sq512 | Split Sq1024 |
   | --- | ---: | ---: | ---: | ---: |
   | tok/s | ~1100 | ~1734 | **~2000** | ~1932 |

4. **Speedup at matched (near-full-chunk) prompt length** — pick the smallest
   Sq ≥ N:

   | Prompt N | Dense | Best split | Speedup |
   | ---: | ---: | ---: | ---: |
   | ~247 | 241 ms | Sq256: 148 ms | **1.63×** |
   | ~492 | 449 ms | Sq512: 256 ms | **1.75×** |
   | ~973 | 833 ms | Sq1024: 532 ms | **1.57×** |

**Caveat — chunk underfill.** Split pays the full Sq cost even for short
prompts (a 30-token prompt at Sq=256 takes 147 ms vs dense's 54 ms). The win
only materializes when the prompt fills most of the chunk, so in practice pick
the smallest Sq ≥ prompt length. At the 256/512/1024 granularity split is
**~1.6–1.75× faster** than dense/mono on chunk-filling prompts.

**Bottom line.** Of the three prefill architectures, only **split** beats dense:
it runs the bulk QKV/O/MLP matmuls once per chunk at M=Sq (amortizing the weight
read over the whole sequence) and only the cheap attention per-qb at M=BQ=32.
Mono sync matches dense; split delivers ~1.6–1.75×. (Device-to-device absolute
times vary — this device is faster than the a615391a unit measured in the
optimization #1–#3 sections above; the *ratios* are what carry over.)

### Regression fixed: spill-fill context config must be opt-in

Optimization #2 originally applied the `QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS`
(shared spill-fill) config unconditionally in `QNNBackend::loadContext`. That
**crashes the dense and mono runners at `initQnnBackend`** (few-graph contexts
don't tolerate the group registration). It's now gated behind the env var
**`MLLM_QNN_SPILLFILL_MB`** (megabytes; unset/0 = off, original behaviour). The
split runner exports it (`=128`) to opt in; dense/mono leave it unset and are
unaffected. See [`QNNBackend::loadContext`](../../mllm/backends/qnn/QNNBackend.cpp).

---

## Why split — the M=BQ=32 efficiency cliff

The existing per-qb causal block-sparse model
([modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp))
compiles the **whole** decoder stack (embedding + L × {input_norm, per-head
Q/K/V proj, RoPE, K/V quant, sparse attn, O-proj, MLP, residuals} + final
norm + lm_head) into one QNN graph. The runner dispatches that graph `num_qb
= Sq/BQ` times per prefill. So at Sq=1024 we run **every layer's QKV/O/MLP
matmul 32 times at M=32**, even though only the attention itself benefits
from per-qb sparsity.

The HMX MAC/cycle efficiency at M=32 is ~8.8× worse than at large M
(see [block_sparse_attention.md § "matmul_av is slower per MAC in big-batch sparse"](block_sparse_attention.md#per-op-profile-what-actually-dominates-inside-the-graph)).
Moving QKV/O/MLP to a single full-Sq dispatch per layer recovers that
factor on those matmuls.

The split keeps the attention itself per-qb (where the sparsity lives) and
unifies everything else to M=Sq.

## Structure: 2L+1 graphs in one QNN context

For L=28, the compiled context holds 57 distinct QNN graphs:

| Graph kind | Count | M | Contains |
|---|---:|---:|---|
| `chunk_0` | 1 | Sq | embedding + sin/cos LUT gather + `layer0_pre_attn` |
| `chunk_i` (1..L-1) | L-1 | Sq | `layer_{i-1}_post_attn` + `layer_i_pre_attn` |
| `chunk_L` | 1 | Sq | `layer_{L-1}_post_attn` + final norm + lm_head |
| `attn_i` (0..L-1) | L | BQ | layer i's per-qb attention compute |

Per-layer compute split:

- **pre-attn**: `input_layernorm` → per-head Q/K/V Conv2D → per-head RMSNorm
  + RoPE → uint8 K/V quantize. Output: `Q [1, Hq, Sq, D]`,
  `K_curr [1, Hkv, D, Sq]`, `V_curr [1, Hkv, Sq, D]`.
- **attn (per-qb)**: per-head `matmul(Q, K_full)` → scale → mask+softmax →
  `matmul(P, V_full)`, where K_full/V_full concat historical (gathered) and
  current K/V along the sequence dim. Output: `attn_output [1, Hq, BQ, D]`.
- **post-attn**: head-concat → `o_proj` Conv2D → residual_add →
  `post_attention_layernorm` → MLP (gate/up/silu/down) → residual_add.

Dispatches per prefill at Sq=1024, num_qb=32, L=28:
- Non-attention: L+1 = 29 chunk dispatches
- Attention: L × num_qb = 896 attn dispatches
- **Total: 925 graph dispatches per prefill** (vs 32 for the monolithic
  per-qb path; far more dispatch overhead, but vastly less per-dispatch
  cycle cost because the non-attention work is no longer running 32× at
  M=BQ).

## Why attention isn't shared across layers

Initial sketch had **one** shared attention graph reused across all L
layers (attention is weight-free — just MatMul + Softmax). But the
per-head QDQ scale/zp constants are baked **per layer** in QNN AOT
(`model.layers.{i}.self_attn.attn_value_matmul_output_qdq_h{h}.fake_quant.*`,
likewise for `qk_matmul_output_qdq_h{h}`, `softmax_output_qdq_h{h}`, etc.).
Sharing one attention graph would force all L layers to use a single
layer's calibration, which is lossy.

So we trace L attention graphs that are **structurally identical** but each
carries its own layer's QDQ constants. Same C++ class
(`Qwen3AttnSplit`) instantiated L times.

## Module hierarchy

Mirrors the monolithic variant's naming so PTQ weight files load unchanged:

```
Qwen3ForCausalLM_SHABlockSparseCausalSplit       (root, no module name)
├── lm_head                  Conv2D
└── llm: Qwen3TextSplit      ("model")
    ├── embedding            Embedding
    ├── rope_sin / rope_cos  Param
    ├── decoders_[i]: Qwen3DecoderSplit          ("layers.{i}")
    │   ├── input_layernorm  RMSNorm
    │   ├── self_attn: Qwen3AttnSplit            ("self_attn")
    │   │   ├── q_projs_[h] / k_projs_[h] / v_projs_[h]   Conv2D
    │   │   ├── q_norm.{h} / k_norm.{h}                   RMSNorm
    │   │   └── o_proj                                    Conv2D
    │   ├── post_attention_layernorm  RMSNorm
    │   └── mlp: Qwen3MLP             ("mlp")
    │       └── gate_proj / up_proj / down_proj / act
    └── norm                 RMSNorm
```

`Qwen3DecoderSplit` and `Qwen3AttnSplit` expose **`pre()` / `attn()` /
`post()` member functions** instead of one `forward()`. Their `forward()`
asserts — they should never be called via `operator()`. The split happens
when the top-level CausalLM's `trace()` calls these methods from each
chunk's body.

## The chunk-wrapper requirement: `CallGraphOp` only comes from `operator()`

The trace machinery — specifically `Module::__trace` at
[mllm/nn/Module.cpp:144](../../mllm/nn/Module.cpp#L144) — creates a
`CallGraphOp` wrapping the forward's ops in a `SubGraphOp` region.
**Plain member-function calls do not trigger this.** Calling
`decoders_[i].pre(...)` directly emits IR ops at the top level of the
module with no `CallGraphOp` wrapping them. `MarkTensorIOPass` then walks
for a `CallGraphOp`, finds none (or fails an assertion), and the lowering
crashes.

Fix: tiny **chunk-wrapper Modules** whose `forward()` delegates to the
underlying decoder's pre/attn/post:

```
Chunk0Module       embedding + sin/cos + decoders_[0].pre()
MidChunkModule     decoders_[i].post() + decoders_[i+1].pre()
FinalChunkModule   decoders_[L-1].post() + norm + lm_head
AttnChunkModule    decoders_[i].attn()
```

Each chunk wrapper is instantiated inline in the trace, invoked via
`operator()`, which produces exactly one `CallGraphOp` + one `SubGraphOp`
per chunk. The wrappers hold no weights — `ptq::QDQ(this, ...)` inside
pre/attn/post resolves against the underlying decoder/attn module's name
(passed as the `this` pointer), so PTQ key lookup is unaffected by the
wrapper hierarchy.

**Gotcha**: composite Modules called inside a chunk wrapper (`mlp_(h)`)
will ALSO create a nested `CallGraphOp`. The top-level lowering's
subgraph walk doesn't capture nested subgraphs and the call dangles,
crashing QNN's HTP graph prep. Fix: call `mlp_.forward({h}, {})` directly
instead of via `operator()`. Single-op composite modules (RMSNorm, Conv2D)
are inlined by the trace dispatcher and don't need this workaround. See
[modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp:Qwen3DecoderSplit::post()](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp)
for the inline call site.

## Chunk-boundary tensors

Per layer i (0..L-1), the trace_inputs map declares 5 boundary tensors
plus the runner-gathered historicals:

| Tensor | Shape | Dtype | Producer | Consumer |
|---|---|---|---|---|
| `residual_pre_attn_{i}` | `[1, Sq, hidden]` | uint16 asym | chunk_i (residual lane) | chunk_{i+1} (residual_add input) |
| `q_{i}` | `[1, Hq, Sq, D]` | uint16 asym | chunk_i (per-head Q concat) | attn_i (via per-qb stage) |
| `k_curr_{i}` | `[1, Hkv, D, Sq]` | uint8 sym | chunk_i (K quant, transposed) | attn_i (via per-qb stage) + KV cache |
| `v_curr_{i}` | `[1, Hkv, Sq, D]` | uint8 sym | chunk_i (V quant) | attn_i (via per-qb stage) + KV cache |
| `attn_output_{i}` | `[1, Hq, Sq, D]` | uint16 asym | L attn_i dispatches (stitched) | chunk_{i+1} (O-proj input) |

Plus globally:
- `input_ids` / `position_ids`: `[1, Sq]` int32
- `mask`: `[1, 1, BQ, kTopKBK]` uint16 (shared across every attn_i in a qb)
- `K_arranged_{i}` / `V_arranged_{i}`: runner-gathered, same shape as
  monolithic variant

## Boundary QDQ aliases (most likely correctness culprit)

Every chunk-boundary tensor needs declared scale/zp metadata — QNN has no
implicit cross-graph requant. Most map cleanly to an existing PTQ key:

| Boundary | PTQ source (existing scale) |
|---|---|
| `residual_pre_attn_{i}` (i ≥ 1) | `model.layers.{i}.input_layernorm_input_qdq.fake_quant.*` |
| `q_{i}` / `q_{i}_qb` | `model.layers.{i}.self_attn.q_rope_add_0_output_qdq_h0.fake_quant.*` |
| `k_curr_{i}` / `k_curr_{i}_qb` | `model.layers.{i}.self_attn.k_cast_to_int8_qdq.fake_quant.*` |
| `v_curr_{i}` / `v_curr_{i}_qb` | `model.layers.{i}.self_attn.v_cast_to_int8_qdq.fake_quant.*` |
| `K_arranged_{i}` / `V_arranged_{i}` | same as k_curr / v_curr |

Two boundaries had **no direct PTQ analogue** because the monolithic graph
handled them via implicit cross-op requant. These got new keys, aliased
from a "semantically nearby" existing key:

| New key | Aliased from | Where applied |
|---|---|---|
| `embed_tokens_output_qdq` | `model.layers.1.input_layernorm_input_qdq` | Explicit `QDQ(x, "embed_tokens_output_qdq")` at the end of chunk_0's embedding (declares `residual_pre_attn_0`'s scale) |
| `model.layers.{i}.self_attn.attn_output_boundary_qdq` | `model.layers.{i}.self_attn.attn_value_matmul_output_qdq_h0` | Explicit `QDQ(y, "attn_output_boundary_qdq")` at end of `Qwen3AttnSplit::attn()` (declares `attn_output_{i}`'s scale across the per-head concat) |

These aliases are **best-guess approximations**, not properly calibrated
scales. The embedding output's range is not exactly layer 1's input range;
head 0's attn_value range isn't necessarily representative of all 16 heads'
concatenated range. Over 28 layers, accumulated requant error from these
mis-aliased scales is the most likely cause of the current output
degeneracy (see "Current correctness state" below).

A proper fix needs a calibration pass: run an fp16 (no quant) trace on a
few prompts, observe per-boundary activation statistics, set scales
accordingly.

## AOT pipeline patches

The standard AOT pipeline at
[mllm/backends/qnn/aot/passes/AOTPipeline.cpp](../../mllm/backends/qnn/aot/passes/AOTPipeline.cpp)
assumes a **single graph named `"model"`** that gets internally split into
per-layer subgraphs by `SplitLLMGraphPass`. Our pre-split structure broke
three passes; the fix is a config-driven opt-in.

### `chunk_graph_name` config flag

Added to the AOT compile context. When set (non-empty string), passes
take the **split path**:

- [SplitLLMGraphPass.cpp](../../mllm/backends/qnn/aot/passes/SplitLLMGraphPass.cpp):
  early-return without splitting. **Also** attaches `use_qnn` attribute to
  the chunk's SubGraphOp and `qnn_graph_name` + `qnn_context_name` StrAttrs
  to every linalg op inside (these are normally set as part of the
  split-and-rename; `LLM2QnnLoweringPass`'s op walk requires them).
- [MergeLLMHeadIntoMainGraphPass.cpp](../../mllm/backends/qnn/aot/passes/MergeLLMHeadIntoMainGraphPass.cpp):
  early-return (lm_head already lives in `chunk_L`'s subgraph).
- [LLM2QnnLoweringPass.cpp](../../mllm/backends/qnn/aot/passes/LLM2QnnLoweringPass.cpp):
  skip the `"model"` name check and the `^model(\.\d+\.s\d+)?$` subgraph
  regex validation; directly look up and capture the chunk subgraph by
  name.

The compile driver mutates `chunk_graph_name` in `AOTCompileContext`'s
in-memory config between `pm.run()` calls. `createQnnAOTLoweringPipeline`
re-reads the config file on each call, so the mutation must happen
**after** pipeline creation and **before** `pm.run()`.

## Compile flow

```text
[load PTQ params]                     mllm::load(model_path, ModelFileVersion::kV2)
[SHA prep]                            prepareParametersForSHA (slice MHA, copy QDQ per head)
[rotary bake]                         bakeRotaryEmbeddings (sin/cos LUT at max_cache_length × D)
[boundary QDQ aliases]                aliasQdqParam for embed_tokens_output_qdq + attn_output_boundary_qdq
[mask + constant_zero QDQ]            push mask.scale=0.001/65535, zp=65535 etc.
[construct model + load]              Qwen3ForCausalLM_SHABlockSparseCausalSplit(cfg).load(params)
[QnnAOTEnv setup]                     parseQcomTargetMachineFromJSONFile(aot_cfg)
[build trace_inputs map ~283 entries] input_ids, position_ids, mask, K/V_arranged_i, residual/q/k/v_curr/attn_output_i, _qb variants
[model.trace()]                       returns IROutput {chunk_0, attn_0, chunk_1, attn_1, ..., chunk_L}  (2L+1 IRs)
[loop chunks in chunk_order]
  pm = PassManager(ir[name])
  pm.reg(createQnnAOTLoweringPipeline(env, aot_cfg_path, params))   // re-reads config file
  cfg["graph_on_qnn"] = [name]                                        // override AFTER pipeline build
  cfg["chunk_graph_name"] = name                                      // override AFTER pipeline build
  pm.run()
[saveContext("context.0", "*.bin")]
[_exit(0)]                            // bypass exit-time dtor crash (see below)
```

Chunk order:

```
chunk_0, attn_0, chunk_1, attn_1, ..., chunk_{L-1}, attn_{L-1}, chunk_L
```

(Order matters: attn_i's K_arranged / V_arranged scales reference layer i's
existing PTQ params; chunks share the qnn_aot_env's context.0 so weights
aren't duplicated across the binary.)

## Runtime: `ShaBlockSparsePromptProcessorSplit`

State:

- 57 `QnnAOTModule` instances (`chunk_0`..`chunk_L` + `attn_0`..`attn_{L-1}`)
- L per-layer full-Sq chunk-boundary buffers (residual / q / k_curr /
  v_curr / attn_output), allocated once in rpcmem.
- L per-layer per-qb staging buffers (q_qb / k_curr_qb / v_curr_qb /
  attn_output_qb), small (~256 KB each), allocated once. **Per-layer**
  rather than shared across all attentions — sharing triggered
  rename-warning churn and was suspected of allocator aliasing.
- L per-layer K_arranged / V_arranged (runner-gathered).
- Shared mask buffer + shared logits buffer.

Prefill flow:

```text
Fill input_ids + position_ids at full Sq (pad past prompt with token 0)

Dispatch chunk_0(input_ids, position_ids)
    → residual_full_[0], q_full_[0], k_curr_full_[0], v_curr_full_[0]

For each layer i in [0, L):
    copy_kv_to_cache(i, base_pos=0, n_tokens=num_tokens)
        // critical: must run BEFORE the per-qb loop so per-qb gathers
        // see freshly-computed K/V at positions 0..qb*BQ-1 instead of
        // stale cache bytes. Discovered the hard way; original ordering
        // (cache write AFTER qb loop) gave the qb=1+ gather garbage.

    For each qb in [0, num_qb):
        CPU: build mask, gather K_hist/V_hist into K_arranged_i / V_arranged_i
        CPU: strided memcpy q_full_[i] / k_curr_full_[i] / v_curr_full_[i]
             qb-slice → q_qb_[i] / k_curr_qb_[i] / v_curr_qb_[i]
        NPU: dispatch attn_i with the L=6 input tensors
        CPU: strided memcpy attn_output_qb_[i] back into
             attn_output_full_[i] at qb offset

    Dispatch chunk_{i+1}(residual_full_[i], attn_output_full_[i], position_ids)
        → residual_full_[i+1], q_full_[i+1], k_curr_full_[i+1], v_curr_full_[i+1]
        (or → logits if i+1 == L)

Greedy sample from logits[num_tokens - 1, :]
```

Per-qb staging via strided memcpy because Sq isn't a contiguous slice
dim for the chunk-boundary layouts. For `q [1, Hq, Sq, D]`: each head's
qb-slice is `BQ × D` contiguous bytes, but head h+1's slice starts at
offset `(h+1) × Sq × D` from the buffer start. Per qb: Hq memcpys of
`BQ × D × sizeof(uint16)`. For k_curr `[1, Hkv, D, Sq]`: per (head, c row):
BQ contiguous bytes (one head row's qb-slice). Hkv × D memcpys of BQ
bytes. v_curr is contiguous like q. Total ~280 MB of CPU memcpy per
prefill at Sq=1024, L=28, num_qb=32, ≈ 12 ms at 23 GB/s. Acceptable;
small compared to NPU compute.

## HTP PD memory cap blocker (and the Sq=256 workaround)

At **L=28 + Sq=1024**, the QNN HTP PD memory size estimate is **~4.2 GB**:

| Component | Estimated contribution at Sq=1024 |
|---|---:|
| Weights (1.6 GB on disk, ~unchanged at runtime) | 1.6 GB |
| Per-chunk full-Sq MLP intermediates (gate/up/silu/down at intermediate=6144) | ~1.7 GB (29 chunks × ~5 stages × 12 MB) |
| Per-chunk full-Sq I/O tensors (residual/q/k_curr/v_curr/attn_output) | ~0.4 GB |
| Logits at Sq=1024 vocab=151936 fp16 | 311 MB |
| Per-graph QNN overhead + VTCM reservation | ~0.5 GB |
| **Total runtime PD estimate** | **~4.2 GB** |

`HtpSignedPd` caps around 1.5-2 GB on SM8650 V79. `HtpUnsignedPd` caps
around 3-4 GB. Both reject our 4.2 GB context (`QnnDsp <E> Failed to
find available PD for contextId 1 ... context size estimate 4226106368`).

**Sq=256 workaround**: at Sq=256 the per-chunk intermediates shrink 4×
linearly, dropping the PD estimate to ~2.7 GB → fits `HtpUnsignedPd`.
Architectural HMX-efficiency win at QKV/O/MLP is preserved because
M=256 is still well above the M=32 cliff. The trade is that single-pass
prefill is capped at 256 tokens; longer prompts would need multi-pass
prefill (not currently implemented).

Required AOT config change for L=28:
[qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split.json](../../examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split.json)
sets `"htp_security_pd_session": "HtpUnsignedPd"`.

For future L=28 at full Sq=1024: multi-context split (chunks 0-13 +
attn 0-13 in one context, 14-28 + attn 14-27 in another), each fitting
~2 GB. Requires patching `LLM2QnnLoweringPass`'s
`MLLM_RT_ASSERT_EQ(split_graph, 1)` and the runtime QNNBackend to
support multiple loaded contexts simultaneously. Not yet attempted.

## Exit-time `align_free` double-free (bypassed with `_exit(0)`)

At Sq=1024, the compile driver's `_exit` runs through a destructor chain
that crashes on `align_free` of an already-freed pointer:

```
align_free (mllm CPU allocator)
  ↑
TensorStorage::~TensorStorage  (mem_type_ == kNormal → memoryManager()->free)
  ↑
TensorViewImpl::~
  ↑
QNNTensorWrapper::~  (holds clientBuf.data raw pointer into a TensorStorage)
  ↑
QnnAOTNodeTensor::~  (captured during LLM2QnnLoweringPass)
  ↑
QnnAOTGraph::~  →  QnnDeviceAndContext::~  (~qnn_aot_env)
```

Root cause: shared data-pointer ownership between mllm's CPU-allocated
`TensorStorage` (mem_type=kNormal) and QNN's `QNNTensorWrapper.clientBuf.data`
(raw pointer). The same pointer ends up being freed twice across the two
destructor chains. Sq=256 doesn't trip it (smaller wrapper count and
pointer ownership happens to land in a working order).

Workaround in [compile_sha_blocksparse_causal_split.cpp](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp):
`_exit(0)` after `saveContext` + final print. The `.bin` is already
written; OS reclaims memory on process exit. Proper fix needs to audit
which `TensorStorage` instance the QNN wrappers should NOT free —
likely set `mem_type_ = kManual` on the host-side trace_inputs tensors.

## File index

| File | Purpose |
|---|---|
| [examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp) | Split model: Qwen3AttnSplit (pre/attn/post), Qwen3DecoderSplit, Qwen3TextSplit, chunk-wrapper Modules, CausalLM with multi-chunk trace |
| [examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp) | Compile driver: builds trace_inputs map, runs per-chunk lowering with `chunk_graph_name` override, saves single context binary |
| [examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split.json](../../examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split.json) | AOT config (HtpUnsignedPd for L=28) |
| [mllm/backends/qnn/aot/passes/SplitLLMGraphPass.cpp](../../mllm/backends/qnn/aot/passes/SplitLLMGraphPass.cpp) | Split-path early-return + use_qnn / qnn_graph_name / qnn_context_name attribute attachment |
| [mllm/backends/qnn/aot/passes/MergeLLMHeadIntoMainGraphPass.cpp](../../mllm/backends/qnn/aot/passes/MergeLLMHeadIntoMainGraphPass.cpp) | Split-path early-return (lm_head already in chunk_L) |
| [mllm/backends/qnn/aot/passes/LLM2QnnLoweringPass.cpp](../../mllm/backends/qnn/aot/passes/LLM2QnnLoweringPass.cpp) | Split-path subgraph capture + relaxed name regex |
| [mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.hpp](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.hpp) | Runtime processor header |
| [mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp) | Runtime: 57-graph orchestration, per-qb staging, KV cache write |
| [examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal_split.cpp](../../examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal_split.cpp) | Runner: prefill-only + decode-by-reprefill diagnostic (`--gen N`) |
| [tests/qnn/GraphSwitchLatencyTest.cpp](../../tests/qnn/GraphSwitchLatencyTest.cpp) | Microbenchmark for graph-switch RPC overhead (sweep N=1..64 graphs in one context) |

## Build & run

Compile (x86 host, AOT):
```bash
cd /path/to/mllm
python task.py tasks/build_x86_qnn_aot.yaml

# Compile split context (use HtpUnsignedPd for L=28). The _split AOT cfg
# enables it; the model cfg is whatever sets num_hidden_layers correctly.
LD_LIBRARY_PATH=/tmp/mllm-qnn-host-libs:$NDK_LIB_DIR:$LD_LIBRARY_PATH \
  ./build-qnn-aot/bin/mllm-qwen3-aot-sha-blocksparse-causal-split-c \
    -m /path/to/qwen3_1.7b_ptq_lpbq.mllm \
    -c ./examples/qwen3_qnn_aot/config_1.7B_w4a16_blocksparse_causal.json \
    -aot_cfg ./examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split.json \
    --sq 256

# Output: qwen3-lpbq-sha-blocksparse-causal-split.bin (~1.6 GB at L=28)
```

Cross-build runtime (Android arm64):
```bash
python task.py tasks/build_android_qnn.yaml
# Output: build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-sha-blocksparse-split-runner
```

Push & run on device (after the standard QNN lib push from `scripts/adb_push.sh`):
```bash
adb push qwen3-lpbq-sha-blocksparse-causal-split.bin /data/local/tmp/
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-sha-blocksparse-split-runner /data/local/tmp/

adb shell 'cd /data/local/tmp && export LD_LIBRARY_PATH=. \
  && export "ADSP_LIBRARY_PATH=.:/data/local/tmp" \
  && echo "your prompt here" \
  | ./mllm-qwen3-aot-sha-blocksparse-split-runner \
      -m qwen3-lpbq-sha-blocksparse-causal-split.bin \
      -t qwen3-tokenizer.json \
      -c config_1.7B_w4a16_blocksparse_causal.json \
      --sq 256 \
      --gen 20'
```

`--gen N` runs the **decode-by-reprefill** diagnostic: append each sampled
token to the prompt, re-run the full prefill, sample the next position.
~Sq× slower than a proper decode but proves whether the model can sustain
a coherent generation.

## Measured latency (architecture only — see correctness caveat below)

L=4 + Sq=1024, 21-token prompt: **167 ms** for full prefill.
L=28 + Sq=256, 197-token prompt: **195 ms** for full prefill.

Extrapolating L=4 at Sq=1024 linearly to L=28 (assuming HMX/HVX scales as
expected): **~1.17 s for a 1024-token Sq=1024 prefill** vs the existing
monolithic per-qb's ~7-10 s at the same shape — projected **6-8× speedup**.
Not yet measured at Sq=1024 + L=28 because the PD memory blocker forces
Sq=256 there. The full Sq=1024 + L=28 number needs multi-context support.

## Current correctness state (open)

**The architecture compiles, loads, and dispatches correctly. Output is
wrong — but the breakage is upstream of the split.**

### Upstream blocker: AOT visitor edits break baseline models

The working tree carries several uncommitted modifications to AOT visitors
that pre-date this split work and break the underlying QNN AOT lowering for
**any** Qwen3 SHA model:

- [`mllm/backends/qnn/aot/visitor/Linear.cpp`](../../mllm/backends/qnn/aot/visitor/Linear.cpp)
  — bias-less Linears (i.e. almost every Q/K/V/O/MLP projection in the model)
  now lower to QNN `MatMul` with `transpose_in1=true` instead of
  `FullyConnected`. Comment in the diff cites "validator rejects 4D output"
  as the motivation. Changes quantization semantics for every linear in the
  model.
- [`mllm/backends/qnn/aot/visitor/RMSNorm.cpp`](../../mllm/backends/qnn/aot/visitor/RMSNorm.cpp)
  — synthesises an fp16 fake bias when weight is fp16 (originally hard-coded
  to uint16 — segfaulted on fp16 path). Changes RMSNorm output handling.
- [`mllm/backends/qnn/aot/visitor/Transpose.cpp`](../../mllm/backends/qnn/aot/visitor/Transpose.cpp),
  [`LLMQuantRecipePass.cpp`](../../mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp),
  [`QNNBackend.cpp`](../../mllm/backends/qnn/QNNBackend.cpp),
  [`QNNModel.cpp`](../../mllm/backends/qnn/QNNModel.cpp) — also have
  uncommitted local edits.

Verified at the end of the May 2026 split session: on a freshly-wiped
SM8650 V79 device with all libs re-pushed:

- `mllm-qwen3-aot-sha-blocksparse-runner` (monolithic per-qb): produces
  Arabic-letter / Cyrillic / Japanese noise for English chat-template prompts.
- `mllm-qwen3-aot-runner` (regular SHA, non-blocksparse): runs prefill OK,
  then `QnnDsp <E> Graph model.0.s1 failed in execution with err 1100` on
  every decode step (each decode dispatch fails, runner emits the same
  fallback token "d" 2031 times).

So the QNN AOT pipeline itself is in a known-broken state on this checkout
— `compile_sha`'s "known-good" baseline status no longer holds with the
current visitor edits applied. Until those edits are reverted or fixed, the
split's correctness can't be assessed independently.

### Split-specific behaviour on top of the upstream breakage

What works:
- 57 graphs load from the 1.6 GB context binary with correct
  per-graph tensor metadata (verified via `QNNModel::loadGraphTensorInfo`
  log)
- All 57 graphs dispatch without QNN errors
- Logits buffer fills with **real, prompt-dependent data** (verified via
  diagnostic hash of `residual_full_[0]` after chunk_0 — different prompts
  produce different bytes)
- The chunk-boundary buffer aliasing flows data correctly (chunk_i's
  output and chunk_{i+1}'s input bind to the same rpcmem memhandle)

What's broken:
- Greedy sampling produces **token 271 = `\n\n`** at every step regardless
  of prompt content. The top-5 logits across drastically different prompts
  all rank the same 5 whitespace/structural tokens (`\n\n`, ` `, ` "`, `.`,
  ` \n\n`). No content tokens like ` Paris` (12095) or ` The` (576) appear
  in the top-5 for "The capital of France is".

**Comparison with monolithic baseline**:

Running the same `qwen3-lpbq-sha-blocksparse-causal.bin` model file via the
existing `mllm-qwen3-aot-sha-blocksparse-runner` (the monolithic per-qb
path) **also produces garbage** on chat-template prompts — Arabic-letter
noise and whitespace, not coherent English. So the baseline blocksparse-causal
pipeline itself appears to be in a broken state independent of our split.

The split inherits the underlying breakage and amplifies it: where
monolithic produces *varied* garbage tokens, split produces the *same*
token. This is the additional split-specific issue.

Two diagnostic avenues open:

1. **Verify the monolithic blocksparse-causal pipeline still produces
   coherent text on a recent known-good test.** The user recalls it has
   worked. If a rerun confirms, we have a clear bisect target. If not,
   the pipeline regression is the upstream blocker and split correctness
   work is moot until it's fixed.

2. **Replace alias-based boundary QDQs with calibrated scales.** Run a
   pure-fp16 trace on a few calibration prompts, observe per-boundary
   activation statistics, set scales (and zero-points for asym) accordingly.
   This addresses the split-specific extra error layer regardless of the
   monolithic-pipeline status.

## Things investigated and ruled out

| Hypothesis | Verdict |
|---|---|
| Per-qb staging tensor rename warnings → real binding bug | **Ruled out.** QNN's `QNNBackend::graphExecute` binds inputs/outputs by **position** in the vector, not by tensor name. The runtime tensor's name doesn't affect dispatch. Verified by removing all renames + allocating per-layer staging (no shared buffer) — output unchanged. |
| Shared per-qb staging across layers → allocator aliasing | **Ruled out.** Switched to per-layer staging (one buffer per layer per kind). Output unchanged. |
| KV cache write happens after per-qb loop, so qb≥1 gather reads stale cache | **Real bug** in original ordering; fix moved cache write before the per-qb loop. Output didn't change qualitatively (still 271 every step), but the fix is correct on its own merits. |
| Chunk-boundary buffer aliasing not flowing between chunks | **Ruled out.** Verified via `residual_full_[0]` byte hash that data flows between chunks. |
| Shape mismatch (logits Tensor [1,Sq,vocab] vs compiled [1,1,Sq,vocab]) | **Ruled out.** QNN's positional binding takes the wrapper's shape (from binary) and the runtime tensor's data pointer; shape mismatch in the runtime tensor doesn't affect dispatch. |
| Token 271 is just a chat-template artifact (model genuinely thinks `\n\n` comes first) | **Partially yes** (271 = `\n\n` IS the chat-template's response-start token), **but** the reprefill loop continues to sample 271 indefinitely, which a healthy model wouldn't do. So 271 is the surface symptom; the underlying issue is that prompt content isn't propagating through the 28 layers strongly enough to disambiguate. |

## Open / cleanup items

- **Numerical correctness** (above)
- **Decode for the split path**. Current decode-by-reprefill is functional
  but ~Sq× slower than ideal. A proper decode needs either (a) a separate
  ar_len=1 split context, or (b) fall back to the monolithic per-qb decode
  graph after the split prefill writes the KV cache.
- **Exit-time `align_free` dtor crash** — currently bypassed with
  `_exit(0)`. Proper fix needs auditing `QnnAOTNodeTensor` /
  `QNNTensorWrapper` / `wrapTensors2TensorIR` ownership semantics.
- **L=28 at full Sq=1024** — requires multi-context support: split
  chunks into multiple QNN context binaries (each ≤ ~2 GB), patch
  `LLM2QnnLoweringPass`'s `MLLM_RT_ASSERT_EQ(split_graph, 1)` and the
  runtime QNNBackend to maintain multiple loaded contexts simultaneously.
- **Per-axis quant for `attn_output_boundary_qdq`** instead of per-tensor
  alias-from-head-0. Preserves each head's dynamic range; should
  meaningfully reduce per-layer requant error.
- **Graph-switch latency confirmation**. Microbenchmark exists; numbers
  not collected yet. Worth bounding the 925-dispatch-per-prefill RPC
  overhead so future tuning has a budget target.

## 2026-05-18 update — saturation diagnosis + fp16 boundary direction

This session resolved the earlier "always sample token 271" mystery and
narrowed the remaining correctness issue to the boundary QDQ scales.

### V79 skel — was masking everything

Earlier sessions saw both monolithic and split produce `wapwapnpwap...` /
err 1100 / `crc32 failed in cpu`. Root cause was the
`/data/local/tmp/libQnnHtpV79Skel.so` we'd been pushing — the
`/vendor/lib64/hw/audio/libQnnHtpV79Skel.so` audio variant had a CRC32
mismatch against QAIRT 2.43's `libQnnHtpPrepare.so`. The DSP rejected
the skel before any inference happened, but the host runner's `MLLM_INFO`
output was block-buffered behind `adb shell`'s pipe and lost on
`abort()`, so we never saw the failure chain.

Fixes:

- **Skel**: push
  `/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so`
  (10588380 bytes, dated 2026-01-30) to `/data/local/tmp/`. Verified
  monolithic SHA L=28 produces coherent text again.
- **Runner buffering**: added `std::setvbuf(stdout, nullptr, _IONBF, 0);`
  to [`examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal_split.cpp`](../../examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal_split.cpp)
  so MLLM logs flush before any subsequent abort. Keep this — it's
  cheap and saved hours of guessing.
- **g++**: reverted `-DCMAKE_CXX_COMPILER=/usr/bin/g++-15` pin in
  [`tasks/build_x86_qnn_aot.yaml`](../../tasks/build_x86_qnn_aot.yaml).
  g++-15 breaks `mllm/ffi/vendors/tvm-ffi/src/ffi/extra/module.cc`
  (recursive `is_constructible_v` constraint in libstdc++ 15's
  `std::optional`). g++-13 (system default) compiles cleanly and
  produces identical-output bins.

### Logits comparison localised the bug

With the skel fixed, mono L=28 produces coherent text. Split L=28 still
samples token 271 every step. Dumping the lm-head logits at the last
prompt position (MLLM_DUMP_LOGITS env var added to both
`ShaBlockSparsePromptProcessor::prefill` and
`ShaBlockSparsePromptProcessorSplit::prefill`) showed:

| variant | min | max | mean | argmax | top-5 indices |
|---|---|---|---|---|---|
| mono L=4   |  7009 | 35282 | 21426 | 61886 | 61886, 354, 99632, 76565, 68699 |
| split L=28 (no fix) | 2266 | **63968** | 17011 | 271 | 271, 198, 319, 2146, 4710 |
| split L=28 + 4× boundary scales | 1309 | 32981 | 17706 | 713 | 713, 151643, 8908, 5, 271 |
| split L=28 + 16× boundary scales | 4624 | 33755 | 19683 | 38207 | 38207, …, all ~33500 |

Key findings:

1. **Without fix**: split logits hit uint16 max (63968 ≈ 65535).
   Saturation collapses argmax to whichever index saturated first → 271
   wins by lexicographic position. Argmax is meaningless.
2. **4× boundary headroom**: saturation gone. Different argmax each
   compile but top-1 still wins by tiny margin (382 in uint16 units).
   Output is degenerate-but-different ("ich", "warfare", "驷" depending
   on the exact alias choice).
3. **16× headroom**: even smaller top1−top2 gap (176) → worse
   precision per step. Scale tuning alone won't fix this.

### Why scale tuning isn't enough

Each chunk boundary applies `ptq::QDQ(this, y, "..._boundary_qdq")`,
which is metadata attach + uint16-round-trip through QNN's graph
boundary. Mono has **zero** boundary requantizes; per-op QDQs inside the
single graph are tuned per-op via PTQ calibration. Split has
**2L+1 = 57** graph crossings, each adding a requantize whose scale
came from a borrowed analogue (head 0's `attn_value_matmul_output_qdq`
for `attn_output_boundary_qdq`; layer 1's `input_layernorm_input_qdq`
for `embed_tokens_output_qdq`). Cumulative noise flattens the lm-head
distribution into ambiguity.

### Recommended next step — fp16 boundaries

Drop the boundary QDQ entirely. Make chunk-output tensors `kFloat16`
and chunk-input tensors `kFloat16`. fp16's 5-bit exponent easily covers
activation ranges in transformers, and the precision loss across 57
boundaries is negligible compared to uint16's 16-bit-per-channel error.

Concrete edit points:

- [`examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp):
  replace each `y = ptq::QDQ(this, y, "..._boundary_qdq")` with an
  explicit `CastTo(kFloat16)` (or framework-equivalent), and remove
  `hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq")`
  at the start of each chunk's `pre()` since the input is now fp16.
- [`examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp):
  drop the `aliasQdqParam` calls for `embed_tokens_output_qdq` and
  `attn_output_boundary_qdq`. Change the boundary entries in
  `trace_inputs` to `kFloat16` instead of `kUInt16PerTensorAsy`.
- [`mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp):
  switch `residual_full_`, `attn_output_full_`, the per-qb `attn_output_qb_`
  to `kFloat16`. Staging memcpys are unchanged byte-for-byte
  (uint16 → fp16 is still 2 bytes/elem).
- The chunk_L logit dump path stays uint16 (lm_head's output is
  unchanged).

The `kBoundaryScaleMul = 1.f` line currently in the compile driver is a
hook for the failed scale-tuning approach — remove it together with the
aliasing once the fp16 path lands.

### What's still in the working tree from today

- `MLLM_DUMP_LOGITS` env-gated logit dumper in both
  `Sha*PromptProcessor.cpp` (keep — useful diagnostic)
- `setvbuf(_IONBF)` in `aot_run_sha_blocksparse_causal_split.cpp`
  (keep — saves the next debugging session hours)
- `aliasQdqParam` block in `compile_sha_blocksparse_causal_split.cpp`
  reading `max_h` across heads + `kBoundaryScaleMul = 1.f` (revert
  when fp16 path replaces it)
- `tasks/build_x86_qnn_aot.yaml` g++-15 pin **reverted** (system default
  g++-13 works correctly)
- `tasks/build_android_qnn.yaml` HexagonMakeTask now targets `htp_v79`
  (was `htp_v75`; required for SM8750 device)

## 2026-05-18 update (later) — fp16 boundary refactor landed

Implemented the fp16-boundary plan from the previous section. Key edits:

- [`examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp):
  - `Qwen3AttnSplit::attn` line 246: `ptq::QDQ(this, y, "attn_output_boundary_qdq")` → `y.to(kFloat16)`. Chunk output is now fp16.
  - `Chunk0Module::forward` line 390: same swap for `embed_tokens_output_qdq`.
  - `Qwen3DecoderSplit::pre`: dropped the `layer_idx_ != 0` guard, every layer now runs `ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq")` — the helper's new kFloat16 case (see below) turns this into a Quantize op at the chunk input.
  - `Qwen3DecoderSplit::post`: explicit `residual = ptq::QDQ(this, residual, "input_layernorm_input_qdq")` at start, so the residual+attn_output add downstream has matching uint16+scale dtypes. Without this QNN's ElementWiseAdd validates with "mismatching datatypes 0x216 != 0x416".
  - `Qwen3AttnSplit::post`: added `attn_output = ptq::QDQ(this, attn_output, "attn_value_matmul_output_qdq_h0")` at start. Without this, PTQPass can't solve o_proj's input quant spec (tensor X "is not solved, produced by Op o_proj").

- [`examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp) (shared `ptq::QDQ` helper):
  - Added `case kFloat16:` to the switch — emits `in.to(kUInt16PerTensorAsy)` (lowered to Quantize via the CastType visitor's "isFloat→isInt" branch) then attaches the named scale/zp. Used by the split modeling's chunk-input QDQs.

- [`examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp):
  - Removed the `aliasQdqParam`/`kBoundaryScaleMul` block for `embed_tokens_output_qdq` and `attn_output_boundary_qdq` — boundary QDQs no longer exist.
  - Added `aliasQdqParam(params, "model.layers.1.input_layernorm_input_qdq", "model.layers.0.input_layernorm_input_qdq")` — PTQ doesn't have layer 0's input_layernorm scale (the original model used embed_tokens_output_qdq instead); the new chunk pre() applies the QDQ uniformly across all layers and needs layer 0's key.
  - `residual_pre_attn_i` and `attn_output_i` in trace_inputs: `kUInt16` → `kFloat16`. Dropped the boundary scale/zp attaches.

- [`mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessorSplit.cpp):
  - `residual_full_[i]`, `attn_output_full_[i]`, `attn_output_qb_[i]`: dtype `kUInt16` → `kFloat16`. Staging memcpys are unchanged (still 2 bytes/elem).

### Outcomes

| L=4 variant | min | max | mean | top1−top2 | argmax | first sampled token |
| --- | --- | --- | --- | --- | --- | --- |
| split + uint16 boundary (Sq=1024) | 2266 | **63968** | 17011 | 1 | 271 | `\n\n` (saturated) |
| split + uint16 boundary + 4× headroom | 7069 | 33241 | 20415 | 1990 | 120003 | `驷` (random) |
| split + **fp16 boundary** (Sq=1024) | 9868 | 38930 | 23010 | **2028** | 57662 | `/topics` |

| L=28 variant (Sq=256, prompt "Hello, my name is") | min | max | mean | top1−top2 | argmax | first sampled token |
| --- | --- | --- | --- | --- | --- | --- |
| split + uint16 boundary (no headroom mul) | 1309 | 32981 | 17706 | 382 | 713 | `ich` |
| split + fp16 boundary | 11999 | 33569 | 23179 | **1** | 24926 | `cplusplus` |
| **mono** SHA-blocksparse-causal (L=28) — direct reference | 3560 | 39183 | 20508 | **862** | 65662 | `userManager` |

L=4 fp16 looks healthy — clean top1, no saturation. L=28 fp16 has top1−top2=1 (essentially noise), and decode-by-reprefill samples `cplusplus`, `idis`, `cplusplus`, `cplusplus`, `cplusplus` (responds to context across iterations but degenerate within an iteration).

**The mono L=28 SHA-blocksparse-causal also produces nonsense** ("userManager ۇ\n\n\n\n\n\n\n") with top1−top2=862. The block-sparse-causal model itself is the deeper bottleneck — that's a higher-level numerical accuracy problem in the block selection / per-qb attention, not in the split machinery. Main branch's dense `compile_sha` produces fully coherent text on the same device/skel, so the device/runtime stack is fine.

### Where the residual ~860× confidence gap with mono comes from

Each chunk boundary now goes uint16 → fp16 (Dequantize, lossless for the value range) → fp16 over the wire → fp16 → uint16 (Quantize at chunk input). The Quantize-at-chunk-input is the new noise source the fp16 boundary doesn't eliminate. With 56 chunk crossings × 2 boundary tensors per crossing × 2 Quantize ops each, that's quantization rounding error accumulating across ~200+ requantize sites. Each one uses a per-op scale that was calibrated on the training-time activation, not on the actual runtime boundary tensor.

The next steps to close that gap are scale-quality, not structural:

1. **Per-tensor boundary calibration**. Add a CPU calibration mode that runs the monolithic blocksparse-causal model on a representative prompt, captures the actual value ranges of the boundary tensors (residual_pre_attn_i, attn_output_i), and writes per-tensor scales to use at the chunk INPUT Quantize. Replaces the current head-0 / layer-1 alias proxies.

2. **Skip the Quantize-at-chunk-input for o_proj specifically**. o_proj is a quantized Conv2D; QNN should support fp16-in / quantized-weights / quantized-out. Tried this — fails in PTQPass.cpp with "TensorValue is not solved". The PTQ pass needs to know the input quant spec to derive the matmul output's spec. Possible workaround: extend PTQPass to accept "fp16 input → derive output spec from weights' quant spec".

3. **Address the underlying mono block-sparse correctness issue first**. The fact that mono L=28 itself samples `userManager` then `ۇ` then newlines means the model can't continue any prompt. Compare against the dense main-branch baseline at the same prompt to see where block-sparse diverges — could be the block-selection algorithm picking wrong K/V blocks, could be the causal-mask placement.

### Mono blocksparse-causal is itself the deeper bottleneck

After landing fp16 boundaries, tested mono SHA-blocksparse-causal at L=28 directly:

```
prompt: "Hello, my name is"
output: " userManager ۇ\n\n\n\n\n\n\n"

prompt: "Once upon a time, in a small village nestled at the foot of a great
        mountain, there lived a young girl named Maya who"  (38 tokens, 2 qbs)
output: "    to       "
```

Both incoherent. Main branch's dense `compile_sha` on the same device + skel produces fully coherent text — so the runtime stack is fine. The block-sparse-causal *model* (per-qb dispatch + top-k=8 K/V gather) doesn't produce coherent output regardless of prompt length.

Block-selection in [`ShaBlockSparsePromptProcessor::selectTopKBlocks`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.cpp) is a **sliding-window stub** — picks the most recent 7 historical q-blocks, no actual top-k by attention score. For a 17-token prompt (1 qb), this is irrelevant (no history to select); the diagonal slot covers all valid tokens and the causal mask handles the BQ-padding. Yet output is still wrong, so the selection algorithm isn't the only thing broken.

Likely candidates to investigate next session:

- The per-qb causal mask layout vs the dense model's `[1, 1, Sq, CL]` mask — `mask.equalConstant(zero_c)` + `where(mask==0, a, a_vv)` is unusual; verify the QDQ scale wiring matches what was calibrated during PTQ.
- The K_curr / V_curr diagonal-slot concat ordering — `concat({K_hist_h, K_curr_h}, -1)` puts historical first, current last. Position-to-slot mapping in the causal mask must match.
- **Most likely**: the W4A16 LPBQ PTQ was calibrated on a different model topology than the block-sparse-causal split. The per-op QDQ scales were captured during PTQ training under a specific attention structure; when that structure changes (dense → block-sparse, single dispatch → per-qb dispatch), the same intermediate tensors flow through different shapes and op fusion paths, and the calibrated scales no longer reflect runtime distributions. Mono on multiple prompts (`Hi`, `The capital of France is`, `1+1=`, `Hello, my name is`, 38-token narrative) all returned incoherent output — different gibberish per prompt (so input does reach output), but no prompt produced sensible continuation.
- The `attn_value_matmul_output_qdq_h0`-borrow-for-boundary heuristic was inherited from the mono model; if mono is wrong, the heuristic is also wrong.

Verification path: run the model under fp32 / fp16 (no quantization) using `compile_fp16_blocksparse_causal.cpp` (rebuilt against today's modeling). If fp16 produces coherent output, the algorithm + per-qb dispatch + gather + mask are all correct, and the issue is **purely PTQ calibration**. If fp16 is also broken, the algorithm itself has a structural bug (per-qb mask / concat ordering / something else). Today's fp16 bin from May 16 has an I/O contract mismatch (runner sends 115 inputs, bin expects 19) so it can't be tested without recompiling.

### Recommended next-session focus (priority order)

1. **Fix mono blocksparse-causal first.** Until mono produces coherent text on a short prompt, split correctness can't be evaluated — split is bounded below by mono. Diff the IR (`qwen3_qnn_aot_sha_blocksparse_causal_*.mir`) at L=4 against `qwen3_qnn_aot_sha_32.mir` (dense main) to find the structural difference. The fp16 reference [`modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp) bypasses PTQ; running that and comparing to the W4A16 SHA version isolates "algorithm broken" vs "quantization broken".
2. **Per-tensor boundary calibration.** Run mono on a representative prompt with chunk-boundary-equivalent tap points instrumented; capture actual value ranges; write proper per-tensor scales for the split's chunk-input Quantize ops. Replaces the current head-0 / layer-1 aliases.
3. **Extend PTQPass to derive op input quant spec from weight quant spec** when input is fp16. Would let `o_proj` consume fp16 directly, eliminating ~28 of the ~56 chunk-input Quantize ops in split.

### Working-tree state at session end (2026-05-18)

- All three boundary edits live in the working tree (modeling header, compile driver, runtime processor). No stash needed.
- `kBoundaryScaleMul` and the boundary alias block in the compile driver are gone (removed when fp16 boundary landed).
- `aliasQdqParam("model.layers.1.input_layernorm_input_qdq", "model.layers.0.input_layernorm_input_qdq")` added — required for the new uniform-per-layer pre()-side QDQ since PTQ has no layer-0 input_layernorm scale.
- `ptq::QDQ` helper in [`modeling_qwen_qnn_aot_sha.hpp`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp) extended with a `kFloat16` case that emits Quantize before attaching scale/zp. Shared with other SHA models — they don't currently feed fp16 into it, so no behavior change for them.
- `MLLM_DUMP_LOGITS` env-gated logits dumper still in both `Sha*PromptProcessor.cpp` — keep.
- `setvbuf(_IONBF)` now in **both** `aot_run_sha_blocksparse_causal_split.cpp` **and** `aot_run_blocksparse_causal.cpp` — keep both. Mono runner needed it for the same reason (logs were lost behind `adb shell`'s pipe buffer on abort).
- `tasks/build_x86_qnn_aot.yaml` — no g++ pin (system default g++-13 works).
- `tasks/build_android_qnn.yaml` — `htp_v79` target (was v75).

## 2026-05-18 session-end summary

Validated working:
- fp16 boundary refactor across modeling + compile + runtime — compiles clean, runs clean, no logit saturation.
- L=4 split + fp16 logits look healthy (top1−top2 = 2028, unambiguous argmax).

Newly bounded today:
- Mono SHA-blocksparse-causal at L=28 is itself incoherent across **every** prompt tried (`Hi`, `Hello, my name is`, `The capital of France is`, `1+1=`, 38-token narrative). Different garbage per prompt → input does reach output, so tokenizer/dispatch/gather are plumbed correctly.
- Main-branch dense `compile_sha` on the same device + skel produces fully coherent text → runtime stack, device, and V79 skel are all fine.
- Therefore: the block-sparse-causal *model* (per-qb dispatch + top-k gather + per-op QDQ scale wiring) is the bottleneck, not the split machinery layered on top of it. Split correctness is bounded below by mono and can't be assessed independently until mono is fixed.

Leading hypothesis: W4A16 LPBQ PTQ scales were calibrated under a different attention topology (dense / single-dispatch) than what the block-sparse-causal model runs at execution time; per-op scales misfit the actual runtime distributions.

Verification path for next session: rebuild [`compile_fp16_blocksparse_causal.cpp`](../../examples/qwen3_qnn_aot/compile_fp16_blocksparse_causal.cpp) against today's modeling (no PTQ involved) and run. fp16 coherent ⇒ purely a PTQ-calibration issue. fp16 also incoherent ⇒ structural bug in per-qb mask layout / K_curr-V_curr concat ordering / block-selection.

## 2026-05-18 follow-up — attention-chain structural audit (no on-device work)

Did a static audit comparing dense SHA vs mono SHA-blocksparse-causal at the per-op level, to bound where the bug *isn't* before next session. Important deltas (and non-deltas) for the SHA-blocksparse-causal:

**Identical to dense (ruled out as bug source):**
- The 9 per-head attention QDQ keys ([`modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp:251-270`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp#L251-L270) vs [`modeling_qwen_qnn_aot_sha.hpp:407-426`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp#L407-L426)) are byte-for-byte the same: `qk_matmul_output_qdq_h{h}` → `scaling_qdq_h{h}` → `mul_0_output_qdq_h{h}` → `reduce_min_output_qdq_h{h}` → `neg_20_qdq_h{h}` → `minus_0_output_qdq_h{h}` → `where_attn_qdq_h{h}` → `softmax_output_qdq_h{h}` → `attn_value_matmul_output_qdq_h{h}`. Same ops, same order, same QDQ wiring.
- The K_arranged trace-input borrows `model.layers.{i}.self_attn.k_cast_to_int8_qdq.fake_quant.{scale,zero_point}` ([`compile_sha_blocksparse_causal.cpp:158`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal.cpp#L158)). That layer-level key is the source for the per-head `_h{h}` copies via [`prepareParametersForSHA`'s `copyQDQParams`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp#L836) (line 869), which literally clones the same scalar — so per-layer and per-head K-cast scales are equal. Earlier "scale mismatch between K_hist and K_curr" hypothesis is **ruled out**.
- K/V concat ordering matches: both place historical-first, current/diagonal-last. Dense at [`modeling_qwen_qnn_aot_sha.hpp:387`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp#L387) (`concat({past_k_h, k_h}, -1)`), blocksparse at [`modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp:247`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha_blocksparse_causal.hpp#L247) (`concat({K_hist_h, K_curr_h}, -1)`). The mask in [`ShaBlockSparsePromptProcessor::build_mask`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.cpp#L184) (slots 0..6 historical, slot 7 diagonal triangle) matches this layout.

**Different from dense (but mathematically consistent):**
- Dense's `causal_mask` is `[1, 1, ar_len, context_len]` = `[1, 1, 32, 1024]` and selects via `where(causal_mask.equalConstant(zero_constant), attn, attn_vv)` with mask scale `0.001/65535` and zp `65535` ([`compile_sha.cpp:82-83`](../../examples/qwen3_qnn_aot/compile_sha.cpp#L82-L83)). Blocksparse's `mask` is `[1, 1, kBQ, kTopKBK]` = `[1, 1, 32, 256]` with **identical** scale/zp ([`compile_sha_blocksparse_causal.cpp:112-113`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal.cpp#L112-L113)). The where-pattern math is the same.
- Dense bakes `position_ids = [0..ar_len-1]` at trace time then re-binds at runtime ([`modeling_qwen_qnn_aot_sha.hpp:618`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp#L618); runner writes `pos_ids_ptr[i] = start_pos + i` in [`PromptProcessor.cpp:107`](../../mllm/backends/qnn/aot_rt/PromptProcessor.cpp#L107)). Blocksparse takes position_ids as an explicit trace-input and the runner writes `global_pos + qb_in_chunk*kBQ + i` ([`ShaBlockSparsePromptProcessor.cpp:206`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.cpp#L206)). Both produce 0..31 for a first-qb prompt — equivalent for qb_global=0.

**The remaining suspects after this audit:**
1. **PTQ scale mismatch with runtime distribution.** Same hypothesis as before — calibration was done under the dense per-prefill activation distributions; the per-qb dispatch sees BQ=32 rows in isolation, which can produce different intermediate-tensor ranges through e.g. RMSNorm, mul scale, softmax. The op chain is structurally correct, but a calibrated scale that was right for "Sq=32 with full-context Q" may be wrong for "BQ=32 with only-historical-context Q" at later qbs. At qb_global=0 the two are identical, yet output is still wrong — so this can't be the *only* thing.
2. **fp16 blocksparse-causal uses a different attention algorithm.** [`modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp:299-307`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_fp16_blocksparse_causal.hpp#L299-L307) uses **additive masking** (`attn + padding_mask + triangle_mask`) and **non-transposed K** (`concat axis=-2`, `K_full_h_T = K_full_h.transpose(...)`). The SHA path uses **where-based masking** (`where(mask==0, attn, attn_min-20)`) and **transposed K** for HMX (`concat axis=-1`). So **a fp16 test ONLY validates the gather + high-level algorithm + position handling**, not the SHA's where-pattern masking or the transposed-K matmul. If fp16 is coherent, the gather/pipeline is sound and the bug is in SHA quant/masking; if fp16 is also incoherent, the bug is in the shared algorithm (gather indices, position_ids, mask convention).

## Ready-to-run fp16 verification recipe (next session)

The compiled artefacts already exist on disk from the previous fp16 session:

- Bin: `qwen3-fp16-blocksparse-causal.bin` (1.65 GB, L=4, dated 2026-05-16)
- Compile driver: `build-qnn-aot/bin/mllm-qwen3-aot-fp16-blocksparse-causal-c` (rebuilt 2026-05-18 — current against source)
- Runner: `build-qnn-aot/bin/mllm-qwen3-aot-blocksparse-runner` (rebuilt 2026-05-18 — has the `setvbuf(_IONBF)` for unbuffered logs)
- Configs: [`config_1.7B_fp16_blocksparse_causal_4L.json`](../../examples/qwen3_qnn_aot/config_1.7B_fp16_blocksparse_causal_4L.json) (`num_hidden_layers=4`) + [`qnn_aot_cfg_1.7B_fp16_blocksparse_causal_4L.json`](../../examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_fp16_blocksparse_causal_4L.json)

The previous-session note "runner sends 115 inputs, bin expects 19" was a config mismatch — using the L=28 config against an L=4 bin. The L=4 configs above produce a runner that sends `2 + 4×4 + 1 = 19` inputs, matching the bin.

Cross-build the Android runner and push:

```bash
python task.py tasks/build_android_qnn.yaml
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-blocksparse-runner /data/local/tmp/
adb push qwen3-fp16-blocksparse-causal.bin /data/local/tmp/
adb push examples/qwen3_qnn_aot/config_1.7B_fp16_blocksparse_causal_4L.json /data/local/tmp/
```

Run a short prompt (avoid >32 tokens — the runner doesn't multi-qb prefill yet, see `BlockSparsePromptProcessor::prefill`):

```bash
adb shell 'cd /data/local/tmp && export LD_LIBRARY_PATH=. \
  && export "ADSP_LIBRARY_PATH=.:/data/local/tmp" \
  && echo "Hello, my name is" | \
     MLLM_DUMP_LOGITS=/data/local/tmp/fp16_logits.bin \
     ./mllm-qwen3-aot-blocksparse-runner \
       -m qwen3-fp16-blocksparse-causal.bin \
       -c config_1.7B_fp16_blocksparse_causal_4L.json \
       -t qwen3-tokenizer.json'
```

Interpretation:
- L=4 isn't a fair coherence test (4 layers can't produce sensible continuations even with a perfect implementation). What it *can* establish: argmax stability, no logit saturation, top-1 vs top-2 gap.
- Compare against the **dense main-branch `compile_sha` running the same prompt at L=4**. If both produce roughly the same logit distribution shape (mean/std/top-1 gap), the per-qb gather/algorithm is structurally fine and the bug is W4A16-PTQ-specific.
- If fp16 L=4 logits look *very different* from dense L=4 logits on the same prompt, the algorithm itself has a bug (gather direction, position_ids, mask convention, K-transpose vs non-transpose layout mismatch with the runner).
- To make this a real coherence test, rebuild fp16 at L=28: same compile driver + [`config_1.7B_fp16_blocksparse_causal.json`](../../examples/qwen3_qnn_aot/config_1.7B_fp16_blocksparse_causal.json) (already L=28). At L=28 the fp16 weights are ~3.3 GB so the PD memory cap applies — use `HtpUnsignedPd` in the aot config (same edit as the split's L=28 path).

## MIR-level equivalence proof (2026-05-18, late session)

Static analysis of the lowered MIRs ([qwen3_qnn_aot_sha_32.mir](../../qwen3_qnn_aot_sha_32.mir) vs [qwen3_qnn_aot_sha_blocksparse_causal_BQ32.mir](../../qwen3_qnn_aot_sha_blocksparse_causal_BQ32.mir)) confirms that at layer 0's `MatMul.0` (head 0's QK^T):

| Op | Dense MIR (`sha_32.mir:2081`) | Blocksparse MIR (`sha_blocksparse_causal_BQ32.mir:2045`) |
|---|---|---|
| Q input  | `[1, 1, 32, 128] UInt16` uuid=**133** | `[1, 1, 32, 128] UInt16` uuid=**133** |
| K input  | `[1, 1, 128, 1024] UInt8` uuid=**3** | `[1, 1, 128, 256] UInt8` uuid=**3** |
| Output   | `[1, 1, 32, 1024] UInt16` uuid=**288** | `[1, 1, 32, 256] UInt16` uuid=**288** |

The Q/K/output **QDQ uuids match** (133 / 3 / 288 ↔ 133 / 3 / 288). uuid 133 is `q_rope_add_0_output_qdq_h0`, uuid 3 is the per-layer `k_cast_to_int8_qdq`, uuid 288 is `qk_matmul_output_qdq_h0`. Each scale is loaded from the same param key in the same .mllm. The Concat that feeds the K input (Concat.24 in both MIRs) inherits its output scale from the K_hist/past_key input slot — so the requantize-at-concat semantics are identical.

**Consequence:** at `qb_global=0` with a prompt ≤ 32 tokens, dense and blocksparse should produce **numerically equivalent attention outputs** for the 32 valid Q positions (masked positions in both contribute < e^-20 weight to softmax). The remaining bug must be either:

(a) **Outside the attention compute** — in input_layernorm-input QDQ paths, RoPE sin/cos gather (rank-1 vs rank-2 indices), MLP, residual paths, embedding output, final-norm/lm_head, OR

(b) **Manifesting only at qb_global ≥ 1** — i.e., something specific to KV-cache update + gather round-trip that doesn't show up at qb=0.

The runner reports incoherent output even for short prompts (one qb), which rules out (b). That leaves (a) as the remaining candidate.

## Diagnostic playbook for next session

Static analysis is exhausted. The next session needs **per-layer tap-point dumps on-device** to identify which layer's output first diverges between mono blocksparse-causal and dense. Concrete steps:

1. **Single-token prompt comparison.** Run dense (`mllm-qwen3-aot-runner` with the mono SHA path) and mono blocksparse (`mllm-qwen3-aot-sha-blocksparse-runner`) on the same single-token prompt (e.g. just the BOS token). With one qb and one real query position, the attention behavior should be IDENTICAL between the two paths. If the logits differ on this case, the bug is **upstream of attention** (embedding, layer-0 input prep, position_ids gather, RoPE wiring). If they match, push to a longer prompt to localize the qb-boundary issue.

2. **Layer-N output tap.** Add an env-gated tap in both processors: after `current_pos += n_update`, dump the `present_key_<layer>` and `present_value_<layer>` buffer bytes for a chosen layer (start with N=0). Compare byte-for-byte between dense and mono blocksparse at the same prompt. The first layer where bytes diverge is the bug locus.
   - Hook for dense: in [`PromptProcessor.cpp:141`](../../mllm/backends/qnn/aot_rt/PromptProcessor.cpp#L141) before `updateCache`, dump `k_caches[N].output_buffer` (bytes = D × ar_len × sizeof(uint8) per head, total = Hkv × D × ar_len).
   - Hook for blocksparse: in [`ShaBlockSparsePromptProcessor.cpp:232`](../../mllm/backends/qnn/aot_rt/ShaBlockSparsePromptProcessor.cpp#L232) before `updateCache`, dump `k_caches[N].output_buffer` (same shape).
   - Quick equality test on CPU after running both with `cmp` or `python -c "import sys; a=open(sys.argv[1],'rb').read(); b=open(sys.argv[2],'rb').read(); print(sum(x!=y for x,y in zip(a,b)))"`.

3. **lm_head input tap.** After all layers, before lm_head, dump `residual` (or equivalent in both processors). The dense and mono blocksparse final-norm output should ALSO be byte-identical for the same single-token prompt. If layer-0 K bytes match but final-norm bytes don't, the divergence is in some middle layer's MLP or residual path.

4. **Direct logit dump and argmax comparison.** Both runners now have `MLLM_DUMP_LOGITS=` env support — generate logits.bin from both runners on the SAME single-token prompt and `cmp` them. Easiest first signal.

If even diagnostic (1) shows differing logits, the bug is structural in something we haven't inspected statically — most likely candidates: lm_head input QDQ key wiring, the `norm_input_qdq` at the top-level final norm, or an embedding output handling quirk specific to one path.

## Related docs

- [block_sparse_attention.md](block_sparse_attention.md) — block-sparse
  attention design rationale, M=BQ HMX efficiency cliff data, decomposed
  vs per-qb pipelined comparison.
- [custom_hvx_op_skill.md](custom_hvx_op_skill.md) — "should I write a
  custom op?" framing.
- [softmax_block_sparse_causal_op.md](softmax_block_sparse_causal_op.md) —
  the fused custom HVX softmax+mask+scale op (orthogonal optimisation to
  this doc; both could be combined).
- [aot_execute.rst](aot_execute.rst) — general QNN AOT compilation
  framework.
