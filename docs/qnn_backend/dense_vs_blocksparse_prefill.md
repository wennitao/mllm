# Dense vs block-sparse prefill (BK=32 / BK=64 / dense, Sq 128–2048)

On-device prefill throughput of **dense full-attention** vs **block-sparse**
attention at block sizes **BK=32** and **BK=64**, swept over sequence length
Sq ∈ {128, 256, 512, 1024, 2048}, on **SM8750 / V79 HTP** (QAIRT 2.43,
Qwen3-1.7B W4A16 LPBQ, L=28).

Getting an apples-to-apples answer required two enabling changes, each of which
removed a wall that previously made a cell of the matrix unmeasurable:

1. **Multi-context split** — block-sparse at Sq≥1024 overflows the ~3.6 GB
   *per-context* HTP PD ceiling as a single context. Splitting the model into
   several context bins loaded as one spill-fill group fits it.
   (§[Enabler 1](#enabler-1-multi-context-split-for-block-sparse-sq1024).)
2. **Dense last-token lm_head** (`MLLM_DENSE_LASTTOK`) — the dense graph ran
   lm_head over *all* Sq positions, which (a) made the comparison unfair (block-
   sparse slices last-token) and (b) hit a 622 MB rpcmem allocation limit at
   Sq=2048. Reducing lm_head to one position fixes both.
   (§[Enabler 2](#enabler-2-dense-last-token-lm_head).)

## TL;DR

Prefill throughput, tok/s (Sq / prefill_wall, single-graph pass), device
`2a38935c`, one session:

| Sq | **Dense** | BK=32 | BK=64 |
|---|---:|---:|---:|
| 128 | 2166 | 2106 | 2304 |
| 256 | 3464 | 2617 | 2870 |
| 512 | **3703** | 2534 | 2767 |
| 1024 | **3558** | 2146 | 2294 |
| 2048 | **2704** | 2200 | 2362 |

- **Dense full-attention beats block-sparse at every Sq** — by a wide margin at
  mid/long context (Sq=1024: dense 3558 vs BK=64 2294, **+55 %**). The block-
  sparse split's per-qb dispatch + CPU gather/selection + multi-context overhead
  outweighs its sparse-attention savings.
- Dense peaks around Sq=512 (3703) then declines at Sq=2048 (2704) as the
  O(Sq²) attention term grows — but still leads.
- **BK=64 > BK=32 at every Sq** (~7–10 %): fewer, larger-block dispatches win.
  (An earlier "crossover" where BK=32 led at Sq=2048 was a measurement
  artifact — see [Methodology](#methodology).)
- **Block-sparse's value is long-context memory capacity, not prefill speed.**
  Its sparsity does not buy throughput here; the projections/MLP/lm_head, not
  attention, dominate prefill at these sizes.

> kTopK=8 throughout, so BK=64 covers 512 history tokens per query block and
> BK=32 covers 256. Block-sparse is only *actually* sparse once num_blocks >
> kTopK, i.e. Sq > 8·BK (Sq > 512 for BK=64, Sq > 256 for BK=32); below that it
> selects all blocks and is effectively dense.

---

## Enabler 1 — multi-context split (for block-sparse Sq≥1024)

### The wall

The QNN/HTP "PD" (program-domain) limit on V79 is a **per-context** ceiling of
≈3.6 GB on a single context's reserved space (firmware/device dependent), plus
a much larger per-device pool. A single block-sparse split-prefill context at
Sq≥1024 exceeds the per-context ceiling and fails to load:

| context | reserved | result |
|---|---:|---|
| BK=64 Sq=1024 (one context, 57 graphs) | 3.68 GB | **`Failed to find available PD`** |
| BK=64 Sq=2048 (one context) | 5.88 GB | fails |
| 4-layer split bin | 1.26 GB | loads |
| 28-layer split bin | 2.79 GB | loads |

Measured with a probe that loads N weight-bearing contexts into one spill-fill
group and reads the per-context reserved bytes from the HTP verbose log
(`Setting context N reserved space to <bytes>`): the device held **8 × 1.26 GB
= ~10 GB co-resident** before the 9th failed, while a single **3.68 GB** context
fails outright. So the binding constraint is **per-context**, and splitting the
model into sub-ceiling contexts fits — the total pool (~10 GB here) has ample
room.

### The fix

Compile the 2L+1 split-prefill chunks across **N HTP context bins** and load
them as one spill-fill group.

- **AOT** —
  [`compile_sha_blocksparse_causal_split.cpp`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal_split.cpp)
  gains `--num_contexts N`; it assigns each chunk to context ⌊j·N/L⌋ by layer
  and `saveContext`s one `.bin` per context. A per-chunk `chunk_context_name` is
  threaded through
  [`SplitLLMGraphPass`](../../mllm/backends/qnn/aot/passes/SplitLLMGraphPass.cpp)
  (sets the per-op `qnn_context_name`) and
  [`LLM2QnnLoweringPass`](../../mllm/backends/qnn/aot/passes/LLM2QnnLoweringPass.cpp)
  (creates/captures into that context). `split_graph` stays 1, so the existing
  guards are satisfied, not tripped.
- **Runtime** —
  [`QNNBackend::loadContextGroup`](../../mllm/backends/qnn/QNNBackend.cpp) loads
  N bins as one group (bin 0 registers a new spill-fill group,
  `firstGroupHandle=0`; the rest join via `firstGroupHandle=anchor`), merging
  all graphs into one index map. `QNNModel` now stores its owning context, and
  `graphExecute` points the allocator at that context before binding I/O. The
  [`QNNAllocator`](../../mllm/backends/qnn/QNNAllocator.cpp) registration cache
  is keyed by `(ptr, context)` so the two cross-bin seam buffers
  (`residual` + `attn_output` at the cut) register against both contexts.
  `initQnnBackendGroup(paths)` is the entry point; the split runner's `-m`
  accepts a comma-separated bin list. The processor itself is unchanged.

### Result

BK=64 Sq=1024 as a 2-bin split: **c0 = 2.02 GB (28 graphs) + c1 = 1.66 GB
(29 graphs)**, both under the ceiling, where the single 3.68 GB context failed.
Validated **bit-exact** vs the single-bin path at Sq=512 (identical token
sequence `9707, 11, 847, 829, 374`).

---

## Enabler 2 — dense last-token lm_head

### The wall

The dense graph (`compile_sha.cpp`, one `model.0.sN` graph) ran lm_head over
**all N** positions: `[N,2048] × [2048,151936]`. That matmul is a large share
of dense prefill (≈48 ms over 512 positions), so leaving it in penalized dense
~18–35 % vs block-sparse (which already slices last-token in `chunk_L`). Worse,
the lm_head **output** is `[1,1,N,151936]` uint16 = 311 MB at N=1024 and
**622 MB at N=2048**, which exceeds an rpcmem/ION single-allocation limit
(`Mod 4 failed … size: 622329856 … err 8003`) — so dense N=2048 *loaded*
(3.52 GB PD) but could not execute.

The `MLLM_DENSE_LASTTOK` optimization (compute lm_head for one position) fixes
both, but the original slice-based attempt crashed the AOT pipeline: the
outer-graph slice landed on the **CPU side** of the `op_on_qnn=[lm_head]`
partition, so lm_head consumed it across a CPU→QNN boundary with no QNN
quant-recipe (cascading failures in `LLMQuantRecipePass`, then `PTQPass`).

### The fix

Mirror the block-sparse `chunk_L`: do the last-token reduction **inside the QNN
graph** with a **gather** whose index comes from a graph input.

- [`modeling_qwen_qnn_aot_sha.hpp`](../../examples/qwen3_qnn_aot/modeling_qwen_qnn_aot_sha.hpp)
  — in `Qwen3TextSHA::forward`, after `model.norm`, gather position Sq-1:

  ```cpp
  if (std::getenv("MLLM_DENSE_LASTTOK")) {
    const int sq = x.shape()[1];
    auto last_idx = position_ids.slice({{sq - 1, sq}}, /*ssa=*/true);  // native index
    x = nn::functional::gather(x, 1, last_idx);  // [1,Sq,hidden] -> [1,1,hidden]
  }
  ```

  Two subtleties that each caused a failure: the gather must be **inside the
  llm/QNN graph** (so the recipe propagates to lm_head), and the index must be a
  **native (input-derived) tensor** — a baked constant makes QNN Gather fail to
  construct (`err 6007`). Slicing the existing `position_ids` graph input
  satisfies both (its value at Sq-1 is Sq-1 for prefill).
- [`LLMQuantRecipePass.cpp`](../../mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp)
  — the Gather recipe pattern now also assigns the **index** operand a recipe
  (previously only input[0]/output), so `PTQPass` doesn't dereference a null
  spec. Safe and general (graph-input indices already had one).

The runtime needed **no change** —
[`PromptProcessor`](../../mllm/backends/qnn/aot_rt/PromptProcessor.cpp) already
sizes the logits output to 1 position and reads index 0 under
`MLLM_DENSE_LASTTOK`.

### Result

**Prefill logits are bit-identical** to the full-lm_head dense (argmax=2132,
max-abs-diff = 0 over all 151,936 vocab entries, validated with `MLLM_DUMP_LOGITS`
on a prompt tokenized to exactly Sq). It is **prefill-only** by design — decode
garbles because the gather is fixed at the window's last position, valid only
for a full-window prefill (num_tokens == Sq). N=2048 now executes (lm_head
output 0.3 MB instead of 622 MB).

The lm_head reduction is the entire reason the fair dense column beats the old
pessimistic one:

| Sq | Dense fair (last-token) | Dense full-lm_head |
|---|---:|---:|
| 128 | 2166 | 1787 |
| 256 | 3464 | 2425 |
| 512 | 3703 | 2734 |
| 1024 | 3558 | 2708 |
| 2048 | 2704 | — (622 MB rpcmem block) |

---

## Methodology

**Throughput is `Sq / prefill_wall`** — a single-graph pass over Sq positions.
The graph is fixed-size, so wall time is ~constant in actual prompt length; the
*compute* is what we compare.

- **Block-sparse** (`mllm-qwen3-aot-sha-blocksparse-split-runner`): fed
  `NIAH1_4k.txt` (~3700 tokens), which the runner **truncates** to Sq, so it
  reports `num_tokens = Sq` and `tok/s = Sq/time` directly.
- **Dense** (`mllm-qwen3-aot-runner`): the dense runner does *chunked
  autoregressive* prefill, so a long prompt overflows the KV cache. Feed a
  **short prompt** (one pass; the graph still does full-N work) and compute
  `Sq / prefill_time` from the reported `Prefill time` (the runner's printed
  `tok/s` is `prompt_tokens/time`, an undercount).

**Consistency matters.** An earlier round mixed `prompt_tokens/time` with
different-length prompts and produced a spurious BK crossover. Measuring all
cells as `Sq/time` removes it; BK=64 leads throughout.

**Spill-fill gotchas:**
- Block-sparse multi-context groups need `MLLM_QNN_SPILLFILL_MB` set: **128** for
  Sq=1024, **≥192** for Sq=2048 (the group buffer is sized by the anchor bin;
  Sq=2048 needs ~134.6 MB > the 128 default, else
  `Shared spill-fill size … smaller than required`, err 0x1392).
- The **dense** single graph must **not** set `MLLM_QNN_SPILLFILL_MB` — it caps
  the graph's spill-fill too small and aborts at load. Without it the graph uses
  its own baked spill-fill (e.g. 2.61 GB at N=1024) and fits.

**Variance:** QDC devices show ~10–15 % cross-device/session swing (thermal /
DVFS). Trust *within-session* relative ordering, not absolute cross-session
deltas. All numbers above are one device, one session.

### Raw wall times (µs)

| Sq | Dense (lasttok) | BK=32 | BK=64 |
|---|---:|---:|---:|
| 128 | 59 107 | 60 769 | 55 559 |
| 256 | 73 913 | 97 808 | 89 208 |
| 512 | 138 255 | 202 061 | 185 018 |
| 1024 | 287 784 | 477 099 | 446 475 |
| 2048 | 757 494 | 930 779 | 867 208 |

## Reproduction

Compile (x86 AOT host; needs `libunwind.so.1` on `LD_LIBRARY_PATH`):

```bash
export QAIRT_SDK_ROOT=/path/to/qairt/2.43.0.260128
export LD_LIBRARY_PATH=$QAIRT_SDK_ROOT/lib/x86_64-linux-clang:/path/to/HOST_TOOLCHAIN/lib

# Block-sparse, 2-context split at Sq>=1024 (kBK set in the two split headers):
mllm-qwen3-aot-sha-blocksparse-causal-split-c \
  -m qwen3_1.7b_ptq_lpbq.mllm -c config_1.7B_w4a16_blocksparse_causal.json \
  -aot_cfg qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split.json \
  --sq 1024 --num_contexts 2

# Dense, last-token lm_head (use a CL=N+margin config to avoid the empty-past crash):
MLLM_DENSE_LASTTOK=1 mllm-qwen3-aot-sha-c \
  -m qwen3_1.7b_ptq_lpbq.mllm -c config_cl1280.json \
  -aot_cfg qnn_aot_cfg_1.7B_dense.json -N 1024 -o qwen3-dense-N1024-lasttok.bin
```

Run on device:

```bash
# Block-sparse Sq=1024 (2-bin group):
cd /data/local/tmp && export LD_LIBRARY_PATH=.:/data/local/tmp \
  ADSP_LIBRARY_PATH=.:/data/local/tmp MLLM_QNN_SPILLFILL_MB=128
tr '\n' ' ' < NIAH1_4k.txt | ./mllm-qwen3-aot-sha-blocksparse-split-runner \
  -m qwen3-bsBK64-sq1024-c0of2.bin,qwen3-bsBK64-sq1024-c1of2.bin \
  -t qwen3-tokenizer.json -c config_bs.json --sq 1024 --gen 0

# Dense Sq=1024 (no MLLM_QNN_SPILLFILL_MB; throughput = 1024 / Prefill_time):
export MLLM_DENSE_LASTTOK=1; unset MLLM_QNN_SPILLFILL_MB
cat shortprompt.txt | ./mllm-qwen3-aot-runner -m qwen3-dense-N1024-lasttok.bin \
  -t qwen3-tokenizer.json -c config_cl1280.json --ar_len 1024 --gen 0
```

## Caveats / scope

- **`MLLM_DENSE_LASTTOK` is prefill-only.** Decode needs the un-reduced path.
- **Dense CL padding.** Dense uses a `CL = N + 256` cache (a margin to dodge a
  `CL == N` empty-past compile crash); the small extra attention is a slight,
  Sq-dependent penalty on dense (larger at small Sq) — so the dense column is if
  anything mildly *conservative*.
- **kBK is a compile-time constant** in two headers
  (`modeling_qwen_qnn_aot_sha_blocksparse_causal_split.hpp` and
  `ShaBlockSparsePromptProcessorSplit.hpp`) — switch BK=32↔64 by editing both
  and rebuilding the compiler + runner.
- The multi-context PD result holds where the device's total pool ≫ the
  per-context ceiling (≈10 GB vs 3.6 GB here). On a small-RAM device where the
  pool ≈ the per-context ceiling, splitting would not help.

## Related

- [`split_prefill.md`](split_prefill.md) — why the block-sparse split-prefill
  structure exists.
- [`blocksparse_pipeline_profile_sq1024.md`](blocksparse_pipeline_profile_sq1024.md)
  — where the ~490 ms block-sparse Sq=1024 prefill goes (per-component).
- [`dense_regression_bisect.md`](dense_regression_bisect.md) — dense whole-graph
  baseline lineage.
