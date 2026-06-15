# Block-sparse two-pass GEMM attention (Adreno 830)

Applies the dense two-pass GEMM prefill ([flash_attention_optimization.md](flash_attention_optimization.md),
Stage 2/3, ~1 TF) to **block-sparse** attention. The earlier OpenCL block-sparse
kernel (branch `blocksparse-opencl`) was built on the *old* fused FA skeleton
(~116 GF/s era); this rebuilds it on the image-A GEMM that reaches ~1 TF.

Bench: [examples/bsa_twopass_bench/](../../examples/bsa_twopass_bench/)
(`mllm-bsa-twopass-bench`, kernels in `kernels.cl.inc`), validated against a CPU
block-sparse reference (max_abs ≤2.2e-4, no NaN — the fp16 rounding floor).

## Design: index-driven (no gather) + image-A GEMM

The naive "gather selected K/V into contiguous per-query-block buffers" would
reintroduce the gather bottleneck — with per-qb replication it moves *more* data
than dense. Instead this keeps **both** wins:

- **Index-driven (no gather)**: K/V are transposed/copied to FULL contiguous
  images ONCE (the dense `pack_q`/`trans_k`/`copy_v` pre-passes, reused
  verbatim). The QK and P·V GEMMs read those images at the **selected block
  offsets** from `block_idx` — no per-qb gather materialization.
- **Image-A GEMM**: the validated variant-C 8×4 image-A / packed-B GEMM, so the
  selected-key matmul runs at GEMM-class throughput.

Pipeline (per head, batched over B·H·num_qb via grid dim 2):
1. `pack_q` Q → Qp (packed; same as dense)
2. `trans_k` K → Kt[d,k] image (full, transposed; same as dense)
3. `copy_v` V → Vc[k,d] image (full; **vectorized half8**, coalesced)
4. `bs_qk_gemm` A=Kt image read at selected block offsets, B=Qp → S[q_local, slot]
   query-major (sel = top_k·BK wide), epilogue = scale + causal/padding mask + clamp
5. `bs_softmax` online (m,l) + write, in place over the `sel` slots
6. `bs_pv_gemm` A=Vc image read at selected offsets, B=P direct (4×vload4);
   outer loop over selected blocks (block_idx once/block, padding skipped) → O

Correctness mirrors the dense kernel: online softmax is order-independent over
any distinct key subset, and the per-element causal test handles the diagonal
block's triangle. Bit-exact to dense at full selection; the "sparse"
approximation is entirely in *which* blocks the host selects.

## Measured (SM8750 / Adreno 830, H=16, D=128, BQ=BK=64, sink+recent selection)

E2E ms / speedup vs the dense two-pass (dense: 18.6 ms @2048, 69.8 @4096, ~280 @8192):

| top_k (sel, density) | S=2048 | S=4096 | S=8192 |
|---:|---:|---:|---:|
| 4 (256, ~6%)  | 5.5 ms / **3.4×** | 11.9 / **5.9×** | 33.1 / **~8.5×** |
| 8 (512, ~12%) | 9.1 / 2.0× | 21.3 / 3.3× | 51.8 / ~5.4× |
| 16 (1024, ~25%) | 15.6 / **1.2×** | 34.4 / 2.0× | 78.5 / ~3.6× |

Block-sparse two-pass **wins across the whole sweep** — the advantage grows with
sparsity and context length, exactly as block-sparse should, and now on the fast
GEMM. (The old block-sparse claimed 4× over dense @2048, but that was over the
*116 GF/s* dense; against today's ~1 TF dense the saving is purely the smaller
selected-key GEMM — and it still wins.) Throughput on the selected work reaches
**~880 GF/s** (top_k=16 @4096).

### Two overhead fixes that mattered (each measured)
- **Vectorized `copy_v`** (half8, coalesced) — the element-per-thread strided
  copy was **6.65 ms (32% of E2E)** at top_k=4/4096; vectorizing dropped it to
  **0.53 ms**.
- **Hoisted `block_idx` in P·V** (outer loop over selected blocks, read once per
  block, skip padding) — P·V 5.63 → 3.32 ms.

Per-stage @4096/top_k=4 (E2E 12.3 ms): qk 3.22 / pv 3.32 / softmax 2.28 /
trans_k 1.87 / copy_v 0.53 — the two GEMMs balanced, the rest is the full-K/V
pre-pass + softmax glue.

## Notes / next
- **Crossover**: dense still wins when sel ≈ the causal average (very high
  density at short S); pick block-sparse when top_k·BK is meaningfully below the
  causal window, which is the long-context regime it's meant for.
- **Selection is a host concern** (XAttention / sink+recent / top-k) — the kernel
  only consumes `block_idx`; it never scores. Must hold DISTINCT ids per (bh,qb)
  and include the diagonal block.
- **Productionization** (not yet wired): an op path mirroring the dense
  `tryForwardTwoPass`, taking the selection as an extra input. The pre-passes and
  scratch are identical to dense; only qk/softmax/pv differ.
- The full-K/V pre-passes (`trans_k`/`copy_v`) are an O(S) floor; at extreme
  sparsity + long S they become the limit (and `trans_k`'s transpose is strided).
  A tiled transpose or skipping unselected blocks (when the selection union is
  sparse) would push further.
