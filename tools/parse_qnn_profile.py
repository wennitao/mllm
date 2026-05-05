"""Aggregate qnn_profile.csv into per-layer attention/MLP timings for layers 0-1.

Output: prefill / decode breakdown for the first two transformer layers of Qwen3-1.7B.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "qnn_profile.csv"
LAYER_OP = re.compile(r"^model\.layers\.(\d+)\.(.+?):OpId_(\d+)\s*\(cycles\)$")

# Block selection: drop block 1 (10 ms warmup) and block 5 (graph-switch overhead).
PREFILL_BLOCKS = [1, 2, 3, 4]         # 1-indexed, includes warmup block 1
DECODE_BLOCKS = list(range(6, 34))    # 1-indexed: 6..33

LAYERS = (0, 1)


def classify(sub: str) -> str:
    """Map an op sub-name (everything after `model.layers.N.`) to a bucket."""
    if sub.startswith("self_attn.q_proj"):       return "q_proj"
    if sub.startswith("self_attn.k_proj"):       return "k_proj"
    if sub.startswith("self_attn.v_proj"):       return "v_proj"
    if sub.startswith("self_attn.o_proj"):       return "o_proj"
    if sub.startswith("self_attn.Softmax"):      return "softmax"
    if sub.startswith("self_attn.MatMul."):
        idx = int(sub.rsplit(".", 1)[-1])
        return "matmul_qk" if idx % 2 == 0 else "matmul_av"
    if sub.startswith(("self_attn.Where", "self_attn.Equal", "self_attn.Slice")):
        return "mask"
    if sub.startswith("self_attn.Add"):
        idx = int(sub.rsplit(".", 1)[-1])
        return "mask" if idx >= 24 else "skip"   # Add.0..23 are RoPE
    if sub.startswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
        return "mlp"
    if sub.startswith(("mlp.Sigmoid", "mlp.Mul")):
        return "mlp"
    return "skip"


def parse_blocks():
    """Yield (accel_us, total_cycles, [(layer, sub, cycles)]) per QNN execute call."""
    blocks = []
    rows, accel_us, total_cyc = [], None, None
    with CSV_PATH.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            depth, _pid, _eid, ev, _unit, value, _ts, ident = r
            ident = ident.strip()
            if ev == "EXECUTE" and "QNN (execute) time" in ident:
                blocks.append((accel_us, total_cyc, rows))
                rows, accel_us, total_cyc = [], None, None
                continue
            if depth == "0" and "QNN accelerator (execute) time" in ident:
                accel_us = int(value)
            elif depth == "0" and "Accelerator (execute) time (cycles)" in ident:
                total_cyc = int(value)
            else:
                m = LAYER_OP.match(ident)
                if m:
                    rows.append((int(m.group(1)), m.group(2), int(value)))
    return blocks


def phase_us_per_cycle(blocks, indices):
    """Single conversion factor for the phase: sum(accel_us) / sum(total_cycles)
    averaged over the chosen blocks."""
    sum_us = sum(blocks[i - 1][0] for i in indices)
    sum_cy = sum(blocks[i - 1][1] for i in indices)
    return sum_us / sum_cy


def total_us(blocks, indices, layer, us_per_cycle):
    """For each bucket, SUM cycles across all chosen blocks, then convert
    with the phase-level us/cycle. Returns total time the bucket consumed
    across the entire phase (not per-call mean)."""
    totals = defaultdict(int)
    for i in indices:
        _, _, rows = blocks[i - 1]
        for L, sub, cyc in rows:
            if L != layer:
                continue
            b = classify(sub)
            if b != "skip":
                totals[b] += cyc
    return {b: totals[b] * us_per_cycle for b in BUCKETS}


BUCKETS = ["q_proj", "k_proj", "v_proj",
           "matmul_qk", "mask", "softmax", "matmul_av",
           "o_proj", "mlp"]

LABEL = {
    "q_proj":    "Attention(q_proj)",
    "k_proj":    "Attention(k_proj)",
    "v_proj":    "Attention(v_proj)",
    "matmul_qk": "Attention(matmul Q@K^T)",
    "mask":      "Attention(mask)",
    "softmax":   "Attention(softmax)",
    "matmul_av": "Attention(matmul attn@V)",
    "o_proj":    "Attention(out)",
    "mlp":       "MLP (3 linear + SiLU)",
}


def emit(title, us_by_bucket):
    print(title)
    for b in BUCKETS:
        print(f"  {LABEL[b]:<26s} {us_by_bucket[b]:8.1f} us")
    print()


def main():
    blocks = parse_blocks()
    pre_upc = phase_us_per_cycle(blocks, PREFILL_BLOCKS)
    dec_upc = phase_us_per_cycle(blocks, DECODE_BLOCKS)

    print(f"Parsed {len(blocks)} QNN execute blocks "
          f"(prefill: 1-4, decode: 5-33).")
    print(f"Prefill conversion (avg over blocks 2-4): "
          f"{pre_upc * 1e3:.4f} ns/cycle "
          f"({1/pre_upc:.0f} cycles/us)")
    print(f"Decode  conversion (avg over blocks 6-33): "
          f"{dec_upc * 1e3:.4f} ns/cycle "
          f"({1/dec_upc:.0f} cycles/us)\n")

    emit(f"Prefill — Layer 0 (total over {len(PREFILL_BLOCKS)} prefill calls)",
         total_us(blocks, PREFILL_BLOCKS, 0, pre_upc))
    emit(f"Prefill — Layer 1 (total over {len(PREFILL_BLOCKS)} prefill calls)",
         total_us(blocks, PREFILL_BLOCKS, 1, pre_upc))
    emit(f"Decode  — Layer 0 (total over {len(DECODE_BLOCKS)} decode calls)",
         total_us(blocks, DECODE_BLOCKS,  0, dec_upc))
    emit(f"Decode  — Layer 1 (total over {len(DECODE_BLOCKS)} decode calls)",
         total_us(blocks, DECODE_BLOCKS,  1, dec_upc))


if __name__ == "__main__":
    main()
