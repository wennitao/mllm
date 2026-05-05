"""For each of (prefill L0, prefill L1, decode L0, decode L1), print:
  - exactly which op names go into each bucket
  - per-block cycle sums
  - mean cycles, conversion ratio, final us
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "qnn_profile.csv"
LAYER_OP = re.compile(r"^model\.layers\.(\d+)\.(.+?):OpId_(\d+)\s*\(cycles\)$")

PREFILL_BLOCKS = [2, 3, 4]
DECODE_BLOCKS = list(range(6, 34))


def classify(sub):
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
        return "mask" if idx >= 24 else "skip"
    if sub.startswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                       "mlp.Sigmoid", "mlp.Mul")):
        return "mlp"
    return "skip"


def parse_blocks():
    blocks = []
    rows, accel_us, total_cyc = [], None, None
    with CSV_PATH.open() as f:
        r = csv.reader(f); next(r)
        for row in r:
            depth, _pid, _eid, ev, _u, val, _ts, ident = row
            ident = ident.strip()
            if ev == "EXECUTE" and "QNN (execute) time" in ident:
                blocks.append((accel_us, total_cyc, rows))
                rows, accel_us, total_cyc = [], None, None
                continue
            if depth == "0" and "QNN accelerator (execute) time" in ident:
                accel_us = int(val)
            elif depth == "0" and "Accelerator (execute) time (cycles)" in ident:
                total_cyc = int(val)
            else:
                m = LAYER_OP.match(ident)
                if m:
                    rows.append((int(m.group(1)), m.group(2), int(val)))
    return blocks


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


def show_section(title, blocks, indices, layer):
    sum_us = sum(blocks[i - 1][0] for i in indices)
    sum_cy = sum(blocks[i - 1][1] for i in indices)
    upc = sum_us / sum_cy

    # Bucket name -> set of op-subnames present in this layer
    bucket_ops = defaultdict(set)
    for i in indices:
        for L, sub, _ in blocks[i - 1][2]:
            if L == layer:
                b = classify(sub)
                if b != "skip":
                    bucket_ops[b].add(sub)

    # Per-block cycles per bucket
    bucket_cyc = defaultdict(list)
    for i in indices:
        per = defaultdict(int)
        for L, sub, c in blocks[i - 1][2]:
            if L == layer:
                b = classify(sub)
                if b != "skip":
                    per[b] += c
        for b in BUCKETS:
            bucket_cyc[b].append(per[b])

    print("=" * 78)
    print(f"{title}")
    print("=" * 78)
    print(f"phase blocks averaged: {indices}")
    print(f"phase total accel_us : {sum_us:>14,}")
    print(f"phase total cycles   : {sum_cy:>14,}")
    print(f"  -> us / cycle      : {upc:.6f}  ({1/upc:.0f} cycles/us)\n")

    for b in BUCKETS:
        ops = sorted(bucket_ops[b])
        cyc = bucket_cyc[b]
        mean_cyc = sum(cyc) / len(cyc)
        us = mean_cyc * upc
        print(f"--- {LABEL[b]}  ({len(ops)} op{'s' if len(ops)!=1 else ''} summed) ---")
        if ops:
            for o in ops:
                print(f"      model.layers.{layer}.{o}")
        per_block_str = "  ".join(f"{c:>7,}" for c in cyc)
        print(f"   cycles by block : [{per_block_str}]   mean = {mean_cyc:,.0f}")
        print(f"   time            : {mean_cyc:,.0f} cyc * {upc:.6f} us/cyc = {us:.1f} us\n")


def main():
    blocks = parse_blocks()
    show_section("PREFILL — Layer 0 (first layer)",  blocks, PREFILL_BLOCKS, 0)
    show_section("PREFILL — Layer 1 (second layer)", blocks, PREFILL_BLOCKS, 1)
    show_section("DECODE  — Layer 0 (first layer)",  blocks, DECODE_BLOCKS,  0)
    show_section("DECODE  — Layer 1 (second layer)", blocks, DECODE_BLOCKS,  1)


if __name__ == "__main__":
    main()
