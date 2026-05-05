"""Print the exact CSV line numbers used to compute each bucket
for the four sections (Prefill L0, Prefill L1, Decode L0, Decode L1).
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "qnn_profile.csv"
LAYER_OP = re.compile(r"^model\.layers\.(\d+)\.(.+?):OpId_(\d+)\s*\(cycles\)$")

PREFILL_BLOCKS = [1, 2, 3, 4]
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


# Pass 1: parse the CSV but ALSO record the 1-based line number of each op
# Each block: list of (lineno, layer, sub, cycles) plus the lineno
# of "QNN accelerator (execute) time" and "Accelerator (execute) time (cycles)".
def parse_blocks_with_lines():
    blocks = []
    rows = []
    accel_us_line = total_cyc_line = None
    accel_us_val = total_cyc_val = None

    with CSV_PATH.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        lineno = 1  # we just consumed line 1 (header)
        for row in reader:
            lineno += 1
            depth, _pid, _eid, ev, _u, val, _ts, ident = row
            ident = ident.strip()
            if ev == "EXECUTE" and "QNN (execute) time" in ident:
                blocks.append({
                    "rows": rows,
                    "accel_us_line": accel_us_line,
                    "accel_us_val": accel_us_val,
                    "total_cyc_line": total_cyc_line,
                    "total_cyc_val": total_cyc_val,
                })
                rows = []
                accel_us_line = total_cyc_line = None
                accel_us_val = total_cyc_val = None
                continue
            if depth == "0" and "QNN accelerator (execute) time" in ident:
                accel_us_line = lineno
                accel_us_val = int(val)
            elif depth == "0" and "Accelerator (execute) time (cycles)" in ident:
                total_cyc_line = lineno
                total_cyc_val = int(val)
            else:
                m = LAYER_OP.match(ident)
                if m:
                    rows.append((lineno, int(m.group(1)), m.group(2), int(val)))
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


def fmt_lines(lines):
    """Compress a sorted list of line numbers into 'a-b, c, d-e' ranges."""
    lines = sorted(set(lines))
    if not lines:
        return ""
    out = []
    start = prev = lines[0]
    for x in lines[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = x
    out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(out)


def show(title, blocks, indices, layer):
    print("=" * 78)
    print(title)
    print("=" * 78)

    # Conversion lines
    upc_us_lines  = [blocks[i - 1]["accel_us_line"]  for i in indices]
    upc_cyc_lines = [blocks[i - 1]["total_cyc_line"] for i in indices]
    sum_us = sum(blocks[i - 1]["accel_us_val"]  for i in indices)
    sum_cy = sum(blocks[i - 1]["total_cyc_val"] for i in indices)
    upc = sum_us / sum_cy

    print(f"  blocks averaged    : {indices}")
    print(f"  'QNN accelerator (execute) time' lines: {upc_us_lines}")
    print(f"  'Accelerator (execute) time (cycles)' lines: {upc_cyc_lines}")
    print(f"  sum_us = {sum_us:,}, sum_cycles = {sum_cy:,}, us/cycle = {upc:.6f}\n")

    # Per bucket: gather lines and cycles
    bucket_lines = defaultdict(list)
    bucket_cyc_per_block = defaultdict(lambda: defaultdict(int))  # bucket -> {block_idx: cycles_sum}
    for i in indices:
        for ln, L, sub, c in blocks[i - 1]["rows"]:
            if L == layer:
                b = classify(sub)
                if b != "skip":
                    bucket_lines[b].append(ln)
                    bucket_cyc_per_block[b][i] += c

    for b in BUCKETS:
        print(f"--- {LABEL[b]} ---")
        # Per-block summary:
        for i in indices:
            block_lines = sorted(ln for ln, L, sub, _ in blocks[i - 1]["rows"]
                                 if L == layer and classify(sub) == b)
            cyc = bucket_cyc_per_block[b][i]
            print(f"  block {i}: cycles = {cyc:>9,}   from CSV lines: {fmt_lines(block_lines)}")
        all_cyc = [bucket_cyc_per_block[b][i] for i in indices]
        total_cyc = sum(all_cyc)
        print(f"  total cycles (sum across {len(all_cyc)} sequential blocks) "
              f"= {total_cyc:,}")
        print(f"  total time   = {total_cyc:,} * {upc:.6f} = {total_cyc*upc:.1f} us")
        print()


def main():
    blocks = parse_blocks_with_lines()
    show("PREFILL — Layer 0",  blocks, PREFILL_BLOCKS, 0)
    show("PREFILL — Layer 1",  blocks, PREFILL_BLOCKS, 1)
    show("DECODE  — Layer 0",  blocks, DECODE_BLOCKS,  0)
    show("DECODE  — Layer 1",  blocks, DECODE_BLOCKS,  1)


if __name__ == "__main__":
    main()
