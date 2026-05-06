"""For decode (blocks 6-33), compute per-layer total time and the
multiplied estimate `per_layer * 28 * decode_tokens`."""

import csv
import re
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "qnn_profile_old.csv"
LAYER_OP = re.compile(r"^model\.layers\.(\d+)\.(.+?):OpId_(\d+)\s*\(cycles\)$")

DECODE_BLOCKS = list(range(6, 34))   # 1-indexed, 28 blocks

BUCKETS = {
    "q_proj":     "q_proj",
    "k_proj":     "k_proj",
    "v_proj":     "v_proj",
    "MatMul":     "matmul (Q@K^T + attn@V)",
    "Softmax":    "softmax",
    "Where":      "mask",
    "Equal":      "mask",
    "Slice":      "mask",
    "o_proj":     "o_proj",
    "gate_proj":  "mlp",
    "up_proj":    "mlp",
    "down_proj":  "mlp",
    "Sigmoid":    "mlp",
    "mlp.Mul":    "mlp",
    # Untracked-by-original-buckets:
    "Add":        "residual+rope+mask",
    "Mul":        "rope",
    "Neg":        "rope",
    "Concat":     "rope",
    "Transpose":  "rope",
    "View":       "view",
    "CastType":   "cast",
    "ReduceMin":  "misc",
    "q_norm":     "qk_norm",
    "k_norm":     "qk_norm",
    "input_layernorm":           "layernorm",
    "post_attention_layernorm":  "layernorm",
}


def classify_full(sub):
    """Map an op to a bucket; return None to skip."""
    if "mlp.Mul" in sub:                                  return "mlp"
    if "self_attn.MatMul" in sub:                          return "matmul (Q@K^T + attn@V)"
    if "self_attn.Add" in sub:                             return "residual+rope+mask"
    if "self_attn.Mul" in sub or "self_attn.Neg" in sub:   return "rope"
    if "self_attn.Concat" in sub:                          return "rope"
    if "self_attn.Transpose" in sub:                       return "rope"
    if sub.startswith("self_attn.q_proj"):                 return "q_proj"
    if sub.startswith("self_attn.k_proj"):                 return "k_proj"
    if sub.startswith("self_attn.v_proj"):                 return "v_proj"
    if sub.startswith("self_attn.o_proj"):                 return "o_proj"
    if sub.startswith("self_attn.Softmax"):                return "softmax"
    if sub.startswith(("self_attn.Where",
                       "self_attn.Equal",
                       "self_attn.Slice")):                return "mask"
    if sub.startswith(("self_attn.q_norm",
                       "self_attn.k_norm")):               return "qk_norm"
    if sub.startswith(("mlp.gate_proj",
                       "mlp.up_proj",
                       "mlp.down_proj",
                       "mlp.Sigmoid")):                    return "mlp"
    if sub == "input_layernorm" or sub.startswith("input_layernorm"):
        return "layernorm"
    if sub == "post_attention_layernorm" or sub.startswith("post_attention_layernorm"):
        return "layernorm"
    if sub == "Add" or sub.startswith("Add"):
        return "residual"
    if "View" in sub or "CastType" in sub or "ReduceMin" in sub:
        return None
    return "other"


def main():
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

    sum_us = sum(blocks[i - 1][0] for i in DECODE_BLOCKS)
    sum_cy = sum(blocks[i - 1][1] for i in DECODE_BLOCKS)
    upc = sum_us / sum_cy
    print(f"Decode phase: {len(DECODE_BLOCKS)} blocks (6..33), "
          f"sum_us={sum_us:,}, sum_cyc={sum_cy:,}, us/cyc={upc:.6f}\n")

    # Per-layer per-call cycles (mean across 28 blocks)
    per_layer_cycles_per_call = defaultdict(int)
    per_layer_total_cycles    = defaultdict(int)
    for L in range(28):
        for i in DECODE_BLOCKS:
            cyc = sum(c for ll, sub, c in blocks[i - 1][2] if ll == L)
            per_layer_total_cycles[L] += cyc
        per_layer_cycles_per_call[L] = per_layer_total_cycles[L] / len(DECODE_BLOCKS)

    print(f"{'layer':>5} {'per-call us':>14} {'×28 layers':>14} {'×29 tokens (proxy)':>20}")
    grand_total_per_call = 0
    for L in range(28):
        per_call_us = per_layer_cycles_per_call[L] * upc
        grand_total_per_call += per_call_us
        print(f"{L:>5} {per_call_us:>14.1f} {per_call_us*28:>14.1f} {per_call_us*28*29:>20.1f}")

    print()
    avg = grand_total_per_call / 28
    total_per_token = grand_total_per_call          # already sum across 28 layers
    total_28_layers_29_tokens = total_per_token * 29
    print(f"avg per-layer per-call          : {avg:>12,.1f} μs")
    print(f"sum 28 layers, one decode call  : {total_per_token:>12,.1f} μs  "
          f"(time spent inside the 28 layers per token)")
    print(f"× 29 decode tokens              : {total_28_layers_29_tokens:>12,.1f} μs  "
          f"(= {total_28_layers_29_tokens/1e6:.2f} s)")

    # Compare to actual decode wall (sum of accel_us for blocks 6-33)
    print(f"\nactual NPU wall (sum block 6-33 accel_us)  : {sum_us:>12,.0f} μs  "
          f"(= {sum_us/1e6:.2f} s)")
    print(f"live perf=true reported decode             : 9,146,332 μs  (= 9.15 s)")


if __name__ == "__main__":
    main()
