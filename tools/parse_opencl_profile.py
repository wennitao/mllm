"""Aggregate per-Module / per-Layer wall-clock timings from
`--module_profile_path` CSV into the requested attention/MLP buckets.

CSV schema (written by mllm::engine::ModuleProfiler):
    phase,call,module_name,us
    prefill,0,model.layers.0.self_attn.q_proj,612.5
    ...

Bucket mapping (per layer):
    Attention(q_proj)        = self_attn.q_proj
    Attention(k_proj)        = self_attn.k_proj
    Attention(v_proj)        = self_attn.v_proj
    Attention(out)           = self_attn.o_proj
    Attention(mask)          = self_attn.mask
    Attention(softmax)       = self_attn.softmax
    Attention(matmul, est.)  = self_attn  −  (sum of all the above
                                              + q_norm + k_norm + q_rope + k_rope)
                              ≈ the two functional matmuls (Q@K^T and attn@V)
                                plus a bit of scale/view/transpose overhead
    MLP (3 lin + SiLU)       = mlp.gate_proj + mlp.up_proj + mlp.down_proj + mlp.act

Usage:
    python3 tools/parse_opencl_profile.py opencl_profile.csv [--layers 0 1]
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


LAYER_RE = re.compile(r"^model\.layers\.(\d+)(?:\.(.+))?$")

# Specific-leaf names to bucket
ATTN_LEAVES = {
    "self_attn.q_proj":  "q_proj",
    "self_attn.k_proj":  "k_proj",
    "self_attn.v_proj":  "v_proj",
    "self_attn.o_proj":  "o_proj",
    "self_attn.mask":    "mask",
    "self_attn.softmax": "softmax",
}
ATTN_OTHER_LEAVES = {  # extra Qwen3 attention leaves (norm + rope) — count toward
                       # "named leaves" so the residual reflects only matmul
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.q_rope",
    "self_attn.k_rope",
}
MLP_LEAVES = {
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.act",
}
SELF_ATTN_PARENT = "self_attn"
MLP_PARENT = "mlp"


def aggregate(csv_path: Path):
    """Return data[phase][layer][bucket] = total_us."""
    # Per-layer raw sums for everything we care about.
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            phase, call, name, us = r["phase"], int(r["call"]), r["module_name"], float(r["us"])
            m = LAYER_RE.match(name)
            if not m:
                continue
            layer = int(m.group(1))
            sub = m.group(2) or ""
            raw[phase][layer][sub] += us
    return raw


def emit(raw, layers):
    for phase in ["prefill", "decode"]:
        if phase not in raw:
            continue
        for layer in layers:
            if layer not in raw[phase]:
                continue
            sub = raw[phase][layer]
            row = lambda key: sub.get(key, 0.0)

            q_proj  = row("self_attn.q_proj")
            k_proj  = row("self_attn.k_proj")
            v_proj  = row("self_attn.v_proj")
            o_proj  = row("self_attn.o_proj")
            mask    = row("self_attn.mask")
            sm      = row("self_attn.softmax")
            qk_mm   = row("self_attn.qk_matmul")
            av_mm   = row("self_attn.av_matmul")
            other_named = sum(row(k) for k in ATTN_OTHER_LEAVES)

            self_attn_total = row(SELF_ATTN_PARENT)
            named_leaves_total = (q_proj + k_proj + v_proj + o_proj
                                  + mask + sm + qk_mm + av_mm + other_named)
            unaccounted = self_attn_total - named_leaves_total  # ≈ scale + view + transpose + tiny CPU

            mlp_leaves_total = sum(row(k) for k in MLP_LEAVES)

            # If the new MatMul Layers haven't been built yet, fall back to residual
            have_matmul_leaves = (qk_mm > 0) or (av_mm > 0)

            print()
            print(f"{phase.upper()} — Layer {layer}")
            print("-" * 60)
            print(f"  Attention(q_proj)              {q_proj:>10.1f} us")
            print(f"  Attention(k_proj)              {k_proj:>10.1f} us")
            print(f"  Attention(v_proj)              {v_proj:>10.1f} us")
            if have_matmul_leaves:
                print(f"  Attention(matmul Q@K^T)        {qk_mm:>10.1f} us")
                print(f"  Attention(mask)                {mask:>10.1f} us")
                print(f"  Attention(softmax)             {sm:>10.1f} us")
                print(f"  Attention(matmul attn@V)       {av_mm:>10.1f} us")
            else:
                print(f"  Attention(matmul, residual)    {unaccounted:>10.1f} us  "
                      f"(self_attn − named-leaves)")
                print(f"  Attention(mask)                {mask:>10.1f} us")
                print(f"  Attention(softmax)             {sm:>10.1f} us")
            print(f"  Attention(out)                 {o_proj:>10.1f} us")
            print(f"  MLP (3 linear + SiLU)          {mlp_leaves_total:>10.1f} us")
            print(f"  -- info --")
            print(f"    self_attn parent (total)     {self_attn_total:>10.1f} us")
            print(f"    q_norm + k_norm + ropes      {other_named:>10.1f} us  "
                  f"(included in named-leaves total)")
            if have_matmul_leaves:
                print(f"    unaccounted in self_attn     {unaccounted:>10.1f} us  "
                      f"(scale, view, transpose, KV-cache write)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    raw = aggregate(args.csv_path)
    emit(raw, args.layers)


if __name__ == "__main__":
    main()
