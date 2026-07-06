#!/usr/bin/env python3
# ARCH-LEVEL aggregation: turn a qaic-opstats trace.json into per-PROJECTION cycle totals.
# The qaic-opstats summary.txt is OP-LEVEL (groups by op kind across the whole graph);
# this recovers the per-linear (q/k/v/o/gate/up/down/lm_head) view from opName ONNX paths.
#
# Usage: perop_by_projection.py <trace.json> [out.csv]
#   prints a human table to stdout; if out.csv given, also writes a CSV.
import re, sys, collections

if len(sys.argv) < 2:
    sys.exit("usage: perop_by_projection.py <trace.json> [out.csv]")
T = sys.argv[1]
CSV = sys.argv[2] if len(sys.argv) > 2 else None

PROJ = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]

name_re = re.compile(r'"opName"\s*:\s*"([^"]*)"')
pc_re   = re.compile(r'"opPCycle"\s*:\s*"(\d+)"')
uc_re   = re.compile(r'"opUCycle"\s*:\s*"(\d+)"')
kind_re = re.compile(r'"opKind"\s*:\s*"([^"]*)"')
add1_re = re.compile(r'/layers\.\d+/Add_1\b')   # MLP residual  <- down_proj matmul named here
add_re  = re.compile(r'/layers\.\d+/Add\b')     # attn residual <- o_proj  matmul named here

def bucket(opname):
    for p in PROJ:
        if "/"+p+"/" in opname or opname.endswith("/"+p) or p == opname:
            return p
    if add1_re.search(opname): return "down_proj"   # exporter names o_proj/down_proj
    if add_re.search(opname):  return "o_proj"       # after their consuming residual-Add
    if "self_attn" in opname:  return "attn_other"
    if "mlp" in opname:        return "mlp_other"
    return "other"

pc, uc, cnt = collections.Counter(), collections.Counter(), collections.Counter()
by_kind = collections.Counter()
tot_pc = 0
with open(T, "r", errors="ignore") as f:
    for line in f:
        if '"opPCycle"' not in line: continue
        m = pc_re.search(line)
        if not m: continue
        p = int(m.group(1))
        nm = name_re.search(line)
        b = bucket(nm.group(1)) if nm else "other"
        k = kind_re.search(line)
        kind = k.group(1).strip() if k else "?"
        pc[b] += p; cnt[b] += 1; tot_pc += p
        um = uc_re.search(line)
        if um: uc[b] += int(um.group(1))
        by_kind[(b, kind)] += p

order = PROJ + ["attn_other","mlp_other","other"]
print(f"# trace: {T.split('/')[-1]}")
print(f"# total opPCycle summed over ALL 16 NSP cores & threads = {tot_pc:,}\n")
print(f"{'projection':<12} {'pcycles':>15} {'share%':>8} {'ucycles':>15} {'op_count':>9}")
print("-"*64)
for b in order:
    if pc[b]==0 and cnt[b]==0: continue
    print(f"{b:<12} {pc[b]:>15,} {100*pc[b]/tot_pc:>7.2f}% {uc[b]:>15,} {cnt[b]:>9,}")
print("-"*64)
print(f"{'TOTAL':<12} {tot_pc:>15,} {'100.00%':>8}")

print("\n# per-projection split by op-kind (pcycles) — top kinds")
for b in order:
    kinds = sorted([(k[1],v) for k,v in by_kind.items() if k[0]==b], key=lambda x:-x[1])
    if not kinds: continue
    tot = sum(v for _,v in kinds)
    print(f"  {b:<12} " + ", ".join(f"{k}={100*v/tot:.0f}%" for k,v in kinds[:4] if v>0))

if CSV:
    with open(CSV, "w") as f:
        f.write("projection,pcycles,share_pct,ucycles,op_count\n")
        for b in order:
            if pc[b]==0 and cnt[b]==0: continue
            f.write(f"{b},{pc[b]},{100*pc[b]/tot_pc:.4f},{uc[b]},{cnt[b]}\n")
        f.write(f"TOTAL,{tot_pc},100.0,,\n")
    print(f"\n# CSV written: {CSV}")
