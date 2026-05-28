# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Heterogeneous-pipeline schedule simulator for the Qwen3-1.7B MLP.
#
# Given measured per-(op, engine) latencies + transfer costs (qwen3_1p7b_mlp.py),
# enumerate engine assignments and tile counts, simulate the timeline respecting
# DAG deps + engine occupancy + cross-engine transfers, and report the predicted
# makespan for each config vs the fused-on-NPU baseline.
#
# Scope (per request):
#   - No NPU dispatch overhead modeled.
#   - No custom kernels — only measured ops.
#   - Qwen3-1.7B shapes only.
#   - Matmuls (gr, dn) pinned to NPU (no LPBQ on CPU/GPU).
#   - Free choices: engine for `silu` and `gateup_mul`, tile count T, thread model.
#
# Usage:
#   python het_sim.py validate                # re-predict configs we measured
#   python het_sim.py sweep [--total 1024]   # enumerate all configs, rank by makespan
#   python het_sim.py one --silu GPU --gateup_mul CPU --T 2 --mode threaded

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qwen3_1p7b_mlp import SPEC

ENGINES = ("NPU", "CPU", "GPU")


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------

def latency_ms(op: str, engine: str, Sq: int) -> float:
    """Latency of `op` on `engine` for tile size Sq (tokens), in ms."""
    if op in SPEC["npu_matmul_latency_ms"]:
        assert engine == "NPU", f"{op} is NPU-only (LPBQ matmul)"
        table = SPEC["npu_matmul_latency_ms"][op]
        if Sq in table:
            return table[Sq]
        # Linear interp / extrapolation by nearest measurement
        keys = sorted(table.keys())
        if Sq < keys[0]:
            return table[keys[0]] * Sq / keys[0]
        if Sq > keys[-1]:
            return table[keys[-1]] * Sq / keys[-1]
        for i, k in enumerate(keys):
            if k >= Sq:
                k0, k1 = keys[i - 1], k
                t = (Sq - k0) / (k1 - k0)
                return table[k0] + t * (table[k1] - table[k0])
    if op in SPEC["activation"]:
        info = SPEC["activation"][op]
        rate = info["rate_gbps"][engine]
        bufs = info["buffers"]
        I = SPEC["shape"]["I"]
        bytes_streamed = bufs * Sq * I * 2  # fp16
        return bytes_streamed / (rate * 1e9) * 1000.0
    raise ValueError(f"Unknown op {op!r}")


def transfer_ms(src: str, dst: str, n_bytes: int) -> float:
    """Cost to move `n_bytes` from engine `src` to engine `dst` (ms)."""
    if src == dst:
        return 0.0
    rate = SPEC["transfer_ms_per_mb"].get((src, dst))
    if rate is None:
        raise ValueError(f"No transfer cost for {src}->{dst}")
    return rate * (n_bytes / 1e6)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class Event:
    tile: int
    op: str
    engine: str
    deps: List["Event"] = field(default_factory=list)
    latency: float = 0.0
    output_bytes: int = 0
    start: float = -1.0
    end: float = -1.0
    timeline: str = ""

    def label(self) -> str:
        return f"t{self.tile}.{self.op}"


def engines_to_timelines(num_timelines: int) -> Dict[str, str]:
    """Map each engine to a timeline name. Timelines serialize (one task at a time)."""
    if num_timelines == 1:
        return {e: "main" for e in ENGINES}     # everything serial on main thread
    if num_timelines == 2:
        return {"NPU": "main", "CPU": "worker", "GPU": "worker"}  # NPU vs activation worker
    if num_timelines == 3:
        return {e: e for e in ENGINES}          # theoretical max overlap
    raise ValueError(f"num_timelines must be 1/2/3, got {num_timelines}")


def fused_npu_latency_ms(Sq: int) -> float:
    """Measured fused-on-NPU MLP latency at Sq (the single-dispatch baseline)."""
    table = SPEC["fused_npu_reference_ms"]
    if Sq in table:
        return table[Sq]
    keys = sorted(table.keys())
    if Sq < keys[0]:
        return table[keys[0]] * Sq / keys[0]
    if Sq > keys[-1]:
        return table[keys[-1]] * Sq / keys[-1]
    for i, k in enumerate(keys):
        if k >= Sq:
            k0, k1 = keys[i - 1], k
            t = (Sq - k0) / (k1 - k0)
            return table[k0] + t * (table[k1] - table[k0])


def simulate(Sq: int, T: int,
             assignment: Dict[str, str],
             num_timelines: int = 2,
             apply_cpu_worker_penalty: bool = True) -> dict:
    """Simulate one schedule. Returns {makespan, busy[timeline], events, assignment}.

    All-NPU assignments hit a special path that uses the *measured fused* MLP latency
    (one NPU dispatch with intermediates in VTCM), since summing individual op latencies
    badly over-counts that case — VTCM fusion is exactly what the NPU graph compiler buys.
    """
    # Special case: all activations on NPU → one fused dispatch per tile.
    is_fused = (assignment.get("silu", "NPU") == "NPU" and
                assignment.get("gateup_mul", "NPU") == "NPU")
    if is_fused:
        per_tile = fused_npu_latency_ms(Sq)
        # Build a single "fused" event per tile for the timeline/report.
        events: List[Event] = []
        t = 0.0
        for tile in range(T):
            ev = Event(tile=tile, op="fused_mlp", engine="NPU",
                       latency=per_tile, output_bytes=SPEC["output_bytes_per_token"]["o"] * Sq,
                       timeline="main")
            ev.start = t
            ev.end = t + per_tile
            t = ev.end
            events.append(ev)
        return {"makespan": t, "busy": {"main": T * per_tile}, "events": events,
                "op_engine": {"fused_mlp": "NPU"}, "Sq": Sq, "T": T,
                "num_timelines": num_timelines, "fused": True}

    e2t = engines_to_timelines(num_timelines)
    timeline_free: Dict[str, float] = {tl: 0.0 for tl in set(e2t.values())}
    timeline_busy: Dict[str, float] = {tl: 0.0 for tl in timeline_free}

    op_engine = {op: ("NPU" if "NPU" in SPEC["ops"][op]["engines"] and
                              SPEC["ops"][op]["engines"] == ["NPU"]
                      else assignment[op])
                 for op in SPEC["op_order"]}
    # Matmuls pinned to NPU regardless of assignment
    op_engine["gr"] = "NPU"
    op_engine["dn"] = "NPU"

    def lat(op: str, engine: str) -> float:
        x = latency_ms(op, engine, Sq)
        if apply_cpu_worker_penalty and engine == "CPU" and e2t["CPU"] == "worker":
            x *= SPEC.get("cpu_worker_penalty", 1.0)
        return x

    # Build events for each tile.
    events: List[Event] = []
    tile_events: Dict[Tuple[int, str], Event] = {}
    for tile in range(T):
        for op in SPEC["op_order"]:
            engine = op_engine[op]
            deps_events: List[Event] = []
            for in_name in SPEC["ops"][op]["deps"]:
                # find producer event for this tensor in this tile
                producer = None
                for prev_op in SPEC["op_order"]:
                    if in_name in SPEC["ops"][prev_op].get("outputs", []):
                        producer = tile_events.get((tile, prev_op))
                        break
                if producer is not None:
                    deps_events.append(producer)
            out_bytes = sum(SPEC["output_bytes_per_token"][o] * Sq
                            for o in SPEC["ops"][op]["outputs"])
            ev = Event(tile=tile, op=op, engine=engine, deps=deps_events,
                       latency=lat(op, engine), output_bytes=out_bytes,
                       timeline=e2t[engine])
            tile_events[(tile, op)] = ev
            events.append(ev)

    # Scheduling: pick events in the order our real code uses for the threaded
    # pipeline: all `gr` first across tiles, then activations across tiles, then
    # all `dn` (this maximizes overlap between NPU matmuls and the worker's
    # activation chain). For 1-timeline (serial) and 3-timeline (max overlap)
    # cases the same order still works — list-scheduling respects engine free
    # times so ops backfill naturally.
    schedule_order: List[Event] = []
    for tile in range(T):
        schedule_order.append(tile_events[(tile, "gr")])
    for tile in range(T):
        schedule_order.append(tile_events[(tile, "silu")])
        schedule_order.append(tile_events[(tile, "gateup_mul")])
    for tile in range(T):
        schedule_order.append(tile_events[(tile, "dn")])

    for ev in schedule_order:
        deps_ready = max((d.end for d in ev.deps), default=0.0)
        # Transfer in from each dep's engine to this op's engine.
        xfer = 0.0
        for d in ev.deps:
            if d.engine != ev.engine:
                # Per-input bytes: the output of `d` carries multiple tensors; here
                # treat the dep's output as the data to move. (For `gr` which has
                # two outputs g,u, each downstream op reads only one — model as
                # half the total. Close enough for this granularity.)
                bytes_for_this_input = d.output_bytes // max(1, len(SPEC["ops"][d.op]["outputs"]))
                xfer += transfer_ms(d.engine, ev.engine, bytes_for_this_input)
        ev.start = max(deps_ready + xfer, timeline_free[ev.timeline])
        ev.end = ev.start + ev.latency
        timeline_free[ev.timeline] = ev.end
        timeline_busy[ev.timeline] += ev.latency

    makespan = max(ev.end for ev in events)
    return {"makespan": makespan, "busy": timeline_busy, "events": events,
            "op_engine": op_engine, "Sq": Sq, "T": T, "num_timelines": num_timelines}


# ---------------------------------------------------------------------------
# Per-engine reporting + ascii Gantt
# ---------------------------------------------------------------------------

def engine_breakdown(result: dict) -> Dict[str, Tuple[float, float]]:
    """Per-engine (busy, total_busy_time) — sum events per engine."""
    out: Dict[str, float] = {e: 0.0 for e in ENGINES}
    for ev in result["events"]:
        out[ev.engine] += ev.latency
    return out


def ascii_gantt(result: dict, width: int = 60) -> str:
    """Compact text Gantt: one line per timeline."""
    span = result["makespan"] or 1e-9
    lines = []
    timelines = sorted({ev.timeline for ev in result["events"]})
    for tl in timelines:
        row = [" "] * width
        for ev in result["events"]:
            if ev.timeline != tl:
                continue
            s = int(ev.start / span * width)
            e = max(int(ev.end / span * width), s + 1)
            ch = ev.op[0].upper()
            for i in range(s, min(e, width)):
                row[i] = ch
        lines.append(f"  {tl:>7}: |{''.join(row)}|")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: validate / sweep / one
# ---------------------------------------------------------------------------

VALIDATION_CONFIGS = [
    # (label, Sq, T, assignment, num_timelines, measured_ms, note)
    ("fused-NPU @1024",         1024, 1, {"silu": "NPU", "gateup_mul": "NPU"}, 1, 12.5, "LPBQ microbench full"),
    ("fused-NPU @512 (1 tile)",  512, 1, {"silu": "NPU", "gateup_mul": "NPU"}, 1,  2.68, "LPBQ microbench full"),
    ("in-graph tiled 2x512",     512, 2, {"silu": "NPU", "gateup_mul": "NPU"}, 1,  5.36, "= 2 x fused-512 (microbench)"),
    ("CPU mul-only @1024",      1024, 1, {"silu": "NPU", "gateup_mul": "CPU"}, 1, 12.5, "mul-only break-even"),
    ("CPU silu+2mul @1024",     1024, 1, {"silu": "CPU", "gateup_mul": "CPU"}, 1, 30.0, "loses (scalar expf)"),
    ("GPU silu, CPU mul @1024", 1024, 1, {"silu": "GPU", "gateup_mul": "CPU"}, 1, 12.35, "3-way budget"),
    ("tiled (1-thread) 2x512",   512, 2, {"silu": "GPU", "gateup_mul": "CPU"}, 1, 12.8, "het_mlp --tiled"),
    ("threaded 2x512",           512, 2, {"silu": "GPU", "gateup_mul": "CPU"}, 2, 14.0, "het_mlp --threaded (throttled ~11-21)"),
    ("threaded 4x256",           256, 4, {"silu": "GPU", "gateup_mul": "CPU"}, 2, 17.3, "het_mlp --threaded"),
    ("threaded 8x128",           128, 8, {"silu": "GPU", "gateup_mul": "CPU"}, 2, 23.2, "het_mlp --threaded"),
]


def cmd_validate(_args):
    print("Predicted vs measured — qwen3-1.7B MLP, 1024 tok\n")
    fmt = "  {:<26} {:>8} {:>8} {:>8}  {}"
    print(fmt.format("config", "pred(ms)", "meas(ms)", "Δ%", "note"))
    print("  " + "-" * 78)
    for label, Sq, T, assign, ntl, measured, note in VALIDATION_CONFIGS:
        r = simulate(Sq, T, assign, num_timelines=ntl)
        pred = r["makespan"]
        delta_pct = (pred - measured) / measured * 100.0
        print(fmt.format(label, f"{pred:.2f}", f"{measured:.2f}", f"{delta_pct:+.0f}%", note))
    print()
    print("Notes: predictions exclude thermal throttling (measured ~+30% when all 3 engines hot)")
    print("       and OpenCL kernel JIT (~35 ms one-time, amortized in warm reps).")


def cmd_sweep(args):
    total = args.total
    rows = []
    for T in [1, 2, 4, 8, 16]:
        Sq = total // T
        if Sq < 64 or Sq * T != total:
            continue
        for silu_e, gu_e in itertools.product(ENGINES, repeat=2):
            for ntl in [1, 2, 3]:
                is_het = not (silu_e == "NPU" and gu_e == "NPU")
                r = simulate(Sq, T, {"silu": silu_e, "gateup_mul": gu_e},
                             num_timelines=ntl)
                rows.append({"T": T, "Sq": Sq, "silu": silu_e, "gu": gu_e,
                             "ntl": ntl, "makespan": r["makespan"], "het": is_het})
    rows.sort(key=lambda r: r["makespan"])

    # Best all-NPU configuration (the real baseline to beat — fused-tiled wins among them).
    all_npu = [r for r in rows if not r["het"]]
    best_fused = min(all_npu, key=lambda r: r["makespan"]) if all_npu else None
    best_het = min((r for r in rows if r["het"]), key=lambda r: r["makespan"])
    base = best_fused["makespan"] if best_fused else 0
    print(f"Sweep for {total} tokens.")
    print(f"  best all-NPU (fused-tiled):  {base:.3f} ms  "
          f"(T={best_fused['T']}, Sq={best_fused['Sq']})")
    print(f"  best heterogeneous (optimistic, no thermal): {best_het['makespan']:.3f} ms  "
          f"(T={best_het['T']}, silu={best_het['silu']}, gu={best_het['gu']}, ntl={best_het['ntl']})")
    print(f"  → het lower bound is {best_het['makespan']/base:.2f}x fused-tiled "
          f"({'WINS' if best_het['makespan'] < base else 'LOSES'})\n")

    print("  rank  type    T  Sq    silu  gu    ntl  makespan(ms)  vs fused-tiled")
    print("  " + "-" * 73)
    for i, r in enumerate(rows[:args.top], 1):
        kind = "het " if r["het"] else "fused"
        ratio = r["makespan"] / base
        mark = " ✓" if r["makespan"] < base else ""
        print(f"  {i:>4}  {kind}  {r['T']:<2} {r['Sq']:<4}  {r['silu']:<5} {r['gu']:<5} "
              f"{r['ntl']:<3}  {r['makespan']:>10.3f}    {ratio:>5.2f}x{mark}")
    print(f"\n  (showing top {args.top} of {len(rows)})")
    n_het_win = sum(1 for r in rows if r["het"] and r["makespan"] < base)
    print(f"  heterogeneous configs predicted to BEAT fused-tiled "
          f"(at the optimistic no-thermal lower bound): {n_het_win} / {sum(1 for r in rows if r['het'])}")


def cmd_one(args):
    Sq = args.total // args.T if args.Sq is None else args.Sq
    assign = {"silu": args.silu, "gateup_mul": args.gateup_mul}
    r = simulate(Sq, args.T, assign, num_timelines=args.mode)
    print(f"Sq={Sq}  T={args.T}  total={Sq*args.T}  silu={args.silu}  "
          f"gateup_mul={args.gateup_mul}  mode={args.mode}-thread(s)")
    print(f"  makespan: {r['makespan']:.3f} ms")
    print(f"  per-timeline busy:", {k: f"{v:.3f}" for k, v in r["busy"].items()})
    print(f"  per-engine busy:",
          {k: f"{v:.3f}" for k, v in engine_breakdown(r).items()})
    print(f"  Gantt:\n{ascii_gantt(r)}")
    fused = SPEC["fused_npu_reference_ms"].get(Sq * args.T, "?")
    print(f"  vs fused-NPU @ {Sq*args.T}: {fused} ms")


# ---------------------------------------------------------------------------
# BLOCK-LEVEL simulator
#
# Extends the MLP-only sim to a full decoder block (excluding the GQA core).
# Uses the kind-dispatched latency model below — same DAG/scheduler machinery
# as the MLP path, but the SPEC is qwen3_1p7b_block.SPEC and ops carry a
# `kind` attribute (matmul / norm / rope / add / act / gqa).
#
# Het sweep knobs (grouped — same engine for ops sharing kind):
#   norm_engine, rope_engine, add_engine, silu_engine, gu_engine, T, timelines
# Plus GQA scheduling mode + per-tile cost:
#   gqa_mode: "tile" (per-tile pipelined) | "barrier" (one barrier after all qkv/rope)
#   gqa_ms_per_tile: float (default 0 — pretend GQA is free)
# ---------------------------------------------------------------------------

def _interp_table(table, Sq):
    """Linear interp / nearest-extrap by Sq into a {Sq: ms} measured table."""
    if Sq in table:
        return table[Sq]
    keys = sorted(table.keys())
    if Sq < keys[0]:
        return table[keys[0]] * Sq / keys[0]
    if Sq > keys[-1]:
        return table[keys[-1]] * Sq / keys[-1]
    for i, k in enumerate(keys):
        if k >= Sq:
            k0, k1 = keys[i - 1], k
            t = (Sq - k0) / (k1 - k0)
            return table[k0] + t * (table[k1] - table[k0])


def block_latency_ms(spec, op_name, engine, Sq, gqa_ms_per_tile=0.0):
    """Latency of `op_name` on `engine` for tile size Sq (block-level spec)."""
    info = spec["ops"][op_name]
    kind = info.get("kind")
    H = spec["shape"]["H"]
    I = spec["shape"]["I"]
    D = spec["shape"]["head_dim"]

    if kind == "matmul":
        assert engine == "NPU", f"{op_name} is NPU-only (LPBQ matmul)"
        return _interp_table(spec["npu_matmul_latency_ms"][op_name], Sq)

    if kind == "norm":
        width = info["width"]
        rate_table = spec["norm_rate_gbps"].get(width)
        if rate_table is None:
            # Fall back to the closest measured width.
            widths = sorted(spec["norm_rate_gbps"].keys())
            closest = min(widths, key=lambda w: abs(w - width))
            rate_table = spec["norm_rate_gbps"][closest]
        rate = rate_table[engine]
        bufs = spec["norm_buffers"]
        # For q_norm/k_norm (per-head): #rows = Sq * heads, width = head_dim.
        # We model it via bytes_streamed: width is the inner dim, heads scale rows.
        # If "heads" present, multiply Sq by heads to get total rows.
        rows = Sq * (info.get("heads", 1) if width == D else 1)
        # Special case: q_norm / k_norm width is head_dim with #rows = Sq * heads.
        # In our spec, q_norm/k_norm width=128=D, no `heads` attribute — derive
        # from op name.
        if width == D and op_name in ("q_norm", "k_norm"):
            rows = Sq * (spec["shape"]["n_q_heads"] if op_name == "q_norm"
                         else spec["shape"]["n_kv_heads"])
        bytes_streamed = bufs * rows * width * 2
        return bytes_streamed / (rate * 1e9) * 1000.0

    if kind == "rope":
        rate = spec["rope_rate_gbps"][engine]
        bufs = spec["rope_buffers"]
        heads = info["heads"]
        bytes_streamed = bufs * heads * Sq * D * 2
        ms = bytes_streamed / (rate * 1e9) * 1000.0
        # GPU rope in real driver pays extra rpcmem ↔ heap memcpys per op
        # beyond what the elementwise bench measured. See spec comment.
        # Caller can set spec["_zero_copy"] = True to use the validated
        # zero-copy overhead (≈0) instead of the copy-path 0.6 ms/op.
        if engine == "GPU":
            key = "gpu_x2x_overhead_ms_per_op_zerocopy" if spec.get("_zero_copy") \
                  else "gpu_x2x_overhead_ms_per_op"
            ms += spec.get(key, 0.0)
        return ms

    if kind == "add":
        width = info["width"]
        rate_table = spec["add_rate_gbps"].get(width)
        if rate_table is None:
            widths = sorted(spec["add_rate_gbps"].keys())
            closest = min(widths, key=lambda w: abs(w - width))
            rate_table = spec["add_rate_gbps"][closest]
        rate = rate_table[engine]
        bufs = spec["add_buffers"]
        bytes_streamed = bufs * Sq * width * 2
        return bytes_streamed / (rate * 1e9) * 1000.0

    if kind == "act":
        act = spec["activation"][op_name]
        rate = act["rate_gbps"][engine]
        bufs = act["buffers"]
        bytes_streamed = bufs * Sq * I * 2
        return bytes_streamed / (rate * 1e9) * 1000.0

    if kind == "gqa":
        # Virtual op — cost is the per-tile GQA estimate provided by caller.
        return gqa_ms_per_tile

    raise ValueError(f"Unknown op kind {kind!r} for op {op_name!r}")


def simulate_block(spec, Sq, T, assignment,
                   num_timelines=2,
                   gqa_mode="tile",
                   gqa_ms_per_tile=0.0,
                   apply_cpu_worker_penalty=True,
                   schedule="tile-major"):
    """Simulate one block schedule with grouped het assignment.

    assignment: dict with engine choice per kind, e.g.
      {"norm": "NPU", "rope": "NPU", "add": "CPU", "silu": "NPU", "gateup_mul": "NPU"}
    Matmuls are pinned to NPU regardless.

    gqa_mode:
      - "tile":    per-tile pipeline — gqa[t] runs after qr/kr/v[t]; o_proj[t]
                   waits on gqa[t]. Standard tile-level pipelining.
      - "barrier": one barrier — gqa_all runs after ALL tiles' qkv/rope finish,
                   then all o_proj's can run. Matches the sparse-attn pattern
                   where the scorer needs the full K context before launching
                   per-block GQA.

    All-NPU MLP special case: if silu+gateup_mul are both NPU, the MLP portion
    of each tile uses the measured fused_mlp_npu_ms (single-dispatch VTCM
    fusion) instead of summing gr+silu+gateup+dn.
    """
    # Per-op engine resolution.
    kind_to_engine_key = {
        "norm": "norm", "rope": "rope", "add": "add",
        "act": None,    # silu / gateup_mul handled individually
        "matmul": None, # NPU-pinned
        "gqa": None,    # NPU-only
    }
    op_engine = {}
    for op, info in spec["ops"].items():
        kind = info["kind"]
        if kind == "matmul" or kind == "gqa":
            op_engine[op] = "NPU"
        elif kind == "act":
            op_engine[op] = assignment.get(op, "NPU")
        else:
            key = kind_to_engine_key[kind]
            op_engine[op] = assignment.get(key, "NPU")

    e2t = engines_to_timelines(num_timelines)
    timeline_free = {tl: 0.0 for tl in set(e2t.values())}
    timeline_busy = {tl: 0.0 for tl in timeline_free}

    is_mlp_fused = (op_engine["silu"] == "NPU" and op_engine["gateup_mul"] == "NPU")
    fused_mlp_per_tile = _interp_table(spec["fused_mlp_npu_ms"], Sq) if is_mlp_fused else None

    def lat(op, engine):
        x = block_latency_ms(spec, op, engine, Sq, gqa_ms_per_tile=gqa_ms_per_tile)
        if apply_cpu_worker_penalty and engine == "CPU" and e2t["CPU"] == "worker":
            x *= spec.get("cpu_worker_penalty", 1.0)
        return x

    # Build events for each tile. For tile-pipeline gqa mode, gqa is per-tile.
    # For barrier mode, gqa is replaced by a single gqa_all event (cost T·per_tile).
    events = []
    tile_events = {}  # (tile, op) -> Event

    # Map each tensor name to its producer op (for cross-op dep wiring).
    produced_by = {}
    for op, info in spec["ops"].items():
        for out in info["outputs"]:
            produced_by[out] = op

    for tile in range(T):
        for op in spec["op_order"]:
            info = spec["ops"][op]
            engine = op_engine[op]
            # In barrier mode, the per-tile gqa is replaced by deps on a global
            # gqa barrier; we still create a placeholder event but with 0 latency
            # and rewire deps below.
            if op == "gqa" and gqa_mode == "barrier":
                continue
            deps_events = []
            for in_name in info["deps"]:
                producer = produced_by.get(in_name)
                if producer is None:
                    continue
                pe = tile_events.get((tile, producer))
                if pe is not None:
                    deps_events.append(pe)
            out_bytes = sum(spec["output_bytes_per_token"][o] * Sq
                            for o in info["outputs"])
            # MLP-fused special case: when silu+gateup_mul are both NPU, we
            # collapse the gr→silu→gateup→dn chain to one event using the
            # measured fused MLP latency. Implemented by: gr inherits the
            # fused cost, dn becomes 0; silu/gateup_mul become 0. (The chain
            # already serializes on NPU, so total = fused number, correct.)
            this_lat = lat(op, engine)
            if is_mlp_fused:
                if op == "gr":
                    this_lat = fused_mlp_per_tile
                elif op in ("silu", "gateup_mul", "dn"):
                    this_lat = 0.0
            ev = Event(tile=tile, op=op, engine=engine, deps=deps_events,
                       latency=this_lat, output_bytes=out_bytes,
                       timeline=e2t[engine])
            tile_events[(tile, op)] = ev
            events.append(ev)

    # Barrier-mode GQA: one event after ALL qr/kr/v across tiles, cost T*per_tile.
    if gqa_mode == "barrier":
        barrier_deps = []
        for tile in range(T):
            for need in ("qr", "kr", "v"):
                pop = produced_by[need]
                if (tile, pop) in tile_events:
                    barrier_deps.append(tile_events[(tile, pop)])
        gqa_ev = Event(tile=-1, op="gqa_all", engine="NPU", deps=barrier_deps,
                       latency=T * gqa_ms_per_tile,
                       output_bytes=spec["output_bytes_per_token"]["_attn_out"] * Sq * T,
                       timeline=e2t["NPU"])
        events.append(gqa_ev)
        # Each tile's o_proj must wait on the barrier (in addition to its own
        # _attn_out producer, which doesn't exist in barrier mode — we make
        # o_proj depend directly on the barrier).
        for tile in range(T):
            o_ev = tile_events[(tile, "o_proj")]
            o_ev.deps = [gqa_ev]

    # Schedule order — controls the priority list for resource conflicts:
    #   "op-major"   : in_norm_0, in_norm_1, q_proj_0, q_proj_1, ...
    #                  Each engine batches all tiles of one op before moving on.
    #                  Predicts the OPTIMAL upper-bound overlap that a perfect
    #                  work-stealing executor could achieve. Doesn't reflect the
    #                  way our threaded drivers actually loop.
    #   "tile-major" : in_norm_0 → q_proj_0 → ... → res2_0 → in_norm_1 → ...
    #                  Each thread completes tile t fully before starting tile
    #                  t+1's first op. Matches mllm-het-block / mllm-het-mlp
    #                  --threaded loop structure (one cv-signaled chain per
    #                  thread, sequential across tiles). REALISTIC bound for
    #                  the way we actually build pipelines.
    # Diff at T=2, Sq=512 for the top block het config: op-major ~11.0 ms,
    # tile-major ~12.8 ms, real measured 17.4 ms. The 1.8 ms tile-major delta
    # is the dep-chain fill the realistic schedule loses; the remaining ~4.5 ms
    # to measured is OMP/transfer/sync overhead (separate spec adjustments).
    schedule_order = []
    if schedule == "op-major":
        for op in spec["op_order"]:
            if op == "gqa" and gqa_mode == "barrier":
                continue
            for tile in range(T):
                schedule_order.append(tile_events[(tile, op)])
    elif schedule == "tile-major":
        for tile in range(T):
            for op in spec["op_order"]:
                if op == "gqa" and gqa_mode == "barrier":
                    continue
                schedule_order.append(tile_events[(tile, op)])
    else:
        raise ValueError(f"schedule must be 'op-major' or 'tile-major', got {schedule!r}")
    if gqa_mode == "barrier":
        # Insert the barrier event after all qr/kr/v's, before o_proj's.
        cut = next(i for i, e in enumerate(schedule_order)
                   if e.op == "o_proj")
        schedule_order.insert(cut, gqa_ev)

    for ev in schedule_order:
        deps_ready = max((d.end for d in ev.deps), default=0.0)
        xfer = 0.0
        for d in ev.deps:
            if d.engine != ev.engine:
                bytes_for_this_input = d.output_bytes // max(1, len(spec["ops"].get(d.op, {"outputs": ["x"]})["outputs"]) if d.op in spec["ops"] else 1)
                xfer += transfer_ms(d.engine, ev.engine, bytes_for_this_input)
        ev.start = max(deps_ready + xfer, timeline_free[ev.timeline])
        ev.end = ev.start + ev.latency
        timeline_free[ev.timeline] = ev.end
        timeline_busy[ev.timeline] += ev.latency

    makespan = max(ev.end for ev in events)
    return {"makespan": makespan, "busy": timeline_busy, "events": events,
            "op_engine": op_engine, "Sq": Sq, "T": T,
            "num_timelines": num_timelines, "gqa_mode": gqa_mode,
            "gqa_ms_per_tile": gqa_ms_per_tile, "fused_mlp": is_mlp_fused,
            "schedule": schedule}


# ---------------------------------------------------------------------------
# Block CLI: block-one / block-sweep
# ---------------------------------------------------------------------------

def cmd_block_one(args):
    from qwen3_1p7b_block import SPEC as BLOCK_SPEC
    BLOCK_SPEC = dict(BLOCK_SPEC)
    BLOCK_SPEC["_zero_copy"] = args.zero_copy
    Sq = args.total // args.T if args.Sq is None else args.Sq
    assignment = {
        "norm": args.norm, "rope": args.rope, "add": args.add,
        "silu": args.silu, "gateup_mul": args.gateup_mul,
    }
    r = simulate_block(BLOCK_SPEC, Sq, args.T, assignment,
                       num_timelines=args.mode,
                       gqa_mode=args.gqa_mode,
                       gqa_ms_per_tile=args.gqa_ms_per_tile,
                       schedule=args.schedule)
    print(f"Sq={Sq}  T={args.T}  total={Sq*args.T}  "
          f"norm={args.norm} rope={args.rope} add={args.add} "
          f"silu={args.silu} gu={args.gateup_mul}  "
          f"mode={args.mode}-thread(s) schedule={args.schedule} "
          f"gqa={args.gqa_mode} gqa_ms/tile={args.gqa_ms_per_tile}")
    print(f"  makespan: {r['makespan']:.3f} ms  (fused_mlp={r['fused_mlp']})")
    print(f"  per-timeline busy:", {k: f"{v:.3f}" for k, v in r["busy"].items()})
    per_engine = {e: 0.0 for e in ENGINES}
    for ev in r["events"]:
        per_engine[ev.engine] += ev.latency
    print(f"  per-engine busy:", {k: f"{v:.3f}" for k, v in per_engine.items()})
    print(f"  Gantt:\n{ascii_gantt(r)}")


def cmd_block_sweep(args):
    from qwen3_1p7b_block import SPEC as BLOCK_SPEC
    BLOCK_SPEC = dict(BLOCK_SPEC)  # shallow copy so we can stamp _zero_copy
    BLOCK_SPEC["_zero_copy"] = args.zero_copy
    total = args.total
    rows = []
    tile_options = [1, 2, 4, 8, 16]
    for T in tile_options:
        Sq = total // T
        if Sq < 64 or Sq * T != total:
            continue
        for norm_e, rope_e, add_e, silu_e, gu_e in itertools.product(ENGINES, repeat=5):
            for ntl in [1, 2, 3]:
                assign = {"norm": norm_e, "rope": rope_e, "add": add_e,
                          "silu": silu_e, "gateup_mul": gu_e}
                r = simulate_block(BLOCK_SPEC, Sq, T, assign,
                                   num_timelines=ntl,
                                   gqa_mode=args.gqa_mode,
                                   gqa_ms_per_tile=args.gqa_ms_per_tile,
                                   schedule=args.schedule)
                is_all_npu = all(v == "NPU" for v in assign.values())
                rows.append({"T": T, "Sq": Sq, "norm": norm_e, "rope": rope_e,
                             "add": add_e, "silu": silu_e, "gu": gu_e,
                             "ntl": ntl, "makespan": r["makespan"],
                             "het": not is_all_npu})
    rows.sort(key=lambda r: r["makespan"])

    all_npu = [r for r in rows if not r["het"]]
    best_fused = min(all_npu, key=lambda r: r["makespan"]) if all_npu else None
    het_rows = [r for r in rows if r["het"]]
    best_het = min(het_rows, key=lambda r: r["makespan"]) if het_rows else None
    base = best_fused["makespan"] if best_fused else 0
    print(f"Block sweep for {total} tokens.  GQA mode={args.gqa_mode} "
          f"gqa_ms/tile={args.gqa_ms_per_tile}")
    print(f"  best all-NPU:                 {base:.3f} ms  "
          f"(T={best_fused['T']}, Sq={best_fused['Sq']})")
    if best_het:
        print(f"  best heterogeneous (lower bound): {best_het['makespan']:.3f} ms  "
              f"(T={best_het['T']}, norm={best_het['norm']}, rope={best_het['rope']}, "
              f"add={best_het['add']}, silu={best_het['silu']}, gu={best_het['gu']}, "
              f"ntl={best_het['ntl']})")
        print(f"  → het lower bound is {best_het['makespan']/base:.2f}x all-NPU "
              f"({'WINS' if best_het['makespan'] < base else 'LOSES'})\n")

    print("  rank  type    T  Sq    norm rope add  silu gu   ntl  makespan(ms)  vs all-NPU")
    print("  " + "-" * 87)
    for i, r in enumerate(rows[:args.top], 1):
        kind = "het " if r["het"] else "fused"
        ratio = r["makespan"] / base if base else 0
        mark = " ✓" if r["makespan"] < base else ""
        print(f"  {i:>4}  {kind}  {r['T']:<2} {r['Sq']:<4}  "
              f"{r['norm']:<4} {r['rope']:<4} {r['add']:<4} {r['silu']:<4} {r['gu']:<4} "
              f"{r['ntl']:<3}  {r['makespan']:>10.3f}    {ratio:>5.2f}x{mark}")
    print(f"\n  (showing top {args.top} of {len(rows)})")
    n_het_win = sum(1 for r in rows if r["het"] and r["makespan"] < base)
    print(f"  heterogeneous configs predicted to BEAT all-NPU "
          f"(at the optimistic lower bound): {n_het_win} / {len(het_rows)}")


def main():
    ap = argparse.ArgumentParser(description="Het schedule simulator (Qwen3-1.7B).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="re-predict MLP configs we measured")

    s = sub.add_parser("sweep", help="MLP-only sweep")
    s.add_argument("--total", type=int, default=1024)
    s.add_argument("--top", type=int, default=15)

    o = sub.add_parser("one", help="simulate one MLP config")
    o.add_argument("--silu", choices=ENGINES, default="NPU")
    o.add_argument("--gateup_mul", choices=ENGINES, default="NPU")
    o.add_argument("--T", type=int, default=1)
    o.add_argument("--Sq", type=int, default=None)
    o.add_argument("--total", type=int, default=1024)
    o.add_argument("--mode", type=int, choices=[1, 2, 3], default=2,
                   help="# parallel timelines: 1=all serial, 2=NPU‖worker, 3=NPU‖CPU‖GPU")

    bs = sub.add_parser("block-sweep", help="enumerate block-level configs")
    bs.add_argument("--total", type=int, default=1024)
    bs.add_argument("--top", type=int, default=15)
    bs.add_argument("--gqa-mode", choices=["tile", "barrier"], default="tile")
    bs.add_argument("--gqa-ms-per-tile", type=float, default=0.0)
    bs.add_argument("--schedule", choices=["op-major", "tile-major"], default="tile-major",
                    help="tile-major (default) matches the real threaded-driver loop "
                         "structure; op-major models the theoretical optimal overlap "
                         "(also matches the zero-copy + pinned driver — validated)")
    bs.add_argument("--zero-copy", action="store_true",
                    help="use the validated zero-copy GPU overhead (~0 ms/op) instead "
                         "of the copy-path default (~0.6 ms/op)")

    bo = sub.add_parser("block-one", help="simulate one block config")
    bo.add_argument("--norm", choices=ENGINES, default="NPU")
    bo.add_argument("--rope", choices=ENGINES, default="NPU")
    bo.add_argument("--add", choices=ENGINES, default="NPU")
    bo.add_argument("--silu", choices=ENGINES, default="NPU")
    bo.add_argument("--gateup_mul", choices=ENGINES, default="NPU")
    bo.add_argument("--T", type=int, default=1)
    bo.add_argument("--Sq", type=int, default=None)
    bo.add_argument("--total", type=int, default=1024)
    bo.add_argument("--mode", type=int, choices=[1, 2, 3], default=2)
    bo.add_argument("--gqa-mode", choices=["tile", "barrier"], default="tile")
    bo.add_argument("--gqa-ms-per-tile", type=float, default=0.0)
    bo.add_argument("--schedule", choices=["op-major", "tile-major"], default="tile-major")
    bo.add_argument("--zero-copy", action="store_true",
                    help="use the validated zero-copy GPU overhead (~0 ms/op)")

    args = ap.parse_args()
    {"validate": cmd_validate, "sweep": cmd_sweep, "one": cmd_one,
     "block-sweep": cmd_block_sweep, "block-one": cmd_block_one}[args.cmd](args)


if __name__ == "__main__":
    main()
