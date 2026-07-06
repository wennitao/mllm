#!/bin/bash
# One-shot AI 100 per-op profiler for Qwen3-8B: PREFILL (seq_len=128) + DECODE (seq_len=1).
# Produces two granularities per regime, written under mllm/perop/:
#   * OP LEVEL   -> out/*.qaic-opstats.summary.txt  (per-operator-kind, from qaic-opstats)
#   * ARCH LEVEL -> per_projection_<regime>.txt/.csv (per q/k/v/o/gate/up/down/lm_head)
# Chain per regime: qaic-compile(-stats-level=70) -> qaic-runner(raw_device_stats) -> qaic-opstats -> perop_by_projection.py
set -e
source /home/chihao/qeff-venv/bin/activate 2>/dev/null || true
export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH

# ---- inputs / knobs -------------------------------------------------------
ONNX=${ONNX:-/home/chihao/models/qeff_native_cache/Qwen3ForCausalLM/Qwen3ForCausalLM-d165ffa1b02f89bc/Qwen3ForCausalLM.onnx}
CUSTOM_IO=${CUSTOM_IO:-/home/chihao/models/perop_plain/custom_io.yaml}
CTX=${CTX:-256}          # context length (both regimes)
PREFILL_SEQ=${PREFILL_SEQ:-128}
CORES=${CORES:-16}
DEV=${DEV:-0}            # single card
SAMPLES=${SAMPLES:-4}   # profiling samples to capture/decode per regime
OUTROOT=${OUTROOT:-/home/chihao/mllm/perop}
TOOLS=$(cd "$(dirname "$0")" && pwd)
# ---------------------------------------------------------------------------

run_regime () {   # $1=name  $2=seq_len
  local name=$1 seq=$2          # NOTE: keep D on its own line — `local a=$1 b=$OUTROOT/$a`
  local D="$OUTROOT/$name"      # would expand $a before name=$1 takes effect (bash gotcha)
  echo "############### $name (ctx=$CTX seq_len=$seq) ###############"
  mkdir -p "$D/stats" "$D/out"
  printf '{"specializations":[{"batch_size":"1","ctx_len":"%s","seq_len":"%s"}]}\n' "$CTX" "$seq" > "$D/spec.json"
  cp "$CUSTOM_IO" "$D/"

  if [ -f "$D/qpc/programqpc.bin" ]; then
    echo "[1/4] compile: reusing existing $D/qpc/programqpc.bin"
  else
    echo "[1/4] compile (stats-level=70)"
    qaic-compile -aic-hw -aic-hw-version=ai100 -m="$ONNX" \
      -convert-to-fp16 -mxfp6-matmul -aic-num-cores=$CORES -mos=1 -aic-enable-depth-first \
      -network-specialization-config="$D/spec.json" -custom-IO-list-file="$D/custom_io.yaml" \
      -stats-level=70 -compile-only -aic-binary-dir="$D/qpc"
    [ -f "$D/qpc/programqpc.bin" ] || { echo "COMPILE FAILED ($name)"; exit 1; }
  fi

  echo "[2/4] run + profile (raw_device_stats, $SAMPLES samples, card $DEV)"
  rm -f "$D"/stats/*
  qaic-runner -t "$D/qpc" -d "$DEV" --aic-profiling-type raw_device_stats \
    --aic-profiling-num-samples "$SAMPLES" --aic-profiling-out-dir "$D/stats"

  echo "[3/4] decode OP-LEVEL (qaic-opstats --summary --trace)"
  rm -f "$D"/out/*
  # qaic-opstats can crash enumerating a missing sample slot; it still writes the
  # samples it decoded first, so tolerate a nonzero exit and verify output exists.
  qaic-opstats -q "$D/qpc/programqpc.bin" -i "$D/stats" --summary --trace -o "$D/out" \
    || echo "  (qaic-opstats exited nonzero — proceeding with the samples it decoded)"
  ls "$D"/out/*.qaic-opstats.trace.json >/dev/null 2>&1 \
    || { echo "DECODE FAILED: no trace decoded ($name)"; exit 1; }
  # representative op-level summary copy
  cp "$(ls "$D"/out/*.qaic-opstats.summary.txt | head -1)" "$D/op_level_${name}.summary.txt"

  echo "[4/4] aggregate ARCH-LEVEL (per projection)"
  local TR; TR=$(ls "$D"/out/*.qaic-opstats.trace.json | head -1)
  python3 "$TOOLS/perop_by_projection.py" "$TR" "$D/per_projection_${name}.csv" \
      > "$D/per_projection_${name}.txt"
  echo "  -> $D/per_projection_${name}.txt (+ .csv)"
}

mkdir -p "$OUTROOT"
run_regime prefill "$PREFILL_SEQ"
run_regime decode  1

# ---- combined side-by-side arch-level CSV ---------------------------------
python3 - "$OUTROOT" <<'PY'
import csv, sys, os
root = sys.argv[1]
def load(p):
    d={}
    with open(p) as f:
        for r in csv.DictReader(f):
            d[r['projection']] = r
    return d
pre = load(f"{root}/prefill/per_projection_prefill.csv")
dec = load(f"{root}/decode/per_projection_decode.csv")
keys = [k for k in pre if k!='TOTAL']
out = f"{root}/prefill_vs_decode_by_projection.csv"
with open(out,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["projection","prefill_pcycles","prefill_share%","decode_pcycles","decode_share%"])
    for k in keys:
        w.writerow([k, pre[k]['pcycles'], pre[k]['share_pct'],
                    dec.get(k,{}).get('pcycles',''), dec.get(k,{}).get('share_pct','')])
print("combined:", out)
PY

echo
echo "===== DONE. Outputs under $OUTROOT ====="
find "$OUTROOT" -maxdepth 2 -name 'per_projection_*.txt' -o -name 'op_level_*.summary.txt' -o -name 'prefill_vs_decode_*.csv' | sort
