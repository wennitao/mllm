#!/usr/bin/env python3
"""LFM2.5-8B-A1B (LiquidAI `lfm2_moe`) -> mllm V2 converter (standalone, no pymllm).

The checkpoint already stores experts per-expert (`...experts.{e}.w1/w2/w3.weight`)
and uses module names that match the mllm reference 1:1, so the conversion is:
  * cast every weight to fp32
  * squeeze the depthwise conv weight `...conv.conv.weight` [H,1,K] -> [H,K]
    (mllm reads it as a bare [H,K] param)
  * keep all other keys verbatim
  * `lm_head.weight` is absent (tied); the C++ loader aliases model.embed_tokens.weight

We write the mllm V2 binary format directly (magic 0x519A, 532-byte header,
352-byte per-param descriptors), independent of the pymllm native extension.

Usage:
  python convert_lfm2_moe.py convert --src <hf-snapshot-dir> --out lfm2.5-8b-a1b.mllm
  python convert_lfm2_moe.py dump-logits --src <dir> --prompt "..." --out /tmp/lfm2_ref.json
"""

import argparse
import glob
import json
import os
import struct
import sys

# ---- mllm V2 binary format constants ----
_MAGIC = 0x519A
_VERSION = 2
_NAME_LEN = 512
_PNAME_LEN = 256
_SHAPE_LEN = 16
_HDR_SIZE = 532   # <II512sIQ
_DESC_SIZE = 352  # <IIQQQ16i256s
_FP32_CODE = 0    # mllm DataTypes::kFloat32

_DEFAULT_SRC = (
    "/mnt/raid0_ssd/wentao/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/"
    "snapshots/5492b17c7128ec966b5fc661e374ee7edba7423d"
)


def _pack_header(model_name, num_params, params_desc_offset):
    nb = model_name.encode("utf-8")[:_NAME_LEN].ljust(_NAME_LEN, b"\0")
    return struct.pack(f"<II{_NAME_LEN}sIQ", _MAGIC, _VERSION, nb, num_params, params_desc_offset)


def _pack_desc(pid, ptype, psize, poff, shape, name):
    nb = name.encode("utf-8")[:_PNAME_LEN].ljust(_PNAME_LEN, b"\0")
    shp = list(shape) + [0] * (_SHAPE_LEN - len(shape))
    return struct.pack(f"<IIQQQ{_SHAPE_LEN}i{_PNAME_LEN}s", pid, ptype, psize, poff, len(shape), *shp, nb)


def convert(src: str, out: str):
    import numpy as np
    import torch
    from safetensors import safe_open

    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
        files = [f for f in files if os.path.getsize(f) > 1_000_000]
    else:
        files = [src]
    if not files:
        raise FileNotFoundError(f"no .safetensors under {src}")

    # First pass: collect (file, key) of every tensor (all are kept).
    plan = []
    handles = {}
    for fpath in files:
        f = safe_open(fpath, framework="pt")
        handles[fpath] = f
        for key in f.keys():
            if key == "lm_head.weight":
                continue  # tied; the C++ loader aliases embed_tokens
            plan.append((fpath, key))

    n = len(plan)
    squeezed = 0
    with open(out, "wb") as g:
        g.write(b"\x00" * (_HDR_SIZE + n * _DESC_SIZE))
        descs = []
        for pid, (fpath, key) in enumerate(plan):
            arr = handles[fpath].get_tensor(key).to(torch.float32).contiguous().numpy()
            # Depthwise conv weight [H,1,K] -> [H,K] (read as a bare param in mllm).
            if key.endswith("conv.conv.weight") and arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr.reshape(arr.shape[0], arr.shape[2])
                squeezed += 1
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            data = arr.tobytes()
            off = g.tell()
            g.write(data)
            descs.append((pid, _FP32_CODE, len(data), off, list(arr.shape), key))
        for d in descs:
            g.seek(_HDR_SIZE + d[0] * _DESC_SIZE)
            g.write(_pack_desc(*d))
        g.seek(0)
        g.write(_pack_header("lfm2_moe", n, _HDR_SIZE))
    print(f"[convert] kept={n} conv-squeezed={squeezed} -> {out} ({os.path.getsize(out)} bytes)")


def dump_logits(src: str, prompt: str, out: str):
    """Run the HF reference (eager / CPU / fp32) and dump golden last-token logits
    plus per-layer hidden-state fingerprints for localizing a divergence."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.float32, attn_implementation="eager", trust_remote_code=True
    )
    model.eval()
    model.config._attn_implementation = "eager"

    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outp = model(ids, output_hidden_states=True, use_cache=False)
    logits = outp.logits[0, -1].float()
    top = torch.topk(logits, 10)

    # Per-layer fingerprint: L2 norm + first 16 dims of the LAST token.
    hs_fp = []
    for i, hs in enumerate(outp.hidden_states):  # len = num_layers + 1 (embeddings first)
        v = hs[0, -1].float()
        hs_fp.append({"layer": i, "l2": float(v.norm()), "head16": v[:16].tolist()})

    rec = {
        "prompt": prompt,
        "input_ids": ids[0].tolist(),
        "argmax": int(logits.argmax()),
        "top10_ids": top.indices.tolist(),
        "top10_logits": top.values.tolist(),
        "hidden_state_fingerprints": hs_fp,
    }
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    ids_out = os.path.splitext(out)[0] + ".ids.txt"
    with open(ids_out, "w") as f:
        f.write(" ".join(str(i) for i in ids[0].tolist()))
    print(f"[dump-logits] argmax={rec['argmax']} top1={rec['top10_ids'][0]} "
          f"-> {out} (ids -> {ids_out})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", default=_DEFAULT_SRC)
    c.add_argument("--out", default="lfm2.5-8b-a1b.mllm")
    d = sub.add_parser("dump-logits")
    d.add_argument("--src", default=_DEFAULT_SRC)
    d.add_argument("--prompt", default="The capital of France is")
    d.add_argument("--out", default="/tmp/lfm2_ref.json")
    args = ap.parse_args()
    if args.cmd == "convert":
        convert(args.src, args.out)
    elif args.cmd == "dump-logits":
        dump_logits(args.src, args.prompt, args.out)


if __name__ == "__main__":
    sys.exit(main())
