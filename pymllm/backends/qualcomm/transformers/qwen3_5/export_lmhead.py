#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export the Qwen3.5 head (final RMSNorm + tied lm_head) for the NPU graph
# (compile_lmhead.cpp). Companion to export_{deltanet,attn}_decode.py.
#
#   1. Load HF Qwen3.5-2B, hook lm.norm to grab a real final-decoder hidden state x.
#   2. LPBQ-pack lm_head (= tied embed_tokens, hidden -> vocab) deploy buffers.
#   3. fp32 reference: logits = (norm(x, 1+norm_w) @ W_lmhead.t()) using dequant-int4
#      weights (matches on-device), cross-checked vs HF logits (argmax + max err).
#
#   <out>-lpbq.mllm : model.lm_head.{weight,scale1,scale2}, model.lmhead_*_qdq.*
#   <out>-ref.mllm  : x, norm_w (1+w), eps, exp_logits
#
#   python3 export_lmhead.py --out lmhead
#
import argparse
import os
import struct
import sys
import types

import numpy as np
import torch


_MAGIC, _VERSION = 0x519A, 2
_NAME_LEN, _PNAME_LEN, _SHAPE_LEN = 512, 256, 16
_HDR_SIZE, _DESC_SIZE = 532, 352
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18, np.dtype("uint8"): 129}


def write_mllm_v2(path, tensors, model_name="qwen3_5_lmhead"):
    items = list(tensors.items())
    n = len(items)
    with open(path, "wb") as g:
        g.write(b"\x00" * (_HDR_SIZE + n * _DESC_SIZE))
        descs = []
        for pid, (name, arr) in enumerate(items):
            arr = np.ascontiguousarray(arr)
            code = _CODE[arr.dtype]
            data = arr.tobytes()
            off = g.tell()
            g.write(data)
            nb = name.encode("utf-8")[:_PNAME_LEN].ljust(_PNAME_LEN, b"\0")
            shp = list(arr.shape) + [0] * (_SHAPE_LEN - len(arr.shape))
            descs.append(struct.pack(f"<IIQQQ{_SHAPE_LEN}i{_PNAME_LEN}s",
                                     pid, code, len(data), off, len(arr.shape), *shp, nb))
        for pid, d in enumerate(descs):
            g.seek(_HDR_SIZE + pid * _DESC_SIZE)
            g.write(d)
        g.seek(0)
        mn = model_name.encode("utf-8")[:_NAME_LEN].ljust(_NAME_LEN, b"\0")
        g.write(struct.pack(f"<II{_NAME_LEN}sIQ", _MAGIC, _VERSION, mn, n, _HDR_SIZE))
    print(f"[write] {path}: {n} tensors, {os.path.getsize(path)} bytes")


def _import_qlinear():
    here = os.path.dirname(os.path.abspath(__file__))
    core = os.path.normpath(os.path.join(here, "..", "core"))
    chain = {
        "pymllm": os.path.normpath(os.path.join(here, "..", "..", "..", "..")),
        "pymllm.backends": os.path.normpath(os.path.join(here, "..", "..", "..")),
        "pymllm.backends.qualcomm": os.path.normpath(os.path.join(here, "..", "..")),
        "pymllm.backends.qualcomm.transformers": os.path.normpath(os.path.join(here, "..")),
        "pymllm.backends.qualcomm.transformers.core": core,
    }
    for name, path in chain.items():
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    from pymllm.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ
    return QLinearLPBQ


QLinearLPBQ = _import_qlinear()


def _default_model_path():
    import glob
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "Qwen/Qwen3.5-2B"


def dequant_from_deploy(buf, in_features, out_features, block_size):
    n_blk = in_features // block_size
    w = buf["weight"].reshape(in_features, out_features).astype(np.int32)
    w = np.where(w >= 8, w - 16, w)
    s1 = buf["scale1"].reshape(out_features, n_blk).astype(np.float32)
    s2 = buf["scale2"].reshape(out_features).astype(np.float32)
    block_scale = s1 * s2[:, None]
    bs_full = np.repeat(block_scale, block_size, axis=1)
    w_oi = w.T.astype(np.float32)
    return torch.from_numpy(w_oi * bs_full).float()


@torch.no_grad()
def lpbq_pack_weight(weight, block_size):
    """LPBQ-pack a raw [out,in] weight tensor (lm_head = tied embed_tokens)."""
    out_features, in_features = weight.shape
    q = QLinearLPBQ(in_features, out_features, bias=False, block_size=block_size)
    q.weight.data.copy_(weight.float())
    q = q.to(torch.float32)
    q.freeze_weight()
    q.enable_fakequant()
    q.convert_to_conv2d_deploy_hwio()
    buf = {
        "weight": q.weight.cpu().numpy().astype(np.int8),
        "scale1": q.scale1.cpu().numpy().astype(np.uint8),
        "scale2": q.scale2.cpu().numpy().astype(np.float32),
    }
    w_dequant = dequant_from_deploy(buf, in_features, out_features, block_size)
    return buf, w_dequant


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3.5 head (final RMSNorm + tied lm_head): LPBQ + ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out", default="lmhead")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model_path} -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    hidden = cfg.hidden_size
    vocab = cfg.vocab_size
    rms_eps = float(cfg.rms_norm_eps)
    print(f"[cfg] hidden={hidden} vocab={vocab} eps={rms_eps} tie={cfg.tie_word_embeddings}")

    # ---- 1) hook lm.norm to grab a real final hidden state x (last token) ----
    cap = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) > 0 else kw.get("hidden_states")
        cap["x"] = hs.detach()
        return None
    h1 = lm.norm.register_forward_pre_hook(pre_hook, with_kwargs=True)
    ids = tok(args.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**ids)
    h1.remove()
    x = cap["x"][0, -1].float().cpu().reshape(1, hidden)          # [1,hidden] last-token pre-norm
    hf_logits = out.logits[0, -1].float().cpu()                  # [vocab] HF logits, last token
    hf_argmax = int(hf_logits.argmax())
    print(f"[hook] x range[{x.min():.3f},{x.max():.3f}]  HF argmax={hf_argmax} ({tok.decode([hf_argmax])!r})")

    # ---- 2) LPBQ pack lm_head (tied = embed_tokens) ----
    lm_head_w = lm.embed_tokens.weight.detach().float().cpu()    # [vocab, hidden] tied
    buf, W_dq = lpbq_pack_weight(lm_head_w, args.block_size)
    norm_w = (1.0 + lm.norm.weight.detach().float().cpu())       # [hidden] add_unit_offset

    # ---- 3) fp32 reference (dequant-int4 weights, matches graph) ----
    def head_ref(W):
        xf = x.reshape(hidden)
        xn = xf * torch.rsqrt((xf * xf).mean() + rms_eps) * norm_w
        return xn @ W.t()                                        # [vocab]
    logits_i4 = head_ref(W_dq)
    logits_f32 = head_ref(lm_head_w)
    d_hf = (logits_f32 - hf_logits).abs().max().item()
    d_i4 = (logits_i4 - logits_f32).abs().max().item()
    rng = logits_f32.abs().max().item()
    am_i4, am_f32 = int(logits_i4.argmax()), int(logits_f32.argmax())
    print(f"[ref] fp32 vs HF: max|err|={d_hf:.4f}  | int4 vs fp32: max|err|={d_i4:.4f}  |logit|max={rng:.3f}")
    print(f"[ref] argmax: int4={am_i4} fp32={am_f32} HF={hf_argmax}  "
          f"({'OK' if am_i4 == hf_argmax else 'INT4 SHIFTS ARGMAX'})")

    # ---- 4) bundles ----
    ref = {
        "x": x.numpy().astype(np.float32),
        "norm_w": norm_w.reshape(1, 1, hidden).numpy().astype(np.float32),
        "eps": np.array([rms_eps], dtype=np.float32),
        "exp_logits": logits_i4.numpy().astype(np.float32),
    }
    write_mllm_v2(args.out + "-ref.mllm", ref)

    W = {}
    W["model.lm_head.weight"] = buf["weight"]
    W["model.lm_head.scale1"] = buf["scale1"]
    W["model.lm_head.scale2"] = buf["scale2"]
    xn = (x.reshape(hidden) * torch.rsqrt((x.reshape(hidden) ** 2).mean() + rms_eps) * norm_w)
    qdq("model.lmhead_input_qdq", xn.abs().max().item(), W)
    qdq("model.lmhead_output_qdq", logits_i4.abs().max().item(), W)
    write_mllm_v2(args.out + "-lpbq.mllm", W)
    print(f"[done] -> {args.out}-lpbq.mllm + {args.out}-ref.mllm")


if __name__ == "__main__":
    main()
