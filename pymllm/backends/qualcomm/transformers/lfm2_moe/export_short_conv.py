#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export ONE LFM2 short-conv layer (decode step) as a W4A16-LPBQ bundle + golden ref.
#   in_proj (H->3H) and out_proj (H->H) are LPBQ; the depthwise causal conv (kernel
#   L_cache, NO activation) + double gating (Bx=B*x, y=C*conv) run fp16 in-graph.
#   cw (conv weight, [1,K,H]) and cs (conv state, [1,K-1,H]) are graph inputs.
#
# Validates on a REAL token with a REAL conv state: run a prompt, capture the conv
# layer's input (operator_norm output) for all tokens, pick a token t, build cs from
# the prior tokens' Bx frames so all K taps are exercised.
#
#   python3 export_short_conv.py --layer 0 --out /path/conv
#     -> <out>-lpbq.mllm (model.{in,out}_proj.* + 4 *_qdq.*)
#        <out>-ref.mllm  (x[1,H], cw[1,K,H], cs[1,K-1,H], exp_y[1,H])
import argparse
import glob
import os
import struct
import sys
import types

import numpy as np
import torch

_MAGIC, _VERSION = 0x519A, 2
_NAME_LEN, _PNAME_LEN, _SHAPE_LEN = 512, 256, 16
_HDR_SIZE, _DESC_SIZE = 532, 352
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18, np.dtype("uint8"): 129, np.dtype("float16"): 1}


def write_mllm_v2(path, tensors, model_name="lfm2_conv"):
    items = list(tensors.items())
    n = len(items)
    with open(path, "wb") as g:
        g.write(b"\x00" * (_HDR_SIZE + n * _DESC_SIZE))
        descs = []
        for pid, (name, arr) in enumerate(items):
            arr = np.ascontiguousarray(arr)
            data = arr.tobytes()
            off = g.tell()
            g.write(data)
            nb = name.encode("utf-8")[:_PNAME_LEN].ljust(_PNAME_LEN, b"\0")
            shp = list(arr.shape) + [0] * (_SHAPE_LEN - len(arr.shape))
            descs.append(struct.pack(f"<IIQQQ{_SHAPE_LEN}i{_PNAME_LEN}s", pid, _CODE[arr.dtype], len(data), off, len(arr.shape), *shp, nb))
        for pid, d in enumerate(descs):
            g.seek(_HDR_SIZE + pid * _DESC_SIZE)
            g.write(d)
        g.seek(0)
        mn = model_name.encode("utf-8")[:_NAME_LEN].ljust(_NAME_LEN, b"\0")
        g.write(struct.pack(f"<II{_NAME_LEN}sIQ", _MAGIC, _VERSION, mn, n, _HDR_SIZE))
    print(f"[write] {os.path.basename(path)}: {n} tensors, {os.path.getsize(path)} bytes")


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


def dequant_from_deploy(buf, in_features, out_features, block_size):
    n_blk = in_features // block_size
    w = buf["weight"].reshape(in_features, out_features).astype(np.int32)
    w = np.where(w >= 8, w - 16, w)
    s1 = buf["scale1"].reshape(out_features, n_blk).astype(np.float32)
    s2 = buf["scale2"].reshape(out_features).astype(np.float32)
    bs = np.repeat(s1 * s2[:, None], block_size, axis=1)
    return torch.from_numpy(w.T.astype(np.float32) * bs).float()  # [out, in]


@torch.no_grad()
def lpbq_pack(weight, block_size):
    out_f, in_f = weight.shape
    q = QLinearLPBQ(in_f, out_f, bias=False, block_size=block_size)
    q.weight.data.copy_(weight.float())
    q = q.to(torch.float32)
    q.freeze_weight(); q.enable_fakequant(); q.convert_to_conv2d_deploy_hwio()
    buf = {"weight": q.weight.cpu().numpy().astype(np.int8),
           "scale1": q.scale1.cpu().numpy().astype(np.uint8),
           "scale2": q.scale2.cpu().numpy().astype(np.float32)}
    return buf, dequant_from_deploy(buf, in_f, out_f, block_size)


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def _default_model_path():
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "LiquidAI/LFM2.5-8B-A1B"


def main():
    ap = argparse.ArgumentParser(description="Export one LFM2 short-conv decode step (W4A16 LPBQ) + ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=0, help="a conv layer index")
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out", default="/mnt/raid0_ssd/wentao/mllm/build-qnn-aot/bin/lfm2_conv")
    args = ap.parse_args()
    bs = args.block_size

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32, attn_implementation="eager").eval()
    lm, cfg = model.model, model.config
    H = cfg.hidden_size
    K = cfg.conv_L_cache
    layer = lm.layers[args.layer]
    assert hasattr(layer, "conv"), f"layer {args.layer} is not a conv layer"
    conv = layer.conv

    captured = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) else kw.get("hidden_states")
        captured["x"] = hs.detach(); return None
    h = conv.register_forward_pre_hook(pre_hook, with_kwargs=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids, use_cache=False)
    h.remove()
    X = captured["x"][0].float()                      # [T, H] conv inputs (operator_norm output)
    T = X.shape[0]
    t = T - 1

    Wi = conv.in_proj.weight.data.float()             # [3H, H]
    Wo = conv.out_proj.weight.data.float()            # [H, H]
    cw = conv.conv.weight.data.float().reshape(H, K)  # [H, K]  (squeezed from [H,1,K])

    in_buf, Wi_dq = lpbq_pack(Wi, bs)
    out_buf, Wo_dq = lpbq_pack(Wo, bs)

    # reference forward with DEQUANT weights (device-equivalent), fp32 math
    proj = X @ Wi_dq.t()                              # [T, 3H]
    Bg, Cg, xin = proj[:, :H], proj[:, H:2 * H], proj[:, 2 * H:]
    Bx = Bg * xin                                     # [T, H]
    # window for token t = [Bx[t-K+1..t]] (zeros for t<frame); conv = sum_j cw[:,j]*win[j]
    win = torch.zeros(K, H)
    for j in range(K):
        idx = t - (K - 1) + j
        if idx >= 0:
            win[j] = Bx[idx]
    conv_out = (cw.t() * win).sum(0)                  # [H]  cw.t():[K,H]
    y = Cg[t] * conv_out                              # [H]
    out = y @ Wo_dq.t()                               # [H] = exp_y

    cs = win[:K - 1]                                  # [K-1, H]  (state going into token t)

    # bundle
    W = {}
    for nm, b in [("in_proj", in_buf), ("out_proj", out_buf)]:
        W[f"model.{nm}.weight"] = b["weight"]; W[f"model.{nm}.scale1"] = b["scale1"]; W[f"model.{nm}.scale2"] = b["scale2"]
    qdq("model.in_proj_input_qdq", float(X[t].abs().max()), W)
    qdq("model.in_proj_output_qdq", float(proj[t].abs().max()), W)
    qdq("model.out_proj_input_qdq", float(y.abs().max()), W)
    qdq("model.out_proj_output_qdq", float(out.abs().max()), W)
    write_mllm_v2(f"{args.out}-lpbq.mllm", W)

    # ref: x, cw[1,K,H] (cw[0,j,h]=conv_weight[h,j]), cs[1,K-1,H], exp_y
    cw_in = cw.t().contiguous().numpy().reshape(1, K, H)        # [1,K,H]
    write_mllm_v2(f"{args.out}-ref.mllm", {
        "x": X[t:t + 1].contiguous().numpy().astype(np.float32).reshape(1, H),
        "cw": cw_in.astype(np.float32),
        "cs": cs.contiguous().numpy().astype(np.float32).reshape(1, K - 1, H),
        "exp_y": out.contiguous().numpy().astype(np.float32).reshape(1, H),
    })
    print(f"[done] layer={args.layer} H={H} K={K} token={t}/{T} |x|max={float(X[t].abs().max()):.4f} "
          f"|y|max={float(out.abs().max()):.4f}")


if __name__ == "__main__":
    main()
