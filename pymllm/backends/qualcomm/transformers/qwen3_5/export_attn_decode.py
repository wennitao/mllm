#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export ONE Qwen3.5 full-attention decode step for the NPU graph
# (compile_attn_decode.cpp). Companion to export_deltanet_decode.py.
#
#   1. Load HF Qwen3.5-2B (fp32), hook a full-attention layer L's self_attn to grab
#      a real input x (all prompt positions) + the rotary cos/sin + the HF output.
#   2. LPBQ-pack q/k/v/o_proj (deploy int4 weight + uint4 scale1 + fp32 scale2), slice
#      the per-head projections, calibrate uint16 activation QDQ scales.
#   3. Build a fp32 reference of the DECODE step (last token attends over the whole
#      prompt KV cache), using the int4-DEQUANT weights (matches on-device) — plus a
#      full-precision-weight reference to size the int4 cost. q/k-norm uses 1+w
#      (add_unit_offset) baked into the exported norm weight.
#
#   <out>-lpbq.mllm : model.{q,k,v}_proj.<h>.{weight,scale1,scale2}, model.o_proj.*,
#                     all model.*_qdq.fake_quant.{scale,zero_point}
#   <out>-ref.mllm  : x, sin, cos, q_norm_w, k_norm_w, eps, past_k, past_v, mask,
#                     exp_y, exp_k_new, exp_v_new
#
#   python3 export_attn_decode.py --layer 3 --ctx 256 --out attn-l3
#
import argparse
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- mllm V2 binary writer (fp32=0, int8=16, int32=18, uint8=129) ----
_MAGIC, _VERSION = 0x519A, 2
_NAME_LEN, _PNAME_LEN, _SHAPE_LEN = 512, 256, 16
_HDR_SIZE, _DESC_SIZE = 532, 352
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18, np.dtype("uint8"): 129}


def write_mllm_v2(path, tensors, model_name="qwen3_5_attn"):
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
def lpbq_pack(lin, block_size):
    q = QLinearLPBQ(lin.in_features, lin.out_features, bias=False, block_size=block_size)
    q.weight.data.copy_(lin.weight.data.float())
    q = q.to(torch.float32)
    q.freeze_weight()
    q.enable_fakequant()
    q.convert_to_conv2d_deploy_hwio()
    buf = {
        "weight": q.weight.cpu().numpy().astype(np.int8),
        "scale1": q.scale1.cpu().numpy().astype(np.uint8),
        "scale2": q.scale2.cpu().numpy().astype(np.float32),
    }
    w_dequant = dequant_from_deploy(buf, lin.in_features, lin.out_features, block_size)
    return buf, w_dequant


def slice_head(buf, c0, c1, n_blk):
    return {
        "weight": np.ascontiguousarray(buf["weight"][:, :, :, c0:c1]),
        "scale1": np.ascontiguousarray(buf["scale1"][c0 * n_blk:c1 * n_blk]),
        "scale2": np.ascontiguousarray(buf["scale2"][c0:c1]),
    }


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def rmsnorm(x, w, eps):
    # x [..., D], w [D] (already 1+weight); fp32 mean-of-squares.
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def partial_rope(x, cos, sin, rot):
    # x [T,H,D]; cos/sin [T,rot] (rotate-half convention, halves duplicated).
    # rotate only first rot dims; pass the rest through.
    T, H, D = x.shape
    xr = x[..., :rot]                         # [T,H,rot]
    xp = x[..., rot:]
    half = rot // 2
    x1 = xr[..., :half]
    x2 = xr[..., half:]
    rh = torch.cat([-x2, x1], dim=-1)         # [T,H,rot]
    c = cos[:, None, :]                       # [T,1,rot]
    s = sin[:, None, :]
    xr_roped = xr * c + rh * s
    return torch.cat([xr_roped, xp], dim=-1)  # [T,H,D]


def main():
    ap = argparse.ArgumentParser(description="Export one Qwen3.5 full-attention decode step: LPBQ weights + ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--layer", type=int, default=3, help="full-attention layer index (interval 4: 3,7,11,...)")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256, help="KV cache context length (graph fixed size)")
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out", default="attn-l3")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model_path} -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layer = lm.layers[args.layer]
    attn = layer.self_attn
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    H = cfg.num_attention_heads
    KV = cfg.num_key_value_heads
    D = cfg.head_dim
    hidden = cfg.hidden_size
    rms_eps = float(cfg.rms_norm_eps)
    rot = int(round(D * cfg.partial_rotary_factor))
    rot -= rot & 1
    grp = H // KV
    scale = 1.0 / (D ** 0.5)
    print(f"[cfg] layer={args.layer} H={H} KV={KV} D={D} hidden={hidden} rot={rot} eps={rms_eps} scale={scale:.5f}")
    assert not lm.config.layer_types[args.layer].startswith("linear"), \
        f"layer {args.layer} is not full-attention ({lm.config.layer_types[args.layer]})"

    # ---- 1) hook self_attn: capture x (all positions), cos/sin, HF output ----
    cap = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) > 0 else kw.get("hidden_states")
        cap["x"] = hs.detach()
        pe = kw.get("position_embeddings")
        if pe is not None:
            cap["cos"], cap["sin"] = pe[0].detach(), pe[1].detach()
        return None
    def post_hook(mod, inp, out):
        cap["out"] = (out[0] if isinstance(out, tuple) else out).detach()
        return None
    h1 = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.register_forward_hook(post_hook)
    ids = tok(args.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        model(**ids)
    h1.remove(); h2.remove()

    X_all = cap["x"][0].float().cpu()                 # [T, hidden]
    hf_out_last = cap["out"][0, -1].float().cpu()     # [hidden] HF attn output, last token
    cos_all = cap["cos"][0].float().cpu()             # [T, ?]
    sin_all = cap["sin"][0].float().cpu()
    T = X_all.shape[0]
    last = T - 1
    # rotary cos/sin: HF may emit width D (full) or rot. Take the first rot columns.
    cos_all = cos_all[:, :rot]
    sin_all = sin_all[:, :rot]
    print(f"[hook] T={T} x{tuple(X_all.shape)} cos{tuple(cos_all.shape)} out_range[{hf_out_last.min():.3f},{hf_out_last.max():.3f}]")
    assert T <= args.ctx, f"prompt T={T} exceeds ctx={args.ctx}; raise --ctx"

    # ---- 2) LPBQ pack q/k/v/o (deploy + dequant) ----
    q_buf, Wq_dq = lpbq_pack(attn.q_proj, args.block_size)   # hidden -> H*2D
    k_buf, Wk_dq = lpbq_pack(attn.k_proj, args.block_size)   # hidden -> KV*D
    v_buf, Wv_dq = lpbq_pack(attn.v_proj, args.block_size)   # hidden -> KV*D
    o_buf, Wo_dq = lpbq_pack(attn.o_proj, args.block_size)   # H*D -> hidden
    n_blk = hidden // args.block_size                        # 2048/16 = 128 (q/k/v contract over hidden)
    n_blk_o = (H * D) // args.block_size                     # o_proj contracts over H*D

    # norm weights with add_unit_offset baked in (1+w)
    q_norm_w = (1.0 + attn.q_norm.weight.detach().float().cpu())   # [D]
    k_norm_w = (1.0 + attn.k_norm.weight.detach().float().cpu())

    # ---- 3) fp32 decode reference using DEQUANT-int4 weights (matches on-device) ----
    @torch.no_grad()
    def decode_ref(Wq, Wk, Wv, Wo):
        qg = (X_all @ Wq.t()).reshape(T, H, 2 * D)
        q = qg[..., :D]                                       # [T,H,D]
        gate = qg[..., D:]                                    # [T,H,D]
        k = (X_all @ Wk.t()).reshape(T, KV, D)
        v = (X_all @ Wv.t()).reshape(T, KV, D)
        q = rmsnorm(q, q_norm_w, rms_eps)
        k = rmsnorm(k, k_norm_w, rms_eps)
        q = partial_rope(q, cos_all, sin_all, rot)
        k = partial_rope(k, cos_all, sin_all, rot)
        # decode: last query attends over all T positions
        ql = q[last]                                          # [H,D]
        outs = torch.zeros(H, D)
        for hh in range(H):
            kv = hh // grp
            sc = (ql[hh] @ k[:, kv].t()) * scale              # [T]
            p = torch.softmax(sc, dim=-1)
            outs[hh] = p @ v[:, kv]                           # [D]
        out = outs.reshape(H * D) * torch.sigmoid(gate[last].reshape(H * D))
        y = out @ Wo.t()                                      # [hidden]
        return dict(y=y, k=k, v=v, q=q, gate=gate)

    rq = decode_ref(Wq_dq, Wk_dq, Wv_dq, Wo_dq)               # int4-weight (graph match)
    rf = decode_ref(attn.q_proj.weight.detach().float().cpu(),
                    attn.k_proj.weight.detach().float().cpu(),
                    attn.v_proj.weight.detach().float().cpu(),
                    attn.o_proj.weight.detach().float().cpu())  # fp32-weight
    y_int4 = rq["y"]; y_fp32 = rf["y"]
    d_hf = (y_fp32 - hf_out_last).abs().max().item()
    d_i4 = (y_int4 - y_fp32).abs().max().item()
    rng = y_fp32.abs().max().item()
    print(f"[ref] fp32-ref vs HF: max|err|={d_hf:.6f}  | int4 vs fp32: max|err|={d_i4:.6f}  |y|max={rng:.4f}")

    # ---- 4) build the cache + mask for the decode-step graph ----
    # graph attends over ctx = [past (ctx-1) ++ new]; past slots 0..last-1 are real, rest empty.
    P = args.ctx - 1
    k_ref = rq["k"]; v_ref = rq["v"]                          # [T,KV,D]
    past_k = np.zeros((KV, D, P), dtype=np.float32)           # [KV,D,P] (transposed K layout)
    past_v = np.zeros((KV, P, D), dtype=np.float32)
    for p in range(last):                                     # real past tokens 0..last-1
        for kv in range(KV):
            past_k[kv, :, p] = k_ref[p, kv].numpy()
            past_v[kv, p, :] = v_ref[p, kv].numpy()
    exp_k_new = np.zeros((KV, D, 1), dtype=np.float32)        # new token = position `last`
    exp_v_new = np.zeros((KV, 1, D), dtype=np.float32)
    for kv in range(KV):
        exp_k_new[kv, :, 0] = k_ref[last, kv].numpy()
        exp_v_new[kv, 0, :] = v_ref[last, kv].numpy()
    mask = np.full((args.ctx,), -50000.0, dtype=np.float32)
    mask[:last] = 0.0                                         # real past tokens
    mask[args.ctx - 1] = 0.0                                  # the new token slot

    # ---- 5) reference bundle ----
    ref = {
        "x": X_all[last].reshape(1, hidden).numpy().astype(np.float32),
        "sin": sin_all[last].reshape(1, 1, rot).numpy().astype(np.float32),
        "cos": cos_all[last].reshape(1, 1, rot).numpy().astype(np.float32),
        "q_norm_w": q_norm_w.reshape(1, 1, D).numpy().astype(np.float32),
        "k_norm_w": k_norm_w.reshape(1, 1, D).numpy().astype(np.float32),
        "eps": np.array([rms_eps], dtype=np.float32),
        "past_k": past_k, "past_v": past_v,
        "mask": mask,
        "exp_y": y_int4.numpy().astype(np.float32),
        "exp_k_new": exp_k_new, "exp_v_new": exp_v_new,
    }
    write_mllm_v2(args.out + "-ref.mllm", ref)

    # ---- 6) LPBQ weight bundle (per-head slices + QDQ scales) ----
    W = {}
    qdq("model.qkv_input_qdq", X_all[last].abs().max().item(), W)
    # q_proj per head: head h owns out-cols [h*2D : (h+1)*2D]
    qg_all = X_all @ Wq_dq.t()                                # [T, H*2D] for QDQ calibration
    for h in range(H):
        sl = slice_head(q_buf, h * 2 * D, (h + 1) * 2 * D, n_blk)
        W[f"model.q_proj.{h}.weight"] = sl["weight"]
        W[f"model.q_proj.{h}.scale1"] = sl["scale1"]
        W[f"model.q_proj.{h}.scale2"] = sl["scale2"]
        qdq(f"model.q_out_qdq_h{h}", qg_all[:, h * 2 * D:(h + 1) * 2 * D].abs().max().item(), W)
    kk_all = X_all @ Wk_dq.t()
    vv_all = X_all @ Wv_dq.t()
    for kv in range(KV):
        sk = slice_head(k_buf, kv * D, (kv + 1) * D, n_blk)
        sv = slice_head(v_buf, kv * D, (kv + 1) * D, n_blk)
        W[f"model.k_proj.{kv}.weight"] = sk["weight"]
        W[f"model.k_proj.{kv}.scale1"] = sk["scale1"]
        W[f"model.k_proj.{kv}.scale2"] = sk["scale2"]
        W[f"model.v_proj.{kv}.weight"] = sv["weight"]
        W[f"model.v_proj.{kv}.scale1"] = sv["scale1"]
        W[f"model.v_proj.{kv}.scale2"] = sv["scale2"]
        qdq(f"model.k_out_qdq_h{kv}", kk_all[:, kv * D:(kv + 1) * D].abs().max().item(), W)
        qdq(f"model.v_out_qdq_h{kv}", vv_all[:, kv * D:(kv + 1) * D].abs().max().item(), W)
    # o_proj (full)
    W["model.o_proj.weight"] = o_buf["weight"]
    W["model.o_proj.scale1"] = o_buf["scale1"]
    W["model.o_proj.scale2"] = o_buf["scale2"]
    # o_proj input = gated attention output (pre o_proj); calibrate from int4 gated magnitude
    outs_full = []
    for hh in range(H):
        kv = hh // grp
        ql = rq["q"][last, hh]
        sc = (ql @ rq["k"][:, kv].t()) * scale
        p = torch.softmax(sc, -1)
        outs_full.append(p @ rq["v"][:, kv])
    gated_out = torch.stack(outs_full).reshape(H * D) * torch.sigmoid(rq["gate"][last].reshape(H * D))
    qdq("model.o_proj_input_qdq", gated_out.abs().max().item(), W)
    qdq("model.o_proj_output_qdq", y_int4.abs().max().item(), W)
    write_mllm_v2(args.out + "-lpbq.mllm", W)

    print(f"[done] layer={args.layer} -> {args.out}-lpbq.mllm + {args.out}-ref.mllm  (T={T}, ctx={args.ctx})")


if __name__ == "__main__":
    main()
