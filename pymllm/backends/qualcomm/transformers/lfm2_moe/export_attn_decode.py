#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export ONE LFM2 full-attention decode step for the NPU graph (compile_lfm2_attn_decode).
# Adapts the Qwen3.5 attn export to LFM2: NO output gate (q_proj -> D), FULL RoPE (rot=D),
# PLAIN q/k-norm (weight = w, no +1), attrs q_layernorm/k_layernorm/out_proj.
#
#   <out>-lpbq.mllm : model.{q,k,v}_proj.<h>.{weight,scale1,scale2}, model.o_proj.*, all *_qdq.*
#   <out>-ref.mllm  : x, sin, cos, q_norm_w, k_norm_w, eps, past_k, past_v, mask, exp_y, exp_k_new, exp_v_new
#
#   python3 export_attn_decode.py --layer 2 --ctx 256 --out attn-l2
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
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18, np.dtype("uint8"): 129}


def write_mllm_v2(path, tensors, model_name="lfm2_attn"):
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
    return torch.from_numpy(w.T.astype(np.float32) * bs).float()


@torch.no_grad()
def lpbq_pack(lin, block_size):
    q = QLinearLPBQ(lin.in_features, lin.out_features, bias=False, block_size=block_size)
    q.weight.data.copy_(lin.weight.data.float())
    q = q.to(torch.float32)
    q.freeze_weight(); q.enable_fakequant(); q.convert_to_conv2d_deploy_hwio()
    buf = {"weight": q.weight.cpu().numpy().astype(np.int8),
           "scale1": q.scale1.cpu().numpy().astype(np.uint8),
           "scale2": q.scale2.cpu().numpy().astype(np.float32)}
    return buf, dequant_from_deploy(buf, lin.in_features, lin.out_features, block_size)


def slice_head(buf, c0, c1, n_blk):
    return {"weight": np.ascontiguousarray(buf["weight"][:, :, :, c0:c1]),
            "scale1": np.ascontiguousarray(buf["scale1"][c0 * n_blk:c1 * n_blk]),
            "scale2": np.ascontiguousarray(buf["scale2"][c0:c1])}


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def rmsnorm(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def full_rope(x, cos, sin):
    # x [T,H,D]; cos/sin [T,D] (rotate-half, halves duplicated). FULL rotary (rot==D).
    D = x.shape[-1]; half = D // 2
    x1 = x[..., :half]; x2 = x[..., half:]
    rh = torch.cat([-x2, x1], dim=-1)
    return x * cos[:, None, :] + rh * sin[:, None, :]


def _default_model_path():
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "LiquidAI/LFM2.5-8B-A1B"


def main():
    ap = argparse.ArgumentParser(description="Export one LFM2 full-attention decode step: LPBQ + ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--layer", type=int, default=2, help="full_attention layer index (2,6,10,14,18,21)")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out", default="/mnt/raid0_ssd/wentao/mllm/build-qnn-aot/bin/lfm2_attn")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32, attn_implementation="eager").eval()
    lm, cfg = model.model, model.config
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    hidden = cfg.hidden_size
    D = hidden // H
    rms_eps = float(cfg.norm_eps)
    rot = D  # FULL rope
    grp = H // KV
    scale = 1.0 / (D ** 0.5)
    layer = lm.layers[args.layer]
    attn = layer.self_attn
    assert cfg.layer_types[args.layer] == "full_attention", f"layer {args.layer} is {cfg.layer_types[args.layer]}"
    print(f"[cfg] layer={args.layer} H={H} KV={KV} D={D} hidden={hidden} rot={rot} eps={rms_eps}")

    cap = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) else kw.get("hidden_states")
        cap["x"] = hs.detach()
        pe = kw.get("position_embeddings")
        if pe is None and len(inp) > 1:
            pe = inp[1]
        if pe is not None:
            cap["cos"], cap["sin"] = pe[0].detach(), pe[1].detach()
        return None
    def post_hook(mod, inp, out):
        cap["out"] = (out[0] if isinstance(out, tuple) else out).detach()
    h1 = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.register_forward_hook(post_hook)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids, use_cache=False)
    h1.remove(); h2.remove()

    X_all = cap["x"][0].float()                  # [T, hidden]
    hf_out_last = cap["out"][0, -1].float()      # [hidden]
    cos_all = cap["cos"][0].float()[:, :rot]     # [T, rot]
    sin_all = cap["sin"][0].float()[:, :rot]
    T = X_all.shape[0]; last = T - 1
    assert T <= args.ctx
    print(f"[hook] T={T} x{tuple(X_all.shape)} cos{tuple(cos_all.shape)}", flush=True)

    q_buf, Wq_dq = lpbq_pack(attn.q_proj, args.block_size)   # hidden -> H*D
    k_buf, Wk_dq = lpbq_pack(attn.k_proj, args.block_size)   # hidden -> KV*D
    v_buf, Wv_dq = lpbq_pack(attn.v_proj, args.block_size)
    o_buf, Wo_dq = lpbq_pack(attn.out_proj, args.block_size) # H*D -> hidden
    n_blk = hidden // args.block_size

    q_norm_w = attn.q_layernorm.weight.detach().float()      # PLAIN (no +1)
    k_norm_w = attn.k_layernorm.weight.detach().float()

    @torch.no_grad()
    def decode_ref(Wq, Wk, Wv, Wo):
        q = (X_all @ Wq.t()).reshape(T, H, D)        # NO gate
        k = (X_all @ Wk.t()).reshape(T, KV, D)
        v = (X_all @ Wv.t()).reshape(T, KV, D)
        q = full_rope(rmsnorm(q, q_norm_w, rms_eps), cos_all, sin_all)
        k = full_rope(rmsnorm(k, k_norm_w, rms_eps), cos_all, sin_all)
        ql = q[last]
        outs = torch.zeros(H, D)
        for hh in range(H):
            kv = hh // grp
            sc = (ql[hh] @ k[:, kv].t()) * scale
            outs[hh] = torch.softmax(sc, -1) @ v[:, kv]
        out = outs.reshape(H * D)                     # NO output gate
        return dict(y=out @ Wo.t(), k=k, v=v, q=q, attn_out=out)

    rq = decode_ref(Wq_dq, Wk_dq, Wv_dq, Wo_dq)
    rf = decode_ref(attn.q_proj.weight.detach().float(), attn.k_proj.weight.detach().float(),
                    attn.v_proj.weight.detach().float(), attn.out_proj.weight.detach().float())
    y_int4, y_fp32 = rq["y"], rf["y"]
    print(f"[ref] fp32-ref vs HF max|err|={float((y_fp32-hf_out_last).abs().max()):.6f}  "
          f"int4 vs fp32 max|err|={float((y_int4-y_fp32).abs().max()):.6f}  |y|max={float(y_fp32.abs().max()):.4f}", flush=True)

    # cache + mask
    P = args.ctx - 1
    k_ref, v_ref = rq["k"], rq["v"]
    past_k = np.zeros((KV, D, P), dtype=np.float32)
    past_v = np.zeros((KV, P, D), dtype=np.float32)
    for p in range(last):
        for kv in range(KV):
            past_k[kv, :, p] = k_ref[p, kv].numpy()
            past_v[kv, p, :] = v_ref[p, kv].numpy()
    exp_k_new = np.zeros((KV, D, 1), dtype=np.float32)
    exp_v_new = np.zeros((KV, 1, D), dtype=np.float32)
    for kv in range(KV):
        exp_k_new[kv, :, 0] = k_ref[last, kv].numpy()
        exp_v_new[kv, 0, :] = v_ref[last, kv].numpy()
    mask = np.full((args.ctx,), -50000.0, dtype=np.float32)
    mask[:last] = 0.0
    mask[args.ctx - 1] = 0.0

    write_mllm_v2(args.out + "-ref.mllm", {
        "x": X_all[last].reshape(1, hidden).numpy().astype(np.float32),
        "sin": sin_all[last].reshape(1, 1, rot).numpy().astype(np.float32),
        "cos": cos_all[last].reshape(1, 1, rot).numpy().astype(np.float32),
        "q_norm_w": q_norm_w.reshape(1, 1, D).numpy().astype(np.float32),
        "k_norm_w": k_norm_w.reshape(1, 1, D).numpy().astype(np.float32),
        "eps": np.array([rms_eps], dtype=np.float32),
        "past_k": past_k, "past_v": past_v, "mask": mask,
        "exp_y": y_int4.numpy().astype(np.float32),
        "exp_k_new": exp_k_new, "exp_v_new": exp_v_new,
    })

    W = {}
    qdq("model.qkv_input_qdq", float(X_all[last].abs().max()), W)
    qg_all = X_all @ Wq_dq.t()
    for h in range(H):
        sl = slice_head(q_buf, h * D, (h + 1) * D, n_blk)
        W[f"model.q_proj.{h}.weight"] = sl["weight"]; W[f"model.q_proj.{h}.scale1"] = sl["scale1"]; W[f"model.q_proj.{h}.scale2"] = sl["scale2"]
        qdq(f"model.q_out_qdq_h{h}", float(qg_all[:, h * D:(h + 1) * D].abs().max()), W)
    kk_all = X_all @ Wk_dq.t(); vv_all = X_all @ Wv_dq.t()
    for kv in range(KV):
        sk = slice_head(k_buf, kv * D, (kv + 1) * D, n_blk)
        sv = slice_head(v_buf, kv * D, (kv + 1) * D, n_blk)
        for nm, s in [("k_proj", sk), ("v_proj", sv)]:
            W[f"model.{nm}.{kv}.weight"] = s["weight"]; W[f"model.{nm}.{kv}.scale1"] = s["scale1"]; W[f"model.{nm}.{kv}.scale2"] = s["scale2"]
        qdq(f"model.k_out_qdq_h{kv}", float(kk_all[:, kv * D:(kv + 1) * D].abs().max()), W)
        qdq(f"model.v_out_qdq_h{kv}", float(vv_all[:, kv * D:(kv + 1) * D].abs().max()), W)
    W["model.o_proj.weight"] = o_buf["weight"]; W["model.o_proj.scale1"] = o_buf["scale1"]; W["model.o_proj.scale2"] = o_buf["scale2"]
    qdq("model.o_proj_input_qdq", float(rq["attn_out"].abs().max()), W)
    qdq("model.o_proj_output_qdq", float(y_int4.abs().max()), W)
    write_mllm_v2(args.out + "-lpbq.mllm", W)
    print(f"[done] layer={args.layer} -> {args.out}-lpbq.mllm + -ref.mllm (T={T}, ctx={args.ctx})")


if __name__ == "__main__":
    main()
