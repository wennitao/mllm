#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export ONE Qwen3.5 full-attention layer's BLOCK PREFILL (Sq=B causal) for the NPU graph
# (compile_attn_prefill.cpp). Companion to export_attn_decode.py (Sq=1) and
# export_deltanet_prefill.py. Processes a block of B tokens with dense causal attention.
#
# I/O contract (first block; no past KV):
#   inputs : x[B,hidden], sin[1,B,rot], cos[1,B,rot], q_norm_w[1,1,D], k_norm_w[1,1,D],
#            eps[1,1,1], cmask[1,1,B,B] (additive: 0 if j<=i else -50000)
#   outputs: y[B,hidden] (per-token attn output), k_all[1,KV,D,B], v_all[1,KV,B,D]
#            (post-norm/rope K, raw V — the block's KV to seed the decode ring buffer)
#
#   python3 export_attn_prefill.py --layer 3 --seq 128 --out attn-prefill-l3
#
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_attn_decode import (  # noqa: E402
    write_mllm_v2, lpbq_pack, slice_head, qdq, _default_model_path, rmsnorm, partial_rope,
)


def main():
    ap = argparse.ArgumentParser(description="Export one Qwen3.5 full-attention BLOCK PREFILL: LPBQ weights + ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--layer", type=int, default=3, help="full-attention layer index (interval 4: 3,7,11,...)")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128, help="block length B (prompt padded/truncated to this)")
    ap.add_argument("--prompt", default="Artificial intelligence has transformed the way we live and work over the "
                                        "past decade. From natural language processing to computer vision, machine "
                                        "learning models now power search engines and autonomous vehicles.")
    ap.add_argument("--out", default="attn-prefill-l3")
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
    H, KV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    hidden, rms_eps = cfg.hidden_size, float(cfg.rms_norm_eps)
    rot = int(round(D * cfg.partial_rotary_factor)); rot -= rot & 1
    grp = H // KV
    scale = 1.0 / (D ** 0.5)
    B = args.seq
    print(f"[cfg] layer={args.layer} H={H} KV={KV} D={D} hidden={hidden} rot={rot} B={B} scale={scale:.5f}")
    assert not lm.config.layer_types[args.layer].startswith("linear"), \
        f"layer {args.layer} is not full-attention ({lm.config.layer_types[args.layer]})"

    # ---- 1) hook self_attn over the whole prompt ----
    cap = {}
    def pre_hook(mod, inp, kw):
        cap["x"] = (inp[0] if len(inp) > 0 else kw.get("hidden_states")).detach()
        pe = kw.get("position_embeddings")
        if pe is not None:
            cap["cos"], cap["sin"] = pe[0].detach(), pe[1].detach()
    def post_hook(mod, inp, out):
        cap["out"] = (out[0] if isinstance(out, tuple) else out).detach()
    h1 = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.register_forward_hook(post_hook)
    with torch.no_grad():
        model(**tok(args.prompt, return_tensors="pt").to(dev))
    h1.remove(); h2.remove()

    Xf = cap["x"][0].float().cpu()                    # [T,hidden]
    HFY = cap["out"][0].float().cpu()                 # [T,hidden]
    cosf = cap["cos"][0].float().cpu()[:, :rot]        # [T,rot]
    sinf = cap["sin"][0].float().cpu()[:, :rot]
    T = Xf.shape[0]
    nreal = min(T, B)
    # pad/truncate to B
    def padB(t, v=0.0):
        if t.shape[0] >= B:
            return t[:B]
        # EDGE-REPLICATE pad (repeat the last real row), NOT zeros: a zero-variance row makes
        # the fp16 rmsnorm hit rsqrt(0)=inf -> nan, which poisons real tokens through the
        # masked score matmul. Padded rows are causally masked out of every real query anyway.
        pad = t[-1:].repeat((B - t.shape[0],) + (1,) * (t.dim() - 1))
        return torch.cat([t, pad], 0)
    X_all = padB(Xf); cos_all = padB(cosf); sin_all = padB(sinf)
    print(f"[hook] T={T} -> B={B}; x range [{X_all.min():.3f},{X_all.max():.3f}]")

    # ---- 2) LPBQ pack q/k/v/o ----
    q_buf, Wq_dq = lpbq_pack(attn.q_proj, args.block_size)
    k_buf, Wk_dq = lpbq_pack(attn.k_proj, args.block_size)
    v_buf, Wv_dq = lpbq_pack(attn.v_proj, args.block_size)
    o_buf, Wo_dq = lpbq_pack(attn.o_proj, args.block_size)
    n_blk = hidden // args.block_size
    q_norm_w = (1.0 + attn.q_norm.weight.detach().float().cpu())   # [D] (1+w)
    k_norm_w = (1.0 + attn.k_norm.weight.detach().float().cpu())

    # ---- 3) fp32 PREFILL reference (all B tokens, causal) with int4-dequant weights ----
    @torch.no_grad()
    def prefill_ref(Wq, Wk, Wv, Wo):
        qg = (X_all @ Wq.t()).reshape(B, H, 2 * D)
        q = qg[..., :D]; gate = qg[..., D:]               # [B,H,D]
        k = (X_all @ Wk.t()).reshape(B, KV, D)
        v = (X_all @ Wv.t()).reshape(B, KV, D)
        q = rmsnorm(q, q_norm_w, rms_eps)
        k = rmsnorm(k, k_norm_w, rms_eps)
        q = partial_rope(q, cos_all, sin_all, rot)        # [B,H,D]
        k = partial_rope(k, cos_all, sin_all, rot)        # [B,KV,D]
        cmask = torch.full((B, B), float("-inf"))
        cmask = torch.triu(cmask, diagonal=1)             # 0 on/below diag, -inf above
        outs = torch.zeros(B, H, D)
        for hh in range(H):
            kv = hh // grp
            sc = (q[:, hh] @ k[:, kv].t()) * scale + cmask  # [B,B]
            p = torch.softmax(sc, dim=-1)
            outs[:, hh] = p @ v[:, kv]                      # [B,D]
        gated = (outs.reshape(B, H * D)) * torch.sigmoid(gate.reshape(B, H * D))
        y = gated @ Wo.t()                                  # [B,hidden]
        return dict(y=y, k=k, v=v, q=q, gate=gate, gated=gated)

    rq = prefill_ref(Wq_dq, Wk_dq, Wv_dq, Wo_dq)
    rf = prefill_ref(attn.q_proj.weight.detach().float().cpu(), attn.k_proj.weight.detach().float().cpu(),
                     attn.v_proj.weight.detach().float().cpu(), attn.o_proj.weight.detach().float().cpu())
    d_hf = (rf["y"][:nreal] - HFY[:nreal]).abs().max().item()
    d_i4 = (rq["y"][:nreal] - rf["y"][:nreal]).abs().max().item()
    print(f"[ref] fp32-ref vs HF (all {nreal} tok): max|err|={d_hf:.6f}  | int4 vs fp32: {d_i4:.6f}  "
          f"|y|max={rf['y'][:nreal].abs().max():.4f} -> {'OK' if d_hf < 0.02 else 'MISMATCH'}")

    # ---- 4) reference bundle ----
    # k_all: [KV,D,B] (transposed K layout for q@K^T); v_all: [KV,B,D]
    k_all = rq["k"].permute(1, 2, 0).contiguous().numpy().astype(np.float32).reshape(1, KV, D, B)
    v_all = rq["v"].permute(1, 0, 2).contiguous().numpy().astype(np.float32).reshape(1, KV, B, D)
    ref = {
        "x": X_all.numpy().astype(np.float32).reshape(B, hidden),
        "sin": sin_all.numpy().astype(np.float32).reshape(1, B, rot),
        "cos": cos_all.numpy().astype(np.float32).reshape(1, B, rot),
        "q_norm_w": q_norm_w.numpy().astype(np.float32).reshape(1, 1, D),
        "k_norm_w": k_norm_w.numpy().astype(np.float32).reshape(1, 1, D),
        "eps": np.array([rms_eps], dtype=np.float32).reshape(1, 1, 1),
        "exp_y": rq["y"].numpy().astype(np.float32).reshape(B, hidden),
        "exp_k_all": k_all, "exp_v_all": v_all,
    }
    write_mllm_v2(args.out + "-ref.mllm", ref)

    # ---- 5) LPBQ weight bundle (per-head slices + QDQ scales calibrated over the block) ----
    W = {}
    qdq("model.qkv_input_qdq", X_all.abs().max().item(), W)
    qg_all = X_all @ Wq_dq.t()
    for h in range(H):
        sl = slice_head(q_buf, h * 2 * D, (h + 1) * 2 * D, n_blk)
        W[f"model.q_proj.{h}.weight"] = sl["weight"]; W[f"model.q_proj.{h}.scale1"] = sl["scale1"]
        W[f"model.q_proj.{h}.scale2"] = sl["scale2"]
        qdq(f"model.q_out_qdq_h{h}", qg_all[:, h * 2 * D:(h + 1) * 2 * D].abs().max().item(), W)
    kk_all = X_all @ Wk_dq.t(); vv_all = X_all @ Wv_dq.t()
    for kv in range(KV):
        sk = slice_head(k_buf, kv * D, (kv + 1) * D, n_blk); sv = slice_head(v_buf, kv * D, (kv + 1) * D, n_blk)
        W[f"model.k_proj.{kv}.weight"] = sk["weight"]; W[f"model.k_proj.{kv}.scale1"] = sk["scale1"]
        W[f"model.k_proj.{kv}.scale2"] = sk["scale2"]
        W[f"model.v_proj.{kv}.weight"] = sv["weight"]; W[f"model.v_proj.{kv}.scale1"] = sv["scale1"]
        W[f"model.v_proj.{kv}.scale2"] = sv["scale2"]
        qdq(f"model.k_out_qdq_h{kv}", kk_all[:, kv * D:(kv + 1) * D].abs().max().item(), W)
        qdq(f"model.v_out_qdq_h{kv}", vv_all[:, kv * D:(kv + 1) * D].abs().max().item(), W)
    W["model.o_proj.weight"] = o_buf["weight"]; W["model.o_proj.scale1"] = o_buf["scale1"]
    W["model.o_proj.scale2"] = o_buf["scale2"]
    qdq("model.o_proj_input_qdq", rq["gated"].abs().max().item(), W)
    qdq("model.o_proj_output_qdq", rq["y"].abs().max().item(), W)
    write_mllm_v2(args.out + "-lpbq.mllm", W)

    print(f"[done] layer={args.layer} B={B} -> {args.out}-lpbq.mllm + {args.out}-ref.mllm (T={T})")


if __name__ == "__main__":
    main()
