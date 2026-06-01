"""Export ONE Qwen3.5 GatedDeltaNet layer's BLOCK PREFILL (Sq=B tokens) as real-weight
LPBQ + an fp32 reference, for validating compile_deltanet_prefill.

Unlike export_deltanet_decode (Sq=1, one recurrence step), this processes a block of B
tokens in parallel via the CHUNKED gated-delta-rule (the graph uses the nilpotent-doubling
UT transform; the reference here uses the exact sequential recurrence, which is
mathematically identical and trivially correct, and we ALSO check it against the real HF
deltanet output over all B tokens).

Block-prefill I/O contract (matches the graph + future host chaining of blocks):
  inputs : X[B,hidden], S0[H,Dk,Dv] (zeros for first block), conv state cs_{q,k,v}[3,width]
           (zeros for first block), + the usual constants (eps, qscale, gated norm_w, Wa/Wb
           PRE-TRANSPOSED [hidden,LH], A_log, dt_bias, cw_{q,k,v}[4,width]).
  outputs: y[B,hidden] (per-token deltanet output), Sp[H,Dk,Dv] (state AFTER the block),
           new conv state cs'_{q,k,v}[3,width] (last 3 pre-conv windows of the block).

Reuses the LPBQ packing / QDQ helpers from export_deltanet_decode.

Usage:
  python -m pymllm.backends.qualcomm.transformers.qwen3_5.export_deltanet_prefill \
      [--model_path <hf>] [--layer 0] [--seq 128] [--out deltanet-prefill-l0] [--prompt ...]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Run BY FILE PATH (the pymllm package __init__ needs a .so that isn't built). Pull the
# shared helpers from the sibling decode exporter via direct file import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_deltanet_decode import (  # noqa: E402
    write_mllm_v2, lpbq_pack, slice_head, qdq, _default_model_path,
)


def main():
    ap = argparse.ArgumentParser(description="Export one Qwen3.5 deltanet BLOCK PREFILL: LPBQ weights + fp32 ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128, help="block length B (prompt is padded/truncated to this)")
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is Tokyo. "
                                        "Question: name a large prime number used in cryptography and explain why.")
    ap.add_argument("--out", default="deltanet-prefill-l0")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model_path} -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layer = lm.layers[args.layer]
    attn = layer.linear_attn
    H, Dk, Dv = attn.num_k_heads, attn.head_k_dim, attn.head_v_dim
    hidden, key_dim, value_dim = attn.hidden_size, attn.key_dim, attn.value_dim
    eps = float(attn.layer_norm_epsilon)
    print(f"[cfg] layer={args.layer} H={H} Dk={Dk} Dv={Dv} hidden={hidden} key_dim={key_dim} value_dim={value_dim}")

    # ---- 1) hook layer L linear_attn in/out over the WHOLE prompt ----
    captured = {}
    def pre_hook(mod, inp, kw):
        captured["x"] = (inp[0] if len(inp) > 0 else kw.get("hidden_states")).detach()
    def post_hook(mod, inp, out):
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach()
    h = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.register_forward_hook(post_hook)
    ids = tok(args.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        model(**ids)
    h.remove(); h2.remove()
    X_full = captured["x"][0].float().cpu()           # [T,hidden]
    HFY_full = captured["out"][0].float().cpu()       # [T,hidden]
    Tprompt = X_full.shape[0]
    B = args.seq
    if Tprompt >= B:
        X_all = X_full[:B]; hf_out_all = HFY_full[:B]
    else:  # pad with zeros to B (block prefill processes a fixed-size block)
        X_all = F.pad(X_full, (0, 0, 0, B - Tprompt)); hf_out_all = F.pad(HFY_full, (0, 0, 0, B - Tprompt))
    print(f"[hook] prompt T={Tprompt} -> block B={B}; X range [{X_all.min():.3f},{X_all.max():.3f}]")

    # ---- 2) LPBQ pack the three projections (deploy + dequant) ----
    qkv_buf, Wqkv_dq = lpbq_pack(attn.in_proj_qkv, args.block_size)
    z_buf, Wz_dq = lpbq_pack(attn.in_proj_z, args.block_size)
    out_buf, Wout_dq = lpbq_pack(attn.out_proj, args.block_size)
    n_blk = hidden // args.block_size

    # ---- 3) reference: sequential gated-delta recurrence over the B-token block (conv from
    #         ZERO state, S0=zeros) — mathematically identical to the chunked rule the graph
    #         uses. Produces per-token y[B,hidden] + final Sp + final conv state. ----
    A_log = attn.A_log.detach().float().cpu()
    dt_bias = attn.dt_bias.detach().float().cpu()
    norm_w = attn.norm.weight.detach().float().cpu()       # [Dv]
    Wa = attn.in_proj_a.weight.detach().float().cpu()      # [LH, hidden]
    Wb = attn.in_proj_b.weight.detach().float().cpu()
    qscale = 1.0 / (Dk ** 0.5)
    conv_dim = 2 * key_dim + value_dim
    cw_full = attn.conv1d.weight.detach().float().cpu().reshape(conv_dim, -1)   # [6144,4]
    print(f"[cfg] conv_dim={conv_dim} kernel={cw_full.shape[1]} B={B}")

    def prefill_block(Wqkv, Wz, Wout):
        mixed_pre = X_all @ Wqkv.t()                       # [B,6144] raw projections (pre-conv)
        # causal depthwise conv1d + silu over the block (left-pad 3 with zeros = fresh block)
        pad = torch.zeros(3, conv_dim)
        mp = torch.cat([pad, mixed_pre], dim=0)            # [B+3,6144]
        win = torch.stack([mp[i:i + B] for i in range(4)], dim=-1)  # [B,6144,4]
        co = (win * cw_full).sum(-1)                       # [B,6144]
        mixed = co * torch.sigmoid(co)
        q = mixed[:, :key_dim].reshape(B, H, Dk)
        k = mixed[:, key_dim:2 * key_dim].reshape(B, H, Dk)
        v = mixed[:, 2 * key_dim:].reshape(B, H, Dv)
        z = (X_all @ Wz.t()).reshape(B, H, Dv)
        qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * qscale
        kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        self_qn0 = qn[:, 0].contiguous()   # [B,Dk] head-0 normed query (graph qc probe)
        a_all = X_all @ Wa.t(); b_all = X_all @ Wb.t()
        gt_all = torch.exp(-torch.exp(A_log) * F.softplus(a_all + dt_bias))    # [B,H]
        beta_all = torch.sigmoid(b_all)                                       # [B,H]
        S = torch.zeros(H, Dk, Dv)
        gated = torch.zeros(B, H, Dv)
        core = torch.zeros(B, H, Dv)
        for t in range(B):
            for hh in range(H):
                S[hh] = S[hh] * gt_all[t, hh]
                kv = kn[t, hh] @ S[hh]
                delta = (v[t, hh] - kv) * beta_all[t, hh]
                S[hh] = S[hh] + torch.outer(kn[t, hh], delta)
                out = qn[t, hh] @ S[hh]
                core[t, hh] = out
                normed = out * torch.rsqrt((out * out).mean(-1, keepdim=True) + eps) * norm_w
                gated[t, hh] = normed * (z[t, hh] * torch.sigmoid(z[t, hh]))
        y = (gated.reshape(B, value_dim) @ Wout.t())       # [B,hidden]
        # final conv state = last 3 pre-conv windows of the block (graph's cs' output)
        cs_final = mixed_pre[B - 3:B].t().contiguous()     # [6144,3]
        # PRE-conv raw projections per head — the graph applies q/k/v_out_qdq to the raw
        # conv2d output (BEFORE depthwise conv1d+silu), so QDQ must be calibrated on these.
        q_pre = mixed_pre[:, :key_dim].reshape(B, H, Dk)
        k_pre = mixed_pre[:, key_dim:2 * key_dim].reshape(B, H, Dk)
        v_pre = mixed_pre[:, 2 * key_dim:].reshape(B, H, Dv)
        return dict(q=q_pre, k=k_pre, v=v_pre, z=z, gated=gated, core=core, Sp=S.clone(), y=y,
                    cs_final=cs_final, mixed_pre=mixed_pre, qn0=self_qn0)

    rq = prefill_block(Wqkv_dq, Wz_dq, Wout_dq)            # int4-weight reference (== graph)
    rf = prefill_block(attn.in_proj_qkv.weight.detach().float().cpu(),
                       attn.in_proj_z.weight.detach().float().cpu(),
                       attn.out_proj.weight.detach().float().cpu())  # fp32-weight reference
    # validate against the REAL HF deltanet output (only the real prompt rows, not padding)
    nreal = min(Tprompt, B)
    hferr = (rf["y"][:nreal] - hf_out_all[:nreal]).abs().max().item()
    werr = (rq["y"][:nreal] - rf["y"][:nreal]).abs().max().item()
    print(f"[ref] my fp32 ref vs ACTUAL HF deltanet (all {nreal} tokens): max|y err| = {hferr:.6f} "
          f"(HF range [{hf_out_all[:nreal].min():.3f},{hf_out_all[:nreal].max():.3f}]) "
          f"-> {'OK' if hferr < 0.05 else 'MISMATCH'}")
    print(f"[ref] int4-weight vs fp32-weight: max|y err| = {werr:.5f}")

    # ---- 4) calibrate 16-bit activation QDQ scales (max over the whole block) ----
    qdq_store = {}
    qdq("model.qkv_input_qdq", X_all.abs().max().item(), qdq_store)
    for hh in range(H):
        qdq(f"model.q_out_qdq_h{hh}", rq["q"][:, hh].abs().max().item(), qdq_store)
        qdq(f"model.k_out_qdq_h{hh}", rq["k"][:, hh].abs().max().item(), qdq_store)
        qdq(f"model.v_out_qdq_h{hh}", rq["v"][:, hh].abs().max().item(), qdq_store)
        qdq(f"model.z_out_qdq_h{hh}", rq["z"][:, hh].abs().max().item(), qdq_store)
    qdq("model.out_proj_input_qdq", rq["gated"].abs().max().item(), qdq_store)
    qdq("model.out_proj_output_qdq", rq["y"].abs().max().item(), qdq_store)

    # ---- 5a) write weights .mllm (per-head sliced) ----
    W = {}
    for hh in range(H):
        for nm, buf, base in (("q_proj", qkv_buf, 0), ("k_proj", qkv_buf, key_dim),
                              ("v_proj", qkv_buf, 2 * key_dim), ("z_proj", z_buf, 0)):
            D = Dk if nm in ("q_proj", "k_proj") else Dv
            sl = slice_head(buf, base + hh * D, base + (hh + 1) * D, n_blk)
            W[f"model.{nm}.{hh}.weight"] = sl["weight"]
            W[f"model.{nm}.{hh}.scale1"] = sl["scale1"]
            W[f"model.{nm}.{hh}.scale2"] = sl["scale2"]
    W["model.out_proj.weight"] = out_buf["weight"]
    W["model.out_proj.scale1"] = out_buf["scale1"]
    W["model.out_proj.scale2"] = out_buf["scale2"]
    W.update(qdq_store)
    write_mllm_v2(f"{args.out}-lpbq.mllm", W, model_name="qwen3_5_deltanet_prefill")

    # ---- 5b) write reference I/O .mllm (fp32; C++ runner converts to fp16/fp32) ----
    def cwslab(c0, c1):  # cw_full[width,4] -> [1,4,width]
        return cw_full[c0:c1].t().contiguous().numpy().astype(np.float32)[None]
    R = {
        "x": X_all.numpy().astype(np.float32).reshape(B, hidden),
        "eps": np.array([eps], dtype=np.float32).reshape(1, 1, 1),
        "qscale": np.array([qscale], dtype=np.float32).reshape(1, 1, 1),
        "norm_w": norm_w.numpy().astype(np.float32).reshape(1, 1, Dv),
        "A_log": A_log.numpy().astype(np.float32).reshape(1, 1, H),
        "dt_bias": dt_bias.numpy().astype(np.float32).reshape(1, 1, H),
        "Wa": Wa.t().contiguous().numpy().astype(np.float32).reshape(1, hidden, H),   # PRE-TRANSPOSED
        "Wb": Wb.t().contiguous().numpy().astype(np.float32).reshape(1, hidden, H),
        "cw_q": cwslab(0, key_dim),
        "cw_k": cwslab(key_dim, 2 * key_dim),
        "cw_v": cwslab(2 * key_dim, conv_dim),
        # first-block initial states (zeros): S0 and conv windows
        "S0": np.zeros((H, Dk, Dv), dtype=np.float32),
        "cs_q": np.zeros((1, 3, key_dim), dtype=np.float32),
        "cs_k": np.zeros((1, 3, key_dim), dtype=np.float32),
        "cs_v": np.zeros((1, 3, value_dim), dtype=np.float32),
        # expected outputs
        "exp_y": rq["y"].numpy().astype(np.float32).reshape(B, hidden),
        "exp_Sp": rq["Sp"].numpy().astype(np.float32).reshape(H, Dk, Dv),
        "exp_gated_h0": rq["gated"][:, 0].contiguous().numpy().astype(np.float32).reshape(B, Dv),  # head-0 gated (post-norm)
        "exp_gated_full": rq["gated"].reshape(B, value_dim).contiguous().numpy().astype(np.float32),  # [B,H*Dv] all heads gated (pre out_proj)
        "exp_core_full": rq["core"].reshape(B, value_dim).contiguous().numpy().astype(np.float32),    # [B,H*Dv] all heads core (pre gated-norm)
        "exp_qn_h0": rq["qn0"].numpy().astype(np.float32).reshape(B, Dk),  # head-0 normed+scaled query
    }
    write_mllm_v2(f"{args.out}-ref.mllm", R, model_name="qwen3_5_deltanet_prefill_ref")
    print(f"[done] B={B} H={H} Dk={Dk} Dv={Dv}. "
          f"Compile: --params {args.out}-lpbq.mllm --seq {B}; validate: --ref {args.out}-ref.mllm")


if __name__ == "__main__":
    main()
