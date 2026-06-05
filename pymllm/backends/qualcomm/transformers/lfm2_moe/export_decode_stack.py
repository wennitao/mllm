#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export the data a host-orchestrated LFM2 DECODE-STEP runner needs to chain N layers
# (run_lfm2_decode). For a prompt, runs HF prefill + 1 decode step and dumps, per layer:
#   * the mixer bundle  (conv: in/out_proj LPBQ + cw ; attn: per-head q/k/v/o LPBQ + KV seed)
#   * the ffn bundle    (dense: gate/up/down LPBQ ; MoE: 32 expert bundles + router)
#   * the seed state going INTO the decode step (conv cs[K-1,H] ; attn KV[KV,T,D])
# plus consts (operator_norm_w, ffn_norm_w per layer, embedding_norm_w, eps, rope cos/sin
# at the decode position), the embed row for the decode token, and a per-layer GOLDEN
# (HF fp32 residual-stream hidden after each layer at the decode step) + final logits.
#
# All into <out_dir>/. The runner validates hidden-after-layer-L vs golden (loose int4
# tolerance) and finally argmax(logits) vs the golden next token.
#
#   python3 export_decode_stack.py --layers 4 --out_dir /path/wm_dec
import argparse
import glob
import json
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

_MAGIC, _VERSION = 0x519A, 2
_NAME_LEN, _PNAME_LEN, _SHAPE_LEN = 512, 256, 16
_HDR_SIZE, _DESC_SIZE = 532, 352
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18, np.dtype("uint8"): 129, np.dtype("float16"): 1}


def write_mllm_v2(path, tensors, model_name="lfm2_dec"):
    items = list(tensors.items()); n = len(items)
    with open(path, "wb") as g:
        g.write(b"\x00" * (_HDR_SIZE + n * _DESC_SIZE))
        descs = []
        for pid, (name, arr) in enumerate(items):
            arr = np.ascontiguousarray(arr); data = arr.tobytes(); off = g.tell(); g.write(data)
            nb = name.encode("utf-8")[:_PNAME_LEN].ljust(_PNAME_LEN, b"\0")
            shp = list(arr.shape) + [0] * (_SHAPE_LEN - len(arr.shape))
            descs.append(struct.pack(f"<IIQQQ{_SHAPE_LEN}i{_PNAME_LEN}s", pid, _CODE[arr.dtype], len(data), off, len(arr.shape), *shp, nb))
        for pid, d in enumerate(descs):
            g.seek(_HDR_SIZE + pid * _DESC_SIZE); g.write(d)
        g.seek(0)
        mn = model_name.encode("utf-8")[:_NAME_LEN].ljust(_NAME_LEN, b"\0")
        g.write(struct.pack(f"<II{_NAME_LEN}sIQ", _MAGIC, _VERSION, mn, n, _HDR_SIZE))
    print(f"[write] {os.path.basename(path)}: {n} tensors")


def _import_qlinear():
    here = os.path.dirname(os.path.abspath(__file__)); core = os.path.normpath(os.path.join(here, "..", "core"))
    chain = {"pymllm": os.path.normpath(os.path.join(here, "..", "..", "..", "..")),
             "pymllm.backends": os.path.normpath(os.path.join(here, "..", "..", "..")),
             "pymllm.backends.qualcomm": os.path.normpath(os.path.join(here, "..", "..")),
             "pymllm.backends.qualcomm.transformers": os.path.normpath(os.path.join(here, "..")),
             "pymllm.backends.qualcomm.transformers.core": core}
    for name, path in chain.items():
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    from pymllm.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ
    return QLinearLPBQ


QLinearLPBQ = _import_qlinear()


def dequant_from_deploy(buf, in_f, out_f, bs):
    n_blk = in_f // bs
    w = buf["weight"].reshape(in_f, out_f).astype(np.int32); w = np.where(w >= 8, w - 16, w)
    s1 = buf["scale1"].reshape(out_f, n_blk).astype(np.float32); s2 = buf["scale2"].reshape(out_f).astype(np.float32)
    return torch.from_numpy(w.T.astype(np.float32) * np.repeat(s1 * s2[:, None], bs, axis=1)).float()


@torch.no_grad()
def lpbq_pack(weight, bs):
    out_f, in_f = weight.shape
    q = QLinearLPBQ(in_f, out_f, bias=False, block_size=bs); q.weight.data.copy_(weight.float()); q = q.to(torch.float32)
    q.freeze_weight(); q.enable_fakequant(); q.convert_to_conv2d_deploy_hwio()
    buf = {"weight": q.weight.cpu().numpy().astype(np.int8), "scale1": q.scale1.cpu().numpy().astype(np.uint8),
           "scale2": q.scale2.cpu().numpy().astype(np.float32)}
    return buf, dequant_from_deploy(buf, in_f, out_f, bs)


def slice_head(buf, c0, c1, n_blk):
    return {"weight": np.ascontiguousarray(buf["weight"][:, :, :, c0:c1]),
            "scale1": np.ascontiguousarray(buf["scale1"][c0 * n_blk:c1 * n_blk]),
            "scale2": np.ascontiguousarray(buf["scale2"][c0:c1])}


def qdq(name, maxabs, store):
    store[name + ".fake_quant.scale"] = np.array([float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def _default_model_path():
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "LiquidAI/LFM2.5-8B-A1B"


def get_expert(ff, e, inter):
    if hasattr(ff.experts, "gate_up_proj"):
        gu = ff.experts.gate_up_proj.data[e].float()
        return gu[:inter, :].contiguous(), gu[inter:, :].contiguous(), ff.experts.down_proj.data[e].float().contiguous()
    ex = ff.experts[e]
    return ex.w1.weight.data.float(), ex.w3.weight.data.float(), ex.w2.weight.data.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4, help="export layers 0..N-1")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out_dir", default="/mnt/raid0_ssd/wentao/mllm/build-qnn-aot/bin/wm_dec")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bs, NL, ctx = args.block_size, args.layers, args.ctx

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float32, attn_implementation="eager").eval()
    model.config._attn_implementation = "eager"
    lm, cfg = model.model, model.config
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    hidden, D = cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads
    inter_moe, inter_dense = cfg.moe_intermediate_size, cfg.intermediate_size
    E, top_k, K = cfg.num_experts, cfg.num_experts_per_tok, cfg.conv_L_cache
    eps = float(cfg.norm_eps)
    layer_types = list(cfg.layer_types)
    n_dense = cfg.num_dense_layers
    rope_theta = float(cfg.rope_parameters["rope_theta"])

    ids = tok(args.prompt, return_tensors="pt").input_ids
    T = ids.shape[1]
    assert T < ctx

    # capture mixer inputs (operator_norm output) over the PROMPT to rebuild dequant-weight
    # seed states (conv window / attn KV) consistent with the on-device int4 weights.
    mix_in = {}
    hooks = []
    for i in range(NL):
        mod = lm.layers[i].self_attn if layer_types[i] == "full_attention" else lm.layers[i].conv
        def mk(i):
            def f(m, inp, kw):
                hs = inp[0] if len(inp) else kw.get("hidden_states"); mix_in[i] = hs.detach(); return None
            return f
        hooks.append(mod.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.no_grad():
        pf = model(ids, use_cache=True, output_hidden_states=True)
    for h in hooks: h.remove()
    conv_in = mix_in
    cache = pf.past_key_values
    next_tok = int(pf.logits[0, -1].argmax())
    print(f"[prefill] T={T} next_tok={next_tok}", flush=True)

    # decode step: feed next_tok, capture per-layer residual stream + final logits.
    with torch.no_grad():
        dec = model(torch.tensor([[next_tok]]), past_key_values=cache, use_cache=True, output_hidden_states=True)
    hs_dec = dec.hidden_states               # len L+1: [embed, after L0, after L1, ...]
    dec_logits = dec.logits[0, -1].float()
    print(f"[decode] argmax={int(dec_logits.argmax())}", flush=True)

    # rope cos/sin at the decode position (= T)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D, 2).float() / D))
    ang = torch.tensor([float(T)]) [:, None] * inv_freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[0]   # [D]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[0]

    consts = {"eps": np.array([eps], dtype=np.float32),
              "cos": cos.numpy().astype(np.float32).reshape(1, 1, D), "sin": sin.numpy().astype(np.float32).reshape(1, 1, D),
              "embedding_norm_w": f32(lm.embedding_norm.weight),
              "embed_row": lm.embed_tokens.weight.data[next_tok].float().cpu().numpy().astype(np.float32).reshape(1, hidden),
              "next_tok": np.array([next_tok], dtype=np.int32),
              "golden_logits": dec_logits.numpy().astype(np.float32),
              "golden_argmax": np.array([int(dec_logits.argmax())], dtype=np.int32)}

    manifest = {"layers": [], "hidden": hidden, "H": H, "KV": KV, "D": D, "K": K, "ctx": ctx,
                "inter_moe": inter_moe, "inter_dense": inter_dense, "E": E, "top_k": top_k,
                "n_dense": n_dense, "eps": eps, "T": T, "next_tok": next_tok, "vocab": cfg.vocab_size}

    n_blk = hidden // bs
    scale = 1.0 / (D ** 0.5)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D, 2).float() / D))

    def hrms(x, w):  # host RMSNorm (plain weight), fp32 ; x [..,n]
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w

    def rope_row(x, pos):  # x [.., D] full RoPE at position pos
        ang = torch.tensor([float(pos)])[:, None] * inv_freq[None, :]
        c = torch.cat([ang.cos(), ang.cos()], -1)[0]; s = torch.cat([ang.sin(), ang.sin()], -1)[0]
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        return x * c + torch.cat([-x2, x1], -1) * s

    # The chained DEQUANT-weight residual stream (what the on-device int4 graph reproduces).
    hid = torch.from_numpy(consts["embed_row"]).float().reshape(hidden)

    for i in range(NL):
        layer = lm.layers[i]
        op_w = layer.operator_norm.weight.detach().float()
        ffn_w = layer.ffn_norm.weight.detach().float()
        consts[f"l{i}.operator_norm_w"] = f32(op_w)
        consts[f"l{i}.ffn_norm_w"] = f32(ffn_w)
        lt = "attn" if layer_types[i] == "full_attention" else "conv"

        # ---- mixer (dequant-weight decode forward; calibrate QDQ from real activations) ----
        r = hid
        h = hrms(hid, op_w)                                          # [hidden] mixer input
        if lt == "conv":
            c = layer.conv
            cw = c.conv.weight.data.float().reshape(hidden, K)
            in_buf, Wi = lpbq_pack(c.in_proj.weight.data.float(), bs)
            out_buf, Wo = lpbq_pack(c.out_proj.weight.data.float(), bs)
            Xp = mix_in[i][0].float()                               # [T, hidden] prompt mixer inputs
            pp = Xp @ Wi.t(); Bxp = pp[:, :hidden] * pp[:, 2 * hidden:]
            cs = Bxp[T - (K - 1):T] if T >= K - 1 else torch.zeros(K - 1, hidden)
            proj = h @ Wi.t()
            Bg, Cg, xin = proj[:hidden], proj[hidden:2 * hidden], proj[2 * hidden:]
            win = torch.cat([cs, (Bg * xin)[None]], 0)              # [K, hidden]
            conv = (cw.t() * win).sum(0)
            y = Cg * conv; mix = y @ Wo.t()
            M = {}
            for nm, b in [("in_proj", in_buf), ("out_proj", out_buf)]:
                M[f"model.{nm}.weight"] = b["weight"]; M[f"model.{nm}.scale1"] = b["scale1"]; M[f"model.{nm}.scale2"] = b["scale2"]
            qdq("model.in_proj_input_qdq", float(h.abs().max()), M)
            qdq("model.in_proj_output_qdq", float(proj.abs().max()), M)
            qdq("model.out_proj_input_qdq", float(y.abs().max()), M)
            qdq("model.out_proj_output_qdq", float(mix.abs().max()), M)
            write_mllm_v2(os.path.join(args.out_dir, f"l{i}_mixer-lpbq.mllm"), M)
            consts[f"l{i}.cw"] = cw.t().contiguous().numpy().astype(np.float32).reshape(1, K, hidden)
            consts[f"l{i}.cs"] = cs.contiguous().numpy().astype(np.float32).reshape(1, K - 1, hidden)
        else:
            a = layer.self_attn
            q_buf, Wq = lpbq_pack(a.q_proj.weight.data.float(), bs)
            k_buf, Wk = lpbq_pack(a.k_proj.weight.data.float(), bs)
            v_buf, Wv = lpbq_pack(a.v_proj.weight.data.float(), bs)
            o_buf, Wo = lpbq_pack(a.out_proj.weight.data.float(), bs)
            qnw = a.q_layernorm.weight.detach().float(); knw = a.k_layernorm.weight.detach().float()
            Xp = mix_in[i][0].float()                              # [T, hidden]
            # prompt KV (dequant) for the seed + decode K/V at position T
            kp = (Xp @ Wk.t()).reshape(T, KV, D); vp = (Xp @ Wv.t()).reshape(T, KV, D)
            kp = torch.stack([rope_row(hrms(kp[:, kv], knw), p)
                              for p in range(T) for kv in range(KV)]).reshape(T, KV, D) if False else kp
            # (compute per (t,kv) rope/norm)
            kfull = torch.zeros(T + 1, KV, D); vfull = torch.zeros(T + 1, KV, D)
            for t in range(T):
                for kv in range(KV):
                    kfull[t, kv] = rope_row(hrms(kp[t, kv], knw), t)
                    vfull[t, kv] = vp[t, kv]
            kd = (h @ Wk.t()).reshape(KV, D); vd = (h @ Wv.t()).reshape(KV, D)
            qd = (h @ Wq.t()).reshape(H, D)
            for kv in range(KV):
                kfull[T, kv] = rope_row(hrms(kd[kv], knw), T); vfull[T, kv] = vd[kv]
            outs = torch.zeros(H, D)
            for hh in range(H):
                kv = hh // (H // KV)
                q = rope_row(hrms(qd[hh], qnw), T)
                sc = (q @ kfull[:T + 1, kv].t()) * scale
                outs[hh] = torch.softmax(sc, -1) @ vfull[:T + 1, kv]
            attn_out = outs.reshape(H * D); mix = attn_out @ Wo.t()
            M = {}
            qdq("model.qkv_input_qdq", float(h.abs().max()), M)
            qg_all = Xp @ Wq.t()
            for hh in range(H):
                sl = slice_head(q_buf, hh * D, (hh + 1) * D, n_blk)
                M[f"model.q_proj.{hh}.weight"] = sl["weight"]; M[f"model.q_proj.{hh}.scale1"] = sl["scale1"]; M[f"model.q_proj.{hh}.scale2"] = sl["scale2"]
                qdq(f"model.q_out_qdq_h{hh}", float(qg_all[:, hh * D:(hh + 1) * D].abs().max()), M)
            kk_all = Xp @ Wk.t(); vv_all = Xp @ Wv.t()
            for kv in range(KV):
                sk = slice_head(k_buf, kv * D, (kv + 1) * D, n_blk); sv = slice_head(v_buf, kv * D, (kv + 1) * D, n_blk)
                for nm, s in [("k_proj", sk), ("v_proj", sv)]:
                    M[f"model.{nm}.{kv}.weight"] = s["weight"]; M[f"model.{nm}.{kv}.scale1"] = s["scale1"]; M[f"model.{nm}.{kv}.scale2"] = s["scale2"]
                qdq(f"model.k_out_qdq_h{kv}", float(kk_all[:, kv * D:(kv + 1) * D].abs().max()), M)
                qdq(f"model.v_out_qdq_h{kv}", float(vv_all[:, kv * D:(kv + 1) * D].abs().max()), M)
            M["model.o_proj.weight"] = o_buf["weight"]; M["model.o_proj.scale1"] = o_buf["scale1"]; M["model.o_proj.scale2"] = o_buf["scale2"]
            qdq("model.o_proj_input_qdq", float(attn_out.abs().max()), M); qdq("model.o_proj_output_qdq", float(mix.abs().max()), M)
            write_mllm_v2(os.path.join(args.out_dir, f"l{i}_mixer-lpbq.mllm"), M)
            consts[f"l{i}.q_norm_w"] = qnw.numpy().astype(np.float32).reshape(1, 1, D)
            consts[f"l{i}.k_norm_w"] = knw.numpy().astype(np.float32).reshape(1, 1, D)
            P = ctx - 1
            past_k = np.zeros((KV, D, P), dtype=np.float32); past_v = np.zeros((KV, P, D), dtype=np.float32)
            for p in range(min(T, P)):
                for kv in range(KV):
                    past_k[kv, :, p] = kfull[p, kv].numpy(); past_v[kv, p, :] = vfull[p, kv].numpy()
            mask = np.full((ctx,), -50000.0, dtype=np.float32); mask[:T] = 0.0; mask[ctx - 1] = 0.0
            consts[f"l{i}.past_k"] = past_k; consts[f"l{i}.past_v"] = past_v; consts[f"l{i}.mask"] = mask
        hid = r + mix

        # ---- ffn (dequant-weight; calibrate QDQ from real activations) ----
        r2 = hid
        h2 = hrms(hid, ffn_w)
        ff = layer.feed_forward
        if i < n_dense:
            g_buf, Wg = lpbq_pack(ff.w1.weight.data.float(), bs); u_buf, Wu = lpbq_pack(ff.w3.weight.data.float(), bs); d_buf, Wd = lpbq_pack(ff.w2.weight.data.float(), bs)
            gate = h2 @ Wg.t(); up = h2 @ Wu.t(); act = F.silu(gate); interm = act * up; ffn = interm @ Wd.t()
            FF = {}
            for nm, b in [("gate_proj", g_buf), ("up_proj", u_buf), ("down_proj", d_buf)]:
                FF[f"model.{nm}.weight"] = b["weight"]; FF[f"model.{nm}.scale1"] = b["scale1"]; FF[f"model.{nm}.scale2"] = b["scale2"]
            qdq("model.up_proj_input_qdq", float(h2.abs().max()), FF); qdq("model.up_proj_output_qdq", float(up.abs().max()), FF)
            qdq("model.gate_proj_output_qdq", float(gate.abs().max()), FF); qdq("model.sigmoid_output_qdq", 1.0, FF)
            qdq("model.act_output_qdq", float(act.abs().max()), FF); qdq("model.down_proj_input_qdq", float(interm.abs().max()), FF)
            qdq("model.down_proj_output_qdq", float(ffn.abs().max()), FF)
            write_mllm_v2(os.path.join(args.out_dir, f"l{i}_ffn-lpbq.mllm"), FF)
            manifest["layers"].append({"idx": i, "mixer": lt, "ffn": "dense"})
        else:
            mdir = os.path.join(args.out_dir, f"l{i}_moe"); os.makedirs(mdir, exist_ok=True)
            gw = ff.gate.weight.data.float(); bias = ff.expert_bias.data.float()
            rw = torch.sigmoid(h2 @ gw.t()); sc = rw + bias
            sel = torch.topk(sc, top_k).indices; w = rw[sel]
            if cfg.norm_topk_prob: w = w / (w.sum() + 1e-6)
            w = w * float(cfg.routed_scaling_factor)
            ffn = torch.zeros(hidden)
            for e in range(E):
                Wg, Wu, Wd = get_expert(ff, e, inter_moe)
                gb, Wg_dq = lpbq_pack(Wg, bs); ub, Wu_dq = lpbq_pack(Wu, bs); db, Wd_dq = lpbq_pack(Wd, bs)
                gate = h2 @ Wg_dq.t(); up = h2 @ Wu_dq.t(); act = F.silu(gate); interm = act * up; ye = interm @ Wd_dq.t()
                hit = (sel == e).nonzero(as_tuple=True)[0]
                if hit.numel(): ffn = ffn + float(w[int(hit[0])]) * ye
                W = {}
                for nm, b in [("gate_proj", gb), ("up_proj", ub), ("down_proj", db)]:
                    W[f"model.{nm}.weight"] = b["weight"]; W[f"model.{nm}.scale1"] = b["scale1"]; W[f"model.{nm}.scale2"] = b["scale2"]
                qdq("model.up_proj_input_qdq", float(h2.abs().max()), W); qdq("model.up_proj_output_qdq", float(up.abs().max()), W)
                qdq("model.gate_proj_output_qdq", float(gate.abs().max()), W); qdq("model.sigmoid_output_qdq", 1.0, W)
                qdq("model.act_output_qdq", float(act.abs().max()), W); qdq("model.down_proj_input_qdq", float(interm.abs().max()), W)
                qdq("model.down_proj_output_qdq", float(ye.abs().max()), W)
                write_mllm_v2(os.path.join(mdir, f"expert{e}-lpbq.mllm"), W)
            write_mllm_v2(os.path.join(mdir, "router.mllm"), {
                "gate_weight": f32(gw), "expert_bias": f32(bias),
                "meta": np.array([i, E, top_k, hidden, inter_moe, int(cfg.norm_topk_prob)], dtype=np.int32),
                "scaling": np.array([float(cfg.routed_scaling_factor)], dtype=np.float32)})
            manifest["layers"].append({"idx": i, "mixer": lt, "ffn": "moe"})
        hid = r2 + ffn
        consts[f"l{i}.golden_hidden"] = f32(hid)
        print(f"  layer {i}: {lt} + {'dense' if i < n_dense else 'moe'} done (|hid|max={float(hid.abs().max()):.3f})", flush=True)

    # ---- lm_head (tied embed_tokens): final RMSNorm in-graph + LPBQ proj -> logits ----
    emb_w = lm.embedding_norm.weight.detach().float()
    lm_buf, Wlm_dq = lpbq_pack(lm.embed_tokens.weight.data.float(), bs)      # [vocab, hidden]
    hn = hrms(hid, emb_w)                                                     # final-normed hidden
    logits_dq = hn @ Wlm_dq.t()
    LM = {"model.lm_head.weight": lm_buf["weight"], "model.lm_head.scale1": lm_buf["scale1"], "model.lm_head.scale2": lm_buf["scale2"]}
    qdq("model.lmhead_input_qdq", float(hn.abs().max()), LM)
    qdq("model.lmhead_output_qdq", float(logits_dq.abs().max()), LM)
    write_mllm_v2(os.path.join(args.out_dir, "lmhead-lpbq.mllm"), LM)
    consts["dequant_argmax"] = np.array([int(logits_dq.argmax())], dtype=np.int32)
    print(f"[lmhead] dequant-weight argmax={int(logits_dq.argmax())} (HF decode argmax={int(dec_logits.argmax())})")

    write_mllm_v2(os.path.join(args.out_dir, "consts.mllm"), consts)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as g:
        json.dump(manifest, g, indent=2)
    print(f"[done] {NL} layers -> {args.out_dir}  next_tok={next_tok} golden_argmax={int(dec_logits.argmax())}")


if __name__ == "__main__":
    main()
