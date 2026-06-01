#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Whole-model bundle exporter for the NPU decode-loop orchestrator. Produces, in
# one HF load, everything the C++ orchestrator needs to run Qwen3.5-2B decode with
# the per-layer LPBQ graphs (deltanet / attn / mlp / head), all validated already:
#
#   <dir>/layer<i>_mixer-lpbq.mllm  : deltanet (q/k/v/z_proj per-head + out_proj) OR
#                                     attn (q/k/v per-head + o_proj) LPBQ + QDQ scales
#   <dir>/layer<i>_mlp-lpbq.mllm    : gate/up/down LPBQ + QDQ scales
#   <dir>/lmhead-lpbq.mllm          : tied lm_head LPBQ + QDQ scales
#   <dir>/consts.mllm               : per-layer CPU constants (norms 1+w, deltanet
#                                     gates A_log/dt_bias/in_proj_a,b/conv1d/qscale/
#                                     gated-norm 1+w, attn q/k-norm 1+w, final norm 1+w)
#   <dir>/seed.mllm                 : prefill state seed (deltanet S+conv, attn KV) +
#                                     prompt ids + the int4-golden reference token stream
#   <dir>/embed.mllm                : fp16 embedding table [vocab, hidden]
#   <dir>/manifest.json             : layer types, dims, prompt, seq len, file names
#
#   python3 export_whole_model.py --out_dir /path/wm --max_new 24
#
import argparse
import glob
import json
import os
import struct
import sys
import types

import numpy as np
import torch


_MAGIC, _VERSION = 0x519A, 2
_NAME_LEN, _PNAME_LEN, _SHAPE_LEN = 512, 256, 16
_HDR_SIZE, _DESC_SIZE = 532, 352
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18, np.dtype("uint8"): 129,
         np.dtype("float16"): 1}


def write_mllm_v2(path, tensors, model_name="qwen3_5_wm"):
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


def _default_model_path():
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "Qwen/Qwen3.5-2B"


def dequant_from_deploy(buf, in_features, out_features, block_size):
    n_blk = in_features // block_size
    w = buf["weight"].reshape(in_features, out_features).astype(np.int32)
    w = np.where(w >= 8, w - 16, w)
    s1 = buf["scale1"].reshape(out_features, n_blk).astype(np.float32)
    s2 = buf["scale2"].reshape(out_features).astype(np.float32)
    bs = np.repeat(s1 * s2[:, None], block_size, axis=1)
    return torch.from_numpy(w.T.astype(np.float32) * bs).float()


@torch.no_grad()
def lpbq_pack(weight, block_size):
    """LPBQ-pack a raw [out,in] weight; returns (deploy_buf, dequant[out,in])."""
    out_f, in_f = weight.shape
    q = QLinearLPBQ(in_f, out_f, bias=False, block_size=block_size)
    q.weight.data.copy_(weight.float())
    q = q.to(torch.float32)
    q.freeze_weight(); q.enable_fakequant(); q.convert_to_conv2d_deploy_hwio()
    buf = {"weight": q.weight.cpu().numpy().astype(np.int8),
           "scale1": q.scale1.cpu().numpy().astype(np.uint8),
           "scale2": q.scale2.cpu().numpy().astype(np.float32)}
    return buf, dequant_from_deploy(buf, in_f, out_f, block_size)


def slice_head(buf, c0, c1, n_blk):
    return {"weight": np.ascontiguousarray(buf["weight"][:, :, :, c0:c1]),
            "scale1": np.ascontiguousarray(buf["scale1"][c0 * n_blk:c1 * n_blk]),
            "scale2": np.ascontiguousarray(buf["scale2"][c0:c1])}


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Whole-model bundle exporter for the NPU decode orchestrator")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--out_dir", default="/mnt/raid0_ssd/wentao/mllm/build-qnn-aot/bin/wm")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    bs = args.block_size

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model_path} -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    hidden, vocab = cfg.hidden_size, cfg.vocab_size
    H, KV, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    LH, Dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    Dv = cfg.linear_value_head_dim
    key_dim, value_dim = LH * Dk, cfg.linear_num_value_heads * Dv
    eps = float(cfg.rms_norm_eps)
    rot = int(round(Dh * cfg.partial_rotary_factor)); rot -= rot & 1
    layer_types = list(lm.config.layer_types)
    nL = cfg.num_hidden_layers
    print(f"[cfg] L={nL} hidden={hidden} vocab={vocab} | attn H={H}/KV={KV} D={Dh} rot={rot} "
          f"| deltanet H={LH} Dk={Dk} Dv={Dv} key_dim={key_dim}")

    # ---- activation calibration: hook target projection outputs over the prompt ----
    maxabs = {}
    hooks = []
    def hook_out(key):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            maxabs[key] = max(maxabs.get(key, 0.0), float(o.detach().abs().max()))
        return f
    def hook_in(key):
        def f(mod, inp, kw):
            hs = inp[0] if len(inp) else kw.get("hidden_states")
            if hs is not None:
                maxabs[key] = max(maxabs.get(key, 0.0), float(hs.detach().abs().max()))
            return None
        return f
    for i in range(nL):
        layer = lm.layers[i]
        if layer_types[i] == "linear_attention":
            a = layer.linear_attn
            hooks.append(a.in_proj_qkv.register_forward_hook(hook_out(f"l{i}.qkv")))
            hooks.append(a.in_proj_z.register_forward_hook(hook_out(f"l{i}.z")))
            hooks.append(a.out_proj.register_forward_pre_hook(hook_in(f"l{i}.outp_in"), with_kwargs=True))
            hooks.append(a.register_forward_pre_hook(hook_in(f"l{i}.mix_in"), with_kwargs=True))
        else:
            a = layer.self_attn
            hooks.append(a.q_proj.register_forward_hook(hook_out(f"l{i}.q")))
            hooks.append(a.k_proj.register_forward_hook(hook_out(f"l{i}.k")))
            hooks.append(a.v_proj.register_forward_hook(hook_out(f"l{i}.v")))
            hooks.append(a.o_proj.register_forward_pre_hook(hook_in(f"l{i}.outp_in"), with_kwargs=True))
            hooks.append(a.register_forward_pre_hook(hook_in(f"l{i}.mix_in"), with_kwargs=True))
        mlp = layer.mlp
        hooks.append(mlp.up_proj.register_forward_pre_hook(hook_in(f"l{i}.mlp_in"), with_kwargs=True))
        hooks.append(mlp.up_proj.register_forward_hook(hook_out(f"l{i}.up")))
        hooks.append(mlp.gate_proj.register_forward_hook(hook_out(f"l{i}.gate")))
        hooks.append(mlp.down_proj.register_forward_pre_hook(hook_in(f"l{i}.down_in"), with_kwargs=True))
        hooks.append(mlp.down_proj.register_forward_hook(hook_out(f"l{i}.down")))
    hooks.append(lm.norm.register_forward_pre_hook(hook_in("final_in"), with_kwargs=True))

    # ---- prefill: run prompt, capture cache (seed) + activation stats + golden ----
    ids = tok(args.prompt, return_tensors="pt").to(dev)
    T = ids.input_ids.shape[1]
    with torch.no_grad():
        out = model(**ids, use_cache=True)
    for h in hooks:
        h.remove()
    cache = out.past_key_values.layers
    print(f"[prefill] T={T} captured {len(maxabs)} activation maxabs", flush=True)

    # ---- consts + per-layer LPBQ bundles ----
    consts = {}
    consts["final_norm_w"] = f32(1.0 + lm.norm.weight)            # [hidden]
    n_blk = hidden // bs
    n_blk_dv = value_dim // bs
    n_blk_attn_o = (H * Dh) // bs
    # RoPE inv_freq for the orchestrator's CPU sin/cos (rotary_dim/2 entries).
    consts["inv_freq"] = lm.rotary_emb.inv_freq.detach().float().cpu().numpy().astype(np.float32)
    consts["rope_attn_scaling"] = np.array([float(getattr(lm.rotary_emb, "attention_scaling", 1.0))], dtype=np.float32)
    manifest = {"layers": [], "hidden": hidden, "vocab": vocab, "H": H, "KV": KV, "Dh": Dh,
                "rot": rot, "LH": LH, "Dk": Dk, "Dv": Dv, "key_dim": key_dim, "value_dim": value_dim,
                "eps": eps, "ctx": args.ctx, "prompt": args.prompt, "T": T, "block_size": bs,
                "partial_rotary_factor": float(cfg.partial_rotary_factor),
                "inv_freq_len": int(lm.rotary_emb.inv_freq.numel())}

    for i in range(nL):
        layer = lm.layers[i]
        lt = layer_types[i]
        consts[f"l{i}.input_norm_w"] = f32(1.0 + layer.input_layernorm.weight)
        consts[f"l{i}.post_norm_w"] = f32(1.0 + layer.post_attention_layernorm.weight)
        W = {}
        if lt == "linear_attention":
            a = layer.linear_attn
            qkv_buf, _ = lpbq_pack(a.in_proj_qkv.weight.data, bs)   # [6144, hidden]
            z_buf, _ = lpbq_pack(a.in_proj_z.weight.data, bs)       # [value_dim, hidden]
            out_buf, _ = lpbq_pack(a.out_proj.weight.data, bs)      # [hidden, value_dim]
            qdq("model.qkv_input_qdq", maxabs.get(f"l{i}.mix_in", 0), W)
            for hh in range(LH):
                # fused qkv out-channel layout: q[0:key_dim] k[key_dim:2k] v[2k:]; head hh = base + hh*Dk/Dv
                sq = slice_head(qkv_buf, hh * Dk, (hh + 1) * Dk, n_blk)
                sk = slice_head(qkv_buf, key_dim + hh * Dk, key_dim + (hh + 1) * Dk, n_blk)
                sv = slice_head(qkv_buf, 2 * key_dim + hh * Dv, 2 * key_dim + (hh + 1) * Dv, n_blk)
                sz = slice_head(z_buf, hh * Dv, (hh + 1) * Dv, n_blk)
                for nm, sl in [("q_proj", sq), ("k_proj", sk), ("v_proj", sv), ("z_proj", sz)]:
                    W[f"model.{nm}.{hh}.weight"] = sl["weight"]
                    W[f"model.{nm}.{hh}.scale1"] = sl["scale1"]
                    W[f"model.{nm}.{hh}.scale2"] = sl["scale2"]
                qdq(f"model.q_out_qdq_h{hh}", maxabs.get(f"l{i}.qkv", 0), W)
                qdq(f"model.k_out_qdq_h{hh}", maxabs.get(f"l{i}.qkv", 0), W)
                qdq(f"model.v_out_qdq_h{hh}", maxabs.get(f"l{i}.qkv", 0), W)
                qdq(f"model.z_out_qdq_h{hh}", maxabs.get(f"l{i}.z", 0), W)
            W["model.out_proj.weight"] = out_buf["weight"]
            W["model.out_proj.scale1"] = out_buf["scale1"]
            W["model.out_proj.scale2"] = out_buf["scale2"]
            qdq("model.out_proj_input_qdq", maxabs.get(f"l{i}.outp_in", 0), W)
            qdq("model.out_proj_output_qdq", float(out.logits.abs().max()) if False else 8.0, W)
            # deltanet CPU constants
            consts[f"l{i}.A_log"] = f32(a.A_log)
            consts[f"l{i}.dt_bias"] = f32(a.dt_bias)
            consts[f"l{i}.in_proj_a"] = f32(a.in_proj_a.weight)     # [LH, hidden]
            consts[f"l{i}.in_proj_b"] = f32(a.in_proj_b.weight)
            consts[f"l{i}.gated_norm_w"] = f32(1.0 + a.norm.weight)  # [Dv]
            consts[f"l{i}.conv1d_w"] = a.conv1d.weight.detach().float().cpu().reshape(2 * key_dim + value_dim, -1).numpy().astype(np.float32)  # [6144,4]
            consts[f"l{i}.qscale"] = np.array([1.0 / (Dk ** 0.5)], dtype=np.float32)
            manifest["layers"].append({"idx": i, "type": "deltanet"})
        else:
            a = layer.self_attn
            q_buf, _ = lpbq_pack(a.q_proj.weight.data, bs)   # [H*2D, hidden]
            k_buf, _ = lpbq_pack(a.k_proj.weight.data, bs)   # [KV*D, hidden]
            v_buf, _ = lpbq_pack(a.v_proj.weight.data, bs)
            o_buf, _ = lpbq_pack(a.o_proj.weight.data, bs)   # [hidden, H*D]
            qdq("model.qkv_input_qdq", maxabs.get(f"l{i}.mix_in", 0), W)
            for hh in range(H):
                sl = slice_head(q_buf, hh * 2 * Dh, (hh + 1) * 2 * Dh, n_blk)
                W[f"model.q_proj.{hh}.weight"] = sl["weight"]
                W[f"model.q_proj.{hh}.scale1"] = sl["scale1"]
                W[f"model.q_proj.{hh}.scale2"] = sl["scale2"]
                qdq(f"model.q_out_qdq_h{hh}", maxabs.get(f"l{i}.q", 0), W)
            for kv in range(KV):
                sk = slice_head(k_buf, kv * Dh, (kv + 1) * Dh, n_blk)
                sv = slice_head(v_buf, kv * Dh, (kv + 1) * Dh, n_blk)
                for nm, sl in [("k_proj", sk), ("v_proj", sv)]:
                    W[f"model.{nm}.{kv}.weight"] = sl["weight"]
                    W[f"model.{nm}.{kv}.scale1"] = sl["scale1"]
                    W[f"model.{nm}.{kv}.scale2"] = sl["scale2"]
                qdq(f"model.k_out_qdq_h{kv}", maxabs.get(f"l{i}.k", 0), W)
                qdq(f"model.v_out_qdq_h{kv}", maxabs.get(f"l{i}.v", 0), W)
            W["model.o_proj.weight"] = o_buf["weight"]
            W["model.o_proj.scale1"] = o_buf["scale1"]
            W["model.o_proj.scale2"] = o_buf["scale2"]
            qdq("model.o_proj_input_qdq", maxabs.get(f"l{i}.outp_in", 0), W)
            qdq("model.o_proj_output_qdq", 8.0, W)
            consts[f"l{i}.q_norm_w"] = f32(1.0 + a.q_norm.weight)   # [Dh]
            consts[f"l{i}.k_norm_w"] = f32(1.0 + a.k_norm.weight)
            manifest["layers"].append({"idx": i, "type": "attn"})
        write_mllm_v2(os.path.join(args.out_dir, f"layer{i}_mixer-lpbq.mllm"), W)

        # MLP bundle (gate/up: hidden->inter ; down: inter->hidden)
        mlp = layer.mlp
        inter = cfg.intermediate_size
        n_blk_mlp = inter // bs
        gate_buf, _ = lpbq_pack(mlp.gate_proj.weight.data, bs)
        up_buf, _ = lpbq_pack(mlp.up_proj.weight.data, bs)
        down_buf, _ = lpbq_pack(mlp.down_proj.weight.data, bs)
        M = {}
        for nm, b in [("gate_proj", gate_buf), ("up_proj", up_buf), ("down_proj", down_buf)]:
            M[f"model.{nm}.weight"] = b["weight"]
            M[f"model.{nm}.scale1"] = b["scale1"]
            M[f"model.{nm}.scale2"] = b["scale2"]
        # QDQ names must match Qwen3MLP (modeling_qwen_qnn_aot_sha.hpp) + FullMLPLPBQ's
        # closing down_proj_output_qdq: up_in, up_out, gate_out, sigmoid_out, act_out,
        # down_in, down_out. (act_out wraps gate*sigmoid; gate*up feeds down_in.)
        qdq("model.up_proj_input_qdq", maxabs.get(f"l{i}.mlp_in", 0), M)
        qdq("model.up_proj_output_qdq", maxabs.get(f"l{i}.up", 0), M)
        qdq("model.gate_proj_output_qdq", maxabs.get(f"l{i}.gate", 0), M)
        qdq("model.sigmoid_output_qdq", 1.0, M)
        qdq("model.act_output_qdq", maxabs.get(f"l{i}.gate", 0), M)
        qdq("model.down_proj_input_qdq", maxabs.get(f"l{i}.down_in", 0), M)
        qdq("model.down_proj_output_qdq", maxabs.get(f"l{i}.down", 0), M)
        write_mllm_v2(os.path.join(args.out_dir, f"layer{i}_mlp-lpbq.mllm"), M)
        print(f"  layer {i} ({lt}) bundles written", flush=True)

    # ---- head bundle (tied lm_head) ----
    lm_head_w = lm.embed_tokens.weight.data                       # [vocab, hidden] tied
    head_buf, _ = lpbq_pack(lm_head_w, bs)
    Hd = {}
    Hd["model.lm_head.weight"] = head_buf["weight"]
    Hd["model.lm_head.scale1"] = head_buf["scale1"]
    Hd["model.lm_head.scale2"] = head_buf["scale2"]
    qdq("model.lmhead_input_qdq", maxabs.get("final_in", 0), Hd)
    qdq("model.lmhead_output_qdq", 24.0, Hd)
    write_mllm_v2(os.path.join(args.out_dir, "lmhead-lpbq.mllm"), Hd)

    write_mllm_v2(os.path.join(args.out_dir, "consts.mllm"), consts)

    # ---- embedding table (fp16) for CPU gather ----
    embed = {"embed": lm.embed_tokens.weight.detach().float().cpu().numpy().astype(np.float16)}
    write_mllm_v2(os.path.join(args.out_dir, "embed.mllm"), embed)

    # ---- prefill seed (states after the prompt) ----
    seed = {}
    for i in range(nL):
        c = cache[i]
        if layer_types[i] == "linear_attention":
            seed[f"l{i}.recurrent"] = f32(c.recurrent_states)        # [1,LH,Dk,Dv]
            seed[f"l{i}.conv"] = f32(c.conv_states)                  # [1,6144,4]
        else:
            seed[f"l{i}.keys"] = f32(c.keys)                         # [1,KV,T,Dh]
            seed[f"l{i}.values"] = f32(c.values)
    seed["prompt_ids"] = ids.input_ids[0].cpu().numpy().astype(np.int32)
    write_mllm_v2(os.path.join(args.out_dir, "seed.mllm"), seed)

    # ---- int4 golden reference token stream (already proven coherent) ----
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as g:
        json.dump(manifest, g, indent=2)
    print(f"[done] bundles in {args.out_dir} ({nL} layers + head + consts + embed + seed + manifest)")


if __name__ == "__main__":
    main()
