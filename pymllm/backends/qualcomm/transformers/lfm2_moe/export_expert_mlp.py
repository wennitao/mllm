#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export ONE LFM2.5-8B-A1B MoE expert's SwiGLU MLP as a W4A16-LPBQ bundle for the
# QNN AOT MLP-decode compiler, plus a fp32 (dequant-weight) reference for on-device
# validation. This is the per-expert unit of the MoE-on-HTP block (host-dispatch design).
#
# An LFM2 expert is `w2(silu(w1(x)) * w3(x))` — structurally identical to the existing
# Qwen3MLP (gate=w1, up=w3, down=w2). So we emit the bundle under the gate/up/down keys
# the `mllm-qwen3-aot-mlp-decode-c` compiler expects (inter = moe_intermediate_size=1792).
#
# Because experts are conditionally routed, we calibrate the expert on a REAL activation:
# we run a prompt, capture the chosen MoE layer's ffn_norm output (the expert input), and
# pick a token that the router actually sends to the chosen expert.
#
#   python3 export_expert_mlp.py --layer 2 --expert <auto|N> --out /path/expert
#     -> <out>-lpbq.mllm  (model.{gate,up,down}_proj.{weight,scale1,scale2} + 7 *_qdq.fake_quant.*)
#        <out>-ref.mllm   (x[1,hidden], exp_y[1,hidden], meta)
import argparse
import glob
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

# ---- mllm V2 writer (matches export_whole_model.py) ----
_MAGIC, _VERSION = 0x519A, 2
_NAME_LEN, _PNAME_LEN, _SHAPE_LEN = 512, 256, 16
_HDR_SIZE, _DESC_SIZE = 532, 352
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18,
         np.dtype("uint8"): 129, np.dtype("float16"): 1}


def write_mllm_v2(path, tensors, model_name="lfm2_expert"):
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


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def _default_model_path():
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "LiquidAI/LFM2.5-8B-A1B"


def main():
    ap = argparse.ArgumentParser(description="Export one LFM2 MoE expert SwiGLU MLP (W4A16 LPBQ) + ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=2, help="decoder layer index (must be a MoE layer >= num_dense_layers)")
    ap.add_argument("--expert", default="auto", help="expert id, or 'auto' = most-routed expert for the prompt")
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out", default="/mnt/raid0_ssd/wentao/mllm/build-qnn-aot/bin/lfm2_expert")
    args = ap.parse_args()
    bs = args.block_size

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").eval()
    model.config._attn_implementation = "eager"
    lm = model.model
    cfg = model.config
    hidden = cfg.hidden_size
    inter = cfg.moe_intermediate_size
    L = args.layer
    layer = lm.layers[L]
    ff = layer.feed_forward
    assert hasattr(ff, "experts"), f"layer {L} is not a MoE layer (no experts)"

    # ---- capture the expert input = ffn_norm output (feed_forward input) ----
    captured = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) else kw.get("hidden_states")
        captured["x"] = hs.detach()
        return None
    h = ff.register_forward_pre_hook(pre_hook, with_kwargs=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        model(ids, use_cache=False)
    h.remove()
    X = captured["x"][0].float()                       # [T, hidden] feed_forward inputs
    T = X.shape[0]

    # ---- router (sigmoid + expert_bias select) to find which expert each token uses ----
    gate_w = ff.gate.weight.data.float()               # [num_experts, hidden]
    expert_bias = ff.expert_bias.data.float()          # [num_experts]
    top_k = cfg.num_experts_per_tok
    logits = X @ gate_w.t()                             # [T, E]
    rw = torch.sigmoid(logits)
    score = rw + expert_bias
    sel = torch.topk(score, top_k, dim=-1).indices     # [T, top_k]

    if args.expert == "auto":
        counts = torch.bincount(sel.reshape(-1), minlength=gate_w.shape[0])
        e = int(counts.argmax())
    else:
        e = int(args.expert)
    # pick a token routed to expert e (fallback: token 0)
    rows = (sel == e).any(dim=-1).nonzero(as_tuple=True)[0]
    t = int(rows[0]) if rows.numel() else 0
    x = X[t:t + 1]                                     # [1, hidden]
    print(f"[pick] layer={L} expert={e} token={t}/{T} (routed={'yes' if rows.numel() else 'NO-fallback'}) "
          f"|x|max={float(x.abs().max()):.4f}", flush=True)

    # transformers stores experts FUSED: experts.gate_up_proj [E, 2*inter, hidden]
    # (rows 0:inter = gate=w1, inter:2*inter = up=w3) and down_proj [E, hidden, inter] = w2.
    experts = ff.experts
    if hasattr(experts, "gate_up_proj"):
        gate_up = experts.gate_up_proj.data[e].float()   # [2*inter, hidden]
        Wg = gate_up[:inter, :].contiguous()             # gate (w1) [inter, hidden]
        Wu = gate_up[inter:, :].contiguous()             # up   (w3) [inter, hidden]
        Wd = experts.down_proj.data[e].float().contiguous()  # down (w2) [hidden, inter]
    else:  # per-expert ModuleList fallback
        ex = experts[e]
        Wg = ex.w1.weight.data.float()
        Wu = ex.w3.weight.data.float()
        Wd = ex.w2.weight.data.float()

    gate_buf, Wg_dq = lpbq_pack(Wg, bs)
    up_buf, Wu_dq = lpbq_pack(Wu, bs)
    down_buf, Wd_dq = lpbq_pack(Wd, bs)

    # ---- reference forward with DEQUANT (device-equivalent int4) weights, fp32 acts ----
    gate = x @ Wg_dq.t()                  # [1, inter]
    up = x @ Wu_dq.t()                    # [1, inter]
    act = F.silu(gate)                    # silu(gate)
    inter_act = act * up                  # [1, inter]
    y = inter_act @ Wd_dq.t()             # [1, hidden]

    # full-precision-weight reference (sanity: int4 weight error)
    gate_fp = x @ Wg.t(); up_fp = x @ Wu.t()
    y_fp = (F.silu(gate_fp) * up_fp) @ Wd.t()
    werr = float((y - y_fp).abs().max())
    print(f"[ref] int4-weight vs fp32-weight  max|y err|={werr:.5f}  |y|max={float(y.abs().max()):.4f}", flush=True)

    # ---- bundle (compiler) : gate/up/down LPBQ + 7 QDQ scales ----
    W = {}
    for nm, b in [("gate_proj", gate_buf), ("up_proj", up_buf), ("down_proj", down_buf)]:
        W[f"model.{nm}.weight"] = b["weight"]
        W[f"model.{nm}.scale1"] = b["scale1"]
        W[f"model.{nm}.scale2"] = b["scale2"]
    qdq("model.up_proj_input_qdq", float(x.abs().max()), W)
    qdq("model.up_proj_output_qdq", float(up.abs().max()), W)
    qdq("model.gate_proj_output_qdq", float(gate.abs().max()), W)
    qdq("model.sigmoid_output_qdq", 1.0, W)
    qdq("model.act_output_qdq", float(act.abs().max()), W)
    qdq("model.down_proj_input_qdq", float(inter_act.abs().max()), W)
    qdq("model.down_proj_output_qdq", float(y.abs().max()), W)
    write_mllm_v2(f"{args.out}-lpbq.mllm", W)

    # ---- reference (runner validation) : x, exp_y ----
    R = {
        "x": x.contiguous().numpy().astype(np.float32).reshape(1, hidden),
        "exp_y": y.contiguous().numpy().astype(np.float32).reshape(1, hidden),
        "meta": np.array([L, e, t, hidden, inter], dtype=np.int32),
    }
    write_mllm_v2(f"{args.out}-ref.mllm", R)
    print(f"[done] layer={L} expert={e} hidden={hidden} inter={inter}. "
          f"Compile: --params {args.out}-lpbq.mllm --hidden {hidden} --inter {inter} ; "
          f"validate: --ref {args.out}-ref.mllm")


if __name__ == "__main__":
    main()
