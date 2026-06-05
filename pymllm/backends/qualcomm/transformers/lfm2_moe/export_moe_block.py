#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Export a whole LFM2.5-8B-A1B MoE block (one decoder layer's feed_forward) for the
# host-dispatch design: 32 per-expert W4A16-LPBQ MLP bundles + the host router consts
# (gate.weight, expert_bias) + a block golden (x, selected ids/weights, block output y).
#
#   <out>/expert{e}-lpbq.mllm  : model.{gate,up,down}_proj.* + 7 *_qdq.* (compile each to a graph)
#   <out>/router.mllm          : gate_weight[E,hidden] fp32, expert_bias[E] fp32, meta
#   <out>/block-ref.mllm       : x[1,hidden] fp32, sel_ids[top_k] int32, sel_w[top_k] fp32,
#                                exp_y[1,hidden] fp32 (dequant-weight block output for the token)
#
#   python3 export_moe_block.py --layer 2 --out /path/wm_moe_l2
import argparse
import glob
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
_CODE = {np.dtype("float32"): 0, np.dtype("int8"): 16, np.dtype("int32"): 18,
         np.dtype("uint8"): 129, np.dtype("float16"): 1}


def write_mllm_v2(path, tensors, model_name="lfm2_moe_block"):
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
            descs.append(struct.pack(f"<IIQQQ{_SHAPE_LEN}i{_PNAME_LEN}s",
                                     pid, _CODE[arr.dtype], len(data), off, len(arr.shape), *shp, nb))
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
    ap = argparse.ArgumentParser(description="Export a full LFM2 MoE block (32 experts + router + golden)")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--out", default="/mnt/raid0_ssd/wentao/mllm/build-qnn-aot/bin/wm_moe_l2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bs = args.block_size

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").eval()
    model.config._attn_implementation = "eager"
    lm, cfg = model.model, model.config
    hidden, inter = cfg.hidden_size, cfg.moe_intermediate_size
    E, top_k = cfg.num_experts, cfg.num_experts_per_tok
    norm_topk = bool(cfg.norm_topk_prob)
    scaling = float(cfg.routed_scaling_factor)
    L = args.layer
    ff = lm.layers[L].feed_forward
    assert hasattr(ff, "experts"), f"layer {L} is not MoE"

    # ---- capture feed_forward input (ffn_norm output) for the last token ----
    captured = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) else kw.get("hidden_states")
        captured["x"] = hs.detach(); return None
    h = ff.register_forward_pre_hook(pre_hook, with_kwargs=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        block_out_hf = None
        def out_hook(mod, inp, out):
            nonlocal block_out_hf
            block_out_hf = (out[0] if isinstance(out, tuple) else out).detach()
        ho = ff.register_forward_hook(out_hook)
        model(ids, use_cache=False)
        ho.remove()
    h.remove()
    X = captured["x"][0].float()                # [T, hidden]
    t = X.shape[0] - 1                           # last token
    x = X[t:t + 1]                               # [1, hidden]

    # ---- router (LFM2: sigmoid -> +bias select -> gather sigmoid -> /(sum+1e-6) -> *scale) ----
    gate_w = ff.gate.weight.data.float()         # [E, hidden]
    expert_bias = ff.expert_bias.data.float()    # [E]
    logits = x @ gate_w.t()                       # [1, E]
    rw = torch.sigmoid(logits)[0]                 # [E]
    score = rw + expert_bias
    sel_ids = torch.topk(score, top_k).indices    # [top_k]
    sel_w = rw[sel_ids]
    if norm_topk:
        sel_w = sel_w / (sel_w.sum() + 1e-6)
    sel_w = sel_w * scaling
    print(f"[router] layer={L} token={t} sel_ids={sel_ids.tolist()} sel_w={[round(float(w),4) for w in sel_w]}",
          flush=True)

    # ---- per-expert LPBQ bundles + block golden (dequant-weight) ----
    def get_expert(e):
        if hasattr(ff.experts, "gate_up_proj"):
            gu = ff.experts.gate_up_proj.data[e].float()
            return gu[:inter, :].contiguous(), gu[inter:, :].contiguous(), ff.experts.down_proj.data[e].float().contiguous()
        ex = ff.experts[e]
        return ex.w1.weight.data.float(), ex.w3.weight.data.float(), ex.w2.weight.data.float()

    y_block = torch.zeros(1, hidden)
    for e in range(E):
        Wg, Wu, Wd = get_expert(e)
        gate_buf, Wg_dq = lpbq_pack(Wg, bs)
        up_buf, Wu_dq = lpbq_pack(Wu, bs)
        down_buf, Wd_dq = lpbq_pack(Wd, bs)
        gate = x @ Wg_dq.t(); up = x @ Wu_dq.t()
        act = F.silu(gate); inter_act = act * up
        ye = inter_act @ Wd_dq.t()               # [1, hidden] expert output
        # accumulate weighted output if this expert is selected
        hit = (sel_ids == e).nonzero(as_tuple=True)[0]
        if hit.numel():
            y_block = y_block + float(sel_w[int(hit[0])]) * ye

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
        qdq("model.down_proj_output_qdq", float(ye.abs().max()), W)
        write_mllm_v2(os.path.join(args.out, f"expert{e}-lpbq.mllm"), W)

    # sanity: dequant-weight block output vs HF fp32 block output for this token
    if block_out_hf is not None:
        hf_y = block_out_hf[0, t:t + 1].float()
        err = float((y_block - hf_y).abs().max())
        print(f"[golden] dequant-weight block y vs HF fp32 block: max|err|={err:.5f} "
              f"|y|max={float(hf_y.abs().max()):.4f}", flush=True)

    write_mllm_v2(os.path.join(args.out, "router.mllm"), {
        "gate_weight": gate_w.contiguous().numpy().astype(np.float32),
        "expert_bias": expert_bias.contiguous().numpy().astype(np.float32),
        "meta": np.array([L, E, top_k, hidden, inter, int(norm_topk)], dtype=np.int32),
        "scaling": np.array([scaling], dtype=np.float32),
    })
    write_mllm_v2(os.path.join(args.out, "block-ref.mllm"), {
        "x": x.contiguous().numpy().astype(np.float32).reshape(1, hidden),
        "sel_ids": sel_ids.contiguous().numpy().astype(np.int32),
        "sel_w": sel_w.contiguous().numpy().astype(np.float32),
        "exp_y": y_block.contiguous().numpy().astype(np.float32).reshape(1, hidden),
    })
    print(f"[done] layer={L}: {E} expert bundles + router + block-ref in {args.out}")


if __name__ == "__main__":
    main()
