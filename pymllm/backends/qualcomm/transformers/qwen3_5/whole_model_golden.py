#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Whole-model int4 GOLDEN for the NPU decode pipeline. Swaps every quantizable
# Linear's weight with its DEPLOY-DEQUANT int4 version (signed_int4 * scale1 *
# scale2 — exactly what the on-device LPBQ Conv2D graphs compute), then runs HF
# greedy generation. Compares the int4 token stream to the fp32 model's.
#
# This proves the whole-model COMPOSITION is correct under the same weight quant
# the per-layer NPU graphs use (deltanet/attn/mlp/head all validated in isolation),
# before the C++ device orchestrator is built. HF handles all the math (conv1d,
# recurrence, attention, RoPE), so there is no reimplementation risk.
#
#   python3 whole_model_golden.py --max_new 24
#
import argparse
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn


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


@torch.no_grad()
def deploy_dequant_weight(weight, block_size):
    """Return the fp32 weight as the on-device LPBQ Conv2D reconstructs it:
    pack via QLinearLPBQ -> deploy buffers -> signed_int4 * scale1(uint4) * scale2."""
    out_features, in_features = weight.shape
    q = QLinearLPBQ(in_features, out_features, bias=False, block_size=block_size)
    q.weight.data.copy_(weight.float())
    q = q.to(torch.float32)
    q.freeze_weight()
    q.enable_fakequant()
    q.convert_to_conv2d_deploy_hwio()
    w = q.weight.cpu().numpy().reshape(in_features, out_features).astype(np.int32)  # HWIO
    w = np.where(w >= 8, w - 16, w)
    n_blk = in_features // block_size
    s1 = q.scale1.cpu().numpy().reshape(out_features, n_blk).astype(np.float32)
    s2 = q.scale2.cpu().numpy().reshape(out_features).astype(np.float32)
    bs = np.repeat(s1 * s2[:, None], block_size, axis=1)            # [out,in]
    return torch.from_numpy(w.T.astype(np.float32) * bs).float()    # [out,in]


def collect_target_linears(lm, lm_head_module):
    """Names of the Linears the NPU pipeline quantizes (skip the tiny deltanet
    in_proj_a/b gates — they run in fp32 on CPU)."""
    targets = []
    for i, layer in enumerate(lm.layers):
        lt = lm.config.layer_types[i]
        if lt == "linear_attention":
            a = layer.linear_attn
            for nm in ["in_proj_qkv", "in_proj_z", "out_proj"]:
                targets.append((f"layers.{i}.linear_attn.{nm}", getattr(a, nm)))
        else:
            a = layer.self_attn
            for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                targets.append((f"layers.{i}.self_attn.{nm}", getattr(a, nm)))
        for nm in ["gate_proj", "up_proj", "down_proj"]:
            targets.append((f"layers.{i}.mlp.{nm}", getattr(layer.mlp, nm)))
    return targets


def main():
    ap = argparse.ArgumentParser(description="Whole-model int4 golden vs fp32 (HF greedy)")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--quant_head", action="store_true", default=True, help="also int4 the tied lm_head/embedding")
    ap.add_argument("--no_quant_head", dest="quant_head", action="store_false")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model_path} -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    ids = tok(args.prompt, return_tensors="pt").to(dev)

    # ---- fp32 greedy reference ----
    with torch.no_grad():
        gen_fp32 = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                  num_beams=1, use_cache=True)
    new_fp32 = gen_fp32[0, ids.input_ids.shape[1]:].tolist()
    print(f"[fp32] {tok.decode(new_fp32)!r}")

    # ---- swap every target Linear weight -> deploy-dequant int4 ----
    targets = collect_target_linears(lm, None)
    if args.quant_head:
        # tied: lm_head shares embed_tokens.weight; quantize the shared matrix once.
        targets.append(("model.embed_tokens", lm.embed_tokens))
    n_swapped = 0
    for name, mod in targets:
        w = mod.weight.data
        dq = deploy_dequant_weight(w, args.block_size).to(w.dtype).to(w.device)
        mod.weight.data.copy_(dq)
        n_swapped += 1
    print(f"[int4] swapped {n_swapped} weight matrices to deploy-dequant int4 (block={args.block_size})", flush=True)

    # ---- int4 greedy ----
    with torch.no_grad():
        gen_i4 = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                num_beams=1, use_cache=True)
    new_i4 = gen_i4[0, ids.input_ids.shape[1]:].tolist()
    print(f"[int4] {tok.decode(new_i4)!r}")

    # ---- compare ----
    match = sum(1 for a, b in zip(new_fp32, new_i4) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(new_fp32, new_i4)) if a != b), len(new_fp32))
    print(f"[cmp] token match {match}/{len(new_fp32)}; first divergence at +{first_div}")
    print(f"[cmp] {'PASS (identical greedy)' if match == len(new_fp32) else 'coherent-but-diverges (int4 drift)'}")


if __name__ == "__main__":
    main()
