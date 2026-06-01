"""Export ONE Qwen3.5 GatedDeltaNet layer's decode step as real-weight LPBQ +
a conv1d-free fp32 reference, for validating compile_deltanet_decode (full mode).

Pipeline:
  1. Load HF Qwen3.5-2B (fp32), hook layer L's linear_attn input to grab a real
     hidden-state x for one token.
  2. LPBQ-quantize (int4, block16) the deltanet projections in_proj_qkv / in_proj_z
     / out_proj (the same QLinearLPBQ used by quantize_weights.py), keep both the
     deploy buffers (int4 weight + uint4 scale1 + fp32 scale2) AND the dequantized
     fp32 weights (= fakequant forward) for the reference.
  3. Compute the conv1d-free fp32 reference with the *dequantized* weights, matching
     HF torch_recurrent_gated_delta_rule exactly (l2norm q/k, qscale=1/sqrt(Dk),
     S=S*gt; kv=k@S; delta=(v-kv)*beta; S+=k⊗delta; out=q@S), then gated RMSNorm·silu
     and out_proj. Uses a deterministic non-zero initial state S0 so every recurrence
     term is exercised. Also computes the FULL-PRECISION-weight reference to report
     the int4 weight-quant error.
  4. Calibrate 16-bit activation QDQ scales (scale = maxabs/32768, zp = 32768) from
     the reference activations — w4a16 means activations are ~lossless.
  5. Slice the fused qkv per-head and write:
       <out>-lpbq.mllm : per-head model.{q,k,v,z}_proj.<h>.{weight,scale1,scale2},
                         model.out_proj.{weight,scale1,scale2}, + all *_qdq.fake_quant.*
       <out>-ref.mllm  : x, gt, beta, eps, qscale, norm_w, S0, exp_y, exp_Sp

Usage:
  python -m pymllm.backends.qualcomm.transformers.qwen3_5.export_deltanet_decode \
      [--model_path <hf>] [--layer 0] [--block_size 16] [--out deltanet-l0] [--prompt ...]
"""

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


def write_mllm_v2(path, tensors, model_name="qwen3_5_deltanet"):
    """tensors: dict name -> np.ndarray (dtype in {float32,int8,int32,uint8})."""
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
    """Reconstruct fp32 weight EXACTLY as QNN does on-device from the deploy buffers:
      w[o,i] = signed_int4(weight[0,0,i,o]) * scale1[o, i//block] * scale2[o]
    scale1 is uint4-quantized, so this includes the two-level scale-quant error that
    the QLinearLPBQ fakequant (full-precision per-block scale) does NOT — the graph
    uses THIS, so the reference must too."""
    n_blk = in_features // block_size
    w = buf["weight"].reshape(in_features, out_features).astype(np.int32)  # HWIO [1,1,in,out]
    w = np.where(w >= 8, w - 16, w)                                        # masked 0x0F -> signed -7..7
    s1 = buf["scale1"].reshape(out_features, n_blk).astype(np.float32)     # uint4 level-1
    s2 = buf["scale2"].reshape(out_features).astype(np.float32)            # fp32 level-2
    block_scale = s1 * s2[:, None]                                         # [out, n_blk]
    # expand per-block scale to per-input-element, then dequant; return [out, in]
    bs_full = np.repeat(block_scale, block_size, axis=1)                   # [out, in]
    w_oi = w.T.astype(np.float32)                                          # [out, in]
    return torch.from_numpy(w_oi * bs_full).float()


@torch.no_grad()
def lpbq_pack(lin, block_size):
    """Return (deploy_buffers, w_dequant_fp32) for an nn.Linear via QLinearLPBQ.
    w_dequant is reconstructed from the DEPLOY buffers (scale1*scale2), matching the
    on-device QNN dequant — not the full-precision-scale fakequant."""
    q = QLinearLPBQ(lin.in_features, lin.out_features, bias=False, block_size=block_size)
    q.weight.data.copy_(lin.weight.data.float())
    q = q.to(torch.float32)
    q.freeze_weight()
    q.enable_fakequant()
    q.convert_to_conv2d_deploy_hwio()
    buf = {  # deploy: weight HWIO [1,1,in,out] int8, scale1 uint8 flat, scale2 fp32 [out]
        "weight": q.weight.cpu().numpy().astype(np.int8),
        "scale1": q.scale1.cpu().numpy().astype(np.uint8),
        "scale2": q.scale2.cpu().numpy().astype(np.float32),
    }
    w_dequant = dequant_from_deploy(buf, lin.in_features, lin.out_features, block_size)
    return buf, w_dequant


def slice_head(buf, c0, c1, n_blk):
    """Slice deploy buffers to out-channels [c0:c1]. weight is HWIO [1,1,in,Out]."""
    return {
        "weight": np.ascontiguousarray(buf["weight"][:, :, :, c0:c1]),
        "scale1": np.ascontiguousarray(buf["scale1"][c0 * n_blk:c1 * n_blk]),
        "scale2": np.ascontiguousarray(buf["scale2"][c0:c1]),
    }


def qdq(name, maxabs, store):
    s = float(maxabs) / 32768.0 if maxabs > 0 else 1.0 / 256.0
    store[name + ".fake_quant.scale"] = np.array([s], dtype=np.float32)
    store[name + ".fake_quant.zero_point"] = np.array([32768], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description="Export one Qwen3.5 deltanet decode step: LPBQ weights + conv1d-free ref")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is Paris, and the capital of Japan is")
    ap.add_argument("--conv1d", action="store_true", default=True, help="include depthwise causal conv1d (kernel 4) + silu on q/k/v")
    ap.add_argument("--no-conv1d", dest="conv1d", action="store_false")
    ap.add_argument("--out", default="deltanet-l0")
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
    eps = float(attn.layer_norm_epsilon)  # HF rms_norm_eps (1e-6); graph keeps it via scaling
    print(f"[cfg] layer={args.layer} H={H} Dk={Dk} Dv={Dv} hidden={hidden} key_dim={key_dim} value_dim={value_dim}")

    # ---- 1) hook layer L linear_attn input to grab a real x (last token) ----
    captured = {}
    def pre_hook(mod, inp, kw):
        hs = inp[0] if len(inp) > 0 else kw.get("hidden_states")
        captured["x"] = hs.detach()
        return None
    def post_hook(mod, inp, out):
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach()
        return None
    h = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = attn.register_forward_hook(post_hook)
    ids = tok(args.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        model(**ids)
    h.remove(); h2.remove()
    hf_out_last = captured["out"][0, -1].float().cpu()       # HF deltanet output, last token
    X_all = captured["x"][0].float().cpu()                   # [T, hidden] all prompt positions
    x = X_all[-1:].reshape(1, hidden)                        # last token = the decode step
    print(f"[hook] captured X {tuple(X_all.shape)} (T={X_all.shape[0]}); last-token x range [{x.min():.3f}, {x.max():.3f}]")

    # ---- 2) LPBQ pack the three projections (deploy + dequant) ----
    qkv_buf, Wqkv_dq = lpbq_pack(attn.in_proj_qkv, args.block_size)
    z_buf, Wz_dq = lpbq_pack(attn.in_proj_z, args.block_size)
    out_buf, Wout_dq = lpbq_pack(attn.out_proj, args.block_size)
    n_blk = hidden // args.block_size           # blocks along contraction (2048/16=128) for qkv/z
    n_blk_out = value_dim // args.block_size    # out_proj contracts over value_dim

    # ---- 3) conv1d-free reference (fp32), with dequant-int4 AND full-precision weights ----
    A_log = attn.A_log.detach().float().cpu()
    dt_bias = attn.dt_bias.detach().float().cpu()
    norm_w = attn.norm.weight.detach().float().cpu()       # [Dv]
    Wa = attn.in_proj_a.weight.detach().float().cpu()      # [16, hidden]
    Wb = attn.in_proj_b.weight.detach().float().cpu()
    xc = x.float().cpu()
    qscale = 1.0 / (Dk ** 0.5)

    # per-token gates over the whole prompt (from the unquantized tiny a/b projections)
    a_all = X_all @ Wa.t()                                   # [T,16]
    b_all = X_all @ Wb.t()
    gt_all = torch.exp(-torch.exp(A_log) * F.softplus(a_all + dt_bias))  # [T,H]
    beta_all = torch.sigmoid(b_all)                                      # [T,H]
    conv_dim = 2 * key_dim + value_dim
    cw_full = attn.conv1d.weight.detach().float().cpu().reshape(conv_dim, -1)   # [6144,4]
    T = X_all.shape[0]
    last = T - 1
    print(f"[cfg] conv1d={args.conv1d} conv_dim={conv_dim} kernel={cw_full.shape[1]} bias={attn.conv1d.bias is not None} T={T}")

    # REAL decode validation: run the full prefill recurrence over tokens 0..T-2 to get
    # the accumulated state S0 entering the last token, then the graph computes token T-1.
    # This gives a well-conditioned out=q@S0 (unlike a random S0, which the scale-invariant
    # gated RMSNorm would turn into amplified noise).
    def prefill_and_step(Wqkv, Wz, Wout, use_conv=None):
        if use_conv is None:
            use_conv = args.conv1d
        mixed_pre = X_all @ Wqkv.t()                  # [T,6144] raw projections (pre-conv)
        if use_conv:
            # causal depthwise conv1d + silu over the sequence (left-pad 3 with zeros)
            pad = torch.zeros(3, conv_dim)
            mp = torch.cat([pad, mixed_pre], dim=0)   # [T+3,6144]
            win = torch.stack([mp[i:i + T] for i in range(4)], dim=-1)  # [T,6144,4] = taps
            co = (win * cw_full).sum(-1)              # [T,6144]
            mixed = co * torch.sigmoid(co)
        else:
            mixed = mixed_pre
        q = mixed[:, :key_dim].reshape(T, H, Dk)
        k = mixed[:, key_dim:2 * key_dim].reshape(T, H, Dk)
        v = mixed[:, 2 * key_dim:].reshape(T, H, Dv)
        z = (X_all @ Wz.t()).reshape(T, H, Dv)
        qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * qscale
        kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        S = torch.zeros(H, Dk, Dv)
        S0_snapshot = None
        gated = torch.zeros(H, Dv)
        Sp = torch.zeros(H, Dk, Dv)
        for t in range(T):
            if t == last:
                S0_snapshot = S.clone()               # state ENTERING the last token
            for hh in range(H):
                S[hh] = S[hh] * gt_all[t, hh]
                kv = kn[t, hh] @ S[hh]
                delta = (v[t, hh] - kv) * beta_all[t, hh]
                S[hh] = S[hh] + torch.outer(kn[t, hh], delta)
                if t == last:
                    out = qn[t, hh] @ S[hh]
                    Sp[hh] = S[hh]
                    normed = out * torch.rsqrt((out * out).mean(-1, keepdim=True) + eps) * norm_w
                    gated[hh] = normed * (z[t, hh] * torch.sigmoid(z[t, hh]))
        y = (gated.reshape(1, value_dim) @ Wout.t()).reshape(hidden)
        # last-token pre-conv projections (for activation-scale calibration), conv state
        # (graph input = pre-conv window of the 3 prior tokens), gates, S0 entering last token.
        q_pre = mixed_pre[last, :key_dim].reshape(H, Dk)
        k_pre = mixed_pre[last, key_dim:2 * key_dim].reshape(H, Dk)
        v_pre = mixed_pre[last, 2 * key_dim:].reshape(H, Dv)
        cs0 = mixed_pre[last - 3:last].t().contiguous() if use_conv else None  # [6144,3] pre-conv window
        return dict(q=q_pre, k=k_pre, v=v_pre, z=z[last], gated=gated, Sp=Sp, y=y,
                    S0=S0_snapshot, cs0=cs0, gt=gt_all[last], beta=beta_all[last])

    rq = prefill_and_step(Wqkv_dq, Wz_dq, Wout_dq)    # int4-weight reference (matches graph)
    rf = prefill_and_step(attn.in_proj_qkv.weight.detach().float().cpu(),
                          attn.in_proj_z.weight.detach().float().cpu(),
                          attn.out_proj.weight.detach().float().cpu())  # fp32-weight reference
    S0 = rq["S0"]
    gt = rq["gt"]
    beta = rq["beta"]
    cs0 = rq["cs0"]
    werr = (rq["y"] - rf["y"]).abs().max().item()
    print(f"[ref] int4-weight vs fp32-weight  max|y err| = {werr:.5f}  "
          f"(y range [{rf['y'].min():.3f},{rf['y'].max():.3f}])")
    hferr = (rf["y"] - hf_out_last).abs().max().item()
    print(f"[ref] my fp32 reference vs ACTUAL HF deltanet output: max|y err| = {hferr:.5f}  "
          f"(HF range [{hf_out_last.min():.3f},{hf_out_last.max():.3f}]) -> {'conv OK' if hferr < 0.05 else 'CONV MISMATCH'}")

    # ---- 4) calibrate 16-bit activation QDQ scales from the int4-weight reference ----
    qdq_store = {}
    qdq("model.qkv_input_qdq", xc.abs().max().item(), qdq_store)
    for hh in range(H):
        qdq(f"model.q_out_qdq_h{hh}", rq["q"][hh].abs().max().item(), qdq_store)
        qdq(f"model.k_out_qdq_h{hh}", rq["k"][hh].abs().max().item(), qdq_store)
        qdq(f"model.v_out_qdq_h{hh}", rq["v"][hh].abs().max().item(), qdq_store)
        qdq(f"model.z_out_qdq_h{hh}", rq["z"][hh].abs().max().item(), qdq_store)
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
    write_mllm_v2(f"{args.out}-lpbq.mllm", W)

    # ---- 5b) write reference I/O .mllm (fp32; C++ runner converts to fp16) ----
    R = {
        "x": xc.numpy().astype(np.float32).reshape(1, hidden),
        "gt": gt.numpy().astype(np.float32).reshape(H, 1, 1),
        "beta": beta.numpy().astype(np.float32).reshape(H, 1, 1),
        "eps": np.array([eps], dtype=np.float32).reshape(1, 1, 1),
        "qscale": np.array([qscale], dtype=np.float32).reshape(1, 1, 1),
        "norm_w": norm_w.numpy().astype(np.float32).reshape(1, 1, Dv),
        "S0": S0.numpy().astype(np.float32).reshape(H, Dk, Dv),
        "exp_y": rq["y"].numpy().astype(np.float32).reshape(1, hidden),
        "exp_Sp": rq["Sp"].numpy().astype(np.float32).reshape(H, Dk, Dv),
    }
    if args.conv1d:
        # graph layout: cw_* [1,4,width] (tap-major), cs_* [1,3,width] (time-major)
        def cwslab(c0, c1):  # [width,4] -> [1,4,width]
            return cw_full[c0:c1].t().contiguous().numpy().astype(np.float32)[None]
        def csslab(c0, c1):  # [width,3] -> [1,3,width]
            return cs0[c0:c1].t().contiguous().numpy().astype(np.float32)[None]
        R["cw_q"] = cwslab(0, key_dim)
        R["cw_k"] = cwslab(key_dim, 2 * key_dim)
        R["cw_v"] = cwslab(2 * key_dim, conv_dim)
        R["cs_q"] = csslab(0, key_dim)
        R["cs_k"] = csslab(key_dim, 2 * key_dim)
        R["cs_v"] = csslab(2 * key_dim, conv_dim)
    write_mllm_v2(f"{args.out}-ref.mllm", R)
    print(f"[done] H={H} Dk={Dk} Dv={Dv}. Compile: --params {args.out}-lpbq.mllm ; validate: --ref {args.out}-ref.mllm")


if __name__ == "__main__":
    main()
