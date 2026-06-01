"""Weight-only LPBQ (int4, two-scale block) PTQ for Qwen3.5 — quality study.

Unlike the Qwen3 QNN flow (pymllm/.../qwen3/{modeling,runner,train}.py), this does
NOT instrument activations or build a QNN graph. It only quantizes the *weights* of
every Linear in the text backbone to int4 LPBQ (the same `QLinearLPBQ` core), then
measures how much that hurts quality (perplexity + a sample generation) vs fp32.

Weight LPBQ needs no calibration data: `QLinearLPBQ.freeze_weight()` derives the
per-block scales from the weight tensor itself. Activations stay fp32 here.

Usage:
  python -m pymllm.backends.qualcomm.transformers.qwen3_5.quantize_weights \
      [--model_path <hf dir>] [--block_size 16] [--include_lm_head] \
      [--eval_tokens 8192] [--save_dir <dir>]
"""

import argparse
import glob
import os
import sys
import types

import torch
import torch.nn as nn


def _import_qlinear_without_pymllm_init():
    """Import QLinearLPBQ without executing `pymllm/__init__.py` (which loads a
    native FFI .so that isn't built in this env). We pre-stub the parent packages
    in sys.modules so only the leaf modules (observer.py, qlinear.py) execute."""
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
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    from pymllm.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ
    return QLinearLPBQ


QLinearLPBQ = _import_qlinear_without_pymllm_init()


def _default_model_path():
    hub = "/mnt/raid0_ssd/wentao/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots"
    hits = glob.glob(os.path.join(hub, "*", "config.json"))
    return os.path.dirname(hits[0]) if hits else "Qwen/Qwen3.5-2B"


@torch.no_grad()
def swap_linear_to_lpbq(model, block_size=16, include_lm_head=True,
                        skip_prefixes=("model.visual",)):
    """Replace every nn.Linear under the text backbone with a frozen int4 LPBQ linear.

    Returns (n_swapped, n_params_quantized, skipped_names). Skips:
      * anything under `skip_prefixes` (the vision tower),
      * Linears whose in_features is not a multiple of `block_size` (LPBQ needs it),
      * lm_head unless `include_lm_head`.
    """
    n_swapped, n_params, skipped = 0, 0, []
    targets = []  # (parent_module, attr_name, full_name, linear)
    for name, module in model.named_modules():
        for attr, child in list(module.__dict__.get("_modules", {}).items()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{attr}" if name else attr
            if any(full.startswith(p) for p in skip_prefixes):
                skipped.append((full, "vision"))
                continue
            if (not include_lm_head) and full.endswith("lm_head"):
                skipped.append((full, "lm_head excluded"))
                continue
            if child.in_features % block_size != 0:
                skipped.append((full, f"in_features {child.in_features} % {block_size} != 0"))
                continue
            targets.append((module, attr, full, child))

    for parent, attr, full, lin in targets:
        q = QLinearLPBQ(lin.in_features, lin.out_features,
                        bias=lin.bias is not None, block_size=block_size)
        q.weight.data.copy_(lin.weight.data.float())
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.data.float())
        q = q.to(lin.weight.device, dtype=torch.float32)
        q.freeze_weight()          # derive int4 block scales from the weights
        q.enable_fakequant()       # forward() now returns fake-quantized weights
        setattr(parent, attr, q)
        n_swapped += 1
        n_params += lin.weight.numel()

    return n_swapped, n_params, skipped


@torch.no_grad()
def perplexity(model, input_ids, max_len=2048, stride=2048, device="cuda"):
    """Standard concatenated-corpus perplexity over one long token stream."""
    nlls, n_tok = [], 0
    seq_len = input_ids.size(1)
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_len, seq_len)
        ids = input_ids[:, begin:end].to(device)
        if ids.size(1) < 2:
            break
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="sum")
        nlls.append(loss)
        n_tok += shift_labels.numel()
    return torch.exp(torch.stack(nlls).sum() / n_tok).item(), n_tok


def load_eval_text(tokenizer, eval_tokens):
    """Wikitext-2 test split, falling back to a built-in passage if offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
    except Exception as e:  # noqa: BLE001
        print(f"[eval] datasets unavailable ({e}); using built-in passage")
        text = (_BUILTIN_PASSAGE + "\n\n") * 64
    ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, : max(eval_tokens, 8)]


@torch.no_grad()
def sample_generate(model, tokenizer, prompt, max_new_tokens=128, device="cuda"):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer([text], return_tensors="pt").to(device)
    gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    return tokenizer.decode(gen[0][ids.input_ids.size(1):], skip_special_tokens=True).strip()


_BUILTIN_PASSAGE = (
    "The history of science is the study of the development of science, including both "
    "the natural and social sciences. Science is a body of empirical, theoretical, and "
    "practical knowledge about the natural world, produced by scientists who emphasize "
    "the observation, explanation, and prediction of real-world phenomena."
)


def main():
    ap = argparse.ArgumentParser(description="Weight-only LPBQ int4 PTQ quality study for Qwen3.5")
    ap.add_argument("--model_path", default=_default_model_path())
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--include_lm_head", action="store_true")
    ap.add_argument("--eval_tokens", type=int, default=8192)
    ap.add_argument("--prompt", default="Explain why the sky is blue in two sentences.")
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForImageTextToText

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    print(f"[load] {args.model_path}  (fp32 -> {device})")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="eager").to(device).eval()

    eval_ids = load_eval_text(tok, args.eval_tokens)
    print(f"[eval] {eval_ids.size(1)} eval tokens")

    ppl_fp32, ntok = perplexity(model, eval_ids, device=device)
    gen_fp32 = sample_generate(model, tok, args.prompt, device=device)
    print(f"\n[fp32] perplexity = {ppl_fp32:.4f}  over {ntok} tokens")
    print(f"[fp32] sample: {gen_fp32}\n")

    n, nparams, skipped = swap_linear_to_lpbq(
        model, block_size=args.block_size, include_lm_head=args.include_lm_head)
    print(f"[quant] swapped {n} Linears to int4 LPBQ (block_size={args.block_size}), "
          f"{nparams/1e6:.1f}M weights quantized")
    if skipped:
        from collections import Counter
        print("[quant] skipped:", dict(Counter(r for _, r in skipped)))

    ppl_q, _ = perplexity(model, eval_ids, device=device)
    gen_q = sample_generate(model, tok, args.prompt, device=device)
    print(f"\n[int4] perplexity = {ppl_q:.4f}  (fp32 {ppl_fp32:.4f}, "
          f"+{100*(ppl_q-ppl_fp32)/ppl_fp32:.2f}%)")
    print(f"[int4] sample: {gen_q}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        from safetensors.torch import save_model
        for m in model.modules():
            if isinstance(m, QLinearLPBQ):
                m.convert_to_conv2d_deploy_hwio()
        save_model(model, os.path.join(args.save_dir, "model.safetensors"))
        print(f"[save] int4 LPBQ weights -> {args.save_dir}/model.safetensors")


if __name__ == "__main__":
    main()
