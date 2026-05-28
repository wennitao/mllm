# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Measured Qwen3-1.7B FULL DECODER BLOCK latency model — extends the MLP-only
# spec (qwen3_1p7b_mlp.py) to cover every op in a decoder layer except the
# group-query attention core (Q·K^T → mask → softmax → @V), which is excluded
# because it runs on a separate sparse-attention path with sparsity-dependent
# cost.
#
# Scope:
#   - INCLUDED: input_layernorm, q/k/v_proj, q/k_norm, q/k RoPE, o_proj,
#     residual #1, post_attention_layernorm, gate+up/silu/gate*up/down (MLP),
#     residual #2.
#   - EXCLUDED: GQA scaled-dot-product attention (Q·K^T, mask, softmax, A·V).
#     Modeled as either a fixed cost per tile (tile-pipeline mode) or a single
#     barrier between qkv/rope and o_proj (barrier mode).
#
# All numbers collected on SM8750 (Adreno 830 + Hexagon V79) under QAIRT
# 2.43.0.260128, w4a16 LPBQ. Geometry: hidden=2048, intermediate=6144,
# head_dim=128, Hq=16, Hkv=8.
#
# Refresh procedure: see tools/het_sim/README.md "Refreshing the block-level
# measurements" section.

SPEC = {
    "shape": {
        "H": 2048,
        "I": 6144,
        "head_dim": 128,
        "n_q_heads": 16,
        "n_kv_heads": 8,
    },

    # --------------------------------------------------------------------
    # DAG. Each op declares deps (tensor names it reads), outputs (tensor
    # names it produces), engines (where it can run), and kind (latency
    # category — used by het_sim.latency_block_ms to pick the right model).
    # GQA is a virtual op with cost = --gqa-ms-per-tile (default 0); see
    # het_sim.py for scheduling modes (tile-pipeline vs. barrier).
    # --------------------------------------------------------------------
    "ops": {
        "in_norm":    {"deps": ["_in"],                "outputs": ["xn"],         "engines": ["NPU", "CPU", "GPU"], "kind": "norm",   "width": 2048},
        "q_proj":     {"deps": ["xn"],                 "outputs": ["q"],          "engines": ["NPU"],               "kind": "matmul"},
        "k_proj":     {"deps": ["xn"],                 "outputs": ["k"],          "engines": ["NPU"],               "kind": "matmul"},
        "v_proj":     {"deps": ["xn"],                 "outputs": ["v"],          "engines": ["NPU"],               "kind": "matmul"},
        "q_norm":     {"deps": ["q"],                  "outputs": ["qn"],         "engines": ["NPU", "CPU", "GPU"], "kind": "norm",   "width": 128},
        "k_norm":     {"deps": ["k"],                  "outputs": ["kn"],         "engines": ["NPU", "CPU", "GPU"], "kind": "norm",   "width": 128},
        "q_rope":     {"deps": ["qn"],                 "outputs": ["qr"],         "engines": ["NPU", "CPU", "GPU"], "kind": "rope",   "heads": 16},
        "k_rope":     {"deps": ["kn"],                 "outputs": ["kr"],         "engines": ["NPU", "CPU", "GPU"], "kind": "rope",   "heads": 8},
        "gqa":        {"deps": ["qr", "kr", "v"],      "outputs": ["_attn_out"],  "engines": ["NPU"],               "kind": "gqa"},
        "o_proj":     {"deps": ["_attn_out"],          "outputs": ["o"],          "engines": ["NPU"],               "kind": "matmul"},
        "res1":       {"deps": ["o", "_in"],           "outputs": ["x1"],         "engines": ["NPU", "CPU", "GPU"], "kind": "add",    "width": 2048},
        "post_norm":  {"deps": ["x1"],                 "outputs": ["x1n"],        "engines": ["NPU", "CPU", "GPU"], "kind": "norm",   "width": 2048},
        "gr":         {"deps": ["x1n"],                "outputs": ["g", "u"],     "engines": ["NPU"],               "kind": "matmul"},
        "silu":       {"deps": ["g"],                  "outputs": ["sig_g"],      "engines": ["NPU", "CPU", "GPU"], "kind": "act"},
        "gateup_mul": {"deps": ["sig_g", "u"],         "outputs": ["h"],          "engines": ["NPU", "CPU", "GPU"], "kind": "act"},
        "dn":         {"deps": ["h"],                  "outputs": ["mlp_out"],    "engines": ["NPU"],               "kind": "matmul"},
        "res2":       {"deps": ["mlp_out", "x1"],      "outputs": ["_out"],       "engines": ["NPU", "CPU", "GPU"], "kind": "add",    "width": 2048},
    },
    "op_order": [
        "in_norm",
        "q_proj", "k_proj", "v_proj",
        "q_norm", "k_norm",
        "q_rope", "k_rope",
        "gqa",
        "o_proj",
        "res1", "post_norm",
        "gr", "silu", "gateup_mul", "dn",
        "res2",
    ],

    # Output sizes (bytes per token, fp16).
    "output_bytes_per_token": {
        "_in":      2048 * 2,
        "xn":       2048 * 2,
        "q":        2048 * 2,    # 16 * 128
        "k":        1024 * 2,    # 8  * 128
        "v":        1024 * 2,
        "qn":       2048 * 2,
        "kn":       1024 * 2,
        "qr":       2048 * 2,
        "kr":       1024 * 2,
        "_attn_out": 2048 * 2,
        "o":        2048 * 2,
        "x1":       2048 * 2,
        "x1n":      2048 * 2,
        "g":        6144 * 2,
        "u":        6144 * 2,
        "sig_g":    6144 * 2,
        "h":        6144 * 2,
        "mlp_out":  2048 * 2,
        "_out":     2048 * 2,
    },

    # --------------------------------------------------------------------
    # NPU LPBQ matmul latency (ms) at compiled tile width Sq.
    #
    # MLP gr/dn: preserved from qwen3_1p7b_mlp.py (mllm-het-mlp --npu-only,
    # LPBQ `gatedown` bins).
    #
    # q/k/v/o projection: measured this session via
    #   mllm-qwen3-aot-attn-lpbq-microbench-c --mode {q,kv,o} --sq <Sq>  (compile)
    #   mllm-qwen3-aot-attn-lpbq-bench --mode {q,kv,o} --sq <Sq>          (host wall)
    # Synthetic LPBQ-w4a16 weights + uint16 QDQ scaffolding; matches the
    # production AOT recipe (qnn_aot_cfg_attn_lpbq_microbench.json).
    # --------------------------------------------------------------------
    "npu_matmul_latency_ms": {
        "gr": {64: 0.42, 128: 0.60, 256: 1.59, 512: 1.83, 1024: 4.07},
        "dn": {64: 0.25, 128: 0.42, 256: 0.76, 512: 1.44, 1024: 5.09},
        # Anchored on REAL ptq_lpbq.mllm weights via
        # mllm-qwen3-aot-attn-lpbq-microbench-c --params ... --mode {q,kv,o}.
        # Synthetic uniform-weight bins were 15-40% optimistic vs real.
        "q_proj": {64: 0.1337, 128: 0.2419, 256: 0.3963, 512: 0.4782, 1024: 1.0216},
        "k_proj": {64: 0.1089, 128: 0.1609, 256: 0.2746, 512: 0.3749, 1024: 0.9560},
        "v_proj": {64: 0.1089, 128: 0.1609, 256: 0.2746, 512: 0.3749, 1024: 0.9560},
        "o_proj": {64: 0.1699, 128: 0.2588, 256: 0.3581, 512: 0.4769, 1024: 1.0325},
    },

    # Fused q+k+v graph latency. Real-params measurement.
    "fused_qkv_npu_ms": {
        64: 0.2372, 128: 0.2846, 256: 0.4604, 512: 0.7341, 1024: 1.6928,
    },

    # --------------------------------------------------------------------
    # Bandwidth-bound op rates (effective GB/s, at Sq=1024).
    # Source: mllm-bench-elementwise --device {cpu,opencl} --ops {rmsnorm,add}
    # at attention widths (2048 hidden, 128 head_dim).
    # NPU rates from docs/qnn_backend/gemm_latency.md (GemmLatencyTest):
    # bandwidth-bound on-chip kernels — rate doesn't vary much by width.
    # --------------------------------------------------------------------
    "norm_rate_gbps": {
        # width 2048 (in_norm / post_norm at hidden)
        2048: {"NPU": 43.8, "CPU": 15.7, "GPU": 12.5},
        # width 128 (q_norm / k_norm at head_dim — bench at width=128, rows=Sq*heads)
        128:  {"NPU": 43.8, "CPU":  6.0, "GPU":  2.1},
        # width 6144 (kept for reference / MLP gateup-side norms if needed)
        6144: {"NPU": 43.8, "CPU": 16.1, "GPU": 14.0},
    },
    "norm_buffers": 2,    # in + out (gamma/sigmoid intermediate on-chip)

    # RoPE: 7-node decomposed kernel; rate measured at q-shape [16, 1024, 128].
    # k-side has half the heads (8) so half the bytes — same per-byte rate.
    # Source: gemm_latency.md elementwise three-way table.
    "rope_rate_gbps": {"NPU": 9.9, "CPU": 2.2, "GPU": 10.4},
    "rope_buffers":   4,    # in + sin + cos + out

    # Per-GPU-op transfer overhead beyond the rate-bandwidth model.
    # In the default (copy) path, a GPU op moves data:
    #   QNN rpcmem → CPU heap → OpenCL device → kernel → OpenCL → CPU heap → QNN rpcmem
    # mllm-bench-elementwise --ops rope (source of rope_rate_gbps["GPU"]=10.4)
    # only does the CPU↔OpenCL pair; the rpcmem hops are extra.
    #
    # IMPLEMENTED zero-copy path via cl_khr_external_memory_dma_buf (rpcmem
    # is dmabuf-backed; OpenCLAllocator::createIonAlias imports the fd as a
    # cl_mem aliasing the same physical pages — true zero-copy, analog of
    # CUDA pinned memory but stronger). Driver flag: `--zero-copy` in
    # mllm-het-block. With it, the per-GPU-op overhead is essentially zero.
    #
    # Measured calibration (Sq=512 × T=2, pinned cpu=6,7):
    #   without --zero-copy : GPU busy 4.79 ms → overhead 0.6 ms/op
    #   with    --zero-copy : GPU busy 2.21 ms → overhead 0.0 ms/op
    "gpu_x2x_overhead_ms_per_op": 0.6,         # default; pessimistic
    "gpu_x2x_overhead_ms_per_op_zerocopy": 0.0, # when using --zero-copy

    # Residual add at hidden=2048 width.
    # NPU: 20.8 GB/s @ 2048-wide (gemm_latency.md ElementwiseAddTcmCliff test).
    # CPU/GPU: this-session measurement at Sq=1024 width=2048.
    "add_rate_gbps": {
        2048: {"NPU": 20.8, "CPU": 90.5, "GPU": 12.4},
    },
    "add_buffers": 3,   # 2 in + 1 out

    # MLP activations — preserved from qwen3_1p7b_mlp.py.
    "activation": {
        "silu":       {"buffers": 2, "rate_gbps": {"NPU": 5.2,  "CPU": 7.3,  "GPU": 17.0}},
        "gateup_mul": {"buffers": 3, "rate_gbps": {"NPU": 26.4, "CPU": 72.0, "GPU": 8.9}},
    },

    # --------------------------------------------------------------------
    # Cross-engine transfer (ms per MB). Preserved.
    # --------------------------------------------------------------------
    "transfer_ms_per_mb": {
        ("NPU", "CPU"): 0.0,
        ("CPU", "NPU"): 0.0,
        ("CPU", "GPU"): 0.034,
        ("GPU", "CPU"): 0.034,
        ("NPU", "GPU"): 0.034,
        ("GPU", "NPU"): 0.034,
    },

    # OMP-on-worker-thread penalty.
    "cpu_worker_penalty": 3.0,

    # --------------------------------------------------------------------
    # Fused-on-NPU references.
    #
    # fused_mlp_npu_ms: full MLP single-dispatch (mllm-qwen3-aot-mlp-lpbq-bench
    # --mode full). Preserved.
    #
    # No fused full-block (norm+qkv+norms+rope+...+o_proj) reference yet —
    # would require an attn_no_gqa AOT mode. The simulator sums per-op for
    # the all-NPU attention pre-block, which over-counts VTCM-fusion savings
    # the production graph would get (small for matmul-dominated ops, larger
    # for norm+rope chains). Plug a real fused number here when measured.
    # --------------------------------------------------------------------
    # Anchored from REAL ptq_lpbq.mllm weights via mllm-qwen3-aot-mlp-lpbq-bench
    # --mode full. Synthetic all-int4-mid (value=8) weights trigger a degenerate
    # fast path in the QNN LPBQ kernel (~3.2 ms at Sq=1024 vs ~9 ms real). Use
    # real weights when refreshing this table.
    "fused_mlp_npu_ms": {64: 1.09, 128: 1.09, 256: 2.32, 512: 2.77, 1024: 9.23},

    # Larger-scope fusion lookups for VTCM-resident chains the production NPU
    # graph compiler builds. These supersede per-op summation when ALL ops in
    # the chain are pinned to NPU — eliminates the inter-op DRAM round-trips
    # the per-op rate model implicitly assumes. Measured via the `split` mode
    # of mllm-qwen3-aot-attn-lpbq-microbench-c (pre_attn + post_attn bins,
    # real PTQ params) plus an estimated +1.12 ms / 1024 tokens for per-head
    # q/k_norm + RoPE which our split bin skipped (production uses per-head
    # Conv2D form; bench measures multi-head — these two ops are bandwidth-
    # bound so estimate is reasonable).
    "fused_pre_attn_npu_ms": {  # in_norm + qkv + q/k_norm + q/k_rope
        64:   0.81,
        128:  1.10,
        256:  2.11,
        512:  3.88,
        1024: 8.14,  # NB: this is pre+post+correction. Set up below to be just pre+correction.
    },
    # Actually: fused_pre_attn = pre_only_measured + qknorm_rope_correction
    # See fusion_chains below. Replaced with separate pre / post tables:
    "_fused_pre_only_ms":  {64: 0.19, 128: 0.28, 256: 0.47, 512: 0.80, 1024: 1.58},
    "_fused_post_only_ms": {64: 0.52, 128: 0.70, 256: 1.38, 512: 2.51, 1024: 5.47},
    # qknorm+rope estimate: bandwidth-bound, scales linearly with Sq.
    # At Sq=1024: q/k_norm 0.27 ms (NPU 43.8 GB/s) + q/k_rope 0.85 ms (NPU 9.9 GB/s) = 1.12 ms.
    "_qknorm_rope_ms_per_1024": 1.12,

    # Fusion chains. Each chain lists the ops that fuse into a single
    # VTCM-resident dispatch on NPU. The simulator detects when ALL ops in a
    # chain are pinned to NPU and replaces their summed per-op latency with
    # the measured fused lookup (analog of the existing fused_mlp_npu_ms
    # special case, now generalized).
    #
    # Chains are tried LARGEST first — a longer chain subsumes shorter ones
    # whose ops it overlaps with.
    "fusion_chains": [
        {
            "name": "all_npu_pre_attn",
            "ops": ["in_norm", "q_proj", "k_proj", "v_proj",
                    "q_norm", "k_norm", "q_rope", "k_rope"],
            "lookup": "_fused_pre_only_ms",
            "correction_key": "_qknorm_rope_ms_per_1024",  # add bandwidth-derived q/k_norm+rope
        },
        {
            "name": "all_npu_post_attn",
            "ops": ["o_proj", "res1", "post_norm",
                    "gr", "silu", "gateup_mul", "dn", "res2"],
            "lookup": "_fused_post_only_ms",
        },
        {
            "name": "fused_mlp_only",  # fallback when norm/add are off-NPU
            "ops": ["gr", "silu", "gateup_mul", "dn"],
            "lookup": "fused_mlp_npu_ms",
        },
    ],
}
