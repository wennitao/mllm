# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Measured Qwen3-1.7B MLP per-op latency model for the heterogeneous-pipeline
# simulator (tools/het_sim/het_sim.py). All numbers were collected this session
# on SM8750 (Adreno 830 + Hexagon V79) under QAIRT 2.43.0.260128, for the LPBQ
# w4a16 MLP shape (hidden=2048, intermediate=6144).
#
# Scope (per request): no NPU dispatch overhead, no custom kernels, Qwen3-1.7B
# shapes only. Matmuls are NPU-only (no LPBQ kernel exists on CPU/GPU).

SPEC = {
    "shape": {"H": 2048, "I": 6144},

    # MLP DAG. `gr` packs gate/up matmul (matches our `gateraw`/`gateup` bin
    # granularity); `silu` packs sigmoid + the gate*sigmoid mul (= F::silu);
    # `gateup_mul` is the final silu(g)*up; `dn` is the down matmul.
    "ops": {
        "gr":         {"deps": [],                "outputs": ["g", "u"], "engines": ["NPU"]},
        "silu":       {"deps": ["g"],             "outputs": ["sig_g"],  "engines": ["NPU", "CPU", "GPU"]},
        "gateup_mul": {"deps": ["sig_g", "u"],    "outputs": ["h"],      "engines": ["NPU", "CPU", "GPU"]},
        "dn":         {"deps": ["h"],             "outputs": ["o"],      "engines": ["NPU"]},
    },
    "op_order": ["gr", "silu", "gateup_mul", "dn"],

    # Output sizes (bytes per token at Sq=1, fp16).
    "output_bytes_per_token": {
        "x":     2048 * 2,   # input, hidden
        "g":     6144 * 2,
        "u":     6144 * 2,
        "sig_g": 6144 * 2,
        "h":     6144 * 2,
        "o":     2048 * 2,
    },

    # NPU matmul latency (ms) at compiled tile width Sq.
    # Source: `mllm-het-mlp --npu-only` sweep over gatedown bins at Sq={64,128,256,512},
    # and the LPBQ MLP microbench at Sq=1024 (split into gr/dn shares from the per-op
    # profile: the .260128 down@1024 cliff inflates dn relative to gr).
    "npu_matmul_latency_ms": {
        "gr": {64: 0.42, 128: 0.60, 256: 1.59, 512: 1.83, 1024: 4.07},
        "dn": {64: 0.25, 128: 0.42, 256: 0.76, 512: 1.44, 1024: 5.09},
    },

    # Activation rates (effective GB/s) — bandwidth-bound, ms scales with Sq.
    # Source: mllm-bench-elementwise three-way table (fp16, rows=1024, width=6144).
    # `silu` rate uses 2 buffers (in + out, the sigmoid intermediate is on-chip).
    # `gateup_mul` rate uses 3 buffers (2 read + 1 write).
    "activation": {
        "silu":       {"buffers": 2, "rate_gbps": {"NPU": 5.2,  "CPU": 7.3,  "GPU": 17.0}},
        "gateup_mul": {"buffers": 3, "rate_gbps": {"NPU": 26.4, "CPU": 72.0, "GPU": 8.9}},
    },

    # Cross-engine transfer cost (ms per MB transferred, one direction).
    # CPU<->GPU: 0.82 ms round-trip @ 12 MB tensor (mllm-bench-elementwise --ops xfer)
    #             = 0.41 ms per direction per 12 MB = 0.034 ms/MB.
    # NPU<->CPU: rpcmem is shared host memory -> zero-copy.
    # NPU<->GPU: routed through CPU (X2X has no direct kQNN<->kOpenCL path); dominant
    #             cost is CPU<->GPU, NPU<->CPU is free.
    "transfer_ms_per_mb": {
        ("NPU", "CPU"): 0.0,
        ("CPU", "NPU"): 0.0,
        ("CPU", "GPU"): 0.034,
        ("GPU", "CPU"): 0.034,
        ("NPU", "GPU"): 0.034,   # via CPU
        ("GPU", "NPU"): 0.034,
    },

    # CPU OMP-on-worker-thread penalty. Measured (memory `omp-on-worker-thread-3x-penalty`):
    # an OMP parallel-for launched from a pinned non-main worker costs ~3x serial.
    # Applies to CPU ops in any mode where CPU runs on the worker thread (i.e. mode=='threaded').
    "cpu_worker_penalty": 3.0,

    # Fused-on-NPU reference (ms) — what we're trying to beat. Single dispatch,
    # keeps the 6144-wide intermediate in VTCM, no split overhead. Source: LPBQ
    # microbench (`mllm-qwen3-aot-mlp-lpbq-bench --mode full`).
    "fused_npu_reference_ms": {64: 1.09, 128: 1.09, 256: 2.32, 512: 2.68, 1024: 12.5},
}
