# AOT targets run on x86
if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  add_executable(mllm-qwen3-aot-c compile.cpp)
  target_link_libraries(mllm-qwen3-aot-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-c PRIVATE ${MLLM_INCLUDE_DIR})

  # SHA (Single Head Attention) version - MHA to SHA optimization
  add_executable(mllm-qwen3-aot-sha-c compile_sha.cpp)
  target_link_libraries(mllm-qwen3-aot-sha-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-c PRIVATE ${MLLM_INCLUDE_DIR})

  # fp16 block-sparse attention version (Phase 1+2: model + compile, no runner yet).
  add_executable(mllm-qwen3-aot-fp16-blocksparse-c compile_fp16_blocksparse.cpp)
  target_link_libraries(mllm-qwen3-aot-fp16-blocksparse-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-fp16-blocksparse-c PRIVATE ${MLLM_INCLUDE_DIR})

  # fp16 PER-QB CAUSAL block-sparse attention version (compile only).
  # Graph processes one BQ-row chunk per dispatch; runner-side gather
  # + num_qb dispatches is in BlockSparsePromptProcessor / aot_run_blocksparse.cpp.
  add_executable(mllm-qwen3-aot-fp16-blocksparse-causal-c compile_fp16_blocksparse_causal.cpp)
  target_link_libraries(mllm-qwen3-aot-fp16-blocksparse-causal-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-fp16-blocksparse-causal-c PRIVATE ${MLLM_INCLUDE_DIR})

  # SHA + LPBQ projections + block-sparse attention compute (fp16 K/V inputs).
  add_executable(mllm-qwen3-aot-sha-blocksparse-c compile_sha_blocksparse.cpp)
  target_link_libraries(mllm-qwen3-aot-sha-blocksparse-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-blocksparse-c PRIVATE ${MLLM_INCLUDE_DIR})

  # SHA + LPBQ projections + per-qb CAUSAL block-sparse attention (uint8 KV).
  # Uses ptq_lpbq.mllm; per dispatch processes one BQ-row chunk, runner gathers
  # historical K/V from past_key/past_value into K_arranged/V_arranged on CPU.
  add_executable(mllm-qwen3-aot-sha-blocksparse-causal-c compile_sha_blocksparse_causal.cpp)
  target_link_libraries(mllm-qwen3-aot-sha-blocksparse-causal-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-blocksparse-causal-c PRIVATE ${MLLM_INCLUDE_DIR})

  # SPLIT-PREFILL variant: 2L+1 graphs (L+1 non-attention chunks at M=Sq + L
  # per-layer attention chunks at M=BQ). Non-attention compute escapes the
  # M=BQ=32 HMX efficiency cliff. Same .mllm params load unchanged.
  add_executable(mllm-qwen3-aot-sha-blocksparse-causal-split-c compile_sha_blocksparse_causal_split.cpp)
  target_link_libraries(mllm-qwen3-aot-sha-blocksparse-causal-split-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-blocksparse-causal-split-c PRIVATE ${MLLM_INCLUDE_DIR})

  # Fake-scale w4a16-LPBQ MLP microbench compiler (synthetic params, no checkpoint).
  add_executable(mllm-qwen3-aot-mlp-lpbq-microbench-c compile_mlp_lpbq_microbench.cpp)
  target_link_libraries(mllm-qwen3-aot-mlp-lpbq-microbench-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-mlp-lpbq-microbench-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-exp-microbench-c compile_exp_microbench.cpp)
  target_link_libraries(mllm-qwen3-aot-exp-microbench-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-exp-microbench-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-deltanet-step-c compile_deltanet_step.cpp)
  target_link_libraries(mllm-qwen3-aot-deltanet-step-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-deltanet-step-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-deltanet-perhead-c compile_deltanet_perhead.cpp)
  target_link_libraries(mllm-qwen3-aot-deltanet-perhead-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-deltanet-perhead-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-deltanet-decode-c compile_deltanet_decode.cpp)
  target_link_libraries(mllm-qwen3-aot-deltanet-decode-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-deltanet-decode-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-deltanet-prefill-c compile_deltanet_prefill.cpp)
  target_link_libraries(mllm-qwen3-aot-deltanet-prefill-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-deltanet-prefill-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-attn-decode-c compile_attn_decode.cpp)
  target_link_libraries(mllm-qwen3-aot-attn-decode-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-attn-decode-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-attn-prefill-c compile_attn_prefill.cpp)
  target_link_libraries(mllm-qwen3-aot-attn-prefill-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-attn-prefill-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-lmhead-c compile_lmhead.cpp)
  target_link_libraries(mllm-qwen3-aot-lmhead-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-lmhead-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-mlp-decode-c compile_mlp_decode.cpp)
  target_link_libraries(mllm-qwen3-aot-mlp-decode-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-mlp-decode-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-stack-test-c compile_stack_test.cpp)
  target_link_libraries(mllm-qwen3-aot-stack-test-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-stack-test-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-monolithic-c compile_monolithic.cpp)
  target_link_libraries(mllm-qwen3-aot-monolithic-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-monolithic-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-monolithic-prefill-c compile_monolithic_prefill.cpp)
  target_link_libraries(mllm-qwen3-aot-monolithic-prefill-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-monolithic-prefill-c PRIVATE ${MLLM_INCLUDE_DIR})

  # Fake-scale w4a16-LPBQ attention-projection microbench compiler — same
  # synthetic-params pattern as the MLP microbench, for q/k/v/o_proj. Feeds
  # tools/het_sim/qwen3_1p7b_block.py with measured LPBQ projection costs.
  add_executable(mllm-qwen3-aot-attn-lpbq-microbench-c compile_attn_lpbq_microbench.cpp)
  target_link_libraries(mllm-qwen3-aot-attn-lpbq-microbench-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-attn-lpbq-microbench-c PRIVATE ${MLLM_INCLUDE_DIR})

  # MONOLITHIC block-sparse: ONE graph for the whole model with in-graph block
  # selection (score->softmax->pool->TopK) and gather (GatherNd). No CPU pipeline.
  add_executable(mllm-qwen3-aot-sha-blocksparse-monolithic-c compile_sha_blocksparse_monolithic.cpp)
  target_link_libraries(mllm-qwen3-aot-sha-blocksparse-monolithic-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-sha-blocksparse-monolithic-c PRIVATE ${MLLM_INCLUDE_DIR})
endif()

add_executable(mllm-qwen3-aot-runner aot_run.cpp)
target_link_libraries(mllm-qwen3-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})

# Runner that drives the per-qb causal block-sparse model: CPU gather +
# num_qb dispatches via BlockSparsePromptProcessor.
add_executable(mllm-qwen3-aot-blocksparse-runner aot_run_blocksparse_causal.cpp)
target_link_libraries(mllm-qwen3-aot-blocksparse-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-blocksparse-runner PRIVATE ${MLLM_INCLUDE_DIR})

# Runner for the LPBQ SHA per-qb causal block-sparse model: same per-qb
# dispatch flow but with uint8 KV cache + uint16-quant mask.
add_executable(mllm-qwen3-aot-sha-blocksparse-runner aot_run_sha_blocksparse_causal.cpp)
target_link_libraries(mllm-qwen3-aot-sha-blocksparse-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-sha-blocksparse-runner PRIVATE ${MLLM_INCLUDE_DIR})

# Runner for the SPLIT-PREFILL variant: drives the 2L+1 chunked context via
# ShaBlockSparsePromptProcessorSplit. Prefill only — decode TBD.
add_executable(mllm-qwen3-aot-sha-blocksparse-split-runner aot_run_sha_blocksparse_causal_split.cpp)
target_link_libraries(mllm-qwen3-aot-sha-blocksparse-split-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-sha-blocksparse-split-runner PRIVATE ${MLLM_INCLUDE_DIR})

# Monolithic block-sparse runner: single graph "model", greedy decode-by-reprefill.
add_executable(mllm-qwen3-aot-sha-blocksparse-monolithic-runner aot_run_sha_blocksparse_monolithic.cpp)
target_link_libraries(mllm-qwen3-aot-sha-blocksparse-monolithic-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-sha-blocksparse-monolithic-runner PRIVATE ${MLLM_INCLUDE_DIR})

# Attn-only AOT pipeline micro-benchmark: sync vs gather-overlap pipeline on the
# real quantized attn graphs (qwen3-attn-only.bin). Verifies the pipeline does
# not beat sync in the short-dispatch + slow-gather regime.
add_executable(mllm-qwen3-aot-attn-bench aot_bench_attn_pipeline.cpp)
target_link_libraries(mllm-qwen3-aot-attn-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-attn-bench PRIVATE ${MLLM_INCLUDE_DIR})

# Driver for the fake-scale w4a16-LPBQ MLP microbench bin — faithful full-MLP latency.
add_executable(mllm-qwen3-aot-mlp-lpbq-bench aot_bench_mlp_lpbq.cpp)
target_link_libraries(mllm-qwen3-aot-mlp-lpbq-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-mlp-lpbq-bench PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-exp-run run_exp_microbench.cpp)
target_link_libraries(mllm-qwen3-aot-exp-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-exp-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-deltanet-step-run run_deltanet_step.cpp)
target_link_libraries(mllm-qwen3-aot-deltanet-step-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-deltanet-step-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-deltanet-decode-run run_deltanet_decode.cpp)
target_link_libraries(mllm-qwen3-aot-deltanet-decode-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-deltanet-decode-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-deltanet-prefill-run run_deltanet_prefill.cpp)
target_link_libraries(mllm-qwen3-aot-deltanet-prefill-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-deltanet-prefill-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-attn-decode-run run_attn_decode.cpp)
target_link_libraries(mllm-qwen3-aot-attn-decode-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-attn-decode-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-attn-prefill-run run_attn_prefill.cpp)
target_link_libraries(mllm-qwen3-aot-attn-prefill-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-attn-prefill-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-lmhead-run run_lmhead.cpp)
target_link_libraries(mllm-qwen3-aot-lmhead-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-lmhead-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-mono-run run_monolithic.cpp)
target_link_libraries(mllm-qwen3-aot-mono-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-mono-run PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-mono-prefill-run run_monolithic_prefill.cpp)
target_link_libraries(mllm-qwen3-aot-mono-prefill-run PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-mono-prefill-run PRIVATE ${MLLM_INCLUDE_DIR})

# Driver for the fake-scale w4a16-LPBQ attention-projection microbench bins.
add_executable(mllm-qwen3-aot-attn-lpbq-bench aot_bench_attn_lpbq.cpp)
target_link_libraries(mllm-qwen3-aot-attn-lpbq-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-attn-lpbq-bench PRIVATE ${MLLM_INCLUDE_DIR})

# Driver for the split (pre_attn + post_attn) all-NPU baseline bin.
add_executable(mllm-qwen3-aot-block-split-bench aot_bench_block_split.cpp)
target_link_libraries(mllm-qwen3-aot-block-split-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-block-split-bench PRIVATE ${MLLM_INCLUDE_DIR})

# End-to-end LPBQ decode bench: dispatch the per-layer Sq=1 split bin
# L=28 times to simulate one decode step, generates N tokens. Measures
# real all-NPU ms/token (compute only, no GQA/norm/rope/lm_head).
add_executable(mllm-qwen3-decode-e2e bench_decode_e2e.cpp)
target_link_libraries(mllm-qwen3-decode-e2e PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-decode-e2e PRIVATE ${MLLM_INCLUDE_DIR})

# CPU LPBQ matmul microbench — drives mllm::cpu::lpbq_matmul_fp16 on real
# Qwen3-1.7B q/k/v/o/gate/up/down weights. Feeds the het-pipeline simulator
# (tools/het_sim) with measured CPU LPBQ latency for tensor-parallel matmul
# partitioning experiments.
add_executable(mllm-qwen3-cpu-lpbq-bench bench_cpu_lpbq.cpp)
target_link_libraries(mllm-qwen3-cpu-lpbq-bench PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-cpu-lpbq-bench PRIVATE ${MLLM_INCLUDE_DIR})

# LPBQ matmul TP toy bench — measure real cross-engine sync overhead at
# decode (Sq=1) to validate the simulator's 50 μs/edge assumption. Runs
# NPU q_proj concurrently with CPU shard via std::thread + join.
add_executable(mllm-qwen3-tp-toy bench_matmul_tp_toy.cpp)
target_link_libraries(mllm-qwen3-tp-toy PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-tp-toy PRIVATE ${MLLM_INCLUDE_DIR})

# OpenCL LPBQ matmul microbench — same params + correctness check vs CPU,
# but kernel runs on Adreno via an inline OpenCL program. Built only when
# the OpenCL backend is enabled.
if(MLLM_BUILD_OPENCL_BACKEND)
  add_executable(mllm-qwen3-opencl-lpbq-bench bench_opencl_lpbq.cpp)
  target_link_libraries(mllm-qwen3-opencl-lpbq-bench PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-opencl-lpbq-bench PRIVATE ${MLLM_INCLUDE_DIR})

  # Op-level correctness test for the OpenCL LPBQ Linear path (vs CPU ref).
  add_executable(mllm-qwen3-opencl-linear-lpbq-test test_opencl_linear_lpbq.cpp)
  target_link_libraries(mllm-qwen3-opencl-linear-lpbq-test PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-opencl-linear-lpbq-test PRIVATE ${MLLM_INCLUDE_DIR})

  # Layer-level E2E test: real .mllm weights through the LPBQ op (fp32 IO) vs CPU.
  add_executable(mllm-qwen3-opencl-linear-lpbq-layer test_opencl_linear_lpbq_layer.cpp)
  target_link_libraries(mllm-qwen3-opencl-linear-lpbq-layer PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-opencl-linear-lpbq-layer PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-probe-ext probe_ext.cpp)
  target_link_libraries(mllm-probe-ext PRIVATE MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-probe-ext PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-fp16-gemm-va bench_opencl_fp16_va.cpp)
  target_link_libraries(mllm-qwen3-fp16-gemm-va PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-fp16-gemm-va PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-fp16-gemm-vb bench_opencl_fp16_vb.cpp)
  target_link_libraries(mllm-qwen3-fp16-gemm-vb PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-fp16-gemm-vb PRIVATE ${MLLM_INCLUDE_DIR})

  # Roofline anchors: raw fp16 FMA compute ceiling + global read bandwidth.
  add_executable(mllm-qwen3-opencl-roofline bench_opencl_roofline.cpp)
  target_link_libraries(mllm-qwen3-opencl-roofline PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-opencl-roofline PRIVATE ${MLLM_INCLUDE_DIR})

  # fp32 twin of variant A — A/B on arithmetic precision (half vs float).
  add_executable(mllm-qwen3-fp32-gemm-va bench_opencl_fp32_va.cpp)
  target_link_libraries(mllm-qwen3-fp32-gemm-va PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-fp32-gemm-va PRIVATE ${MLLM_INCLUDE_DIR})

  # variant C — fp16 port of llama.cpp q4_0 GEMM (amortized packed weight loads).
  add_executable(mllm-qwen3-fp16-gemm-vc bench_opencl_fp16_vc.cpp)
  target_link_libraries(mllm-qwen3-fp16-gemm-vc PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-fp16-gemm-vc PRIVATE ${MLLM_INCLUDE_DIR})

  # fp32 twin of variant C — precision A/B at the optimized operating point.
  add_executable(mllm-qwen3-fp32-gemm-vc bench_opencl_fp32_vc.cpp)
  target_link_libraries(mllm-qwen3-fp32-gemm-vc PRIVATE
    MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-fp32-gemm-vc PRIVATE ${MLLM_INCLUDE_DIR})
endif()
