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

  # Fake-scale w4a16-LPBQ attention-projection microbench compiler — same
  # synthetic-params pattern as the MLP microbench, for q/k/v/o_proj. Feeds
  # tools/het_sim/qwen3_1p7b_block.py with measured LPBQ projection costs.
  add_executable(mllm-qwen3-aot-attn-lpbq-microbench-c compile_attn_lpbq_microbench.cpp)
  target_link_libraries(mllm-qwen3-aot-attn-lpbq-microbench-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-attn-lpbq-microbench-c PRIVATE ${MLLM_INCLUDE_DIR})
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

# Driver for the fake-scale w4a16-LPBQ attention-projection microbench bins.
add_executable(mllm-qwen3-aot-attn-lpbq-bench aot_bench_attn_lpbq.cpp)
target_link_libraries(mllm-qwen3-aot-attn-lpbq-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-attn-lpbq-bench PRIVATE ${MLLM_INCLUDE_DIR})

# Driver for the split (pre_attn + post_attn) all-NPU baseline bin — the
# apples-to-apples measurement that anchors tools/het_sim/qwen3_1p7b_block.py.
add_executable(mllm-qwen3-aot-block-split-bench aot_bench_block_split.cpp)
target_link_libraries(mllm-qwen3-aot-block-split-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-block-split-bench PRIVATE ${MLLM_INCLUDE_DIR})
