add_executable(mllm-mlp-npu-bench-lpbq main.cpp)
target_link_libraries(mllm-mlp-npu-bench-lpbq PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-mlp-npu-bench-lpbq PRIVATE ${MLLM_INCLUDE_DIR})
