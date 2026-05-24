add_executable(mllm-mlp-npu-bench main.cpp)
target_link_libraries(mllm-mlp-npu-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-mlp-npu-bench PRIVATE ${MLLM_INCLUDE_DIR})
