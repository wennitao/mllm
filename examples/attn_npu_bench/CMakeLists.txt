add_executable(mllm-attn-npu-bench main.cpp)
target_link_libraries(mllm-attn-npu-bench PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-attn-npu-bench PRIVATE ${MLLM_INCLUDE_DIR})
