add_executable(mllm-qwen3-decode-bench-q4 main.cpp)
target_link_libraries(mllm-qwen3-decode-bench-q4 PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-decode-bench-q4 PRIVATE ${MLLM_INCLUDE_DIR})
