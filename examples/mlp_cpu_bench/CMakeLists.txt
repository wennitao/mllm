add_executable(mllm-mlp-cpu-bench main.cpp)
target_link_libraries(mllm-mlp-cpu-bench PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-mlp-cpu-bench PRIVATE ${MLLM_INCLUDE_DIR})
