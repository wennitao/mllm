add_executable(mllm-mlp-cpu-bench-q4 main.cpp)
target_link_libraries(mllm-mlp-cpu-bench-q4 PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-mlp-cpu-bench-q4 PRIVATE ${MLLM_INCLUDE_DIR})
