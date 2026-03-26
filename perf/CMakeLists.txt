add_executable(mllm-perf main.cpp)
target_link_libraries(mllm-perf PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-perf PRIVATE ${MLLM_INCLUDE_DIR})
