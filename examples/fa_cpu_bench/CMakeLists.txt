add_executable(mllm-fa-cpu-bench main.cpp)
target_link_libraries(mllm-fa-cpu-bench PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-fa-cpu-bench PRIVATE ${MLLM_INCLUDE_DIR})
