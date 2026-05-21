add_executable(mllm-bench-block-selection bench_block_selection.cpp)
target_include_directories(mllm-bench-block-selection PRIVATE ${MLLM_INCLUDE_DIR})

if(MLLM_BUILD_OPENCL_BACKEND)
  target_link_libraries(mllm-bench-block-selection PRIVATE MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_compile_definitions(mllm-bench-block-selection PRIVATE BENCH_HAS_OPENCL=1)
else()
  target_link_libraries(mllm-bench-block-selection PRIVATE MllmRT MllmCPUBackend)
endif()
