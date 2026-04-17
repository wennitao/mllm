add_executable(mllm-qwen3-runner main.cpp)
target_link_libraries(mllm-qwen3-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-qwen3-runner PRIVATE ${MLLM_INCLUDE_DIR})

if(MLLM_BUILD_OPENCL_BACKEND)
  add_executable(mllm-qwen3-opencl-runner main_opencl.cpp)
  target_link_libraries(mllm-qwen3-opencl-runner PRIVATE MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-qwen3-opencl-runner PRIVATE ${MLLM_INCLUDE_DIR})
endif()
