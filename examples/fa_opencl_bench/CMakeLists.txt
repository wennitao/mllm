if(MLLM_BUILD_OPENCL_BACKEND)
  add_executable(mllm-fa-opencl-bench main.cpp)
  target_link_libraries(mllm-fa-opencl-bench PRIVATE MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-fa-opencl-bench PRIVATE ${MLLM_INCLUDE_DIR})

  # Standalone variant-C fp16 GEMM microbench at FA-relevant shapes, to measure
  # whether the ~1080 GF/s GEMM structure survives K=128 (QK^T) / N=128 (P*V).
  add_executable(mllm-fa-gemm-vc-bench gemm_vc_ref.cpp)
  target_link_libraries(mllm-fa-gemm-vc-bench PRIVATE MllmRT MllmCPUBackend MllmOpenCLBackend)
  target_include_directories(mllm-fa-gemm-vc-bench PRIVATE ${MLLM_INCLUDE_DIR})
endif()
