# Runtime: runs on Android device (loads pre-compiled .bin)
add_executable(mllm-qwen3-npu main.cpp)
target_link_libraries(mllm-qwen3-npu PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-npu PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/examples)

# Compiler: runs on x86 with QNN SDK to produce the prefill .bin
if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  add_executable(mllm-qwen3-npu-compile compile.cpp)
  target_link_libraries(mllm-qwen3-npu-compile PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-npu-compile PRIVATE ${MLLM_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/examples)
endif()
