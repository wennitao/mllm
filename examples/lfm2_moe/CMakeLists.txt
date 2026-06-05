add_executable(mllm-lfm2_moe-logits logits_check.cpp)
target_link_libraries(mllm-lfm2_moe-logits PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-lfm2_moe-logits PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-lfm2_moe-runner main.cpp)
target_link_libraries(mllm-lfm2_moe-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-lfm2_moe-runner PRIVATE ${MLLM_INCLUDE_DIR})
