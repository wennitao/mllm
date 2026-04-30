add_executable(Mllm-Test-QNN-LLaMAMulOp LLaMAMulOpTest.cpp)

target_link_libraries(Mllm-Test-QNN-LLaMAMulOp PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-LLaMAMulOp PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-FlashAttentionOp FlashAttentionOpTest.cpp)

target_link_libraries(Mllm-Test-QNN-FlashAttentionOp PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-FlashAttentionOp PRIVATE ${MLLM_INCLUDE_DIR})

# On-device deployment (adb):
#   push libQnnHtp.so, libQnnSystem.so, libQnnLLaMAPackage_CPU.so, and
#   libQnnLLaMAPackage.so (hexagon variant) to the same directory as this binary.
#   Run with: LD_LIBRARY_PATH=. ./Mllm-Test-QNN-LLaMAMulOp
#   The test binary prepends . and /data/local/tmp to ADSP_LIBRARY_PATH so the
#   DSP JIT compiler can resolve the hexagon library. See scripts/adb_push.sh.
