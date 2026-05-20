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

add_executable(Mllm-Test-QNN-FlashAttentionInt8KVOp FlashAttentionInt8KVOpTest.cpp)

target_link_libraries(Mllm-Test-QNN-FlashAttentionInt8KVOp PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-FlashAttentionInt8KVOp PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-FlashAttentionCompare FlashAttentionCompareTest.cpp)

target_link_libraries(Mllm-Test-QNN-FlashAttentionCompare PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-FlashAttentionCompare PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-FlashAttentionDecomposed FlashAttentionDecomposedTest.cpp)

target_link_libraries(Mllm-Test-QNN-FlashAttentionDecomposed PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-FlashAttentionDecomposed PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-FlashAttentionTopK FlashAttentionTopKTest.cpp)

target_link_libraries(Mllm-Test-QNN-FlashAttentionTopK PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-FlashAttentionTopK PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-MemBandwidth MemBandwidthTest.cpp)

target_link_libraries(Mllm-Test-QNN-MemBandwidth PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
    MllmCPUBackend
)

target_compile_options(Mllm-Test-QNN-MemBandwidth PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(Mllm-Test-QNN-MemBandwidth PRIVATE ${OpenMP_CXX_FLAGS})
if(OpenMP_CXX_INCLUDE_DIR)
    target_include_directories(Mllm-Test-QNN-MemBandwidth PRIVATE ${OpenMP_CXX_INCLUDE_DIR})
endif()

target_include_directories(Mllm-Test-QNN-MemBandwidth PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-BlockSparse BlockSparseAttentionTest.cpp)

target_link_libraries(Mllm-Test-QNN-BlockSparse PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
    MllmCPUBackend
)

target_compile_options(Mllm-Test-QNN-BlockSparse PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(Mllm-Test-QNN-BlockSparse PRIVATE ${OpenMP_CXX_FLAGS})
if(OpenMP_CXX_INCLUDE_DIR)
    target_include_directories(Mllm-Test-QNN-BlockSparse PRIVATE ${OpenMP_CXX_INCLUDE_DIR})
endif()

target_include_directories(Mllm-Test-QNN-BlockSparse PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-BlockSparseCausal BlockSparseAttentionCausalTest.cpp)

target_link_libraries(Mllm-Test-QNN-BlockSparseCausal PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
    MllmCPUBackend
)

target_include_directories(Mllm-Test-QNN-BlockSparseCausal PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-GemmLatency GemmLatencyTest.cpp)

target_link_libraries(Mllm-Test-QNN-GemmLatency PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-GemmLatency PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-SoftmaxBlockSparseCausalOp SoftmaxBlockSparseCausalOpTest.cpp)

target_link_libraries(Mllm-Test-QNN-SoftmaxBlockSparseCausalOp PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
    MllmCPUBackend
)

target_include_directories(Mllm-Test-QNN-SoftmaxBlockSparseCausalOp PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(Mllm-Test-QNN-GraphSwitchLatency GraphSwitchLatencyTest.cpp)

target_link_libraries(Mllm-Test-QNN-GraphSwitchLatency PRIVATE
    gtest_main
    MllmRT
    MllmQNNBackend
)

target_include_directories(Mllm-Test-QNN-GraphSwitchLatency PRIVATE ${MLLM_INCLUDE_DIR})

# On-device deployment (adb):
#   push libQnnHtp.so, libQnnSystem.so, libQnnLLaMAPackage_CPU.so, and
#   libQnnLLaMAPackage.so (hexagon variant) to the same directory as this binary.
#   Run with: LD_LIBRARY_PATH=. ./Mllm-Test-QNN-LLaMAMulOp
#   The test binary prepends . and /data/local/tmp to ADSP_LIBRARY_PATH so the
#   DSP JIT compiler can resolve the hexagon library. See scripts/adb_push.sh.
