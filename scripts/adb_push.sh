# adb push qwen3-1.7B-lpbq-sha.bin /data/local/tmp/

QNN_SDK_ROOT=$QAIRT_SDK_ROOT

# Push QNN libraries and Op Packages
ANDR_LIB=$QNN_SDK_ROOT/lib/aarch64-android
OP_PATH=mllm/backends/qnn/custom-op-package/LLaMAPackage/build
NDK_LIBOMP=${ANDROID_NDK_PATH:-/mnt/raid0_ssd/wentao/android-ndk-r29}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/21/lib/linux/aarch64/libomp.so

adb push $ANDR_LIB/libQnnHtp.so /data/local/tmp
adb push $ANDR_LIB/libQnnHtpV79Stub.so /data/local/tmp
adb push $ANDR_LIB/libQnnHtpPrepare.so /data/local/tmp
adb push $ANDR_LIB/libQnnHtpProfilingReader.so /data/local/tmp
adb push $ANDR_LIB/libQnnHtpOptraceProfilingReader.so /data/local/tmp
adb push $ANDR_LIB/libQnnHtpV79CalculatorStub.so /data/local/tmp
adb push $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so /data/local/tmp
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp

# ARM prepare requires both op packages to be registered:
#   libQnnLLaMAPackage_CPU.so -> target CPU
#   libQnnLLaMAPackage.so     -> target HTP, loaded by the skel via ADSP_LIBRARY_PATH
adb push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so /data/local/tmp/libQnnLLaMAPackage_CPU.so
adb push $OP_PATH/hexagon-v79/libQnnLLaMAPackage.so /data/local/tmp/libQnnLLaMAPackage.so

# Push mllm runner and libs to device
adb push build-android-arm64-v8a-qnn/bin/*.so /data/local/tmp
adb push $NDK_LIBOMP /data/local/tmp
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner /data/local/tmp
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-npu /data/local/tmp
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-runner /data/local/tmp

# Push the runner-specific config file. Do not use the raw HuggingFace config.json here:
# the AOT runner expects extra numeric fields such as max_cache_length.
adb push examples/qwen3_qnn_aot/config_1.7B.json /data/local/tmp/config_1.7B.json
adb push examples/qwen3_npu/config_1.7B_q4.json /data/local/tmp
adb push examples/qwen3_npu/config_1.7B_kai.json /data/local/tmp
adb push Qwen3-1.7b/tokenizer.json /data/local/tmp/qwen3-tokenizer.json
