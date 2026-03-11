# Build on Paladin

## Step 2 offline compilation to generate qnn context

```
# In the mllm-v2 project root directory
python task.py tasks/build_x86_qnn_aot.yaml

# fix libunwind.so, change to your android-ndk path
mkdir -p /tmp/mllm-qnn-host-libs
ln -sfn /mnt/raid0_ssd/wentao/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/lib/x86_64-unknown-linux-gnu/libunwind.so /tmp/mllm-qnn-host-libs/libunwind.so.1

# Run the compiler program
LD_LIBRARY_PATH=/tmp/mllm-qnn-host-libs:/mnt/raid0_ssd/wentao/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/lib/:$LD_LIBRARY_PATH ./build-qnn-aot/bin/mllm-qwen3-aot-sha-c -m /mnt/raid0_ssd/wentao/mllm/Qwen3-1.7b-mllm/qwen3_1.7b.mllm -c ./examples/qwen3_qnn_aot/config_1.7B.json --aot_config ./examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B.json
# Optional, default value is /opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/
# --qnn_env_path path/to/qnn_sdk.
```
