# Build on Paladin

## Model Quantization

```
build/tools/mllm-quantizer/mllm-quantizer -i Qwen3-1.7b-mllm/qwen3_1.7b.mllm -c examples/qwen3_npu/quant_cfg_1.7B_q4_gguf_weight.json -iv v2 -o Qwen3-1.7b-mllm/qwen3_1.7b_q4.mllm -ov v2
```

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

```
# Cross-compile the aot_run program for the target device (e.g., Android)
python task.py tasks/build_android_qnn.yaml
```

bash scripts/adb_push.sh

```
# Execute on the device
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./mllm-qwen3-aot-runner -m qwen3-1.7B-lpbq-sha.bin -t qwen3-tokenizer.json -c config_1.7B.json --ar_len 32"
```

If Android reports `library "libomp.so" not found`, push the NDK OpenMP runtime as well:

```
adb push $ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/21/lib/linux/aarch64/libomp.so /data/local/tmp
```

```
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./mllm-qwen3-npu --npu_model_path /data/local/tmp/qwen3_1.7b_q4.mllm --cpu_model_path qwen3_1.7b.mllm --npu_config_path config_1.7B_q4.json --cpu_config_path config_1.7B_cpu.json --tokenizer_path qwen3-tokenizer.json --model_version v2"
```

QQUF Q4
```
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./mllm-qwen3-runner --model_path qwen3_1.7b_q4.mllm --model_version v2 --config_path config_1.7B_q4.json --tokenizer_path qwen3-tokenizer.json"
```

FP32
```
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./mllm-qwen3-runner --model_path qwen3_1.7b.mllm --model_version v2 --config_path config_1.7B_cpu.json --tokenizer_path qwen3-tokenizer.json"
```

Run on server
```
./build-sdk-x86/bin/mllm-qwen3-runner --model_path Qwen3-1.7b-mllm/qwen3_1.7b_q4.mllm --model_version v2 --config_path examples/qwen3_npu/config_1.7B_q4.json --tokenizer_path Qwen3-1.7b/tokenizer.json
```