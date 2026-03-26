mllm-convertor   --input_path ~/models/Qwen3-0.6B/model.safetensors   --output_path ~/models/Qwen3-0.6B/qwen3-0.6B-cpu.mllm   --model_name qwen3   --format v2   --pipeline cast2fp32_pipeline   --verbose

adb -P $PORT -s $DEVICE shell '
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp
export ADSP_LIBRARY_PATH=/data/local/tmp
echo "Hello, introduce yourself." | ./mllm-qwen3-runner \
  -m qwen3-0.6B-cpu.mllm \
  -mv v2 \
  -t tokenizer.json \
  -c config_0.6B_cpu.json
'


DEVICE=d00e762e
PORT=5041

ANDR_LIB=$QNN_SDK_ROOT/lib/aarch64-android
OP_PATH=~/mllm/mllm/backends/qnn/custom-op-package/LLaMAPackage/build
LIBOMP=/mnt/data/chihao/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so

BIN_DIR=~/mllm/build-android-arm64-v8a-qnn/bin
MODEL=~/models/Qwen3-0.6B/qwen3-0.6B-cpu.mllm
TOKENIZER=~/models/Qwen3-0.6B/tokenizer.json
CFG=~/mllm/examples/qwen3/config_0.6B_cpu.json

adb -P $PORT -s $DEVICE shell 'mkdir -p /data/local/tmp'

adb -P $PORT -s $DEVICE push $MODEL /data/local/tmp/qwen3-0.6B-cpu.mllm
adb -P $PORT -s $DEVICE push $TOKENIZER /data/local/tmp/tokenizer.json
adb -P $PORT -s $DEVICE push $CFG /data/local/tmp/config_0.6B_cpu.json

adb -P $PORT -s $DEVICE push $BIN_DIR/mllm-qwen3-runner /data/local/tmp/
adb -P $PORT -s $DEVICE push $BIN_DIR/*.so /data/local/tmp/
adb -P $PORT -s $DEVICE push $LIBOMP /data/local/tmp/

adb -P $PORT -s $DEVICE shell '
cd /data/local/tmp
chmod +x ./mllm-qwen3-runner
export LD_LIBRARY_PATH=/data/local/tmp
export ADSP_LIBRARY_PATH=/data/local/tmp
./mllm-qwen3-runner --help
'
