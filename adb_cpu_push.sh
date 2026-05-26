PORT=5041
DEVICE=eb49fb9d
REMOTE_DIR=/data/local/tmp/qwen3_cpu

MLLM_ROOT=/home/chihao/mllm
RUNNER=$MLLM_ROOT/build-android-arm64-v8a-cpu/bin/mllm-qwen3-runner
BIN_DIR=$MLLM_ROOT/build-android-arm64-v8a-cpu/bin

MODEL=/mnt/data/chihao/qwen3-1.7B-q4.mllm
CONFIG=$MLLM_ROOT/examples/qwen3_npu/config_1.7B_q4.json
TOKENIZER=/mnt/data/chihao/models/Qwen3-1.7B/tokenizer.json
LIBOMP=/mnt/data/chihao/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so

TRACE_NAME=qwen3.perf
HOST_TRACE=$HOME/mllm/qwen3_cpu.perf

# ==============================
# Push (one-time per build)
# ==============================
adb -P $PORT -s $DEVICE shell "mkdir -p $REMOTE_DIR"
adb -P $PORT -s $DEVICE push $RUNNER                  $REMOTE_DIR/mllm-qwen3-runner
adb -P $PORT -s $DEVICE push $MODEL                   $REMOTE_DIR/qwen3-1.7B-q4.mllm
adb -P $PORT -s $DEVICE push $CONFIG                  $REMOTE_DIR/config_1.7B_q4.json
adb -P $PORT -s $DEVICE push $TOKENIZER               $REMOTE_DIR/qwen3-tokenizer.json
adb -P $PORT -s $DEVICE push $BIN_DIR/libMllm*.so     $REMOTE_DIR/
adb -P $PORT -s $DEVICE push $LIBOMP                  $REMOTE_DIR/libomp.so
adb -P $PORT -s $DEVICE shell "chmod +x $REMOTE_DIR/mllm-qwen3-runner"

# ==============================
# Run — choose ONE of the three input modes below:
# ==============================

# A) Real-text prompt via echo
adb -P $PORT -s $DEVICE shell "
cd $REMOTE_DIR &&
rm -f $TRACE_NAME &&
export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH &&
echo 'What is UCSD?' | ./mllm-qwen3-runner \
--model_path qwen3-1.7B-q4.mllm \
--model_version v2 \
--config_path config_1.7B_q4.json \
--tokenizer_path qwen3-tokenizer.json \
--perf_path $TRACE_NAME \
--max_new_tokens 40
"

# B) Real-text prompt from a file (push the file first)
# adb -P $PORT -s $DEVICE push ~/mllm/NIAH1_1k.txt $REMOTE_DIR/NIAH1_1k.txt
# adb -P $PORT -s $DEVICE shell "
# cd $REMOTE_DIR &&
# rm -f $TRACE_NAME &&
# export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH &&
# ./mllm-qwen3-runner \
# --model_path qwen3-1.7B-q4.mllm --model_version v2 \
# --config_path config_1.7B_q4.json --tokenizer_path qwen3-tokenizer.json \
# --perf_path $TRACE_NAME --max_new_tokens 40 < NIAH1_1k.txt
# "

# C) Synthetic input of exactly N tokens (no tokenizer/chat template)
# adb -P $PORT -s $DEVICE shell "
# cd $REMOTE_DIR &&
# rm -f $TRACE_NAME &&
# export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH &&
# ./mllm-qwen3-runner \
# --model_path qwen3-1.7B-q4.mllm --model_version v2 \
# --config_path config_1.7B_q4.json --tokenizer_path qwen3-tokenizer.json \
# --perf_path $TRACE_NAME --seq_len 128 --max_new_tokens 30 < /dev/null
# "

# ==============================
# Pull the Perfetto trace
# ==============================
adb -P $PORT -s $DEVICE pull $REMOTE_DIR/$TRACE_NAME $HOST_TRACE
ls -lh $HOST_TRACE
