PORT=5041
DEVICE=eb49fb9d
REMOTE_DIR=/data/local/tmp/qwen3_aot

# ============================================================
# Source paths
# ============================================================
MLLM_ROOT=/home/chihao/mllm
QNN_SDK_ROOT=/mnt/data/chihao/qairt/2.44.0.260225
ANDR_LIB=$QNN_SDK_ROOT/lib/aarch64-android
HEX_LIB=$QNN_SDK_ROOT/lib/hexagon-v79/unsigned
OP_PATH=$MLLM_ROOT/mllm/backends/qnn/custom-op-package/LLaMAPackage/build
LIBOMP=/mnt/data/chihao/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so

COMPILE=$MLLM_ROOT/build-qnn-aot/bin/mllm-qwen3-aot-sha-c
RUNNER=$MLLM_ROOT/build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner
RUNNER_DIR=$MLLM_ROOT/build-android-arm64-v8a-qnn/bin

MLLM_MODEL=/mnt/data/chihao/output/qwen3_1.7b.mllm     # source .mllm for compile
CONFIG=$MLLM_ROOT/examples/qwen3_qnn_aot/config_1.7B.json
AOT_CFG=$MLLM_ROOT/examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B.json
TOKENIZER=/mnt/data/chihao/models/Qwen3-1.7B/tokenizer.json

# ============================================================
# Pick the context_len for this run. The .bin will be specific
# to this CL. Must match the runner's --context_len at runtime.
# ============================================================
CL=1024                                                    # change as needed
BIN_NAME=qwen3-1.7B-lpbq-sha-ctx${CL}.bin
LOCAL_BIN=/mnt/data/chihao/$BIN_NAME
REMOTE_BIN=$BIN_NAME

# ============================================================
# STEP 1 — Compile the AOT context (.bin) on the host
#   - Skips if the .bin already exists locally
#   - Sources QAIRT env so QNN tools resolve their libs
#   - Takes ~10-15 min per CL
# ============================================================
if [ ! -f "$LOCAL_BIN" ]; then
    source $QNN_SDK_ROOT/bin/envsetup.sh
    $COMPILE \
        -m $MLLM_MODEL \
        -c $CONFIG \
        --aot_config $AOT_CFG \
        --context_len $CL \
        --ar_len 32 \
        -o $LOCAL_BIN
fi
ls -lh $LOCAL_BIN

# ============================================================
# STEP 2 — Push the QNN stack + the compiled .bin to the device
#   (push QNN libs only the first time; runner + bin every time)
# ============================================================
adb -P $PORT -s $DEVICE shell "mkdir -p $REMOTE_DIR"

# AOT context + tokenizer + config
adb -P $PORT -s $DEVICE push $LOCAL_BIN $REMOTE_DIR/$REMOTE_BIN
adb -P $PORT -s $DEVICE push $TOKENIZER $REMOTE_DIR/qwen3-tokenizer.json
adb -P $PORT -s $DEVICE push $CONFIG    $REMOTE_DIR/config_1.7B.json

# QNN common libs (skip after first push)
for f in libQnnSystem.so libQnnHtp.so libQnnHtpPrepare.so \
         libQnnHtpProfilingReader.so libQnnHtpOptraceProfilingReader.so \
         libQnnHtpV79Stub.so libQnnHtpV79CalculatorStub.so; do
    adb -P $PORT -s $DEVICE push $ANDR_LIB/$f $REMOTE_DIR/
done
adb -P $PORT -s $DEVICE push $HEX_LIB/libQnnHtpV79Skel.so $REMOTE_DIR/

# Custom op packages (skip after first push)
adb -P $PORT -s $DEVICE push $OP_PATH/aarch64-android/libQnnLLaMAPackage.so \
    $REMOTE_DIR/libQnnLLaMAPackage_CPU.so
adb -P $PORT -s $DEVICE push $OP_PATH/hexagon-v79/libQnnLLaMAPackage.so \
    $REMOTE_DIR/libQnnLLaMAPackage_HTP.so

# MLLM runner + libs + libomp
adb -P $PORT -s $DEVICE push $RUNNER_DIR/*.so $REMOTE_DIR/
adb -P $PORT -s $DEVICE push $RUNNER          $REMOTE_DIR/
adb -P $PORT -s $DEVICE push $LIBOMP          $REMOTE_DIR/

adb -P $PORT -s $DEVICE shell "chmod +x $REMOTE_DIR/mllm-qwen3-aot-runner"

# ============================================================
# STEP 3 — Run. Pick ONE of A / B / C below.
# IMPORTANT: --context_len $CL must match the .bin's compile-time CL
# ============================================================

# A) Real prompt via echo (uses chat template, prefill_tokens = ~30 + text)
adb -P $PORT -s $DEVICE shell "
cd $REMOTE_DIR &&
rm -f qnn_profile.csv qwen3.perfetto &&
export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH='$REMOTE_DIR;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp' &&
echo 'What is UCSD?' | ./mllm-qwen3-aot-runner \
-m $REMOTE_BIN \
-t qwen3-tokenizer.json \
-c config_1.7B.json \
--ar_len 32 \
--context_len $CL \
--max_new_tokens 30 \
--quiet 2>/dev/null
"

# B) Real prompt from a file (e.g., NIAH)
# adb -P $PORT -s $DEVICE push ~/mllm/NIAH1_1k.txt $REMOTE_DIR/NIAH1_1k.txt
# adb -P $PORT -s $DEVICE shell "
# cd $REMOTE_DIR && rm -f qnn_profile.csv &&
# export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH &&
# export ADSP_LIBRARY_PATH='$REMOTE_DIR;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp' &&
# ./mllm-qwen3-aot-runner -m $REMOTE_BIN \
# -t qwen3-tokenizer.json -c config_1.7B.json \
# --ar_len 32 --context_len $CL --max_new_tokens 40 --quiet < NIAH1_1k.txt 2>/dev/null
# "

# C) Synthetic input of exactly N tokens (for clean per-op profiling)
# Make sure SEQ + decode <= CL, and SEQ <= CL - 32
# SEQ=128
# adb -P $PORT -s $DEVICE shell "
# cd $REMOTE_DIR && rm -f qnn_profile.csv &&
# export LD_LIBRARY_PATH=$REMOTE_DIR:\$LD_LIBRARY_PATH &&
# export ADSP_LIBRARY_PATH='$REMOTE_DIR;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp' &&
# ./mllm-qwen3-aot-runner -m $REMOTE_BIN \
# -t qwen3-tokenizer.json -c config_1.7B.json \
# --ar_len 32 --context_len $CL \
# --seq_len $SEQ --max_new_tokens 30 < /dev/null 2>&1 | grep -vE 'QNN profile event' | tail -10
# "

# ============================================================
# STEP 4 — Pull per-op profile CSV (only if --quiet was OFF)
# ============================================================
adb -P $PORT -s $DEVICE pull $REMOTE_DIR/qnn_profile.csv \
    $MLLM_ROOT/qnn_profile_ctx${CL}.csv 2>/dev/null && \
    ls -lh $MLLM_ROOT/qnn_profile_ctx${CL}.csv
