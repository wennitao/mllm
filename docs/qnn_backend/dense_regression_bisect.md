# Dense SHA regression on npu_kernel — investigation log (2026-05-18)

## ROOT CAUSE — confirmed 2026-05-19

**The bug:** compile-time `CL` (KV cache length) and runtime `config.context_len` disagree, causing tensor-shape mismatch → garbage outputs (`**wapwap...` / `**npnp...`) and `err 1100` on decode dispatches.

- **Compile-time:** `compile_sha.cpp` line 46 hardcodes `int CL = 1024;`
- **Runtime (after `170f1e16`):** `aot_run.cpp` line 36 reads `config.context_len = qwen3_cfg.max_cache_length;`
- **Config:** `examples/qwen3_qnn_aot/config_1.7B.json` has `"max_cache_length": 2048`

So the **bin is compiled for 1024-position KV cache** but the runner allocates **2048-position KV cache buffers** at runtime, then passes them to the graph that expects 1024. → past_key/past_value buffer shape mismatch → undefined results → `**wapwap...` / `**npnp...` garbage and HTP execute errors.

**Why main works:** main's `aot_run.cpp` hardcodes `config.context_len = 1024;` (line 35 in `git show main:examples/qwen3_qnn_aot/aot_run.cpp`), which coincidentally matches the compile-time `CL = 1024`. Commit `170f1e16 "enable profiling"` (Mar 11) changed this to read from config:

```diff
-  config.context_len = 1024;
+  config.context_len = qwen3_cfg.max_cache_length;
```

That single line is the entire regression.

**Verified fix:** push `config_1.7B.json` with `max_cache_length: 1024` to device, npu_kernel runner produces `The capital of France is Paris.` (coherent) ✓

**Same bug exists in other runners** (all read `config.context_len = qwen3_cfg.max_cache_length`):
- `examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal.cpp:42` — explains the mono blocksparse-causal garbage documented in [split_prefill.md](split_prefill.md)
- `examples/qwen3_qnn_aot/aot_run_sha_blocksparse_causal_split.cpp:63` — explains the split-prefill `token 271 (\n\n)` symptom
- `examples/qwen3_qnn_aot/aot_run_blocksparse_causal.cpp:62` hardcodes 1024 ← unaffected

The matching `compile_*.cpp` drivers all hardcode `CL = 1024`:
- `compile_sha.cpp:46`
- `compile_sha_blocksparse_causal.cpp:91`
- (compile_sha_blocksparse_causal_split.cpp uses `CL = 1024` indirectly via shape declarations)

So the **entire investigation** in [split_prefill.md](split_prefill.md) about PTQ scale calibration, mask wiring, top-k gather, structural delta analysis — all of it was chasing a phantom. The real bug was a 1024 vs 2048 mismatch in one line of each runner.

### Fix applied (2026-05-19)

**Option 2 chosen — config drives both compile and runtime.** All hardcoded `1024`s in the AOT compile and run paths were replaced with `model_cfg.max_cache_length` / `qwen3_cfg.max_cache_length`:

- [`examples/qwen3_qnn_aot/compile_sha.cpp:46-59`](../../examples/qwen3_qnn_aot/compile_sha.cpp#L46-L59) — `int CL = 1024;` → `const int CL = model_cfg.max_cache_length;` (declared after config load).
- [`examples/qwen3_qnn_aot/compile_sha_blocksparse_causal.cpp:91-96`](../../examples/qwen3_qnn_aot/compile_sha_blocksparse_causal.cpp#L91-L96) — same pattern, CL moved past `model_cfg` construction.
- [`examples/qwen3_qnn_aot/compile_fp16_blocksparse_causal.cpp:108-116`](../../examples/qwen3_qnn_aot/compile_fp16_blocksparse_causal.cpp#L108-L116) — same.
- [`examples/qwen3_qnn_aot/aot_run_blocksparse_causal.cpp:62`](../../examples/qwen3_qnn_aot/aot_run_blocksparse_causal.cpp#L62) — `config.context_len = 1024;` → `config.context_len = qwen3_cfg.max_cache_length;`.

After rebuilding the x86 AOT compiler, recompiling `qwen3-1.7B-lpbq-sha.bin` (now 1619099648 bytes at CL=2048, was 1618366464 at CL=1024), and rebuilding the Android libs/runner, the npu_kernel-built dense path on device produces:

```
The capital of France is Paris.<|im_end|>
```

with no config workaround. Same fix applies transitively to mono blocksparse-causal and the split path — they were already reading `qwen3_cfg.max_cache_length` from config; recompiling their bins with the new compile-time `CL = max_cache_length` aligns the compile/runtime contract.

### End-to-end verification across the four AOT variants (2026-05-19)

After the CL fix, each variant was recompiled (where applicable), pushed to the SM8650/V79 device, and tested with prompt "What is the capital of France?". Result summary:

| Variant | Compile | Runtime | Output | Notes |
|---|---|---|---|---|
| Dense SHA (`qwen3-1.7B-lpbq-sha.bin`) | ✅ CL=2048, 1619099648 B | ✅ | **"The capital of France is Paris.\<\|im_end\|>"** | CL fix resolves it end-to-end. |
| Mono blocksparse-causal (`qwen3-lpbq-sha-blocksparse-causal.bin`) | ✅ CL=4096, 1612468224 B (after both fixes) | ✅ | **"The capital of France is Paris.\<\|im_end\|>"** | Required CL fix + rotary fix. The earlier "Arabic gibberish" was caused by `bakeRotaryEmbeddings` writing fp16 bytes into a slot that `QDQ_ROPE` then reinterprets as uint16 with scale 1/32768, zp 32768. See [split_prefill.md § "Tap-point bisect (2026-05-19) — root cause"](split_prefill.md#tap-point-bisect-2026-05-19--root-cause). |
| Split blocksparse-causal (`qwen3-lpbq-sha-blocksparse-causal-split.bin`) | ✅ L=4: 935149568 B, 9 graphs. L=28: 1638563840 B, 57 graphs | ⚠️ L=4 runs but model lobotomized; L=28 hits PD memory cap (~4.2GB > ~3-4GB cap) | L=4: `/topics` (expected from 4-layer slice); L=28: load fails | Rotary fix applied (same as mono). L=28 PD memory blocker is a separate pre-existing issue per [split_prefill.md § "HTP PD memory cap"](split_prefill.md#htp-pd-memory-cap-blocker-and-the-sq256-workaround). |
| fp16 blocksparse-causal (`qwen3-fp16-blocksparse-causal.bin`) | ❌ aborts in `LLMQuantRecipePass` `embedding` pattern: `Failed at pass: embedding on op(ptr): model.embed_tokens` | n/a | n/a | Pre-existing compile-time blocker unrelated to CL or rotary. `LLMQuantRecipeEmbeddingPattern::rewrite` ([LLMQuantRecipePass.cpp:1023](../../mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp#L1023)) requires `weight_tensor->tensor_.dtype() == kUInt16 \|\| kUInt16PerTensorAsy`, but `qwen3_1.7b_fp16.mllm` ships an fp16 `embed_tokens.weight`. The same hardcoded check exists in the RMSNorm/Linear patterns and the visitor stack — making fp16 work end-to-end requires ~5+ correlated patches. Out of scope; mono/split work without it now that the rotary bug is fixed. |

**Conclusion.** The CL fix resolves the dense SHA regression cleanly. Mono and split block-sparse variants compile/run after the fix but still produce incoherent text from a separate, pre-existing correctness bug in the per-qb block-sparse attention pipeline (PTQ calibration / per-qb gather / mask wiring — investigation tracked in [split_prefill.md](split_prefill.md)). The fp16 path has its own pre-existing compile-time blocker in `LLMQuantRecipePass`, also unrelated to CL.

**Reproduction commands** (executed from `/data/local/tmp/split_test/` after push):

```bash
# Dense
LD_LIBRARY_PATH=.:/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
  ./mllm-qwen3-aot-runner -m qwen3-1.7B-lpbq-sha.bin -c config_1.7B.json -t tokenizer.json

# Mono blocksparse-causal
LD_LIBRARY_PATH=.:/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
  ./mllm-qwen3-aot-sha-blocksparse-runner -m qwen3-lpbq-sha-blocksparse-causal.bin \
    -c config_1.7B_w4a16_blocksparse_causal.json -t tokenizer.json

# Split (decode-by-reprefill)
LD_LIBRARY_PATH=.:/data/local/tmp ADSP_LIBRARY_PATH=/data/local/tmp \
  ./mllm-qwen3-aot-sha-blocksparse-split-runner -m qwen3-lpbq-sha-blocksparse-causal-split.bin \
    -c config_1.7B_w4a16_blocksparse_causal_4L.json -t tokenizer.json --sq 1024 --gen 16
```

The split bin was compiled with:
```
./build-qnn-aot/bin/mllm-qwen3-aot-sha-blocksparse-causal-split-c \
  -m Qwen3-1.7b-mllm/qwen3_1.7b_ptq_lpbq.mllm \
  -c examples/qwen3_qnn_aot/config_1.7B_w4a16_blocksparse_causal_4L.json \
  -aot_cfg examples/qwen3_qnn_aot/qnn_aot_cfg_1.7B_w4a16_blocksparse_causal_split_4L.json \
  -e /mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/
```
→ `qwen3-lpbq-sha-blocksparse-causal-split.bin` (934100992 bytes, 9 graphs: chunk_0…chunk_4 + attn_0…attn_3).

---

## Original investigation log (now superseded by root cause above)



## Symptom

On `npu_kernel` branch, both dense SHA and mono blocksparse-causal produce incoherent output:
- Dense top-1 = `**` (token 334) for every prompt ("Hi", "The capital of France is", "Hello, my name is", etc.)
- Decode fails with `QnnDsp <E> Graph model.0.s1 failed in execution with err 1100` on every step
- Mono blocksparse-causal: top-1 = `<|im_end|>` (151645), garbage continuation

On `main` branch with the same device + same .mllm + same QNN libs + same skel:
- Dense SHA produces "The capital of France is Paris.<|im_end|>" (coherent)

## Definitive isolation

| Test | Result | Implication |
|---|---|---|
| md5 of `qwen3-1.7B-lpbq-sha.bin` compiled on main vs npu_kernel | **identical** (`94a69fd541256b986e9f1a9a6f6e09b0`) | Compile pipeline produces bit-identical bin → bug is NOT in the bin |
| md5 of `qwen3_qnn_aot_sha_32.mir` and `qwen3_qnn_aot_sha_1.mir` | identical between branches | IR lowering is deterministic |
| npu_kernel bin + main's Android libs on same device | **"Paris" coherent** | The bin is fine when run against main's libs |
| npu_kernel bin + npu_kernel's Android libs on same device | **`**npnpnp...` garbage** | npu_kernel libs are the regression |

**Conclusion**: the bug is in npu_kernel's Android runtime libs (`libMllmQNNBackend.so` / `libMllmRT.so`).

## Working artefact locations

- Main's working libs (Android): `/tmp/mllm-main-test/build-android-arm64-v8a-qnn/bin/`
  - `libMllmQNNBackend.so` 26157248 bytes md5=`812a9721b77f1797bf776d28dd4dc2e1`
  - `libMllmRT.so` 145003488 bytes
  - `libMllmCPUBackend.so` 52262032 bytes
  - `mllm-qwen3-aot-runner` 3381600 bytes
- Main worktree compile-time bin (md5 match with npu_kernel): `/tmp/mllm-main-test/qwen3-1.7B-lpbq-sha.bin`
- Main's QNNBackend.cpp diff vs HEAD: `git diff main HEAD -- mllm/backends/qnn/QNNBackend.cpp` shows 222 lines of profile event logging additions (probably benign)

## Paths investigated and ruled out

### Ruled out
1. **Custom op package additions (FlashAttention*, SoftmaxBlockSparse*, REGISTER_PACKAGE_OPTIMIZATIONS)** — stashed the uncommitted XML + Interface.cpp diffs, rebuilt `libQnnLLaMAPackage.so` for v79, pushed to device. Logits md5 identical to with-new-ops version → no effect on dense output.
2. **My `MLLM_DUMP_LOGITS` edit in `PromptProcessor.cpp`** — stashed PromptProcessor.cpp only, rebuilt Android `libMllmQNNBackend.so`. Dense still broken with same `**` top-1.
3. **All uncommitted `aot_rt/` source changes** (PromptProcessor.cpp + QnnAOTConfig.hpp + QnnAOTModule.cpp + the new BlockSparse* / ShaBlockSparse* processors) — stashed with `--include-untracked`, rebuilt + ran. Still broken.
4. **`MLLM_PERFETTO_ENABLE=ON`** — reconfigured npu_kernel build with `MLLM_PERFETTO_ENABLE=OFF`, rebuilt, pushed. Still broken (`**npnpnp...`). Perfetto is not the cause.
5. **V79 skel** — `/data/local/tmp/libQnnHtpV79Skel.so` already correct (10588380 bytes, dated 2026-01-30). Confirmed via stat.
6. **Bin compilation differences** — md5 match between main-compiled and npu_kernel-compiled bin (see Definitive isolation table).

### Under test
- **`profilingLevel_ = ProfilingLevel::OFF` (npu_kernel) vs `DETAILED` (main)** — flipped to DETAILED, rebuilt, pushing now. Result: pending.

### Not yet tested
- **Specific commit bisect** of the 9 commits between `10d3d6a4` (main) and `9d983b9b` (npu_kernel HEAD):
  - `9d983b9b npu kernel testing` — adds FlashAttention custom op (DSP-side .so, mostly ruled out via #1)
  - `4a0c68bc flashattention opencl backend: generation not correct` — OpenCL only
  - `d745559a CPU Decode Topk Mask` — adds TopKMask op (additive enum, should be safe)
  - `ae0ea885 q4 opencl backend pass` — OpenCL only
  - `32ede3e3 update README` — docs only
  - `e4abd87a Prefill on NPU passed` — changes `profilingLevel_ = OFF` (under test above)
  - `2f54711c fix gguf q4 quantize` — adds profile event dump logging
  - `170f1e16 enable profiling` — changes `htp_arch: V75 → V79` in aot_cfg_1.7B.json + adds perfetto::start/stop in aot_run.cpp
  - `803262f6 Build for Snapdragon 8 Elite` — host-side x86 lib path code

### Suspicions ordered by likelihood (current)
1. **`profilingLevel_ = OFF`** (commit `e4abd87a`) — main has DETAILED, npu_kernel has OFF; passing nullptr `profileHandle` to `graphExecute` might trigger some HTP-side issue. Pending test.
2. Some interaction between npu_kernel's NEW custom ops being LINKED into libQnnLLaMAPackage.so (even if not invoked) and existing op dispatch. Partial test in #1 (custom op revert) but only the Interface.cpp/XML were reverted — the new .cpp files were still compiled in.
3. The QNNBackend.cpp profile event dump code (`extractBackendProfilingInfo`) somehow has side effects.

## Recipe to reproduce (broken state)

```bash
cd /mnt/raid0_ssd/wentao/mllm  # on npu_kernel branch
python task.py tasks/build_android_qnn.yaml
# After build, push libs to device:
adb push build-android-arm64-v8a-qnn/bin/libMllmQNNBackend.so /data/local/tmp/split_test/
adb push build-android-arm64-v8a-qnn/bin/libMllmRT.so /data/local/tmp/split_test/
adb push build-android-arm64-v8a-qnn/bin/libMllmCPUBackend.so /data/local/tmp/split_test/
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner /data/local/tmp/split_test/
adb push qwen3-1.7B-lpbq-sha.bin /data/local/tmp/split_test/

# Run:
adb shell 'cd /data/local/tmp/split_test && export LD_LIBRARY_PATH=.:/data/local/tmp \
  && export ADSP_LIBRARY_PATH=.:/data/local/tmp \
  && echo "The capital of France is" | ./mllm-qwen3-aot-runner \
       -m qwen3-1.7B-lpbq-sha.bin -c config_1.7B.json -t tokenizer.json --ar_len 32'
# Expected (broken): "**npnpnpnp..." + err 1100 on decode
```

## Recipe to reproduce (working state)

```bash
# Build main's libs in a worktree:
git worktree add /tmp/mllm-main-test main
cd /tmp/mllm-main-test
# Copy submodules from main checkout (avoids slow `git submodule update --init`)
for d in third_party/*/; do [ -d "/mnt/raid0_ssd/wentao/mllm/$d" ] && \
  cp -r "/mnt/raid0_ssd/wentao/mllm/$d." "/tmp/mllm-main-test/$d" 2>/dev/null; done
# Patch the LLaMAPackage Makefile to auto-detect Hexagon Tools version:
cp /mnt/raid0_ssd/wentao/mllm/mllm/backends/qnn/custom-op-package/LLaMAPackage/Makefile \
   /tmp/mllm-main-test/mllm/backends/qnn/custom-op-package/LLaMAPackage/Makefile
# Configure + build (skip the HexagonMakeTask):
ANDROID_NDK_PATH=/mnt/raid0_ssd/wentao/android-ndk-r29 \
ANDROID_NDK_ROOT=/mnt/raid0_ssd/wentao/android-ndk-r29 \
  cmake -B build-android-arm64-v8a-qnn -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/mnt/raid0_ssd/wentao/android-ndk-r29/build/cmake/android.toolchain.cmake \
    -DCMAKE_MAKE_PROGRAM=/mnt/raid0_ssd/wentao/miniconda3/envs/mllm/bin/ninja \
    -DMLLM_BUILD_ARM_BACKEND=ON -DMLLM_BUILD_QNN_BACKEND=ON \
    -DANDROID_PLATFORM=android-28 -DANDROID_ABI=arm64-v8a \
    "-DMLLM_CPU_BACKEND_COMPILE_OPTIONS=-march=armv8.2-a+fp16+fp16fml+dotprod+i8mm;-ffast-math;-Wno-nan-infinity-disabled" \
    -DMLLM_KERNEL_USE_THREADS=ON -DMLLM_KERNEL_THREADS_VENDOR_OPENMP=ON \
    -DMLLM_KERNEL_USE_THREADS_VENDOR_MLLM=OFF
cmake --build build-android-arm64-v8a-qnn --target mllm-qwen3-aot-runner -j 8

# Push main's libs to device:
adb push build-android-arm64-v8a-qnn/bin/libMllmQNNBackend.so /data/local/tmp/split_test/
adb push build-android-arm64-v8a-qnn/bin/libMllmRT.so /data/local/tmp/split_test/
adb push build-android-arm64-v8a-qnn/bin/libMllmCPUBackend.so /data/local/tmp/split_test/
adb push build-android-arm64-v8a-qnn/bin/mllm-qwen3-aot-runner /data/local/tmp/split_test/

# Run with SAME npu_kernel-compiled bin → coherent "Paris" output
```

## Bisect status

`git bisect` between main and npu_kernel should isolate the single commit responsible. Suggested first cut:
- Test at `e4abd87a` (profile OFF change) — under test now
- If profile flip doesn't fix, test halfway: `170f1e16` (the V79+perfetto enable commit)

## Bisect progress (2026-05-18 21:00–21:40)

Iterative tests on /tmp/mllm-main-test worktree by progressively applying npu_kernel changes:

| Test | Source state | Result | Conclusion |
|---|---|---|---|
| T0 | main pristine (10d3d6a4) | "Paris" | baseline working |
| T1 | main + d745559a's TopKMask files copied in | "Paris" | TopKMask is NOT the bug |
| T2 | T1 + uncommitted aot_rt/* + aot/passes/* + QNNBackend.hpp from npu_kernel | "Paris" | uncommitted aot_rt is NOT the bug |
| T3 | T2 + npu_kernel HEAD's committed QNNBackend.cpp | "Paris" | QNNBackend.cpp profile event dump is NOT the bug |
| **npu_kernel build** | full npu_kernel checkout (HEAD + uncommitted) | `**npnpnp...` (broken) | **The combination of all npu_kernel state breaks dense** |

After T3, applied EVERYTHING I could find from npu_kernel to main worktree → still works. So the bug is in some file I haven't yet copied, OR it's a build-environment / stale .o issue.

Currently running: clean rebuild of npu_kernel libs (`rm -rf build-android-arm64-v8a-qnn && python task.py tasks/build_android_qnn.yaml`) to rule out stale .o files.

If clean rebuild produces working libs: → it was a stale build issue, and the fix is just `rm -rf build-android-arm64-v8a-qnn`.

If clean rebuild produces broken libs: → there's some file I haven't applied to main worktree. Candidates I haven't tried yet:
- `mllm/backends/cpu/CPUBackend.cpp` (committed change)
- `mllm/backends/cpu/kernels/common/ggml/quantize/quantize.hpp` (committed)
- `mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q8.cpp` (committed)
- `mllm/core/aops/ConcatOp.cpp` (uncommitted, adds error reporting only)
- `mllm/models/qwen3/modeling_qwen3_opencl.hpp` (uncommitted, OpenCL only)
- `mllm/models/qwen3/modeling_qwen3.hpp` (committed, used by tracing models)
- `mllm/models/qwen3/modeling_qwen3_fa2.hpp` (committed)
- `mllm/nn/lmcache/StaticCache.cpp` (committed, OpenCL FA2 change)

## Logits comparison data

| Variant | Bin md5 | Lib md5 | Logits top-1 | Top-5 |
|---|---|---|---|---|
| npu_kernel + npu_kernel libs (today's build) | 94a69fd5 | 315f5f63 | 334 (`**`) | `**`, `âĢĭâĢĭ`, `mes`, `mo`, `bo` |
| npu_kernel + main's libs | 94a69fd5 | 812a9721 | (Paris flow, no dump because lib lacks MLLM_DUMP_LOGITS) | coherent text |
| npu_kernel + may15-install lib | 94a69fd5 | 75533ca9 | (broken, no dump) | `**npnpnp...` |

