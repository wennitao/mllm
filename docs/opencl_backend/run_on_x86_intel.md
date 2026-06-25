# Running the OpenCL backend on x86 + Intel iGPU

This documents how to build and run the OpenCL FlashAttention benches **and**
the Qwen3 OpenCL runner on a plain x86-64 Linux box with an **Intel integrated
GPU** (verified on Ubuntu 22.04, Intel Core i7-13700E, UHD Graphics `0xa780`).

The repo's OpenCL kernels were tuned for Qualcomm Adreno; everything below is
the host/Intel-GPU port. The OpenCL backend `dlopen`s `libOpenCL.so` at runtime
(no link-time OpenCL dependency), so the same binaries work on any installed ICD.

---

## 1. One-time system setup (needs sudo)

```bash
# OpenCL runtime for the Intel iGPU + ICD loader + clinfo
sudo apt-get install -y intel-opencl-icd ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo

# Build tools: the repo needs CMake >= 3.25 (Ubuntu 22.04 ships 3.22) and Ninja
sudo apt-get install -y ninja-build
pip3 install --user "cmake>=3.25"          # installs to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"        # put the new cmake first (add to ~/.bashrc)
```

### Give yourself GPU access

The Intel render node is group-restricted. Without this, `clinfo` reports
**0 platforms** even though the runtime is installed.

```bash
sudo usermod -aG render "$USER"
# Either log out/in, OR prefix GPU commands with `sg render -c "..."` for this session.
```

Verify the device is visible:

```bash
sg render -c "clinfo -l"
# Expect: Platform #0: Intel(R) OpenCL HD Graphics
#          `-- Device #0: Intel(R) Graphics [0xa780]
```

---

## 2. Submodules

Only a handful are needed for the OpenCL benches / Qwen3 runner (skip the huge
`update=none` ones like llvm/cutlass):

```bash
cd /home/chihao/mllm
git submodule update --init third_party/fmt third_party/xxHash \
    third_party/flatbuffers third_party/googletest third_party/benchmark
git submodule update --init --recursive mllm/ffi/vendors/tvm-ffi   # pulls libbacktrace + dlpack
```

---

## 3. Build

A ready-made task recipe is included: [`tasks/build_x86_opencl.yaml`](../../tasks/build_x86_opencl.yaml).
It enables the OpenCL backend, sets `-march=native`, disables tests/benchmarks,
and builds the FA benches + both Qwen3 runners.

```bash
export PATH="$HOME/.local/bin:$PATH"        # ensure cmake>=3.25 is used
python3 task.py tasks/build_x86_opencl.yaml
```

Output binaries land in `build-x86-opencl/bin/`:

| Binary | Purpose |
|---|---|
| `mllm-fa-opencl-bench` | FlashAttention op bench (fused + two-pass + decode), fp16/fp32 |
| `mllm-fa-twopass-bench` | Standalone two-pass GEMM FA, validated vs a CPU fp32 reference |
| `mllm-qwen3-opencl-runner` | Qwen3 chat on the OpenCL/GPU backend |
| `mllm-qwen3-runner` | Qwen3 chat on the CPU backend |

> All GPU runs below are wrapped in `sg render -c "..."` and set
> `LD_LIBRARY_PATH=.` so the runtime finds the `libMllm*.so` next to the binary.

---

## 4. Run the FlashAttention benches

```bash
BIN=/home/chihao/mllm/build-x86-opencl/bin

# Two-pass GEMM FA, validated against a CPU fp32 reference (ground truth):
sg render -c "cd $BIN && LD_LIBRARY_PATH=. ./mllm-fa-twopass-bench --reps 3"

# Full FA op bench (fp16). Note: its built-in fp16-vs-fp32 *kernel* check is not
# a ground-truth check — see the caveat below.
sg render -c "cd $BIN && LD_LIBRARY_PATH=. FA_DTYPE=fp16 ./mllm-fa-opencl-bench"
```

> If you paste the `sg render` line by itself, set `BIN` first (or inline the
> absolute path). `sg` starts a fresh shell, so don't rely on `$PWD` or your
> current directory — always pass absolute paths.

**Results on the Intel UHD `0xa780`:** the two-pass GEMM prefill is numerically
correct (`max_abs ≈ 2.1e-4` vs CPU fp32, matching the Adreno reference) but runs
at only **~23-26 GF/s** (vs 529-985 GF/s on Adreno) — the Adreno tuning doesn't
transfer to Intel's EU architecture.

**Caveat:** `mllm-fa-opencl-bench`'s internal validation compares the fp16 path
against the **fp32 *fused* kernel**, and that fused kernel is itself incorrect on
Intel — so it reports `max_abs ≈ 1.0`. That's a broken *reference*, not a broken
two-pass path. Trust `mllm-fa-twopass-bench` (CPU ground truth) instead.

---

## 5. Run Qwen3 on the OpenCL backend

### 5a. Get a complete model + tokenizer

The OpenCL backend supports **fp32** and **GGUF Q4_0** weights only (no Q4_K).
mllmTeam hosts Qwen3-0.6B; use the **fp32** build for the GPU.

```bash
cd /home/chihao/mllm/qwen3_cpu

# Model (fp32, ~3.0 GB) — runs on both CPU and OpenCL:
curl -sL -o qwen-3-0.6b-fp32.mllm \
  "https://huggingface.co/mllmTeam/qwen-3-0.6b-mllm/resolve/main/qwen-3-0.6b-fp32.mllm"

# Tokenizer (standard HF tokenizer.json):
curl -sL -o tokenizer.json \
  "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
```

Config files for the 0.6B are included next to the model:
[`config_0.6B_fp32.json`](../../qwen3_cpu/config_0.6B_fp32.json) (`linear_impl_type: Default`, for fp32)
and [`config_0.6B_gguf.json`](../../qwen3_cpu/config_0.6B_gguf.json) (`linear_impl_type: GGUF`, for Q4_0/Q4_K on CPU).

### 5b. Run on the GPU

Interactive (type a prompt, then `exit`):

```bash
sg render -c "cd /home/chihao/mllm/build-x86-opencl/bin && LD_LIBRARY_PATH=. ./mllm-qwen3-opencl-runner \
  -m /home/chihao/mllm/qwen3_cpu/qwen-3-0.6b-fp32.mllm -mv v1 \
  -t /home/chihao/mllm/qwen3_cpu/tokenizer.json \
  -c /home/chihao/mllm/qwen3_cpu/config_0.6B_fp32.json"
# Expected: "Response: The capital of France is **Paris**."
```

Non-interactive (pipe the prompt in):

```bash
printf 'What is the capital of France?\nexit\n' | \
sg render -c "cd /home/chihao/mllm/build-x86-opencl/bin && LD_LIBRARY_PATH=. ./mllm-qwen3-opencl-runner \
  -m /home/chihao/mllm/qwen3_cpu/qwen-3-0.6b-fp32.mllm -mv v1 \
  -t /home/chihao/mllm/qwen3_cpu/tokenizer.json \
  -c /home/chihao/mllm/qwen3_cpu/config_0.6B_fp32.json"
```

### 5c. CPU reference (optional, accepts Q4_K too)

```bash
cd /home/chihao/mllm/build-x86-opencl/bin
LD_LIBRARY_PATH=. ./mllm-qwen3-runner \
  -m /home/chihao/mllm/qwen3_cpu/qwen-3-0.6b-q4_k.mllm -mv v1 \
  -t /home/chihao/mllm/qwen3_cpu/tokenizer.json \
  -c /home/chihao/mllm/qwen3_cpu/config_0.6B_gguf.json --max_new_tokens 30
# Q4_K is CPU-only; the CPU runner needs no `sg render` wrapper.
```

Runner flags: `-m` model, `-mv` version (`v1`/`v2`), `-t` tokenizer.json,
`-c` config, `--max_new_tokens N` (CPU runner), `--seq_len N` (synthetic prefill,
bypasses the tokenizer).

---

## Notes / gotchas

- **`-mv` must match the file.** The 0.6B mllmTeam models are **v1**. A wrong
  version aborts at load. Inspect a `.mllm` with:
  `LD_LIBRARY_PATH=build-x86-opencl/bin build-x86-opencl/bin/mllm-params-inspector -i model.mllm -iv v2`
- **Truncated model = SIGSEGV in `CPUEmbeddingOp`.** If a `.mllm` is much smaller
  than the sum of its tensor sizes (the inspector prints shapes/dtypes), the
  weight data is missing and the first embedding read segfaults. Re-download.
- **The OpenCL Qwen3 model uses matmul attention, not the FlashAttention
  kernels** ([modeling_qwen3_opencl.hpp:122-134](../../mllm/models/qwen3/modeling_qwen3_opencl.hpp#L122-L134),
  `use_fa2=false`). The FA benches in §4 exercise the FA op standalone; they do
  not affect Qwen3's output correctness.
- **Performance** on the iGPU is modest (fp32 0.6B: prefill ~23 tok/s, decode
  ~4 tok/s) — correct, but the kernels are untuned for Intel.
