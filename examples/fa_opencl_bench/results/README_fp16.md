# OpenCL FlashAttention — fp16 vs fp32 (Adreno 830, SM8750)

Profiling of the OpenCL FlashAttention **v1** kernel
(`flash_attention.cl` + `OpenCLFlashAttention2Op`) at fp16, alongside the
existing fp32 baseline. Same device (Adreno 830), same B=1 H=16 D=128 sweep.

The bench now selects dtype at runtime — `FA_DTYPE=fp16` or `FA_DTYPE=fp32`
(default fp32). The op dispatches to `flash_attention_fp16` vs
`flash_attention_fp32` by input tensor dtype.

## Result: fp16 buys nothing over fp32

Prefill (min latency, S_q=S_kv), fp32 vs fp16:

| S    | fp32      | fp16      | GF/s (both) |
|-----:|----------:|----------:|------------:|
| 128  | 12.57 ms  | 12.23 ms  | ~5.5        |
| 256  | 48.75 ms  | 48.03 ms  | ~5.6        |
| 1024 | 801 ms    | 765 ms    | ~5.5        |
| 2048 | 3282 ms   | 3119 ms   | ~5.5        |

Decode (min latency, S_q=1):

| S_kv | fp32     | fp16     |
|-----:|---------:|---------:|
| 512  | 2.42 ms  | 2.40 ms  |
| 1024 | 4.71 ms  | 4.69 ms  |
| 2048 | 9.21 ms  | 9.20 ms  |
| 4096 | 18.20 ms | 18.26 ms |

- **fp16 ≈ fp32, within run-to-run noise.** This v1 kernel is
  structure/occupancy-bound, not dtype-bound, so halving the element size does
  not help. Both plateau at **~5.5 GF/s ≈ 0.2% of the ~3 TFLOP/s Adreno 830
  fp16 peak**.
- Prefill **S=4096 trips the GPU watchdog (TDR)** and is omitted — same as fp32.
- **Correctness:** the fp16 kernel is numerically sane (no NaN/Inf, mean abs
  error ~1e-5 vs the fp32 kernel) — see `correctness_fp16.txt`.
- **Reading the GB/s column:** the fp16 files compute traffic with 2 bytes/elem,
  so fp16 GB/s is half the fp32 GB/s at the *same* wall time. It is not a
  regression; compare latency / GF-s instead.

The kernel needs real optimization (tiling / vectorization / multiple row-blocks
per workgroup) before fp16-vs-fp32 is even a meaningful question.

## Files

- `prefill_sq_fp16.{txt,csv}`, `decode_skv_fp16.{txt,csv}` — fp16 sweep
- `correctness_fp16.txt` — fp16-vs-fp32 numeric diff
- `prefill_sq.{txt,csv}`, `decode_skv.{txt,csv}` — fp32 baseline (unchanged)

## Reproduce

```bash
cmake --build build-android-arm64-v8a --target mllm-fa-opencl-bench -j
adb push build-android-arm64-v8a/bin/mllm-fa-opencl-bench \
         build-android-arm64-v8a/bin/libMllm{RT,CPUBackend,OpenCLBackend}.so \
         <ndk>/.../aarch64/libomp.so  /data/local/tmp/fa_ocl/
# fp16 (runs the correctness check first, then the sweep):
adb shell 'cd /data/local/tmp/fa_ocl && LD_LIBRARY_PATH=. FA_DTYPE=fp16 ./mllm-fa-opencl-bench'
# fp32 baseline:
adb shell 'cd /data/local/tmp/fa_ocl && LD_LIBRARY_PATH=. FA_DTYPE=fp32 ./mllm-fa-opencl-bench'
```
