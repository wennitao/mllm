# Writing a Hexagon HVX Custom Op for the LLaMAPackage QNN Op Package

This is a recipe for adding a new compute-heavy custom operator to
`mllm/backends/qnn/custom-op-package/LLaMAPackage` that runs on the Hexagon
HTP. It is distilled from implementing `FlashAttention.cpp`. Use it when you
need to wire a new op end-to-end: package XML, interface registration, HVX
kernel, ARM reference fallback, and an on-device gtest.

---

## 1. Files you will touch

| File | Why |
|------|-----|
| `src/ops/<NewOp>.cpp` | Op kernel (HVX path + REFERENCE_OP fallback). |
| `src/LLaMAPackageInterface.cpp` | `DECLARE_PKG_OPS_OPTS_LIST`, bump `sg_opNames` size, add a `validateOpConfig` clause. |
| `config/LLaMAOpPackageHtp.xml` | `<OpDef>` (inputs / outputs / params), add to `<SupportedOps>`, add `<SupplementalOpDef>` listing supported datatypes. |
| `tests/qnn/<NewOp>OpTest.cpp` | gtest that builds a 1-node QNN graph, runs it on device, compares against a host reference. |
| `tests/qnn/CMakeLists.txt` | Register the test executable. |

The `Makefile` is generic — it picks up new `.cpp` files in `src/ops/`
automatically; you do not need to edit it.

---

## 2. Op skeleton

Use the existing `CausalMask.cpp` or `LLaMADequantizeAdd.cpp` as a starting
template. The minimal structure:

```cpp
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"
#include <math.h>
#include <stddef.h>

BEGIN_PKG_OP_DEFINITION(PKG_NewOp);

template<typename T>
GraphStatus newopImpl(T& out_0, const T& in_0, /*...*/, const PlainFloatTensor& scalar_param, const Tensor& uint_param);

DEF_PACKAGE_OP((newopImpl<Tensor>), "NewOp")

DEF_PACKAGE_PARAM_ORDER("NewOp",
                        "scalar_param", true, nullptr,
                        "uint_param",   true, nullptr)

#ifndef REFERENCE_OP
  #include <hexagon_types.h>
  #include "hvx_internal.h"
  // HVX kernels here
#endif

template<typename T>
GraphStatus newopImpl(T& out_0, const T& in_0, /*...*/) {
  out_0.set_dims(in_0);
  auto [B, H, W, D] = in_0.dims();
  // dispatch to HVX or reference path
  return GraphStatus::Success;
}

END_PKG_OP_DEFINITION(PKG_NewOp);
```

Two compile contexts share this file:
- `hexagon-vXX` build defines no `REFERENCE_OP`. HVX intrinsics + Hexagon
  runtime. Real on-device path.
- `aarch64-android` build is built with `-DREFERENCE_OP`. Plain C++.
  This is the **ARM-side prepare/validation** package — the QNN graph
  builder needs it to type-check ops before the DSP loads the real one.
  It runs only briefly during graph creation, not in the hot path.

---

## 3. The five Hexagon gotchas (each one cost a debug iteration)

### 3.1 `libm expf` returns 0 on the HTP

Every existing op in this package only calls `expf`/`expf16` inside the
`#ifdef REFERENCE_OP` block. On the Hexagon side, `expf` symbol "resolves"
but at runtime returns 0 silently — your op produces all-zero outputs and
the QNN runtime gives no error.

**Fix:** ship your own polynomial expf. A Cephes-style range reduction
(`x = n·ln(2) + r`, polynomial for `exp(r)`, scale by `2^n` via bit
manipulation) is ~7 digits accurate and works fine:

```cpp
static inline float fa_expf(float x) {
  if (x < -87.336544f) return 0.0f;
  if (x >  88.722839f) x =  88.722839f;
  const float LOG2EF = 1.44269504088896341f;
  const float C1 = 0.693359375f;
  const float C2 = -2.12194440e-4f;
  float fx = x * LOG2EF + 0.5f;
  int32_t n = (int32_t)fx;
  if ((float)n > fx) n -= 1;            // floor for negative x
  float r = x - (float)n * C1 - (float)n * C2;
  float r2 = r * r;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r2 + r + 1.0f;
  union { float f; int32_t i; } u;
  u.i = (n + 127) << 23;                // 2^n
  return p * u.f;
}
```

The same applies to `expf16`, `tanhf`, etc. If the reference path uses a
libm function and the HVX path needs it too, write your own.

### 3.2 The `vlalign`-based qf32 horizontal sum is fragile

The pattern that works in `RMSNorm.cpp`:

```cpp
for (int32_t i = 64; i >= 4; i >>= 1)
  sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vlalign_VVR(sum, zero, i));
sum = Q6_Vsf_equals_Vqf32(sum);
float result = *((float*)&sum + 31);
```

works in RMSNorm because the result is read **once per op invocation** and
then used in scalar code (the `1/sqrt(...)` step). When you reuse the same
reduction inside a hot loop (e.g., one dot product per K row in attention),
results come out subtly wrong — the `&acc + 31` read interacts badly with
the compiler's register/spill choices when the function is inlined many
times.

**Fix:** convert qf32 → sf and sum the 32 lanes in scalar code via a union:

```cpp
static inline float fa_hvx_qf32_hsum(HVX_Vector v_qf32) {
  HVX_Vector acc_sf = Q6_Vsf_equals_Vqf32(v_qf32);
  union { HVX_Vector v; float f[32]; } u;
  u.v = acc_sf;
  float s = 0.0f;
  for (int i = 0; i < 32; ++i) s += u.f[i];
  return s;
}
```

This is just as fast (still one HVX add tree → one spill → 32 scalar adds)
and never produces wrong values.

### 3.3 Use `HVX_UVector*` for read-modify-write of output buffers

QNN tensor base addresses *are* page-aligned, but row offsets within a
multi-head tensor (e.g., `head_idx * D * sizeof(float)`) may not always
hit a 128-byte boundary, and aligned `vmem` stores **silently truncate**
the address to the previous 128-byte boundary on Hexagon — corrupting
neighbouring rows. Match the pattern in `RMSNorm.cpp` / `LLaMADequantizeAdd.cpp`:

```cpp
HVX_Vector*  iptr = (HVX_Vector*)input;    // aligned reads OK
HVX_UVector* optr = (HVX_UVector*)output;  // unaligned store, defensive
```

For a kernel that *both reads and writes* the output (online accumulation
like FA), use `HVX_UVector*` for both.

### 3.4 Intrinsic names that don't exist

There is no `Q6_Vqf32_vmpy_Vqf32Vsf`. The canonical pattern when you have
a qf32 accumulator and want to multiply by an sf vector is:

```cpp
HVX_Vector prod_qf = Q6_Vqf32_vmpy_VsfVsf(alpha_sf, src_sf);   // sf*sf -> qf32
HVX_Vector dst_qf  = Q6_Vqf32_vadd_VsfVsf(dst_sf, zero);       // sf -> qf32 lift
HVX_Vector sum_qf  = Q6_Vqf32_vadd_Vqf32Vqf32(prod_qf, dst_qf);
HVX_Vector out_sf  = Q6_Vsf_equals_Vqf32(sum_qf);
```

When in doubt, grep existing ops for the source/destination type pair you
want — only those combinations are supported.

### 3.5 No heap allocation, watch the stack

The HTP runtime explicitly forbids `malloc`/`new` and STL containers with
default allocators in op kernels. Stack arrays are fine, but keep them
small (a few KB). For FlashAttention I keep an `s_buf[FA_BC]` (256 bytes)
for one tile of softmax scores; that's well under the limit.

---

## 4. The HVX kernel pattern

For elementwise / inner-product / weighted-sum kernels, three primitives
cover most of what you need. Each loops over `D / 32` HVX vectors:

```cpp
// dot product (read-only)
static inline float hvx_dot_f32(const float* q, const float* k, uint32_t D) {
  const HVX_UVector* qp = (const HVX_UVector*)q;
  const HVX_UVector* kp = (const HVX_UVector*)k;
  HVX_Vector acc = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
  for (uint32_t i = 0; i < D / 32; ++i)
    acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(qp[i], kp[i]));
  return hvx_qf32_hsum(acc);   // scalar reduction!
}

// dst += alpha * src  (read-modify-write)
static inline void hvx_axpy_f32(float* dst, const float* src, float alpha, uint32_t D) {
  HVX_UVector* dp = (HVX_UVector*)dst;
  const HVX_UVector* sp = (const HVX_UVector*)src;
  HVX_Vector a = Q6_V_vsplat_R(float_to_bits(alpha));
  HVX_Vector z = Q6_V_vzero();
  for (uint32_t i = 0; i < D / 32; ++i) {
    HVX_Vector prod = Q6_Vqf32_vmpy_VsfVsf(a, sp[i]);
    HVX_Vector dqf  = Q6_Vqf32_vadd_VsfVsf(dp[i], z);
    dp[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(prod, dqf));
  }
}

// dst *= alpha
static inline void hvx_scale_f32(float* dst, float alpha, uint32_t D) {
  HVX_UVector* dp = (HVX_UVector*)dst;
  HVX_Vector a = Q6_V_vsplat_R(float_to_bits(alpha));
  for (uint32_t i = 0; i < D / 32; ++i)
    dp[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, dp[i]));
}
```

`l2fetch(addr, bytes_per_row, bytes_per_row, n_rows, 0)` warms the next
tile. Issue it before the inner D-loop so DMA overlaps with HVX compute:

```cpp
if (next_tile_in_range)
  l2fetch(K + next_j0 * kv_stride, D * sizeof(float), D * sizeof(float),
          tile_rows, 0);
```

---

## 5. Wiring into the package

### 5.1 `LLaMAPackageInterface.cpp`

Three edits:

```cpp
// 1. Declare the new op-table chunk:
DECLARE_PKG_OPS_OPTS_LIST(PKG_NewOp)

// 2. Bump the array size and append the type name:
static std::array<const char*, N+1> sg_opNames{{ ..., "NewOp" }};

// 3. Add a validation clause inside validateOpConfig:
} else if (std::string(opConfig.v1.typeName) == "NewOp") {
  if (opConfig.v1.numOfParams != /*P*/ ||
      opConfig.v1.numOfInputs != /*I*/ ||
      opConfig.v1.numOfOutputs != /*O*/) {
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }
}
```

The IDE often complains about the `DECLARE_PKG_OPS_OPTS_LIST` macros
producing a `built_array<…,1>` mismatch with `ba_op_table<N>`. This is an
IDE limitation — the actual hexagon-clang build is fine.

### 5.2 `LLaMAOpPackageHtp.xml`

Three sections:

1. `<OpDef>` block under `<OpDefList>`: declares inputs, outputs,
   parameters, default values. `<Datatype>BACKEND_SPECIFIC</Datatype>` is
   correct here — actual datatypes come from the supplemental block.
2. `<OpName>NewOp</OpName>` inside `<SupportedOps>`.
3. `<SupplementalOpDef>` listing the QNN datatypes you actually support
   (e.g. `QNN_DATATYPE_FLOAT_16`, `QNN_DATATYPE_FLOAT_32`).

---

## 6. On-device gtest

Pattern (see `tests/qnn/LLaMAMulOpTest.cpp` and
`tests/qnn/FlashAttentionOpTest.cpp`):

1. Test fixture's `SetUpTestSuite` creates `QNNBackend`, registers the
   ARM op package as target `CPU` and the Hexagon package as target `HTP`,
   creates the QNN context, and registers the dispatcher.
2. Each `TEST_F` allocates input/output `Tensor`s in `kQNN` shared
   memory, fills with deterministic random data
   (`std::mt19937 rng(seed)`), computes a host reference into a
   `std::vector<float>`, then:
   - `createQnnGraph` with a unique name per case;
   - `addTensor` for each input / output;
   - `graphAddNode` with `packageName="LLaMAPackage"` and your scalar params via
     `QNNParamScalarWrapper::create<float>("name", value)`;
   - `graphFinalize` (validation runs here — failures show up as a non-zero
     low16 error code);
   - `graphExecute`;
   - elementwise tolerance check (`5e-3` is fine for fp32 attention with
     random uniform inputs).

**Hex-literal pitfall:** `0xA77ENu` is parsed as `0xA77E` + invalid suffix
`Nu`. Use `0xA77E0001u` or just `12345u`.

**Param wiring:** scalar params go via `QNNParamScalarWrapper::create<T>`,
where T determines the QNN datatype:
- `bool`  → `QNN_DATATYPE_BOOL_8`
- `uint32_t` / `int32_t` → `QNN_DATATYPE_UINT_32` / `INT_32`
- `float` → `QNN_DATATYPE_FLOAT_32`

The names must match `DEF_PACKAGE_PARAM_ORDER` in the kernel exactly.

---

## 7. Build & deploy loop

```bash
# 1. Build Hexagon and ARM packages
cd mllm/backends/qnn/custom-op-package/LLaMAPackage
make htp_v79 htp_aarch64        # adjust v79 to your target

# 2. Build host test
cmake --build build-android-arm64-v8a-qnn --target Mllm-Test-QNN-NewOp

# 3. Push to device (paths from repo root)
adb push mllm/backends/qnn/custom-op-package/LLaMAPackage/build/hexagon-v79/libQnnLLaMAPackage.so      /data/local/tmp/libQnnLLaMAPackage.so
adb push mllm/backends/qnn/custom-op-package/LLaMAPackage/build/aarch64-android/libQnnLLaMAPackage.so  /data/local/tmp/libQnnLLaMAPackage_CPU.so
adb push build-android-arm64-v8a-qnn/bin/Mllm-Test-QNN-NewOp                                            /data/local/tmp/

# 4. Run
adb shell "cd /data/local/tmp && export LD_LIBRARY_PATH=. && ./Mllm-Test-QNN-NewOp"
```

`scripts/adb_push.sh` handles a full deploy including `libQnnHtp.so`,
`libQnnSystem.so`, the V79 stub, and the V79 skel. Run it once to get the
QNN runtime onto the device, then iterate with the targeted pushes above.

When `make htp_v79` prints `Nothing to be done`, force a rebuild with
`rm build/hexagon-v79/ops/<NewOp>.o`. Plain `touch` of the source isn't
always enough.

---

## 8. Debugging checklist when results are wrong

The HTP gives no useful runtime error — wrong output is your only signal.
Bisect like this:

1. **All zeros?** Almost certainly a libm function returning 0 on HTP. See
   §3.1.
2. **First row correct, later rows corrupt?** Alignment-truncation in an
   aligned `HVX_Vector*` store. Switch the output to `HVX_UVector*`. See
   §3.3.
3. **Per-row results subtly wrong, no clear pattern?** Suspect the qf32
   horizontal sum. Replace with the union-spill scalar version. See §3.2.
4. **Bisect by replacing HVX functions with scalar one at a time.** A pure
   scalar implementation (still using `fa_expf`) is the ground truth for
   the algorithm. Once that passes, swap one HVX primitive in at a time.
5. The host reference in the gtest is the source of truth for the
   algorithm. Make it as simple as possible (naive softmax, no online
   recurrence) so any divergence is the kernel's fault, not the test's.

---

## 9. Checklist — minimum viable new op

- [ ] `src/ops/NewOp.cpp` with `DEF_PACKAGE_OP`, `DEF_PACKAGE_PARAM_ORDER`,
      both REFERENCE_OP and HVX paths.
- [ ] `LLaMAPackageInterface.cpp`: `DECLARE_PKG_OPS_OPTS_LIST`,
      `sg_opNames` size + entry, `validateOpConfig` clause.
- [ ] `LLaMAOpPackageHtp.xml`: `<OpDef>`, `<SupportedOps>` entry,
      `<SupplementalOpDef>`.
- [ ] `make htp_v79 htp_aarch64` succeeds with `-Werror`.
- [ ] gtest passes against a host reference within fp32 tolerance.
