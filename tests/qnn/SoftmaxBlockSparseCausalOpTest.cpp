// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// On-device integration test for the SoftmaxBlockSparseCausal custom QNN op.
//
// Validates that one fused custom-op call produces the same numerical output
// as the existing 3-op decomposition:
//   QKs = ElementWiseMul(QK, scale)
//   QKm = ElementWiseAdd(QKs, mask)            (mask = causal triangle on
//                                              diagonal slot, -1e4 elsewhere)
//   P   = Softmax(QKm, axis=-1)
//
// The custom op SoftmaxBlockSparseCausal(QK, scale, q_block_idx, bk) does the
// same math without a mask tensor — it derives the masking pattern from
// q_block_idx and bk.
//
// Two graphs are built:
//   - "ref": the 3-op decomposition with an explicit mask tensor.
//   - "fused": the single custom-op SoftmaxBlockSparseCausal.
// We feed identical QK inputs to both and compare outputs element-wise.
//
// Prerequisites — push these to the same directory before running, then
// LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=.;/data/local/tmp ./Mllm-Test-QNN-SoftmaxBlockSparseCausalOp:
//
//   libQnnHtp.so                              (QNN HTP runtime, ARM side)
//   libQnnSystem.so                           (QNN system interface)
//   libQnnLLaMAPackage_CPU.so                 (ARM/aarch64 package, target CPU)
//   libQnnLLaMAPackage.so                     (hexagon-vXX package, target HTP)

#include <gtest/gtest.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include "mllm/backends/cpu/CPUBackend.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "QnnBackend.h"

using namespace mllm;
using namespace mllm::qnn;

namespace {

constexpr float kMaskNeg = -1.0e4f;  // -inf surrogate (matches the rest of the suite).

bool readableFile(const std::string& path) { return access(path.c_str(), R_OK) == 0; }

std::string findReadableFile(const std::vector<std::string>& candidates) {
  for (const auto& candidate : candidates) {
    if (readableFile(candidate)) return candidate;
  }
  return {};
}

void unbufferOutput() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
}

#define STEP(msg) fprintf(stderr, "[STEP] " msg "\n")

}  // namespace

// ---------------------------------------------------------------------------
// Test fixture — registers the LLaMAPackage custom op package on both CPU and
// HTP targets, then creates the QNN context.
// ---------------------------------------------------------------------------
class SoftmaxBlockSparseCausalOpTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    unbufferOutput();
    STEP("SetUpTestSuite start");

    {
      const char* existing = std::getenv("ADSP_LIBRARY_PATH");
      std::string adsp_path = existing && existing[0] != '\0'
                                  ? (std::string(".;/data/local/tmp;") + existing)
                                  : ".;/data/local/tmp";
      setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), /*overwrite=*/1);
      fprintf(stderr, "[STEP] ADSP_LIBRARY_PATH=%s\n", adsp_path.c_str());
    }

    ASSERT_TRUE(isQnnAvailable()) << "QNN runtime libraries not found";

    auto& ctx = Context::instance();
    auto host_backend = cpu::createCPUBackend();
    ctx.registerBackend(host_backend);
    ctx.memoryManager()->registerAllocator(kCPU, host_backend->allocator(), MemoryManagerOptions());

    backend_ = std::make_shared<QNNBackend>();

    const std::string cpu_package_path = findReadableFile({
        "./libQnnLLaMAPackage_CPU.so",
        "/data/local/tmp/libQnnLLaMAPackage_CPU.so",
    });
    ASSERT_FALSE(cpu_package_path.empty())
        << "Missing ARM package. Push build/aarch64-android/libQnnLLaMAPackage.so as libQnnLLaMAPackage_CPU.so.";

    const std::string htp_package_path = findReadableFile({
        "./libQnnLLaMAPackage.so",
        "/data/local/tmp/libQnnLLaMAPackage.so",
        "./libQnnLLaMAPackage_HTP.so",
        "/data/local/tmp/libQnnLLaMAPackage_HTP.so",
    });
    ASSERT_FALSE(htp_package_path.empty()) << "Missing Hexagon HTP op package.";

    auto registerPackage = [](const std::shared_ptr<QNNBackend>& backend, const std::string& package_path,
                              const char* target) {
      auto ret = backend->qnnInterface().backendRegisterOpPackage(backend->backendHandle(), package_path.c_str(),
                                                                  "LLaMAPackageInterfaceProvider", target);
      fprintf(stderr, "[STEP] backendRegisterOpPackage(%s) returned 0x%x (low16=%d)\n", target, (unsigned)ret,
              (int)(ret & 0xFFFF));
      return ret;
    };

    auto ret = registerPackage(backend_, cpu_package_path, "CPU");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, (int)(ret & 0xFFFF));
    ret = registerPackage(backend_, htp_package_path, "HTP");
    ASSERT_EQ(QNN_BACKEND_NO_ERROR, (int)(ret & 0xFFFF));

    ASSERT_TRUE(backend_->createContext()) << "QNN context creation failed";
    ctx.registerBackend(backend_);
    ctx.memoryManager()->registerAllocator(
        kQNN, backend_->allocator(), {.really_large_tensor_threshold = 0, .using_buddy_mem_pool = false});
    ctx.dispatcherManager()->registerDispatcher(
        createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), QNNDispatcherOptions()));
  }

  static void TearDownTestSuite() { backend_.reset(); }
  static std::shared_ptr<QNNBackend> backend_;
};
std::shared_ptr<QNNBackend> SoftmaxBlockSparseCausalOpTest::backend_ = nullptr;

// ---------------------------------------------------------------------------
// Build the per-q-block causal mask [BQ, top_k*bk]: zeros for valid columns,
// kMaskNeg for masked columns. Same logic as buildCausalMaskOneQb in
// BlockSparseAttentionCausalTest.cpp — replicated here so this file is
// self-contained.
// ---------------------------------------------------------------------------
static void buildMaskOneQb(__fp16* mask_qb, int q_block_idx, int BQ, int bk, int top_k) {
  const int W = top_k * bk;
  const int hist_slots = (q_block_idx < top_k - 1) ? q_block_idx : (top_k - 1);
  for (int q = 0; q < BQ; ++q) {
    __fp16* row = mask_qb + (size_t)q * W;
    for (int slot = 0; slot < top_k - 1; ++slot) {
      const bool is_padding = (slot >= hist_slots);
      for (int s = 0; s < bk; ++s) row[slot * bk + s] = is_padding ? (__fp16)kMaskNeg : (__fp16)0.0f;
    }
    const int diag_off = (top_k - 1) * bk;
    for (int s = 0; s < bk; ++s) row[diag_off + s] = (s <= q) ? (__fp16)0.0f : (__fp16)kMaskNeg;
  }
}

// ---------------------------------------------------------------------------
// Host reference: scaled + masked softmax, exactly the operation the fused
// op should implement. Operates in fp32 internally for stability.
// ---------------------------------------------------------------------------
static void hostRefSoftmax(const __fp16* qk, __fp16* p_out, int Hq, int BQ, int W, float scale,
                            int q_block_idx, int bk) {
  const int top_k = W / bk;
  const int diag_slot = top_k - 1;
  const int n_real_hist = (q_block_idx < top_k - 1) ? q_block_idx : (top_k - 1);
  std::vector<float> tmp(W);
  for (int h = 0; h < Hq; ++h) {
    for (int q = 0; q < BQ; ++q) {
      const __fp16* srow = qk + ((size_t)h * BQ + q) * W;
      __fp16* prow = p_out + ((size_t)h * BQ + q) * W;
      float row_max = -INFINITY;
      for (int c = 0; c < W; ++c) {
        const int slot = c / bk;
        const int col_in_slot = c - slot * bk;
        bool valid;
        if (slot < n_real_hist) {
          valid = true;
        } else if (slot < diag_slot) {
          valid = false;
        } else {
          valid = (col_in_slot <= q);
        }
        if (!valid) {
          tmp[c] = -INFINITY;
          continue;
        }
        float v = (float)srow[c] * scale;
        tmp[c] = v;
        if (v > row_max) row_max = v;
      }
      float denom = 0.0f;
      for (int c = 0; c < W; ++c) {
        if (tmp[c] == -INFINITY) {
          tmp[c] = 0.0f;
          continue;
        }
        tmp[c] = std::exp(tmp[c] - row_max);
        denom += tmp[c];
      }
      const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
      for (int c = 0; c < W; ++c) prow[c] = (__fp16)(tmp[c] * inv);
    }
  }
}

// ---------------------------------------------------------------------------
// Run one compare case: build a ref graph (Mul + Add + Softmax) and a fused
// graph (SoftmaxBlockSparseCausal), feed identical QK + matching mask, then
// compare both against the host reference.
// ---------------------------------------------------------------------------
static void runCompare(const std::shared_ptr<QNNBackend>& backend, int Hq, int BQ, int bk, int top_k,
                       int q_block_idx, const std::string& tag) {
  const int W = top_k * bk;
  fprintf(stderr, "[CASE %s] Hq=%d BQ=%d bk=%d top_k=%d (W=%d) q_block_idx=%d\n", tag.c_str(), Hq, BQ, bk, top_k, W,
          q_block_idx);

  // -------- Allocate tensors --------
  auto QK = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN).alloc();
  auto P_ref = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN).alloc();
  auto P_fused = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN).alloc();

  // The reference graph needs a static mask tensor [1, BQ, W] (broadcast over Hq).
  auto Mask = Tensor::empty({1, BQ, W}, kFloat16, kQNN).alloc();
  buildMaskOneQb(Mask.ptr<__fp16>(), q_block_idx, BQ, bk, top_k);

  // Static scale tensor (used by the ref graph's Mul).
  const float scale = 1.0f / std::sqrt(128.0f);  // emulate D=128 attention
  auto ScaleT = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  ScaleT.ptr<__fp16>()[0] = (__fp16)scale;

  // -------- Fill QK with deterministic random fp16 in [-1, 1) --------
  std::mt19937 rng(0xCA0501u + (uint32_t)q_block_idx + (uint32_t)(W * 7));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qkp = QK.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * BQ * W; ++i) qkp[i] = (__fp16)dist(rng);

  // -------- Host reference --------
  std::vector<__fp16> ref((size_t)Hq * BQ * W);
  hostRefSoftmax(qkp, ref.data(), Hq, BQ, W, scale, q_block_idx, bk);

  // -------- Graph 1: reference decomposition (Mul + Add + Softmax) --------
  const std::string g_ref = "ref_" + tag;
  ASSERT_NE(backend->createQnnGraph(g_ref), nullptr);
  backend->addTensor(g_ref, "QK", QNN_TENSOR_TYPE_APP_WRITE, QK);
  backend->addTensor(g_ref, "P", QNN_TENSOR_TYPE_APP_READ, P_ref);
  backend->addStaticTensor(g_ref, "scale", ScaleT);
  backend->addStaticTensor(g_ref, "mask", Mask);

  auto QKs_t = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN);
  auto QKm_t = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN);
  backend->addTensor(g_ref, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend->addTensor(g_ref, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);

  backend->graphAddNode(g_ref, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  backend->graphAddNode(g_ref, "add_mask", "ElementWiseAdd", {"QKs", "mask"}, {"QKm"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
      QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend->graphAddNode(g_ref, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
  ASSERT_TRUE(backend->graphFinalize(g_ref));

  std::vector<Tensor> ins_ref = {QK};
  std::vector<Tensor> outs_ref = {P_ref};
  backend->graphExecute(g_ref, ins_ref, outs_ref);  // warmup
  const auto t0_ref = std::chrono::steady_clock::now();
  backend->graphExecute(g_ref, ins_ref, outs_ref);
  const auto t1_ref = std::chrono::steady_clock::now();
  const double ref_ms = std::chrono::duration<double, std::milli>(t1_ref - t0_ref).count();

  // -------- Graph 2: fused custom op --------
  // q_block_idx is now a runtime APP_WRITE input (uint32 scalar tensor),
  // not a compile-time scalar param.
  auto QBlockIdx = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
  QBlockIdx.ptr<uint32_t>()[0] = (uint32_t)q_block_idx;

  const std::string g_fused = "fused_" + tag;
  ASSERT_NE(backend->createQnnGraph(g_fused), nullptr);
  backend->addTensor(g_fused, "QK", QNN_TENSOR_TYPE_APP_WRITE, QK);
  backend->addTensor(g_fused, "q_block_idx", QNN_TENSOR_TYPE_APP_WRITE, QBlockIdx);
  backend->addTensor(g_fused, "P", QNN_TENSOR_TYPE_APP_READ, P_fused);
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> fused_params = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<uint32_t>("bk", (uint32_t)bk),
  };
  backend->graphAddNode(g_fused, "fused_softmax", "SoftmaxBlockSparseCausal",
                        {"QK", "q_block_idx"}, {"P"}, {}, fused_params, "LLaMAPackage");
  ASSERT_TRUE(backend->graphFinalize(g_fused));

  std::vector<Tensor> ins_fused = {QK, QBlockIdx};
  std::vector<Tensor> outs_fused = {P_fused};
  backend->graphExecute(g_fused, ins_fused, outs_fused);  // warmup
  const auto t0_fused = std::chrono::steady_clock::now();
  backend->graphExecute(g_fused, ins_fused, outs_fused);
  const auto t1_fused = std::chrono::steady_clock::now();
  const double fused_ms = std::chrono::duration<double, std::milli>(t1_fused - t0_fused).count();

  // -------- Compare both graphs against the host reference --------
  const __fp16* gref = P_ref.ptr<__fp16>();
  const __fp16* gfused = P_fused.ptr<__fp16>();
  size_t miss_ref = 0, miss_fused = 0, miss_pair = 0;
  float max_err_ref = 0.f, max_err_fused = 0.f, max_err_pair = 0.f;
  // Sample a few mismatches to help debug.
  int sampled = 0;
  for (size_t i = 0; i < (size_t)Hq * BQ * W; ++i) {
    float dr = std::fabs((float)gref[i] - (float)ref[i]);
    float df = std::fabs((float)gfused[i] - (float)ref[i]);
    float dp = std::fabs((float)gref[i] - (float)gfused[i]);
    if (dr > max_err_ref) max_err_ref = dr;
    if (df > max_err_fused) max_err_fused = df;
    if (dp > max_err_pair) max_err_pair = dp;
    if (dr > 5e-3f) ++miss_ref;
    if (df > 5e-3f) ++miss_fused;
    if (dp > 5e-3f) ++miss_pair;
    if (df > 5e-3f && sampled < 8) {
      int h = (int)(i / ((size_t)BQ * W));
      int rest = (int)(i - (size_t)h * BQ * W);
      int q = rest / W;
      int c = rest - q * W;
      fprintf(stderr, "  miss[h=%d q=%d c=%d]: host=%.6f fused=%.6f ref=%.6f\n", h, q, c, (float)ref[i],
              (float)gfused[i], (float)gref[i]);
      ++sampled;
    }
  }
  fprintf(stderr,
          "[CMP %-14s]\n"
          "  ref   (Mul+Add+Softmax) : %7.3f ms  err_vs_host=%.4f miss=%zu\n"
          "  fused (custom op)        : %7.3f ms  err_vs_host=%.4f miss=%zu\n"
          "  fused vs ref (per-elt)   : max=%.4f miss=%zu\n",
          tag.c_str(), ref_ms, max_err_ref, miss_ref, fused_ms, max_err_fused, miss_fused, max_err_pair, miss_pair);

  EXPECT_EQ(miss_ref, 0u) << "ref decomposition diverged from host reference (test bug?)";
  EXPECT_EQ(miss_fused, 0u) << "fused custom op diverged from host reference";
}

// ---------------------------------------------------------------------------
// Test cases. Each TEST_F runs in a fresh process — see BlockSparseAttention*
// for the multi-graph state-accumulation note.
//
// Mid-prefill (q_block_idx >= top_k - 1) — all top_k-1 historical slots are
// real, no padding. This is the common steady-state case.
// ---------------------------------------------------------------------------
TEST_F(SoftmaxBlockSparseCausalOpTest, Mid_TopK8_Hq16_BQ32) {
  runCompare(backend_, /*Hq=*/16, /*BQ=*/32, /*bk=*/32, /*top_k=*/8, /*q_block_idx=*/16, "mid_tk8");
}

// Early-prefill cases — q_block_idx < top_k-1 exercises the padding-slot
// branch. These are the easy-to-get-wrong rows: q_block_idx=0 has no real
// historical blocks at all (only the diagonal triangle is valid).
TEST_F(SoftmaxBlockSparseCausalOpTest, Early_QbIdx0_TopK8) {
  runCompare(backend_, 16, 32, 32, 8, /*q_block_idx=*/0, "qb0_tk8");
}
TEST_F(SoftmaxBlockSparseCausalOpTest, Early_QbIdx3_TopK8) {
  runCompare(backend_, 16, 32, 32, 8, /*q_block_idx=*/3, "qb3_tk8");
}

// Smaller top_k — covers a different W shape (top_k=4 -> W=128).
TEST_F(SoftmaxBlockSparseCausalOpTest, Mid_TopK4) {
  runCompare(backend_, 16, 32, 32, 4, /*q_block_idx=*/8, "mid_tk4");
}

// ---------------------------------------------------------------------------
// Context-dump variant: builds both the ref (Mul + Add + Softmax) and the
// fused custom-op graph at the Mid_TopK8 shape, executes once for warmup,
// then saves the QNN context binary + raw fp16 input tensors + per-graph
// input_list.txt files.
//
// Outputs (all in cwd, which on-device is /data/local/tmp):
//   sbsc_ctx.bin            — QNN context containing both graphs
//   sbsc_QK.raw             — fp16 input scores [Hq=16, BQ=32, top_k_BK=256]
//   sbsc_ref_inputs.txt     — input_list line for the ref graph
//   sbsc_fused_inputs.txt   — input_list line for the fused graph
//   ref_dump_*_schematic.bin / fused_dump_*_schematic.bin — auto-emitted by
//                              MLLM_QNN_PROFILE=DETAILED at graphFinalize.
//
// Run with `MLLM_QNN_PROFILE=DETAILED ./Mllm-Test-QNN-... --gtest_filter=*Dump*`
// then push qnn-net-run + the matching reader .so to the device and re-run
// the saved context under it for per-HW-unit (HMX/HVX) cycle accounting.
// ---------------------------------------------------------------------------
TEST_F(SoftmaxBlockSparseCausalOpTest, DumpContext_Mid_TopK8) {
  const int Hq = 16, BQ = 32, bk = 32, top_k = 8, q_block_idx = 16;
  const int W = top_k * bk;

  auto QK = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN).alloc();
  auto P_ref = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN).alloc();
  auto P_fused = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN).alloc();
  auto Mask = Tensor::empty({1, BQ, W}, kFloat16, kQNN).alloc();
  buildMaskOneQb(Mask.ptr<__fp16>(), q_block_idx, BQ, bk, top_k);

  const float scale = 1.0f / std::sqrt(128.0f);
  auto ScaleT = Tensor::empty({1, 1, 1}, kFloat16, kQNN).alloc();
  ScaleT.ptr<__fp16>()[0] = (__fp16)scale;

  std::mt19937 rng(0xCA0501u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  __fp16* qkp = QK.ptr<__fp16>();
  for (size_t i = 0; i < (size_t)Hq * BQ * W; ++i) qkp[i] = (__fp16)dist(rng);

  // Ref graph (Mul + Add + Softmax).
  const std::string g_ref = "ref_dump_mid_tk8";
  ASSERT_NE(backend_->createQnnGraph(g_ref), nullptr);
  backend_->addTensor(g_ref, "QK", QNN_TENSOR_TYPE_APP_WRITE, QK);
  backend_->addTensor(g_ref, "P", QNN_TENSOR_TYPE_APP_READ, P_ref);
  backend_->addStaticTensor(g_ref, "scale", ScaleT);
  backend_->addStaticTensor(g_ref, "mask", Mask);
  auto QKs_t = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN);
  auto QKm_t = Tensor::empty({Hq, BQ, W}, kFloat16, kQNN);
  backend_->addTensor(g_ref, "QKs", QNN_TENSOR_TYPE_NATIVE, QKs_t);
  backend_->addTensor(g_ref, "QKm", QNN_TENSOR_TYPE_NATIVE, QKm_t);
  backend_->graphAddNode(g_ref, "scale_qk", "ElementWiseMultiply", {"QK", "scale"}, {"QKs"}, {}, {}, "qti.aisw");
  backend_->graphAddNode(g_ref, "add_mask", "ElementWiseAdd", {"QKs", "mask"}, {"QKm"}, {}, {}, "qti.aisw");
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> sm = {
      QNNParamScalarWrapper::create<uint32_t>("axis", 2u), QNNParamScalarWrapper::create<float>("beta", 1.0f)};
  backend_->graphAddNode(g_ref, "softmax", "Softmax", {"QKm"}, {"P"}, {}, sm, "qti.aisw");
  ASSERT_TRUE(backend_->graphFinalize(g_ref));

  // Fused graph (custom op). q_block_idx is now an APP_WRITE input tensor.
  auto QBlockIdx = Tensor::empty({1, 1, 1, 1}, kUInt32, kQNN).alloc();
  QBlockIdx.ptr<uint32_t>()[0] = (uint32_t)q_block_idx;

  const std::string g_fused = "fused_dump_mid_tk8";
  ASSERT_NE(backend_->createQnnGraph(g_fused), nullptr);
  backend_->addTensor(g_fused, "QK", QNN_TENSOR_TYPE_APP_WRITE, QK);
  backend_->addTensor(g_fused, "q_block_idx", QNN_TENSOR_TYPE_APP_WRITE, QBlockIdx);
  backend_->addTensor(g_fused, "P", QNN_TENSOR_TYPE_APP_READ, P_fused);
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> fp = {
      QNNParamScalarWrapper::create<float>("softmax_scale", scale),
      QNNParamScalarWrapper::create<uint32_t>("bk", (uint32_t)bk),
  };
  backend_->graphAddNode(g_fused, "fused_softmax", "SoftmaxBlockSparseCausal",
                         {"QK", "q_block_idx"}, {"P"}, {}, fp, "LLaMAPackage");
  ASSERT_TRUE(backend_->graphFinalize(g_fused));

  // Warmup execute (so the saved context's runtime state is "warmed").
  std::vector<Tensor> ins_ref = {QK};
  std::vector<Tensor> ins_fused = {QK, QBlockIdx};
  std::vector<Tensor> outs_ref = {P_ref}, outs_fused = {P_fused};
  backend_->graphExecute(g_ref, ins_ref, outs_ref);
  backend_->graphExecute(g_fused, ins_fused, outs_fused);

  // Save context binary (contains both graphs).
  backend_->saveContext("sbsc_ctx.bin");

  // Save raw inputs.
  {
    std::ofstream f("sbsc_QK.raw", std::ios::binary);
    f.write(reinterpret_cast<const char*>(qkp), (size_t)Hq * BQ * W * sizeof(__fp16));
  }
  {
    std::ofstream f("sbsc_qblock_idx.raw", std::ios::binary);
    uint32_t v = (uint32_t)q_block_idx;
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
  }

  // input_list lines (one input per execute, name=path syntax).
  {
    std::ofstream f("sbsc_ref_inputs.txt");
    f << "QK:=sbsc_QK.raw\n";
  }
  {
    std::ofstream f("sbsc_fused_inputs.txt");
    f << "QK:=sbsc_QK.raw q_block_idx:=sbsc_qblock_idx.raw\n";
  }

  fprintf(stderr,
          "[DUMP] context=sbsc_ctx.bin inputs=sbsc_QK.raw\n"
          "       ref graph    : %s   input_list=sbsc_ref_inputs.txt\n"
          "       fused graph  : %s  input_list=sbsc_fused_inputs.txt\n"
          "       (schematics: %s_schematic.bin / %s_schematic.bin in cwd)\n",
          g_ref.c_str(), g_fused.c_str(), g_ref.c_str(), g_fused.c_str());
}
