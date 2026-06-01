// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// QUANTIZED GatedDeltaNet BLOCK-PREFILL graph (Sq=B tokens) — the parallel-prefill
// counterpart of compile_deltanet_decode (Sq=1). Instead of one recurrence step, it
// processes a whole block of B tokens at once via the CHUNKED gated-delta-rule
// (Yang et al. chunkwise-parallel form): per chunk of C tokens it builds the within-
// chunk UT transform T=(I-L)^{-1} (L strictly-lower => nilpotent => computed by the
// doubling product (I+L)(I+L^2)(I+L^4)... , ~log2(C) batched matmuls, HTP-friendly),
// then carries the recurrent state S across chunks. Math is identical to the sequential
// recurrence (verified in /tmp/verify_chunk.py to 1e-7) but parallel within a chunk.
//
// VALIDATED on V79 (real Qwen3.5 L0): chunk C<=32 -> y 0.29% / state 0.34% rel.
// Two HTP gotchas were essential: (1) HTP ElementWiseExp returns GARBAGE (negative,
// ~-64k) for large-negative inputs, so every decay-exp arg is F::clip'd to [-50,0]
// (ReluMinMax); (2) the q@S near-cancellation core underflows the fp16 RMSNorm, so the
// gated norm uses the K=200 scale-invariant trick. Chunk math is effectively fp16 on HTP,
// so keep C small (C=64 -> state ~5%, C=128 -> overflow/garbage).
//
// Weight-heavy projections (q/k/v/z + out_proj) are LPBQ Conv2D (int4 w, uint16 act,
// block 16) exactly like decode; the chunk math runs in fp32 (HTP fp32 matmul, proven).
//
// I/O contract (block prefill; first block has S0=0 and zero conv state):
//   inputs : x[B,hidden], eps[1,1,1], qscale[1,1,1], norm_w[1,1,Dv], A_log[1,1,H],
//            dt_bias[1,1,H], Wa[1,hidden,H] (PRE-TRANSPOSED), Wb[1,hidden,H],
//            cw_q[1,4,H*Dk],cw_k[1,4,H*Dk],cw_v[1,4,H*Dv],
//            cs_q[1,3,H*Dk],cs_k[1,3,H*Dk],cs_v[1,3,H*Dv], S0[H,Dk,Dv],
//            Ltri[1,1,C,C], strict[1,1,C,C], eye[1,1,C,C]  (constant masks)
//   outputs: y[B,hidden] (per-token deltanet output), Sp[H,Dk,Dv] (state after block)
//
//   ./mllm-qwen3-aot-deltanet-prefill-c -aot_cfg qnn_aot_cfg_deltanet_decode.json \
//        --params deltanet-prefill-l0-lpbq.mllm --seq 128 --chunk 64
//
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/passes/AOTCompileContext.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot_sha.hpp"  // CONV2D_PROPERTY + ptq::QDQ helpers

using mllm::Argparse;
using mllm::Tensor;
namespace sha = mllm::models::qwen3::sha;

namespace {
std::string defaultQnnEnvPath() {
  if (const char* r = std::getenv("QAIRT_SDK_ROOT")) { return std::string(r) + "/lib/x86_64-linux-clang/"; }
  return "/mnt/raid0_ssd/wentao/qairt/2.43.0.260128/lib/x86_64-linux-clang/";
}
}  // namespace

namespace mllm::models::qwen3::sha {

// Quantized per-head deltanet BLOCK PREFILL (chunked gated-delta-rule).
class DeltaNetPrefillLPBQ final : public nn::Module {
  int H_ = 0, Dk_ = 0, Dv_ = 0, hidden_ = 0, B_ = 0, C_ = 0;
  bool dbg_ = false, emit_cat_ = false;
  int emit_core_ = 0;
  bool dbg_chunk_ = false;
  std::vector<nn::Conv2D> q_proj_, k_proj_, v_proj_, z_proj_;
  nn::Conv2D out_proj_;

 public:
  DeltaNetPrefillLPBQ() = default;
  void setDbg(bool d) { dbg_ = d; }
  void setEmitCat(bool d) { emit_cat_ = d; }
  void setEmitCore(int d) { emit_core_ = d; }
  void setDbgChunk(bool d) { dbg_chunk_ = d; }
  DeltaNetPrefillLPBQ(const std::string& name, int H, int Dk, int Dv, int hidden, int B, int C)
      : nn::Module(name), H_(H), Dk_(Dk), Dv_(Dv), hidden_(hidden), B_(B), C_(C) {
    for (int h = 0; h < H; ++h) {
      auto hs = std::to_string(h);
      q_proj_.emplace_back(reg<nn::Conv2D>("q_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      k_proj_.emplace_back(reg<nn::Conv2D>("k_proj." + hs, hidden, Dk, CONV2D_PROPERTY));
      v_proj_.emplace_back(reg<nn::Conv2D>("v_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
      z_proj_.emplace_back(reg<nn::Conv2D>("z_proj." + hs, hidden, Dv, CONV2D_PROPERTY));
    }
    out_proj_ = reg<nn::Conv2D>("out_proj", H * Dv, hidden, CONV2D_PROPERTY);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& in, const std::vector<AnyValue>&) override {
    namespace F = nn::functional;
    const int kd = H_ * Dk_, vd = H_ * Dv_, nc = B_ / C_;
    int p = 0;
    auto x = in[p++];                 // [B,hidden] fp16
    auto eps = in[p++];               // [1,1,1]
    auto qscale = in[p++];            // [1,1,1]
    auto norm_w = in[p++];            // [1,1,Dv]
    auto A_log = in[p++];             // [1,1,H]
    auto dt_bias = in[p++];           // [1,1,H]
    auto Wa = in[p++];                // [1,hidden,H]  (pre-transposed)
    auto Wb = in[p++];                // [1,hidden,H]
    auto cw_q = in[p++], cw_k = in[p++], cw_v = in[p++];   // [1,4,*]
    auto cs_q = in[p++], cs_k = in[p++], cs_v = in[p++];   // [1,3,*]
    auto S0 = in[p++];                // [H,Dk,Dv] fp32
    auto Ltri = in[p++];              // [1,1,C,C] lower-tri ones incl diag
    auto strict = in[p++];            // [1,1,C,C] strictly-lower ones
    auto eye = in[p++];               // [1,1,C,C] identity
    auto eps_f = eps.to(kFloat32);

    // ---- in-graph gates (fp32, log-decay form): g_log = -exp(A_log)*softplus(x@Wa+dt_bias);
    //      beta = sigmoid(x@Wb). a/b GEMVs over the whole block. 4D matmul, weights
    //      PRE-TRANSPOSED ([hidden,H]) — HTP MatMul rejects transB. ----
    auto x4 = x.view({1, 1, B_, hidden_}, true).to(kFloat32);
    auto a = F::matmul(x4, Wa.view({1, 1, hidden_, H_}, true).to(kFloat32));   // [1,1,B,H]
    auto bb = F::matmul(x4, Wb.view({1, 1, hidden_, H_}, true).to(kFloat32));  // [1,1,B,H]
    auto Al = A_log.view({1, 1, 1, H_}, true).to(kFloat32);
    auto db = dt_bias.view({1, 1, 1, H_}, true).to(kFloat32);
    auto g_log = F::neg(F::exp(Al) * F::softplus(a + db));   // [1,1,B,H]
    auto beta = F::sigmoid(bb);                              // [1,1,B,H]

    // Shared QDQ'd input to every per-head projection ([1,1,B,hidden] uint16).
    auto xq = ptq::QDQ(this, x, "qkv_input_qdq").view({1, 1, -1, hidden_}, true);

    // raw LPBQ projection over the block -> [1,1,B,D] fp32.
    auto proj = [&](std::vector<nn::Conv2D>& pr, int h, int D, const std::string& tag) {
      return ptq::QDQ(this, pr[h](xq), tag + std::to_string(h)).to(kFloat32).view({1, 1, B_, D}, true);
    };
    // causal depthwise conv1d (kernel 4) + silu over the block. cs/cw stacked over heads.
    auto conv_silu = [&](Tensor raw, Tensor cw_full, Tensor cs_full, int h, int D) {
      auto cw = cw_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true).to(kFloat32);  // [1,4,D]
      auto cs = cs_full.slice({kAll, kAll, {h * D, (h + 1) * D}}, true).to(kFloat32)
                    .view({1, 1, 3, D}, true);                                          // [1,1,3,D]
      auto full = F::concat({cs, raw}, 2);                                              // [1,1,B+3,D]
      Tensor acc;
      for (int j = 0; j < 4; ++j) {
        auto tap = cw.slice({kAll, {j, j + 1}, kAll}, true).view({1, 1, 1, D}, true);   // [1,1,1,D]
        auto sl = full.slice({kAll, kAll, {j, j + B_}, kAll}, true);                    // [1,1,B,D]
        acc = (j == 0) ? (sl * tap) : (acc + sl * tap);
      }
      return acc * F::sigmoid(acc);   // silu, [1,1,B,D]
    };

    std::vector<Tensor> y_heads;     // each [1,1,B,Dv] gated output per head
    std::vector<Tensor> Sp_heads;    // each [1,1,Dk,Dv] final state per head
    std::vector<Tensor> core_heads;  // each [1,1,B,Dv] pre-norm core per head
    std::vector<Tensor> attn_heads, vnew_heads, T_heads, qc_heads, U_heads, dm_heads, vb_heads, v_heads;  // chunk-0 probes (B==C only)
    std::vector<Tensor> gccol_heads, gcrow_heads, diffmask_heads;
    Tensor dbg_q, dbg_T, dbg_core, dbg_L, dbg_gc;
    const int Hloop = dbg_chunk_ ? 1 : H_;   // dbg_chunk: only head 0 (few tensors -> dump fits DSP)
    for (int h = 0; h < Hloop; ++h) {
      // projections + conv + silu + l2norm (q,k); z is NOT convolved.
      auto q = conv_silu(proj(q_proj_, h, Dk_, "q_out_qdq_h"), cw_q, cs_q, h, Dk_);  // [1,1,B,Dk]
      auto k = conv_silu(proj(k_proj_, h, Dk_, "k_out_qdq_h"), cw_k, cs_k, h, Dk_);
      auto v = conv_silu(proj(v_proj_, h, Dv_, "v_out_qdq_h"), cw_v, cs_v, h, Dv_);  // [1,1,B,Dv]
      auto z = proj(z_proj_, h, Dv_, "z_out_qdq_h");                                 // [1,1,B,Dv]
      if (h == 0) dbg_L = z;   // head-0 z probe (mode 12)
      // NOTE: keep scalar operands rank-4 ([1,1,1,1]) to match q/k rank-4 — a rank-3
      // [1,1,1] operand against a rank-4 tensor mis-broadcasts on HTP and corrupts values.
      // l2-norm eps: HF uses 1e-6, but HTP runs this in fp16 where 1e-6 underflows to 0 ->
      // a zero/degenerate token has sum(q^2)=0, rsqrt(0)=inf, q*inf=NaN, which then poisons
      // real tokens through nan*0 in the causal attn matmul. Use ~1e-3 (fp16-representable,
      // negligible vs real-token norms). mulConstant on the eps INPUT keeps the recipe pass ok.
      auto eps_l2 = eps_f.view({1, 1, 1, 1}, true).mulConstant(Tensor::constant(1000.0f, kFloat32));  // ~1e-3
      const float qsv = 1.0f / std::sqrt((float)Dk_);
      q = (q * F::rsqrt(F::sum(q * q, -1, true) + eps_l2)).mulConstant(Tensor::constant(qsv, kFloat32));  // l2norm + qscale
      k = k * F::rsqrt(F::sum(k * k, -1, true) + eps_l2);
      if (true) qc_heads.push_back(q);   // l2-normed q (chunk-0 probe via mode 6; B==C)
      auto gh = g_log.slice({kAll, kAll, kAll, {h, h + 1}}, true);   // [1,1,B,1] log-decay
      auto bh = beta.slice({kAll, kAll, kAll, {h, h + 1}}, true);    // [1,1,B,1]

      Tensor S = S0.slice({{h, h + 1}, kAll, kAll}, true).view({1, 1, Dk_, Dv_}, true).to(kFloat32);
      std::vector<Tensor> outc;   // per-chunk core output [1,1,C,Dv]
      for (int c = 0; c < nc; ++c) {
        auto sl = [&](Tensor t, int D) {  // chunk c tokens -> [1,1,C,D]
          return t.slice({kAll, kAll, {c * C_, (c + 1) * C_}, kAll}, true);
        };
        auto qc = sl(q, Dk_), kc = sl(k, Dk_), vc = sl(v, Dv_);   // [1,1,C,D]
        auto gc_in = gh.slice({kAll, kAll, {c * C_, (c + 1) * C_}, kAll}, true);  // [1,1,C,1]
        auto bc = bh.slice({kAll, kAll, {c * C_, (c + 1) * C_}, kAll}, true);     // [1,1,C,1]
        // cumulative decay within chunk: gc = Ltri @ g_log  -> [1,1,C,1]
        auto gc = F::matmul(Ltri, gc_in);                       // [1,1,C,1]
        auto gcT = gc.transpose(2, 3);                          // [1,1,1,C]
        // HTP ElementWiseExp returns NEGATIVE GARBAGE for large-negative inputs (exp(-329)
        // dumped as -64256). The 128-token cumulative decay reaches ~-329, so clamp every
        // decay-exp argument to [-30,0]: exp(-30)=9e-14~=0 (those far-token weights ARE ~0),
        // near-diagonal (small |x|) untouched. (decode never hit this — tiny per-token args.)
        const float kClampLo = -50.0f, kClampHi = 0.0f;
        auto cexp = [&](Tensor t) { return F::exp(F::clip(t, kClampLo, kClampHi)); };
        auto egc = cexp(gc);                                    // [1,1,C,1]
        // decay_mask[i,j] = exp(gc_i - gc_j) for i>=j else 0. Build the [C,C] difference by
        // OUTER-PRODUCT MATMULs (gc·1ᵀ and 1·gcᵀ) instead of the broadcast gc[C,1]-gcT[1,C]:
        // HTP's elementwise does NOT do the both-dims-expand outer broadcast, so the bare
        // subtraction mis-aligns and the *Ltri mask fails to zero the upper tri -> exp(+gc)
        // overflows to inf. MASK THE EXPONENT FIRST so the upper tri is exp(0)=1, then *Ltri.
        auto ones = Ltri + strict.transpose(2, 3);              // [1,1,C,C] all-ones
        auto ones_row = ones.slice({kAll, kAll, {0, 1}, kAll}, true);   // [1,1,1,C]
        auto ones_col = ones.slice({kAll, kAll, kAll, {0, 1}}, true);   // [1,1,C,1]
        auto gc_col = F::matmul(gc, ones_row);                  // [1,1,C,C] entry=gc_i
        auto gc_row = F::matmul(ones_col, gcT);                 // [1,1,C,C] entry=gc_j
        auto diffmask = (gc_col - gc_row) * Ltri;               // exp argument [1,1,C,C]
        auto dm = cexp(diffmask) * Ltri;                      // [1,1,C,C]
        if (c == 0) { gccol_heads.push_back(gc_col); gcrow_heads.push_back(gc_row); diffmask_heads.push_back(diffmask); }
        auto kb = kc * bc;                                      // k_beta [1,1,C,Dk]
        auto vb = vc * bc;                                      // v_beta [1,1,C,Dv]
        if (c == 0) { vb_heads.push_back(vb); v_heads.push_back(vc); }
        // L = -(k_beta @ k^T * dm) strictly-lower
        auto L = F::neg(F::matmul(kb, kc.transpose(2, 3)) * dm) * strict;   // [1,1,C,C]
        // T = (I - L)^{-1} = (I+L)(I+L^2)(I+L^4)... (L nilpotent, L^C=0).
        // P=matmul(P,P) is a SELF-ALIASED matmul (same tensor handle for both operands)
        // which HTP mishandles -> garbage that explodes through the doubling. Feed a
        // distinct copy (identity transpose-transpose -> fresh IR node, same values).
        Tensor T = eye, P = L;
        for (int pw = 1; pw < C_; pw *= 2) {
          T = F::matmul(T, eye + P);
          auto Pc = P * ones;                            // distinct handle (P*1), same values
          P = F::matmul(P, Pc);
        }
        auto U = F::matmul(T, vb);                              // [1,1,C,Dv]
        if (c == 0) { T_heads.push_back(T); U_heads.push_back(U); dm_heads.push_back(dm); }
        auto Wk = F::matmul(T, kb * egc);                       // [1,1,C,Dk]  (k_cumdecay)
        // intra-chunk + cross-chunk recurrence
        auto attn = F::matmul(qc, kc.transpose(2, 3)) * dm;     // [1,1,C,C] causal
        auto v_prime = F::matmul(Wk, S);                        // [1,1,C,Dv]
        auto v_new = U - v_prime;                               // [1,1,C,Dv]
        auto a_inter = F::matmul(qc * egc, S);                  // [1,1,C,Dv]
        if (c == 0) { attn_heads.push_back(attn); vnew_heads.push_back(v_new); }
        outc.push_back(a_inter + F::matmul(attn, v_new));       // core out [1,1,C,Dv]
        // S = S*exp(gc_last) + (k * exp(gc_last - gc))^T @ v_new
        auto gc_last = gc.slice({kAll, kAll, {C_ - 1, C_}, kAll}, true);   // [1,1,1,1]
        auto kdec = kc * cexp(gc_last - gc);                  // [1,1,C,Dk]
        S = S * cexp(gc_last) + F::matmul(kdec.transpose(2, 3), v_new);  // [1,1,Dk,Dv]
      }
      auto core = (nc == 1) ? outc[0] : F::concat(outc, 2);     // [1,1,B,Dv]
      core_heads.push_back(core);
      // gated RMSNorm * silu(z). core=q@S is a tiny near-cancellation (~1e-3); the HTP path
      // is effectively fp16, so mean(core^2)~1e-6 underflows the fp16 normal floor (6e-5) ->
      // rsqrt(0+eps)=inf. RMSNorm is scale-invariant: scale core up by K (eps by K^2) to keep
      // core^2 in fp16 range while preserving HF eps. K=200 (K^2=40000 < fp16 max 65504). Same
      // fix as the validated decode graph [[qdeltanet-decode-graph-next]].
      const float Kc = 200.0f;
      auto cs = core.mulConstant(Tensor::constant(Kc, kFloat32));
      auto inv = F::rsqrt(F::mean(cs * cs, -1, true) + eps_f.mulConstant(Tensor::constant(Kc * Kc, kFloat32)));
      auto normed = (cs * inv) * norm_w.to(kFloat32).view({1, 1, 1, Dv_}, true);
      auto zsilu = z * F::sigmoid(z);
      auto gated = normed * zsilu;                              // [1,1,B,Dv]
      if (h == 0) dbg_core = gated;   // head-0 gated, emitted WITHOUT concat (mode 10)
      y_heads.push_back(gated.to(kFloat16));
      Sp_heads.push_back(S);                                    // [1,1,Dk,Dv]
    }

    if (dbg_chunk_) {  // head-0 chunk pipeline only, NO cat/out_proj — minimal tensor count so
                       // MLLM_QNN_DEBUG_ALL_READ + _DUMP fits the DSP shared-buffer limit.
                       // MUST return before cat (cat uses vd=H*Dv; 1 head mismatches). Output
                       // bare fp32 (like the Sp state output) — no QDQ/fp16 cast.
      std::vector<Tensor> dbgouts;
      dbgouts.push_back(core_heads[0]);   // fp32 [1,1,B,Dv] raw (dump self-allocs from QNN dims)
      return dbgouts;
    }

    // concat heads -> [1,1,B,H*Dv], LPBQ out_proj -> [1,1,B,hidden] -> [B,hidden]
    auto cat = F::concat(y_heads, -1);                                       // [1,1,B,H*Dv]
    auto cq = ptq::QDQ(this, cat, "out_proj_input_qdq").view({1, 1, -1, vd}, true);
    auto y = ptq::QDQ(this, out_proj_(cq), "out_proj_output_qdq").to(kFloat16).view({B_, hidden_}, true);

    std::vector<Tensor> outs;
    // emit modes swap output 0 (a reliable front slot) for isolating the y-path:
    //   emit_cat_  -> pre-out_proj concatenated gated [B,H*Dv]
    //   emit_core_ -> pre-gated-norm concatenated core [B,H*Dv]
    // modes 2-9: HEAD-0 SINGLE tensor (no concat), [B,Dv] — use --chunk 128 (C==B==Dv).
    if (emit_core_ == 1) {
      outs.push_back(F::concat(core_heads, -1).to(kFloat16).view({B_, vd}, true));   // core
    } else if (emit_core_ == 2) {
      outs.push_back(attn_heads[0].to(kFloat16).view({B_, Dv_}, true));   // attn_h0
    } else if (emit_core_ == 3) {
      outs.push_back(vnew_heads[0].to(kFloat16).view({B_, Dv_}, true));   // v_new_h0
    } else if (emit_core_ == 4) {
      outs.push_back(T_heads[0].to(kFloat16).view({B_, Dv_}, true)); // T_h0
    } else if (emit_core_ == 5) {
      outs.push_back(U_heads[0].to(kFloat16).view({B_, Dv_}, true));       // U_h0
    } else if (emit_core_ == 6) {
      outs.push_back(qc_heads[0].to(kFloat16).view({B_, Dv_}, true));// qc_h0
    } else if (emit_core_ == 7) {
      outs.push_back(dm_heads[0].to(kFloat16).view({B_, Dv_}, true)); // dm_h0
    } else if (emit_core_ == 8) {
      outs.push_back(vb_heads[0].to(kFloat16).view({B_, Dv_}, true));      // vb_h0
    } else if (emit_core_ == 9) {
      outs.push_back(v_heads[0].to(kFloat16).view({B_, Dv_}, true));       // v_h0
    } else if (emit_core_ == 10) {
      outs.push_back(dbg_core.to(kFloat16).view({B_, Dv_}, true));  // head-0 gated, NO concat
    } else if (emit_core_ == 11) {
      outs.push_back(core_heads[0].to(kFloat16).view({B_, Dv_}, true));  // head-0 core, NO concat
    } else if (emit_core_ == 12) {
      outs.push_back(dbg_L.to(kFloat16).view({B_, Dv_}, true));     // head-0 z, NO concat
    } else if (emit_core_ == 13) {
      outs.push_back(gccol_heads[0].to(kFloat16).view({B_, Dv_}, true));   // gc_col_h0 (B==C)
    } else if (emit_core_ == 14) {
      outs.push_back(gcrow_heads[0].to(kFloat16).view({B_, Dv_}, true));   // gc_row_h0 (B==C)
    } else if (emit_core_ == 15) {
      outs.push_back(diffmask_heads[0].to(kFloat16).view({B_, Dv_}, true));// (gc_col-gc_row)*Ltri exp arg (B==C)
    } else {
      outs.push_back(emit_cat_ ? cat.to(kFloat16).view({B_, vd}, true) : y);  // index 0
    }
    for (int h = 0; h < H_; ++h) outs.push_back(Sp_heads[h].view({1, Dk_, Dv_}, true));  // Sp_h (1..H)
    if (dbg_) {  // head-0 chunk-0 probes: H+1 qc, H+2 attn, H+3 v_new(U), H+4 U
      outs.push_back(dbg_q.view({1, C_, Dk_}, true));      // qc
      outs.push_back(dbg_L.view({1, C_, C_}, true));       // attn
      outs.push_back(dbg_gc.view({1, C_, Dv_}, true));     // v_new
      outs.push_back(dbg_T.view({1, C_, Dv_}, true));      // U
    }
    return outs;
  }
};

}  // namespace mllm::models::qwen3::sha

static std::vector<Tensor> buildInputs(int H, int Dk, int Dv, int hidden, int B, int C) {
  const int kd = H * Dk, vd = H * Dv;
  std::vector<Tensor> ti;
  auto addf16 = [&](std::vector<int> s, const std::string& n) { ti.push_back(Tensor::zeros(s, mllm::kFloat16).setName(n)); };
  auto addf32 = [&](std::vector<int> s, const std::string& n) { ti.push_back(Tensor::zeros(s, mllm::kFloat32).setName(n)); };
  addf16({B, hidden}, "x");
  addf32({1, 1, 1}, "eps");
  addf32({1, 1, 1}, "qscale");
  addf32({1, 1, Dv}, "norm_w");
  addf32({1, 1, H}, "A_log");
  addf32({1, 1, H}, "dt_bias");
  addf32({1, hidden, H}, "Wa");
  addf32({1, hidden, H}, "Wb");
  addf32({1, 4, kd}, "cw_q"); addf32({1, 4, kd}, "cw_k"); addf32({1, 4, vd}, "cw_v");
  addf32({1, 3, kd}, "cs_q"); addf32({1, 3, kd}, "cs_k"); addf32({1, 3, vd}, "cs_v");
  addf32({H, Dk, Dv}, "S0");
  addf32({1, 1, C, C}, "Ltri");
  addf32({1, 1, C, C}, "strict");
  addf32({1, 1, C, C}, "eye");
  return ti;
}

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& aot_cfg = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path").def(defaultQnnEnvPath());
  auto& params_arg = Argparse::add<std::string>("--params").help("real per-head LPBQ .mllm (from export_deltanet_prefill.py)").def("");
  auto& seq_arg = Argparse::add<int>("--seq").help("block length B (multiple of --chunk)").def(128);
  auto& chunk_arg = Argparse::add<int>("--chunk").help("delta-rule chunk size C (<=32: HTP fp16 stable; 64=state~5%, 128 breaks)").def(32);
  auto& H_arg = Argparse::add<int>("--heads").def(16);
  auto& dk_arg = Argparse::add<int>("--dk").def(128);
  auto& dv_arg = Argparse::add<int>("--dv").def(128);
  auto& out_arg = Argparse::add<std::string>("--out").def("");
  auto& dbg_arg = Argparse::add<bool>("--dbg").help("append head-0 intermediate outputs (q, T, core)").def(false);
  auto& dbg_chunk_arg = Argparse::add<bool>("--dbg_chunk").help("ONLY head-0 chunk pipeline, no out_proj (for ALL_READ dump)").def(false);
  auto& emit_cat_arg = Argparse::add<bool>("--emit_cat").help("output 0 = pre-out_proj concatenated gated [B,H*Dv]").def(false);
  auto& emit_core_arg = Argparse::add<int>("--emit_core").help("out0: 1=core 2=attn 3=v_new (B==C)").def(0);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  if (!aot_cfg.isSet()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No aot config"); return -1; }
  if (params_arg.get().empty()) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--params required (real LPBQ checkpoint)"); return -1; }

  const int H = H_arg.get(), Dk = dk_arg.get(), Dv = dv_arg.get(), hidden = H * Dv;
  const int B = seq_arg.get(), C = chunk_arg.get();
  if (B % C != 0) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--seq ({}) must be a multiple of --chunk ({})", B, C); return -1; }

  auto real = mllm::load(params_arg.get(), mllm::ModelFileVersion::kV2, mllm::kCPU, /*mmap=*/false);
  auto ti = buildInputs(H, Dk, Dv, hidden, B, C);
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(), mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(aot_cfg.get()));
  sha::DeltaNetPrefillLPBQ m("model", H, Dk, Dv, hidden, B, C);
  m.setDbg(dbg_arg.get());
  m.setEmitCat(emit_cat_arg.get());
  m.setEmitCore(emit_core_arg.get());
  m.setDbgChunk(dbg_chunk_arg.get());
  m.load(real);
  auto ir = mllm::ir::trace_(m, ti);
  mllm::ir::PassManager pm(ir);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, aot_cfg.get(), real));
  pm.run();
  const std::string bin = out_arg.get().empty() ? ("qwen3-deltanet-prefill-s" + std::to_string(B) + ".bin") : out_arg.get();
  qnn_aot_env.saveContext("context.0", bin);
  mllm::print(fmt::format("DeltaNet block-prefill (LPBQ) B={} C={} H={} -> {} (graph model.0.s{})", B, C, H, bin, hidden));
});
