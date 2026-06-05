// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Host-orchestrated LFM2 DECODE-STEP runner: chains N per-layer graphs (compiled distinctly
// by compile_lfm2_decode_stack) with host norms / residuals / MoE-routing / state, and
// validates the residual stream after each layer vs the golden (export_decode_stack.py).
//
//   hidden = embed_row
//   for L: r=hidden; h=op_norm(hidden); mix=(conv_lL|attn_lL)(h); hidden=r+mix;
//          r2=hidden; h2=ffn_norm(hidden); ffn=(ffn_lL | sum_j w_j*moe_lL_e{sel_j})(h2); hidden=r2+ffn
//
//   ./mllm-lfm2-aot-decode-run -m stack.bin --dir wm_dec --layers 2
//
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <set>
#include <string>
#include <vector>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"

using mllm::Argparse;
using mllm::Tensor;
using mllm::qnn::aot::QnnAOTModule;

static bool isAttn(int i) { return i == 2 || i == 6 || i == 10 || i == 14 || i == 18 || i == 21; }
static bool isDense(int i) { return i < 2; }

MLLM_MAIN({
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  auto& help = Argparse::add<bool>("-h|--help").help("Show help");
  auto& model_path = Argparse::add<std::string>("-m|--model").def("lfm2-decode-stack.bin");
  auto& bins_arg = Argparse::add<std::string>("--bins").help("comma-separated context .bins (multi-context group)").def("");
  auto& dir_arg = Argparse::add<std::string>("--dir").required(true);
  auto& nl_arg = Argparse::add<int>("--layers").def(2);
  auto& hidden_arg = Argparse::add<int>("--hidden").def(2048);
  auto& ctx_arg = Argparse::add<int>("--ctx").def(256);
  auto& vocab_arg = Argparse::add<int>("--vocab").def(128000);
  auto& head_arg = Argparse::add<bool>("--lm_head").help("run lm_head + argmax after the layers").def(false);
  Argparse::parse(argc, argv);
  if (help.isSet()) { Argparse::printHelp(); return 0; }
  const int NL = nl_arg.get(), hidden = hidden_arg.get(), ctx = ctx_arg.get(), vocab = vocab_arg.get();
  const int H = 32, KV = 8, D = 64, rot = 64, K = 3, P = ctx - 1;

  auto consts = mllm::load(dir_arg.get() + "/consts.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
  const float eps = consts->pull("eps").ptr<float>()[0];

  // host RMSNorm (plain weight), fp32
  auto rmsnorm = [&](const std::vector<float>& x, const Tensor& w) {
    double ms = 0; for (float v : x) ms += (double)v * v; ms /= x.size();
    float inv = 1.f / std::sqrt((float)ms + eps);
    std::vector<float> o(x.size());
    const float* wp = w.ptr<float>();
    for (size_t i = 0; i < x.size(); ++i) o[i] = x[i] * inv * wp[i];
    return o;
  };
  // dispatch a named graph: fp32 host inputs -> fp16 kQNN; outputs fp16 -> fp32 host.
  auto dispatch = [&](const std::string& name, const std::vector<std::pair<std::vector<int>, std::vector<float>>>& ins_spec,
                      const std::vector<std::vector<int>>& outs_shape) {
    std::vector<Tensor> ins, outs;
    for (auto& [shp, data] : ins_spec) {
      auto t = Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc();
      for (int64_t i = 0; i < t.numel(); ++i) t.ptr<__fp16>()[i] = (__fp16)data[i];
      ins.push_back(t);
    }
    for (auto& shp : outs_shape) outs.push_back(Tensor::empty(shp, mllm::kFloat16, mllm::kQNN).alloc());
    QnnAOTModule g(name); g.to(mllm::kQNN); g.setOutputTensors(outs); (void)g(ins);
    std::vector<std::vector<float>> res;
    for (auto& o : outs) {
      std::vector<float> v(o.numel());
      for (int64_t i = 0; i < o.numel(); ++i) v[i] = (float)o.ptr<__fp16>()[i];
      res.push_back(std::move(v));
    }
    return res;
  };
  auto tovec = [&](const Tensor& t) { std::vector<float> v(t.numel()); for (int64_t i = 0; i < t.numel(); ++i) v[i] = t.ptr<float>()[i]; return v; };

  if (!bins_arg.get().empty()) {  // multi-context group load
    std::vector<std::string> bins;
    std::string s = bins_arg.get(), tok;
    size_t p = 0, q;
    while ((q = s.find(',', p)) != std::string::npos) { bins.push_back(s.substr(p, q - p)); p = q + 1; }
    bins.push_back(s.substr(p));
    mllm::initQnnBackendGroup(bins);
  } else {
    mllm::initQnnBackend(model_path.get());
  }

  std::vector<float> hidden_v = tovec(consts->pull("embed_row"));  // [hidden]

  for (int L = 0; L < NL; ++L) {
    const std::string li = "l" + std::to_string(L);
    // ---- mixer ----
    auto r = hidden_v;
    auto h = rmsnorm(hidden_v, consts->pull(li + ".operator_norm_w"));
    std::vector<float> mix;
    if (isAttn(L)) {
      auto pk = tovec(consts->pull(li + ".past_k"));
      auto pv = tovec(consts->pull(li + ".past_v"));
      auto res = dispatch("attn_" + li,
                          {{{1, hidden}, h},
                           {{1, 1, rot}, tovec(consts->pull("sin"))},
                           {{1, 1, rot}, tovec(consts->pull("cos"))},
                           {{1, 1, D}, tovec(consts->pull(li + ".q_norm_w"))},
                           {{1, 1, D}, tovec(consts->pull(li + ".k_norm_w"))},
                           {{1, 1, 1}, {eps}},
                           {{1, KV, D, P}, pk},
                           {{1, KV, P, D}, pv},
                           {{1, 1, 1, ctx}, tovec(consts->pull(li + ".mask"))}},
                          {{1, hidden}, {1, KV, D, 1}, {1, KV, 1, D}});
      mix = res[0];
    } else {
      auto res = dispatch("conv_" + li,
                          {{{1, hidden}, h}, {{1, K, hidden}, tovec(consts->pull(li + ".cw"))},
                           {{1, K - 1, hidden}, tovec(consts->pull(li + ".cs"))}},
                          {{1, hidden}, {1, K - 1, hidden}});
      mix = res[0];
    }
    for (int i = 0; i < hidden; ++i) hidden_v[i] = r[i] + mix[i];

    // ---- ffn ----
    auto r2 = hidden_v;
    auto h2 = rmsnorm(hidden_v, consts->pull(li + ".ffn_norm_w"));
    std::vector<float> ffn(hidden, 0.f);
    if (isDense(L)) {
      ffn = dispatch("ffn_" + li, {{{1, hidden}, h2}}, {{1, hidden}})[0];
    } else {
      // host router (sigmoid(h2@gate^T)+bias -> top-k -> /(sum+1e-6) -> *scaling), then dispatch selected experts.
      auto router = mllm::load(dir_arg.get() + "/" + li + "_moe/router.mllm", mllm::ModelFileVersion::kV2, mllm::kCPU, false);
      auto gw = router->pull("gate_weight"); auto bias = router->pull("expert_bias");
      auto meta = router->pull("meta"); auto scaling = router->pull("scaling").ptr<float>()[0];
      const int E = meta.ptr<int32_t>()[1], top_k = meta.ptr<int32_t>()[2];
      const bool norm_topk = meta.ptr<int32_t>()[5] != 0;
      const float* gwp = gw.ptr<float>(); const float* bp = bias.ptr<float>();
      std::vector<float> sig(E), score(E);
      for (int e = 0; e < E; ++e) {
        double acc = 0; const float* row = gwp + (size_t)e * hidden;
        for (int i = 0; i < hidden; ++i) acc += (double)h2[i] * row[i];
        sig[e] = 1.f / (1.f + std::exp(-(float)acc)); score[e] = sig[e] + bp[e];
      }
      std::vector<int> ord(E); std::iota(ord.begin(), ord.end(), 0);
      std::partial_sort(ord.begin(), ord.begin() + top_k, ord.end(),
                        [&](int a, int b) { return score[a] > score[b] || (score[a] == score[b] && a < b); });
      float wsum = 0; for (int j = 0; j < top_k; ++j) wsum += sig[ord[j]];
      float inv = norm_topk ? (1.f / (wsum + 1e-6f)) : 1.f;
      // Grouped experts: dispatch only the groups containing a selected expert; each group
      // graph returns GS outputs, take the selected positions and weighted-sum them.
      const int GS = 8;
      std::vector<float> wsel(top_k);
      for (int j = 0; j < top_k; ++j) wsel[j] = sig[ord[j]] * inv * scaling;
      std::set<int> groups;
      for (int j = 0; j < top_k; ++j) groups.insert(ord[j] / GS);
      for (int g : groups) {
        std::vector<std::vector<int>> osh(GS, std::vector<int>{1, hidden});
        auto gouts = dispatch("moe_" + li + "_g" + std::to_string(g), {{{1, hidden}, h2}}, osh);
        for (int j = 0; j < top_k; ++j) {
          if (ord[j] / GS == g) {
            int pos = ord[j] % GS;
            for (int i = 0; i < hidden; ++i) ffn[i] += wsel[j] * gouts[pos][i];
          }
        }
      }
    }
    for (int i = 0; i < hidden; ++i) hidden_v[i] = r2[i] + ffn[i];

    // ---- validate residual stream after layer L vs golden (HF fp32; loose int4 tol) ----
    auto gold = consts->pull(li + ".golden_hidden");
    double e = 0, range = 0;
    for (int i = 0; i < hidden; ++i) {
      e = std::max(e, (double)std::fabs(hidden_v[i] - gold.ptr<float>()[i]));
      range = std::max(range, (double)std::fabs(gold.ptr<float>()[i]));
    }
    fmt::print("[layer {} {}+{}] hidden vs golden: max|err|={:.4f} |h|max={:.4f} rel={:.2f}pct\n", L,
               isAttn(L) ? "attn" : "conv", isDense(L) ? "dense" : "moe", e, range, 100.0 * e / (range + 1e-9));
  }

  // ---- lm_head (final RMSNorm in-graph + LPBQ proj) -> argmax ----
  if (head_arg.isSet()) {
    auto logits = dispatch("lm_head",
                           {{{1, hidden}, hidden_v},
                            {{1, 1, hidden}, tovec(consts->pull("embedding_norm_w"))},
                            {{1, 1, 1}, {eps}}},
                           {{1, vocab}})[0];
    int am = 0; float best = logits[0];
    for (int i = 1; i < vocab; ++i) { if (logits[i] > best) { best = logits[i]; am = i; } }
    int gold = consts->pull("golden_argmax").ptr<int32_t>()[0];
    int deq = consts->has("dequant_argmax") ? consts->pull("dequant_argmax").ptr<int32_t>()[0] : gold;
    fmt::print("[lm_head] device argmax={} ({:.4f})  dequant-golden={}  HF-golden={}  -> {}\n", am, best, deq, gold,
               (am == deq || am == gold) ? "MATCH" : "MISMATCH");
  } else {
    fmt::print("[decode] HF-golden next_tok={} ; stack ran {} layers.\n",
               consts->pull("golden_argmax").ptr<int32_t>()[0], NL);
  }
  mllm::shutdownContext();
});
