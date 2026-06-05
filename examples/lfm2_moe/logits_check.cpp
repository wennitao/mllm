// Dump LFM2.5-8B-A1B last-token logits for a fixed input id sequence, for numerical
// comparison against the HF reference.
//
//   mllm-lfm2_moe-logits -m lfm2.5-8b-a1b.mllm -c <dir>/config.json --ids lfm2_ref.ids.txt --out lfm2_mllm_logits.bin
//
// --ids : a text file of whitespace-separated int64 token ids (one sequence).
// --out : raw float32 little-endian vector of length vocab_size (last position).
#include <algorithm>
#include <fstream>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/lfm2_moe/configuration_lfm2_moe.hpp>
#include <mllm/models/lfm2_moe/modeling_lfm2_moe.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Path to .mllm model").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("config.json path").required(true);
  auto& ids_path = Argparse::add<std::string>("--ids").help("text file of whitespace-separated int64 ids").required(true);
  auto& out_path = Argparse::add<std::string>("--out").help("output float32 logits file").def("lfm2_mllm_logits.bin");
  Argparse::parse(argc, argv);

  auto cfg = mllm::models::lfm2_moe::Lfm2MoeConfig(config_path.get());
  auto model = mllm::models::lfm2_moe::Lfm2MoeForCausalLM(cfg);
  auto param = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  model.load(param);

  std::vector<int64_t> ids;
  {
    std::ifstream f(ids_path.get());
    long long v;
    while (f >> v) { ids.push_back((int64_t)v); }
  }
  if (ids.empty()) {
    fmt::print("no ids read from {}\n", ids_path.get());
    mllm::shutdownContext();
    return 1;
  }
  int S = (int)ids.size();
  auto seq = mllm::Tensor::empty({1, S}, mllm::kInt64, mllm::kCPU).alloc();
  auto* sp = seq.ptr<int64_t>();
  for (int i = 0; i < S; ++i) { sp[i] = ids[i]; }

  auto out = model.forward({{"sequence", seq}}, {});
  auto logits = out.at("sequence");
  const auto& shp = logits.shape();
  int V = shp[(int)shp.size() - 1];
  const float* lp = logits.ptr<float>();

  {
    std::ofstream g(out_path.get(), std::ios::binary);
    g.write(reinterpret_cast<const char*>(lp), sizeof(float) * (size_t)V);
  }

  // argmax + top-10
  std::vector<int> idx(V);
  for (int i = 0; i < V; ++i) { idx[i] = i; }
  int K = std::min(10, V);
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                    [&](int a, int b) { return lp[a] > lp[b]; });
  fmt::print("S={} V={} argmax={} ({:.6f})\n", S, V, idx[0], lp[idx[0]]);
  fmt::print("top10:");
  for (int i = 0; i < K; ++i) { fmt::print(" {}:{:.4f}", idx[i], lp[idx[i]]); }
  fmt::print("\nwrote {} ({} floats)\n", out_path.get(), V);

  mllm::shutdownContext();
})
