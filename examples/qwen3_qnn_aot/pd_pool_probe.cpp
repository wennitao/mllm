// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// PD-pooling / cap-locality proof probe.
//
// The multi-context-split plan (fit a 28-layer block-sparse model under the V79
// HTP PD cap by compiling layers as several context bins loaded into ONE
// shared-spill-fill group) rests on this open question: is the ~3.6 GB PD cap
// PER-CONTEXT (so splitting into bins trivially fits) or PER-DEVICE-TOTAL (so
// splitting only saves the shared spill-fill delta)? The only on-device proof so
// far (QNNBackend::beginAuxContext) used a WEIGHTLESS aux graph, which says
// nothing about co-resident weight-bearing bins.
//
// This probe loads a LIST of real weight-bearing context bins into ONE spill-fill
// group (bins[0] = anchor firstGroupHandle=0, the rest = joiners
// firstGroupHandle=anchor) and reports HOW MANY reserve PD before the device runs
// out. Use -n to repeat one bin (capacity sweep): each ~877 MB 4L bin co-resident.
//
//   per-context cap  => loaded keeps growing well past 3.6 GB total => plan GO
//   per-device-total => loaded caps where cumulative PD ~= 3.6 GB   => split only
//                       helps via spill-fill sharing (marginal)
//
// No graph execute is needed — PD is reserved at contextCreateFromBinary.
// Raise QNN/HTP log verbosity with MLLM_QNN_LOG_LEVEL=VERBOSE to capture the
// skel's "context size estimate <bytes>" / "Failed to find available PD" lines.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mllm/mllm.hpp>

#include "mllm/backends/qnn/QNNBackend.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& bins_arg =
      Argparse::add<std::string>("-a|--bins").help("Comma-separated context bin paths, loaded in order into ONE group").required(true);
  auto& repeat = Argparse::add<int>("-n|--repeat").help("Repeat the bin list N times (capacity sweep)").def(1);
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  uint64_t sf_mb = 128;
  if (const char* e = std::getenv("MLLM_QNN_SPILLFILL_MB")) {
    uint64_t v = std::strtoull(e, nullptr, 10);
    if (v > 0) { sf_mb = v; }
  }

  // Split the comma-separated list, then repeat it N times.
  std::vector<std::string> base;
  {
    const std::string s = bins_arg.get();
    size_t p = 0;
    while (true) {
      size_t c = s.find(',', p);
      base.push_back(c == std::string::npos ? s.substr(p) : s.substr(p, c - p));
      if (c == std::string::npos) { break; }
      p = c + 1;
    }
  }
  std::vector<std::string> bins;
  const int reps = repeat.get() < 1 ? 1 : repeat.get();
  for (int r = 0; r < reps; ++r) {
    for (const auto& b : base) { bins.push_back(b); }
  }

  std::cout << "PD_POOL_PROOF nbins=" << bins.size() << " sf_mb=" << sf_mb << std::endl;

  auto backend = std::make_shared<mllm::qnn::QNNBackend>();
  int loaded = backend->loadBinsOneGroup(bins, sf_mb);

  std::cout << "\nPD_POOL_PROOF loaded=" << loaded << "/" << bins.size() << std::endl;
  return loaded == static_cast<int>(bins.size()) ? 0 : 2;
});
