// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <chrono>
#include <string>

namespace mllm::engine {

// Per-module wall-clock profiler. Records (phase, call_idx, module_name, us)
// rows from Module::__main, intended for backends where ops dispatch
// asynchronously (OpenCL) and per-op cycle counters aren't available.
//
// Usage:
//   ModuleProfiler::setEnabled(true);
//   ModuleProfiler::setPhase("prefill");      // before first prefill call
//   ModuleProfiler::nextCall();               // before each model.forward()
//   ...
//   ModuleProfiler::setPhase("decode");
//   ModuleProfiler::nextCall();
//   ...
//   ModuleProfiler::dumpCSV("opencl_profile.csv");
class ModuleProfiler {
 public:
  using clock = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  static void setEnabled(bool on);
  static bool isEnabled();

  static void setPhase(const std::string& tag);
  static void nextCall();

  // Append one row. Caller takes timestamps before+after the module dispatch
  // (and a syncWait on the device for async backends).
  static void record(const std::string& module_name, time_point t0, time_point t1);

  static void dumpCSV(const std::string& path);
  static void clear();
};

}  // namespace mllm::engine
