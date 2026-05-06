// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/ModuleProfiler.hpp"

#include <atomic>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace mllm::engine {

namespace {

struct Row {
  std::string phase;
  int call;
  std::string name;
  double us;
};

std::atomic<bool> g_enabled{false};
std::mutex g_mu;
std::string g_phase = "init";
int g_call = -1;
std::vector<Row> g_rows;

}  // namespace

void ModuleProfiler::setEnabled(bool on) { g_enabled.store(on, std::memory_order_relaxed); }

bool ModuleProfiler::isEnabled() { return g_enabled.load(std::memory_order_relaxed); }

void ModuleProfiler::setPhase(const std::string& tag) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_phase = tag;
  g_call = -1;
}

void ModuleProfiler::nextCall() {
  std::lock_guard<std::mutex> lk(g_mu);
  ++g_call;
}

void ModuleProfiler::record(const std::string& module_name, time_point t0, time_point t1) {
  if (!isEnabled()) return;
  if (module_name.empty()) return;
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

  std::lock_guard<std::mutex> lk(g_mu);
  g_rows.push_back(Row{g_phase, g_call < 0 ? 0 : g_call, module_name, us});
}

void ModuleProfiler::dumpCSV(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_mu);
  std::ofstream out(path);
  if (!out) return;
  out << "phase,call,module_name,us\n";
  for (const auto& r : g_rows) { out << r.phase << "," << r.call << "," << r.name << "," << r.us << "\n"; }
}

void ModuleProfiler::clear() {
  std::lock_guard<std::mutex> lk(g_mu);
  g_rows.clear();
  g_phase = "init";
  g_call = -1;
}

}  // namespace mllm::engine
