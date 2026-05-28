// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/base/Allocator.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include <map>
#include <memory>
#include <mutex>

namespace mllm::opencl {

class OpenCLAllocator final : public Allocator {
 public:
  explicit OpenCLAllocator(std::shared_ptr<OpenCLRuntime> runtime);
  ~OpenCLAllocator();

  inline bool ctrlByMemManager() override {
    // OpenCLAllocator manages its own memory pool
    return false;
  }

  bool alloc(Storage* storage) override;
  bool alloc(const Storage::ptr_t& storage) override;
  void free(Storage* storage) override;
  void free(const Storage::ptr_t& storage) override;
  bool generalAlloc(void** ptr, size_t cap, size_t align) override;
  void generalFree(void* ptr) override;
  size_t allocSize(Storage* storage) override;
  size_t allocSize(const Storage::ptr_t& storage) override;
  [[nodiscard]] size_t alignSize() const override;

  // Zero-copy alias: create a cl_mem that maps the SAME physical pages as an
  // ION (dmabuf) buffer the host already owns — typically a QNN rpcmem
  // allocation. Uses Qualcomm's cl_qcom_ion_host_ptr extension. The returned
  // cl_mem must be released by the caller (NOT pool-managed — destroying it
  // doesn't affect the underlying ION allocation, only this OpenCL view).
  //
  // host_cache_policy: CL_MEM_HOST_UNCACHED_QCOM is safest for shared CPU↔GPU
  // access (matches rpcmem's typical cache mode). WRITEBACK is faster for
  // CPU-only reuse but needs explicit flushes before handing back to NPU.
  //
  // Returns nullptr on failure (e.g., extension not supported, fd invalid).
  cl_mem createIonAlias(int ion_fd, void* hostptr, size_t size,
                        cl_uint host_cache_policy = 0x40A4 /*UNCACHED*/);

 private:
  std::multimap<size_t, cl_mem> memory_pool_;
  std::mutex pool_mutex_;
  std::shared_ptr<OpenCLRuntime> runtime_;
};

}  // namespace mllm::opencl
