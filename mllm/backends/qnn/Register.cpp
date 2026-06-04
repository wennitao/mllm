// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include <filesystem>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "CustomLayers.hpp"

namespace mllm {

// export initQnnBackend function to initialize QNN backend
void initQnnBackend(const std::string& context_path) {
  MLLM_RT_ASSERT(isQnnAvailable());
  auto& ctx = Context::instance();

  // 1. Register backend
  auto backend = std::make_shared<qnn::QNNBackend>();
  if (std::filesystem::exists(context_path)) {
    MLLM_INFO("QNN context path exists: {}", context_path);
    if (!backend->loadContext(context_path)) {
      MLLM_ERROR_EXIT(1, "Failed to load QNN context from {}", context_path);
    } else {
      MLLM_INFO("QNN context loaded successfully from {}", context_path);
    }
  } else {
    if (!backend->createContext()) {
      MLLM_ERROR_EXIT(1, "Failed to create QNN context");
    } else {
      MLLM_INFO("QNN context created successfully");
    }
  }
  ctx.registerBackend(backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kQNN, backend->allocator(),
                                         {
                                             .really_large_tensor_threshold = 0,
                                             .using_buddy_mem_pool = false,
                                         });
  MLLM_INFO("QNN memory manager registered");

  // 3. Initialize dispatcher manager
  ctx.dispatcherManager()->registerDispatcher(
      createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), qnn::QNNDispatcherOptions()));

  // register QNN custom ops
  Context::instance().registerCustomizedOp(kQNN, "DequantizeAdd",
                                           std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::DequantizeAddFactory())));
}

// Multi-context (grouped-bin) init: load several .bin partitions of one model as
// a single spill-fill group. Mirrors initQnnBackend but uses loadContextGroup.
void initQnnBackendGroup(const std::vector<std::string>& context_paths) {
  MLLM_RT_ASSERT(isQnnAvailable());
  auto& ctx = Context::instance();

  // 1. Register backend
  auto backend = std::make_shared<qnn::QNNBackend>();
  for (const auto& p : context_paths) {
    if (!std::filesystem::exists(p)) { MLLM_ERROR_EXIT(1, "QNN context bin not found: {}", p); }
  }
  if (!backend->loadContextGroup(context_paths)) {
    MLLM_ERROR_EXIT(1, "Failed to load QNN context group ({} bins)", context_paths.size());
  }
  MLLM_INFO("QNN context group loaded successfully ({} bins)", context_paths.size());
  ctx.registerBackend(backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kQNN, backend->allocator(),
                                         {
                                             .really_large_tensor_threshold = 0,
                                             .using_buddy_mem_pool = false,
                                         });

  // 3. Initialize dispatcher manager
  ctx.dispatcherManager()->registerDispatcher(
      createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), qnn::QNNDispatcherOptions()));

  // register QNN custom ops
  Context::instance().registerCustomizedOp(kQNN, "DequantizeAdd",
                                           std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::DequantizeAddFactory())));
}
}  // namespace mllm
