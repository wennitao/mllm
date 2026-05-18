// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/opencl/OpenCLDispatcher.hpp"
#include "mllm/backends/opencl/OpenCLBackend.hpp"
#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/tracy_perf/Tracy.hpp"

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"
#endif

namespace mllm::opencl {

OpenCLDispatcher::OpenCLDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                                   const OpenCLDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void OpenCLDispatcher::receive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule:
    case TaskTypes::kExecuteOp: {
      process(task);
      break;
    }
    default: NYI("Only execute op task is supported receive");
  }
}

TaskResult::sender_t OpenCLDispatcher::asyncReceive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule: {
      MLLM_EMPTY_SCOPE;
      break;
    }
    default: NYI("Only execute module task is supported asyncReceive");
  }
  auto scheduler = thread_pool_.get_scheduler();
  return stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });
}

void OpenCLDispatcher::process(const Task::ptr_t& task) {
  MLLM_TRACY_ZONE_SCOPED;
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
#ifdef MLLM_PERFETTO_ENABLE
      // One slice per op covering reshape + setup + forward + clFinish, so the
      // duration is the real GPU wall-clock for this op (not just the
      // host-side clEnqueueNDRangeKernel return time). Costs a per-op sync —
      // only paid in MLLM_PERFETTO_ENABLE builds.
      auto op_name = optype2Str(task->op->getOpType());
      MLLM_PERF_TRACE_BEGIN("mllm.kernel", perfetto::DynamicString{op_name}, [&](perfetto::EventContext ctx) {
        int cnt = 0;
        for (auto& i : task->inputs) {
          ctx.AddDebugAnnotation(perfetto::DynamicString{"inputs-" + std::to_string(cnt++)}, i.shape());
        }
      });
#endif
      auto op = task->op;
      auto& inputs = task->inputs;
      auto& outputs = task->outputs;
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
#ifdef MLLM_PERFETTO_ENABLE
      {
        auto backend = std::static_pointer_cast<OpenCLBackend>(Context::instance().getBackend(kOpenCL));
        if (backend && backend->runtime()) { backend->runtime()->commandQueue().finish(); }
      }
      MLLM_PERF_TRACE_END("mllm.kernel");
#endif
      break;
    }
    case TaskTypes::kExecuteModule: {
      task->outputs = ((nn::Module*)(task->custom_context_ptr))->forward(task->inputs, task->args);
      break;
    }
    default: NYI("OpenCLDispatcher::process not supported task type");
  }
}

void OpenCLDispatcher::syncWait() {
  // Drain the GPU queue so wall-clock timers in callers (e.g. ModuleProfiler
  // in Module::__main) reflect actual GPU completion, not enqueue return.
  auto backend = std::static_pointer_cast<OpenCLBackend>(Context::instance().getBackend(kOpenCL));
  if (!backend) return;
  auto runtime = backend->runtime();
  if (!runtime) return;
  runtime->commandQueue().finish();
}

OpenCLDispatcher::ptr_t createOpenCLDispatcher(exec::static_thread_pool& thread_pool, const OpenCLDispatcherOptions& options) {
  return std::make_shared<OpenCLDispatcher>(thread_pool, Dispatcher::opencl_dispatcher_id, options);
}

}  // namespace mllm::opencl