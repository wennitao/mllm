#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"
#include "mllm/backends/base/Backend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/backends/qnn/QNNModel.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

static const std::string QNN_Custom_Op_Package = "LLaMAPackage";
static const std::string QNN_Context_File = "qnn_context.bin";

enum class ProfilingLevel { OFF, BASIC, DETAILED, INVALID };
class QNNPerf {
 public:
  static std::unique_ptr<QNNPerf> create(const QNN_INTERFACE_VER_TYPE* qnnInterface) {
    return std::make_unique<QNNPerf>(qnnInterface);
  }
  explicit QNNPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface);
  ~QNNPerf();

  // Explicitly destroy power config. Call this while QNN HTP infrastructure is still alive.
  void shutdown();

  void setRpcLatencyAndPolling();
  void setPowerConfigBurst();
  void setPowerConfigBalanced();

 private:
  const QNN_INTERFACE_VER_TYPE* qnnInterface_ = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perfInfra_{};
  uint32_t powerConfigId_ = 0;
  QnnHtpPerfInfrastructure_PowerConfig_t powerConfigBurst_{};
  QnnHtpPerfInfrastructure_PowerConfig_t powerConfigBalanced_{};
  bool isShutdown_ = false;
};

class QNNRuntime {
  friend class QNNBackend;

 public:
  ~QNNRuntime();

  static std::unique_ptr<QNNRuntime> create(ProfilingLevel profilingLevel = ProfilingLevel::OFF,
                                            QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_VERBOSE) {
    return std::unique_ptr<QNNRuntime>(initRuntime(profilingLevel, qnnLogLevel));
  }

  bool createContext(Qnn_ContextHandle_t& context, QnnContext_Config_t** contextConfig = nullptr);
  bool retrieveContext(const std::string& contextBinaryPath, Qnn_ContextHandle_t& context,
                       std::vector<std::shared_ptr<QNNModel>>& qnnModels, QnnContext_Config_t** contextConfig = nullptr);

 private:
  QNN_INTERFACE_VER_TYPE qnnInterface;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;

  Qnn_LogHandle_t logHandle = nullptr;
  Qnn_BackendHandle_t backendHandle = nullptr;
  Qnn_DeviceHandle_t deviceHandle = nullptr;
  Qnn_ProfileHandle_t profileHandle = nullptr;

  QNNRuntime(QNN_INTERFACE_VER_TYPE qnnInterface, QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface,
             Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle,
             Qnn_ProfileHandle_t qnnProfileHandle = nullptr)
      : qnnInterface(qnnInterface),
        qnnSystemInterface(qnnSystemInterface),
        logHandle(qnnLogHandle),
        backendHandle(qnnBackendHandle),
        deviceHandle(qnnDeviceHandle),
        profileHandle(qnnProfileHandle) {}

  std::string getBackendBuildId(QNN_INTERFACE_VER_TYPE& qnnInterface) {
    char* backendBuildId{nullptr};
    if (QNN_SUCCESS != qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
      MLLM_ERROR_EXIT(1, "Unable to get build Id from the backend.");
    }
    return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
  }

  static QNNRuntime* initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel);
};

class QNNBackend final : public Backend {
 public:
  QNNBackend();
  ~QNNBackend();

  bool loadContext(const std::string& contextPath);
  // Load SEVERAL context bins as ONE spill-fill group (multi-context split):
  // bins[0] anchors a new group (firstGroupHandle=0), the rest JOIN it
  // (firstGroupHandle=anchor). All graphs across all bins merge into one index
  // map; graphExecute routes each to its owning context, and the allocator keys
  // registrations by (ptr,context) so seam buffers register against both bins.
  // Used to fit a model whose single-context PD reservation would exceed the
  // ~3.6 GB per-context ceiling. A 1-path list falls back to loadContext.
  bool loadContextGroup(const std::vector<std::string>& contextPaths);
  bool createContext();
  void saveContext(const std::string& contextPath = "qnn_context.bin");

  // Open a SECOND HTP context for building extra graphs at runtime (e.g. the
  // block-selection score matmul) alongside the loaded binary model context
  // (which is immutable). The aux context joins the model context's spill-fill
  // group (firstGroupHandle = model context) so it shares the model's spill-fill
  // buffer instead of reserving its own — the documented multi-context pattern.
  // While open, context_/allocator point at the aux context, so the normal
  // createQnnGraph/addTensor/graphAddNode/graphFinalize + kQNN tensor allocs all
  // land in (and memRegister against) the aux context. Call endAuxContext()
  // when done; graphExecute() afterwards still works (graphs hold their handle).
  bool beginAuxContext(uint64_t max_spill_fill_mb);
  void endAuxContext();

  // PD-pooling / cap-locality proof: load a LIST of weight-bearing context bins
  // into ONE HTP spill-fill group (bins[0] = anchor, firstGroupHandle=0; the
  // rest join via firstGroupHandle=anchor) and report how many reserve PD before
  // the device runs out. Determines whether the ~3.6 GB V79 PD cap is
  // per-context (so splitting 28L into bins trivially fits) or per-device-total
  // (so splitting only saves the shared spill-fill delta) — the open question
  // for the multi-context-split plan. beginAuxContext only proved the join for a
  // WEIGHTLESS aux graph; here every bin carries weights. No graph execute: PD
  // is reserved at contextCreateFromBinary. Returns the number of bins loaded.
  int loadBinsOneGroup(const std::vector<std::string>& bins, uint64_t sf_mb);

  bool isWeightOnDevice() override { return false; }

  // QNN Graph build interfaces
  std::shared_ptr<QNNModel> createQnnGraph(const std::string& graphName);

  void graphAddNode(const std::string& graphName, const std::string& nodeName, const std::string& nodeType,
                    const std::vector<std::string>& inputTensorNames, const std::vector<std::string>& outputTensorNames,
                    const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                    const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                    const std::string& packageName = "qti.aisw");

  bool graphFinalize(const std::string& graphName);

  void graphExecute(const std::string& graphName, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

  // Clear a graph's I/O tensor-wrapper bindings so the next graphExecute
  // re-binds them to the tensors passed in that call (for dispatching one AOT
  // graph against alternating double-buffer slots). Cheap: registration cached.
  void resetGraphIOForRebind(const std::string& graphName);

  // Tensor management interfaces
  bool addTensor(const std::string& graphName, const std::string& tensorName, Qnn_TensorType_t type, const Tensor& tensor,
                 Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  bool addStaticTensor(const std::string& graphName, const std::string& tensorName, const Tensor& tensor,
                       Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  // Get tensor wrapper by name from specific graph
  std::shared_ptr<QNNTensorWrapper> getTensorWrapper(const std::string& graphName, const std::string& tensorName);

  // Getters for runtime components
  [[nodiscard]] const QNN_INTERFACE_VER_TYPE& qnnInterface() const { return runtime_->qnnInterface; }
  [[nodiscard]] Qnn_BackendHandle_t backendHandle() const { return runtime_->backendHandle; }
  [[nodiscard]] Qnn_ContextHandle_t context() const { return context_; }

 private:
  bool debug_;  // controlled by -DMLLM_QNN_DEBUG compile flag
  ProfilingLevel profilingLevel_;
  Qnn_ContextHandle_t context_ = nullptr;
  Qnn_ContextHandle_t aux_context_ = nullptr;   // 2nd context for runtime-built graphs
  Qnn_ContextHandle_t main_context_ = nullptr;  // saved model context while aux is active
  std::vector<Qnn_ContextHandle_t> group_contexts_;  // all model bins when loaded as a group
  std::unique_ptr<QNNRuntime> runtime_;
  std::unique_ptr<QNNPerf> perf_;

  // Hold QNN library handles to control unload order
  // These libraries will only be unloaded when QNNBackend is destroyed
  void* qnnHtpLibHandle_ = nullptr;
  void* qnnSystemLibHandle_ = nullptr;

  // Graph management
  std::map<std::string, int> qnnModelIndexMap_;
  std::vector<std::shared_ptr<QNNModel>> qnnModels_;
  int currentQnnModelIndex_ = -1;

  // Helper methods
  void extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);
};

}  // namespace mllm::qnn
