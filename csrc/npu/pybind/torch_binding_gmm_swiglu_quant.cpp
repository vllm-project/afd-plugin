// SPDX-License-Identifier: MIT
// Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
// Copyright contributors to the AFD plugin project
//
// Adapted from cam_async/src/comm_operator/pybind/gmm_swiglu_quant_v2_layered.cpp.
// Registered under the plugin-owned afd_ascend namespace (umdk uses
// umdk_cam_op_lib). The ACLNN call, output allocation, and Meta/autograd
// dispatch are retained; only the namespace and the schema lookup name change.

#include <torch/extension.h>
#include <torch/library.h>

#include "pytorch_extension/op_api_common.h"
#include "grouped_matmul_swiglu_quant_v2_layered/op_api/aclnn_grouped_matmul_swiglu_quant_v2_layered.h"

namespace afd_plugin::gmm_swiglu_quant {
namespace {

using tensor_list = std::vector<at::Tensor>;

// quantMode == 2 selects mx-style quantization (per-64-block scale pairs).
constexpr int64_t QUANT_MODE_MX = 2;

// Source: gmm_swiglu_quant_v2_layered.cpp::AllocSwigluQuantOutputs.
// Output shapes follow the op infershape: m from x_scale dim0, n from the
// per-channel weight scale last dim (halved by swiglu).
void AllocSwigluQuantOutputs(const at::Tensor &xScale,
                             const at::TensorList &allWeightScale,
                             const int64_t quantMode, at::Tensor &yOut,
                             at::Tensor &yScaleOut) {
  int64_t m = xScale.size(0);
  int64_t n = allWeightScale[0].size(-1);
  int64_t nAfterHalve = n / 2;
  yOut = at::empty({m, nAfterHalve}, xScale.options().dtype(at::kChar));
  if (quantMode == QUANT_MODE_MX) {
    // mx-style quant: per-64-block scale pairs
    int64_t nAfterSplit = (nAfterHalve + 63) / 64;
    yScaleOut = at::empty({m, nAfterSplit, 2},
                          xScale.options().dtype(at::kFloat));
  } else {
    yScaleOut = at::empty({m}, xScale.options().dtype(at::kFloat));
  }
}

// Source: gmm_swiglu_quant_v2_layered.cpp::cam_gmm_swiglu_quant_v2_layered_impl_npu.
tensor_list impl_npu(
    const at::Tensor &x, const at::TensorList &allWeight,
    const at::TensorList &allWeightScale,
    const at::TensorList &allWeightAssistMatrix, const at::Tensor &xScale,
    const at::Tensor &groupList, const at::Tensor &layerIndex,
    const int64_t dequantMode, const int64_t quantMode,
    const int64_t groupListType,
    const c10::optional<std::vector<int64_t>> &tuningConfigOptional) {
  at::Tensor yOut;
  at::Tensor yScaleOut;
  AllocSwigluQuantOutputs(xScale, allWeightScale, quantMode, yOut, yScaleOut);

  // bias / smoothScale are not exposed: bias must stay null on the A8W4/A8W8
  // paths and smoothScale is A4W4-only; pass empty placeholders the op_api
  // normalizes to nullptr.
  at::Tensor biasPlaceholder;
  at::Tensor smoothScalePlaceholder;

  c10::optional<at::IntArrayRef> tuningConfigRef;
  if (tuningConfigOptional.has_value() && tuningConfigOptional->size() > 0) {
    tuningConfigRef =
        at::IntArrayRef(tuningConfigOptional->data(), tuningConfigOptional->size());
  }

  // EXEC_NPU_CMD's parameter packing requires lvalues; materialize the derived
  // attrs. The aclnn API has no quant_dtype argument -- the op_api derives it
  // from the output tensor's dtype via output->GetDataType().
  int64_t dequantDtype = static_cast<int64_t>(at::kFloat);
  EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantV2Layered,
               x, allWeight, allWeightScale, allWeightAssistMatrix,
               biasPlaceholder, xScale, smoothScalePlaceholder, groupList,
               layerIndex, dequantMode, dequantDtype, quantMode, groupListType,
               tuningConfigRef, yOut, yScaleOut);
  return {yOut, yScaleOut};
}

// Source: gmm_swiglu_quant_v2_layered.cpp::cam_gmm_swiglu_quant_v2_layered_impl_meta.
tensor_list impl_meta(
    const at::Tensor &x, const at::TensorList &allWeight,
    const at::TensorList &allWeightScale,
    const at::TensorList &allWeightAssistMatrix, const at::Tensor &xScale,
    const at::Tensor &groupList, const at::Tensor &layerIndex,
    const int64_t dequantMode, const int64_t quantMode,
    const int64_t groupListType,
    const c10::optional<std::vector<int64_t>> &tuningConfigOptional) {
  at::Tensor yOut;
  at::Tensor yScaleOut;
  AllocSwigluQuantOutputs(xScale, allWeightScale, quantMode, yOut, yScaleOut);
  return {yOut, yScaleOut};
}

// Source: gmm_swiglu_quant_v2_layered.cpp::cam_gmm_swiglu_quant_v2_layered_impl.
tensor_list impl_dispatch(
    const at::Tensor &x, const at::TensorList &allWeight,
    const at::TensorList &allWeightScale,
    const at::TensorList &allWeightAssistMatrix, const at::Tensor &xScale,
    const at::Tensor &groupList, const at::Tensor &layerIndex,
    const int64_t dequantMode, const int64_t quantMode,
    const int64_t groupListType,
    const c10::optional<std::vector<int64_t>> &tuningConfigOptional) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow(
                           "afd_ascend::gmm_swiglu_quant_v2_layered", "")
                       .typed<decltype(impl_dispatch)>();
  return op.call(x, allWeight, allWeightScale, allWeightAssistMatrix, xScale,
                 groupList, layerIndex, dequantMode, quantMode, groupListType,
                 tuningConfigOptional);
}

// Source: gmm_swiglu_quant_v2_layered.cpp::ExtCamGmmSwigluQuantV2Layered.
// forward/backward via torch::autograd::Function subclass. Backward is
// intentionally empty (gradients stop here): the operator is inference-only.
class ExtGmmSwigluQuantV2Layered
    : public torch::autograd::Function<ExtGmmSwigluQuantV2Layered> {
 public:
  static tensor_list forward(
      torch::autograd::AutogradContext *ctx, const at::Tensor &x,
      const at::TensorList &allWeight, const at::TensorList &allWeightScale,
      const at::TensorList &allWeightAssistMatrix, const at::Tensor &xScale,
      const at::Tensor &groupList, const at::Tensor &layerIndex,
      const int64_t dequantMode, const int64_t quantMode,
      const int64_t groupListType,
      const c10::optional<std::vector<int64_t>> &tuningConfigOptional) {
    at::AutoDispatchBelowADInplaceOrView guard;

    auto result = impl_dispatch(x, allWeight, allWeightScale,
                                allWeightAssistMatrix, xScale, groupList,
                                layerIndex, dequantMode, quantMode,
                                groupListType, tuningConfigOptional);
    return result;
  }

  static tensor_list backward(torch::autograd::AutogradContext *ctx,
                              tensor_list grad_outputs) {
    return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

// Source: gmm_swiglu_quant_v2_layered.cpp::cam_gmm_swiglu_quant_v2_layered_impl_autograd.
tensor_list impl_autograd(
    const at::Tensor &x, const at::TensorList &allWeight,
    const at::TensorList &allWeightScale,
    const at::TensorList &allWeightAssistMatrix, const at::Tensor &xScale,
    const at::Tensor &groupList, const at::Tensor &layerIndex,
    const int64_t dequantMode, const int64_t quantMode,
    const int64_t groupListType,
    const c10::optional<std::vector<int64_t>> &tuningConfigOptional) {
  auto result = ExtGmmSwigluQuantV2Layered::apply(
      x, allWeight, allWeightScale, allWeightAssistMatrix, xScale, groupList,
      layerIndex, dequantMode, quantMode, groupListType, tuningConfigOptional);
  return result;
}

}  // namespace
}  // namespace afd_plugin::gmm_swiglu_quant

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
  ops.def("gmm_swiglu_quant_v2_layered(Tensor x, Tensor[] all_weight, "
          "Tensor[] all_weight_scale, Tensor[] all_weight_assist_matrix, "
          "Tensor x_scale, Tensor group_list, Tensor layer_index, "
          "int dequant_mode=0, int quant_mode=0, int group_list_type=0, "
          "int[]? tuning_config=None) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, ops) {
  ops.impl("gmm_swiglu_quant_v2_layered",
           &afd_plugin::gmm_swiglu_quant::impl_npu);
}

TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, ops) {
  ops.impl("gmm_swiglu_quant_v2_layered",
           &afd_plugin::gmm_swiglu_quant::impl_autograd);
}

TORCH_LIBRARY_IMPL(afd_ascend, Meta, ops) {
  ops.impl("gmm_swiglu_quant_v2_layered",
           &afd_plugin::gmm_swiglu_quant::impl_meta);
}