// SPDX-License-Identifier: MIT
// Copyright contributors to the AFD plugin project
//
// Adapted from cam_async/src/comm_operator/pybind/gmm_layered.cpp.
// AFD adapter: shared allocation and validation between NPU and Meta, template
// argument selects execution only. The source's autograd Function subclass is
// replaced by registering the same adapter for AutogradPrivateUse1, matching the
// other AFD Ascend bindings; these are inference-only operators.

#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#include "pytorch_extension/op_api_common.h"
#include "grouped_matmul_layered/op_api/aclnn_grouped_matmul_layered.h"

namespace afd_plugin::grouped_matmul_layered {
namespace {

using tensor_list = std::vector<at::Tensor>;

constexpr int64_t B4_PER_B32 = 8;  // eight int4 nibbles per int32 word

// Source: gmm_layered.cpp::DeriveOutputDtype.
// AFD adaptation: none; kept verbatim so the layered operator derives y exactly
// as the CAM build does. An explicit output_dtype wins; otherwise A8W4
// antiquant (int8 x with int32-packed int4 weight) yields bfloat16 while the
// quantized A8W8 family keeps the activation dtype. Never inherit from the
// packed weight tensor, whose dtype describes the storage, not the result.
at::ScalarType derive_output_dtype(
    const at::TensorList &x, const at::TensorList &all_weight,
    const c10::optional<at::ScalarType> &output_dtype) {
  if (output_dtype.has_value()) {
    return *output_dtype;
  }
  if (x[0].scalar_type() == at::kChar && all_weight[0].scalar_type() == at::kInt) {
    return at::kBFloat16;
  }
  return x[0].scalar_type();
}

// Source: gmm_layered.cpp::IsWeightTransposed.
// AFD adaptation: none. Detects the native is_weight_trans layout by the
// last-two-dimension stride swap; a transposed int32-packed int4 weight carries
// its nibbles along a different axis, which this adapter does not model, so it
// is rejected rather than used to compute a wrong N.
bool is_weight_transposed(const at::Tensor &tensor) {
  if (tensor.dim() < 2) {
    return false;
  }
  const int64_t dim1 = tensor.dim() - 1;
  const int64_t dim2 = tensor.dim() - 2;
  return tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2);
}

// Source: gmm_layered.cpp::AllocGmmOutputs.
// AFD adaptation: allocate rows from merged activation shape, without reading
// the device group list. The layered output model gives one
// [rows, n] tensor whose rows are the per-group rows concatenated; the kernel
// writes each group at its own offset inside that single buffer. A fresh
// at::empty storage is format-neutral, so the weight's FRACTAL_NZ tag cannot
// leak into y.
tensor_list alloc_outputs(const at::TensorList &x, const at::TensorList &all_weight,
                          const c10::optional<at::ScalarType> &output_dtype) {
  // torch_npu expresses int4 weight as int32, eight nibbles per word along the
  // last axis, so the logical N is the unpacked int4 extent.
  int64_t n = all_weight[0].size(-1);
  if (all_weight[0].scalar_type() == at::kInt) {
    TORCH_CHECK(!is_weight_transposed(all_weight[0]),
                "all_weight: transposed int32-packed int4 weight is not supported");
    n *= B4_PER_B32;
  }
  const auto out_options =
      all_weight[0].options().dtype(derive_output_dtype(x, all_weight, output_dtype));
  return {at::empty({x[0].size(0), n}, out_options)};
}

// Source: gmm_layered.cpp::cam_gmm_layered_impl_npu.
// AFD adaptation: the device guard is kept (it is what makes the kernel launch
// on the device the tensors live on rather than the process default), the
// symbols carry the AFD prefix, and argument validation is stated explicitly.
void check_lists(const at::TensorList &x, const at::TensorList &all_weight,
                 const at::TensorList &all_bias, const at::TensorList &all_scale,
                 const at::Tensor &layer_index) {
  TORCH_CHECK(!x.empty() && !all_weight.empty() && !all_bias.empty() && !all_scale.empty(),
              "x, all_weight, all_bias and all_scale must be non-empty lists");
  // The all_* lists each hold one element per layer, so their lengths must agree
  // and every element of a list must describe the same layer geometry.
  const size_t layer_count = all_weight.size();
  TORCH_CHECK(all_bias.size() == layer_count && all_scale.size() == layer_count,
              "all_weight, all_bias and all_scale must have the same length; got ",
              all_weight.size(), ", ", all_bias.size(), ", ", all_scale.size());
  TORCH_CHECK(layer_index.defined() && layer_index.scalar_type() == at::kLong &&
                  layer_index.numel() == 1,
              "layer_index must be a one-element int64 tensor");
  for (size_t i = 0; i < layer_count; ++i) {
    TORCH_CHECK(all_weight[i].sizes() == all_weight[0].sizes(),
                "all_weight[", i, "] must match all_weight[0]");
  }
}

// The device binding avoids materializing routing counts on the host. The
// merged activation supplies the output row count even when it has spare rows.
template <bool EXECUTE_NPU>
tensor_list grouped_matmul_layered(
    const at::TensorList &x, const at::TensorList &all_weight,
    const at::TensorList &all_bias, const at::TensorList &all_scale,
    const at::Tensor &layer_index, const at::Tensor &group_list,
    const c10::optional<at::Tensor> &per_token_scale_optional,
    const int64_t group_list_type, const int64_t split_item,
    const c10::optional<at::ScalarType> &output_dtype) {
  check_lists(x, all_weight, all_bias, all_scale, layer_index);
  TORCH_CHECK(x.size() == 1 && x[0].dim() == 2,
              "grouped_matmul_layered requires one merged 2D activation");
  TORCH_CHECK((group_list_type == 0 || group_list_type == 1) && split_item == 3,
              "grouped_matmul_layered requires group_list_type=0/1 and split_item=3");
  TORCH_CHECK(group_list.dim() == 1 && group_list.scalar_type() == at::kLong &&
                  group_list.numel() > 0 &&
                  group_list.device() == x[0].device(),
              "group_list must be a nonempty 1D int64 tensor on the activation device");
  TORCH_CHECK(layer_index.dim() == 1 && layer_index.device() == x[0].device(),
              "layer_index must have shape [1] on the activation device");
  for (size_t i = 0; i < all_weight.size(); ++i) {
    TORCH_CHECK(all_weight[i].dim() == 3 &&
                    all_weight[i].size(0) == group_list.numel() &&
                    all_weight[i].device() == x[0].device() &&
                    all_weight[i].scalar_type() == all_weight[0].scalar_type(),
                "all_weight must contain matching per-layer expert tensors");
    TORCH_CHECK(all_bias[i].device() == x[0].device() &&
                    all_scale[i].device() == x[0].device() &&
                    all_bias[i].sizes() == all_bias[0].sizes() &&
                    all_scale[i].sizes() == all_scale[0].sizes() &&
                    all_bias[i].scalar_type() == all_bias[0].scalar_type() &&
                    all_scale[i].scalar_type() == all_scale[0].scalar_type(),
                "all_bias/all_scale must have matching device, shape and dtype");
  }
  TORCH_CHECK(!per_token_scale_optional.has_value() ||
                  (per_token_scale_optional->device() == x[0].device() &&
                   per_token_scale_optional->scalar_type() == at::kFloat &&
                   per_token_scale_optional->dim() == 1 &&
                   per_token_scale_optional->size(0) == x[0].size(0)),
              "per_token_scale must be capacity-sized float32 on the activation device");
  const c10::OptionalDeviceGuard device_guard(at::device_of(x[0]));
  auto outputs = alloc_outputs(x, all_weight, output_dtype);
  if constexpr (EXECUTE_NPU) {
    // This A8W4 kernel consumes counts. Difference cumulative offsets on the
    // device so the routing metadata never takes a D2H/H2D round trip.
    at::Tensor group_counts = group_list;
    if (group_list_type == 0) {
      at::Tensor preceding = at::cat(
          {at::zeros({1}, group_list.options()),
           group_list.slice(0, 0, group_list.size(0) - 1)});
      group_counts = group_list - preceding;
    }
    at::Tensor scale = per_token_scale_optional.has_value()
                           ? *per_token_scale_optional
                           : at::empty({0}, x[0].options().dtype(at::kFloat));
    at::TensorList scales(&scale, 1);
    at::TensorList output_list(outputs);
    int64_t group_list_type_norm = 1;
    EXEC_NPU_CMD(aclnnGroupedMatmulLayered,
                 x, all_weight, all_bias, all_scale, scales, layer_index,
                 group_counts, split_item, group_list_type_norm, output_list);
  }
  return outputs;
}

}  // namespace
}  // namespace afd_plugin::grouped_matmul_layered

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
  ops.def("grouped_matmul_layered(Tensor[] x, Tensor[] all_weight, "
          "Tensor[] all_bias, Tensor[] all_scale, Tensor layer_index, "
          "Tensor group_list, Tensor? per_token_scale=None, "
          "int group_list_type=1, int split_item=3, ScalarType? output_dtype=None) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, ops) {
  ops.impl("grouped_matmul_layered",
           &afd_plugin::grouped_matmul_layered::grouped_matmul_layered<true>);
}

// Explicitly enforce the inference-only contract, including when grad mode is on.
TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, ops) {
  ops.impl("grouped_matmul_layered",
           &afd_plugin::grouped_matmul_layered::grouped_matmul_layered<true>);
}

TORCH_LIBRARY_IMPL(afd_ascend, Meta, ops) {
  ops.impl("grouped_matmul_layered",
           &afd_plugin::grouped_matmul_layered::grouped_matmul_layered<false>);
}
