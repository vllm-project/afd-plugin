// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/npu/Module.h>

#include "pytorch_extension/op_api_common.h"
#include "a2e/op_api/aclnn_a2e.h"
#include "e2a/op_api/aclnn_e2a.h"
#include <limits>
#include "a2e/op_kernel/a2e_packet.h"

namespace afd_plugin {
namespace {

// Only A2E arguments opt into FP8 conversion; other operators keep the common converter.
struct A2eTensorArg {
  at::Tensor tensor;
};

aclTensor *ConvertType(const A2eTensorArg &arg) {
  const auto &tensor = arg.tensor;
  if (!tensor.defined()) {
    return nullptr;
  }
#ifdef AFD_HAS_MXFP8_DTYPES
  aclDataType dtype;
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float8_e4m3fn: dtype = ACL_FLOAT8_E4M3FN; break;
    case at::ScalarType::Float8_e5m2: dtype = ACL_FLOAT8_E5M2; break;
    case at::ScalarType::Float8_e8m0fnu: dtype = ACL_FLOAT8_E8M0; break;
    default: return ::ConvertType(tensor);
  }
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  TORCH_CHECK(aclCreateTensor != nullptr, "A2E requires aclCreateTensor");
  aclFormat format = ACL_FORMAT_ND;
  c10::SmallVector<int64_t, 5> storage_dims;
  if (!IsOpInputBaseFormat(tensor)) {
    const auto &desc = vllm_ascend::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    format = desc.npu_format_;
    storage_dims = desc.storage_sizes_;
  } else {
    storage_dims.push_back(tensor.storage().nbytes() / tensor.itemsize());
  }
  return aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype,
                         tensor.strides().data(), tensor.storage_offset(), format,
                         storage_dims.data(), storage_dims.size(),
                         const_cast<void *>(tensor.storage().data()));
#else
  return ::ConvertType(tensor);
#endif
}

}  // namespace

std::vector<at::Tensor> a2e(
    const at::Tensor &x,
    const c10::optional<at::Tensor> &expert_ids,
    const c10::optional<at::Tensor> &scales,
    int64_t batch_size,
    int64_t hidden_size,
    int64_t topk,
    int64_t expert_rank_size,
    int64_t attention_rank_size,
    int64_t rank,
    c10::string_view group_ep,
    int64_t aiv_num,
    int64_t compute_gate,
    const c10::optional<at::Tensor> &x_scale,
    int64_t quant_mode) {
  constexpr int64_t A2E_MX_GROUP_SIZE = Moe::A2E_MX_GROUP_SIZE;
  constexpr int64_t A2E_MAX_ROLE_RANKS = Moe::A2E_A5_MAX_RANKS / 2;
  constexpr int64_t A2E_MAX_PACKET_BYTES = std::numeric_limits<int32_t>::max();
  constexpr int64_t A2E_GATE_UB_BYTES = 192 * 1024 - 1024;
  TORCH_CHECK(quant_mode == 0 || quant_mode == 4,
              "A2E quant_mode must be 0 or 4");
  if (quant_mode == 0) {
    TORCH_CHECK(!x_scale.has_value(), "A2E x_scale requires quant_mode=4");
    TORCH_CHECK(x.scalar_type() != at::ScalarType::Float8_e4m3fn &&
                    x.scalar_type() != at::ScalarType::Float8_e5m2,
                "A2E FP8 x requires quant_mode=4");
  } else {
#ifndef AFD_HAS_MXFP8_DTYPES
  TORCH_CHECK(false, "A2E MXFP8 requires PyTorch/CANN E4M3, E5M2 and E8M0 support");
#else
    TORCH_CHECK(expert_rank_size > 0 && expert_rank_size <= A2E_MAX_ROLE_RANKS &&
                    expert_rank_size == attention_rank_size,
                "A2E MXFP8 requires equal positive attention/expert rank counts <= 32");
    TORCH_CHECK(rank >= 0 && rank < expert_rank_size + attention_rank_size,
                "A2E rank is out of range");
    constexpr int64_t max_int = std::numeric_limits<int32_t>::max();
    TORCH_CHECK(batch_size > 0 && batch_size <= max_int && hidden_size > 0 &&
                    hidden_size <= max_int - 31 && topk > 0 && topk <= max_int,
                "A2E MXFP8 dimensions must be positive int32 values");
    TORCH_CHECK(compute_gate == 0 || compute_gate == 1,
                "A2E MXFP8 compute_gate must be 0 or 1");
    TORCH_CHECK(aiv_num > (compute_gate ? 2 : 0) && aiv_num <= max_int,
                "A2E MXFP8 has insufficient or invalid AIV count");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == hidden_size && x.is_contiguous() &&
                    (x.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                     x.scalar_type() == at::ScalarType::Float8_e5m2),
                "A2E MXFP8 x must be contiguous 2D E4M3 or E5M2 with hidden_size columns");
    TORCH_CHECK(x.size(0) >= 0 && x.size(0) <= max_int,
                "A2E MXFP8 input token count must be a non-negative int32 value");
    const bool attention = rank >= expert_rank_size;
    TORCH_CHECK(!attention || x.size(0) == batch_size,
                "A2E MXFP8 attention ranks require batch_size input rows");
    TORCH_CHECK(attention || x.size(0) == 0 || x.size(0) == batch_size,
                "A2E MXFP8 expert ranks require zero or batch_size input rows");
    TORCH_CHECK(!attention || (x_scale.has_value() && x_scale->defined()),
                "A2E MXFP8 attention ranks require x_scale");
    if (x_scale.has_value()) {
      TORCH_CHECK(x_scale->defined() && x_scale->dim() == 2 &&
                      x_scale->scalar_type() == at::ScalarType::Float8_e8m0fnu &&
                      x_scale->is_contiguous() && x_scale->device() == x.device() &&
                      x_scale->size(0) == x.size(0) &&
                      x_scale->size(1) == (hidden_size + A2E_MX_GROUP_SIZE - 1) / A2E_MX_GROUP_SIZE,
                  "A2E MXFP8 x_scale must be contiguous E8M0 [tokens, ceil(hidden_size/32)] on x device");
    }
    const int64_t scale_width =
        (hidden_size + A2E_MX_GROUP_SIZE - 1) / A2E_MX_GROUP_SIZE;
    TORCH_CHECK(batch_size <= (max_int - aiv_num) / hidden_size &&
                    batch_size <= (max_int / 8) / topk &&
                    batch_size <= max_int / scale_width,
                "A2E MXFP8 packet dimensions overflow int32 addressing");
    const auto layout = Moe::GetA2EPacketLayout(batch_size, hidden_size, topk, 1, quant_mode);
    TORCH_CHECK(layout.endOffset <= A2E_MAX_PACKET_BYTES,
                "A2E MXFP8 packet layout exceeds int32 addressing");
    TORCH_CHECK(compute_gate == 0 || batch_size * topk * sizeof(int32_t) <= A2E_GATE_UB_BYTES,
                "A2E MXFP8 compute_gate=1 routing IDs exceed UB capacity");
    if (attention && compute_gate == 1) {
      TORCH_CHECK(expert_ids.has_value() && scales.has_value() &&
                      expert_ids->defined() && scales->defined(),
                  "A2E compute_gate=1 requires routing ids and scales");
      TORCH_CHECK(expert_ids->dim() == 2 && scales->dim() == 2 &&
                      expert_ids->size(0) == x.size(0) && scales->size(0) == x.size(0) &&
                      expert_ids->size(1) == topk && scales->size(1) == topk &&
                      expert_ids->scalar_type() == at::kInt && scales->scalar_type() == at::kFloat &&
                      expert_ids->is_contiguous() && scales->is_contiguous() &&
                      expert_ids->device() == x.device() && scales->device() == x.device(),
                  "A2E routing tensors must be contiguous int32/float32 [tokens, topk] on x device");
    }
#endif
  }
#ifndef AFD_TARGET_A5
  TORCH_CHECK(quant_mode == 0, "A2E MXFP8 runtime requires an Ascend 950/A5 build");
#endif
  int64_t base_batch_size =
      (rank >= expert_rank_size) ? x.sizes()[0] : batch_size;

  at::Tensor expand_x_out;
  if (rank >= expert_rank_size) {
    expand_x_out = at::empty({1, 1}, x.options().dtype(quant_mode == 4 ? x.scalar_type() : at::kBFloat16));
  } else {
    expand_x_out =
        at::empty({base_batch_size, hidden_size}, x.options().dtype(quant_mode == 4 ? x.scalar_type() : at::kBFloat16));
  }

  at::Tensor simulate_expert_ids;
  at::Tensor simulate_expert_scales;
  at::Tensor atten_batch_size;
  at::Tensor x_active_mask_out;
  if (rank < expert_rank_size && rank < attention_rank_size) {
    simulate_expert_ids =
        at::empty({base_batch_size, topk}, x.options().dtype(at::kInt));
    simulate_expert_scales =
        at::empty({base_batch_size, topk}, x.options().dtype(at::kFloat));
    x_active_mask_out = at::empty(base_batch_size, x.options().dtype(at::kBool));
  } else {
    simulate_expert_ids = at::empty({1, 1}, x.options().dtype(at::kInt));
    simulate_expert_scales = at::empty({1, 1}, x.options().dtype(at::kFloat));
    x_active_mask_out = at::empty(1, x.options().dtype(at::kBool));
  }
  atten_batch_size = at::empty(
      {(attention_rank_size + expert_rank_size - 1) / expert_rank_size},
      x.options().dtype(at::kInt));

  std::vector<char> group_ep_chars(group_ep.begin(), group_ep.end());
  group_ep_chars.push_back('\0');
  char *group_ep_ptr = &group_ep_chars[0];

  at::Tensor x_scale_out;
  if (quant_mode == 4) {
#ifdef AFD_HAS_MXFP8_DTYPES
    auto scale_options = x.options().dtype(at::ScalarType::Float8_e8m0fnu);
    x_scale_out = rank >= expert_rank_size
        ? at::empty({1, 1}, scale_options)
        : at::empty({base_batch_size, (hidden_size + A2E_MX_GROUP_SIZE - 1) / A2E_MX_GROUP_SIZE}, scale_options);
#endif
  }
  A2eTensorArg x_arg{x};
  A2eTensorArg x_scale_arg{x_scale.value_or(at::Tensor())};
  A2eTensorArg expand_x_arg{expand_x_out};
  A2eTensorArg x_scale_out_arg{x_scale_out};
  EXEC_NPU_CMD(aclnnA2e, x_arg, expert_ids, scales, x_scale_arg,
               base_batch_size, hidden_size, topk, expert_rank_size,
               attention_rank_size, rank, group_ep_ptr, aiv_num,
               compute_gate, quant_mode, expand_x_arg, simulate_expert_ids,
               simulate_expert_scales, atten_batch_size, x_active_mask_out,
               x_scale_out_arg);

  std::vector<at::Tensor> outputs = {expand_x_out, simulate_expert_ids,
      simulate_expert_scales, atten_batch_size, x_active_mask_out};
  if (quant_mode == 4) {
    outputs.push_back(x_scale_out);
  }
  return outputs;
}

at::Tensor e2a(const at::Tensor &expand_x,
               const at::Tensor &atten_batch_size,
               int64_t batch_size,
               int64_t hidden_size,
               int64_t topk,
               int64_t expert_rank_size,
               int64_t attention_rank_size,
               int64_t rank,
               c10::string_view group_ep,
               int64_t aiv_num) {
  int32_t base_batch_size =
      (rank >= expert_rank_size) ? expand_x.sizes()[0] : batch_size;

  at::Tensor x_out;
  if (rank < expert_rank_size) {
    x_out = at::empty({1, 1}, expand_x.options().dtype(at::kBFloat16));
  } else {
    x_out = at::empty({base_batch_size, hidden_size},
                      expand_x.options().dtype(at::kBFloat16));
  }

  std::vector<char> group_ep_chars(group_ep.begin(), group_ep.end());
  group_ep_chars.push_back('\0');
  char *group_ep_ptr = &group_ep_chars[0];

  EXEC_NPU_CMD(aclnnE2a,
               expand_x,
               atten_batch_size,
               base_batch_size,
               hidden_size,
               topk,
               expert_rank_size,
               attention_rank_size,
               rank,
               group_ep_ptr,
               aiv_num,
               x_out);

  return x_out;
}

}  // namespace afd_plugin

TORCH_LIBRARY(afd_ascend, ops) {
  ops.def("a2e(Tensor x, Tensor? expert_ids=None, Tensor? scales=None, "
          "int batch_size=0, int hidden_size=0, int topk=0, "
          "int expert_rank_size=0, int attention_rank_size=0, int rank=0, "
          "str group_ep='', int aiv_num=0, int compute_gate=0, "
          "Tensor? x_scale=None, int quant_mode=0) -> Tensor[]");
  ops.impl("a2e", torch::kPrivateUse1, &afd_plugin::a2e);

  ops.def("e2a(Tensor expand_x, Tensor atten_batch_size, "
          "int batch_size=0, int hidden_size=0, int topk=0, "
          "int expert_rank_size=0, int attention_rank_size=0, int rank=0, "
          "str group_ep='', int aiv_num=0) -> Tensor");
  ops.impl("e2a", torch::kPrivateUse1, &afd_plugin::e2a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "AFD Ascend A2E/E2A custom operator bindings";
}
