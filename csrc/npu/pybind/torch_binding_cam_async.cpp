// SPDX-License-Identifier: MIT
// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
// Copyright contributors to the AFD plugin project
//
// Adapted from cam_async/src/comm_operator/pybind/moe_*_async.cpp.
// New AFD inference adapters share allocation and validation between NPU and
// Meta. They retain the cam_async_routed_only_compact_v2 argument order and
// ACLNN calls, without the source's placeholder autograd implementation.

#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#include "pytorch_extension/op_api_common.h"
#include "afd_async_combine_recv/op_api/aclnn_afd_async_combine_recv.h"
#include "afd_async_combine_send/op_api/aclnn_afd_async_combine_send.h"
#include "afd_async_dispatch_recv/op_api/aclnn_afd_async_dispatch_recv.h"
#include "afd_async_dispatch_send/op_api/aclnn_afd_async_dispatch_send.h"

namespace afd_plugin::cam_async {
namespace {

constexpr int64_t MAX_SEQUENCE_LENGTH = 1024 * 256;
constexpr int64_t BATCH_INFO_FIELDS = 5;
constexpr int64_t RESERVED_MAGIC = 0;
constexpr float DEFAULT_BATCH_SIZE_FACTOR = 1.0f;

void check_tensor(const at::Tensor &tensor, const at::Tensor &anchor,
                  const char *name) {
  TORCH_CHECK(!tensor.requires_grad(), name,
              ": CAM asynchronous operators support inference only");
  TORCH_CHECK(tensor.device() == anchor.device(), name,
              " must be on the same device as the input");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_topology(int64_t hidden_size, int64_t top_k, int64_t moe_rank_num,
                    int64_t attn_rank_num, int64_t route_expert_num_per_moe,
                    int64_t world_size) {
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
  TORCH_CHECK(moe_rank_num > 0 && attn_rank_num > 0 &&
                  route_expert_num_per_moe > 0,
              "rank counts and route_expert_num_per_moe must be positive");
  TORCH_CHECK(world_size == moe_rank_num + attn_rank_num,
              "world_size must equal attn_rank_num + moe_rank_num");
  TORCH_CHECK(top_k > 0 && top_k <= moe_rank_num * route_expert_num_per_moe,
              "top_k must be in [1, moe_rank_num * route_expert_num_per_moe]");
}

void check_tp(int64_t max_seq_len, int64_t tp_size, int64_t attn_rank_num) {
  TORCH_CHECK(tp_size > 0 && attn_rank_num % tp_size == 0,
              "tp_size must be positive and divide attn_rank_num");
  TORCH_CHECK(max_seq_len >= tp_size && max_seq_len <= MAX_SEQUENCE_LENGTH,
              "max_seq_len must be in [tp_size, 262144]");
}

void check_float_input(const at::Tensor &tensor) {
  check_tensor(tensor, tensor, "input");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf ||
                  tensor.scalar_type() == at::kBFloat16,
              "input must have dtype float16 or bfloat16");
}

// Source: moe_dispatch_recv_async.cpp::get_max_seq_len_factor.
// AFD adaptation: retain float parsing/truncation and malformed-value fallback,
// then reject a factor that would allocate zero rows instead of hanging DR.
int64_t dispatch_recv_capacity() {
  float factor = DEFAULT_BATCH_SIZE_FACTOR;
  const char *value = std::getenv("BATCH_SIZE_FACTOR");
  if (value != nullptr) {
    try {
      factor = std::stof(value);
    } catch (const std::exception &) {
      TORCH_WARN("Cannot parse BATCH_SIZE_FACTOR; using 1.0");
    }
  }
  TORCH_CHECK(factor > 0.0f && factor <= DEFAULT_BATCH_SIZE_FACTOR,
              "BATCH_SIZE_FACTOR must be in (0, 1]");
  const int64_t capacity = static_cast<int64_t>(MAX_SEQUENCE_LENGTH * factor);
  TORCH_CHECK(capacity > 0, "BATCH_SIZE_FACTOR must allocate at least one row");
  return capacity;
}

}  // namespace

// Source: moe_dispatch_send_async.cpp::cam_dispatch_send_async_impl_npu/meta.
// AFD adapter: shared NPU/Meta validation and allocation; no autograd wrapper.
// Signature retains the source arguments; the template chooses execution only.
template <bool EXECUTE_NPU>
at::Tensor dispatch_send(
    const at::Tensor &x, const at::Tensor &expert_ids,
    const at::Tensor &comm_args, const int64_t comm_id,
    const int64_t max_seq_len, const int64_t batch_size,
    const int64_t hidden_size, const int64_t top_k,
    const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t attn_rank_id,
    const int64_t world_size, const int64_t layer_index,
    const int64_t tp_size, const int64_t dynamic_quant,
    c10::string_view group_name) {
  // Reserved by the source API: HCCL group_name selects communication resources.
  (void)comm_id;
  check_float_input(x);
  check_tensor(expert_ids, x, "expert_ids");
  check_tensor(comm_args, x, "comm_args");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  check_tp(max_seq_len, tp_size, attn_rank_num);
  TORCH_CHECK(batch_size > 0 && batch_size <= max_seq_len / tp_size,
              "batch_size must be in [1, max_seq_len / tp_size]");
  TORCH_CHECK(attn_rank_id >= 0 && attn_rank_id < attn_rank_num,
              "attn_rank_id must identify an Attention rank");
  TORCH_CHECK(layer_index >= 0, "layer_index must be nonnegative");
  TORCH_CHECK(dynamic_quant == 0 || dynamic_quant == 1,
              "dynamic_quant must be 0 or 1");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == batch_size && x.size(1) == hidden_size,
              "x must have shape [batch_size, hidden_size]");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2 &&
                  expert_ids.size(0) == batch_size && expert_ids.size(1) == top_k,
              "expert_ids must be int32[batch_size, top_k]");
  TORCH_CHECK(comm_args.scalar_type() == at::kHalf,
              "comm_args must have dtype float16");
  at::Tensor output = at::empty({1}, x.options().dtype(at::kChar));
  if constexpr (EXECUTE_NPU) {
    std::string group(group_name.data(), group_name.size());
    char *group_ptr = group.data();
    EXEC_NPU_CMD(aclnnAfdAsyncDispatchSend, x, expert_ids, comm_args,
                 RESERVED_MAGIC, max_seq_len, batch_size, hidden_size, top_k,
                 moe_rank_num, attn_rank_num, route_expert_num_per_moe,
                 attn_rank_id, world_size, layer_index, tp_size, dynamic_quant,
                 group_ptr, output);
  }
  return output;
}

// Source: moe_dispatch_recv_async.cpp::cam_dispatch_recv_async_impl_npu/meta.
// AFD adapter: one compact-v2 allocator for both backends, with inference checks.
// Signature retains the source arguments; the template chooses execution only.
template <bool EXECUTE_NPU>
std::vector<at::Tensor> dispatch_recv(
    const at::Tensor &x, const at::Tensor &comm_args, const int64_t comm_id,
    const int64_t max_seq_len, const int64_t hidden_size, const int64_t top_k,
    const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t moe_rank_id,
    const int64_t world_size, const int64_t tp_size,
    const int64_t dynamic_quant, c10::string_view group_name) {
  (void)comm_id;
  check_float_input(x);
  check_tensor(comm_args, x, "comm_args");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  check_tp(max_seq_len, tp_size, attn_rank_num);
  TORCH_CHECK(moe_rank_id >= attn_rank_num && moe_rank_id < world_size,
              "moe_rank_id must be the global rank of a MoE participant");
  TORCH_CHECK(x.dim() == 1 && x.size(0) == 1, "x must be a one-element anchor");
  TORCH_CHECK(comm_args.scalar_type() == at::kHalf,
              "comm_args must have dtype float16");
  TORCH_CHECK(dynamic_quant == 0 || dynamic_quant == 1,
              "dynamic_quant must be 0 or 1");
  const int64_t capacity = dispatch_recv_capacity();
  const int64_t batch_info_size =
      BATCH_INFO_FIELDS + tp_size + route_expert_num_per_moe * tp_size;
  at::Tensor expanded = at::empty(
      {capacity, hidden_size}, dynamic_quant == 1 ? x.options().dtype(at::kChar)
                                                  : x.options());
  at::Tensor scales = at::empty({dynamic_quant == 1 ? capacity : 1},
                                x.options().dtype(at::kFloat));
  at::Tensor batch_info = at::empty({batch_info_size}, x.options().dtype(at::kLong));
  at::Tensor counts = at::empty({route_expert_num_per_moe},
                               x.options().dtype(at::kLong));
  if constexpr (EXECUTE_NPU) {
    std::string group(group_name.data(), group_name.size());
    char *group_ptr = group.data();
    EXEC_NPU_CMD(aclnnAfdAsyncDispatchRecv, x, comm_args,
                 RESERVED_MAGIC, max_seq_len, hidden_size, top_k, moe_rank_num,
                 attn_rank_num, route_expert_num_per_moe, moe_rank_id,
                 world_size, tp_size, dynamic_quant, group_ptr,
                 expanded, scales, batch_info, counts);
  }
  return {expanded, scales, batch_info, counts};
}

// Source: moe_combine_send_async.cpp::cam_combine_send_async_impl_npu/meta.
// AFD adapter: retain compact batch_info, with no shared-expert input/backward.
// Signature retains the source arguments; the template chooses execution only.
template <bool EXECUTE_NPU>
at::Tensor combine_send(
    const at::Tensor &expand_x, const at::Tensor &comm_args,
    const at::Tensor &batch_info, const int64_t comm_id,
    const int64_t max_seq_len, const int64_t hidden_size, const int64_t top_k,
    const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t moe_rank_id,
    const int64_t world_size, const int64_t tp_size,
    c10::string_view group_name) {
  (void)comm_id;
  check_float_input(expand_x);
  check_tensor(comm_args, expand_x, "comm_args");
  check_tensor(batch_info, expand_x, "batch_info");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  check_tp(max_seq_len, tp_size, attn_rank_num);
  TORCH_CHECK(moe_rank_id >= attn_rank_num && moe_rank_id < world_size,
              "moe_rank_id must be the global rank of a MoE participant");
  TORCH_CHECK(expand_x.dim() == 2 && expand_x.size(1) == hidden_size,
              "expand_x must have shape [capacity, hidden_size]");
  TORCH_CHECK(comm_args.scalar_type() == at::kHalf,
              "comm_args must have dtype float16");
  const int64_t batch_info_size =
      BATCH_INFO_FIELDS + tp_size + route_expert_num_per_moe * tp_size;
  TORCH_CHECK(batch_info.scalar_type() == at::kLong && batch_info.dim() == 1 &&
                  batch_info.size(0) == batch_info_size,
              "batch_info must be int64[5 + tp_size + route_expert_num_per_moe * tp_size]");
  at::Tensor output = at::empty({1}, expand_x.options().dtype(at::kChar));
  if constexpr (EXECUTE_NPU) {
    std::string group(group_name.data(), group_name.size());
    char *group_ptr = group.data();
    EXEC_NPU_CMD(aclnnAfdAsyncCombineSend, expand_x, comm_args, batch_info,
                 RESERVED_MAGIC, max_seq_len, hidden_size, top_k, moe_rank_num,
                 attn_rank_num, route_expert_num_per_moe, moe_rank_id,
                 world_size, tp_size, group_ptr, output);
  }
  return output;
}

// Source: moe_combine_recv_async.cpp::cam_combine_recv_async_impl_npu/meta.
// AFD adapter: one positive [batch_size, hidden_size] shape for both backends;
// reject zero-sized requests instead of the source's NPU-only shape substitution.
// Signature retains the source arguments; the template chooses execution only.
template <bool EXECUTE_NPU>
at::Tensor combine_recv(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &expert_scales, const at::Tensor &comm_args,
    const int64_t comm_id, const int64_t batch_size, const int64_t hidden_size,
    const int64_t top_k, const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t attn_rank_id,
    const int64_t world_size, c10::string_view group_name) {
  (void)comm_id;
  check_float_input(expand_x);
  check_tensor(expert_ids, expand_x, "expert_ids");
  check_tensor(expert_scales, expand_x, "expert_scales");
  check_tensor(comm_args, expand_x, "comm_args");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  TORCH_CHECK(batch_size > 0 && batch_size <= MAX_SEQUENCE_LENGTH,
              "batch_size must be in [1, 262144]");
  TORCH_CHECK(attn_rank_id >= 0 && attn_rank_id < attn_rank_num,
              "attn_rank_id must identify an Attention rank");
  TORCH_CHECK(expand_x.dim() == 1 && expand_x.size(0) == 1,
              "expand_x must be a one-element anchor");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2 &&
                  expert_ids.size(0) == batch_size && expert_ids.size(1) == top_k,
              "expert_ids must be int32[batch_size, top_k]");
  TORCH_CHECK(expert_scales.scalar_type() == at::kFloat &&
                  expert_scales.dim() == 2 && expert_scales.size(0) == batch_size &&
                  expert_scales.size(1) == top_k,
              "expert_scales must be float32[batch_size, top_k]");
  TORCH_CHECK(comm_args.scalar_type() == at::kHalf,
              "comm_args must have dtype float16");
  at::Tensor output = at::empty({batch_size, hidden_size}, expand_x.options());
  if constexpr (EXECUTE_NPU) {
    std::string group(group_name.data(), group_name.size());
    char *group_ptr = group.data();
    EXEC_NPU_CMD(aclnnAfdAsyncCombineRecv, expand_x, expert_ids,
                 expert_scales, comm_args, RESERVED_MAGIC, batch_size,
                 hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, attn_rank_id, world_size,
                 group_ptr, output);
  }
  return output;
}

}  // namespace afd_plugin::cam_async

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
  ops.def("afd_async_dispatch_send(Tensor x, Tensor expert_ids, Tensor comm_args, "
          "int comm_id, int max_seq_len, int batch_size, int hidden_size, int top_k, "
          "int moe_rank_num, int attn_rank_num, int route_expert_num_per_moe, "
          "int attn_rank_id, int world_size, int layer_index, int tp_size, "
          "int dynamic_quant, str group_name) -> Tensor");
  ops.def("afd_async_dispatch_recv(Tensor x, Tensor comm_args, "
          "int comm_id, int max_seq_len, int hidden_size, int top_k, "
          "int moe_rank_num, int attn_rank_num, int route_expert_num_per_moe, "
          "int moe_rank_id, int world_size, int tp_size, int dynamic_quant, "
          "str group_name) -> Tensor[]");
  ops.def("afd_async_combine_send(Tensor expand_x, Tensor comm_args, Tensor batch_info, "
          "int comm_id, int max_seq_len, int hidden_size, int top_k, "
          "int moe_rank_num, int attn_rank_num, int route_expert_num_per_moe, "
          "int moe_rank_id, int world_size, int tp_size, str group_name) -> Tensor");
  ops.def("afd_async_combine_recv(Tensor expand_x, Tensor expert_ids, "
          "Tensor expert_scales, Tensor comm_args, int comm_id, int batch_size, "
          "int hidden_size, int top_k, int moe_rank_num, int attn_rank_num, "
          "int route_expert_num_per_moe, int attn_rank_id, int world_size, "
          "str group_name) -> Tensor");
}

TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, ops) {
  ops.impl("afd_async_dispatch_send", &afd_plugin::cam_async::dispatch_send<true>);
  ops.impl("afd_async_dispatch_recv", &afd_plugin::cam_async::dispatch_recv<true>);
  ops.impl("afd_async_combine_send", &afd_plugin::cam_async::combine_send<true>);
  ops.impl("afd_async_combine_recv", &afd_plugin::cam_async::combine_recv<true>);
}

// Explicitly enforce the inference-only contract, including when grad mode is on.
TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, ops) {
  ops.impl("afd_async_dispatch_send", &afd_plugin::cam_async::dispatch_send<true>);
  ops.impl("afd_async_dispatch_recv", &afd_plugin::cam_async::dispatch_recv<true>);
  ops.impl("afd_async_combine_send", &afd_plugin::cam_async::combine_send<true>);
  ops.impl("afd_async_combine_recv", &afd_plugin::cam_async::combine_recv<true>);
}

TORCH_LIBRARY_IMPL(afd_ascend, Meta, ops) {
  ops.impl("afd_async_dispatch_send", &afd_plugin::cam_async::dispatch_send<false>);
  ops.impl("afd_async_dispatch_recv", &afd_plugin::cam_async::dispatch_recv<false>);
  ops.impl("afd_async_combine_send", &afd_plugin::cam_async::combine_send<false>);
  ops.impl("afd_async_combine_recv", &afd_plugin::cam_async::combine_recv<false>);
}
