// SPDX-License-Identifier: MIT
// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
// Copyright contributors to the AFD plugin project
//
// Adapted from cam_async/src/comm_operator/pybind/moe_*_async.cpp.
// Dedicated Mega pull adapters; the original CAM entry points remain unchanged.

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#include "pytorch_extension/op_api_common.h"
#include "utils/op_kernel/afd_pull_layout.h"
#include "afd_mega_moe_comm/op_api/aclnn_afd_mega_moe_comm.h"

namespace afd_plugin::mega_comm {
namespace {

constexpr int64_t MAX_SEQUENCE_LENGTH = 1024 * 256;
constexpr int64_t BATCH_INFO_FIELDS = 5;
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
  TORCH_CHECK(max_seq_len >= tp_size && max_seq_len <= MAX_SEQUENCE_LENGTH && max_seq_len % tp_size == 0,
              "max_seq_len must be in [tp_size, 262144] and divisible by tp_size");
}

void check_pull_state(const at::Tensor &state, int64_t attn_ranks,
                       int64_t tp_size, const at::Tensor &anchor) {
  TORCH_CHECK(tp_size > 0 && attn_ranks % tp_size == 0, "pull TP topology is invalid");
  TORCH_CHECK(anchor.scalar_type() == at::kBFloat16, "pull requires BF16");
  TORCH_CHECK(state.scalar_type() == at::kHalf && state.dim() == 1 &&
      state.numel() >= int64_t(4 * Cam::Pull::StateWords(attn_ranks / tp_size, 0)),
      "pull state is missing; initialize with afd_mega_pull_init");
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

template <bool EXECUTE_NPU>
std::vector<at::Tensor> run_comm(const at::Tensor &x, const at::Tensor &ids,
    const at::Tensor &probabilities, const at::Tensor &state, const at::Tensor &batch,
    at::IntArrayRef c, c10::string_view group_name) {
  Cam::Pull::Layout layout{};
  TORCH_CHECK(c[11] > 0 && c[11] <= Cam::Pull::MAX_CAPACITY, "Mega receive capacity is out of bounds");
  TORCH_CHECK(Cam::Pull::MakeLayout({uint64_t(c[1] / c[8]), uint64_t(c[2]), uint64_t(c[3]),
      uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[8])}, layout), "invalid Mega communication layout");
  std::vector<at::Tensor> out{
      at::empty({c[0] == 2 ? c[11] * c[2] : 1}, x.options().dtype(at::kChar)),
      at::empty({c[0] == 2 ? c[11] : 1}, x.options().dtype(at::kFloat)),
      at::empty({c[0] == 2 ? int64_t(Cam::Pull::MetadataWords(c[8], c[6])) : 1}, x.options().dtype(at::kLong)),
      at::empty({c[0] == 2 ? c[6] : 1}, x.options().dtype(at::kLong)),
      at::empty({c[0] == 4 ? std::max<int64_t>(1, c[9]) * c[2] : 1}, x.options()),
      at::empty({1}, x.options().dtype(at::kLong))};
  if constexpr (EXECUTE_NPU) {
    std::string group(group_name.data(), group_name.size());
    auto group_ptr = group.data();
    auto &packed = out[0], &scales = out[1], &metadata = out[2], &counts = out[3];
    auto &combined = out[4], &completion = out[5];
    EXEC_NPU_CMD(aclnnAfdMegaMoeComm, x, ids, probabilities, state, batch, c, group_ptr,
        packed, scales, metadata, counts, combined, completion);
  }
  return out;
}

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
  TORCH_CHECK(comm_id == Cam::Pull::MAGIC, "Mega communication requires pull_v1");
  check_float_input(x);
  check_pull_state(comm_args, attn_rank_num, tp_size, x);
  check_tensor(expert_ids, x, "expert_ids");
  check_tensor(comm_args, x, "comm_args");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  check_tp(max_seq_len, tp_size, attn_rank_num);
  TORCH_CHECK(batch_size >= 0 &&
                  batch_size <= max_seq_len / tp_size,
              "batch_size is outside the dispatch envelope");
  TORCH_CHECK(attn_rank_id >= 0 && attn_rank_id < attn_rank_num,
              "attn_rank_id must identify an Attention rank");
  TORCH_CHECK(layer_index >= 0, "layer_index must be nonnegative");
  TORCH_CHECK(dynamic_quant == 1, "pull requires dynamic_quant=1");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == batch_size && x.size(1) == hidden_size,
              "x must have shape [batch_size, hidden_size]");
  TORCH_CHECK(expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2 &&
                  expert_ids.size(0) == batch_size && expert_ids.size(1) == top_k,
              "expert_ids must be int32[batch_size, top_k]");
  TORCH_CHECK(comm_args.scalar_type() == at::kHalf,
              "comm_args must have dtype float16");
  auto probs = at::empty({1}, x.options().dtype(at::kFloat));
  auto batch = at::empty({1}, x.options().dtype(at::kLong));
  return run_comm<EXECUTE_NPU>(x, expert_ids, probs, comm_args, batch,
      {1,max_seq_len,hidden_size,top_k,moe_rank_num,attn_rank_num,route_expert_num_per_moe,
       attn_rank_id,tp_size,batch_size,layer_index,1}, group_name)[5];
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
  TORCH_CHECK(comm_id == Cam::Pull::MAGIC, "Mega communication requires pull_v1");
  check_float_input(x);
  check_pull_state(comm_args, attn_rank_num, tp_size, x);
  check_tensor(comm_args, x, "comm_args");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  check_tp(max_seq_len, tp_size, attn_rank_num);
  TORCH_CHECK(moe_rank_id >= attn_rank_num && moe_rank_id < world_size,
              "moe_rank_id must be the global rank of a MoE participant");
  TORCH_CHECK(x.dim() == 1 && x.size(0) == 1, "x must be a one-element anchor");
  TORCH_CHECK(comm_args.scalar_type() == at::kHalf,
              "comm_args must have dtype float16");
  TORCH_CHECK(dynamic_quant == 1, "pull requires dynamic_quant=1");
  const int64_t capacity = dispatch_recv_capacity();
  auto ids = at::empty({1}, x.options().dtype(at::kInt));
  auto probs = at::empty({1}, x.options().dtype(at::kFloat));
  auto batch = at::empty({1}, x.options().dtype(at::kLong));
  auto out = run_comm<EXECUTE_NPU>(x, ids, probs, comm_args, batch,
      {2,max_seq_len,hidden_size,top_k,moe_rank_num,attn_rank_num,route_expert_num_per_moe,
       moe_rank_id,tp_size,0,0,capacity}, group_name);
  return {out[0].view({capacity,hidden_size}),out[1],out[2],out[3]};
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
  TORCH_CHECK(comm_id == Cam::Pull::MAGIC, "Mega communication requires pull_v1");
  check_float_input(expand_x);
  check_pull_state(comm_args, attn_rank_num, tp_size, expand_x);
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
      BATCH_INFO_FIELDS + tp_size + route_expert_num_per_moe * tp_size +
      2;
  TORCH_CHECK(batch_info.scalar_type() == at::kLong && batch_info.dim() == 1 &&
                  batch_info.size(0) == batch_info_size,
              "batch_info does not match the selected protocol layout");
  auto ids = at::empty({1}, expand_x.options().dtype(at::kInt));
  auto probs = at::empty({1}, expand_x.options().dtype(at::kFloat));
  return run_comm<EXECUTE_NPU>(expand_x, ids, probs, comm_args, batch_info,
      {3,max_seq_len,hidden_size,top_k,moe_rank_num,attn_rank_num,route_expert_num_per_moe,
       moe_rank_id,tp_size,expand_x.size(0),0,dispatch_recv_capacity()}, group_name)[5];
}

// Source: moe_combine_recv_async.cpp::cam_combine_recv_async_impl_npu/meta.
// Empty transactions still launch a physical output row to consume completion flags.
// Signature retains the source arguments; the template chooses execution only.
template <bool EXECUTE_NPU>
at::Tensor combine_recv(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &expert_scales, const at::Tensor &comm_args,
    const int64_t comm_id, const int64_t batch_size, const int64_t hidden_size,
    const int64_t top_k, const int64_t moe_rank_num, const int64_t attn_rank_num,
    const int64_t route_expert_num_per_moe, const int64_t attn_rank_id,
    const int64_t world_size, c10::string_view group_name,
    int64_t max_seq_len, int64_t tp_size) {
  TORCH_CHECK(comm_id == Cam::Pull::MAGIC, "Mega communication requires pull_v1");
  check_float_input(expand_x);
  check_tp(max_seq_len, tp_size, attn_rank_num);
  check_pull_state(comm_args, attn_rank_num, tp_size, expand_x);
  check_tensor(expert_ids, expand_x, "expert_ids");
  check_tensor(expert_scales, expand_x, "expert_scales");
  check_tensor(comm_args, expand_x, "comm_args");
  check_topology(hidden_size, top_k, moe_rank_num, attn_rank_num,
                 route_expert_num_per_moe, world_size);
  TORCH_CHECK(batch_size >= 0 &&
                  batch_size <= MAX_SEQUENCE_LENGTH,
              "batch_size is outside the combine envelope");
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
  TORCH_CHECK(batch_size <= max_seq_len / tp_size, "batch_size exceeds Mega source capacity");
  auto batch = at::empty({1}, expand_x.options().dtype(at::kLong));
  auto out = run_comm<EXECUTE_NPU>(expand_x, expert_ids, expert_scales, comm_args, batch,
      {4,max_seq_len,hidden_size,top_k,moe_rank_num,attn_rank_num,route_expert_num_per_moe,
       attn_rank_id,tp_size,batch_size,0,1}, group_name)[4];
  return out.view({std::max<int64_t>(1,batch_size),hidden_size}).narrow(0,0,batch_size);
}

at::Tensor pull_init(const at::Tensor &anchor, at::IntArrayRef config,
                     at::IntArrayRef layer_ids, int64_t epoch, c10::string_view group_name) {
  TORCH_CHECK(config.size() == 8, "pull config must be [Ncap,H,K,F,A,E,rank,TP]");
  const auto c = config.data();
  Cam::Pull::Layout layout{};
  TORCH_CHECK(Cam::Pull::MakeLayout({uint64_t(c[0]), uint64_t(c[1]), uint64_t(c[2]),
      uint64_t(c[3]), uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[7])}, layout),
      "invalid pull layout or uint16 Combine bound");
  TORCH_CHECK(c[6] >= 0 && c[6] < c[3] + c[4] && epoch > 0,
              "pull rank/epoch is invalid");
  TORCH_CHECK(anchor.scalar_type() == at::kBFloat16, "pull requires BF16");
  int64_t map_size = 0;
  for (auto layer : layer_ids) {
    TORCH_CHECK(layer >= 0 && layer < 65536, "pull layer map is out of bounds");
    map_size = std::max(map_size, layer + 1);
  }
  TORCH_CHECK(c[6] < c[4] || !layer_ids.empty(), "FFN pull requires a layer table");
  std::vector<int64_t> values(Cam::Pull::StateWords(c[4] / c[7], map_size), 0);
  values[Cam::Pull::SESSION] = epoch;
  values[Cam::Pull::SOURCE_CAPACITY] = c[0];
  values[Cam::Pull::TP_SIZE] = c[7];
  values[Cam::Pull::MAP_SIZE] = map_size;
  values[Cam::Pull::SOURCE] = -1;
  const auto offset = Cam::Pull::LayerMapOffset(c[4] / c[7]);
  std::fill(values.begin() + offset, values.begin() + offset + map_size, -1);
  for (int64_t slot = 0; slot < int64_t(layer_ids.size()); ++slot) {
    const auto layer = layer_ids[slot];
    TORCH_CHECK(layer >= 0 && values[offset + layer] == -1, "pull layer IDs must be unique and nonnegative");
    values[offset + layer] = slot;
  }
  auto host = at::from_blob(values.data(), {int64_t(values.size())}, at::TensorOptions().dtype(at::kLong));
  auto state = host.to(anchor.device(), at::kLong, false, true).view(at::kHalf);
  if (anchor.is_meta()) return state;
  auto ids = at::empty({1}, anchor.options().dtype(at::kInt));
  auto probs = at::empty({1}, anchor.options().dtype(at::kFloat));
  auto batch = at::empty({1}, anchor.options().dtype(at::kLong));
  run_comm<true>(anchor, ids, probs, state, batch,
      {0,c[0]*c[7],c[1],c[2],c[3],c[4],c[5],c[6],c[7],0,0,1}, group_name);
  return state;
}

}  // namespace afd_plugin::mega_comm

TORCH_LIBRARY_FRAGMENT(afd_ascend, ops) {
  ops.def("afd_mega_pull_init(Tensor anchor, int[] config, int[] layer_ids, int epoch, str group_name) -> Tensor");
  ops.def("afd_mega_dispatch_send(Tensor x, Tensor expert_ids, Tensor(a!) comm_args, "
          "int comm_id, int max_seq_len, int batch_size, int hidden_size, int top_k, "
          "int moe_rank_num, int attn_rank_num, int route_expert_num_per_moe, "
          "int attn_rank_id, int world_size, int layer_index, int tp_size, "
          "int dynamic_quant, str group_name) -> Tensor");
  ops.def("afd_mega_dispatch_recv(Tensor x, Tensor(a!) comm_args, "
          "int comm_id, int max_seq_len, int hidden_size, int top_k, "
          "int moe_rank_num, int attn_rank_num, int route_expert_num_per_moe, "
          "int moe_rank_id, int world_size, int tp_size, int dynamic_quant, "
          "str group_name) -> Tensor[]");
  ops.def("afd_mega_combine_send(Tensor expand_x, Tensor(a!) comm_args, Tensor batch_info, "
          "int comm_id, int max_seq_len, int hidden_size, int top_k, "
          "int moe_rank_num, int attn_rank_num, int route_expert_num_per_moe, "
          "int moe_rank_id, int world_size, int tp_size, str group_name) -> Tensor");
  ops.def("afd_mega_combine_recv(Tensor expand_x, Tensor expert_ids, "
          "Tensor expert_scales, Tensor(a!) comm_args, int comm_id, int batch_size, "
          "int hidden_size, int top_k, int moe_rank_num, int attn_rank_num, "
          "int route_expert_num_per_moe, int attn_rank_id, int world_size, "
          "str group_name, int max_seq_len, int tp_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(afd_ascend, PrivateUse1, ops) {
  ops.impl("afd_mega_pull_init", &afd_plugin::mega_comm::pull_init);
  ops.impl("afd_mega_dispatch_send", &afd_plugin::mega_comm::dispatch_send<true>);
  ops.impl("afd_mega_dispatch_recv", &afd_plugin::mega_comm::dispatch_recv<true>);
  ops.impl("afd_mega_combine_send", &afd_plugin::mega_comm::combine_send<true>);
  ops.impl("afd_mega_combine_recv", &afd_plugin::mega_comm::combine_recv<true>);
}

// Explicitly enforce the inference-only contract, including when grad mode is on.
TORCH_LIBRARY_IMPL(afd_ascend, AutogradPrivateUse1, ops) {
  ops.impl("afd_mega_dispatch_send", &afd_plugin::mega_comm::dispatch_send<true>);
  ops.impl("afd_mega_dispatch_recv", &afd_plugin::mega_comm::dispatch_recv<true>);
  ops.impl("afd_mega_combine_send", &afd_plugin::mega_comm::combine_send<true>);
  ops.impl("afd_mega_combine_recv", &afd_plugin::mega_comm::combine_recv<true>);
}

TORCH_LIBRARY_IMPL(afd_ascend, Meta, ops) {
  ops.impl("afd_mega_pull_init", &afd_plugin::mega_comm::pull_init);
  ops.impl("afd_mega_dispatch_send", &afd_plugin::mega_comm::dispatch_send<false>);
  ops.impl("afd_mega_dispatch_recv", &afd_plugin::mega_comm::dispatch_recv<false>);
  ops.impl("afd_mega_combine_send", &afd_plugin::mega_comm::combine_send<false>);
  ops.impl("afd_mega_combine_recv", &afd_plugin::mega_comm::combine_recv<false>);
}
