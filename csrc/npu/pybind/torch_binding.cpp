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

namespace afd_plugin {

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
    int64_t compute_gate) {
  int32_t base_batch_size =
      (rank >= expert_rank_size) ? x.sizes()[0] : batch_size;

  at::Tensor expand_x_out;
  if (rank >= expert_rank_size) {
    expand_x_out = at::empty({1, 1}, x.options().dtype(at::kBFloat16));
  } else {
    expand_x_out =
        at::empty({base_batch_size, hidden_size}, x.options().dtype(at::kBFloat16));
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

  EXEC_NPU_CMD(aclnnA2e,
               x,
               expert_ids,
               scales,
               base_batch_size,
               hidden_size,
               topk,
               expert_rank_size,
               attention_rank_size,
               rank,
               group_ep_ptr,
               aiv_num,
               compute_gate,
               expand_x_out,
               simulate_expert_ids,
               simulate_expert_scales,
               atten_batch_size,
               x_active_mask_out);

  return {expand_x_out,
          simulate_expert_ids,
          simulate_expert_scales,
          atten_batch_size,
          x_active_mask_out};
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
          "str group_ep='', int aiv_num=0, int compute_gate=0) -> Tensor[]");
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
