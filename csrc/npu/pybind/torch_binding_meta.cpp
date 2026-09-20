// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>

namespace afd_plugin::meta {

std::vector<at::Tensor> a2e_meta(
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
  (void)expert_ids;
  (void)scales;
  (void)group_ep;
  (void)aiv_num;
  (void)compute_gate;
  int32_t base_batch_size =
      (rank >= expert_rank_size) ? x.sizes()[0] : batch_size;

  at::Tensor expand_x_out;
  if (rank >= expert_rank_size) {
    expand_x_out =
        at::empty({1, 1}, x.options().dtype(at::kBFloat16).device(at::kMeta));
  } else {
    expand_x_out = at::empty({base_batch_size, hidden_size},
                             x.options().dtype(at::kBFloat16).device(at::kMeta));
  }

  at::Tensor simulate_expert_ids;
  at::Tensor simulate_expert_scales;
  at::Tensor x_active_mask_out;
  if (rank < expert_rank_size && rank < attention_rank_size) {
    simulate_expert_ids =
        at::empty({base_batch_size, topk}, x.options().dtype(at::kInt).device(at::kMeta));
    simulate_expert_scales = at::empty(
        {base_batch_size, topk}, x.options().dtype(at::kFloat).device(at::kMeta));
    x_active_mask_out =
        at::empty(base_batch_size, x.options().dtype(at::kBool).device(at::kMeta));
  } else {
    simulate_expert_ids =
        at::empty({1, 1}, x.options().dtype(at::kInt).device(at::kMeta));
    simulate_expert_scales =
        at::empty({1, 1}, x.options().dtype(at::kFloat).device(at::kMeta));
    x_active_mask_out =
        at::empty(1, x.options().dtype(at::kBool).device(at::kMeta));
  }
  at::Tensor atten_batch_size = at::empty(
      {(attention_rank_size + expert_rank_size - 1) / expert_rank_size},
      x.options().dtype(at::kInt).device(at::kMeta));

  return {expand_x_out,
          simulate_expert_ids,
          simulate_expert_scales,
          atten_batch_size,
          x_active_mask_out};
}

at::Tensor e2a_meta(const at::Tensor &expand_x,
                    const at::Tensor &atten_batch_size,
                    int64_t batch_size,
                    int64_t hidden_size,
                    int64_t topk,
                    int64_t expert_rank_size,
                    int64_t attention_rank_size,
                    int64_t rank,
                    c10::string_view group_ep,
                    int64_t aiv_num) {
  (void)atten_batch_size;
  (void)topk;
  (void)attention_rank_size;
  (void)group_ep;
  (void)aiv_num;
  int32_t base_batch_size =
      (rank >= expert_rank_size) ? expand_x.sizes()[0] : batch_size;
  if (rank < expert_rank_size) {
    return at::empty({1, 1},
                     expand_x.options().dtype(at::kBFloat16).device(at::kMeta));
  }
  return at::empty({base_batch_size, hidden_size},
                   expand_x.options().dtype(at::kBFloat16).device(at::kMeta));
}

}  // namespace afd_plugin::meta

TORCH_LIBRARY_IMPL(afd_ascend, Meta, ops) {
  ops.impl("a2e", &afd_plugin::meta::a2e_meta);
  ops.impl("e2a", &afd_plugin::meta::e2a_meta);
}
