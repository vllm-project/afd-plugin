/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace afd_async_atten_send {
std::tuple<at::Tensor, at::Tensor, at::Tensor> Run(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topk_ids, const at::Tensor &topk_weights, const std::vector<int64_t> &configuration, const std::vector<int64_t> &dp_ranks, const std::vector<float> &activation_parameters, int64_t layer_index)
{
    TORCH_CHECK(layer_index >= 0 && layer_index < (1LL << 24), "layer_index must be in [0, 2**24)");
    TORCH_CHECK(configuration.size() == 17, "Expected 17 configuration values");
    TORCH_CHECK(configuration[0] > 0 && configuration[1] > 0 && configuration[2] > 0 &&
                configuration[2] % configuration[1] == 0 && configuration[3] > 0 &&
                configuration[4] >= 1024 && configuration[4] <= 8192 && configuration[4] % 64 == 0 &&
                configuration[5] >= 512 && configuration[5] <= 8192 && configuration[5] % 256 == 0 &&
                configuration[6] > 0 && configuration[6] <= configuration[2] && configuration[6] <= 32 &&
                configuration[13] >= 0 && configuration[13] <= configuration[3], "Invalid split geometry");
    TORCH_CHECK(torch_npu::utils::is_npu(context), "context must be on NPU");
    TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.size(0) == configuration[13] &&
                topk_ids.size(1) == configuration[6] && topk_ids.scalar_type() == at::kInt &&
                topk_ids.device() == context.device() && topk_ids.is_contiguous(), "Invalid topk_ids");
    TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes() && topk_weights.scalar_type() == at::kFloat &&
                topk_weights.device() == context.device() && topk_weights.is_contiguous(), "Invalid topk_weights");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == configuration[13] && x.size(1) == configuration[4] &&
                x.scalar_type() == at::kBFloat16 && x.device() == context.device() && x.is_contiguous(), "Invalid activations");
    auto snapshot_data = at::empty({configuration[12] ? configuration[13] : 0, configuration[4]}, context.options().dtype(at::kByte));
    auto snapshot_scales = at::empty({configuration[12] ? configuration[13] : 0, (configuration[4] + 63) / 64 * 2}, context.options().dtype(at::kByte));
    auto completion = at::empty({1}, context.options().dtype(at::kInt));
    at::IntArrayRef configurationRef(configuration);
    at::IntArrayRef dpRanksRef(dp_ranks);
    ACLNN_CMD(aclnnAfdAsyncAttenSend, context, x, topk_ids, topk_weights, configurationRef, dpRanksRef, activation_parameters, layer_index, snapshot_data, snapshot_scales, completion);
    return std::make_tuple(snapshot_data, snapshot_scales, completion);
}
void RegisterAttenSend(pybind11::module_ &module)
{
    module.def("afd_async_atten_send", &Run, pybind11::arg("context"), pybind11::arg("x"),
        pybind11::arg("topk_ids"), pybind11::arg("topk_weights"), pybind11::arg("configuration"),
        pybind11::arg("dp_ranks"), pybind11::arg("activation_parameters"), pybind11::arg("layer_index") = 0);
}
}
