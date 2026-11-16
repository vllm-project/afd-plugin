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

namespace cam_megamoe_combine_recv {
static void CheckExpertTensors(const std::vector<at::Tensor> &first,
                               const std::vector<at::Tensor> &second,
                               const std::vector<at::Tensor> &firstScales,
                               const std::vector<at::Tensor> &secondScales,
                               int64_t experts, int64_t hidden, int64_t intermediateFull)
{
    if (experts == 0) { return; }
    TORCH_CHECK(!first.empty() && !second.empty(), "Missing expert weights");
    bool separate = first[0].dim() == 2;
    auto tensors = separate ? experts : 1;
    TORCH_CHECK(int64_t(first.size()) == tensors && int64_t(second.size()) == tensors &&
                int64_t(firstScales.size()) == tensors && int64_t(secondScales.size()) == tensors,
                "Use one stacked tensor or one tensor per expert, consistently for both layers and scales");
    for (int64_t index = 0; index < tensors; ++index) {
        int64_t packed = first[index].scalar_type() == at::kByte ? 2 : 1;
        TORCH_CHECK(first[index].scalar_type() == second[index].scalar_type(), "Weight dtypes must match");
        std::vector<int64_t> firstShape{intermediateFull, hidden / packed};
        std::vector<int64_t> secondShape{hidden, intermediateFull / 2 / packed};
        std::vector<int64_t> firstScaleShape{intermediateFull, hidden / 64, 2};
        std::vector<int64_t> secondScaleShape{hidden, intermediateFull / 2 / 64, 2};
        if (!separate) {
            firstShape.insert(firstShape.begin(), experts);
            secondShape.insert(secondShape.begin(), experts);
            firstScaleShape.insert(firstScaleShape.begin(), experts);
            secondScaleShape.insert(secondScaleShape.begin(), experts);
        }
        TORCH_CHECK(first[index].sizes().vec() == firstShape && second[index].sizes().vec() == secondShape,
                    "Expert weight shape mismatch (packed FP4 stores two elements per byte)");
        TORCH_CHECK(firstScales[index].sizes().vec() == firstScaleShape &&
                    secondScales[index].sizes().vec() == secondScaleShape &&
                    firstScales[index].element_size() == 1 && secondScales[index].element_size() == 1,
                    "Expert E8M0 scale shape mismatch");
    }
}

std::tuple<at::Tensor, at::Tensor> Run(const at::Tensor &context, const at::Tensor &topk_ids, const at::Tensor &topk_weights, const at::Tensor &snapshot_data, const at::Tensor &snapshot_scales, const std::vector<at::Tensor> &shared_weight1, const std::vector<at::Tensor> &shared_weight2, const std::vector<at::Tensor> &shared_scales1, const std::vector<at::Tensor> &shared_scales2, const std::vector<int64_t> &configuration, const std::vector<int64_t> &dp_ranks, const std::vector<float> &activation_parameters)
{
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
    TORCH_CHECK(snapshot_data.scalar_type() == at::kByte && snapshot_scales.scalar_type() == at::kByte &&
                snapshot_data.device() == context.device() && snapshot_scales.device() == context.device() &&
                snapshot_data.is_contiguous() && snapshot_scales.is_contiguous(), "Invalid shared input snapshots");
    if (configuration[12] > 0) {
        TORCH_CHECK(snapshot_data.numel() == configuration[13] * configuration[4] &&
                    snapshot_scales.numel() == configuration[13] * ((configuration[4] + 63) / 64 * 2),
                    "Shared input snapshot size mismatch");
    }
    CheckExpertTensors(shared_weight1, shared_weight2, shared_scales1, shared_scales2, configuration[12], configuration[4], configuration[5]);
    auto y = at::empty({configuration[13], configuration[4]}, context.options().dtype(at::kBFloat16));
    TORCH_CHECK(!shared_weight1.empty(), "shared_weight1 needs at least one typed tensor");
    for (const auto &tensor : shared_weight1) {
        TORCH_CHECK(tensor.device() == context.device(), "shared_weight1 must share context's NPU");
    }
    for (const auto &tensor : shared_weight2) {
        TORCH_CHECK(tensor.device() == context.device(), "shared_weight2 must share context's NPU");
    }
    for (const auto &tensor : shared_scales1) {
        TORCH_CHECK(tensor.device() == context.device(), "shared_scales1 must share context's NPU");
    }
    for (const auto &tensor : shared_scales2) {
        TORCH_CHECK(tensor.device() == context.device(), "shared_scales2 must share context's NPU");
    }
    TensorListWrapper shared_weight1Wrapper = {at::TensorList(shared_weight1), (shared_weight1[0].scalar_type() == at::kByte ? aclDataType::ACL_FLOAT4_E2M1 : ConvertToAclDataType(shared_weight1[0].scalar_type()))};
    TORCH_CHECK(!shared_weight2.empty(), "shared_weight2 needs at least one typed tensor");
    TensorListWrapper shared_weight2Wrapper = {at::TensorList(shared_weight2), (shared_weight2[0].scalar_type() == at::kByte ? aclDataType::ACL_FLOAT4_E2M1 : ConvertToAclDataType(shared_weight2[0].scalar_type()))};
    TORCH_CHECK(!shared_scales1.empty(), "shared_scales1 needs at least one typed tensor");
    TensorListWrapper shared_scales1Wrapper = {at::TensorList(shared_scales1), aclDataType::ACL_FLOAT8_E8M0};
    TORCH_CHECK(!shared_scales2.empty(), "shared_scales2 needs at least one typed tensor");
    TensorListWrapper shared_scales2Wrapper = {at::TensorList(shared_scales2), aclDataType::ACL_FLOAT8_E8M0};
    auto completion = at::empty({1}, context.options().dtype(at::kInt));
    at::IntArrayRef configurationRef(configuration);
    at::IntArrayRef dpRanksRef(dp_ranks);
    ACLNN_CMD(aclnnCamMegamoeCombineRecv, context, topk_ids, topk_weights, snapshot_data, snapshot_scales, shared_weight1Wrapper, shared_weight2Wrapper, shared_scales1Wrapper, shared_scales2Wrapper, configurationRef, dpRanksRef, activation_parameters, y, completion);
    return std::make_tuple(y, completion);
}
void RegisterMegamoeCombineRecv(pybind11::module_ &module)
{
    module.def("npu_megamoe_combine_recv", &Run);
}
}
