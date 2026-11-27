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
// ### PATCH START: AFD shared Mega headers
#include "mega_layer_selection.h"
// ### PATCH END: AFD shared Mega headers

namespace afd_async_mega_ffn {
static void CheckExpertTensors(const std::vector<at::Tensor> &first,
                               const std::vector<at::Tensor> &second,
                               const std::vector<at::Tensor> &firstScales,
                               const std::vector<at::Tensor> &secondScales,
                               int64_t experts, int64_t hidden, int64_t intermediateFull)
{
    TORCH_CHECK(experts > 0, "local experts per layer must be positive");
    TORCH_CHECK(!first.empty() && !second.empty(), "Missing expert weights");
    TORCH_CHECK(first[0].dim() == 2 || first[0].dim() == 3,
                "Use layer-major 2D expert matrices or one stacked 3D tensor per layer");
    bool separate = first[0].dim() == 2;
    TORCH_CHECK(first.size() <= AfdAsyncLayer::MAX_TENSORS &&
                AfdAsyncLayer::LayerCount(static_cast<uint32_t>(first.size()),
                    static_cast<uint32_t>(experts), separate) != 0,
                "Weight TensorList must contain L*N matrices or L stacked tensors; at most 65536 entries");
    auto tensors = static_cast<int64_t>(first.size());
    TORCH_CHECK(int64_t(second.size()) == tensors &&
                int64_t(firstScales.size()) == tensors && int64_t(secondScales.size()) == tensors,
                "W1, W2 and both scale lists must have the same layer-major layout and length");
    for (int64_t index = 0; index < tensors; ++index) {
        int64_t packed = first[index].scalar_type() == at::kByte ? 2 : 1;
        TORCH_CHECK(first[index].scalar_type() == first[0].scalar_type() &&
                    second[index].scalar_type() == first[0].scalar_type(), "All layer weight dtypes must match");
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

at::Tensor Run(const at::Tensor &context, const std::vector<at::Tensor> &weight1, const std::vector<at::Tensor> &weight2, const std::vector<at::Tensor> &weight_scales1, const std::vector<at::Tensor> &weight_scales2, const std::vector<int64_t> &configuration, const std::vector<float> &activation_parameters)
{
    TORCH_CHECK(configuration.size() == 17, "Expected 17 configuration values");
    TORCH_CHECK(configuration[0] > 0 && configuration[1] > 0 && configuration[2] > 0 &&
                configuration[2] % configuration[1] == 0 && configuration[3] > 0 &&
                configuration[4] >= 1024 && configuration[4] <= 8192 && configuration[4] % 64 == 0 &&
                configuration[5] >= 512 && configuration[5] <= 8192 && configuration[5] % 256 == 0 &&
                configuration[6] > 0 && configuration[6] <= configuration[2] && configuration[6] <= 32 &&
                configuration[13] >= 0 && configuration[13] <= configuration[3], "Invalid split geometry");
    TORCH_CHECK(torch_npu::utils::is_npu(context), "context must be on NPU");
    CheckExpertTensors(weight1, weight2, weight_scales1, weight_scales2, configuration[2] / configuration[1], configuration[4], configuration[5]);
    auto expert_token_nums = at::empty({configuration[2] / configuration[1]}, context.options().dtype(at::kInt));
    TORCH_CHECK(!weight1.empty(), "weight1 needs at least one typed tensor");
    for (const auto &tensor : weight1) {
        TORCH_CHECK(tensor.device() == context.device(), "weight1 must share context's NPU");
    }
    for (const auto &tensor : weight2) {
        TORCH_CHECK(tensor.device() == context.device(), "weight2 must share context's NPU");
    }
    for (const auto &tensor : weight_scales1) {
        TORCH_CHECK(tensor.device() == context.device(), "weight_scales1 must share context's NPU");
    }
    for (const auto &tensor : weight_scales2) {
        TORCH_CHECK(tensor.device() == context.device(), "weight_scales2 must share context's NPU");
    }
    TensorListWrapper weight1Wrapper = {at::TensorList(weight1), (weight1[0].scalar_type() == at::kByte ? aclDataType::ACL_FLOAT4_E2M1 : ConvertToAclDataType(weight1[0].scalar_type()))};
    TORCH_CHECK(!weight2.empty(), "weight2 needs at least one typed tensor");
    TensorListWrapper weight2Wrapper = {at::TensorList(weight2), (weight2[0].scalar_type() == at::kByte ? aclDataType::ACL_FLOAT4_E2M1 : ConvertToAclDataType(weight2[0].scalar_type()))};
    TORCH_CHECK(!weight_scales1.empty(), "weight_scales1 needs at least one typed tensor");
    TensorListWrapper weight_scales1Wrapper = {at::TensorList(weight_scales1), aclDataType::ACL_FLOAT8_E8M0};
    TORCH_CHECK(!weight_scales2.empty(), "weight_scales2 needs at least one typed tensor");
    TensorListWrapper weight_scales2Wrapper = {at::TensorList(weight_scales2), aclDataType::ACL_FLOAT8_E8M0};
    at::IntArrayRef configurationRef(configuration);
    ACLNN_CMD(aclnnAfdAsyncMegaFfn, context, weight1Wrapper, weight2Wrapper, weight_scales1Wrapper, weight_scales2Wrapper, configurationRef, activation_parameters, expert_token_nums);
    return expert_token_nums;
}
void RegisterMegaFfn(pybind11::module_ &module)
{
    module.def("afd_async_mega_ffn", &Run);
}
}
