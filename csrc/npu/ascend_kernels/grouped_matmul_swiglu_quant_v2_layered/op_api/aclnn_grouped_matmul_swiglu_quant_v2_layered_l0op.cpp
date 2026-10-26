/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: aclnn_grouped_matmul_swiglu_quant_v2_layered_l0op source file
 * Create: 2026-09-20
 * Note:
 * History: 2026-09-20 port from cam_async repository
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "util/math_util.h"
#include "grouped_matmul_swiglu_quant_utils.h"
#include "grouped_matmul_swiglu_quant_v2_layered.h"

using namespace op;
using namespace gmm_dsq;

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulSwigluQuantV2Layered);

constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;

// The out/outScale shapes and dtypes of the DAV_3510 arch are different from the generic ones.
void AllocOutTensorsV2(aclOpExecutor *executor, const aclTensorList *weightScale, int64_t m,
                       int64_t quantMode, int64_t quantDtype, bool transposeWeight,
                       aclTensor *&out, aclTensor *&scaleOut)
{
    int64_t n = 0;
    if (transposeWeight) {
        n = (*weightScale)[0]->GetViewShape().GetDim(1); // In the transposed case, dim 1 of weightScale is n
    } else {
        n = (*weightScale)[0]->GetViewShape().GetDim(2); // In the non-transposed case, dim 2 of weightScale is n
    }
    int64_t nAfterHalve = static_cast<int64_t>(n / 2); // outShape needs to be [M, N / 2]
    gert::Shape outShapeV2({m, nAfterHalve});
    gert::Shape scaleOutShapeV2;
    // When quantMode equals 2, the shape of out_scale is three-dimensional
    if (quantMode == 2) {
        int64_t nAfterSplit = static_cast<int64_t>(Ops::Base::CeilDiv(nAfterHalve, SWIGLU_SPLIT_SIZE));
        scaleOutShapeV2 = gert::Shape({m, nAfterSplit, 2});
    } else {
        scaleOutShapeV2 = gert::Shape({m});
    }
    out = executor->AllocTensor(outShapeV2, static_cast<ge::DataType>(quantDtype), ge::FORMAT_ND);
    // When quantMode equals 2, the DataType of outScale is FLOAT8_E8M0
    if (quantMode == 2) {
        scaleOut = executor->AllocTensor(scaleOutShapeV2, DataType::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    } else {
        scaleOut = executor->AllocTensor(scaleOutShapeV2, DataType::DT_FLOAT, ge::FORMAT_ND);
    }
}

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulSwigluQuantV2Layered(
    const aclTensor *x, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensor *xScale, const aclTensorList *weightAssistanceMatrix,
    const aclTensor *bias, const aclTensor *smoothScale, const aclTensor *groupList,
    const aclTensor *layerIndex, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t quantDtype, bool transposeWeight, int64_t groupListType,
    const aclIntArray *tuningConfigOptional, aclOpExecutor *executor)
{
    [[maybe_unused]] static const bool configRegistered = []() {
        op::BinConfigJsonDict::UpdateConfigJsonPath(
            GroupedMatmulSwigluQuantV2LayeredOpTypeId(),
            "grouped_matmul_swiglu_quant_v2_layered.json");
        return true;
    }();
    L0_DFX(GroupedMatmulSwigluQuantV2Layered, x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale,
           groupList, layerIndex, dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight,
           tuningConfigOptional);
    if (x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x is nullptr.");
        return std::tuple(nullptr, nullptr);
    }
    int64_t m = xScale->GetViewShape().GetDim(0);
    int64_t n = (*weightScale)[0]->GetViewShape().GetDim(1);
    int64_t nAfterHalve = static_cast<int64_t>(n / 2);
    gert::Shape outShape({m, nAfterHalve});
    gert::Shape scaleOutShape({m});
    auto out = executor->AllocTensor(outShape, DataType::DT_INT8, ge::FORMAT_ND);
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT, ge::FORMAT_ND);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        AllocOutTensorsV2(executor, weightScale, m, quantMode, quantDtype, transposeWeight, out, scaleOut);
    }
    auto ret = INFER_SHAPE(
        GroupedMatmulSwigluQuantV2Layered,
        OP_INPUT(x, xScale, groupList, layerIndex, weight, weightScale, weightAssistanceMatrix, bias, smoothScale),
        OP_OUTPUT(out, scaleOut),
        OP_ATTR(dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, groupListType,
                tuningConfigOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return std::tuple(nullptr, nullptr);
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GroupedMatmulSwigluQuantV2Layered,
        OP_INPUT(x, xScale, groupList, layerIndex, weight, weightScale, weightAssistanceMatrix, bias, smoothScale),
        OP_OUTPUT(out, scaleOut),
        OP_ATTR(dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, groupListType,
                tuningConfigOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple(nullptr, nullptr);
    }

    return std::tie(out, scaleOut);
}

} // namespace l0op
