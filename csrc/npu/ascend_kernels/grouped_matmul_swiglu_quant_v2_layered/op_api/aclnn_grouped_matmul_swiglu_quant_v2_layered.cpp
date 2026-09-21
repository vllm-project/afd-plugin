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
 * Description: aclnn_grouped_matmul_swiglu_quant_v2_layered source file
 * Create: 2026-09-20
 * Note:
 * History: 2026-09-20 port from cam_async repository
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "gmm_dsq_base.h"
#include "grouped_matmul_swiglu_quant_v2_layered_utils.h"
#include "grouped_matmul_swiglu_quant_v2_layered.h"
#include "aclnn_grouped_matmul_swiglu_quant_v2_layered.h"
// NOTE: WeightNz variant (aclnnGroupedMatmulSwigluQuantWeightNzV2) intentionally not migrated yet.

using namespace op;
using namespace gmm_dsq;
using namespace gmm_dsq_base;

class GmmDsqHandlerFactory {
  private:
    std::unordered_map<NpuArch, std::unique_ptr<GroupedMatmulSwigluQuantHandler>> handlers_;

  public:
    void registerHandler(NpuArch npuArch, std::unique_ptr<GroupedMatmulSwigluQuantHandler> handler)
    {
        handlers_[npuArch] = std::move(handler);
    }

    GroupedMatmulSwigluQuantHandler *getHandler(NpuArch npuArch)
    {
        auto it = handlers_.find(npuArch);
        return it != handlers_.end() ? it->second.get() : nullptr;
    }
};

static aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(const char *interfaceName,
                                                                       GroupedMatmulSwigluQuantParamsBase &params,
                                                                       uint64_t *workspaceSize,
                                                                       aclOpExecutor **executor)
{
    GmmDsqHandlerFactory factory;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    factory.registerHandler(NpuArch::DAV_2201, std::make_unique<gmm_dsq_base::GroupedMatmulSwigluQuantBaseHandler>());
    factory.registerHandler(NpuArch::DAV_3510,
                            std::make_unique<gmmSwigluQuantV2::GroupedMatmulSwigluQuantBaseHandler>());

    if (auto *handler = factory.getHandler(npuArch)) {
        handler->Initialize(interfaceName, params, workspaceSize, executor);
        return handler->Process();
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interfaceName failed: the soc verison is not support");
    }

    return ACLNN_ERR_PARAM_INVALID;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulSwigluQuantV2LayeredGetWorkspaceSize(
    const aclTensor *x, const aclTensorList *allWeight, const aclTensorList *allWeightScale,
    const aclTensorList *allWeightAssistMatrix, const aclTensor *bias, const aclTensor *xScale,
    const aclTensor *smoothScale, const aclTensor *groupList, const aclTensor *layerIndex, int64_t dequantMode,
    int64_t dequantDtype, int64_t quantMode, int64_t groupListType, const aclIntArray *tuningConfigOptional,
    aclTensor *output, aclTensor *outputScale, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantV2Layered,
                   DFX_IN(x, allWeight, allWeightScale, xScale, groupList, layerIndex), DFX_OUT(output, outputScale));
    CHECK_COND((output != nullptr), ACLNN_ERR_PARAM_INVALID,
               "Expected a proper Tensor but got null for argument output.");
    GroupedMatmulSwigluQuantParamsBase params =
        GroupedMatmulSwigluQuantParamsBuilder::Create(x, allWeight, allWeightScale, output, outputScale)
            .SetXScale(xScale)
            .SetSmoothScale(smoothScale)
            .SetGroupList(groupList)
            .SetGroupListType(groupListType)
            .SetWeightAssistMatrix(allWeightAssistMatrix)
            .SetLayerIndex(layerIndex)
            .SetDequantAttr(dequantMode, dequantDtype)
            .SetQuantAttr(quantMode, static_cast<int64_t>(output->GetDataType()))
            .SetTransposeAttr(false)
            .SetBias(bias)
            .SetScenario()
            .SetTuningConfig(tuningConfigOptional)
            .Build();

    // Call the common interface
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(__FUNCTION__, params, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuantV2Layered(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantV2Layered);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantV2Layered launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif