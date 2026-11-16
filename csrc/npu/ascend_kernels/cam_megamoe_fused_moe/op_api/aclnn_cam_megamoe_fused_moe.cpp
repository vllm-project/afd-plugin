/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnnInner_cam_megamoe_fused_moe.h"

#include "aclnn_cam_megamoe_fused_moe.h"

extern "C" {
aclnnStatus aclnnCamMegamoeFusedMoeGetWorkspaceSize(const aclTensor *context, const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weight_scales1, const aclTensorList *weight_scales2, const aclIntArray *configuration, const aclIntArray *dp_ranks, const aclFloatArray *activation_parameters, const aclTensor *expert_token_nums, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCamMegamoeFusedMoeGetWorkspaceSize(context, weight1, weight2, weight_scales1, weight_scales2, configuration, dp_ranks, activation_parameters, expert_token_nums, workspaceSize, executor);
}
aclnnStatus aclnnCamMegamoeFusedMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerCamMegamoeFusedMoe(workspace, workspaceSize, executor, stream);
}
}
