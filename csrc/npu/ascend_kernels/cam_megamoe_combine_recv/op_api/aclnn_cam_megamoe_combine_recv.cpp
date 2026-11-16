/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnnInner_cam_megamoe_combine_recv.h"

#include "aclnn_cam_megamoe_combine_recv.h"

extern "C" {
aclnnStatus aclnnCamMegamoeCombineRecvGetWorkspaceSize(const aclTensor *context, const aclTensor *topk_ids, const aclTensor *topk_weights, const aclTensor *snapshot_data, const aclTensor *snapshot_scales, const aclTensorList *shared_weight1, const aclTensorList *shared_weight2, const aclTensorList *shared_scales1, const aclTensorList *shared_scales2, const aclIntArray *configuration, const aclIntArray *dp_ranks, const aclFloatArray *activation_parameters, const aclTensor *y, const aclTensor *completion, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCamMegamoeCombineRecvGetWorkspaceSize(context, topk_ids, topk_weights, snapshot_data, snapshot_scales, shared_weight1, shared_weight2, shared_scales1, shared_scales2, configuration, dp_ranks, activation_parameters, y, completion, workspaceSize, executor);
}
aclnnStatus aclnnCamMegamoeCombineRecv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerCamMegamoeCombineRecv(workspace, workspaceSize, executor, stream);
}
}
