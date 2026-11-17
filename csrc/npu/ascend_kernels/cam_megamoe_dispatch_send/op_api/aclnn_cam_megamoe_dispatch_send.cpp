/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Expose the ACLNN interface for the AFD MegaMoE operator.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

#include "aclnnInner_cam_megamoe_dispatch_send.h"

#include "aclnn_cam_megamoe_dispatch_send.h"

extern "C" {
aclnnStatus aclnnCamMegamoeDispatchSendGetWorkspaceSize(const aclTensor *context, const aclTensor *x, const aclTensor *topk_ids, const aclTensor *topk_weights, const aclIntArray *configuration, const aclIntArray *dp_ranks, const aclFloatArray *activation_parameters, int64_t layer_index, const aclTensor *snapshot_data, const aclTensor *snapshot_scales, const aclTensor *completion, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerCamMegamoeDispatchSendGetWorkspaceSize(context, x, topk_ids, topk_weights, configuration, dp_ranks, activation_parameters, layer_index, snapshot_data, snapshot_scales, completion, workspaceSize, executor);
}
aclnnStatus aclnnCamMegamoeDispatchSend(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerCamMegamoeDispatchSend(workspace, workspaceSize, executor, stream);
}
}
