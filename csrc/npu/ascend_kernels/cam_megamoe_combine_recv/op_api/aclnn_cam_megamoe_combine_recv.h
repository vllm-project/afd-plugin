/* Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2. */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Expose the ACLNN interface for the AFD MegaMoE operator.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */
#pragma once
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnCamMegamoeCombineRecvGetWorkspaceSize(const aclTensor *context, const aclTensor *topk_ids, const aclTensor *topk_weights, const aclTensor *snapshot_data, const aclTensor *snapshot_scales, const aclTensorList *shared_weight1, const aclTensorList *shared_weight2, const aclTensorList *shared_scales1, const aclTensorList *shared_scales2, const aclIntArray *configuration, const aclIntArray *dp_ranks, const aclFloatArray *activation_parameters, const aclTensor *y, const aclTensor *completion, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnCamMegamoeCombineRecv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
