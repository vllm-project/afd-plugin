/* Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2. */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Expose the ACLNN interface for the AFD Mega operator.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */
#pragma once
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnAfdAsyncMegaFfnGetWorkspaceSize(const aclTensor *context, const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weight_scales1, const aclTensorList *weight_scales2, const aclIntArray *configuration, const aclIntArray *dp_ranks, const aclFloatArray *activation_parameters, const aclTensor *expert_token_nums, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnAfdAsyncMegaFfn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
