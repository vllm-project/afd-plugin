/* Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2. */
#pragma once
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnCamMegamoeDispatchSendGetWorkspaceSize(const aclTensor *context, const aclTensor *x, const aclTensor *topk_ids, const aclTensor *topk_weights, const aclIntArray *configuration, const aclIntArray *dp_ranks, const aclFloatArray *activation_parameters, int64_t layer_index, const aclTensor *snapshot_data, const aclTensor *snapshot_scales, const aclTensor *completion, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnCamMegamoeDispatchSend(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
