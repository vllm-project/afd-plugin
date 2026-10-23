// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#pragma once
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) aclnnStatus aclnnAfdFfnFaasGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *probabilities,
    const aclTensor *w13, const aclTensor *w2, const aclTensor *layers,
    const aclTensor *request, const aclTensor *state, const aclTensor *scratch,
    int64_t mode, const aclIntArray *config, char *group,
    const aclTensor *y, const aclTensor *status, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnAfdFfnFaas(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
