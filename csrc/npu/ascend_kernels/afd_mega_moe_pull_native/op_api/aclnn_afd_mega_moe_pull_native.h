// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#pragma once
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) aclnnStatus aclnnAfdMegaMoePullNativeGetWorkspaceSize(
    const aclTensor *anchor, const aclTensor *commArgs, const aclTensor *layerMap,
    const aclTensor *scratch, const aclTensorList *w13, const aclTensorList *w2,
    const aclTensorList *s13, const aclTensorList *s2, const aclTensorList *b13,
    const aclTensorList *b2, const aclIntArray *config, char *hcclGroupName, double routedScale,
    const aclTensor *completion, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnAfdMegaMoePullNative(void *workspace,
                                                                      uint64_t workspaceSize,
                                                                      aclOpExecutor *executor,
                                                                      aclrtStream stream);
#ifdef __cplusplus
}
#endif
