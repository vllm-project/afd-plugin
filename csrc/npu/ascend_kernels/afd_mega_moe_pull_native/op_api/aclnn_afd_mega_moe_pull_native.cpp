// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "aclnn_afd_mega_moe_pull_native.h"
#include "aclnnInner_afd_mega_moe_pull_native.h"
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, int32_t);
extern "C" aclnnStatus aclnnAfdMegaMoePullNativeGetWorkspaceSize(
    const aclTensor *anchor, const aclTensor *commArgs, const aclTensor *layerMap,
    const aclTensor *scratch, const aclTensorList *w13, const aclTensorList *w2,
    const aclTensorList *s13, const aclTensorList *s2, const aclTensorList *b13,
    const aclTensorList *b2, const aclIntArray *config, char *hcclGroupName, double routedScale,
    const aclTensor *completion, uint64_t *workspaceSize, aclOpExecutor **executor) {
    return aclnnInnerAfdMegaMoePullNativeGetWorkspaceSize(
        anchor, commArgs, layerMap, scratch, w13, w2, s13, s2, b13, b2, config, hcclGroupName,
        routedScale, completion, workspaceSize, executor);
}
extern "C" aclnnStatus aclnnAfdMegaMoePullNative(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, aclrtStream stream) {
    if (NnopbaseSetHcclServerType)
        NnopbaseSetHcclServerType(executor, 1);
    return aclnnInnerAfdMegaMoePullNative(workspace, workspaceSize, executor, stream);
}
