// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "aclnn_afd_ffn_faas.h"
#include "aclnnInner_afd_ffn_faas.h"

// Match CAM's MTE server setup. MC2 establishes peer windows; the kernel uses
// point-to-point window accesses, never an ordered cross-FFN collective.
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, int32_t serverType);
extern "C" aclnnStatus aclnnAfdFfnFaasGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *probabilities,
    const aclTensor *w13, const aclTensor *w2, const aclTensor *layers,
    const aclTensor *request, const aclTensor *state, const aclTensor *scratch,
    int64_t mode, const aclIntArray *config, char *group,
    const aclTensor *y, const aclTensor *status, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerAfdFfnFaasGetWorkspaceSize(x, expertIds, probabilities, w13, w2,
        layers, request, state, scratch, mode, config, group, y, status, workspaceSize, executor);
}
extern "C" aclnnStatus aclnnAfdFfnFaas(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor, aclrtStream stream)
{
    constexpr int32_t MTE_SERVER = 1;
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, MTE_SERVER);
    }
    return aclnnInnerAfdFfnFaas(workspace, workspaceSize, executor, stream);
}
