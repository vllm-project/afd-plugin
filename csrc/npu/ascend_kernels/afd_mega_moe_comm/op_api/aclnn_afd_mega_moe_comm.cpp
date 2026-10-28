// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include "aclnn_afd_mega_moe_comm.h"
#include "aclnnInner_afd_mega_moe_comm.h"
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *, int32_t);
extern "C" aclnnStatus aclnnAfdMegaMoeCommGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *expertScales,
    const aclTensor *state, const aclTensor *batch, const aclIntArray *config, char *hcclGroupName,
    const aclTensor *packed, const aclTensor *scales, const aclTensor *metadata,
    const aclTensor *counts, const aclTensor *combined, const aclTensor *completion,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
    return aclnnInnerAfdMegaMoeCommGetWorkspaceSize(x, expertIds, expertScales, state, batch,
        config, hcclGroupName, packed, scales, metadata, counts, combined, completion, workspaceSize, executor);
}
extern "C" aclnnStatus aclnnAfdMegaMoeComm(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor, aclrtStream stream) {
    if (NnopbaseSetHcclServerType) NnopbaseSetHcclServerType(executor, 1);
    return aclnnInnerAfdMegaMoeComm(workspace, workspaceSize, executor, stream);
}
