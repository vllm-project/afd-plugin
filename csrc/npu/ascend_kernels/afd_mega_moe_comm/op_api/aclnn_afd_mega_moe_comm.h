// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#pragma once
#include "aclnn/aclnn_base.h"
extern "C" {
__attribute__((visibility("default"))) aclnnStatus aclnnAfdMegaMoeCommGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *expertScales,
    const aclTensor *state, const aclTensor *batch, const aclIntArray *config, char *hcclGroupName,
    const aclTensor *packed, const aclTensor *scales, const aclTensor *metadata,
    const aclTensor *counts, const aclTensor *combined, const aclTensor *completion,
    uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default"))) aclnnStatus aclnnAfdMegaMoeComm(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}
