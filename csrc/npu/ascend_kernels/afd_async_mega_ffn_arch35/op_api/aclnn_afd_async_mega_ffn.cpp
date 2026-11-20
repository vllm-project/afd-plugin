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
 * Description: Expose the ACLNN interface for the AFD Mega operator.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#include "aclnnInner_afd_async_mega_ffn_arch35.h"

#include "aclnn_afd_async_mega_ffn.h"

extern "C" {
aclnnStatus aclnnAfdAsyncMegaFfnGetWorkspaceSize(const aclTensor *context, const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weight_scales1, const aclTensorList *weight_scales2, const aclIntArray *configuration, const aclFloatArray *activation_parameters, const aclTensor *expert_token_nums, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerAfdAsyncMegaFfnArch35GetWorkspaceSize(context, weight1, weight2, weight_scales1, weight_scales2, configuration, activation_parameters, expert_token_nums, workspaceSize, executor);
}
aclnnStatus aclnnAfdAsyncMegaFfn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerAfdAsyncMegaFfnArch35(workspace, workspaceSize, executor, stream);
}
}
