/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGAMOE_FUSED_MOE_WORKSPACE_RESET_H
#define MEGAMOE_FUSED_MOE_WORKSPACE_RESET_H

#include "../common/megamoe_fused_moe_utils.h"

namespace MegamoeFusedMoeImpl {

using namespace AscendC;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// 清理连续 flag workspace；prefetch 路径额外清理独立分配的 GMM1 tile 状态区。
// 清零范围一律取自 WorkspaceInfo 在分配处记账的元素数，与布局恒同源。
template <bool TopkWeightsPrefetch>
__aicore__ inline void ResetSyncStatus(const AivJobContext &job, const Params &params, int32_t resetBatchElementCount,
                                       LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || resetBatchElementCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, params.workspaceInfo.flagActivationToGmm2Ptr,
                            static_cast<int32_t>(params.workspaceInfo.flagResetElementCount), resetBatchElementCount,
                            resetTensor);
    if constexpr (TopkWeightsPrefetch) {
        ResetWorkspaceRegion<1>(job, params.workspaceInfo.gmm1TileStatusPtr,
                                static_cast<int32_t>(params.workspaceInfo.gmm1TileStatusElementCount),
                                resetBatchElementCount, resetTensor);
    }
}
#endif

} // namespace MegamoeFusedMoeImpl

#endif // MEGAMOE_FUSED_MOE_WORKSPACE_RESET_H
