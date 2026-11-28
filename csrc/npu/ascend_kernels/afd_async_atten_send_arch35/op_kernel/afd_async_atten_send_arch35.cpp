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
 * Description: Dispatch the AFD Mega device kernel using host tiling parameters.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"
// ### PATCH START: AFD merged source layout
#include "arch35/atten_send_arch35.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "afd_async_atten_send_tiling.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "afd_async_atten_send_tiling_key.h"
// ### PATCH END: AFD merged source layout
using namespace AscendC;
using namespace Mc2Tiling;
template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType,
          uint8_t CommModeType, bool TopkWeightsPrefetch>
__global__ __aicore__ void afd_async_atten_send_arch35(GM_ADDR context, GM_ADDR x, GM_ADDR topk_ids, GM_ADDR topk_weights, GM_ADDR snapshot_data, GM_ADDR snapshot_scales, GM_ADDR completion, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaDispatchSendTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaDispatchSendTilingData, tilingData, tilingGM);
    using Weight = typename std::conditional<DispatchQuantOutType == 3, fp8_e5m2_t, fp8_e4m3fn_t>::type;
    MegaDispatchSendImpl::MegaDispatchSend<bfloat16_t, bfloat16_t, float, Weight, DispatchQuantOutType,
                        CombineQuantOutType, TopkWeightsPrefetch, false> op;
    op.Init(context, x, topk_ids, topk_weights, &tilingData);
    op.ProcessSplit(snapshot_data, snapshot_scales);
    if constexpr (g_coreType == AIV) {
        if (GetBlockIdx() == 0U) {
            WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(completion), int32_t(tilingData.splitSequence));
        }
    }
}
