/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"
// ### PATCH START: AFD merged source layout
#include "arch35/megamoe_combine_recv_arch35.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "cam_megamoe_combine_recv_tiling.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "cam_megamoe_combine_recv_tiling_key.h"
// ### PATCH END: AFD merged source layout
using namespace AscendC;
using namespace Mc2Tiling;
template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType,
          uint8_t CommModeType, bool TopkWeightsPrefetch>
__global__ __aicore__ void cam_megamoe_combine_recv(GM_ADDR context, GM_ADDR topk_ids, GM_ADDR topk_weights, GM_ADDR snapshot_data, GM_ADDR snapshot_scales, GM_ADDR shared_weight1, GM_ADDR shared_weight2, GM_ADDR shared_scales1, GM_ADDR shared_scales2, GM_ADDR y, GM_ADDR completion, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegamoeCombineRecvTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegamoeCombineRecvTilingData, tilingData, tilingGM);
    using Weight = DTYPE_SHARED_WEIGHT1;
    constexpr bool matching = (DispatchQuantOutType == 4 && (Std::IsSame<Weight, fp8_e4m3fn_t>::value ||
                               Std::IsSame<Weight, fp4x2_e2m1_t>::value)) ||
                              (DispatchQuantOutType == 3 && Std::IsSame<Weight, fp8_e5m2_t>::value);
    if constexpr (matching) {
        MegamoeCombineRecvImpl::MegamoeCombineRecv<bfloat16_t, bfloat16_t, float, Weight, DispatchQuantOutType,
                            CombineQuantOutType, TopkWeightsPrefetch, false> op;
        op.Init(context, topk_ids, topk_weights, shared_weight1, shared_weight2,
                shared_scales1, shared_scales2, y, workspaceGM, &tilingData, tilingGM);
        op.ProcessSplit(snapshot_data, snapshot_scales);
    }
    if constexpr (g_coreType == AIV) {
        if (GetBlockIdx() == 0U) {
            WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(completion), int32_t(tilingData.splitSequence));
        }
    }
}
