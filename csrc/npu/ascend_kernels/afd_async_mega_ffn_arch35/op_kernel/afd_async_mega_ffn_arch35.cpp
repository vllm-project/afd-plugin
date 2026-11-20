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
#include "arch35/mega_ffn_wave_a8w8.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "arch35/mega_ffn_wave_a8w4.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "afd_async_mega_ffn_tiling.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "afd_async_mega_ffn_tiling_key.h"
// ### PATCH END: AFD merged source layout
using namespace AscendC;
using namespace Mc2Tiling;
template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType,
          uint8_t CommModeType, bool TopkWeightsPrefetch>
__global__ __aicore__ void afd_async_mega_ffn_arch35(GM_ADDR context, GM_ADDR weight1, GM_ADDR weight2, GM_ADDR weight_scales1, GM_ADDR weight_scales2, GM_ADDR expert_token_nums, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaFusedMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaFusedMoeTilingData, tilingData, tilingGM);
    using Weight = DTYPE_WEIGHT1;
    constexpr bool matching = (DispatchQuantOutType == 4 && (Std::IsSame<Weight, fp8_e4m3fn_t>::value ||
                               Std::IsSame<Weight, fp4x2_e2m1_t>::value)) ||
                              (DispatchQuantOutType == 3 && Std::IsSame<Weight, fp8_e5m2_t>::value);
    if constexpr (matching) {
        if constexpr (Std::IsSame<Weight, fp4x2_e2m1_t>::value) {
            MegaFusedMoeImpl::MegaFusedMoeA8W4Wave<bfloat16_t, bfloat16_t, float, Weight, DispatchQuantOutType,
                                       CombineQuantOutType, TopkWeightsPrefetch> op;
            op.Init(context, weight1, weight2, weight_scales1, weight_scales2, expert_token_nums,
                    workspaceGM, &tilingData, tilingGM);
            op.Process();
        } else {
            MegaFusedMoeImpl::MegaFusedMoeA8W8Wave<bfloat16_t, bfloat16_t, float, Weight, DispatchQuantOutType,
                                       CombineQuantOutType, TopkWeightsPrefetch, false> op;
            op.Init(context, weight1, weight2, weight_scales1, weight_scales2, expert_token_nums,
                    workspaceGM, &tilingData, tilingGM);
            op.Process();
        }
    }
}
