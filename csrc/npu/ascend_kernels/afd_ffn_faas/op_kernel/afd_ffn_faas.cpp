// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#define ASCENDC_CUBE_ONLY
#include "afd_ffn_faas_control.h"

// The CANN-generated launch wrapper references this name at file scope.
using AfdFaas::AfdFfnFaasTilingData;

extern "C" __global__ __aicore__ void afd_ffn_faas(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR probabilities, GM_ADDR w13, GM_ADDR w2,
    GM_ADDR layers, GM_ADDR request, GM_ADDR state, GM_ADDR scratch, GM_ADDR y, GM_ADDR status,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AfdFfnFaasTilingData);
    if (TILING_KEY_IS(1)) {
        // SyncAll<true> requires FFTS even though this branch uses only AIV.
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0);
        GET_TILING_DATA_WITH_STRUCT(AfdFfnFaasTilingData, data, tiling);
        TPipe pipe;
        AfdFaas::FaaSKernel kernel(data.info, pipe);
        kernel.Run(x, expertIds, probabilities, w13, w2, layers, request, state, scratch, y, status);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(AfdFfnFaasTilingData, data, tiling);
        if ASCEND_IS_AIC {
            const uint32_t limit = data.info.mode == AfdFaas::SERVE ? AfdFaas::MAX_WORK_PER_SERVE : 1;
            for (uint32_t step = 0; step < limit; ++step) {
                if (!AfdFaas::CubeExecute(data.info, scratch, w13, w2, tiling)) {
                    break;
                }
            }
        } else {
            TPipe pipe;
            AfdFaas::FaaSKernel kernel(data.info, pipe);
            kernel.Run(x, expertIds, probabilities, w13, w2, layers, request, state, scratch, y, status);
        }
    }
}
