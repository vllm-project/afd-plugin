/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Ascend C kernel entry point for afd_async_combine_send.
 * Create: 2026-11-23
 * Note: The tiling key selects the kernel specialization.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "afd_async_combine_send.h"
#include "afd_async_combine_send_tiling.h"

using namespace AscendC;
using namespace afd_async::combine_send;
using namespace afd_async;
extern "C" __global__ __aicore__ void afd_async_combine_send(
    GM_ADDR expandX, GM_ADDR commArgs, GM_ADDR batchInfo,
    GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AfdAsyncCombineSendTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", AfdAsyncCombineSendTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncCombineSendTilingData, tilingData, tilingGM);
    if (TILING_KEY_IS(100)) {
        AfdAsyncCombineSend<bfloat16_t> op;
        op.Init(expandX, workspaceGM, &pipe, &tilingData, commArgs, batchInfo, isCamComm);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        AfdAsyncCombineSend<float16_t> op;
        op.Init(expandX, workspaceGM, &pipe, &tilingData, commArgs, batchInfo, isCamComm);
        op.Process();
    }
}
