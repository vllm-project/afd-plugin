/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Ascend C kernel entry point for afd_async_dispatch_recv.
 * Create: 2026-11-23
 * Note: The tiling key selects the kernel specialization.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#include "kernel_operator.h"
#include "afd_async_dispatch_recv.h"
#include "afd_async_dispatch_recv_tiling.h"

using namespace AscendC;
using namespace afd_async::dispatch_recv;
using namespace afd_async;

extern "C" __global__ __aicore__ void afd_async_dispatch_recv(
    GM_ADDR x, GM_ADDR commArgs,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR batchInfoOut, GM_ADDR epRecvCountRoutedOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AfdAsyncDispatchRecvTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncDispatchRecvTilingData, tilingData, tilingGM);

    int dynamicQuant = tilingData.afdAsyncDispatchRecvInfo.dynamicQuant;

    if (TILING_KEY_IS(100)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchRecv<bfloat16_t, bfloat16_t, false> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchRecv<bfloat16_t, int8_t, true> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    } else if (TILING_KEY_IS(101)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchRecv<float16_t, float16_t, false> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchRecv<float16_t, int8_t, true> op;
            op.Init(x, expandXOut, dynamicScalesOut,
                batchInfoOut, epRecvCountRoutedOut,
                workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    }
}
