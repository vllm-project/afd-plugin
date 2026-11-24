/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Ascend C kernel entry point for afd_async_dispatch_send.
 * Create: 2026-11-23
 * Note: The tiling key selects the kernel specialization.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#include "kernel_operator.h"
#include "afd_async_dispatch_send.h"
#include "afd_async_dispatch_send_tiling.h"

using namespace AscendC;
using namespace afd_async::dispatch_send;
using namespace afd_async;

// Patch reason: scalar fallback keys must compile without a SIMT VF call graph.
// Patch functionality: dispatch new A5 metadata keys to the SIMT specialization;
// scalar keys retain the original execution path and full UB arena.
// Signature: matches the original kernel entry; no added parameters.
extern "C" __global__ __aicore__ void afd_async_dispatch_send(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR commArgs, GM_ADDR expandXOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AfdAsyncDispatchSendTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000000000", AfdAsyncDispatchSendTilingData);
    TPipe pipe;
    int32_t isCamComm = 1;
    GET_TILING_DATA_WITH_STRUCT(AfdAsyncDispatchSendTilingData, tilingData, tilingGM);

    int dynamicQuant = tilingData.afdAsyncDispatchSendInfo.dynamicQuant;

    // ### PATCH START: Explicit scalar metadata specialization
    if (TILING_KEY_IS(100)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchSend<bfloat16_t, bfloat16_t, false, false> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchSend<bfloat16_t, int8_t, true, false> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    } else if (TILING_KEY_IS(101)) {
        if (dynamicQuant == 0) {
            AfdAsyncDispatchSend<float16_t, float16_t, false, false> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchSend<float16_t, int8_t, true, false> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
    }
    // ### PATCH END: Explicit scalar metadata specialization
    // ### PATCH START: Enumerate A5 keys while isolating their VF bodies
    // CANN discovers keys from preprocessed comparisons, including its first-key infer pass.
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3510)) || defined(__DAV_C310__)
    else if (TILING_KEY_IS(AFD_ASYNC_DISPATCH_SIMT_BF16_KEY)) {
#ifdef AFD_ASYNC_DISPATCH_SIMT_ROUTING
        if (dynamicQuant == 0) {
            AfdAsyncDispatchSend<bfloat16_t, bfloat16_t, false, true> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchSend<bfloat16_t, int8_t, true, true> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
#endif
    } else if (TILING_KEY_IS(AFD_ASYNC_DISPATCH_SIMT_FP16_KEY)) {
#ifdef AFD_ASYNC_DISPATCH_SIMT_ROUTING
        if (dynamicQuant == 0) {
            AfdAsyncDispatchSend<float16_t, float16_t, false, true> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        } else {
            AfdAsyncDispatchSend<float16_t, int8_t, true, true> op;
            op.Init(x, expertIds, workspaceGM, &pipe, &tilingData, commArgs, isCamComm);
            op.Process();
        }
#endif
    }
#endif
    // ### PATCH END: Enumerate A5 keys while isolating their VF bodies
}
