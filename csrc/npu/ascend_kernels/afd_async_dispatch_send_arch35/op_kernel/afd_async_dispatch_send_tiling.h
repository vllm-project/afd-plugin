/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Define the host-to-device tiling data for afd_async_dispatch_send.
 * Create: 2026-11-23
 * Note: The host tiler and the device entry point share this structure layout.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#ifndef AFD_ASYNC_DISPATCH_SEND_TILING_H
#define AFD_ASYNC_DISPATCH_SEND_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// ### PATCH START: Separate scalar and SIMT metadata compilation
#define AFD_ASYNC_DISPATCH_SIMT_BF16_KEY 110
#define AFD_ASYNC_DISPATCH_SIMT_FP16_KEY 111
// ### PATCH END: Separate scalar and SIMT metadata compilation

namespace afd_async {
struct AfdAsyncDispatchSendInfo {
    int64_t magic;
    uint32_t maxBatchSize;
    uint32_t batchSize;
    uint32_t hiddenSize;
    uint32_t topk;
    uint32_t moeRankNum;
    uint32_t attnRankNum;
    uint32_t routeExpertNumPerMoe;
    uint32_t attnRankId;
    uint32_t worldSize;
    uint32_t aivNum;
    uint32_t tpSize;
    uint64_t totalUbSize;
    uint64_t totalWorkspaceSize;
    uint32_t layerIndex;
    int dynamicQuant;
};

struct AfdAsyncDispatchSendTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    AfdAsyncDispatchSendInfo afdAsyncDispatchSendInfo;
};
}  // namespace afd_async

#endif
