/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineSend tiling function header file
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineSend tiling function header file
 */
#ifndef AFD_ASYNC_COMBINE_SEND_TILING_H
#define AFD_ASYNC_COMBINE_SEND_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace afd_async {
struct AfdAsyncCombineSendInfo {
    int64_t magic;
    uint32_t maxSeqLen;
    uint32_t hiddenSize;
    uint32_t topk;
    uint32_t moeRankNum;
    uint32_t attnRankNum;
    uint32_t routeExpertNumPerMoe;
    uint32_t moeRankId;
    uint32_t worldSize;
    uint32_t tpSize;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWorkspaceSize;
};
struct AfdAsyncCombineSendTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    AfdAsyncCombineSendInfo afdAsyncCombineSendInfo;
};
} // namespace afd_async

#endif // AFD_ASYNC_COMBINE_SEND_TILING_H
