/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineRecv tiling function header file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineRecv tiling function header file
 */
#ifndef ASCENDC_AFD_ASYNC_COMBINE_RECV_TILING_H
#define ASCENDC_AFD_ASYNC_COMBINE_RECV_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace Cam {
struct AfdAsyncCombineRecvInfo {
    int64_t magic;
    uint32_t batchSize;
    uint32_t hiddenSize;
    uint32_t topk;
    uint32_t moeRankNum;
    uint32_t attnRankNum;
    uint32_t routeExpertNumPerMoe;
    uint32_t attnRankId;
    uint32_t worldSize;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWorkspaceSize;
};

struct AfdAsyncCombineRecvTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    AfdAsyncCombineRecvInfo moeDistributeCombineInfo;
};
} // namespace Cam

#endif //__ASCENDC_AFD_ASYNC_COMBINE_RECV_TILING_H
