/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "e2a.h"
#include "e2a_tiling.h"

using namespace Moe;
extern "C" __global__ __aicore__ void e2a(GM_ADDR expandX, GM_ADDR attenBatchSize, GM_ADDR x,
            GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(E2ATilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", E2ATilingData);
    GET_TILING_DATA_WITH_STRUCT(E2ATilingData, tiling_data, tiling);

    int batchSize = tiling_data.batchSize;
    int hiddenSize = tiling_data.hiddenSize;
    int topk = tiling_data.topk;
    int expertRankSize = tiling_data.expertRankSize;
    int attentionRankSize = tiling_data.attentionRankSize;
    int rank = tiling_data.rank;
    int rankSize = expertRankSize + attentionRankSize;

    if (TILING_KEY_IS(21)) {
        E2a<bfloat16_t> opKernel(rank, rankSize);
        opKernel.init(expandX, attenBatchSize, x, batchSize, hiddenSize, topk, expertRankSize, \
            attentionRankSize, rank, tiling);
        opKernel.process();
    }
}
