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
#include "a2e.h"
#include "a2e_tiling.h"

using namespace Moe;
extern "C" __global__ __aicore__ void a2e(GM_ADDR x, GM_ADDR expertIds, GM_ADDR expertScales,
            GM_ADDR expandX, GM_ADDR simulateExpertIds, GM_ADDR simulateExpertScales, GM_ADDR attenBatchSize,
            GM_ADDR xActiveMask, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(A2ETilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR < 2000", A2ETilingData);
    GET_TILING_DATA_WITH_STRUCT(A2ETilingData, tiling_data, tiling);

    int batchSize = tiling_data.batchSize;
    int hiddenSize = tiling_data.hiddenSize;
    int topk = tiling_data.topk;
    int expertRankSize = tiling_data.expertRankSize;
    int attentionRankSize = tiling_data.attentionRankSize;
    int rank = tiling_data.rank;
    int computeGate = tiling_data.computeGate;
    int rankSize = expertRankSize + attentionRankSize;

    if (TILING_KEY_IS(21)) {
        A2e<bfloat16_t, bfloat16_t, false> opKernel(rank, rankSize);
        opKernel.init(x, expertIds, expertScales, expandX, simulateExpertIds, simulateExpertScales, \
            attenBatchSize, xActiveMask, batchSize, hiddenSize, topk, expertRankSize, attentionRankSize, \
            rank, tiling, computeGate);
        opKernel.process();
    }
}
