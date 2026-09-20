/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef E2A_TILING_H
#define E2A_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace Moe {
struct E2ATilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    uint32_t batchSize;
    uint32_t hiddenSize;
    uint32_t topk;
    uint32_t expertRankSize;
    uint32_t attentionRankSize;
    uint32_t rank;
};
}

#endif
