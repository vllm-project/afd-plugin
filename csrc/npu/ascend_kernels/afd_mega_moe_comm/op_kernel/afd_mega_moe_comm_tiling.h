// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#pragma once
#include "kernel_tiling/kernel_tiling.h"
namespace Cam {
// mode,maxSeqLen,H,K,F,A,E,rank,TP,rows,layer.
struct AfdMegaMoeCommTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    int64_t config[11];
    uint32_t aivNum;
    uint64_t ubSize, windowSize;
};
}
