// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_MEGA_MOE_PULL_NATIVE_TILING_H
#define AFD_MEGA_MOE_PULL_NATIVE_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "afd_native_mega_workspace.h"
namespace Cam {
struct AfdMegaMoePullNativeTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    int64_t config[14];
    uint32_t aicNum, aivNum, commAivNum, layerMapSize;
    alignas(8) uint8_t cube[1024];
    float routedScale;
};
}
#endif
