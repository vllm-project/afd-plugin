// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#ifndef AFD_NATIVE_MEGA_WORKSPACE_H
#define AFD_NATIVE_MEGA_WORKSPACE_H
#include <cstdint>
#include "afd_pull_layout.h"

#ifdef __CCE_AICORE__
#define AFD_NATIVE_DEVICE __aicore__
#else
#define AFD_NATIVE_DEVICE
#endif

namespace Cam {
constexpr uint32_t NATIVE_MAX_AIVS = 48;
constexpr uint32_t NATIVE_STAGES = 5;
constexpr uint32_t NATIVE_MAX_ROWS = Pull::MAX_CAPACITY;
constexpr uint32_t NATIVE_METADATA_ROWS = 512;
constexpr uint32_t NATIVE_TILE_ROWS = 64;
AFD_NATIVE_DEVICE constexpr uint64_t NativeAlign(uint64_t n) { return (n + 511) / 512 * 512; }

struct NativeMegaWorkspace {
    uint64_t batch, counts, tileCounts, layer, flags, scale1, scale2;
    uint64_t x, a1, a2, c1, c2, bytes;
    AFD_NATIVE_DEVICE NativeMegaWorkspace(uint64_t rows, uint64_t h, uint64_t i,
                                         uint64_t experts, uint64_t tp) {
        batch = 0;
        counts = NativeAlign(8 * Pull::MetadataWords(tp, experts));
        tileCounts = counts + NativeAlign(8 * experts);
        layer = tileCounts + NativeAlign(8 * experts * ((rows + NATIVE_TILE_ROWS - 1) / NATIVE_TILE_ROWS));
        flags = layer + 512;
        scale1 = flags + NativeAlign(NATIVE_STAGES * (NATIVE_MAX_AIVS + 1) * 64);
        scale2 = scale1 + NativeAlign(4 * rows);
        x = scale2 + NativeAlign(4 * rows);
        a1 = x + NativeAlign(rows * h);
        a2 = a1 + NativeAlign(rows * h);
        c1 = a2 + NativeAlign(rows * i);
        c2 = c1 + NativeAlign(8 * rows * i);
        bytes = c2 + NativeAlign(4 * rows * h);
    }
};
} // namespace Cam
#undef AFD_NATIVE_DEVICE
#endif
