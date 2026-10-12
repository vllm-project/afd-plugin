/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 */
#ifndef A2E_PACKET_H
#define A2E_PACKET_H

#include <cstdint>

namespace Moe {
constexpr uint32_t A2E_MX_GROUP_SIZE = 32;
constexpr uint32_t A2E_PACKET_ALIGNMENT = 32;
constexpr uint32_t A2E_QUANT_MODE_MXFP8 = 4;
constexpr uint32_t A2E_A5_MAX_RANKS = 64;

struct A2EPacketLayout {
    uint64_t expertIdsOffset;
    uint64_t expertScalesOffset;
    uint64_t xOffset;
    uint64_t xScaleOffset;
    uint64_t endOffset;
};

// Attribute validation bounds all products before this is used. Keep the
// legacy layout byte-for-byte; MXFP8 regions start on DMA block boundaries.
#ifdef __aicore__
#define A2E_LAYOUT_INLINE __aicore__ inline
#else
#define A2E_LAYOUT_INLINE inline
#endif
A2E_LAYOUT_INLINE uint64_t A2EAlignPacket(uint64_t bytes)
{
    return (bytes + A2E_PACKET_ALIGNMENT - 1) / A2E_PACKET_ALIGNMENT * A2E_PACKET_ALIGNMENT;
}

A2E_LAYOUT_INLINE A2EPacketLayout GetA2EPacketLayout(
    uint64_t batch, uint64_t hidden, uint64_t topk, uint64_t elementBytes, uint32_t quantMode)
{
    A2EPacketLayout layout;
    layout.expertIdsOffset = sizeof(int32_t);
    layout.expertScalesOffset = layout.expertIdsOffset + batch * topk * sizeof(int32_t);
    layout.xOffset = layout.expertScalesOffset + batch * topk * sizeof(float);
    if (quantMode == A2E_QUANT_MODE_MXFP8) {
        layout.xOffset = A2EAlignPacket(layout.xOffset);
    }
    layout.xScaleOffset = layout.xOffset + batch * hidden * elementBytes;
    layout.endOffset = layout.xScaleOffset;
    if (quantMode == A2E_QUANT_MODE_MXFP8) {
        layout.xScaleOffset = A2EAlignPacket(layout.xScaleOffset);
        layout.endOffset = A2EAlignPacket(
            layout.xScaleOffset + batch * ((hidden + A2E_MX_GROUP_SIZE - 1) / A2E_MX_GROUP_SIZE));
    }
    return layout;
}
#undef A2E_LAYOUT_INLINE
}
#endif
