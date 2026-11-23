// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_ROUTING_COLUMNS_H
#define AFD_CAM_ASYNC_ROUTING_COLUMNS_H

#include <cstdint>

// The VF wrapper supplies the toolkit's SIMT callee qualifier. CPU tests use
// ordinary inline functions and instantiate the same loops with an owned backend.
#ifndef CAM_ASYNC_ROUTING_INLINE
#define CAM_ASYNC_ROUTING_INLINE inline
#define CAM_ASYNC_ROUTING_DEFAULT_INLINE
#endif

namespace Cam::AsyncRouting {
// The A5 communicator reserves at least one rank for Attention, leaving at
// most 63 MoE destinations. The caller admits this bound before invoking a VF.
constexpr uint32_t MAX_MASK_RANK_COUNT = 63;

// IDs and counters are UB pointers in production. Counters are initialized once
// before processing tiles; integer UB atomics accumulate all tiles of this AIV.
// Expert counts retain every assignment, while a rank gets one payload per token.
template <typename Backend, typename IdPointer, typename CountPointer>
CAM_ASYNC_ROUTING_INLINE void CountTokensForLane(IdPointer ids, CountPointer counts,
    uint32_t tileTokens, uint32_t topk, uint32_t rankCount, uint32_t expertsPerRank,
    uint32_t lane, uint32_t laneCount)
{
    for (uint32_t token = lane; token < tileTokens; token += laneCount) {
        uint64_t seenRanks = 0;
        for (uint32_t assignment = 0; assignment < topk; ++assignment) {
            const uint32_t expert = Backend::ReadId(
                ids, static_cast<uint64_t>(token) * topk + assignment);
            const uint32_t rank = expert / expertsPerRank;
            const uint64_t mask = uint64_t{1} << rank;
            Backend::AddCount(counts, static_cast<uint64_t>(rankCount) + expert, 1);
            if ((seenRanks & mask) == 0) {
                Backend::AddCount(counts, rank, 1);
                seenRanks |= mask;
            }
        }
    }
}

// This AIV owns a contiguous set of columns, as in the existing padded prefix
// staging path. All reads are from its gathered UB matrix. Row zero stays in
// that matrix; output rows 0..N-2 hold inclusive prefixes for source rows 1..N-1.
// The padded destination stride is preserved, and padding is never written.
template <typename Backend, typename MatrixPointer, typename PrefixPointer>
CAM_ASYNC_ROUTING_INLINE void PrefixColumnsForLane(MatrixPointer matrix,
    PrefixPointer prefix, uint32_t aivCount, uint32_t columnStride,
    uint32_t columnStart, uint32_t columnCount, uint32_t prefixRowStride,
    uint32_t lane, uint32_t laneCount)
{
    for (uint32_t column = lane; column < columnCount; column += laneCount) {
        uint32_t sum = Backend::ReadCount(matrix,
            static_cast<uint64_t>(columnStart) + column);
        for (uint32_t row = 1; row < aivCount; ++row) {
            sum += Backend::ReadCount(matrix,
                static_cast<uint64_t>(row) * columnStride + columnStart + column);
            Backend::WritePrefix(prefix,
                static_cast<uint64_t>(row - 1) * prefixRowStride + column, sum);
        }
    }
}
} // namespace Cam::AsyncRouting

#ifdef CAM_ASYNC_ROUTING_DEFAULT_INLINE
#undef CAM_ASYNC_ROUTING_DEFAULT_INLINE
#undef CAM_ASYNC_ROUTING_INLINE
#endif

#endif // AFD_CAM_ASYNC_ROUTING_COLUMNS_H
