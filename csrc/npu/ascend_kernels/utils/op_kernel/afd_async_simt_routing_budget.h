/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Calculate the unified buffer budget for SIMT routing.
 * Create: 2026-11-23
 * Note: The budget includes space for the SIMT data cache and compiler workspace.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#ifndef AFD_ASYNC_SIMT_ROUTING_BUDGET_H
#define AFD_ASYNC_SIMT_ROUTING_BUDGET_H

#include <cstdint>

#ifndef AFD_ASYNC_ROUTING_BUDGET_INLINE
#define AFD_ASYNC_ROUTING_BUDGET_INLINE constexpr
#define AFD_ASYNC_ROUTING_BUDGET_DEFAULT_INLINE
#endif

namespace afd_async::routing {
constexpr uint64_t SIMT_MAX_DATA_CACHE_BYTES = 128 * 1024;
constexpr uint64_t SIMT_COMPILER_RESERVED_BYTES = 8 * 1024;
// Conservative documented upper bound, not an observed/default cache setting.
// The separate scalar keys retain their complete baseline UB arena.
constexpr uint64_t SIMT_RESERVED_UB_BYTES =
    SIMT_MAX_DATA_CACHE_BYTES + SIMT_COMPILER_RESERVED_BYTES;
constexpr uint64_t SIMT_MIN_TOKENS_PER_AIV = 32;
constexpr uint64_t SIMT_ID_TILE_BYTES = 64 * 1024;
constexpr uint64_t SIMT_MAX_TOPK = SIMT_ID_TILE_BYTES / sizeof(int32_t);

AFD_ASYNC_ROUTING_BUDGET_INLINE bool CanUseSimtRouting(uint64_t physicalUbBytes,
    uint64_t baselinePeakUbBytes, uint64_t batchTokens, uint64_t activeAivCount,
    uint64_t topk)
{
    return activeAivCount != 0 && topk != 0 && topk <= SIMT_MAX_TOPK
        && batchTokens / activeAivCount >= SIMT_MIN_TOKENS_PER_AIV
        && baselinePeakUbBytes <= physicalUbBytes
        && SIMT_RESERVED_UB_BYTES <= physicalUbBytes - baselinePeakUbBytes;
}

AFD_ASYNC_ROUTING_BUDGET_INLINE bool CanFitPrefixStaging(
    uint64_t arenaBytes, uint64_t usedBytes, uint64_t stagingBytes)
{
    return usedBytes <= arenaBytes && stagingBytes <= arenaBytes - usedBytes;
}
} // namespace afd_async::routing

#ifdef AFD_ASYNC_ROUTING_BUDGET_DEFAULT_INLINE
#undef AFD_ASYNC_ROUTING_BUDGET_DEFAULT_INLINE
#undef AFD_ASYNC_ROUTING_BUDGET_INLINE
#endif

#endif // AFD_ASYNC_SIMT_ROUTING_BUDGET_H
