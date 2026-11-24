/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Implement the SIMT routing kernels for afd_async_dispatch_send.
 * Create: 2026-11-23
 * Note: Each SIMT lane counts expert assignments and deduplicates payload destinations
 *       for its token.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#ifndef AFD_ASYNC_DISPATCH_SEND_SIMT_H
#define AFD_ASYNC_DISPATCH_SEND_SIMT_H

#include "kernel_operator.h"
#include "simt_api/cpp/kernel_simt_utils.h"

// This header is included only by A5 keys 110/111. All VF inputs/outputs are UB;
// the native caller owns GM copies and the public pipeline handoffs.
#define AFD_ASYNC_ROUTING_INLINE __simt_callee__ inline
#include "afd_async_routing_columns.h"
#undef AFD_ASYNC_ROUTING_INLINE

namespace afd_async::dispatch_send {
constexpr uint32_t ROUTING_VF_THREAD_COUNT = 128;

struct UbRoutingBackend {
    static __simt_callee__ inline uint32_t ReadId(__ubuf__ const int32_t *ids, uint64_t index)
    {
        return static_cast<uint32_t>(ids[index]);
    }

    static __simt_callee__ inline void AddCount(__ubuf__ uint32_t *counts,
        uint64_t index, uint32_t value)
    {
        AscendC::Simt::AtomicAdd(counts + index, value);
    }

    static __simt_callee__ inline uint32_t ReadCount(__ubuf__ const uint32_t *matrix,
        uint64_t index)
    {
        return matrix[index];
    }

    static __simt_callee__ inline void WritePrefix(__ubuf__ uint32_t *prefix,
        uint64_t index, uint32_t value)
    {
        prefix[index] = value;
    }
};

// One SIMT lane owns each token; UB atomics count assignments while a lane-local
// rank mask deduplicates that token's payload destinations.
__simt_vf__ __launch_bounds__(ROUTING_VF_THREAD_COUNT) inline void count_routing_tokens_vf(
    __ubuf__ const int32_t *ids, __ubuf__ uint32_t *counts, uint32_t tileTokens,
    uint32_t topk, uint32_t rankCount, uint32_t expertsPerRank)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lanes = AscendC::Simt::GetThreadNum<0>();
    afd_async::routing::CountTokensForLane<UbRoutingBackend>(
        ids, counts, tileTokens, topk, rankCount, expertsPerRank, lane, lanes);
}

// Each lane owns complete columns and scans source AIV rows in their original
// order. It writes only valid words in the caller's aligned UB staging rows.
__simt_vf__ __launch_bounds__(ROUTING_VF_THREAD_COUNT) inline void prefix_routing_columns_vf(
    __ubuf__ const uint32_t *matrix, __ubuf__ uint32_t *prefix, uint32_t aivCount,
    uint32_t columnStride, uint32_t columnStart, uint32_t columnCount,
    uint32_t prefixRowStride)
{
    const uint32_t lane = AscendC::Simt::GetThreadIdx<0>();
    const uint32_t lanes = AscendC::Simt::GetThreadNum<0>();
    afd_async::routing::PrefixColumnsForLane<UbRoutingBackend>(matrix, prefix,
        aivCount, columnStride, columnStart, columnCount, prefixRowStride, lane, lanes);
}
} // namespace afd_async::dispatch_send

#endif // AFD_ASYNC_DISPATCH_SEND_SIMT_H
