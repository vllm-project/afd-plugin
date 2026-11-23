// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef AFD_CAM_ASYNC_HOST_TILING_H
#define AFD_CAM_ASYNC_HOST_TILING_H

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../op_kernel/cam_async_simt_routing_budget.h"

namespace Cam::AsyncHost {
constexpr uint64_t UB_ALIGNMENT = 32;
constexpr uint64_t MAX_ACTIVE_AIV = 48;
// MC2 commEngine stores HcclAccelerator::AIV for the A5 MTE backend.
constexpr uint8_t MTE_HCCL_ACCELERATOR = 3;
constexpr int64_t A5_MAX_WORLD_SIZE = 64;
constexpr int64_t MAX_SOURCE_TOKEN_COUNT = std::numeric_limits<uint16_t>::max();
constexpr int64_t MAX_KERNEL_DIMENSION = std::numeric_limits<uint32_t>::max();
constexpr uint64_t IDS_CACHE_BYTES = 64 * 1024;
constexpr uint64_t BATCH_INFO_FIELDS = 5;
constexpr uint64_t QUANT_REDUCTION_FLOATS = 8;
constexpr uint64_t WINDOW_CURSOR_BYTES = 32;

inline uint64_t AlignBytes(uint64_t bytes)
{
    return (bytes + UB_ALIGNMENT - 1) / UB_ALIGNMENT * UB_ALIGNMENT;
}

inline uint64_t GetActiveAivCount(uint64_t platformAivCount)
{
    // The compact-v2 window reservation stays at 48 rows for every operator.
    return std::min(platformAivCount, MAX_ACTIVE_AIV);
}

// DS and CR must select this same count for a given source batch. FFN-side
// DR/CS continue to use GetActiveAivCount; budgets keep the baseline cap.
inline uint64_t GetSourceActiveAivCount(uint64_t platformAivCount, uint64_t batchTokens)
{
    return std::min(GetActiveAivCount(platformAivCount), batchTokens);
}

inline uint64_t DispatchSendUbBytes(uint64_t moeRanks, uint64_t experts,
    uint64_t hiddenSize, uint64_t activeAivCount, bool dynamicQuant)
{
    uint64_t prefix = IDS_CACHE_BYTES + AlignBytes(sizeof(int32_t) * experts)
        + AlignBytes(sizeof(uint8_t) * moeRanks) + AlignBytes(sizeof(uint16_t) * moeRanks)
        + AlignBytes(sizeof(uint16_t) * experts);
    // PreProcess statistics reuse the storage later occupied by the send buffers.
    uint64_t preprocess = prefix
        + AlignBytes(sizeof(uint32_t) * (moeRanks + experts) * activeAivCount);
    uint64_t send = prefix + AlignBytes(sizeof(uint32_t) * moeRanks)
        + AlignBytes(sizeof(uint32_t) * experts) + AlignBytes(sizeof(int64_t) * BATCH_INFO_FIELDS)
        + AlignBytes(sizeof(int16_t) * hiddenSize) + AlignBytes(UB_ALIGNMENT * experts)
        + AlignBytes(sizeof(uint32_t) * experts * 2);
    if (dynamicQuant) {
        send += AlignBytes(sizeof(float) * hiddenSize) * 2
            + AlignBytes(sizeof(float) * QUANT_REDUCTION_FLOATS);
    }
    // The optional prefix staging has an existing bounded scalar fallback.
    return std::max(preprocess, send);
}

inline uint64_t DispatchRecvUbBytes(uint64_t localExperts, uint64_t tpSize,
    uint64_t hiddenSize, uint64_t activeAivCount, uint64_t sourceCapacity,
    uint64_t topk, uint64_t receiveCapacity, bool dynamicQuant)
{
    // A5 may stage a second row only after the existing address slice fits.
    // Keep the mandatory one-row bound: insufficient spare UB uses B0 instead.
    uint64_t mandatory = AlignBytes(sizeof(int64_t) * BATCH_INFO_FIELDS)
        + AlignBytes(sizeof(int64_t) * (BATCH_INFO_FIELDS + tpSize + localExperts * tpSize))
        + AlignBytes(sizeof(int64_t) * localExperts)
        + AlignBytes(sizeof(int32_t) * localExperts) * 2
        + AlignBytes(dynamicQuant ? hiddenSize + UB_ALIGNMENT : sizeof(int16_t) * hiddenSize)
        + AlignBytes(sizeof(uint32_t) * localExperts * 2);
    // A chunk fits receiveCapacity, but one source can own all of its rows.
    uint64_t sourceRows = std::min(receiveCapacity, sourceCapacity * topk);
    uint64_t rowsPerAiv = (sourceRows + activeAivCount - 1) / activeAivCount;
    return mandatory + AlignBytes(sizeof(uint16_t) * rowsPerAiv);
}

inline uint64_t CombineSendUbBytes(uint64_t localExperts, uint64_t tpSize,
    uint64_t moeRanks, uint64_t hiddenSize)
{
    // The A5 two-slot pipeline splits spare reusable UB at runtime. Its optional
    // second row must not raise this baseline acceptance threshold.
    uint64_t metadata = AlignBytes(sizeof(int64_t)
        * (BATCH_INFO_FIELDS + tpSize + localExperts * tpSize))
        + AlignBytes(sizeof(uint32_t) * localExperts * tpSize)
        + AlignBytes(sizeof(uint32_t) * tpSize);
    // Token staging and the final completion-prefix scratch share one region.
    return metadata + std::max(AlignBytes(sizeof(int16_t) * hiddenSize),
        AlignBytes(sizeof(uint32_t) * moeRanks));
}

inline uint64_t CombineRecvUbBytes(uint64_t moeRanks, uint64_t experts,
    uint64_t hiddenSize, uint64_t activeAivCount)
{
    uint64_t prefix = IDS_CACHE_BYTES + AlignBytes(sizeof(uint16_t) * experts);
    uint64_t preprocess = prefix
        + AlignBytes(sizeof(uint32_t) * (moeRanks + experts) * activeAivCount);
    uint64_t reduce = prefix + AlignBytes(sizeof(uint32_t) * experts)
        + AlignBytes(sizeof(uint16_t) * experts) + AlignBytes(sizeof(float) * hiddenSize) * 3
        + AlignBytes(sizeof(int16_t) * hiddenSize) + AlignBytes(sizeof(uint32_t) * moeRanks);
    return std::max(preprocess, reduce);
}
} // namespace Cam::AsyncHost

#endif // AFD_CAM_ASYNC_HOST_TILING_H
