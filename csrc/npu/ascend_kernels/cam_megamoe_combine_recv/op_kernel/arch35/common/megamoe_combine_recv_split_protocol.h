/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

namespace MegamoeCombineRecvImpl {

// Split protocol, MTE only. Each control word has exactly one remote writer.
// Counts retain the original in-band epoch; a full sequence gates DP admission.
constexpr uint32_t SPLIT_PUBLISHED = 0U;
constexpr uint32_t SPLIT_TAKEN = 1U;
constexpr uint32_t SPLIT_READY = 2U;
constexpr uint32_t SPLIT_CONSUMED = 3U;
__aicore__ inline __gm__ int32_t *SplitWord(GM_ADDR window, uint32_t peer, uint32_t field)
{
    return reinterpret_cast<__gm__ int32_t *>(window + static_cast<uint64_t>(peer) * 64U + field * 4U);
}
__aicore__ inline void SplitWait(__gm__ int32_t *word, uint32_t sequence)
{
    while (static_cast<uint32_t>(ReadGmByPassDCache(word)) < sequence) {
        int64_t start = GetSystemCycle();
        while (GetSystemCycle() - start < GM_FLAG_POLL_BACKOFF_CYCLES) {}
    }
}
__aicore__ inline bool SplitMember(const MegamoeCombineRecvTilingData *td, uint32_t rank)
{
    for (uint32_t index = 0; index < td->splitDpSize; ++index) {
        if (td->splitDpRanks[index] == rank) { return true; }
    }
    return false;
}
__aicore__ inline void SplitPublish(const Params &params, GM_ADDR *windows, uint32_t peer, uint32_t field)
{
    WriteGmByPassDCache(SplitWord(windows[peer], params.tilingData->splitRank, field),
                       static_cast<int32_t>(params.tilingData->splitSequence));
}
// Nonblocking fan-in: the last AIV1 to finish all of its pulls publishes the ACK.
// No AIC waits for this fan-in, so it does not cut the existing tile pipeline.
__aicore__ inline void SplitDispatchTaken(const Params &params, GM_ADDR *windows, uint32_t block, uint32_t blocks)
{
    const auto *td = params.tilingData;
    GM_ADDR own = params.peermemInfo.rankSyncInWorldPtr;
    auto *done = reinterpret_cast<__gm__ int32_t *>(own + 48U * 1024U);
    WriteGmByPassDCache(done + block * 16U, static_cast<int32_t>(td->splitLaunch));
    for (uint32_t index = 0; index < blocks; ++index) {
        if (static_cast<uint32_t>(ReadGmByPassDCache(done + index * 16U)) != td->splitLaunch) { return; }
    }
    for (uint32_t index = 0; index < td->splitDpSize; ++index) {
        SplitPublish(params, windows, td->splitDpRanks[index], SPLIT_TAKEN);
    }
}
__aicore__ inline void SplitWaitOutputSlot(const Params &params, uint32_t source)
{
    SplitWait(SplitWord(params.peermemInfo.rankSyncInWorldPtr, source, SPLIT_CONSUMED),
              params.tilingData->splitSequence - 1U);
}


// Each AIV1 advances its completed GMM2/Combine prefix at an existing Wave boundary.
// Completion is per source Attention, with a nonblocking local fan-in: no new
// inter-core barrier and no wait for unrelated later source ranges / experts.
__aicore__ inline void SplitCombineProgress(const Params &params, GM_ADDR *windows,
                                           uint32_t block, uint32_t blocks, uint32_t completedRows,
                                           __gm__ int32_t *prefix)
{
    const auto *td = params.tilingData;
    auto *progress = reinterpret_cast<__gm__ int32_t *>(params.peermemInfo.rankSyncInWorldPtr + 48U * 1024U);
    WriteGmByPassDCache(progress + block * 16U + 1U, static_cast<int32_t>(completedRows));
    uint32_t commonEnd = completedRows;
    for (uint32_t index = 0; index < blocks; ++index) {
        uint32_t end = static_cast<uint32_t>(ReadGmByPassDCache(progress + index * 16U + 1U));
        commonEnd = end < commonEnd ? end : commonEnd;
    }
    for (uint32_t member = 0; member < td->splitDpSize; ++member) {
        uint32_t source = td->splitDpRanks[member];
        // Prefix is expert-major, then source-major, and has already been snapshotted
        // in local workspace. It remains valid after dispatch_taken releases input metadata.
        uint32_t lastRequired = 0U;
        for (uint32_t expert = td->moeExpertPerRank; expert > 0U; --expert) {
            uint32_t offset = (expert - 1U) * td->epWorldSize + source;
            uint32_t end = static_cast<uint32_t>(ReadGmByPassDCache(prefix + offset));
            uint32_t begin = offset == 0U ? 0U : static_cast<uint32_t>(ReadGmByPassDCache(prefix + offset - 1U));
            if (end > begin) { lastRequired = end; break; }
        }
        if (lastRequired != 0U && commonEnd >= lastRequired) {
            SplitPublish(params, windows, source, SPLIT_READY);
        }
    }
}
}
