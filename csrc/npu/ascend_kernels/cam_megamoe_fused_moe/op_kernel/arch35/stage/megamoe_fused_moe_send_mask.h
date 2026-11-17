/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Implement the megamoe fused moe send mask stage of the AFD MegaMoE pipeline.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

#ifndef MEGAMOE_FUSED_MOE_SEND_MASK_H
#define MEGAMOE_FUSED_MOE_SEND_MASK_H

#include "../common/megamoe_fused_moe_utils.h"

namespace MegamoeFusedMoeImpl {

using namespace AscendC;

struct SendMaskConfig {
    uint32_t splitAttentionRanks = 0;
    uint64_t expertCountWinOffset;
    uint64_t routeIndexAlignSize;
    uint64_t routeIndexWinOffset;
    MegamoeFusedMoeSendMaskBufferConfig bufferConfig;
};

// 装配 MTE 路径唯一的 compact route 发送配置。
__aicore__ inline SendMaskConfig CreateSendMaskConfig(const Params &params, uint32_t aivCoreIdx)
{
    uint64_t routeIndexWinOffset =
        static_cast<uint64_t>(params.peermemInfo.maskRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    uint64_t expertCountWinOffset =
        static_cast<uint64_t>(params.peermemInfo.expertCountRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    const MegamoeFusedMoeSendMaskBufferConfig &bufferConfig = aivCoreIdx < params.tilingData->sendMaskCoreCountWithExtraExpert ?
                                                          params.tilingData->sendMaskConfigForCoreWithExtraExpert :
                                                          params.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    return {.expertCountWinOffset = expertCountWinOffset,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .routeIndexWinOffset = routeIndexWinOffset,
            .bufferConfig = bufferConfig};
}

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    LocalTensor<int32_t> topkIndexTensor;
    // Contiguous runtime-sized ring. Slot i starts at i * bufferConfig.bufferBytes.
    LocalTensor<uint8_t> routeRingTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

struct ExpertRouteInfo {
    int32_t ownedIdx;
    int32_t globalExpertId;
    int32_t bufferIdx;
};

// MTE Wave：把当前 route batch 中命中某专家的全局 topkIndex 压紧后直接写入对端槽。
// 每个 slot 只保存 index；raw count 在所有 ring 写完成后由 PublishExpertCounts 独立发布。
__aicore__ inline void LoadRouteBatch(SendMaskScratch &scratch, GlobalTensor<int32_t> &topkIdsGm, int32_t batchStart,
                                      int32_t validLen)
{
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    CreateVecIndex(scratch.topkIndexTensor, batchStart, validLen);
}

__aicore__ inline void GatherAndSendExpertCompactRoute(const MoeStageCommonConfig &common, GM_ADDR *winRankAddr,
                                                       const SendMaskConfig &config, SendMaskScratch &scratch,
                                                       const ExpertRouteInfo &routeInfo, int32_t validLen)
{
    const MegamoeFusedMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    const uint32_t compareMaskBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) / BITS_PER_BYTE;
    TEventID eventId = static_cast<TEventID>(routeInfo.bufferIdx);
    uint32_t slotOffset = routeInfo.bufferIdx * bufferConfig.bufferBytes;
    LocalTensor<uint8_t> compareMaskTensor = scratch.routeRingTensor[slotOffset];
    LocalTensor<uint32_t> compareMaskU32Tensor = compareMaskTensor.template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> tokValidIndexTensor =
        scratch.routeRingTensor[slotOffset + compareMaskBytes].template ReinterpretCast<int32_t>();

    WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
    uint64_t batchMatchedRouteCount = 0U;
    if (validLen > 0) {
        CompareScalar(compareMaskTensor, scratch.topkIdsTensor, routeInfo.globalExpertId, AscendC::CMPMODE::EQ,
                      validLen);
        GatherMask(tokValidIndexTensor, scratch.topkIndexTensor, compareMaskU32Tensor, true,
                   static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
    }
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    int32_t previousCount = scratch.sendCntAccTensor.GetValue(routeInfo.ownedIdx);
    int32_t remainingCapacity = static_cast<int32_t>(common.tokenNum) - previousCount;
    remainingCapacity = remainingCapacity > 0 ? remainingCapacity : 0;
    int32_t copiedCount = static_cast<int32_t>(batchMatchedRouteCount);
    copiedCount = remainingCapacity > copiedCount ? copiedCount : remainingCapacity;
    scratch.sendCntAccTensor.SetValue(routeInfo.ownedIdx, previousCount + copiedCount);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

    if (copiedCount > 0) {
        int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
        int32_t dstRank = routeInfo.globalExpertId / expertPerRank + config.splitAttentionRanks;
        int32_t localExpertId = routeInfo.globalExpertId % expertPerRank;
        uint64_t dstOffset = config.routeIndexWinOffset +
                             static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) +
                                                   static_cast<int32_t>(common.rankId)) *
                                 config.routeIndexAlignSize +
                             static_cast<uint64_t>(previousCount) * sizeof(int32_t);
        GlobalTensor<int32_t> dstRouteIndexGm;
        dstRouteIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(winRankAddr[dstRank] + dstOffset));
        DataCopyPad(dstRouteIndexGm, tokValidIndexTensor,
                    {1U, static_cast<uint32_t>(copiedCount * sizeof(int32_t)), 0U, 0U, 0U});
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
}

// 将本核连续专家区间的 raw count 发布到各目标 rank 的 [localExpert][sourceRank] 表。
__aicore__ inline void PublishExpertCounts(const MoeStageCommonConfig &common, GM_ADDR *winRankAddr,
                                           const SendMaskConfig &config, const SendMaskScratch &scratch,
                                           int32_t ownedExpertBegin, int32_t ownedExpertNum, int32_t arrivalEpoch)
{
    if (ownedExpertNum <= 0) {
        return;
    }
    /*
     * 到达标记内嵌：count 与跨卡同步信号走不同源核/不同通道(MTE3 vs scalar)，互连不保证
     * 到达序，接收端凭同步放行后可能读到在途 count(旧序靠共享 GMM1 时间垫层掩盖)。
     * 判据必须内嵌在数据自身的单次写内：高 8 位写 launch epoch(值域[0x80,0xFF]恒非零,
     * 避开窗口零初值)，低 24 位为真实 count(上限 maxOutputSize 远小于 2^24)。
     * 接收端逐槽校验 epoch 后取低 24 位(见 token_dispatch.h PrepareMoeExpertTokenCountTable)。
     */
    constexpr uint32_t EXPERT_COUNT_EPOCH_SHIFT = 24U; // Low 24 bits store the count; high 8 bits store the epoch.
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t rawCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        scratch.sendCntAccTensor.SetValue(ownedIdx, (arrivalEpoch << EXPERT_COUNT_EPOCH_SHIFT) | rawCount);
    }

    int32_t sourceRank = static_cast<int32_t>(common.rankId);
    int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
    int32_t worldSize = static_cast<int32_t>(common.worldSize);
    int32_t ownedOffset = 0;
    GlobalTensor<int32_t> dstCountGm;

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    while (ownedOffset < ownedExpertNum) {
        int32_t globalExpertId = ownedExpertBegin + ownedOffset;
        int32_t dstRank = globalExpertId / expertPerRank + config.splitAttentionRanks;
        int32_t localExpertBegin = globalExpertId % expertPerRank;
        int32_t remainingInRank = expertPerRank - localExpertBegin;
        int32_t segmentExpertCount = ownedExpertNum - ownedOffset;
        segmentExpertCount = segmentExpertCount < remainingInRank ? segmentExpertCount : remainingInRank;

        uint64_t dstOffset = config.expertCountWinOffset +
                             static_cast<uint64_t>(localExpertBegin * worldSize + sourceRank) * sizeof(int32_t);
        dstCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(winRankAddr[dstRank] + dstOffset));
        DataCopyExtParams countCopyParams{
            static_cast<uint16_t>(segmentExpertCount), static_cast<uint32_t>(sizeof(int32_t)), 0,
            static_cast<int64_t>(worldSize - 1) * static_cast<int64_t>(sizeof(int32_t)), 0U};

        if (ownedOffset % static_cast<int32_t>(INT32_PER_256B) == 0) {
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.sendCntAccTensor[ownedOffset],
                                                       countCopyParams);
        } else {
            // compact route 已发送完成，复用 topkIndexTensor 将非对齐 count 段重排到对齐起点。
            for (int32_t expertIdx = 0; expertIdx < segmentExpertCount; ++expertIdx) {
                scratch.topkIndexTensor.SetValue(expertIdx, scratch.sendCntAccTensor.GetValue(ownedOffset + expertIdx));
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.topkIndexTensor, countCopyParams);
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
        }
        ownedOffset += segmentExpertCount;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// 三条 MTE Wave 路径的唯一 route 入口。专家 ownership 仍使用连续 quotient/remainder 分配，
// 因而 count 表可继续按目标 rank 合并成二维 strided copy。
__aicore__ inline void GatherAndSendExpertCompactRoutes(const AivJobContext &job, const MoeStageCommonConfig &common,
                                                        const Params &params, GM_ADDR *winRankAddr,
                                                        const SendMaskConfig &config, SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const MegamoeFusedMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    WorkRange ownedExpertRange =
        GetBalancedWorkRange(params.tilingData->splitMoeRanks * common.moeExpertPerRank, job.jobIndex, job.totalJobs);
    if (ownedExpertRange.count == 0U) {
        return;
    }
    int32_t ownedExpertBegin = static_cast<int32_t>(ownedExpertRange.start);
    int32_t ownedExpertNum = static_cast<int32_t>(ownedExpertRange.count);

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertIdxGmAddr));
    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        const int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
        const int32_t realSendTotalNum =
            static_cast<int32_t>(static_cast<uint64_t>(common.tokenNum) * static_cast<uint64_t>(common.topK));
        const int32_t realRemain = realSendTotalNum - batchStart;
        int32_t validLen = bufferConfig.routeItemsPerBatch;
        if (realRemain < validLen) {
            validLen = realRemain > 0 ? realRemain : 0;
        }

        if (validLen > 0) {
            SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
            LoadRouteBatch(scratch, topkIdsGm, batchStart, validLen);
        }

        const int32_t batchRingBegin = batchIdx * ownedExpertNum;
        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
            int32_t globalExpertId = ownedExpertBegin + ownedIdx;
            ExpertRouteInfo routeInfo{.ownedIdx = ownedIdx,
                                      .globalExpertId = globalExpertId,
                                      .bufferIdx = (batchRingBegin + ownedIdx) % bufferConfig.bufferCount};
            GatherAndSendExpertCompactRoute(common, winRankAddr, config, scratch, routeInfo, validLen);
        }
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
    int32_t arrivalEpoch = (params.tilingData->splitSequence & 0x7FU) | 0x80U;
    PublishExpertCounts(common, winRankAddr, config, scratch, ownedExpertBegin, ownedExpertNum, arrivalEpoch);
}

} // namespace MegamoeFusedMoeImpl

#endif // MEGAMOE_FUSED_MOE_SEND_MASK_H
