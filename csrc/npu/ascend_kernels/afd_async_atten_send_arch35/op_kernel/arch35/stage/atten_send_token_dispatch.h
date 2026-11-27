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
 * Description: Implement the token dispatch stage of the AFD Mega pipeline.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#ifndef MEGA_DISPATCH_SEND_TOKEN_DISPATCH_H
#define MEGA_DISPATCH_SEND_TOKEN_DISPATCH_H

#include "../common/atten_send_utils.h"
#include "atten_send_token_quant.h"

namespace MegaDispatchSendImpl {

using namespace AscendC;

// Token Dispatch 的阶段私有配置。任务分工(BlockJobContext)与 count 槽位(BlockWorkspaceContext)
// 与 GMM 阶段共用编排层的同一份实例，不在此重复持有。
struct TokenDispatchConfig {
    uint64_t maxOutputSize;
    MegaDispatchSendDispatchBufferConfig bufferConfig;
    uint64_t routeIndexAlignSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
};

template <typename ActivationType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    // Contiguous runtime-sized ring. A slot view is built from this UB base on demand.
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
};

// 当前物理 AIV1 在单个专家内负责的左闭右开 row 区间。
struct ExpertDispatchCoreRange {
    uint32_t expertGlobalRowBegin;
    uint32_t localExpertRowBegin;
    uint32_t localExpertRowEnd;
};

// 由 tiling/peermem/quant 配置装配 Token Dispatch 阶段配置（普通与 wave 编排模板共用）。
__aicore__ inline TokenDispatchConfig CreateTokenDispatchConfig(const Params &params,
                                                                const QuantProcessConfig &quantProcessConfig)
{
    uint64_t quantWinOffset =
        static_cast<uint64_t>(params.peermemInfo.quantTokenScalePtr - params.peermemInfo.rankSyncInWorldPtr);
    return {.maxOutputSize = params.tilingData->maxOutputSize,
            .bufferConfig = params.tilingData->dispatchBufferConfig,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .quantWinOffset = quantWinOffset,
            .quantTokenAlignBytes = quantProcessConfig.quantTokenAlignBytes,
            .quantScaleAlignBytes = quantProcessConfig.quantScaleAlignBytes,
            .quantTokenScaleAlignBytes = quantProcessConfig.quantTokenScaleAlignBytes};
}

template <typename ActivationType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(const TokenDispatchConfig &context,
                                                                    const TokenDispatchScratch<ActivationType> &scratch,
                                                                    int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig &context, const Params &params,
                                                     TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
                                                     int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType> &remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(params.tilingData->topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(
        TOPK_INDEX, topkIndex % static_cast<int32_t>(params.tilingData->topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// copyIdx 同时是段内目标行号：三个调用点的 ring 排空顺序保证写出行与匹配序恒一一对应。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig &context, const Params &params,
                                                     TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
                                                     GlobalTensor<ActivationType> &tokenRevGlobalTensor,
                                                     GlobalTensor<QuantScaleType> &scaleRevGlobalTensor,
                                                     GlobalTensor<int32_t> &metaInfoGlobalTensor, int32_t copyStartIdx,
                                                     int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx);
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(params.tilingData->topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    DataCopyPad(tokenRevGlobalTensor[copyIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[copyIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[copyIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Prototype: MegaDispatchSend::CopyGMToGMPerToken. Copies one dispatch shard and emits its metadata.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(const TokenDispatchConfig &context, const Params &params,
                                                    GM_ADDR *winRankAddr, TokenDispatchScratch<ActivationType> &scratch,
                                                    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx,
                                                    int32_t copyNum)
{
    const MegaDispatchSendDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t bufferCount = bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params.workspaceInfo.dispatchRevDataPtr + rowDstOffset * scratch.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(
        params.workspaceInfo.dispatchRevScalePtr + rowDstOffset * scratch.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    metaInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.metaInfoPtr + rowDstOffset * INT32_PER_256B * sizeof(int32_t)));

    // The only caller enters this function for a non-empty match interval, so copyNum is at least one.
    // DispatchRankTokens completes the compact-index MTE2-to-S synchronization before the first scalar read;
    // the drain below covers cross-call reuse of the copy and metadata rings.
    int32_t firstTopkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, 0, firstTopkIndex,
                                                              remoteRankIdx, remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, issueIdx, topkIndex,
                                                                  remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, params, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyBufferIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, scratch, (copyNum - 1) % bufferCount, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// 发布一个 source-rank 段覆盖到的 GMM1 tile ready 计数。
__aicore__ inline void PublishGmm1TileReady(const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                            uint32_t localExpertId, int32_t gmm1TileRowCount,
                                            int32_t segmentExpertRowBegin, int32_t segmentExpertRowEnd)
{
    if (segmentExpertRowBegin >= segmentExpertRowEnd) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    __gm__ int32_t *gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * syncLayout.dispatchFlagSlotCountPerExpert;
    int32_t firstTileIdx = segmentExpertRowBegin / gmm1TileRowCount;
    int32_t lastTileIdx = (segmentExpertRowEnd - 1) / gmm1TileRowCount;
    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileExpertRowBegin = tileIdx * gmm1TileRowCount;
        int32_t tileExpertRowEnd = tileExpertRowBegin + gmm1TileRowCount;
        int32_t overlapRowBegin =
            segmentExpertRowBegin > tileExpertRowBegin ? segmentExpertRowBegin : tileExpertRowBegin;
        int32_t overlapRowEnd = segmentExpertRowEnd < tileExpertRowEnd ? segmentExpertRowEnd : tileExpertRowEnd;
        AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE, overlapRowEnd - overlapRowBegin);
    }
}

/*
 * 按单个 source rank 的 compact route-index ordinal 区间搬运 token 和 metadata。
 * compact index 已按 [expert][source rank] 分槽保存，因此这里只需按 ordinal 分批顺序读取，无需再次筛选 mask。
 */
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline int32_t DispatchRankTokens(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                             const Params &params, GM_ADDR *winRankAddr,
                                             TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
                                             int32_t expertGlobalRowBegin, uint32_t remoteRankIdx,
                                             int32_t rankSegmentRowBegin, int32_t segmentMatchOrdinalBegin,
                                             int32_t segmentMatchOrdinalEnd)
{
    const MegaDispatchSendDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t processedRouteCount = 0;
    int32_t dispatchedRowCount = 0;
    GlobalTensor<int32_t> remoteRouteIndexGlobal;
    while (processedRouteCount < segmentMatchOrdinalEnd - segmentMatchOrdinalBegin) {
        int32_t remainingRouteCount = segmentMatchOrdinalEnd - segmentMatchOrdinalBegin - processedRouteCount;
        int32_t batchDispatchRowCount = remainingRouteCount < bufferConfig.routeItemsPerBatch ?
                                            remainingRouteCount :
                                            bufferConfig.routeItemsPerBatch;
        int32_t routeOrdinal = segmentMatchOrdinalBegin + processedRouteCount;
        // 直接定位当前专家、source rank 的 compact ordinal，不再扫描完整 topK mask。
        uint64_t slotOffset =
            (static_cast<uint64_t>(localExpertId) * common.worldSize + remoteRankIdx) * context.routeIndexAlignSize;
        remoteRouteIndexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params.peermemInfo.maskRecvPtr + slotOffset + static_cast<uint64_t>(routeOrdinal) * sizeof(int32_t)));
        DataCopyExtParams routeCopyParams{1U, static_cast<uint32_t>(batchDispatchRowCount * sizeof(int32_t)), 0U, 0U,
                                          0U};
        DataCopyPadExtParams<int32_t> routeCopyPad{false, 0U, 0U, 0U};
        DataCopyPad(scratch.validTopkIndexTensor, remoteRouteIndexGlobal, routeCopyParams, routeCopyPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID4>();
        // compact ordinal 同时决定专家输出行位置，保证分批搬运后仍保持专家内连续布局。
        int32_t dispatchDstRowBegin = rankSegmentRowBegin + expertGlobalRowBegin + routeOrdinal;
        CopyTokensAndMetaForDispatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, dispatchDstRowBegin, remoteRankIdx, 0, batchDispatchRowCount);
        processedRouteCount += batchDispatchRowCount;
        dispatchedRowCount += batchDispatchRowCount;
    }

    return dispatchedRowCount;
}

/*
 * 在当前专家的 source-rank 累计行数中二分查找 globalRowIdx 所属的首个 rank。
 * 返回值作为 DispatchOwnedExpertRows 的扫描起点，后续只访问与本核范围相交的连续 rank 段。
 */
template <typename ActivationType>
__aicore__ inline uint32_t FindDispatchSourceRank(const MoeStageCommonConfig &common,
                                                  const TokenDispatchScratch<ActivationType> &scratch,
                                                  uint32_t localExpertId, uint32_t globalRowIdx)
{
    uint32_t expertRankBegin = localExpertId * common.worldSize;
    uint32_t searchBegin = expertRankBegin;
    uint32_t searchEnd = expertRankBegin + common.worldSize;
    while (searchBegin < searchEnd) {
        uint32_t middle = searchBegin + (searchEnd - searchBegin) / 2U;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(middle));
        if (rankGlobalRowEnd <= globalRowIdx) {
            searchBegin = middle + 1U;
        } else {
            searchEnd = middle;
        }
    }
    return searchBegin - expertRankBegin;
}

/*
 * 搬运当前物理 AIV1 在一个专家内负责的连续 row。
 * 该范围先按 source-rank prefix 拆段，再从 compact index 槽搬运 token/metadata；全部 rank 段完成后，
 * 按实际覆盖行数累加 GMM1 tile ready。AtomicAdd 允许同一 tile 被多个 range 分段发布。
 */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchOwnedExpertRows(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                               const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                               GM_ADDR *winRankAddr, TokenDispatchScratch<ActivationType> &scratch,
                                               uint32_t localExpertId, const ExpertDispatchCoreRange &coreRange)
{
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t coreGlobalRowBegin = coreRange.expertGlobalRowBegin + coreRange.localExpertRowBegin;
    uint32_t coreGlobalRowEnd = coreRange.expertGlobalRowBegin + coreRange.localExpertRowEnd;
    // prefix 单调递增，二分找到与本核首行相交的第一个 source rank，避免从 rank 0 线性扫描。
    uint32_t sourceRankIdx = FindDispatchSourceRank(common, scratch, localExpertId, coreGlobalRowBegin);
    int32_t dispatchedRowCount = 0;

    while (sourceRankIdx < common.worldSize) {
        uint32_t prefixIndex = localExpertId * common.worldSize + sourceRankIdx;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex));
        uint32_t rankGlobalRowBegin =
            prefixIndex == 0U ? 0U : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex - 1U));
        if (rankGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }

        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > rankGlobalRowBegin ? coreGlobalRowBegin : rankGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < rankGlobalRowEnd ? coreGlobalRowEnd : rankGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            int32_t rankDispatchedRowCount = DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, common, params, winRankAddr, scratch, localExpertId,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                static_cast<int32_t>(rankGlobalRowBegin - coreRange.expertGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin));
            dispatchedRowCount += rankDispatchedRowCount;
        }
        ++sourceRankIdx;
    }
    // 所有 source-rank 子段完成后统一发布本核贡献；GMM1 按 tile 累计到完整行数后即可启动。
    PublishGmm1TileReady(syncLayout, params, localExpertId, gmm1TileRowCount,
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowBegin) + dispatchedRowCount);
}

/*
 * Dispatch 唯一执行入口：搬运调用方给定的左闭右开专家 token 范围，并发布 GMM1 tile ready。
 * 调用方负责提前规划范围、推进 WAVE/专家位置以及在必要时恢复 prefix；本函数只做范围裁剪、AIV1 分工、
 * 专家/source-rank 拆段和数据发送。A8W8 可传整 WAVE，W4 也可传单个专家 slice，执行路径完全一致。
 */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchTokenRange(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                          const BlockJobContext &blockJob, const MoeSyncWorkspaceLayout &syncLayout,
                                          const Params &params, GM_ADDR *winRankAddr,
                                          TokenDispatchScratch<ActivationType> &scratch, const ExpertTokenRange &range)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || blockJob.totalJobs == 0U) {
        return;
    }

    // maxOutputSize 是 Dispatch workspace 的硬边界，先统一裁剪再做分核。
    uint32_t dispatchGlobalRowBegin = static_cast<uint32_t>(
        range.begin.globalTokenIndex < context.maxOutputSize ? range.begin.globalTokenIndex : context.maxOutputSize);
    uint32_t dispatchGlobalRowEnd = static_cast<uint32_t>(
        range.end.globalTokenIndex < context.maxOutputSize ? range.end.globalTokenIndex : context.maxOutputSize);
    if (dispatchGlobalRowEnd <= dispatchGlobalRowBegin) {
        return;
    }

    uint32_t dispatchRowCount = dispatchGlobalRowEnd - dispatchGlobalRowBegin;
    // 对整个输入范围一次连续均分；按全局起点轮转首 owner，避免余数长期集中在低编号 AIV1。
    WorkRange ownedRange =
        GetRotatedBalancedWorkRange(dispatchRowCount, blockJob.jobIndex, blockJob.totalJobs, dispatchGlobalRowBegin);
    if (ownedRange.count == 0U) {
        return;
    }
    uint32_t coreGlobalRowBegin = dispatchGlobalRowBegin + ownedRange.start;
    uint32_t coreGlobalRowEnd = coreGlobalRowBegin + ownedRange.count;

    // end 位于专家内部时需要包含该专家；恰好位于专家边界时 end.expertIdx 已指向下一专家。
    uint32_t lastExpertExclusive = range.end.expertIdx + (range.end.tokenIndexInExpert == 0U ? 0U : 1U);
    if (lastExpertExclusive > common.moeExpertPerRank) {
        lastExpertExclusive = common.moeExpertPerRank;
    }
    for (uint32_t expertIdx = range.begin.expertIdx; expertIdx < lastExpertExclusive; ++expertIdx) {
        uint32_t expertGlobalRowBegin =
            expertIdx == 0U ?
                0U :
                static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(expertIdx * common.worldSize - 1U));
        uint32_t expertGlobalRowEnd =
            static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue((expertIdx + 1U) * common.worldSize - 1U));
        if (expertGlobalRowEnd <= coreGlobalRowBegin) {
            continue;
        }
        if (expertGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }
        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > expertGlobalRowBegin ? coreGlobalRowBegin : expertGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < expertGlobalRowEnd ? coreGlobalRowEnd : expertGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            // 将本核的全局连续区间投影为当前专家内的局部 row 区间。
            ExpertDispatchCoreRange expertRange{expertGlobalRowBegin, overlapGlobalRowBegin - expertGlobalRowBegin,
                                                overlapGlobalRowEnd - expertGlobalRowBegin};
            DispatchOwnedExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
                context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
        }
    }
}

// Wave 进入逐专家流水前一次准备完整 count 表；每个物理 block 只发布一次 ready。
// W4 Wave-ahead 路径同时将 prefix 持久化到 GM，避免后续 Activation 覆盖 UB 后丢失 Dispatch 状态。
template <bool NeedCumsumReload = false, typename ActivationType>
__aicore__ inline void PrepareMoeExpertTokenCountTable(const MoeStageCommonConfig &common,
                                                       const BlockWorkspaceContext &countWorkspace,
                                                       const Params &params,
                                                       TokenDispatchScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;
    GlobalTensor<int32_t> expertCountGlobal;
    expertCountGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.peermemInfo.expertCountRecvPtr));
    DataCopyPad(scratch.cumsumInfoTensor, expertCountGlobal,
                {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();

    /*
     * 到达校验：count 槽高 8 位为发送侧写入的 launch epoch(与 rankSync 计数槽同源,见
     * send_mask.h PublishExpertCounts)。跨卡同步信号与 count 数据跨源/跨通道无到达序,
     * 同步放行不代表 count 已落地——对高位不匹配的槽单槽自旋重读至到达,再掩出低 24 位。
     */
    const auto *td = params.tilingData;
    int32_t expectEpoch = (td->splitSequence & 0x7FU) | 0x80U;
    auto *rawCountGm = reinterpret_cast<__gm__ int32_t *>(params.peermemInfo.expertCountRecvPtr);
    for (uint32_t slotIdx = 0; slotIdx < rawCountElementCount; ++slotIdx) {
        if (!SplitMember(td, slotIdx % common.worldSize)) {
            scratch.cumsumInfoTensor.SetValue(slotIdx, 0);
            continue;
        }
        int32_t slotValue = ReadGmByPassDCache(rawCountGm + slotIdx);
        while (((static_cast<uint32_t>(slotValue) >> 24U) & 0xFFU) != static_cast<uint32_t>(expectEpoch)) {
            slotValue = ReadGmByPassDCache(rawCountGm + slotIdx);
        }
        scratch.cumsumInfoTensor.SetValue(slotIdx, slotValue & 0x00FFFFFF);
    }
    SyncFuncStatic<AscendC::HardEvent::S_V, SYNC_EVENT_ID2>();

    ComputeExpertCountTables(scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor, common.moeExpertPerRank,
                             common.worldSize, params.tilingData->maxOutputSize);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    DataCopyPad(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    if constexpr (NeedCumsumReload) {
        DataCopyPad(scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
                    {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();

    __gm__ int32_t *countTableReady =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WriteGmByPassDCache(countTableReady, static_cast<int32_t>(1));
}

/*
 * W4 的 Activation/SwiGLU 会覆盖 AIV1 UB 0 开始的 prefix 表。调用方在执行 DispatchTokenRange 前，
 * 从当前物理 block 的 GM 备份恢复“首专家前缀 + 本次范围覆盖的专家前缀”。
 */
template <typename ActivationType>
__aicore__ inline void ReloadDispatchCumsumRange(const MoeStageCommonConfig &common,
                                                 TokenDispatchScratch<ActivationType> &scratch, uint32_t firstExpertIdx,
                                                 uint32_t lastExpertIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    if (firstExpertIdx >= common.moeExpertPerRank || firstExpertIdx > lastExpertIdx) {
        return;
    }
    if (lastExpertIdx >= common.moeExpertPerRank) {
        lastExpertIdx = common.moeExpertPerRank - 1U;
    }
    constexpr uint32_t INT32_PER_32B = ALIGN_32 / sizeof(int32_t);
    uint32_t requiredBeginIndex = firstExpertIdx == 0U ? 0U : firstExpertIdx * common.worldSize - 1U;
    uint32_t alignedBeginIndex = requiredBeginIndex / INT32_PER_32B * INT32_PER_32B;
    uint32_t requiredEndIndex = (lastExpertIdx + 1U) * common.worldSize;
    uint32_t reloadElementCount = requiredEndIndex - alignedBeginIndex;
    DataCopyPad(scratch.cumsumInfoTensor[alignedBeginIndex], scratch.cumsumInfoGlobalTensor[alignedBeginIndex],
                {1U, reloadElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
}

template <typename ActivationType>
__aicore__ inline void ExportCompactExpertTokenCounts(const MoeStageCommonConfig &common,
                                                      const BlockWorkspaceContext &countWorkspace, const Params &params,
                                                      TokenDispatchScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || countWorkspace.blockIdx != 0U) {
        return;
    }

    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    GlobalTensor<int32_t> expertRevTokenNums;
    expertRevTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.expertRevTokenNumsPtr));
    DataCopyPad(scratch.expertTokenNumsOutTensor, expertRevTokenNums[countOffset],
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_MTE3, SYNC_EVENT_ID2>();

    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID2>();
}

} // namespace MegaDispatchSendImpl

#endif // MEGA_DISPATCH_SEND_TOKEN_DISPATCH_H
