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
 * Description: Provide shared utils definitions for split MegaMoE execution.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

/*!
 * \file megamoe_combine_recv_utils.h
 * \brief MegamoeCombineRecv 调度使用的通用 host/device 辅助函数。
 */

#ifndef MEGAMOE_COMBINE_RECV_UTILS_H
#define MEGAMOE_COMBINE_RECV_UTILS_H

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "megamoe_combine_recv_constants.h"
#include "megamoe_combine_recv_types.h"

namespace MegamoeCombineRecvImpl {

using namespace AscendC;

constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint32_t UINT64_BYTE_OFFSET_SHIFT = 3U;

struct WorkRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

/*
 * 对一段连续 int32 数据原地执行 inclusive prefix-scan。
 *
 * 路由计数全程保持 int32 精确累加。本实现专用于一段连续的一维数据，不需要通用多维 CumSum 的
 * 维度转换和额外临时 UB：每个向量寄存器内使用 Hillis-Steele scan，相邻寄存器通过上一寄存器
 * 末值 carry 串接。
 *
 * 本函数是 VF 内部的可复用计算单元，其他 VF 可直接调用，不会产生额外的 VF 启动。
 */
__simd_callee__ inline void InclusivePrefixSumInt32(__ubuf__ int32_t *values, uint32_t elementCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    AscendC::Reg::RegTensor<int32_t> prefixReg;
    AscendC::Reg::RegTensor<int32_t> shiftedReg;
    AscendC::Reg::RegTensor<int32_t> carryReg;
    AscendC::Reg::RegTensor<int32_t> laneIndexReg;
    AscendC::Reg::RegTensor<int32_t> shiftedIndexReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Arange(laneIndexReg, 0);
    AscendC::Reg::Duplicate(carryReg, 0, fullMask);
    for (uint32_t vectorOffset = 0U; vectorOffset < elementCount; vectorOffset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = elementCount - vectorOffset;
        validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
        uint32_t maskCount = validCount;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(maskCount);
        AscendC::Reg::LoadAlign(prefixReg, values + vectorOffset);

        for (uint32_t shift = 1U; shift < validCount; shift <<= 1U) {
            AscendC::Reg::Adds(shiftedIndexReg, laneIndexReg, -static_cast<int32_t>(shift), fullMask);
            AscendC::Reg::Maxs(shiftedIndexReg, shiftedIndexReg, 0, fullMask);
            AscendC::Reg::Gather(shiftedReg, prefixReg,
                                 reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(shiftedIndexReg));
            AscendC::Reg::MaskReg shiftedLaneMask;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(shiftedLaneMask, laneIndexReg,
                                                                  static_cast<int32_t>(shift), activeMask);
            AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(prefixReg, prefixReg, shiftedReg,
                                                                             shiftedLaneMask);
        }

        AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(prefixReg, prefixReg, carryReg, activeMask);
        AscendC::Reg::StoreAlign(values + vectorOffset, prefixReg, activeMask);

        AscendC::Reg::Duplicate(shiftedIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
        AscendC::Reg::Gather(carryReg, prefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(shiftedIndexReg));
    }
}

/*
 * MegamoeCombineRecv count 表的融合版本：先复用 int32 prefix-scan，再在同一次 VF 中提取每个专家的累计尾值
 * 并作差。相比先调用通用前缀和、再启动第二个 VF，融合实现省去一次 asc_vf_call 和外部流水同步。
 */
__simd_vf__ inline void ComputeExpertCountTablesVF(__ubuf__ int32_t *count, __ubuf__ int32_t *expertCounts,
                                                   uint32_t elementCount, uint32_t expertCount, uint32_t worldSize,
                                                   uint32_t maxOutputSize)
{
    InclusivePrefixSumInt32(count, elementCount);

    // prefix 已写回 UB；后续从同一区域 Gather 前建立 VEC_STORE -> VEC_LOAD 可见性。
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    AscendC::Reg::RegTensor<int32_t> expertEndPrefixReg;
    AscendC::Reg::RegTensor<int32_t> previousEndPrefixReg;
    AscendC::Reg::RegTensor<int32_t> expertCountReg;
    AscendC::Reg::RegTensor<int32_t> previousVectorEndReg;
    AscendC::Reg::RegTensor<int32_t> maxOutputSizeReg;
    AscendC::Reg::RegTensor<int32_t> laneIndexReg;
    AscendC::Reg::RegTensor<int32_t> gatherIndexReg;
    AscendC::Reg::RegTensor<int32_t> previousLaneIndexReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    int32_t maxOutputSizeInt32 = static_cast<int32_t>(maxOutputSize);
    AscendC::Reg::Arange(laneIndexReg, 0);
    AscendC::Reg::Duplicate(previousVectorEndReg, 0, fullMask);
    AscendC::Reg::Duplicate(maxOutputSizeReg, maxOutputSizeInt32, fullMask);
    for (uint32_t expertOffset = 0U; expertOffset < expertCount; expertOffset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = expertCount - expertOffset;
        validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
        uint32_t maskCount = validCount;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(maskCount);

        AscendC::Reg::Adds(gatherIndexReg, laneIndexReg, static_cast<int32_t>(expertOffset), fullMask);
        AscendC::Reg::Muls(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize), fullMask);
        AscendC::Reg::Adds(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize - 1U), fullMask);
        AscendC::Reg::Gather(expertEndPrefixReg, count,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(gatherIndexReg), activeMask);

        AscendC::Reg::Adds(previousLaneIndexReg, laneIndexReg, -1, fullMask);
        AscendC::Reg::Maxs(previousLaneIndexReg, previousLaneIndexReg, 0, fullMask);
        AscendC::Reg::Gather(previousEndPrefixReg, expertEndPrefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(previousLaneIndexReg));
        AscendC::Reg::Select(previousEndPrefixReg, previousVectorEndReg, previousEndPrefixReg, firstLaneMask);
        AscendC::Reg::MaskReg overflowMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(overflowMask, expertEndPrefixReg, maxOutputSizeInt32,
                                                              activeMask);
        AscendC::Reg::Select(expertCountReg, maxOutputSizeReg, expertEndPrefixReg, overflowMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(overflowMask, previousEndPrefixReg, maxOutputSizeInt32,
                                                              activeMask);
        AscendC::Reg::Select(previousEndPrefixReg, maxOutputSizeReg, previousEndPrefixReg, overflowMask);
        AscendC::Reg::Sub(expertCountReg, expertCountReg, previousEndPrefixReg, activeMask);
        AscendC::Reg::StoreAlign(expertCounts + expertOffset, expertCountReg, activeMask);

        AscendC::Reg::Duplicate(previousLaneIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
        AscendC::Reg::Gather(previousVectorEndReg, expertEndPrefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t> &>(previousLaneIndexReg));
    }
}

__aicore__ inline void ComputeExpertCountTables(LocalTensor<int32_t> countTensor,
                                                LocalTensor<int32_t> expertCountTensor, uint32_t expertCount,
                                                uint32_t worldSize, uint32_t maxOutputSize)
{
    __ubuf__ int32_t *count = reinterpret_cast<__ubuf__ int32_t *>(countTensor.GetPhyAddr());
    __ubuf__ int32_t *expertCounts = reinterpret_cast<__ubuf__ int32_t *>(expertCountTensor.GetPhyAddr());
    asc_vf_call<ComputeExpertCountTablesVF>(count, expertCounts, expertCount * worldSize, expertCount, worldSize,
                                            maxOutputSize);
}

__aicore__ inline WorkRange TilingByJobContext(uint32_t totalLen, uint32_t jobIndex, uint32_t totalJobs,
                                               uint32_t align = ALIGN_32)
{
    if (totalJobs == 0U || jobIndex >= totalJobs) {
        return {};
    }
    uint32_t lenPerJob = Ops::Base::CeilDiv(totalLen, totalJobs);
    uint32_t alignedLenPerJob = Ops::Base::CeilAlign(lenPerJob, align);
    uint32_t jobOffset = jobIndex * alignedLenPerJob;
    if (jobOffset >= totalLen) {
        return {};
    }
    uint32_t jobLen = jobOffset + alignedLenPerJob > totalLen ? totalLen - jobOffset : alignedLenPerJob;
    return {jobOffset, jobLen};
}

__aicore__ inline GroupSyncSlotLayout CalcGroupSyncSlotLayout(uint32_t expertTokenCount, uint32_t logicalCoreCount)
{
    uint32_t groupCount = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    return {totalSyncSlotCount / groupCount, totalSyncSlotCount % groupCount};
}

__aicore__ inline __gm__ int32_t *GetCombineSyncCounterAddress(__gm__ int32_t *expertCounterBase,
                                                               uint32_t localSyncSlotIndex)
{
    return expertCounterBase + static_cast<uint64_t>(localSyncSlotIndex) * INT_CACHELINE;
}

__aicore__ inline void GetGroupSyncSlotRange(uint32_t groupIndex, const GroupSyncSlotLayout &slotLayout,
                                             uint32_t &firstSyncSlot, uint32_t &syncSlotCount)
{
    syncSlotCount = slotLayout.baseSlotCountPerGroup + (groupIndex < slotLayout.extraSlotGroupCount ? 1U : 0U);
    uint32_t precedingExtraSlotCount =
        groupIndex < slotLayout.extraSlotGroupCount ? groupIndex : slotLayout.extraSlotGroupCount;
    firstSyncSlot = groupIndex * slotLayout.baseSlotCountPerGroup + precedingExtraSlotCount;
}

/*
 * 计算单个专家贡献的独立 M 分组数。不同专家使用不同权重，因此各专家不足
 * rowsPerMGroup 的尾块也必须分别占用一个分组。
 */
__aicore__ inline uint32_t GetMGroupCountForRows(uint64_t rowCount, uint32_t rowsPerMGroup)
{
    if (rowCount == 0U || rowsPerMGroup == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>((rowCount - 1U) / rowsPerMGroup + 1U);
}

// 当前 AIC 未分到该 Wave problem 的任何 tile 时，补偿推进 rolling 游标并返回 true。
__aicore__ inline bool HandleWaveProblemWithoutWork(uint32_t problemTileCount, const BlockJobContext &blockJob,
                                                    uint32_t &startBlockIdx)
{
    if constexpr (g_coreType == AIC) {
        if (problemTileCount < blockJob.totalJobs) {
            uint32_t firstOwnedTile =
                (blockJob.jobIndex < startBlockIdx ? blockJob.jobIndex + blockJob.totalJobs : blockJob.jobIndex) -
                startBlockIdx;
            if (firstOwnedTile >= problemTileCount) {
                startBlockIdx = (startBlockIdx + problemTileCount) % blockJob.totalJobs;
                return true;
            }
        }
    }
    return false;
}

// 计算剩余分组预算允许当前 Wave 在本专家内推进到的结束行偏移。
__aicore__ inline uint32_t GetWaveEndRowOffsetInExpert(uint64_t expertRowCount, uint32_t currentRowOffsetInExpert,
                                                       uint32_t remainingMGroupCount, uint32_t rowsPerMGroup)
{
    if (remainingMGroupCount == 0U || rowsPerMGroup == 0U || currentRowOffsetInExpert >= expertRowCount) {
        return currentRowOffsetInExpert;
    }
    uint64_t maxWaveEndRowOffset =
        static_cast<uint64_t>(currentRowOffsetInExpert) + static_cast<uint64_t>(remainingMGroupCount) * rowsPerMGroup;
    return static_cast<uint32_t>(expertRowCount < maxWaveEndRowOffset ? expertRowCount : maxWaveEndRowOffset);
}

// 连续均衡分配任务，前 totalWorkItems % workerCount 个执行单元各多处理一个任务。
__aicore__ inline WorkRange GetBalancedWorkRange(uint32_t totalWorkItems, uint32_t workerIdx, uint32_t workerCount)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t base = totalWorkItems / workerCount;
    uint32_t remainder = totalWorkItems % workerCount;
    uint32_t extraBefore = workerIdx < remainder ? workerIdx : remainder;
    return {workerIdx * base + extraBefore, base + static_cast<uint32_t>(workerIdx < remainder)};
}

// 以全局任务前缀轮转“多一个任务”的首 owner；Dispatch/Combine 共用这一公式。
__aicore__ inline WorkRange GetRotatedBalancedWorkRange(uint32_t totalWorkItems, uint32_t workerIdx,
                                                        uint32_t workerCount, uint64_t globalWorkPrefix)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t firstOwner = static_cast<uint32_t>(globalWorkPrefix % workerCount);
    uint32_t logicalWorkerIdx = workerIdx >= firstOwner ? workerIdx - firstOwner : workerIdx + workerCount - firstOwner;
    return GetBalancedWorkRange(totalWorkItems, logicalWorkerIdx, workerCount);
}

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// 使用调用方准备的全零 UB Tensor，清理当前 AIV 任务负责的连续 GM 区域。
template <int32_t JobAlignment>
__aicore__ inline void ResetWorkspaceRegion(const AivJobContext &job, GM_ADDR regionPtr, int32_t elementCount,
                                            int32_t batchElementCount, LocalTensor<int32_t> &resetTensor)
{
    WorkRange range = TilingByJobContext(static_cast<uint32_t>(elementCount), job.jobIndex, job.totalJobs,
                                         static_cast<uint32_t>(JobAlignment));
    GlobalTensor<int32_t> regionGm;
    regionGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(regionPtr));
    for (uint32_t offset = 0; offset < range.count; offset += static_cast<uint32_t>(batchElementCount)) {
        uint32_t batchCount = range.count - offset < static_cast<uint32_t>(batchElementCount) ?
                                  range.count - offset :
                                  static_cast<uint32_t>(batchElementCount);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(regionGm[range.start + offset], resetTensor, copyParams);
    }
}

#endif

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

template <bool IsPingPong = false>
__aicore__ inline void EndSync(int32_t vecSetSyncCom, uint16_t pingpongIdx = 0)
{
    if (vecSetSyncCom == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        if constexpr (IsPingPong) {
            if (vecSetSyncCom == 1) {
                WaitForVector(1U - pingpongIdx);
            } else {
                WaitForVector(pingpongIdx);
                WaitForVector(1U - pingpongIdx);
            }
        } else {
            WaitForVector();
        }
    }
}

__aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t &coreOffset, int32_t align = ALIGN_32)
{
    int32_t coreIdx = GetBlockIdx();
    int32_t coreNum = GetBlockNum() * 2;
    int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
    coreLen = lenPerCoreAlign;
    coreOffset = coreIdx * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

// 普通路径沿用 [expert][block] 的 32B 槽；Wave 一次发布完整表后使用
// [block][expert] 连续布局，减少逐专家元数据同步。
__aicore__ inline uint64_t GetExpertCountWorkspaceOffset(const BlockWorkspaceContext &workspace, uint32_t expertCount,
                                                         uint32_t expertIdx, bool useBlockMajorLayout)
{
    if (useBlockMajorLayout) {
        uint32_t blockMajorStride = Ops::Base::CeilAlign(expertCount, static_cast<uint32_t>(INT_CACHELINE));
        return static_cast<uint64_t>(workspace.blockIdx) * blockMajorStride + expertIdx;
    }
    return static_cast<uint64_t>(expertIdx) * INT32_PER_256B * workspace.blockNum +
           static_cast<uint64_t>(INT32_PER_256B) * workspace.blockIdx;
}

__aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sigAddr, int32_t compareValue)
{
    do {
        if (ReadGmByPassDCache(sigAddr) == compareValue) {
            return;
        }
    } while (true);
}

__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext *mc2Context, uint32_t rankId, uint32_t epRankId)
{
    uint32_t index = rankId > epRankId ? rankId - 1U : rankId;
    return mc2Context->hcommHandle_[index];
}

inline GM_ADDR g_winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(g_winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> UINT64_BYTE_OFFSET_SHIFT);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

template <typename ElementType>
__aicore__ inline GM_ADDR GetExpertWeightAddr(GM_ADDR tensorListAddr, bool isPerExpertWeightTensor, uint32_t expertIdx,
                                              uint64_t elementOffset)
{
    uint16_t tensorIdx = isPerExpertWeightTensor ? static_cast<uint16_t>(expertIdx) : 0U;
    GM_ADDR tensorAddr = GetTensorAddr(tensorIdx, tensorListAddr);
    if (isPerExpertWeightTensor) {
        return tensorAddr;
    }
    return tensorAddr + elementOffset * sizeof(ElementType);
}

template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&value)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(value));
}

template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&value)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(value)));
}

__aicore__ inline ExpertLoopState CreateExpertLoopState(const MoeStageCommonConfig &context)
{
    ExpertLoopState state;
    Get<N_VALUE>(state.problemShape) = context.gmm1OutputDim;
    Get<K_VALUE>(state.problemShape) = context.tokenHiddenDim;
    return state;
}

// 按专家顺序推进连续 token 行偏移，并用当前专家的 token 数更新 M 维度。
__aicore__ inline void UpdateExpertLoopState(ExpertLoopState &state, uint32_t expertIdx, uint64_t expertTokenCount)
{
    if (expertIdx != state.expertIdx) {
        state.globalTokenStartIndex += Get<M_VALUE>(state.problemShape);
    }
    state.expertIdx = expertIdx;
    Get<M_VALUE>(state.problemShape) = static_cast<int64_t>(expertTokenCount);
}

// 按当前 Wave 的剩余容量推进专家 token 位置，返回本次推进覆盖的 token 数。
template <uint32_t TileM>
__aicore__ inline uint32_t AdvanceExpertTokenPositionInWave(uint32_t expertTokenCount, uint32_t waveMGroupTarget,
                                                            uint32_t &waveMGroupCount, ExpertTokenPosition &position)
{
    uint32_t expertRemainingTokenCount = expertTokenCount - position.tokenIndexInExpert;
    uint32_t waveRemainingTokenCapacity = (waveMGroupTarget - waveMGroupCount) * TileM;
    uint32_t sliceTokenCount =
        expertRemainingTokenCount < waveRemainingTokenCapacity ? expertRemainingTokenCount : waveRemainingTokenCapacity;
    waveMGroupCount += Ops::Base::CeilDiv(sliceTokenCount, TileM);
    position.tokenIndexInExpert += sliceTokenCount;
    position.globalTokenIndex += sliceTokenCount;
    if (position.tokenIndexInExpert >= expertTokenCount) {
        ++position.expertIdx;
        position.tokenIndexInExpert = 0U;
    }
    return sliceTokenCount;
}

// 轮询 GM 中的 int32 ready flag，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagIsNonZero(__gm__ int32_t *flagAddr)
{
    while (AscendC::ReadGmByPassDCache(flagAddr) == 0) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

/*
 * 等待 Dispatch 发布当前 expert/block 的 token 总数。这里只同步专家级元数据；每个 256-row group
 * 的 Dispatch 数据就绪依赖由 WaitForGmm1InputReady 处理。
 */
__aicore__ inline void WaitForMoeExpertTokenCountReady(GM_ADDR tokenCountReadyFlagWorkspace,
                                                       const BlockWorkspaceContext &countWorkspace, uint32_t expertIdx)
{
    __gm__ int32_t *tokenCountReadyFlag = reinterpret_cast<__gm__ int32_t *>(tokenCountReadyFlagWorkspace) +
                                          static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum * INT_CACHELINE +
                                          static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WaitUntilGmFlagIsNonZero(tokenCountReadyFlag);
}

// 从当前 block 的 [block][expert] workspace 行读取专家 token 数。
__aicore__ inline uint32_t GetExpertTokenCountFromWorkspace(GM_ADDR expertTokenCountWorkspace,
                                                            const BlockWorkspaceContext &countWorkspace,
                                                            uint32_t expertCount, uint32_t expertIdx)
{
    uint64_t countSlotIndex = GetExpertCountWorkspaceOffset(countWorkspace, expertCount, expertIdx, true);
    __gm__ int32_t *expertTokenCountAddr =
        reinterpret_cast<__gm__ int32_t *>(expertTokenCountWorkspace) + countSlotIndex;
    return static_cast<uint32_t>(AscendC::ReadGmByPassDCache(expertTokenCountAddr));
}

/*
 * 在当前 WAVE 的剩余容量内规划下一个非空专家 slice。
 * 单次调用最多返回一个非空专家范围；空专家会被跳过。函数只更新当前 WAVE 的 M-group 计数，
 * 不修改调用方持有的 Dispatch 位置；调用方在 Dispatch 完成后使用返回范围的 end 提交进度。
 */
template <uint32_t TileM>
__aicore__ inline ExpertTokenRange PlanNextExpertTokenRangeInWave(GM_ADDR expertTokenCountWorkspace,
                                                                  const BlockWorkspaceContext &countWorkspace,
                                                                  uint32_t expertCount, uint32_t waveMGroupTarget,
                                                                  uint32_t &waveMGroupCount,
                                                                  const ExpertTokenPosition &position)
{
    ExpertTokenPosition plannedPosition = position;
    while (plannedPosition.expertIdx < expertCount && waveMGroupCount < waveMGroupTarget) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(expertTokenCountWorkspace, countWorkspace,
                                                                     expertCount, plannedPosition.expertIdx);
        // 空专家或已完成专家不占用 WAVE 容量，继续寻找下一个非空专家。
        if (expertTokenCount == 0U || plannedPosition.tokenIndexInExpert >= expertTokenCount) {
            ++plannedPosition.expertIdx;
            plannedPosition.tokenIndexInExpert = 0U;
            continue;
        }

        ExpertTokenRange range{plannedPosition, plannedPosition};
        AdvanceExpertTokenPositionInWave<TileM>(expertTokenCount, waveMGroupTarget, waveMGroupCount, plannedPosition);
        range.end = plannedPosition;
        return range;
    }
    return {plannedPosition, plannedPosition};
}

// 轮询 GM 中的 int32 flag 直至等于期望值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagEquals(__gm__ int32_t *flagAddr, int32_t expectedValue,
                                             int64_t pollBackoffCycles = GM_FLAG_POLL_BACKOFF_CYCLES)
{
    while (AscendC::ReadGmByPassDCache(flagAddr) != expectedValue) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < pollBackoffCycles) {
        }
    }
}

// 轮询 GM 中的 int32 计数直至不小于目标值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagAtLeast(__gm__ int32_t *flagAddr, int32_t targetValue)
{
    while (AscendC::ReadGmByPassDCache(flagAddr) < targetValue) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

// 全卡同步：各 AIV 按任务分工分摊 rank 握手，末尾执行本卡 AIV 同步。
__aicore__ inline void CrossRankSyncInWorldSize(GM_ADDR rankSyncInWorldPtr, uint32_t rankId, uint32_t worldSize,
                                                const AivJobContext &aivJob)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = reinterpret_cast<__gm__ int32_t *>(rankSyncInWorldPtr);
    __gm__ int32_t *syncCount = reinterpret_cast<__gm__ int32_t *>(rankSyncInWorldPtr + 48U * 1024U +
                                                                   static_cast<uint64_t>(aivJob.jobIndex) * 64U);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIdx = aivJob.jobIndex; rankIdx < worldSize; rankIdx += aivJob.totalJobs) {
        __gm__ int32_t *remoteSyncAddr =
            reinterpret_cast<__gm__ int32_t *>(g_winRankAddr_[rankIdx]) + rankId * INT_CACHELINE;
        WriteGmByPassDCache(remoteSyncAddr, count);
        GmSignalWaitBarrier(syncRank + rankIdx * INT_CACHELINE, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

} // namespace MegamoeCombineRecvImpl

#endif // MEGAMOE_COMBINE_RECV_UTILS_H
