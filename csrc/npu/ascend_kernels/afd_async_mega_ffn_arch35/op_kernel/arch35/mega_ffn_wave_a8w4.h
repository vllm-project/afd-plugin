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
 * Description: Implement the device-side AFD Mega execution pipeline.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

/*!
 * \file mega_ffn_wave_a8w4.h
 * \brief MegaFusedMoe A8W4 动态 Wave 调度
 */

#ifndef MEGA_FUSED_MOE_WAVE_A8W4_H
#define MEGA_FUSED_MOE_WAVE_A8W4_H

#include "common/mega_ffn_utils.h"
#include "mega_ffn_arch35.h"

namespace MegaFusedMoeImpl {

#define TemplateMegaFusedMoeA8W4WaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaFusedMoeA8W4WaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaFusedMoeA8W4WaveTypeClass>
class MegaFusedMoeA8W4Wave : public MegaFusedMoe<TemplateMegaFusedMoeA8W4WaveTypeFunc, false> {
private:
    using MegaFusedMoeBase = MegaFusedMoe<TemplateMegaFusedMoeA8W4WaveTypeFunc, false>;
    friend MegaFusedMoeBase;

public:
    using MegaFusedMoeBase::Init;
    __aicore__ inline void Process();

private:
    using QuantOutType = typename MegaFusedMoeBase::QuantOutType;
    using ActivationType = typename MegaFusedMoeBase::ActivationType;
    using QuantScaleOutType = typename MegaFusedMoeBase::QuantScaleOutType;
    using ActivationQuantOutType = typename MegaFusedMoeBase::ActivationQuantOutType;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MegaFusedMoeBase::A_ELEMS_PER_BYTE;
    static constexpr uint32_t GMM1_TILE_M = MegaFusedMoeBase::GMM1_TILE_M;
    static constexpr uint32_t EPILOGUE_TILE_M = MegaFusedMoeBase::EPILOGUE_TILE_M;

    using MegaFusedMoeBase::DispatchBuffInit;
    using MegaFusedMoeBase::RunGmm2CombineForExpert;
    using MegaFusedMoeBase::exceptionDump_;
    using MegaFusedMoeBase::gmmLoopCount_;
    using MegaFusedMoeBase::epilogueOp_;
    using MegaFusedMoeBase::commonConfig_;
    using MegaFusedMoeBase::countWorkspace_;
    using MegaFusedMoeBase::gmmExecutionConfig_;
    using MegaFusedMoeBase::gmmTileSequence_;
    using MegaFusedMoeBase::mGroupsPerWave_;
    using MegaFusedMoeBase::syncWorkspaceLayout_;
    using MegaFusedMoeBase::moeWeightTensorListAddrs_;
    using MegaFusedMoeBase::params_;
    using MegaFusedMoeBase::startBlockIdx_;
    using MegaFusedMoeBase::tokenDispatchConfig_;
    using MegaFusedMoeBase::tokenDispatchScratch_;
    using MegaFusedMoeBase::waveCombineScratch_;
    __aicore__ inline bool IsPositionWithinWave(const ExpertTokenPosition &position, uint32_t waveMGroupCount) const
    {
        return position.expertIdx < commonConfig_.moeExpertPerRank && waveMGroupCount < mGroupsPerWave_;
    }

    __aicore__ inline void RunGmm1ActivationForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                      uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
                                                      uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline ExpertTokenPosition DispatchFirstWave();
    __aicore__ inline void DispatchNextWaveExpertSlice(ExpertTokenPosition &dispatchPosition,
                                                       uint32_t &nextDispatchWaveMGroupCount);
    __aicore__ inline ExpertTokenRange ProcessNextDispatchAndCurrentGmm1(
        const ExpertTokenPosition &waveBeginPosition, ExpertTokenPosition &dispatchPosition, ExpertLoopState &gmm1State,
        GMMAddrInfo &gmm1AddrInfo, uint32_t &startBlockIdx, uint32_t gmm1TilesPerMGroup);
    __aicore__ inline void RunGmm2CombineForWaveRange(const ExpertTokenRange &waveRange, ExpertLoopState &gmm2State,
                                                      GMMAddrInfo &gmm2AddrInfo,
                                                      WaveCombineBufferConfig &combineBufferConfig,
                                                      uint32_t &combineRowSequence);
    __aicore__ inline void ProcessMoeExpertStages();
};

// 更新当前专家切片的输入、输出及同步地址，并执行 GMM1 和 Activation。
template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline void MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::RunGmm1ActivationForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, uint32_t gmm1TilesPerMGroup)
{
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm1TilesPerMGroup;
    if (HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        return;
    }
    UpdateMoeExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                    A_ELEMS_PER_BYTE, true, TopkWeightsPrefetch>(
        gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, epilogueOp_,
        gmmAddrInfo, state, tokenStartIndexInExpert, gmm1TilesPerMGroup);
    ProblemShape sliceProblemShape = state.problemShape;
    Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
    RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                EPILOGUE_TILE_M, TopkWeightsPrefetch, false, true>(
        epilogueOp_, params_, sliceProblemShape, gmmAddrInfo, startBlockIdx, gmmTileSequence_,
        gmmExecutionConfig_.blockJob, static_cast<uint32_t>(state.globalTokenStartIndex) + tokenStartIndexInExpert,
        state.expertIdx);
}

template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline ExpertTokenPosition MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::DispatchFirstWave()
{
    ExpertTokenPosition dispatchPosition{};
    // 启动阶段：GMM1 开始消费输入前，由 AIV1 先 Dispatch 第一个完整 Wave。
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 1U) {
            return dispatchPosition;
        }
        ExpertTokenRange firstDispatchRange{dispatchPosition, dispatchPosition};
        ExpertTokenPosition plannedDispatchPosition = dispatchPosition;
        uint32_t firstDispatchWaveMGroupCount = 0U;
        while (IsPositionWithinWave(plannedDispatchPosition, firstDispatchWaveMGroupCount)) {
            ExpertTokenRange nextDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
                params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank,
                mGroupsPerWave_, firstDispatchWaveMGroupCount, plannedDispatchPosition);
            plannedDispatchPosition = nextDispatchRange.end;
            firstDispatchRange.end = nextDispatchRange.end;
        }
        // count-table 准备刚完成，UB prefix 仍有效；首 WAVE 无需从 GM 备份重复恢复。
        DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
            g_winRankAddr_, tokenDispatchScratch_, firstDispatchRange);
        // 首 WAVE 的全部数据和 ready flag 发布完成后，再提交 Dispatch 进度。
        dispatchPosition = plannedDispatchPosition;
        this->MaybePublishDispatchTaken(dispatchPosition.expertIdx >= commonConfig_.moeExpertPerRank);
    }
    return dispatchPosition;
}

template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline void MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::DispatchNextWaveExpertSlice(
    ExpertTokenPosition &dispatchPosition, uint32_t &nextDispatchWaveMGroupCount)
{
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 1U) {
            return;
        }
        ExpertTokenRange nextExpertDispatchRange = PlanNextExpertTokenRangeInWave<GMM1_TILE_M>(
            params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank,
            mGroupsPerWave_, nextDispatchWaveMGroupCount, dispatchPosition);
        // 规划完成后再判断是否得到有效专家 slice，调度函数本身不参与流程分支。
        if (nextExpertDispatchRange.end.globalTokenIndex > nextExpertDispatchRange.begin.globalTokenIndex) {
            // 当前 WAVE 的 Activation 可能覆盖 prefix UB；只恢复本专家 slice 所需的前缀。
            ReloadDispatchCumsumRange(commonConfig_, tokenDispatchScratch_, nextExpertDispatchRange.begin.expertIdx,
                                      nextExpertDispatchRange.begin.expertIdx);
            DispatchTokenRange<ActivationType, QuantScaleOutType, GMM1_TILE_M, TopkWeightsPrefetch>(
                tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, syncWorkspaceLayout_, params_,
                g_winRankAddr_, tokenDispatchScratch_, nextExpertDispatchRange);
        }
        // 有效 slice 完成 Dispatch，或仅跳过空专家后，提交本次规划的末尾位置。
        dispatchPosition = nextExpertDispatchRange.end;
        this->MaybePublishDispatchTaken(dispatchPosition.expertIdx >= commonConfig_.moeExpertPerRank);
    }
}

template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline ExpertTokenRange MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::ProcessNextDispatchAndCurrentGmm1(
    const ExpertTokenPosition &waveBeginPosition, ExpertTokenPosition &dispatchPosition, ExpertLoopState &gmm1State,
    GMMAddrInfo &gmm1AddrInfo, uint32_t &startBlockIdx, uint32_t gmm1TilesPerMGroup)
{
    ExpertTokenRange waveRange{waveBeginPosition, waveBeginPosition};
    uint32_t currentWaveMGroupCount = 0U;
    bool currentWaveNeedsGmm1 = true;
    uint32_t nextDispatchWaveMGroupCount = 0U;
    bool nextWaveNeedsDispatch = false;
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
        }
    }
    while (currentWaveNeedsGmm1 || nextWaveNeedsDispatch) {
        // AIV1 每轮先发送下一 Wave 的一个专家 slice，再处理当前 Wave 的一个专家 slice。
        if (nextWaveNeedsDispatch) {
            DispatchNextWaveExpertSlice(dispatchPosition, nextDispatchWaveMGroupCount);
            nextWaveNeedsDispatch = IsPositionWithinWave(dispatchPosition, nextDispatchWaveMGroupCount);
        }
        if (currentWaveNeedsGmm1) {
            if (waveRange.end.tokenIndexInExpert == 0U) {
                this->template PrepareGmmExpertState<true>(gmm1State, waveRange.end.expertIdx);
            }
            uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm1State.problemShape));
            uint32_t sliceTokenStartIndexInExpert = waveRange.end.tokenIndexInExpert;
            uint32_t sliceTokenCount = AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(
                expertTokenCount, mGroupsPerWave_, currentWaveMGroupCount, waveRange.end);
            if (sliceTokenCount != 0U) {
                RunGmm1ActivationForExpert(gmm1State, gmm1AddrInfo, startBlockIdx, sliceTokenStartIndexInExpert,
                                           sliceTokenCount, gmm1TilesPerMGroup);
            }
            currentWaveNeedsGmm1 = IsPositionWithinWave(waveRange.end, currentWaveMGroupCount);
        }
    }
    return waveRange;
}

// 消费一个已完成 Wave 的 GMM2/Combine（内联循环的逐字提取；调度本体在基类 RunGmm2CombineForExpert）。
// Wave 在专家内结束时需包含该专家；在专家边界结束时，range.end 已指向下一专家。
template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline void MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::RunGmm2CombineForWaveRange(
    const ExpertTokenRange &waveRange, ExpertLoopState &gmm2State, GMMAddrInfo &gmm2AddrInfo,
    WaveCombineBufferConfig &combineBufferConfig, uint32_t &combineRowSequence)
{
    uint32_t waveGmm2ExpertEndExclusive = waveRange.end.expertIdx + (waveRange.end.tokenIndexInExpert == 0U ? 0U : 1U);
    for (uint32_t expertIdx = waveRange.begin.expertIdx; expertIdx < waveGmm2ExpertEndExclusive; ++expertIdx) {
        uint32_t sliceTokenStartIndexInExpert =
            expertIdx == waveRange.begin.expertIdx ? waveRange.begin.tokenIndexInExpert : 0U;
        // WAVE 从专家起点开始时，准备该专家的 GMM2 状态；跨 WAVE slice 沿用已有状态。
        if (sliceTokenStartIndexInExpert == 0U) {
            this->template PrepareGmmExpertState<false>(gmm2State, expertIdx);
        }
        uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(gmm2State.problemShape));
        uint32_t sliceTokenEndIndexInExpert =
            expertIdx == waveRange.end.expertIdx ? waveRange.end.tokenIndexInExpert : expertTokenCount;
        uint32_t sliceTokenCount = sliceTokenEndIndexInExpert - sliceTokenStartIndexInExpert;
        if (sliceTokenCount != 0U) {
            uint64_t sliceGlobalEndIndex =
                static_cast<uint64_t>(gmm2State.globalTokenStartIndex) + sliceTokenEndIndexInExpert;
            bool isFinalCombine = waveRange.end.expertIdx >= commonConfig_.moeExpertPerRank &&
                                  sliceGlobalEndIndex >= waveRange.end.globalTokenIndex;
            // W4 的 GMM2/Combine 调度集中在基类，派生模板只负责提供当前专家 slice。
            RunGmm2CombineForExpert(gmm2State, gmm2AddrInfo, startBlockIdx_, sliceTokenStartIndexInExpert,
                                    sliceTokenCount, combineBufferConfig, combineRowSequence, gmmTileSequence_,
                                    isFinalCombine);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        DrainCombineRowBuffers(combineRowSequence, combineBufferConfig.rowBufferCount);
    }
    this->PublishSourceCombineProgress(waveRange.end);
}

/*
 * 按 GMM1 调度负载将专家划分为动态 Wave。启动阶段先 Dispatch 第一个完整 Wave；稳态阶段由 AIV1
 * 按专家 slice 交替执行下一 Wave 的 Dispatch 和当前 Wave 的 Activation，同时 AIC/AIV0 执行当前 Wave 的 GMM1。
 * 整 Wave 与专家 slice 都向 DispatchTokenRange 传入显式 [begin, end) 范围。当前 Wave 完成后立即启动
 * GMM2/Combine，无需等待所有专家的 GMM1。
 * 大 bs 下 GMM2 滞后一拍：wave w 的 GMM2/Combine 延至 GMM1(w+1) 之后，用下一 Wave 的 GMM1
 * 覆盖 AIC 等激活的自旋（对齐 A8W8 的 gmm2LaggedTarget 设计；A8W4 的 AIV0 prologue 与
 * AIV1 combine 在 GMM2 调用内均有实活，故三角色整调用同序滞后）。
 */
template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline void MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::ProcessMoeExpertStages()
{
    // GMM1/GMM2 交错流水只记录一次阶段入口，各 Wave 完成轮次由独立计数记录。
    exceptionDump_.UpdateStage(MegaFusedMoeImpl::Stage::MOE_GMM1_ACTIVATION);
    uint64_t gmm1Count = 0U;
    uint64_t gmm2Count = 0U;
    DispatchBuffInit();
    PrepareMoeExpertTokenCountTable<true>(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_,
                                         moeWeightTensorListAddrs_);
    WaveCombineBufferConfig combineBufferConfig{};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineBufferConfig = InitWaveCombineBuffers<CombineQuantMode>(commonConfig_, waveCombineScratch_);
    }
    uint32_t combineRowSequence = 0U;

    ExpertLoopState gmm1State = CreateExpertLoopState(commonConfig_);
    ExpertLoopState gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1AddrInfo{};
    GMMAddrInfo gmm2AddrInfo{};

    // MoE GMM1 与 GMM2 沿用同一分核游标，按 tile 顺序持续滚动。

    const uint32_t gmm1TilesPerMGroup =
        Ops::Base::CeilDiv(commonConfig_.gmm1OutputDim / ACTIVATION_N_HALF, static_cast<uint32_t>(L1_TILE_N));

    ExpertTokenPosition dispatchPosition = DispatchFirstWave();
    // GMM2 滞后一拍门控：与 A8W8 的 GMM2_LAG_MIN_TOKEN_NUM(4096) 同门槛；
    // 关闭态与现役"当前 Wave 完成后立即消费"逐字等价。
    const bool gmm2LagActive = commonConfig_.tokenNum >= 4096U;
    bool hasPendingWave = false;
    ExpertTokenRange pendingWaveRange{};
    ExpertTokenPosition gmm1Position{};
    while (gmm1Position.expertIdx < commonConfig_.moeExpertPerRank) {
        ExpertTokenRange waveRange = ProcessNextDispatchAndCurrentGmm1(
            gmm1Position, dispatchPosition, gmm1State, gmm1AddrInfo, startBlockIdx_, gmm1TilesPerMGroup);
        gmm1Position = waveRange.end;
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM1, ++gmm1Count);
        if (!gmm2LagActive) {
            RunGmm2CombineForWaveRange(waveRange, gmm2State, gmm2AddrInfo, combineBufferConfig, combineRowSequence);
            UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
        } else {
            if (hasPendingWave) {
                RunGmm2CombineForWaveRange(pendingWaveRange, gmm2State, gmm2AddrInfo, combineBufferConfig,
                                           combineRowSequence);
                UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
            }
            pendingWaveRange = waveRange;
            hasPendingWave = true;
        }
    }
    // 滞后流水收尾：三角色共同补跑最后一个 Wave 的 GMM2/Combine。
    if (hasPendingWave) {
        RunGmm2CombineForWaveRange(pendingWaveRange, gmm2State, gmm2AddrInfo, combineBufferConfig, combineRowSequence);
        UpdateGmmLoopCount(gmmLoopCount_, LoopCountIndex::GMM2, ++gmm2Count);
    }
}

template <TemplateMegaFusedMoeA8W4WaveTypeClass>
__aicore__ inline void MegaFusedMoeA8W4Wave<TemplateMegaFusedMoeA8W4WaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

} // namespace MegaFusedMoeImpl

#undef TemplateMegaFusedMoeA8W4WaveTypeClass
#undef TemplateMegaFusedMoeA8W4WaveTypeFunc

#endif
