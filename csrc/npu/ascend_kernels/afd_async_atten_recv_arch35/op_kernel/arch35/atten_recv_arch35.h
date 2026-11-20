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
 * \file atten_recv_arch35.h
 * \brief
 */

#ifndef MEGA_COMBINE_RECV_ARCH35_H
#define MEGA_COMBINE_RECV_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
// ### PATCH START: AFD merged source layout
#include "mc2_kernel_utils.h"
// ### PATCH END: AFD merged source layout

#include "kernel_operator_list_tensor_intf.h"
#include "common/atten_recv_types.h"
#include "common/atten_recv_workspace.h"
#include "common/atten_recv_utils.h"
#include "common/atten_recv_exception_dump_policy.h"
// ### PATCH START: AFD merged source layout
#include "blaze/epilogue/afd_async_atten_recv_block_epilogue_activation_mx_quant.h"
// ### PATCH END: AFD merged source layout
#include "common/atten_recv_split_protocol.h"
#include "stage/atten_recv_token_quant.h"
#include "stage/atten_recv_workspace_reset.h"
#include "stage/atten_recv_token_dispatch.h"
#include "stage/atten_recv_gmm1_activation.h"
#include "stage/atten_recv_gmm2_combine.h"
#include "stage/atten_recv_unpermute.h"
// ### PATCH START: AFD merged source layout
#include "quantize_functions.h"
// ### PATCH END: AFD merged source layout


namespace MegaCombineRecvImpl {

using namespace AscendC;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaCombineRecvTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaCombineRecvTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved

template <TemplateMegaCombineRecvTypeClass>
class MegaCombineRecv {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaCombineRecv(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR topkIds, GM_ADDR topkWeights,
                                GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR workspaceGM, MegaCombineRecvTilingData *tilingData, GM_ADDR tilingGM);

    __aicore__ inline void ProcessSplit(GM_ADDR snapshotData = nullptr, GM_ADDR snapshotScale = nullptr);
protected:
    __aicore__ inline void ResetSplitWorkspace()
    {
        if constexpr (g_coreType == AIV) {
            LocalTensor<int32_t> zeros(TPosition::VECCALC, 0U, DISPATCH_RESET_BATCH);
            Duplicate<int32_t>(zeros, 0, DISPATCH_RESET_BATCH);
            ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, DISPATCH_RESET_BATCH, zeros);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    using UnpermuteBufferConfig = MegaCombineRecvUnpermuteBufferConfig;

    __aicore__ inline void InitEpilogueAndCommonConfig(MegaCombineRecvTilingData *tilingData);
    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                    int32_t activationFlagSlotsPerExpert);
    __aicore__ inline void InitGmmConfigs();
    __aicore__ inline void InitTokenUnpermuteConfig();

protected:
    __aicore__ inline void InitQuantTokenBufferConfig();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void RunSharedExpertGmm1Activation(const GMMAddrInfo &gmmAddrInfo,
                                                         const ProblemShape &problemShape,
                                                         const GmmExecutionConfig &gmmConfig,
                                                         GmmRuntimeState &runtimeState, uint32_t sharedExpertIdx);
    __aicore__ inline void RunSharedExpertGmm2(const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape,
                                               uint32_t &startBlockIdx);
    __aicore__ inline void ProcessSharedExpertGmm1();
    __aicore__ inline void ProcessSharedExpertGmm2();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    MoeStageCommonConfig commonConfig_{};
    GmmExecutionConfig gmmExecutionConfig_{};
    BlockWorkspaceContext countWorkspace_{};
    MoeSyncWorkspaceLayout syncWorkspaceLayout_{};
    // 单次 reset 批量元素数（与 syncWorkspaceLayout_ 描述的清零区域配套）。
    int32_t resetBatchElementCount_ = 0;
    // 输入准备各 stage（quant/mask/reset/shared-prepare/unpermute）共用的逐 AIV 任务分工。
    AivJobContext aivJob_{};
    TokenDispatchConfig tokenDispatchConfig_;
    QuantProcessConfig quantProcessConfig_;
    QuantTokenBufferConfig quantTokenBufferConfig_;
    // Wave Combine 的逻辑任务分工（block 粒度，AIV1 门控在函数内）。
    AivJobContext waveCombineJob_{};
    TokenUnpermuteConfig tokenUnpermuteConfig_;

    uint32_t k_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint16_t gmm1PingPongIdx_ = 0;
    // 同一 AIC/AIV 对复用 gmmToEpilogueFlag；序号在单次 launch 的相关 GMM 阶段间保持单调递增。
    int32_t gmmTileSequence_ = 0;
    // MoE GMM1/GMM2 共用分核游标；A8W8 的共享专家 GMM2 继续沿用该游标。
    uint32_t startBlockIdx_ = 0;
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    uint32_t mGroupsPerWave_ = 1U;

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4: FP8 activation + FP4 weight，GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr bool ENABLE_A8W8 = (QuantMode == E4M3_QUANT && Std::IsSame<Weight1Type, fp8_e4m3fn_t>::value) ||
                                        (QuantMode == E5M2_QUANT && Std::IsSame<Weight1Type, fp8_e5m2_t>::value);
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;
    QuantProcessScratch<ActivationType> quantScratch_;
    LocalTensor<int32_t> resetTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，ActivationQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using ActivationQuantOutType =
        typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // ActivationQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE;

    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, EPILOGUE_TILE_M, L1_TILE_N,
                                                         TopkWeightsPrefetch, IsGmm1Interleaved>;
    using SharedBlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_M_256,
                                                               L1_TILE_N, false, IsGmm1Interleaved>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    WaveCombineScratch waveCombineScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
    MegaCombineRecvImpl::ExceptionDumpEngine exceptionDump_;
    __gm__ MegaCombineRecvImpl::GmmLoopCount *gmmLoopCount_{nullptr};
};

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ =
        CreateQuantProcessConfig<ActivationType, QuantScaleOutType, TopkWeightsPrefetch, A_ELEMS_PER_BYTE>(k_, params_);
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                                                  int32_t activationFlagSlotsPerExpert)
{
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = activationFlagSlotsPerExpert,
                            .gmm1TileStatusCountPerExpert = params_.tilingData->maxTilesPerExpert};
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitGmmConfigs()
{
    gmmExecutionConfig_ = {.blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
                           .groupedMatmulMode = params_.tilingData->groupedMatmulMode,
                           .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    waveCombineJob_ = {.jobIndex = blockIdx_, .totalJobs = blockNum_};
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitTokenUnpermuteConfig()
{
    tokenUnpermuteConfig_ = {.job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
                             .quantTokenSizeBytes = quantTokenBufferConfig_.quantTokenSizeBytes,
                             .fullTokenChunkJobCount = params_.tilingData->unpermuteFullTokenChunkCoreCount,
                             .fullTokenChunkConfig = params_.tilingData->unpermuteConfigForFullTokenChunk,
                             .tailTokenChunkConfig = params_.tilingData->unpermuteConfigForTailTokenChunk};
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitEpilogueAndCommonConfig(MegaCombineRecvTilingData *tilingData)
{
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    commonConfig_ = {.rankId = rankId_,
                     .worldSize = worldSize_,
                     .moeExpertPerRank = moeExpertPerRank_,
                     .sharedExpertNum = sharedExpertNum_,
                     .tokenNum = tilingData->bs,
                     .topK = tilingData->topK,
                     .tokenHiddenDim = k_,
                     .gmm1OutputDim = tilingData->hiddenDim};
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::Init(GM_ADDR context, GM_ADDR topkIds, GM_ADDR topkWeights,
                                GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR workspaceGM, MegaCombineRecvTilingData *tilingData, GM_ADDR tilingGM)
{
    k_ = tilingData->h;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    mGroupsPerWave_ = tilingData->mGroupsPerWave;
    gmm1PingPongIdx_ = 0;
    gmmTileSequence_ = 0;
    startBlockIdx_ = 0;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    tilingData->splitRank = rankId_;
    GM_ADDR dumpBase = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[rankId_]);
    for (int i = 0; i < worldSize_; i++) {
        // g_winRankAddr_从win区地址偏移60K开始用，前面60K是异常dump区
        g_winRankAddr_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]) + EXCEPTION_DUMP_REGION_SIZE;
    }
    params_.expertIdxGmAddr = topkIds;
    if (sharedExpertNum_ > 0U) {
        sharedWeightTensorListAddrs_ = {.weight1 = sharedWeight1,
                                        .weightScales1 = sharedWeightScales1,
                                        .weight2 = sharedWeight2,
                                        .weightScales2 = sharedWeightScales2};
    }
    params_.y2GmAddr = yOut;
    params_.probsGmAddr = topkWeights;
    {
        WorkspaceLayout workspaceLayout(tilingData);
        params_.workspaceInfo.Bind(workspaceGM, workspaceLayout);
    }
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    InitEpilogueAndCommonConfig(tilingData);
    const int64_t maxOutput = static_cast<int64_t>(tilingData->maxOutputSize);
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert = static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    int32_t activationFlagSlotsPerExpert =
        static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, static_cast<int64_t>(L1_TILE_M_256))) * INT_CACHELINE;
    InitInputPrepareConfigs();
    tokenDispatchConfig_ = CreateTokenDispatchConfig(params_, quantProcessConfig_);
    InitSyncWorkspaceConfigs(dispatchFlagSlotsPerExpert, activationFlagSlotsPerExpert);
    InitGmmConfigs();
    InitQuantTokenBufferConfig();
    InitTokenUnpermuteConfig();

    gmmLoopCount_ =
        MegaCombineRecvImpl::RegisterAttenRecvExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData, params_.peermemInfo,
                                                  reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。


template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitQuantTokenBufferConfig()
{
    quantTokenBufferConfig_ = {.quantTokenSizeBytes = 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantTokenBufferConfig_ = CreateQuantTokenBufferConfig(k_);
    }
}

// ======================================================================================
// InitQuantScratchTensors：量化 scratch（mxTemp、xOut 双 buffer、xIn 双 buffer）的地址排布；
//   shared prepare 复用 xOut 双 buffer；返回量化区之后的空闲地址。
// ======================================================================================


// ======================================================================================
// SendAndQuantBuffInit：单核 mask/reset/quant/shared-prepare 模块使用的 buffer 申请。
//   shared prepare 复用 quant 输出双 buffer；reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================


// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline typename MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::UnpermuteBufferConfig
MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::InitTokenUnpermuteBuffers()
{
    return CreateTokenUnpermuteBuffers<TopkWeightsType, CombineQuantMode>(
        tokenUnpermuteConfig_, commonConfig_.tokenHiddenDim, tokenUnpermuteScratch_);
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::RunSharedExpertGmm1Activation(
    const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape, const GmmExecutionConfig &gmmConfig,
    GmmRuntimeState &runtimeState, uint32_t sharedExpertIdx)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * commonConfig_.tokenNum;
    if constexpr (ENABLE_A8W4) {
        RunGmm1A8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    L1_TILE_M_256, false, true, true>(sharedEpilogueOp_, params_, problemShape, gmmAddrInfo,
                                                      runtimeState.startBlockIdx, gmmTileSequence_, gmmConfig.blockJob,
                                                      expertBeforeCnt, sharedExpertIdx);
    } else {
        RunGmm1GenericByWeightFormat<QuantOutType, ActivationQuantOutType, QuantScaleOutType, GMM1_TILE_M,
                                     L1_TILE_M_256, false, IsGmm1Interleaved, true, true>(
            gmmConfig, params_, sharedEpilogueOp_, gmmAddrInfo, problemShape, expertBeforeCnt, runtimeState,
            sharedExpertIdx, nullptr, ENABLE_A8W8);
    }
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::ProcessSharedExpertGmm1()
{
    if (gmmExecutionConfig_.blockJob.totalJobs == 0U ||
        gmmExecutionConfig_.blockJob.jobIndex >= gmmExecutionConfig_.blockJob.totalJobs) {
        return;
    }
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
        .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
        .clampLimit = params_.tilingData->clampLimit,
        .actMode = params_.tilingData->actMode,
        .actSubMode = params_.tilingData->actSubMode,
        .activationAlpha = params_.tilingData->activationAlpha,
        .activationBeta = params_.tilingData->activationBeta};
    sharedEpilogueOp_.Init(epilogueParams);

    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t pingPongIdx = 0U;
    GmmRuntimeState runtimeState{startBlockIdx, vecSetSyncCom, pingPongIdx};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationQuantOutType, QuantScaleOutType,
                                           ENABLE_A8W4>(commonConfig_, gmmExecutionConfig_, params_.workspaceInfo,
                                                        sharedWeightTensorListAddrs_, sharedEpilogueOp_, gmmAddrInfo,
                                                        sharedExpertIdx);
        RunSharedExpertGmm1Activation(gmmAddrInfo, problemShape, gmmExecutionConfig_, runtimeState, sharedExpertIdx);
    }
    EndSync<IsGmm1Interleaved>(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::RunSharedExpertGmm2(const GMMAddrInfo &gmmAddrInfo,
                                                                             const ProblemShape &problemShape,
                                                                             uint32_t &startBlockIdx)
{
    // 共享 GMM1 和 GMM2 的所有量化模式都按 256-token group 交接 activation；
    // TopK weight prefetch 仍关闭，权重 L2 bypass 仍仅用于 A8W8。
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    false, true, false, true>(problemShape, gmmAddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob,
                                              static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U);
    } else {
        RunGmm2GenericByWeightFormat<COMBINE_NO_QUANT, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                     QuantScaleOutType, false, GMM1_TILE_M, false, true, IsGmm1Interleaved, true>(
            gmmExecutionConfig_, problemShape, gmmAddrInfo, startBlockIdx, nullptr, ENABLE_A8W8);
    }
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::ProcessSharedExpertGmm2()
{
    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    // 保持主线分核语义：A8W8 延续 MoE 游标，W4 的共享 GMM2 从局部起点 0 开始。
    // 待共享激活输入的同步修复合入后，再统一为连续滚动；起点归零本身不提供同步保证。
    uint32_t w4SharedGmm2StartBlockIdx = 0U;
    uint32_t &sharedGmm2StartBlockIdx = ENABLE_A8W8 ? startBlockIdx_ : w4SharedGmm2StartBlockIdx;
    GMMAddrInfo gmmAddrInfo{};
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm2GlobalBuffer<ActivationQuantOutType, Weight1Type, QuantScaleOutType, GMM1_TILE_M>(
            commonConfig_, gmmExecutionConfig_, params_.workspaceInfo, sharedWeightTensorListAddrs_, gmmAddrInfo,
            sharedExpertIdx);
        RunSharedExpertGmm2(gmmAddrInfo, problemShape, sharedGmm2StartBlockIdx);
    }
}

template <TemplateMegaCombineRecvTypeClass>
__aicore__ inline void MegaCombineRecv<TemplateMegaCombineRecvTypeFunc>::ProcessSplit(GM_ADDR snapshotData, GM_ADDR snapshotScale)
{
    // Unpermute also runs on AIV1 and idle shared-GMM1 AIV0 cores. They must
    // use the same per-instruction saturation mode as the active epilogue cores.
    int64_t originalOverflow = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    params_.workspaceInfo.sharedExpertInputDataPtr = snapshotData;
    params_.workspaceInfo.sharedExpertInputScalePtr = snapshotScale;
    if constexpr (g_coreType == AIV) {
        ResetSplitWorkspace();
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();
    if (sharedExpertNum_ > 0U && params_.tilingData->bs > 0U) {
        ProcessSharedExpertGmm1();
        ProcessSharedExpertGmm2();
    }
    if constexpr (g_coreType == AIV) {
        const auto *td = params_.tilingData;
        if (aivCoreIdx_ == 0U) {
            auto *ids = reinterpret_cast<__gm__ int32_t *>(params_.expertIdxGmAddr);
            for (uint32_t owner = 0; owner < td->splitMoeRanks; ++owner) {
                bool required = false;
                for (uint32_t item = 0; item < td->bs * td->topK; ++item) {
                    required = required || uint32_t(ReadGmByPassDCache(ids + item)) / moeExpertPerRank_ == owner;
                }
                if (required) {
                    SplitWait(SplitWord(params_.peermemInfo.rankSyncInWorldPtr,
                                        td->splitAttentionRanks + owner, SPLIT_READY), td->splitSequence);
                }
            }
        }
        SyncAll<true>();
        if (td->bs > 0U) {
            auto buffers = InitTokenUnpermuteBuffers();
            UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
                tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, buffers);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        if (aivCoreIdx_ == 0U) {
            for (uint32_t owner = td->splitAttentionRanks; owner < worldSize_; ++owner) {
                SplitPublish(params_, g_winRankAddr_, owner, SPLIT_CONSUMED);
            }
        }
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(originalOverflow);
}
} // namespace MegaCombineRecvImpl
#undef TemplateMegaCombineRecvTypeClass
#undef TemplateMegaCombineRecvTypeFunc
#endif
