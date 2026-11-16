/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file megamoe_fused_moe_arch35.h
 * \brief
 */

#ifndef MEGAMOE_FUSED_MOE_ARCH35_H
#define MEGAMOE_FUSED_MOE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/mc2_kernel_utils.h"
// ### PATCH END: AFD merged source layout
#else
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/mc2_kernel_utils.h"
// ### PATCH END: AFD merged source layout
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "common/megamoe_fused_moe_types.h"
#include "common/megamoe_fused_moe_workspace.h"
#include "common/megamoe_fused_moe_utils.h"
#include "common/megamoe_fused_moe_exception_dump_policy.h"
// ### PATCH START: AFD merged source layout
#include "blaze/epilogue/cam_megamoe_fused_moe_block_epilogue_activation_mx_quant.h"
// ### PATCH END: AFD merged source layout
#include "common/megamoe_fused_moe_split_protocol.h"
#include "stage/megamoe_fused_moe_token_quant.h"
#include "stage/megamoe_fused_moe_workspace_reset.h"
#include "stage/megamoe_fused_moe_token_dispatch.h"
#include "stage/megamoe_fused_moe_gmm1_activation.h"
#include "stage/megamoe_fused_moe_gmm2_combine.h"
#include "stage/megamoe_fused_moe_unpermute.h"
#if __has_include("../../common/quantize_functions.h")
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/quantize_functions.h"
// ### PATCH END: AFD merged source layout
#else
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/quantize_functions.h"
// ### PATCH END: AFD merged source layout
#endif

namespace MegamoeFusedMoeImpl {

using namespace AscendC;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegamoeFusedMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegamoeFusedMoeTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved

template <TemplateMegamoeFusedMoeTypeClass>
class MegamoeFusedMoe {
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
    __aicore__ inline MegamoeFusedMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR expertTokenNumsOut,
                                GM_ADDR workspaceGM, MegamoeFusedMoeTilingData *tilingData, GM_ADDR tilingGM);

protected:
    __aicore__ inline void ResetSplitWorkspace()
    {
        if constexpr (g_coreType == AIV) {
            LocalTensor<int32_t> zeros(TPosition::VECCALC, 0U, DISPATCH_RESET_BATCH);
            Duplicate<int32_t>(zeros, 0, DISPATCH_RESET_BATCH);
            ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, DISPATCH_RESET_BATCH, zeros);
            if (GetSubBlockIdx() == 1U) {
                auto *progress = reinterpret_cast<__gm__ int32_t *>(params_.peermemInfo.rankSyncInWorldPtr + 48U * 1024U);
                WriteGmByPassDCache(progress + blockIdx_ * 16U + 1U, 0);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void MaybePublishDispatchTaken(bool finished)
    {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U && finished && !dispatchTaken_) {
                PipeBarrier<PIPE_ALL>();
                SplitDispatchTaken(params_, g_winRankAddr_, blockIdx_, blockNum_);
                dispatchTaken_ = true;
            }
        }
    }
    __aicore__ inline void PublishSourceCombineProgress(const ExpertTokenPosition &position)
    {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1U) {
                uint32_t bytes = Ops::Base::CeilAlign(worldSize_ * moeExpertPerRank_ * uint32_t(sizeof(int32_t)), uint32_t(ALIGN_32));
                auto *prefix = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + uint64_t(bytes) * blockIdx_);
                uint32_t completed = static_cast<uint32_t>(position.globalTokenIndex);
                if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
                    // A8W8 quant combine can defer an incomplete expert; conservatively
                    // report only complete experts for both variants.
                    if (position.tokenIndexInExpert != 0U) {
                        completed = position.expertIdx == 0U ? 0U : uint32_t(ReadGmByPassDCache(prefix + position.expertIdx * worldSize_ - 1U));
                    }
                }
                PipeBarrier<PIPE_ALL>(); // Drain this writer's result stores before advancing its prefix.
                SplitCombineProgress(params_, g_winRankAddr_, blockIdx_, blockNum_, completed, prefix);
            }
        }
    }
    bool dispatchTaken_ = false;
private:
    using UnpermuteBufferConfig = MegamoeFusedMoeUnpermuteBufferConfig;

    __aicore__ inline void InitEpilogueAndCommonConfig(MegamoeFusedMoeTilingData *tilingData);
    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                    int32_t activationFlagSlotsPerExpert);
    __aicore__ inline void InitGmmConfigs();

protected:
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void RunGmm2CombineForExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                   uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
                                                   uint32_t sliceTokenCount,
                                                   WaveCombineBufferConfig &combineBufferConfig,
                                                   uint32_t &combineRowSequence, int32_t &gmmTileSequence,
                                                   bool isFinalCombine);
    template <bool WaitForTokenCountReady>
    __aicore__ inline void PrepareGmmExpertState(ExpertLoopState &state, uint32_t expertIdx);
    template <typename Derived>
    __aicore__ inline void ProcessWave(Derived &derived);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
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
    MegamoeFusedMoeImpl::ExceptionDumpEngine exceptionDump_;
    __gm__ MegamoeFusedMoeImpl::GmmLoopCount *gmmLoopCount_{nullptr};
};

template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ =
        CreateQuantProcessConfig<ActivationType, QuantScaleOutType, TopkWeightsPrefetch, A_ELEMS_PER_BYTE>(k_, params_);
}

template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                                                  int32_t activationFlagSlotsPerExpert)
{
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = activationFlagSlotsPerExpert,
                            .gmm1TileStatusCountPerExpert = params_.tilingData->maxTilesPerExpert};
}

template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::InitGmmConfigs()
{
    gmmExecutionConfig_ = {.blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
                           .groupedMatmulMode = params_.tilingData->groupedMatmulMode,
                           .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    waveCombineJob_ = {.jobIndex = blockIdx_, .totalJobs = blockNum_};
}



template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::InitEpilogueAndCommonConfig(MegamoeFusedMoeTilingData *tilingData)
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
template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::Init(GM_ADDR context, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR expertTokenNumsOut,
                                GM_ADDR workspaceGM, MegamoeFusedMoeTilingData *tilingData, GM_ADDR tilingGM)
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
    moeWeightTensorListAddrs_ = {
        .weight1 = weight1, .weightScales1 = weightScales1, .weight2 = weight2, .weightScales2 = weightScales2};
    moeWeightTensorListAddrs_.localExperts = moeExpertPerRank_;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
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

    gmmLoopCount_ =
        MegamoeFusedMoeImpl::RegisterMegamoeFusedMoeExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData, params_.peermemInfo,
                                                  reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。
template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::DispatchBuffInit()
{
    const TokenDispatchConfig &context = tokenDispatchConfig_;
    TokenDispatchScratch<ActivationType> &scratch = tokenDispatchScratch_;
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    const MegamoeFusedMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = commonConfig_.tokenHiddenDim / A_ELEMS_PER_BYTE;
    scratch.revScaleElemCnt = Ops::Base::CeilDiv(static_cast<int64_t>(commonConfig_.tokenHiddenDim),
                                                 static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                              MXFP_MULTI_BASE_SIZE;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.worldSize * commonConfig_.moeExpertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr +
            static_cast<uint64_t>(cumsumInfoTensorSize) * countWorkspace_.blockIdx));

    // 按既定顺序落地址。Tensor 保存所有 [expert][source rank] count 的前缀和。
    // Tensor 大小：worldSize_ * moeExpertPerRank_ * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t cumsumInfoTensorAddr = 0U;
    scratch.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // compact route 已消除 mask 扫描，只保留一份接收 index batch。
    // Tensor 用途：接收 compact topkIndex 的当前 batch。
    uint32_t validTopkIndexTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(bufferConfig.routeItemsPerBatch * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    // 路由批次 Tensor 后依次放置 copyTmp 环形缓冲区和 32B metaInfo 环形缓冲区。
    // Tensor 用途：DispatchExpertTokens 中的动态 dispatch 环形缓冲区，配合
    // EVENT_ID0..EVENT_ID(bufferCount-1) 形成软流水；
    // 只记基址：槽视图在热路径由 GetDispatchCopyBuffer 现场构造，
    // Tensor 大小：bufferConfig.bufferCount 块（主线自适应 UB 预算给出的 2~6），
    // 每块 tokenDispatchConfig_.quantTokenScaleAlignBytes；
    // 该值即 Init() 算好的 Align256(token) + Align32(scale) + optional Align32(weight)，与 host
    // CalcDispatchBufferConfig 的 copyBufferBytes 恒相等，故连续 ring 中每个槽位均保持 32B 对齐。
    scratch.copyTmpBaseAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * context.quantTokenScaleAlignBytes;
    uint32_t expertTokenNumsOutTensorAddr = scratch.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.moeExpertPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                            expertTokenNumsOutTensorSize / sizeof(int32_t));
    // Tensor 用途：CopyTokensAndMetaForDispatch 中的 metaInfo 环形缓冲区，逐 token 即时写入 GM；
    // Tensor 大小：bufferCount * 32B，与 copyTmp 槽位和事件编号一一对应。
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoTensorSize = static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoTensorSize / sizeof(int32_t));
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










// W4 基类统一执行原 A8W4/A4W4 的专家 GMM2/Combine；派生模板只提供当前专家 slice。
// 非量化路径由 AIV1 在 GMM2 scheduler 内逐 tile 调用统一的 CombineTokenRange。
template <TemplateMegamoeFusedMoeTypeClass>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::RunGmm2CombineForExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &startBlockIdx, uint32_t tokenStartIndexInExpert,
    uint32_t sliceTokenCount, WaveCombineBufferConfig &combineBufferConfig, uint32_t &combineRowSequence,
    int32_t &gmmTileSequence, bool isFinalCombine)
{
    uint32_t expertTokenCount = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    uint32_t gmm2NTileCount = Ops::Base::CeilDiv(commonConfig_.tokenHiddenDim, static_cast<uint32_t>(L1_TILE_N));
    uint32_t problemTileCount = GetMGroupCountForRows(sliceTokenCount, GMM1_TILE_M) * gmm2NTileCount;
    if (!HandleWaveProblemWithoutWork(problemTileCount, gmmExecutionConfig_.blockJob, startBlockIdx)) {
        UpdateMoeExpertGmm2GlobalBuffer<Weight1Type, ActivationQuantOutType, QuantScaleOutType>(
            gmmExecutionConfig_, syncWorkspaceLayout_, params_.workspaceInfo, moeWeightTensorListAddrs_, gmmAddrInfo,
            state, tokenStartIndexInExpert);
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            // 非量化 Combine 按 GMM2 tile 配对消费，需要为每个 AIC 配置独立完成标记。
            gmmAddrInfo.gmmToEpilogueFlag =
                reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                static_cast<uint64_t>(gmmExecutionConfig_.blockJob.jobIndex) * INT_CACHELINE;
        }
        ProblemShape sliceProblemShape = state.problemShape;
        Get<M_VALUE>(sliceProblemShape) = sliceTokenCount;
        int32_t *gmm2TileSequence = nullptr;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            gmm2TileSequence = &gmmTileSequence;
        }
        RunGmm2A8W4<ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                    TopkWeightsPrefetch, false, false, true, CombineQuantMode == COMBINE_NO_QUANT>(
            sliceProblemShape, gmmAddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob, expertTokenCount,
            tokenStartIndexInExpert, &params_, gmm2TileSequence);
    }

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        // AIV1 已在上述 GMM2 调用中消费当前 slice；无需在外部重建 scheduler。
        return;
    }

    // 量化 Combine 保持原 W4 语义：跨 WAVE 的专家必须等最后一个 slice 完成后再消费。
    if (tokenStartIndexInExpert + sliceTokenCount < expertTokenCount) {
        return;
    }

    NotifyWaveGmm2Ready(waveCombineJob_, params_, state.expertIdx);
    UpdateMoeExpertCombineGlobalBuffer(params_.workspaceInfo, gmmAddrInfo, state);
    if (isFinalCombine) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 0U) {
                // W4 prologue 完成后，末轮 AIV0 才能通过 MTE2 复用同一 UB 区域。
                SyncFuncStatic<HardEvent::MTE3_MTE2, SYNC_EVENT_ID0>();
            }
        }
        combineBufferConfig =
            PrepareFinalWaveCombineBuffers<CombineQuantMode>(commonConfig_, combineBufferConfig, waveCombineScratch_);
        RunWaveCombineStage<CombineQuantMode, true>(commonConfig_, waveCombineJob_, combineBufferConfig,
                                                    waveCombineScratch_, params_, gmmAddrInfo, state, state.expertIdx,
                                                    combineRowSequence);
        return;
    }
    RunWaveCombineStage<CombineQuantMode>(commonConfig_, waveCombineJob_, combineBufferConfig, waveCombineScratch_,
                                          params_, gmmAddrInfo, state, state.expertIdx, combineRowSequence);
}

// 从 workspace 读取 token 数并准备 GMM 专家状态；GMM1 可按需先等待 token count ready。
template <TemplateMegamoeFusedMoeTypeClass>
template <bool WaitForTokenCountReady>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::PrepareGmmExpertState(ExpertLoopState &state,
                                                                               uint32_t expertIdx)
{
    if constexpr (WaitForTokenCountReady) {
        if (GetSubBlockIdx() == 0U && !state.expertCountTableReady) {
            moeWeightTensorListAddrs_.layerIndex = WaitForMoeExpertTokenCountReady(
                params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_, 0U);
            state.expertCountTableReady = true;
        }
    }
    uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
        params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, commonConfig_.moeExpertPerRank, expertIdx);
    UpdateExpertLoopState(state, expertIdx, expertTokenCount);
}



/*
 * 所有 MTE Wave 路径共用相同的阶段边界，派生类只负责 MoE 专家 Wave 的具体编排。
 */
template <TemplateMegamoeFusedMoeTypeClass>
template <typename Derived>
__aicore__ inline void MegamoeFusedMoe<TemplateMegamoeFusedMoeTypeFunc>::ProcessWave(Derived &derived)
{
    int64_t originalOverflow = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    if constexpr (g_coreType == AIV) {
        // This is a local workspace reset, never a reset of the registered control window.
        ResetSplitWorkspace();
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();
    if constexpr (g_coreType == AIV) {
        for (uint32_t member = 0; member < params_.tilingData->splitDpSize; ++member) {
            SplitWait(SplitWord(params_.peermemInfo.rankSyncInWorldPtr,
                              params_.tilingData->splitDpRanks[member], SPLIT_PUBLISHED),
                      params_.tilingData->splitSequence);
        }
    }
    derived.ProcessMoeExpertStages();
    MaybePublishDispatchTaken(true); // Includes the all-empty / trailing-empty expert case.
    if constexpr (g_coreType == AIV) {
        ExportCompactExpertTokenCounts(commonConfig_, countWorkspace_, params_, tokenDispatchScratch_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        if (aivCoreIdx_ == 0U) {
            for (uint32_t member = 0; member < params_.tilingData->splitDpSize; ++member) {
                SplitPublish(params_, g_winRankAddr_, params_.tilingData->splitDpRanks[member], SPLIT_READY);
            }
        }
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(originalOverflow);
}



} // namespace MegamoeFusedMoeImpl
#undef TemplateMegamoeFusedMoeTypeClass
#undef TemplateMegamoeFusedMoeTypeFunc
#endif
