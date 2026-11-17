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
 * Description: Implement the device-side AFD MegaMoE execution pipeline.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

/*!
 * \file megamoe_dispatch_send_arch35.h
 * \brief
 */

#ifndef MEGAMOE_DISPATCH_SEND_ARCH35_H
#define MEGAMOE_DISPATCH_SEND_ARCH35_H

#include <type_traits>
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
#include "common/megamoe_dispatch_send_types.h"
#include "common/megamoe_dispatch_send_workspace.h"
#include "common/megamoe_dispatch_send_utils.h"
#include "common/megamoe_dispatch_send_split_protocol.h"
#include "stage/megamoe_dispatch_send_token_quant.h"
#include "stage/megamoe_dispatch_send_send_mask.h"
#include "stage/megamoe_dispatch_send_shared_expert_input.h"
#if __has_include("../../common/quantize_functions.h")
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/quantize_functions.h"
// ### PATCH END: AFD merged source layout
#else
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/quantize_functions.h"
// ### PATCH END: AFD merged source layout
#endif

namespace MegamoeDispatchSendImpl {
using namespace AscendC;
#define TemplateMegamoeDispatchSendTypeClass     typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode,     int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegamoeDispatchSendTypeFunc     XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, IsGmm1Interleaved

template <TemplateMegamoeDispatchSendTypeClass>
class MegamoeDispatchSend {
public:
    using QuantOutType = typename std::conditional<QuantMode == E5M2_QUANT, fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using ActivationType = QuantOutType;
    using QuantScaleOutType = fp8_e8m0_t;
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR activations, GM_ADDR topkIds,
                                GM_ADDR topkWeights, MegamoeDispatchSendTilingData *tilingData)
    {
        k_ = tilingData->h;
        worldSize_ = tilingData->epWorldSize;
        moeExpertPerRank_ = tilingData->moeExpertPerRank;
        sharedExpertNum_ = tilingData->sharedExpertNum;
        auto *communication = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
        rankId_ = communication->epRankId;
        tilingData->splitRank = rankId_;
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            g_winRankAddr_[peer] = reinterpret_cast<GM_ADDR>(communication->epHcclBuffer_[peer]) + EXCEPTION_DUMP_REGION_SIZE;
        }
        params_.aGmAddr = activations;
        params_.expertIdxGmAddr = topkIds;
        params_.probsGmAddr = topkWeights;
        params_.tilingData = tilingData;
        params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, 1U);
        commonConfig_ = {.rankId = rankId_, .worldSize = worldSize_, .moeExpertPerRank = moeExpertPerRank_,
                         .sharedExpertNum = sharedExpertNum_, .tokenNum = tilingData->bs,
                         .topK = tilingData->topK, .tokenHiddenDim = k_, .gmm1OutputDim = tilingData->hiddenDim};
        aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
        quantProcessConfig_ = CreateQuantProcessConfig<ActivationType, QuantScaleOutType, TopkWeightsPrefetch, 1>(k_, params_);
        sendMaskConfig_ = CreateSendMaskConfig(params_, aivCoreIdx_);
        sendMaskConfig_.splitAttentionRanks = tilingData->splitAttentionRanks;
    }
    __aicore__ inline void ProcessSplit(GM_ADDR snapshotData, GM_ADDR snapshotScale);
private:
    using SendMaskBufferConfig = MegamoeDispatchSendSendMaskBufferConfig;
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline uint32_t InitQuantScratchTensors(uint32_t mxTempTensorAddr);
    Params params_{};
    MoeStageCommonConfig commonConfig_{};
    AivJobContext aivJob_{};
    SendMaskConfig sendMaskConfig_{};
    QuantProcessConfig quantProcessConfig_{};
    QuantProcessScratch<ActivationType> quantScratch_{};
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_{};
    SendMaskScratch sendMaskScratch_{};
    LocalTensor<int32_t> resetTensor_;
    int32_t resetBatchElementCount_ = 0;
    uint32_t rankId_ = 0, worldSize_ = 0, k_ = 0, sharedExpertNum_ = 0, moeExpertPerRank_ = 0;
    uint32_t blockAivNum_ = GetBlockNum() * 2, aivCoreIdx_ = GetBlockIdx();
    static constexpr uint32_t A_ELEMS_PER_BYTE = 1U;
};

template <TemplateMegamoeDispatchSendTypeClass>
__aicore__ inline uint32_t MegamoeDispatchSend<TemplateMegamoeDispatchSendTypeFunc>::InitQuantScratchTensors(uint32_t mxTempTensorAddr)
{
    uint32_t mxTempTensorSize = 2 * 1024;
    // 单个 xOutTensor 槽位与 dispatch 的 token-scale-weight 通信记录使用相同布局。
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));
    if (sharedExpertNum_ > 0U) {
        sharedExpertPrepareScratch_.copyBuffer0 = quantScratch_.xOutTensor0;
        sharedExpertPrepareScratch_.copyBuffer1 = quantScratch_.xOutTensor1;
    }

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t routeRingAddr = xInAlignAddr2 + xInAlignSize;
    // clang-format off
    /*
     * h%64==32（scale 组数为奇数）时，量化链路存在三处"计算不覆盖、却进入定长通信记录或参与
     * 计算"的跨 launch UB 残留：xIn 尾部（进 ComputeMaxExp 尾块 mask 内 lane）、xOut 记录的
     * scale 偶数补齐槽（ComputeScale 掩码写不到）、mxTemp 的 halfScale 补偶槽（被
     * ComputeFp8Data 尾块 E2B 广播进乘法，0×NaN 仍为 NaN）。残留呈 NaN/大指数位型时整行
     * GMM 输出被污染为 NaN，最终 combine 输出成块清零（首轮 UB 干净故仅多轮调用时显形）。
     * 此处对 [mxTempTensorAddr, routeRingAddr) 连续 span（mxTemp/xOut0/xOut1/xIn0/xIn1 五段
     * 量化 scratch）一次性清零：span 边界取 routeRingAddr、与本函数的地址推进公式同源，
     * 中间插入新 buffer 时范围自动跟随；有效区随后每 token 均被完整覆写，残留位恒为良性 0。
     * h%64==0 时不存在上述缝隙，本清零不改变任何可观测行为。
     */
    // clang-format on
    LocalTensor<int16_t> quantScratchSpan(TPosition::VECCALC, mxTempTensorAddr,
                                          (routeRingAddr - mxTempTensorAddr) / sizeof(int16_t));
    Duplicate<int16_t>(quantScratchSpan, 0, static_cast<int32_t>((routeRingAddr - mxTempTensorAddr) / sizeof(int16_t)));
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    return routeRingAddr;
}

template <TemplateMegamoeDispatchSendTypeClass>
__aicore__ inline void MegamoeDispatchSend<TemplateMegamoeDispatchSendTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(params_.workspaceInfo.flagResetElementCount);
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount = static_cast<uint64_t>(params_.workspaceInfo.gmm1TileStatusElementCount);
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                         static_cast<int32_t>(resetElementCountPerCore) :
                                         DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(params_.tilingData->splitMoeRanks * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。compact route 按连续专家段
    // 分核，因此前 remainder 个 core 多处理一个 expert。
    const SendMaskBufferConfig &bufferConfig = sendMaskConfig_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t topkIndexTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    sendMaskScratch_.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIndexTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    if (resetTensorSize > 0U) { Duplicate<int32_t>(resetTensor_, 0, resetTensorSize / sizeof(int32_t)); }
    resetBatchElementCount_ = resetBatchElementCount;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    uint32_t routeRingAddr = InitQuantScratchTensors(mxTempTensorAddr);
    uint32_t routeRingBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.routeRingTensor = LocalTensor<uint8_t>(TPosition::VECCALC, routeRingAddr, routeRingBytes);
    uint32_t sendCntAccAddr = routeRingAddr + routeRingBytes;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
}

template <TemplateMegamoeDispatchSendTypeClass>
__aicore__ inline void MegamoeDispatchSend<TemplateMegamoeDispatchSendTypeFunc>::ProcessSplit(GM_ADDR snapshotData, GM_ADDR snapshotScale)
{
    // Preserve the original MegaMoE entry contract on every core, including
    // cores which never execute an activation epilogue that sets CTRL[60].
    int64_t originalOverflow = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    if constexpr (g_coreType == AIV) {
        const auto *td = params_.tilingData;
        for (uint32_t owner = td->splitAttentionRanks; owner < worldSize_; ++owner) {
            SplitWait(SplitWord(params_.peermemInfo.rankSyncInWorldPtr, owner, SPLIT_TAKEN), td->splitSequence, true);
        }
        params_.workspaceInfo.sharedExpertInputDataPtr = snapshotData;
        params_.workspaceInfo.sharedExpertInputScalePtr = snapshotScale;
        SendAndQuantBuffInit();
        QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
            aivJob_, commonConfig_, params_, quantProcessConfig_, quantScratch_);
        PipeBarrier<PIPE_ALL>();
        // All quant writers finish before ANY route / count is published remotely.
        SyncAll<true>();
        if (sharedExpertNum_ > 0U) {
            PrepareSharedExpertInput<ActivationType, QuantScaleOutType, A_ELEMS_PER_BYTE>(
                aivJob_, commonConfig_, params_, quantProcessConfig_, sharedExpertPrepareScratch_);
        }
        GatherAndSendExpertCompactRoutes(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_, sendMaskScratch_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        if (aivCoreIdx_ == 0U) {
            for (uint32_t owner = td->splitAttentionRanks; owner < worldSize_; ++owner) {
                SplitPublishRouteTask(params_, g_winRankAddr_, owner);
                SplitPublish(params_, g_winRankAddr_, owner, SPLIT_PUBLISHED);
            }
        }
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(originalOverflow);
}
}
#undef TemplateMegamoeDispatchSendTypeClass
#undef TemplateMegamoeDispatchSendTypeFunc
#endif
