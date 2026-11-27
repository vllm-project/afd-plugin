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
 * Description: Implement the token quant stage of the AFD Mega pipeline.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#ifndef MEGA_COMBINE_RECV_TOKEN_QUANT_H
#define MEGA_COMBINE_RECV_TOKEN_QUANT_H

#include "../common/atten_recv_mxfp8_utils.h"
#include "../common/atten_recv_utils.h"

namespace MegaCombineRecvImpl {

using namespace AscendC;

struct QuantProcessConfig {
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t quantScaleNumAlignPerToken;
};

/*
 * 计算量化通信记录的对齐布局（普通与 wave 编排模板共用）。
 * 记录布局 = Align256(token 数据) + Align32(scale) + prefetch 时附加 Align32(topk 权重)，
 * 与 host CalcDispatchBufferConfig 的 copyBufferBytes 契约恒相等。
 */
template <typename ActivationType, typename QuantScaleOutType, bool TopkWeightsPrefetch, uint32_t AElemsPerByte>
__aicore__ inline QuantProcessConfig CreateQuantProcessConfig(uint32_t tokenHiddenDim, const Params &params)
{
    uint32_t quantScaleNumAlignPerToken = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenAlignBytes =
        Ops::Base::CeilAlign(tokenHiddenDim / AElemsPerByte, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    uint32_t quantScaleAlignBytes = Ops::Base::CeilAlign(
        quantScaleNumAlignPerToken * static_cast<uint32_t>(sizeof(QuantScaleOutType)), static_cast<uint32_t>(ALIGN_32));
    uint32_t quantTokenScaleAlignBytes = quantTokenAlignBytes + quantScaleAlignBytes;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK * sizeof(float)),
                                                         static_cast<uint32_t>(ALIGN_32));
        quantTokenScaleAlignBytes += weightAlignBytes;
    }
    return {quantTokenAlignBytes, quantScaleAlignBytes, quantTokenScaleAlignBytes, quantScaleNumAlignPerToken};
}

template <typename ActivationType>
struct QuantProcessScratch {
    LocalTensor<bfloat16_t> xInTensor0;
    LocalTensor<bfloat16_t> xInTensor1;
    LocalTensor<ActivationType> xOutTensor0;
    LocalTensor<ActivationType> xOutTensor1;
    LocalTensor<uint16_t> mxTempTensor;
};

template <typename TopkWeightsType, typename ActivationType, bool TopkWeightsPrefetch>
__aicore__ inline void LoadTopkWeightsToUb(const Params &params, const QuantProcessConfig &config,
                                           QuantProcessScratch<ActivationType> &scratch,
                                           const LocalTensor<ActivationType> &xOutTensor, int32_t tokenIndex,
                                           TEventID event)
{
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<TopkWeightsType> weightGm;
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkWeightsType *>(
            params.probsGmAddr +
            static_cast<uint64_t>(tokenIndex) * params.tilingData->topK * sizeof(TopkWeightsType)));
        uint32_t weightOffsetInUb = config.quantTokenAlignBytes + config.quantScaleAlignBytes;
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            LocalTensor<TopkWeightsType> weightBf16Tmp =
                scratch.mxTempTensor.template ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightBf16Tmp, weightGm,
                        {1U, static_cast<uint32_t>(params.tilingData->topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            LocalTensor<float> weightFp32Ub = xOutTensor[weightOffsetInUb].template ReinterpretCast<float>();
            Cast(weightFp32Ub, weightBf16Tmp, AscendC::RoundMode::CAST_NONE, params.tilingData->topK);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<TopkWeightsType> weightUb =
                xOutTensor[weightOffsetInUb].template ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightUb, weightGm,
                        {1U, static_cast<uint32_t>(params.tilingData->topK * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
    }
}

template <int32_t QuantMode, typename QuantOutType, typename ActivationType>
__aicore__ inline void QuantizeTokenToLocal(const LocalTensor<bfloat16_t> &xInTensor,
                                            const LocalTensor<ActivationType> &xOutTensor,
                                            const QuantProcessConfig &config, uint32_t hiddenDim,
                                            __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *halfScaleAddr)
{
    __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)xInTensor.GetPhyAddr();
    __ubuf__ int8_t *outDataAddr = (__ubuf__ int8_t *)xOutTensor.GetPhyAddr();
    __ubuf__ uint16_t *mxScaleAddr = (__ubuf__ uint16_t *)xOutTensor[config.quantTokenAlignBytes].GetPhyAddr();

    Quant::ComputeMaxExp(srcAddr, maxExpAddr, hiddenDim);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, config.quantScaleNumAlignPerToken);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, hiddenDim);
    }
}

// 原型：MegaCombineRecv::QuantProcessInRank。量化一个逻辑 AIV 任务负责的本卡 token。
template <int32_t QuantMode, typename QuantOutType, typename ActivationType, typename TopkWeightsType,
          bool TopkWeightsPrefetch>
__aicore__ inline void QuantizeLocalTokens(const AivJobContext &job, const MoeStageCommonConfig &common,
                                           const Params &params, const QuantProcessConfig &config,
                                           QuantProcessScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    uint32_t hiddenDim = common.tokenHiddenDim;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.aGmAddr) +
                                    static_cast<uint64_t>(tokenRange.start) * hiddenDim);
    dstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(params.peermemInfo.quantTokenScalePtr) +
                                    static_cast<uint64_t>(tokenRange.start) * config.quantTokenScaleAlignBytes);
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(hiddenDim * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U, config.quantTokenScaleAlignBytes, 0U, 0U, 0U};
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(scratch.mxTempTensor.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        scratch.mxTempTensor[Ops::Base::CeilAlign(config.quantScaleNumAlignPerToken, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    // 量化 scratch（mxTemp/xOut0/xOut1/xIn0/xIn1）的跨 launch 残留清零由各编排的
    // SendAndQuantBuffInit 在分配处一次性完成（span 清零，与布局同源），见其注释。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = useFirstBuffer ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutTensor = useFirstBuffer ? scratch.xOutTensor0 : scratch.xOutTensor1;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * hiddenDim], xCopyInParams,
                    xCopyInPadParams);
        LoadTopkWeightsToUb<TopkWeightsType, ActivationType, TopkWeightsPrefetch>(params, config, scratch, xOutTensor,
                                                                                  tokenRange.start + index, event);
        if constexpr (!TopkWeightsPrefetch) {
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
        QuantizeTokenToLocal<QuantMode, QuantOutType>(xInTensor, xOutTensor, config, hiddenDim, maxExpAddr,
                                                      halfScaleAddr);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor[static_cast<uint64_t>(index) * config.quantTokenScaleAlignBytes], xOutBytesTensor,
                    xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace MegaCombineRecvImpl

#endif // MEGA_COMBINE_RECV_TOKEN_QUANT_H
