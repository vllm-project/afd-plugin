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
 * Description: Apply fused activation and quantization in the GEMM epilogue.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

/*!
 * \file block_epilogue_activation_mx_quant.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
#define BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
// ### PATCH START: AFD merged source layout
#include "quantize_functions.h"
// ### PATCH END: AFD merged source layout

#include "../../common/atten_send_constants.h"
#include "../../common/atten_send_utils.h"
// ### PATCH START: AFD merged source layout
#include "activation/afd_async_atten_send_activation_common.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "activation/afd_async_atten_send_situglu_activation.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "activation/afd_async_atten_send_swiglu_activation.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "activation/afd_async_atten_send_swigluoai_activation.h"
// ### PATCH END: AFD merged source layout
// ### PATCH START: AFD merged source layout
#include "afd_async_atten_send_block_epilogue_ub_layout.h"
// ### PATCH END: AFD merged source layout

namespace ActivationQuantMsg {
using ActMode = MegaDispatchSendImpl::MegaDispatchSendActMode;
using ActSubMode = MegaDispatchSendImpl::MegaDispatchSendActSubMode;

constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
} // namespace ActivationQuantMsg

namespace MegaDispatchSendImpl {

using namespace AscendC;
using namespace ActivationQuantMsg;

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM = 256, uint32_t TileN = 256,
          bool TopkWeightsPrefetch = false, bool IsInterleaved = false>
class BlockEpilogueActivationMxQuant {
public:
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr uint32_t MAX_SINGLE_MN = TileM * TileN;

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        float clampLimit{0.0f};
        uint8_t actMode{static_cast<uint8_t>(ActMode::SWIGLU)};
        uint8_t actSubMode{static_cast<uint8_t>(ActSubMode::DEFAULT)};
        float activationAlpha{1.0f};
        float activationBeta{1.0f};
        Arguments() = default;
    };

    using Params = Arguments;

    __aicore__ inline BlockEpilogueActivationMxQuant() {}

    __aicore__ inline void Init(Params const &params);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);
    __aicore__ inline auto GetTopkWeightTensor();

    // 计算并写回一个激活/量化 tile，不包含流水通知。
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0);

private:
    using UbPointers = ActivationMxQuantUbPointers<DataTypeIn>;

    struct TileGeometry {
        uint32_t rowCount;
        uint32_t columnCount;
        uint32_t outputRowStrideElements;
    };

    struct TileExecutionContext {
        TileGeometry geometry;
        UbPointers buffers;
        uint64_t outputOffset;
        uint64_t scaleOffset;
        uint16_t pingpongIndex;
    };

    static constexpr auto ubLayout = BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved>();
    static constexpr uint32_t ubFirstOffset = ubLayout.firstInputOffsetBytes;
    static constexpr uint32_t ubInputElementCapacity = ubLayout.firstInputElementCapacity;
    static constexpr uint32_t ubSecondOffset = ubLayout.secondInputOffsetBytes;

    __aicore__ inline TileExecutionContext PrepareTileExecutionContext(const BlockShape &blockShape,
                                                                       const BlockCoord &blockCoord,
                                                                       uint16_t pingpongIdx);

    __aicore__ static inline uint32_t ComputeGatedActivationInputRowStride(uint16_t validColumnCount);

    __aicore__ static inline void ConfigureGatedActivationTail(
        Activation::GatedActivationTileContext<DataTypeIn> &context, uint16_t validColumnCount);

    __aicore__ inline Activation::GatedActivationTileContext<DataTypeIn> BuildGatedActivationContext(
        const TileExecutionContext &tileContext);

    __aicore__ inline void RunGatedActivationTile(const TileExecutionContext &tileContext);

    __aicore__ inline void RunMxQuantTile(const TileExecutionContext &tileContext);

    __aicore__ inline void StoreQuantTile(const TileExecutionContext &tileContext);

    __aicore__ inline void StoreQuantOutput(AscendC::GlobalTensor<int8_t> &dst, AscendC::LocalTensor<int8_t> &src,
                                            uint64_t blockCount, uint64_t offset, uint64_t n, uint64_t singleN);

    __aicore__ inline void StoreQuantScaleCompact(AscendC::GlobalTensor<int8_t> &dst, AscendC::LocalTensor<int8_t> &src,
                                                  uint64_t blockCount, uint64_t offset, uint64_t scaleN,
                                                  uint64_t singleN);

    // GM base addresses and bound tensor views
    GM_ADDR yGmAddr_{nullptr};
    GM_ADDR yScaleGmAddr_{nullptr};
    GlobalTensor<int8_t> quantOutputGlobal_;
    GlobalTensor<int8_t> quantScaleGlobal_;

    // UB tensor views ordered by physical layout
    LocalTensor<DataTypeIn> l0cOutUbFirst_{TPosition::VECIN, ubFirstOffset, ubInputElementCapacity};
    LocalTensor<DataTypeIn> l0cOutUbSecond_{TPosition::VECIN, ubSecondOffset, ubInputElementCapacity};
    LocalTensor<bfloat16_t> gluRes_;
    LocalTensor<int8_t> quantOutput_;
    LocalTensor<int8_t> quantScaleOutput_;
    LocalTensor<uint16_t> maxExp_;
    LocalTensor<uint16_t> inverseMxScale_;
    LocalTensor<float> weightUb_;

    // Problem state
    int64_t intermediateHiddenSize_;
    int64_t intermediateHiddenScaleElements_;

    // Activation state
    float clampLimit_{0.0f};
    uint8_t actMode_{static_cast<uint8_t>(ActMode::SWIGLU)};
    uint8_t actSubMode_{static_cast<uint8_t>(ActSubMode::DEFAULT)};
    float activationAlpha_{1.0f};
    float activationBeta_{1.0f};
};

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved>::Init(Params const &params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Params 的生命周期可能短于 Epilogue 对象，因此保存输出 GM 地址。
    yGmAddr_ = params.yGmAddr;
    yScaleGmAddr_ = params.yScaleGmAddr;
    clampLimit_ = params.clampLimit;
    actMode_ = params.actMode;
    actSubMode_ = params.actSubMode;
    activationAlpha_ = params.activationAlpha;
    activationBeta_ = params.activationBeta;

    gluRes_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, ubLayout.activationOutputOffsetBytes,
                                      ubLayout.activationQuantElementCapacity);
    quantOutput_ = LocalTensor<int8_t>(TPosition::VECOUT, ubLayout.quantOutputOffsetBytes,
                                       ubLayout.activationQuantElementCapacity);
    quantScaleOutput_ =
        LocalTensor<int8_t>(TPosition::VECOUT, ubLayout.quantScaleOffsetBytes, ubLayout.scaleElementCapacity);
    maxExp_ = LocalTensor<uint16_t>(TPosition::VECCALC, ubLayout.maxExpOffsetBytes, ubLayout.scaleElementCapacity);
    inverseMxScale_ =
        LocalTensor<uint16_t>(TPosition::VECCALC, ubLayout.inverseMxScaleOffsetBytes, ubLayout.scaleElementCapacity);

    // weight UB: 放在 VECIN 区之后的安全区域，仅 mode=1 分配
    if constexpr (TopkWeightsPrefetch) {
        weightUb_ =
            LocalTensor<float>(TPosition::VECCALC, ubLayout.topkWeightOffsetBytes, ubLayout.topkWeightElementCapacity);
    }
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved>::UpdateGlobalAddr(const BlockCoord &baseOffset)
{
    if constexpr (g_coreType == AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)yGmAddr_ + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)yScaleGmAddr_ + Get<Y_SCALE_IDX>(baseOffset));
    }
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::UpdateNextProblem(const ProblemShape &problemShape)
{
    intermediateHiddenSize_ = Get<N_VALUE>(problemShape);
    intermediateHiddenScaleElements_ =
        Ops::Base::CeilDiv(static_cast<uint64_t>(intermediateHiddenSize_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline auto BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved>::GetTopkWeightTensor()
{
    return weightUb_;
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline typename BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                                                          IsInterleaved>::TileExecutionContext
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::PrepareTileExecutionContext(const BlockShape &blockShape,
                                                                           const BlockCoord &blockCoord,
                                                                           uint16_t pingpongIdx)
{
    TileExecutionContext tileContext{};
    tileContext.geometry.rowCount = Get<M_VALUE>(blockShape);
    tileContext.geometry.columnCount = Get<N_VALUE>(blockShape);
    tileContext.pingpongIndex = pingpongIdx;

    tileContext.geometry.outputRowStrideElements =
        Ops::Base::CeilAlign(tileContext.geometry.columnCount, static_cast<uint32_t>(ONE_BLK_SIZE));
    tileContext.outputOffset = Get<Y_IDX>(blockCoord);
    tileContext.scaleOffset = Get<Y_SCALE_IDX>(blockCoord);

    __ubuf__ DataTypeIn *secondInputBase = nullptr;
    if constexpr (!IsInterleaved) {
        secondInputBase = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    }
    tileContext.buffers = ResolveActivationMxQuantUbPointers<DataTypeIn, MAX_SINGLE_MN, IsInterleaved>(
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr(), secondInputBase,
        (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr(), (__ubuf__ int8_t *)quantOutput_.GetPhyAddr(),
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr(), (__ubuf__ uint16_t *)maxExp_.GetPhyAddr(),
        (__ubuf__ uint16_t *)inverseMxScale_.GetPhyAddr(), tileContext.geometry.columnCount, tileContext.pingpongIndex);
    return tileContext;
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline uint32_t
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::ComputeGatedActivationInputRowStride(uint16_t validColumnCount)
{
    if constexpr (IsInterleaved) {
        // interleaved源布局为[x1, x2]连续存放在同一行，下一行stride是2*validColumnCount。
        // 注意：2*validColumnCount==实际行距(TileN) 仅在满 tile 成立；tiling 已约束 hiddenDim%256==0，
        // 交织调度宽度为完整 hiddenDim，故交织路径不会出现尾 tile；若未来放宽须改用 TileN。
        return static_cast<uint32_t>(validColumnCount) * 2U;
    }
    // 非交织源是两块独立 UB tile，生产端始终按固定行距 TileN 写入；尾块有效列数较小时，
    // 行距也不能随之收缩，否则下一行输入会发生错位。
    return TileN;
}

// 保留主循环、尾 Vector 计算和两段补零能力，兼容后续放宽列方向对齐约束。
template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void BlockEpilogueActivationMxQuant<
    DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
    IsInterleaved>::ConfigureGatedActivationTail(Activation::GatedActivationTileContext<DataTypeIn> &context,
                                                 uint16_t validColumnCount)
{
    const uint32_t vectorLength = Activation::VECTOR_LENGTH_FP32;
    context.fullVectorLoopCount = validColumnCount / vectorLength;
    context.needTailVectorCompute = 0U;
    context.needAdditionalPaddingStore = 0U;
    context.tailComputeMaskElementCount = 0U;
    context.tailStoreMaskElementCount = 0U;
    context.additionalPaddingStoreMaskElementCount = 0U;
    context.gateTail = context.gate;
    context.upTail = context.up;
    context.outputTail = context.output;
    context.additionalPaddingOutput = context.output;

    const uint32_t tailElementCount = validColumnCount % vectorLength;
    if (tailElementCount == 0U) {
        return;
    }

    context.tailComputeMaskElementCount = tailElementCount;
    context.needTailVectorCompute = 1U;
    const uint32_t tailAndPaddingElementCount =
        context.outputRowStrideElements - context.fullVectorLoopCount * vectorLength;
    if (tailAndPaddingElementCount <= vectorLength) {
        context.tailStoreMaskElementCount = tailAndPaddingElementCount;
    } else {
        context.needAdditionalPaddingStore = 1U;
        context.tailStoreMaskElementCount = vectorLength;
        context.additionalPaddingStoreMaskElementCount = tailAndPaddingElementCount - vectorLength;
    }
    const uint32_t tailColumnOffsetElements = context.fullVectorLoopCount * vectorLength;
    context.gateTail = context.gate + tailColumnOffsetElements;
    context.upTail = context.up + tailColumnOffsetElements;
    context.outputTail = context.output + tailColumnOffsetElements;
    context.additionalPaddingOutput = context.outputTail + context.needTailVectorCompute * vectorLength;
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline Activation::GatedActivationTileContext<DataTypeIn>
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::BuildGatedActivationContext(const TileExecutionContext &tileContext)
{
    const UbPointers &ubPointers = tileContext.buffers;
    const uint16_t validColumnCount = static_cast<uint16_t>(tileContext.geometry.columnCount);
    Activation::GatedActivationTileContext<DataTypeIn> gatedActivationContext{};
    gatedActivationContext.gate = ubPointers.firstInput;
    gatedActivationContext.up = ubPointers.secondInput;
    gatedActivationContext.output = ubPointers.activationOutput;
    gatedActivationContext.inputRowStrideElements = ComputeGatedActivationInputRowStride(validColumnCount);
    gatedActivationContext.outputRowStrideElements = tileContext.geometry.outputRowStrideElements;
    gatedActivationContext.rowLoopCount = static_cast<uint16_t>(tileContext.geometry.rowCount);
    ConfigureGatedActivationTail(gatedActivationContext, validColumnCount);
    if constexpr (TopkWeightsPrefetch) {
        gatedActivationContext.topkWeights = (__ubuf__ float *)weightUb_.GetPhyAddr();
    } else {
        gatedActivationContext.topkWeights = nullptr;
    }
    return gatedActivationContext;
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::RunGatedActivationTile(const TileExecutionContext &tileContext)
{
    Activation::GatedActivationTileContext<DataTypeIn> gatedActivationContext =
        BuildGatedActivationContext(tileContext);

    const ActMode actMode = static_cast<ActMode>(actMode_);
    const ActSubMode actSubMode = static_cast<ActSubMode>(actSubMode_);
    if (actMode == ActMode::SITU) {
        const float invBeta = activationBeta_ != 0.0f ? 1.0f / activationBeta_ : 1.0f;
        if (actSubMode == ActSubMode::LINEAR) {
            const float invAlpha = activationAlpha_ != 0.0f ? 1.0f / activationAlpha_ : 1.0f;
            const Activation::SituGluParams situGluParams{clampLimit_, activationBeta_, invBeta, activationAlpha_,
                                                          invAlpha};
            asc_vf_call<Activation::RunSiTUGLU<DataTypeIn, TopkWeightsPrefetch, true>>(gatedActivationContext,
                                                                                       situGluParams);
        } else {
            const Activation::SituGluParams situGluParams{clampLimit_, activationBeta_, invBeta, 1.0f, 1.0f};
            asc_vf_call<Activation::RunSiTUGLU<DataTypeIn, TopkWeightsPrefetch, false>>(gatedActivationContext,
                                                                                        situGluParams);
        }
    } else if (actMode == ActMode::SWIGLU_STEP) {
        asc_vf_call<Activation::RunSwiGLU<DataTypeIn, TopkWeightsPrefetch, true>>(gatedActivationContext, clampLimit_);
    } else if (actMode == ActMode::SWIGLU_OAI) {
        const Activation::SwiGluOaiParams swiGluOaiParams{clampLimit_, activationAlpha_, activationBeta_};
        asc_vf_call<Activation::RunSwiGLUOAI<DataTypeIn, TopkWeightsPrefetch>>(gatedActivationContext, swiGluOaiParams);
    } else {
        asc_vf_call<Activation::RunSwiGLU<DataTypeIn, TopkWeightsPrefetch, false>>(gatedActivationContext, clampLimit_);
    }
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::RunMxQuantTile(const TileExecutionContext &tileContext)
{
    const UbPointers &ubPointers = tileContext.buffers;
    const uint16_t validRowCount = static_cast<uint16_t>(tileContext.geometry.rowCount);
    const uint32_t outputRowStrideElements = tileContext.geometry.outputRowStrideElements;
    const uint32_t dataCount = validRowCount * outputRowStrideElements;
    const uint32_t scaleCount = dataCount / ONE_BLK_SIZE;

    Quant::ComputeMaxExp<bfloat16_t>(ubPointers.activationOutput, ubPointers.maxExp, dataCount);
    Quant::ComputeScale<DataTypeOut>(ubPointers.maxExp, ubPointers.quantScale, ubPointers.inverseMxScale, scaleCount);
    if constexpr (IsSameType<DataTypeOut, fp8_e4m3fn_t>::value || IsSameType<DataTypeOut, fp8_e5m2_t>::value) {
        Quant::ComputeFp8Data<bfloat16_t, DataTypeOut, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            ubPointers.activationOutput, ubPointers.inverseMxScale, ubPointers.quantOutput, dataCount);
    }
    if constexpr (IsSameType<DataTypeOut, fp4x2_e2m1_t>::value || IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        Quant::ComputeFp4Data<bfloat16_t, DataTypeOut, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            ubPointers.activationOutput, ubPointers.inverseMxScale, ubPointers.quantOutput, dataCount);
    }
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::StoreQuantTile(const TileExecutionContext &tileContext)
{
    const TileGeometry &tileGeometry = tileContext.geometry;
    const uint32_t bufferOffset = tileContext.buffers.selectedInt8BufferOffsetElements;
    LocalTensor<int8_t> quantOutputTile = quantOutput_[bufferOffset];
    LocalTensor<int8_t> quantScaleTile = quantScaleOutput_[bufferOffset];

    StoreQuantOutput(quantOutputGlobal_, quantOutputTile, tileGeometry.rowCount, tileContext.outputOffset,
                     intermediateHiddenSize_, tileGeometry.columnCount);
    // scale已按compact布局生成，直接copy到GM，省掉原先TransMxScaleLayout重排scale。
    StoreQuantScaleCompact(quantScaleGlobal_, quantScaleTile, tileGeometry.rowCount, tileContext.scaleOffset,
                           intermediateHiddenScaleElements_, tileGeometry.columnCount);
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::StoreQuantOutput(AscendC::GlobalTensor<int8_t> &dst,
                                                                AscendC::LocalTensor<int8_t> &src, uint64_t blockCount,
                                                                uint64_t offset, uint64_t n, uint64_t singleN)
{
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = blockCount; // 128
    if constexpr (AscendC::IsSameType<DataTypeOut, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<DataTypeOut, fp4x2_e1m2_t>::value) {
        ub2GmParams.blockLen = singleN >> 1;
        ub2GmParams.dstStride = (n - singleN) >> 1;
        offset = offset >> 1;
    } else {
        uint64_t nDstUbAligned =
            Ops::Base::CeilAlign(static_cast<uint64_t>(singleN), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
        ub2GmParams.blockLen = singleN; // 256
        ub2GmParams.srcStride = (nDstUbAligned - singleN) / AscendC::ONE_BLK_SIZE;
        ub2GmParams.dstStride = n - singleN;
    }
    AscendC::DataCopyPad(dst[offset], src, ub2GmParams);
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void
BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                               IsInterleaved>::StoreQuantScaleCompact(AscendC::GlobalTensor<int8_t> &dst,
                                                                      AscendC::LocalTensor<int8_t> &src,
                                                                      uint64_t blockCount, uint64_t offset,
                                                                      uint64_t scaleN, uint64_t singleN)
{
    AscendC::DataCopyExtParams ub2GmParams{0, 0, 0, 0, 0};
    auto blockScaleN = Ops::Base::CeilDiv(static_cast<uint64_t>(singleN), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE;
    // scale layout in UB is already compact: (mSize, blockScaleN). Compact copy avoids (mSize*8)->(mSize,32).
    ub2GmParams.blockCount = blockCount; // 128
    ub2GmParams.blockLen = blockScaleN;  // 8
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = scaleN - blockScaleN;
    AscendC::DataCopyPad<int8_t, AscendC::PaddingMode::Compact>(dst[offset], src, ub2GmParams);
}

template <typename DataTypeOut, typename DataTypeIn, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch,
          bool IsInterleaved>
__aicore__ inline void BlockEpilogueActivationMxQuant<DataTypeOut, DataTypeIn, TileM, TileN, TopkWeightsPrefetch,
                                                      IsInterleaved>::operator()(const BlockShape &blockShape,
                                                                                 const BlockCoord &blockCoord,
                                                                                 uint16_t pingpongIdx)
{
    if (Get<M_VALUE>(blockShape) == 0) {
        return;
    }

    TileExecutionContext tileContext = PrepareTileExecutionContext(blockShape, blockCoord, pingpongIdx);
    RunGatedActivationTile(tileContext);
    RunMxQuantTile(tileContext);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    StoreQuantTile(tileContext);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}

} // namespace MegaDispatchSendImpl

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
