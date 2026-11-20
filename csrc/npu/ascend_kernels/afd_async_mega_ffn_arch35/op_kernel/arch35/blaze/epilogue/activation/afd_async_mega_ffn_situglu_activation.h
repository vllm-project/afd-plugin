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
 * \file situglu_activation.h
 * \brief SiTUGLU DEFAULT 和 LINEAR 门控激活 Reg VF 实现。
 */

#ifndef MEGA_FUSED_MOE_ARCH35_SITUGLU_ACTIVATION_H
#define MEGA_FUSED_MOE_ARCH35_SITUGLU_ACTIVATION_H

#if defined(__DAV_C310__)
// ### PATCH START: AFD merged source layout
#include "afd_async_mega_ffn_activation_common.h"
// ### PATCH END: AFD merged source layout
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"

namespace MegaFusedMoeImpl {
namespace Activation {

// SiTUGLU VF 调用使用的最小参数集。
struct SituGluParams {
    float clampLimit;
    float beta;
    float invBeta;
    float alpha;
    float invAlpha;
};

template <typename InputType>
__simd_callee__ inline void LoadSituGluInputAsFp32(AscendC::Reg::RegTensor<float> &dstReg,
                                                   AscendC::Reg::RegTensor<InputType> &inputReg,
                                                   __ubuf__ InputType *src, AscendC::Reg::MaskReg computeMask)
{
    AscendC::Reg::LoadAlign<InputType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(inputReg, src);
    AscendC::Reg::Cast<float, InputType, CAST_INPUT_TO_FP32>(dstReg, inputReg, computeMask);
}

__simd_callee__ inline void StoreSituGluFp32AsBf16(__ubuf__ bfloat16_t *dst,
                                                   AscendC::Reg::RegTensor<bfloat16_t> &outputReg,
                                                   AscendC::Reg::RegTensor<float> &srcReg,
                                                   AscendC::Reg::MaskReg computeMask, AscendC::Reg::MaskReg storeMask)
{
    AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outputReg, srcReg, computeMask);
    AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst, outputReg, storeMask);
}

__simd_callee__ inline void ComputeSituGluTanhPiecewiseApprox(
    AscendC::Reg::RegTensor<float> &resultReg, AscendC::Reg::RegTensor<float> &zReg,
    AscendC::Reg::RegTensor<float> &oneReg, AscendC::Reg::RegTensor<float> &absReg,
    AscendC::Reg::RegTensor<float> &squareReg, AscendC::Reg::RegTensor<float> &polynomialReg,
    AscendC::Reg::RegTensor<float> &temporaryReg, AscendC::Reg::RegTensor<float> &expReg,
    AscendC::Reg::RegTensor<float> &c1Reg, AscendC::Reg::RegTensor<float> &c2Reg, AscendC::Reg::MaskReg &compareMask,
    AscendC::Reg::MaskReg computeMask)
{
    AscendC::Reg::Mul(squareReg, zReg, zReg, computeMask);
    AscendC::Reg::Muls(polynomialReg, squareReg, 0.0157396831F, computeMask);
    AscendC::Reg::Adds(polynomialReg, polynomialReg, -0.0523039624F, computeMask);
    AscendC::Reg::MulDstAdd(polynomialReg, squareReg, c2Reg, computeMask);
    AscendC::Reg::MulDstAdd(polynomialReg, squareReg, c1Reg, computeMask);
    AscendC::Reg::Mul(polynomialReg, polynomialReg, squareReg, computeMask);
    AscendC::Reg::MulDstAdd(polynomialReg, zReg, zReg, computeMask);

    AscendC::Reg::Muls(expReg, zReg, -2.0F, computeMask);
    AscendC::Reg::Exp(expReg, expReg, computeMask);
    AscendC::Reg::Adds(expReg, expReg, 1.0F, computeMask);
    AscendC::Reg::Div(temporaryReg, oneReg, expReg, computeMask);
    AscendC::Reg::Muls(temporaryReg, temporaryReg, 2.0F, computeMask);
    AscendC::Reg::Adds(temporaryReg, temporaryReg, -1.0F, computeMask);

    AscendC::Reg::Abs(absReg, zReg, computeMask);
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(compareMask, absReg, 0.60000002384185791016F, computeMask);
    AscendC::Reg::Select(resultReg, temporaryReg, polynomialReg, compareMask);
}

template <bool IsLinear>
__simd_callee__ inline void ComputeSituGluRegFlow(
    AscendC::Reg::RegTensor<float> &outputFp32Reg, AscendC::Reg::RegTensor<float> &gateFp32Reg,
    AscendC::Reg::RegTensor<float> &upFp32Reg, AscendC::Reg::RegTensor<float> &negReg,
    AscendC::Reg::RegTensor<float> &expReg, AscendC::Reg::RegTensor<float> &denominatorReg,
    AscendC::Reg::RegTensor<float> &sigmoidReg, AscendC::Reg::RegTensor<float> &zReg,
    AscendC::Reg::RegTensor<float> &betaReg, AscendC::Reg::RegTensor<float> &oneReg,
    AscendC::Reg::RegTensor<float> &absReg, AscendC::Reg::RegTensor<float> &squareReg,
    AscendC::Reg::RegTensor<float> &polynomialReg, AscendC::Reg::RegTensor<float> &tanhTemporaryReg,
    AscendC::Reg::RegTensor<float> &c1Reg, AscendC::Reg::RegTensor<float> &c2Reg, AscendC::Reg::MaskReg &compareMask,
    AscendC::Reg::MaskReg computeMask, float clampLimit, float beta, float invBeta, float alpha, float invAlpha)
{
    AscendC::Reg::Mins(gateFp32Reg, gateFp32Reg, clampLimit, computeMask);
    AscendC::Reg::Mins(upFp32Reg, upFp32Reg, clampLimit, computeMask);
    AscendC::Reg::Maxs(upFp32Reg, upFp32Reg, -clampLimit, computeMask);

    AscendC::Reg::Muls(zReg, gateFp32Reg, invBeta, computeMask);
    ComputeSituGluTanhPiecewiseApprox(sigmoidReg, zReg, oneReg, absReg, squareReg, polynomialReg, tanhTemporaryReg,
                                      expReg, c1Reg, c2Reg, compareMask, computeMask);
    AscendC::Reg::Muls(negReg, gateFp32Reg, -1.0F, computeMask);
    AscendC::Reg::Exp(expReg, negReg, computeMask);
    AscendC::Reg::Adds(denominatorReg, expReg, 1.0F, computeMask);
    AscendC::Reg::Duplicate(betaReg, beta, computeMask);
    AscendC::Reg::Div(expReg, betaReg, denominatorReg, computeMask);
    AscendC::Reg::Mul(outputFp32Reg, sigmoidReg, expReg, computeMask);

    if constexpr (IsLinear) {
        AscendC::Reg::Muls(gateFp32Reg, outputFp32Reg, 1.0F, computeMask);
        AscendC::Reg::Muls(zReg, upFp32Reg, invAlpha, computeMask);
        ComputeSituGluTanhPiecewiseApprox(sigmoidReg, zReg, oneReg, absReg, squareReg, polynomialReg, tanhTemporaryReg,
                                          expReg, c1Reg, c2Reg, compareMask, computeMask);
        AscendC::Reg::Muls(sigmoidReg, sigmoidReg, alpha, computeMask);
        AscendC::Reg::Mul(outputFp32Reg, gateFp32Reg, sigmoidReg, computeMask);
    } else {
        AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, upFp32Reg, computeMask);
    }
}

template <typename InputType, bool TopkWeightsPrefetch, bool IsLinear>
__simd_vf__ inline void RunSiTUGLU(GatedActivationTileContext<InputType> context, SituGluParams params)
{
    __ubuf__ InputType *gate = context.gate;
    __ubuf__ InputType *up = context.up;
    __ubuf__ bfloat16_t *output = context.output;
    __ubuf__ float *topkWeights = context.topkWeights;
    __ubuf__ InputType *gateTail = context.gateTail;
    __ubuf__ InputType *upTail = context.upTail;
    __ubuf__ bfloat16_t *outputTail = context.outputTail;
    __ubuf__ bfloat16_t *additionalPaddingOutput = context.additionalPaddingOutput;

    const uint32_t inputRowStrideElements = context.inputRowStrideElements;
    const uint32_t outputRowStrideElements = context.outputRowStrideElements;
    const uint16_t rowLoopCount = context.rowLoopCount;
    const uint16_t fullVectorLoopCount = context.fullVectorLoopCount;
    const uint16_t needTailVectorCompute = context.needTailVectorCompute;
    const uint16_t needAdditionalPaddingStore = context.needAdditionalPaddingStore;
    uint32_t tailComputeMaskElementCount = context.tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount = context.tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount = context.additionalPaddingStoreMaskElementCount;

    const float clampLimit = params.clampLimit;
    const float beta = params.beta;
    const float invBeta = params.invBeta;
    const float alpha = params.alpha;
    const float invAlpha = params.invAlpha;

    AscendC::Reg::RegTensor<InputType> gateReg;
    AscendC::Reg::RegTensor<InputType> upReg;
    AscendC::Reg::RegTensor<float> gateFp32Reg;
    AscendC::Reg::RegTensor<float> upFp32Reg;
    AscendC::Reg::RegTensor<float> outputFp32Reg;
    AscendC::Reg::RegTensor<float> weightReg;
    AscendC::Reg::RegTensor<float> negReg;
    AscendC::Reg::RegTensor<float> expReg;
    AscendC::Reg::RegTensor<float> denominatorReg;
    AscendC::Reg::RegTensor<float> sigmoidReg;
    AscendC::Reg::RegTensor<float> zReg;
    AscendC::Reg::RegTensor<float> betaReg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::RegTensor<float> absReg;
    AscendC::Reg::RegTensor<float> squareReg;
    AscendC::Reg::RegTensor<float> polynomialReg;
    AscendC::Reg::RegTensor<float> tanhTemporaryReg;
    AscendC::Reg::RegTensor<float> c1Reg;
    AscendC::Reg::RegTensor<float> c2Reg;
    AscendC::Reg::RegTensor<bfloat16_t> outputReg;
    AscendC::Reg::RegTensor<bfloat16_t> zeroReg;

    AscendC::Reg::MaskReg compareMask;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg tailComputeMask = AscendC::Reg::UpdateMask<float>(tailComputeMaskElementCount);
    AscendC::Reg::MaskReg tailStoreMask = AscendC::Reg::UpdateMask<float>(tailStoreMaskElementCount);
    AscendC::Reg::MaskReg additionalPaddingStoreMask =
        AscendC::Reg::UpdateMask<bfloat16_t>(additionalPaddingStoreMaskElementCount);

    AscendC::Reg::Duplicate(oneReg, 1.0F);
    AscendC::Reg::Duplicate(c1Reg, -0.333327681F);
    AscendC::Reg::Duplicate(c2Reg, 0.133152977F);

    for (uint16_t rowIndex = 0; rowIndex < rowLoopCount; ++rowIndex) {
        if constexpr (TopkWeightsPrefetch) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                weightReg, topkWeights + rowIndex * INT32_PER_256B + WEIGHT_INDEX);
        }

        const uint32_t inputRowOffset = static_cast<uint32_t>(rowIndex) * inputRowStrideElements;
        const uint32_t outputRowOffset = static_cast<uint32_t>(rowIndex) * outputRowStrideElements;
        for (uint16_t vectorIndex = 0; vectorIndex < fullVectorLoopCount; ++vectorIndex) {
            const uint32_t vectorOffset = static_cast<uint32_t>(vectorIndex) * VECTOR_LENGTH_FP32;
            LoadSituGluInputAsFp32<InputType>(gateFp32Reg, gateReg, gate + inputRowOffset + vectorOffset, fullMask);
            LoadSituGluInputAsFp32<InputType>(upFp32Reg, upReg, up + inputRowOffset + vectorOffset, fullMask);
            ComputeSituGluRegFlow<IsLinear>(outputFp32Reg, gateFp32Reg, upFp32Reg, negReg, expReg, denominatorReg,
                                            sigmoidReg, zReg, betaReg, oneReg, absReg, squareReg, polynomialReg,
                                            tanhTemporaryReg, c1Reg, c2Reg, compareMask, fullMask, clampLimit, beta,
                                            invBeta, alpha, invAlpha);
            if constexpr (TopkWeightsPrefetch) {
                AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, fullMask);
            }
            StoreSituGluFp32AsBf16(output + outputRowOffset + vectorOffset, outputReg, outputFp32Reg, fullMask,
                                   fullMask);
        }

        for (uint16_t tailIndex = 0; tailIndex < needTailVectorCompute; ++tailIndex) {
            LoadSituGluInputAsFp32<InputType>(gateFp32Reg, gateReg, gateTail + inputRowOffset, tailComputeMask);
            LoadSituGluInputAsFp32<InputType>(upFp32Reg, upReg, upTail + inputRowOffset, tailComputeMask);
            ComputeSituGluRegFlow<IsLinear>(outputFp32Reg, gateFp32Reg, upFp32Reg, negReg, expReg, denominatorReg,
                                            sigmoidReg, zReg, betaReg, oneReg, absReg, squareReg, polynomialReg,
                                            tanhTemporaryReg, c1Reg, c2Reg, compareMask, tailComputeMask, clampLimit,
                                            beta, invBeta, alpha, invAlpha);
            if constexpr (TopkWeightsPrefetch) {
                AscendC::Reg::Mul(outputFp32Reg, outputFp32Reg, weightReg, tailComputeMask);
            }
            StoreSituGluFp32AsBf16(outputTail + outputRowOffset, outputReg, outputFp32Reg, tailComputeMask,
                                   tailStoreMask);
        }

        for (uint16_t additionalPaddingIndex = 0; additionalPaddingIndex < needAdditionalPaddingStore;
             ++additionalPaddingIndex) {
            AscendC::Reg::Duplicate(zeroReg, static_cast<bfloat16_t>(0));
            AscendC::Reg::StoreAlign<bfloat16_t>(additionalPaddingOutput + outputRowOffset, zeroReg,
                                                 additionalPaddingStoreMask);
        }
    }
}

} // namespace Activation
} // namespace MegaFusedMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_FUSED_MOE_ARCH35_SITUGLU_ACTIVATION_H
