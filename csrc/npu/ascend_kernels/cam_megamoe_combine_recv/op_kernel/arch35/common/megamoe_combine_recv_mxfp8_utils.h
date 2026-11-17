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
 * Description: Provide shared mxfp8 utils definitions for split MegaMoE execution.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

#ifndef MEGAMOE_COMBINE_RECV_MXFP8_UTILS_H
#define MEGAMOE_COMBINE_RECV_MXFP8_UTILS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#endif
#include "megamoe_combine_recv_constants.h"
#if __has_include("../../../common/quantize_functions.h")
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/quantize_functions.h"
// ### PATCH END: AFD merged source layout
#else
// ### PATCH START: AFD merged source layout
#include "mc2/common/op_kernel/quantize_functions.h"
// ### PATCH END: AFD merged source layout
#endif

namespace MegamoeCombineRecvImpl {

using namespace AscendC;

namespace Mxfp8 {

template <typename InputType, typename Fp8Type>
__aicore__ inline void ComputeFp8Token(__ubuf__ InputType *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                       __ubuf__ uint16_t *mxScaleAddr, __ubuf__ uint16_t *halfScaleAddr,
                                       __ubuf__ int8_t *outDataAddr, uint32_t processLen, uint32_t scaleNum)
{
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, processLen);
    Quant::ComputeScale<Fp8Type>(maxExpAddr, mxScaleAddr, halfScaleAddr, scaleNum);
    Quant::ComputeFp8Data<InputType, Fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
        srcAddr, halfScaleAddr, outDataAddr, processLen);
}

// 将一行 BF16 token 量化为带 padding 的 MXFP8 data + scale 记录。
template <uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantMxFp8(LocalTensor<ExpandXType> &outLocal, LocalTensor<ExpandXType> &inLocal,
                                  LocalTensor<float> &floatTemp, int32_t processLen)
{
    uint32_t mxScaleNum = Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(static_cast<uint32_t>(processLen), static_cast<uint32_t>(ALIGN_32)), 2U);
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    LocalTensor<Fp8Type> castFp8LocalTensor = outLocal.template ReinterpretCast<Fp8Type>();
    __ubuf__ ExpandXType *srcAddr = (__ubuf__ ExpandXType *)inLocal.GetPhyAddr();
    __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatTemp.GetPhyAddr();
    __ubuf__ uint16_t *halfScaleLocalAddr =
        (__ubuf__ uint16_t *)floatTemp[Ops::Base::CeilAlign(mxScaleNum, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
    __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)castFp8LocalTensor.GetPhyAddr();
    uint32_t tokenStorageElementCount =
        Ops::Base::CeilAlign(static_cast<uint32_t>(processLen), static_cast<uint32_t>(ALIGN_256));
    __ubuf__ uint16_t *mxScaleLocalAddr =
        (__ubuf__ uint16_t *)castFp8LocalTensor[tokenStorageElementCount].GetPhyAddr();
    ComputeFp8Token<ExpandXType, Fp8Type>(srcAddr, maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, outLocalAddr,
                                          static_cast<uint32_t>(processLen), mxScaleNum);
}

// 将一条 MXFP8 token 记录反量化为 FP32，供 Unpermute 累加。
__simd_callee__ inline void ConvertMxScaleToBf16(__ubuf__ fp8_e8m0_t *srcPtr0, __ubuf__ bfloat16_t *dyScaleBf16Ptr,
                                                 uint32_t scaleLen, uint32_t bf16RepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<fp8_e8m0_t> vSrcReg;
    AscendC::Reg::RegTensor<bfloat16_t> vDstReg;
    static constexpr AscendC::Reg::CastTrait FP82BF16CastTraitZero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    AscendC::Reg::MaskReg maskReg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg = AscendC::Reg::UpdateMask<bfloat16_t>(scaleLen);
        Reg::DataCopy<fp8_e8m0_t, Reg::LoadDist::DIST_UNPACK_B8>(vSrcReg, srcPtr0 + i * bf16RepeatSize);
        Reg::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
        Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_INTLV_B16>(dyScaleBf16Ptr + i * bf16RepeatSize * 2, vDstReg,
                                                                  vDstReg, maskReg);
    }
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

__simd_callee__ inline void ConvertMxScaleToFp32(__ubuf__ bfloat16_t *dyScaleBf16Ptr, __ubuf__ float *dyScaleFp32Ptr,
                                                 uint32_t quantCount, uint32_t fp32RepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<bfloat16_t> dyScaleBf16Reg;
    AscendC::Reg::RegTensor<float> dyScaleFp32Reg;
    static constexpr AscendC::Reg::CastTrait FP162FP32CastTraitZero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    AscendC::Reg::MaskReg maskReg1;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg1 = AscendC::Reg::UpdateMask<float>(quantCount);
        Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg, dyScaleBf16Ptr + i * fp32RepeatSize);
        Reg::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(dyScaleFp32Ptr + i * fp32RepeatSize * 2, dyScaleFp32Reg,
                                                             dyScaleFp32Reg, maskReg1);
    }
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename T>
__simd_callee__ inline void DequantizeMxFp8Token(__ubuf__ T *tokenPtr0, __ubuf__ float *dyScaleFp32Ptr,
                                                 __ubuf__ float *sumDstPtr, uint32_t tokenLen, uint32_t fp32RepeatSize,
                                                 uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> tokenSrcReg;
    AscendC::Reg::RegTensor<float> tokenFp32SrcReg;
    AscendC::Reg::RegTensor<float> dyScaleFp32Reg;
    AscendC::Reg::RegTensor<float> sumDstReg;
    AscendC::Reg::RegTensor<float> sumLocalDstReg;
    static constexpr AscendC::Reg::CastTrait FP82BF16CastTraitZero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    AscendC::Reg::MaskReg maskReg2;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg2 = AscendC::Reg::UpdateMask<float>(tokenLen);
        Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
        Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg, tokenPtr0 + i * fp32RepeatSize);
        Reg::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
        Reg::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2);
        Reg::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2);
    }
}

template <typename T, typename XType>
__simd_vf__ inline void DeQuantMxFp8Vf(__ubuf__ XType *inAddr, __ubuf__ float *sumDstPtr,
                                       __ubuf__ bfloat16_t *dyScaleBf16Ptr, __ubuf__ float *dyScaleFp32Ptr,
                                       uint32_t scaleLen, uint32_t tokenLen)
{
    __ubuf__ T *tokenPtr0 = (__ubuf__ T *)inAddr;
    uint32_t tokenStorageElementCount = ((tokenLen + ALIGN_256 - 1U) / ALIGN_256 * ALIGN_256) / 2U;
    __ubuf__ fp8_e8m0_t *srcPtr0 = (__ubuf__ fp8_e8m0_t *)(inAddr + tokenStorageElementCount);
    constexpr uint32_t bf16RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint32_t fp32RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t bf16RepeatTimes = (scaleLen + bf16RepeatSize - 1U) / bf16RepeatSize;
    uint32_t quantCount = scaleLen * 2U;
    uint16_t scaleFp32RepeatTimes = (quantCount + fp32RepeatSize - 1U) / fp32RepeatSize;
    uint16_t tokenFp32RepeatTimes = (tokenLen + fp32RepeatSize - 1U) / fp32RepeatSize;
    ConvertMxScaleToBf16(srcPtr0, dyScaleBf16Ptr, scaleLen, bf16RepeatSize, bf16RepeatTimes);
    ConvertMxScaleToFp32(dyScaleBf16Ptr, dyScaleFp32Ptr, quantCount, fp32RepeatSize, scaleFp32RepeatTimes);
    DequantizeMxFp8Token(tokenPtr0, dyScaleFp32Ptr, sumDstPtr, tokenLen, fp32RepeatSize, tokenFp32RepeatTimes);
}

template <typename T, typename XType>
__aicore__ inline void DeQuantMxFp8(LocalTensor<XType> &inLocal, LocalTensor<float> &sumTensor,
                                    LocalTensor<bfloat16_t> &scaleBf16Tensor, LocalTensor<float> &scaleFP32Tensor,
                                    uint32_t scaleLen, uint32_t tokenLen)
{
    __ubuf__ XType *inAddr = (__ubuf__ XType *)inLocal.GetPhyAddr();
    __ubuf__ bfloat16_t *dyScaleBf16Ptr = (__ubuf__ bfloat16_t *)scaleBf16Tensor.GetPhyAddr();
    __ubuf__ float *dyScaleFp32Ptr = (__ubuf__ float *)scaleFP32Tensor.GetPhyAddr();
    __ubuf__ float *sumDstPtr = (__ubuf__ float *)sumTensor.GetPhyAddr();
    asc_vf_call<DeQuantMxFp8Vf<T, XType>>(inAddr, sumDstPtr, dyScaleBf16Ptr, dyScaleFp32Ptr, scaleLen, tokenLen);
}

} // namespace Mxfp8
} // namespace MegamoeCombineRecvImpl

#endif // MEGAMOE_COMBINE_RECV_MXFP8_UTILS_H
