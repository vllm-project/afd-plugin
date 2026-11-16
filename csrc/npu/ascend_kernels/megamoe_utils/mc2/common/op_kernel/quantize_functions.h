/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize_functions.h
 * \brief
 */

#ifndef QUANTIZE_FUNCTIONS_H
#define QUANTIZE_FUNCTIONS_H

#include <cstdint>

namespace Quant {

constexpr int DIGIT_TWO = 2;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t BF16_NAN_CUSTOM = 0x7f81;
constexpr uint16_t BF16_SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_BF16_MAX_EXP = 0x0000;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t OUT_ELE_NUM_PER_LOOP_FP4 = 128;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57344是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t FP32_BIASED_EXP_MAX_NORMAL = 0x000000fe;
constexpr uint32_t FP32_MANTISSA_HALF = 0x00400000;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint16_t MAX_EXP_FOR_FP16 = 0x7c00;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t EPS_1E_4_FP32 = 0x38D1B717; // 1e-4 的 FP32 bit pattern，用于 cuBLAS scale 路径下限 clamp

using namespace AscendC;

template <typename T>
__aicore__ inline T CeilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

__aicore__ inline constexpr uint32_t GetUbBlockSizeDispatch()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSizeDispatch()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline void ComputeMaxExp(__ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t eleNum)
{
    uint32_t vlB16Double = GetVRegSizeDispatch() / sizeof(T) * 2; // 两个向量寄存器可以存储的元素个数
    uint16_t blockPerReg = GetVRegSizeDispatch() / GetUbBlockSizeDispatch(); // Reduce操作后搬出的元素个数
    uint16_t loopNum = CeilDiv(eleNum, vlB16Double);

    __VEC_SCOPE__
    {
        Reg::RegTensor<half> x0Half, x1Half;
        Reg::RegTensor<bfloat16_t> x0BF16, x1BF16;
        Reg::RegTensor<uint16_t> xExpSelect0, xExpSelect1;
        Reg::RegTensor<uint16_t> xExpExtract0, xExpExtract1;
        Reg::RegTensor<uint16_t> xMaxExp;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::Duplicate(expMaskFP16, MAX_EXP_FOR_FP16);

        Reg::MaskReg validExpMask0, validExpMask1;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg validEleMaskEven = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg validEleMaskOdd = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::UnalignRegForStore ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum; i++) {
            if constexpr (Std::IsSame<T, bfloat16_t>::value) {
                Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    x0BF16, x1BF16, srcAddr, vlB16Double);
            } else if constexpr (Std::IsSame<T, half>::value) {
                Reg::LoadAlign<half, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    x0Half, x1Half, srcAddr, vlB16Double);
                Reg::Cast<bfloat16_t, half, castTraitHalf2Bf16>(x0BF16, x0Half, validEleMaskEven);
                Reg::Cast<bfloat16_t, half, castTraitHalf2Bf16>(x1BF16, x1Half, validEleMaskOdd);
            }
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t> &)x0BF16, expMaskBF16, validEleMaskEven);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t> &)x1BF16, expMaskBF16, validEleMaskOdd);
            if constexpr (Std::IsSame<T, half>::value) {
                Reg::And(xExpSelect0, (Reg::RegTensor<uint16_t> &)x0Half, expMaskFP16, validEleMaskEven);
                Reg::And(xExpSelect1, (Reg::RegTensor<uint16_t> &)x1Half, expMaskFP16, validEleMaskOdd);
                Reg::Compare<uint16_t, CMPMODE::NE>(validExpMask0, xExpSelect0, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(validExpMask1, xExpSelect1, expMaskFP16, maskAll);
                Reg::Select<uint16_t>(xExpExtract0, xExpExtract0, expMaskBF16, validExpMask0);
                Reg::Select<uint16_t>(xExpExtract1, xExpExtract1, expMaskBF16, validExpMask1);
            }

            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, maskAll);
            Reg::ReduceDataBlock<Reg::ReduceType::MAX>(xMaxExp, xMaxExp, maskAll);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg, blockPerReg);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeMaxExpClip(__ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T); // 每个向量寄存器可以存储的元素个数
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch(); // Reduce操作后搬出的元素个数
    uint16_t loopNum = CeilDiv(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            Reg::And((Reg::RegTensor<uint16_t> &)vdExp0, (Reg::RegTensor<uint16_t> &)vdExp0, absMask16Bit, Mask);
            Reg::And((Reg::RegTensor<uint16_t> &)vdExp1, (Reg::RegTensor<uint16_t> &)vdExp1, absMask16Bit, Mask);
            Reg::Max(vdMaxExp, (Reg::RegTensor<uint16_t> &)vdExp0, (Reg::RegTensor<uint16_t> &)vdExp1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, ureg,
                                                                            elementAfterReduce);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                    __ubuf__ uint16_t *recipScaleLocalAddr, uint32_t scaleNum)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(uint16_t);
    uint32_t vlB16Pack = vlB16 / 2;
    uint16_t loopTimes = CeilDiv(scaleNum, vlB16);
    uint16_t maxExponent;
    if constexpr (Std::IsSame<T, fp8_e4m3fn_t>::value) {
        maxExponent = FP8_E4M3_MAX_EXP;
    } else if constexpr (Std::IsSame<T, fp8_e5m2_t>::value) {
        maxExponent = FP8_E5M2_MAX_EXP;
    } else if constexpr (Std::IsSame<T, fp4x2_e2m1_t>::value) {
        maxExponent = FP4_E2M1_BF16_MAX_EXP;
    } else {
        maxExponent = FP4_E1M2_BF16_MAX_EXP;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> recipScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, maxExponent);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::Duplicate(fp8NanU16, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::Duplicate(zeroU16, 0);
        Reg::RegTensor<uint16_t> nanU16;
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < loopTimes; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(scaleNum);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, vlB16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, xMaxExp, zeroU16, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(xMaxExp, maxExpValue, xMaxExp, invalidDataMask);
            Reg::Sub(sharedExp, xMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);
            Reg::Select<uint16_t>(scaleValue, scaleValue, zeroU16, zeroMask);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlB16Pack, preMaskScale); // 128 个scale，占用 128 * 1 Bytes

            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(recipScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(recipScale, recipScale, nanU16, cmpResult);
            Reg::Select<uint16_t>(recipScale, recipScale, zeroU16, zeroMask);
            Reg::Select<uint16_t>(recipScale, specialExpU16, recipScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(recipScaleLocalAddr, recipScale, vlB16,
                                                                          preMaskScale);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeScaleClip(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(uint32_t);
    uint16_t loopNumScale = CeilDiv(totalScaleInUB, vlForHalfNumber);
    uint32_t dtypeMax;
    if constexpr (Std::IsSame<T, fp8_e4m3fn_t>::value) {
        dtypeMax = FP8_E4M3_MAX;
    } else if constexpr (Std::IsSame<T, fp8_e5m2_t>::value) {
        dtypeMax = FP8_E5M2_MAX;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> max16;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> normalExp32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint16_t> expOut;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> recExpOut;

        Reg::RegTensor<uint32_t> invMax;
        Reg::Duplicate(invMax, dtypeMax);
        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        Reg::RegTensor<uint32_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        Reg::RegTensor<uint32_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        Reg::RegTensor<uint32_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::RegTensor<uint32_t> epsTensor;
        Reg::Duplicate(epsTensor, EPS_1E_4_FP32);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        // Reg::MaskReg maskHalf = Reg::CreateMask<uint16_t>();
        uint32_t B16_HALF_MASK_ELEMENT_NUM = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(B16_HALF_MASK_ELEMENT_NUM);
        Reg::MaskReg maskFloat = Reg::CreateMask<uint32_t>();

        static constexpr Reg::CastTrait castTraitHalf2Float = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale; i++) {
            // 单搬 64 个数
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                max16, maxExpAddr, vlForHalfNumber);

            Reg::Cast<float, U, castTraitHalf2Float>((Reg::RegTensor<float> &)max32, (Reg::RegTensor<U> &)max16,
                                                     maskFloat);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, maskFloat);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, maskFloat);

            // 1e-4 下限 clamp：抑制 block 内 max_abs 过小导致 scale 噪声放大
            // 注意：cmpResult/zeroMask 已在 clamp 前用原始 max32 算好，特殊路径不受影响
            Reg::Max((Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)epsTensor,
                     maskFloat);
            // 等价的整数 Max 版本（max32 来自 |x|，非负 IEEE 浮点 bit pattern 与值序一致）：
            // Reg::Max(max32, max32, epsTensor, maskFloat);

            Reg::Mul((Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)invMax,
                     maskFloat);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskFloat);
            Reg::And(man32, max32, manMaskFP32, maskFloat);

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, FP32_BIASED_EXP_MAX_NORMAL, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_ZERO, maskFloat);
            Reg::MaskAnd(p0, p0, p1, maskFloat);
            Reg::MaskAnd(p0, p0, p2, maskFloat);

            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, FP32_MANTISSA_HALF, maskFloat);
            Reg::MaskAnd(p1, p1, p2, maskFloat);
            Reg::MaskOr(p0, p0, p1, maskFloat);

            Reg::Adds(expAddOne32, exp32, 1, maskFloat);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);
            // 64 个 max 计算得到 64 * 1 Bytes = 32 * sizeof(uint16_t) Bytes
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + i * 32, expOut,
                                                                     dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskFloat);
            Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
            Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(halfScaleLocalAddr + i * vlForHalfNumber, recExpOut, dataMaskB16Half);
        }
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeFp8Data(__ubuf__ T *srcAddr, __ubuf__ uint16_t *recipScaleLocalAddr,
                                      __ubuf__ int8_t *outLocalAddr, uint32_t eleNum)
{
    uint32_t vlB16 = GetVecLen() / sizeof(uint16_t);
    uint32_t vlB16Double = vlB16 * 2;
    uint16_t blockPerReg = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopTimes = CeilDiv(eleNum, vlB16Double);
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg dataMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<float> floatScaleForMul;
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<U> x0ZeroFP8;
        Reg::RegTensor<U> x0OneFP8;
        Reg::RegTensor<U> x1ZeroFP8;
        Reg::RegTensor<U> x1OneFP8;

        static constexpr Reg::CastTrait castTraitB16ToB32Zero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitB16ToB32One = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                                Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitB32ToB80 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitB32ToB81 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitB32ToB82 = {Reg::RegLayout::TWO, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitB32ToB83 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < loopTimes; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, srcAddr,
                                                                                                  vlB16Double);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, blockPerReg);
            if constexpr (Std::IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitB16ToB32Zero>(
                    floatScaleForMul, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, maskAll);

                Reg::Cast<float, T, castTraitB16ToB32Zero>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, T, castTraitB16ToB32One>(x0OneFP32, x0, maskAll);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, floatScaleForMul, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, floatScaleForMul, maskAll);

                Reg::Cast<float, T, castTraitB16ToB32Zero>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, T, castTraitB16ToB32One>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, floatScaleForMul, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, floatScaleForMul, maskAll);
            } else if constexpr (Std::IsSame<T, bfloat16_t>::value) {
                Reg::Mul(x0, x0, (Reg::RegTensor<T> &)halfScaleForMul, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<T> &)halfScaleForMul, maskAll);

                Reg::Cast<float, T, castTraitB16ToB32Zero>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, T, castTraitB16ToB32One>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, T, castTraitB16ToB32Zero>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, T, castTraitB16ToB32One>(x1OneFP32, x1, maskAll);
            }

            Reg::Cast<U, float, castTraitB32ToB80>(x0ZeroFP8, x0ZeroFP32, maskAll);
            Reg::Cast<U, float, castTraitB32ToB81>(x1ZeroFP8, x1ZeroFP32, maskAll);
            Reg::Cast<U, float, castTraitB32ToB82>(x0OneFP8, x0OneFP32, maskAll);
            Reg::Cast<U, float, castTraitB32ToB83>(x1OneFP8, x1OneFP32, maskAll);

            Reg::Add((Reg::RegTensor<uint8_t> &)x0ZeroFP8, (Reg::RegTensor<uint8_t> &)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t> &)x0OneFP8, maskAll);
            Reg::Add((Reg::RegTensor<uint8_t> &)x1ZeroFP8, (Reg::RegTensor<uint8_t> &)x1ZeroFP8,
                     (Reg::RegTensor<uint8_t> &)x1OneFP8, maskAll);
            Reg::Add((Reg::RegTensor<uint8_t> &)x0ZeroFP8, (Reg::RegTensor<uint8_t> &)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t> &)x1ZeroFP8, maskAll);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)x0ZeroFP8, vlB16Double, dataMask);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void FP16Convert(Reg::RegTensor<half> &output, Reg::RegTensor<half> &input, Reg::MaskReg &mask)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> specialValueTensor;
        Reg::RegTensor<uint16_t> newMantissa;
        Reg::RegTensor<uint16_t> andResult;
        Reg::RegTensor<uint16_t> newValue;
        Reg::MaskReg specialMask;
        Reg::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (Std::IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        Reg::Duplicate(specialValueTensor, specialValue);
        Reg::Duplicate(newMantissa, NEW_MANTISSA);
        Reg::And(andResult, (Reg::RegTensor<uint16_t> &)input, specialValueTensor, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        Reg::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        Reg::Or(newValue, (Reg::RegTensor<uint16_t> &)input, newMantissa, mask);
        Reg::Select<uint16_t>((Reg::RegTensor<uint16_t> &)output, newValue, (Reg::RegTensor<uint16_t> &)input,
                              specialMask);
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeFp4Data(__ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
                                      __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopNum = CeilDiv(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;

        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;

        Reg::RegTensor<U> vdExp0FP4;
        Reg::RegTensor<U> vdExp1FP4;

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                (Reg::RegTensor<uint32_t> &)vdExp0, (Reg::RegTensor<uint32_t> &)vdExp1, (__ubuf__ uint32_t *&)srcAddr,
                vlForHalfNumber);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (Std::IsSame<T, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT) {
                    FP16Convert<T, U>(vdExp0, vdExp0, dataMask1);
                    FP16Convert<T, U>(vdExp1, vdExp1, dataMask1);
                }
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                Reg::Mul(vdExp0BF16, vdExp0BF16, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                Reg::Mul(vdExp1BF16, vdExp1BF16, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                Reg::Cast<U, bfloat16_t, castTraitZero>(vdExp0FP4, vdExp0BF16, dataMask1);
                Reg::Cast<U, bfloat16_t, castTraitOne>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Cast<U, T, castTraitZero>(vdExp0FP4, vdExp0, dataMask1);
                Reg::Cast<U, T, castTraitOne>(vdExp1FP4, vdExp1, dataMask1);
            }
            Reg::Add((Reg::RegTensor<uint8_t> &)vdExp0FP4, (Reg::RegTensor<uint8_t> &)vdExp0FP4,
                     (Reg::RegTensor<uint8_t> &)vdExp1FP4, dataMask1);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp0FP4, OUT_ELE_NUM_PER_LOOP_FP4, dataMask1);
        }
    }
}

template <typename T, typename U, RoundMode RMode, bool HasSmooth>
__aicore__ inline void ComputePerTileDynamic(__ubuf__ T *srcAddr, __ubuf__ float *smoothLocalAddr,
                                             __ubuf__ float *scaleOutLocalAddr, __ubuf__ int8_t *outLocalAddr,
                                             uint32_t totalCountInUB)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(T);
    uint32_t vlB32 = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = CeilDiv(totalCountInUB, vlB16);
    uint32_t totalCntForB32 = totalCountInUB;
    float maxVal = 0.0f;
    float invMaxVal = 0.0f;
    if constexpr (Std::IsSame<U, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
        invMaxVal = 1.0f / FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
        invMaxVal = 1.0f / FP8_E4M3_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, hifloat8_t>::value) {
        maxVal = HIFP8_MAX_VALUE;
        invMaxVal = 1.0f / HIFP8_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, int8_t>::value) {
        maxVal = INT8_MAX_VALUE;
        invMaxVal = 1.0f / INT8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1;
        Reg::MaskReg dataMask2;
        Reg::MaskReg dataMask3;
        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();

        Reg::RegTensor<T> vInB16;

        Reg::RegTensor<float> vInFP32Zero;
        Reg::RegTensor<float> vInFP32One;
        Reg::RegTensor<float> vSmooth0;
        Reg::RegTensor<float> vSmooth1;
        Reg::RegTensor<float> vTileMax;
        Reg::RegTensor<float> vDynScale;
        Reg::RegTensor<float> vOutputScale;
        Reg::RegTensor<float> vMaxVal;
        Reg::RegTensor<float> vInvMaxVal;

        Reg::RegTensor<U> vOut0;
        Reg::RegTensor<U> vOut1;

        Reg::Duplicate(vMaxVal, maxVal);
        Reg::Duplicate(vInvMaxVal, invMaxVal);

        static constexpr Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, true};

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static Reg::CastTrait castTrait32tof8 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, RMode};

        constexpr static Reg::CastTrait castTrait32tof16 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        constexpr static Reg::CastTrait castTrait16toi8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = Reg::UpdateMask<float>(totalCntForB32);
            dataMask3 = Reg::UpdateMask<float>(totalCntForB32);

            Reg::DataCopy(vInB16, srcAddr + i * vlB16);
            Reg::Cast<float, T, castTraitZero>(vInFP32Zero, vInB16, dataMask1);
            Reg::Cast<float, T, castTraitOne>(vInFP32One, vInB16, dataMask1);
            if constexpr (HasSmooth) {
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    vSmooth0, vSmooth1, smoothLocalAddr, vlB32 * DIGIT_TWO);
                Reg::Mul(vInFP32Zero, vInFP32Zero, vSmooth0, maskAll);
                Reg::Mul(vInFP32One, vInFP32One, vSmooth1, maskAll);
            }
            Reg::Interleave(vSmooth0, vSmooth1, vInFP32Zero, vInFP32One);
            Reg::Abs(vInFP32Zero, vSmooth0, maskAll);
            Reg::Abs(vInFP32One, vSmooth1, maskAll);
            Reg::Max(vTileMax, vInFP32Zero, vInFP32One, maskAll);
            Reg::ReduceMax(vTileMax, vTileMax, dataMask2);
            Reg::Duplicate(vTileMax, vTileMax, maskAll);
            Reg::Div<float, &divMode>(vDynScale, vMaxVal, vTileMax, maskAll);
            Reg::Mul(vSmooth0, vSmooth0, vDynScale, maskAll);
            Reg::Mul(vSmooth1, vSmooth1, vDynScale, maskAll);

            if constexpr (Std::IsSame<U, int8_t>::value) {
                Reg::RegTensor<half> vHalf0;
                Reg::RegTensor<half> vHalf1;
                Reg::Cast<half, float, castTrait32tof16>(vHalf0, vSmooth0, maskAll);
                Reg::Cast<half, float, castTrait32tof16>(vHalf1, vSmooth1, maskAll);
                Reg::Cast<U, half, castTrait16toi8>(vOut0, vHalf0, maskAll);
                Reg::Cast<U, half, castTrait16toi8>(vOut1, vHalf1, maskAll);
            } else {
                Reg::Cast<U, float, castTrait32tof8>(vOut0, vSmooth0, dataMask2);
                Reg::Cast<U, float, castTrait32tof8>(vOut1, vSmooth1, dataMask3);
            }

            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vOut0, OUT_ELE_NUM_ONE_BLK, dataMask2);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vOut1, OUT_ELE_NUM_ONE_BLK, dataMask3);

            Reg::Mul(vOutputScale, vTileMax, vInvMaxVal, maskOne);
            Reg::DataCopy<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOutLocalAddr + i, vOutputScale, maskOne);
        }
    }
}

template <typename T, bool HasSmooth>
__aicore__ inline void ComputePerTileGroupMax(__ubuf__ T *srcAddr, __ubuf__ float *smoothLocalAddr,
                                              __ubuf__ float *groupScaleLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(T);
    uint32_t vlB32 = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = CeilDiv(totalCountInUB, vlB16);
    uint32_t totalCntForB32 = totalCountInUB;

    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<T> vInB16;
        Reg::RegTensor<float> vInFP32Zero;
        Reg::RegTensor<float> vInFP32One;
        Reg::RegTensor<float> vSmooth0;
        Reg::RegTensor<float> vSmooth1;
        Reg::RegTensor<float> vTileMax;

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::DataCopy(vInB16, srcAddr + i * vlB16);
            Reg::Cast<float, T, castTraitZero>(vInFP32Zero, vInB16, maskAll);
            Reg::Cast<float, T, castTraitOne>(vInFP32One, vInB16, maskAll);
            if constexpr (HasSmooth) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vSmooth0, vSmooth1,
                                                                     smoothLocalAddr + i * vlB32 * DIGIT_TWO);
                Reg::Mul(vInFP32Zero, vInFP32Zero, vSmooth0, maskAll);
                Reg::Mul(vInFP32One, vInFP32One, vSmooth1, maskAll);
            }
            Reg::Abs(vInFP32Zero, vInFP32Zero, maskAll);
            Reg::Abs(vInFP32One, vInFP32One, maskAll);
            Reg::Max(vTileMax, vInFP32Zero, vInFP32One, maskAll);
            Reg::ReduceMax(vTileMax, vTileMax, maskAll);
            Reg::DataCopy<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(groupScaleLocalAddr + i, vTileMax, maskAll);
        }
    }
}

template <typename U>
__aicore__ inline void ComputePerTileGroupScale(__ubuf__ float *groupScaleLocalAddr, __ubuf__ float *scaleOutLocalAddr,
                                                uint16_t groupNum)
{
    uint32_t groupNumPerLoop = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = CeilDiv(static_cast<uint32_t>(groupNum), groupNumPerLoop);
    uint32_t remainingGroupNum = groupNum;
    float maxVal = 0.0f;
    if constexpr (Std::IsSame<U, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
    }
    float invMaxVal = 1.0f / maxVal;

    __VEC_SCOPE__
    {
        Reg::MaskReg groupMask;
        Reg::RegTensor<float> vTileMax;
        Reg::RegTensor<float> vDynScale;
        Reg::RegTensor<float> vOutputScale;
        Reg::RegTensor<float> vMaxVal;
        Reg::RegTensor<float> vInvMaxVal;

        Reg::Duplicate(vMaxVal, maxVal);
        Reg::Duplicate(vInvMaxVal, invMaxVal);
        static constexpr Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, true};

        // 一个FP32向量的不同lane分别计算不同group的高精度量化系数。
        for (uint16_t i = 0; i < loopNum; i++) {
            groupMask = Reg::UpdateMask<float>(remainingGroupNum);
            uint32_t groupOffset = i * groupNumPerLoop;
            Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(vTileMax, groupScaleLocalAddr + groupOffset);
            Reg::Div<float, &divMode>(vDynScale, vMaxVal, vTileMax, groupMask);
            Reg::Mul(vOutputScale, vTileMax, vInvMaxVal, groupMask);
            Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(groupScaleLocalAddr + groupOffset, vDynScale,
                                                                groupMask);
            Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(scaleOutLocalAddr + groupOffset, vOutputScale,
                                                                groupMask);
        }
    }
}

template <typename T, typename U, RoundMode RMode, bool HasSmooth>
__aicore__ inline void QuantizePerTileWithGroupScale(__ubuf__ T *srcAddr, __ubuf__ float *smoothLocalAddr,
                                                     __ubuf__ float *groupScaleLocalAddr, __ubuf__ int8_t *outLocalAddr,
                                                     uint32_t totalCountInUB)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(T);
    uint32_t vlB32 = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = CeilDiv(totalCountInUB, vlB16);

    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<T> vInB16;
        Reg::RegTensor<float> vInFP32Zero;
        Reg::RegTensor<float> vInFP32One;
        Reg::RegTensor<float> vSmooth0;
        Reg::RegTensor<float> vSmooth1;
        Reg::RegTensor<float> vDynScale;
        Reg::RegTensor<U> vOut0;
        Reg::RegTensor<U> vOut1;

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        constexpr static Reg::CastTrait castTrait32tof80 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                            Reg::MaskMergeMode::ZEROING, RMode};
        constexpr static Reg::CastTrait castTrait32tof81 = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                            Reg::MaskMergeMode::ZEROING, RMode};

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::DataCopy(vInB16, srcAddr + i * vlB16);
            Reg::Cast<float, T, castTraitZero>(vInFP32Zero, vInB16, maskAll);
            Reg::Cast<float, T, castTraitOne>(vInFP32One, vInB16, maskAll);
            if constexpr (HasSmooth) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vSmooth0, vSmooth1,
                                                                     smoothLocalAddr + i * vlB32 * DIGIT_TWO);
                Reg::Mul(vInFP32Zero, vInFP32Zero, vSmooth0, maskAll);
                Reg::Mul(vInFP32One, vInFP32One, vSmooth1, maskAll);
            }
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(vDynScale, groupScaleLocalAddr + i);
            Reg::Mul(vInFP32Zero, vInFP32Zero, vDynScale, maskAll);
            Reg::Mul(vInFP32One, vInFP32One, vDynScale, maskAll);

            Reg::Cast<U, float, castTrait32tof80>(vOut0, vInFP32Zero, maskAll);
            Reg::Cast<U, float, castTrait32tof81>(vOut1, vInFP32One, maskAll);
            Reg::Add((Reg::RegTensor<int8_t> &)vOut0, (Reg::RegTensor<int8_t> &)vOut0, (Reg::RegTensor<int8_t> &)vOut1,
                     maskAll);

            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vOut0, OUT_ELE_NUM_ONE_BLK * 2, maskAll);
        }
    }
}

template <typename T, typename U, RoundMode RMode, bool HasSmooth>
__aicore__ inline void ComputePerTileDynamicBatchScale(__ubuf__ T *srcAddr, __ubuf__ float *smoothLocalAddr,
                                                       __ubuf__ float *groupScaleLocalAddr,
                                                       __ubuf__ float *scaleOutLocalAddr, __ubuf__ int8_t *outLocalAddr,
                                                       uint32_t totalCountInUB)
{
    uint16_t groupNum =
        static_cast<uint16_t>(CeilDiv(totalCountInUB, static_cast<uint32_t>(GetVRegSizeDispatch() / sizeof(T))));
    ComputePerTileGroupMax<T, HasSmooth>(srcAddr, smoothLocalAddr, groupScaleLocalAddr, totalCountInUB);
    ComputePerTileGroupScale<U>(groupScaleLocalAddr, scaleOutLocalAddr, groupNum);
    QuantizePerTileWithGroupScale<T, U, RMode, HasSmooth>(srcAddr, smoothLocalAddr, groupScaleLocalAddr, outLocalAddr,
                                                          totalCountInUB);
}

} // namespace Quant
#endif
