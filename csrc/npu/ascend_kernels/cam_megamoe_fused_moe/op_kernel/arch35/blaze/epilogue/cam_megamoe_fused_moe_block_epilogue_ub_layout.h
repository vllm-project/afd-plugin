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
 * \file block_epilogue_ub_layout.h
 * \brief UB layout helpers for activation and MX quantization.
 */

#ifndef BLOCK_EPILOGUE_UB_LAYOUT_H
#define BLOCK_EPILOGUE_UB_LAYOUT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/megamoe_fused_moe_constants.h"

namespace MegamoeFusedMoeImpl {

struct ActivationMxQuantUbOffsets {
    uint32_t firstInputOffsetBytes;
    uint32_t firstInputElementCapacity;
    uint32_t secondInputOffsetBytes;
    uint32_t activationQuantElementCapacity;
    uint32_t activationOutputOffsetBytes;
    uint32_t quantOutputOffsetBytes;
    uint32_t quantScaleOffsetBytes;
    uint32_t maxExpOffsetBytes;
    uint32_t inverseMxScaleOffsetBytes;
    uint32_t scaleElementCapacity;
    uint32_t topkWeightOffsetBytes;
    uint32_t topkWeightElementCapacity;
};

template <typename InputType, uint32_t TileM, uint32_t TileN, bool IsInterleaved>
__aicore__ constexpr inline ActivationMxQuantUbOffsets BuildActivationMxQuantUbOffsets()
{
    constexpr uint32_t maxSingleElementCount = TileM * TileN;
    constexpr uint32_t activationQuantElementCapacity =
        IsInterleaved ? maxSingleElementCount / ACTIVATION_N_HALF : maxSingleElementCount;
    constexpr uint32_t activationOutputOffsetBytes = 0;
    constexpr uint32_t quantOutputOffsetBytes =
        activationOutputOffsetBytes + activationQuantElementCapacity * sizeof(bfloat16_t);
    constexpr uint32_t quantScaleOffsetBytes = quantOutputOffsetBytes + activationQuantElementCapacity * sizeof(int8_t);
    constexpr uint32_t scaleElementCapacity = activationQuantElementCapacity / AscendC::ONE_BLK_SIZE;
    constexpr uint32_t maxExpOffsetBytes = quantScaleOffsetBytes + scaleElementCapacity * sizeof(int8_t);
    constexpr uint32_t inverseMxScaleOffsetBytes = maxExpOffsetBytes + scaleElementCapacity * sizeof(uint16_t);
    constexpr uint32_t vecInEndBytes = maxSingleElementCount * sizeof(InputType) * 2U;
    constexpr uint32_t secondInputOffsetBytes =
        (vecInEndBytes <= 256U * 1024U) ? (maxSingleElementCount * sizeof(InputType)) : 0U;

    return {0U,
            maxSingleElementCount,
            secondInputOffsetBytes,
            activationQuantElementCapacity,
            activationOutputOffsetBytes,
            quantOutputOffsetBytes,
            quantScaleOffsetBytes,
            maxExpOffsetBytes,
            inverseMxScaleOffsetBytes,
            scaleElementCapacity,
            vecInEndBytes,
            TileM * INT32_PER_256B};
}

template <typename InputType>
struct ActivationMxQuantUbPointers {
    __ubuf__ InputType *firstInput;
    __ubuf__ InputType *secondInput;
    __ubuf__ bfloat16_t *activationOutput;
    __ubuf__ int8_t *quantOutput;
    __ubuf__ uint16_t *quantScale;
    __ubuf__ uint16_t *maxExp;
    __ubuf__ uint16_t *inverseMxScale;
    uint32_t selectedInt8BufferOffsetElements;
};

template <typename InputType, uint32_t MaxSingleElementCount, bool IsInterleaved>
__aicore__ inline ActivationMxQuantUbPointers<InputType> ResolveActivationMxQuantUbPointers(
    __ubuf__ InputType *firstInputBase, __ubuf__ InputType *secondInputBase, __ubuf__ bfloat16_t *activationOutputBase,
    __ubuf__ int8_t *quantOutputBase, __ubuf__ uint16_t *quantScaleBase, __ubuf__ uint16_t *maxExpBase,
    __ubuf__ uint16_t *inverseMxScaleBase, uint32_t validColumnCount, uint16_t pingpongIdx)
{
    constexpr uint32_t pongElementOfInput = MaxSingleElementCount;
    constexpr uint32_t pongElementOfBfloat16 = MaxSingleElementCount * sizeof(InputType) / sizeof(bfloat16_t);
    constexpr uint32_t pongElementOfInt8 = MaxSingleElementCount * sizeof(InputType);
    constexpr uint32_t pongElementOfUint16 = MaxSingleElementCount * sizeof(InputType) / sizeof(uint16_t);
    const uint32_t pongMultiplier = (IsInterleaved && pingpongIdx == 1U) ? 1U : 0U;

    ActivationMxQuantUbPointers<InputType> pointers{};
    pointers.firstInput = firstInputBase + pongMultiplier * pongElementOfInput;
    if constexpr (IsInterleaved) {
        pointers.secondInput = pointers.firstInput + validColumnCount;
    } else {
        pointers.secondInput = secondInputBase;
    }
    pointers.activationOutput = activationOutputBase + pongMultiplier * pongElementOfBfloat16;
    pointers.quantOutput = quantOutputBase + pongMultiplier * pongElementOfInt8;
    pointers.quantScale = quantScaleBase + pongMultiplier * pongElementOfUint16;
    pointers.maxExp = maxExpBase + pongMultiplier * pongElementOfUint16;
    pointers.inverseMxScale = inverseMxScaleBase + pongMultiplier * pongElementOfUint16;
    pointers.selectedInt8BufferOffsetElements = pongMultiplier * pongElementOfInt8;
    return pointers;
}

} // namespace MegamoeFusedMoeImpl

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_UB_LAYOUT_H
