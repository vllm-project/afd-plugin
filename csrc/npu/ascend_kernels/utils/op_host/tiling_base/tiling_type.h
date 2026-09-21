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
 * \file tiling_type.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace optiling {

enum class AxisEnum {
    B = 0,
    N2 = 1,
    G = 2,
    S1 = 3,
    S2 = 4,
    D = 5,
    NONE = 9,
};

enum class DtypeEnum {
    FLOAT16 = 0,
    FLOAT32 = 1,
    BFLOAT16 = 2,
    FLOAT16_PRECISION = 3,
};

enum class PerformanceOrientedEnum {
    BIG_BUFFER = 1,
    BIG_DOUBLE_BUFFER = 2,
};

enum class MatmulConfig {
    NULL_CONFIG = 0,
    NORMAL_CONFIG = 1,
    MDL_CONFIG = 2
};

enum class PseConfig {
    NO_PSE = 0,
    EXIST_PSE = 1
};

enum class AttenMaskConfig {
    NO_ATTEN_MASK = 0,
    EXIST_ATTEN_MASK = 1
};

enum class DropOutConfig {
    NO_DROP_OUT = 0,
    EXIST_DROP_OUT = 1
};

enum class CubeFormatEnum {
    ND = 0,
    NZ = 1
};
enum class LayoutEnum {
    BSND = 0,
    SBND = 1,
    BNSD = 2,
    TND = 3,
    NTD_TND = 4
};

enum class CubeInputSourceEnum {
    GM = 0,
    L1 = 1
};

enum class OptionEnum {
    DISABLE = 0,
    ENABLE = 1
};

enum class SparseEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

constexpr uint64_t RecursiveSum()
{
    return 0;
}

constexpr int64_t base10Multiplier = 10;

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + base10Multiplier * RecursiveSum(templateIds...);
}

// Rules for generating the TilingKey:
// FlashAttentionScore/FlashAttentionScoreGrad assemble the tiling key from decimal digits, containing the following key
// parameters, ordered from low to high digits: Ub0, Ub1,
// Block, DataType, Format, Sparse, and the specialized template Ub0, Ub1:
//     Represents the axis of the Ub intra-core split, expressed using the AxisEnum enumeration. Since at most two axes
//     may be split, there exist UB0 and UB1; if there is no UB intra-core split, fill in AXIS_NONE. UB0 and UB1 each
//     occupy one decimal digit;
//     Block: represents the axis used by UB for core splitting, expressed using the AxisEnum enumeration, occupying one
//     decimal digit;
//     DataType: represents the input/output data types supported by the current tiling key, expressed using the
//     SupportedDtype enumeration, occupying one decimal digit
//     Format: represents the Format supported by the current tiling key, expressed using the InputLayout enumeration,
//     occupying one decimal digit
//     Sparse: represents whether the current tiling key supports Sparse, expressed using the SparseCapability
//     enumeration, occupying one decimal digit
//     For other specialized scenarios, define your own bit fields and values
// usage: get tilingKey from inputed types
//     uint64_t tilingKey = GET_FLASHATTENTION_TILINGKEY(AxisEnum::AXIS_S1, AxisEnum::AXIS_S2, AxisEnum::AXIS_N2,
//                                     SupportedDtype::FLOAT32, InputLayout::BSH, SparseCapability::SUPPORT_ALL)

constexpr uint64_t TILINGKEYOFFSET = uint64_t(10000000000000000000UL); // 10^19
template <typename... Args> constexpr uint64_t GET_TILINGKEY(Args... templateIds)
{
    return TILINGKEYOFFSET + RecursiveSum(templateIds...);
}

// usage: get tilingKey from inputed types
//     uint64_t tilingKey = TILINGKEY(S2, S1, N2, FLOAT32, BSND, ALL)

#define TILINGKEY(ub2, ub1, block, dtype, layout, sparse)                                                              \
    (GET_TILINGKEY(AxisEnum::ub2, AxisEnum::ub1, AxisEnum::block, DtypeEnum::dtype, LayoutEnum::layout,                \
                   SparseEnum::sparse))

} // namespace optiling
