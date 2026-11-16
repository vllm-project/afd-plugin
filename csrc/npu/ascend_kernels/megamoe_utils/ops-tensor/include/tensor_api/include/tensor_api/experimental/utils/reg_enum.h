/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REG_ENUM_H
#endif

/*!
 * \file reg_enum.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_EXPERIMENTAL_UTILS_REG_ENUM_H
#define INCLUDE_TENSOR_API_EXPERIMENTAL_UTILS_REG_ENUM_H

#include <cstdint>

namespace asc {
namespace te {
namespace experimental {

/** @brief Data layout selected by a width-changing register cast. */
enum class cast_layout {
    unknown = -1,
    zero,
    one,
    two,
    three,
};

/** @brief Rounding behavior for register cast. */
enum class cast_round_mode {
    unknown = -1,
    rint,
    round,
    floor,
    ceil,
    trunc,
};

/** @brief Saturation behavior for register cast. */
enum class cast_sat_mode {
    unknown = -1,
    not_sat,
    sat,
};

struct cast_options {
    cast_layout layout = cast_layout::zero;
    cast_round_mode round = cast_round_mode::rint;
    cast_sat_mode sat = cast_sat_mode::not_sat;
};

enum class load_sideband_mode : uint8_t {
    direct,
    deintlv,
    downsample,
    unpack,
    unpack4,
    upsample,
};

enum class store_sideband_mode : uint8_t {
    direct,
    store_1st,
    intlv,
    pack,
    pack_quarter,
};

enum class broadcast_mode : uint8_t {
    elem,
    datablock,
    elem2datablock,
};

enum class mask_pattern : uint8_t {
    all,
    vl1,
    vl2,
    vl3,
    vl4,
    vl8,
    vl16,
    vl32,
    vl64,
    vl128,
    every3,
    every4,
    half,
    quarter,
    none,
};

enum class reduce_scope : uint8_t { element, datablock, pair };

enum class histogram_half {
    low,
    high,
};
 
enum class histogram_mode {
    frequency,
    cumulative,
};
} // namespace experimental
} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_EXPERIMENTAL_UTILS_REG_ENUM_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REG_ENUM_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_REG_ENUM_H
#endif
