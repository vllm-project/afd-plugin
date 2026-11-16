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
 * \file math_util.h
 * \brief common math opeator funcions
 */

#ifndef OP_COMMON_OP_HOST_UTIL_MATH_UTIL_H
#define OP_COMMON_OP_HOST_UTIL_MATH_UTIL_H

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include "opbase_export.h"

namespace Ops {
namespace Base {
/**
 * if y is 0, return x
 */
template <typename T>
auto FloorDiv(T x, T y) -> typename std::enable_if<std::is_integral<T>::value, T>::type
{
    return y == 0 ? x : x / y;
}

/**
 * if align is 0, return 0
 */
template <typename T>
auto FloorAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
{
    return align == 0 ? 0 : x / align * align;
}

/**
 * if y is 0, return x
 */
template <typename T>
auto CeilDiv(T x, T y) -> typename std::enable_if<std::is_signed<T>::value, T>::type
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
    }

    return x;
}

/**
 * if y is 0, return x
 */
template <typename T>
auto CeilDiv(T x, T y) -> typename std::enable_if<std::is_unsigned<T>::value, T>::type
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0) ? (quotient + 1) : quotient;
    }

    return x;
}

/**
 * if align is 0, return 0
 */
template <typename T>
auto CeilAlign(T x, T align) -> typename std::enable_if<std::is_integral<T>::value, T>::type
{
    if (align == 0) {
        return 0;
    }
    T div = CeilDiv(x, align);
    if (div > std::numeric_limits<T>::max() / align) {
        return std::numeric_limits<T>::max();
    }
    return div * align;
}

template <typename T>
auto IsFloatEqual(T a, T b) -> typename std::enable_if<std::is_floating_point<T>::value, bool>::type
{
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    if (std::isinf(a) || std::isinf(b)) {
        return std::signbit(a) == std::signbit(b);
    }
    return fabs(a - b) <= std::numeric_limits<T>::epsilon();
}

OPBASE_API uint64_t LastPow2(uint64_t n);

OPBASE_API uint32_t MurmurHash(const void* src, uint32_t srcLen, uint32_t seed = 271828);

struct OPBASE_API SplitResult {
    int32_t splitCount = 0;
    int64_t splitFactor = 0;
    int64_t splitTailFactor = 0;
};

OPBASE_API bool SplitIntoEqualByParts(int64_t splitLen, int32_t parts, SplitResult& splitResult);

OPBASE_API bool SplitIntoEqualByFactor(int64_t splitLen, int32_t factor, SplitResult& splitResult);
} // namespace Base
} // namespace Ops

#endif // OP_COMMON_OP_HOST_UTIL_MATH_UTIL_H
