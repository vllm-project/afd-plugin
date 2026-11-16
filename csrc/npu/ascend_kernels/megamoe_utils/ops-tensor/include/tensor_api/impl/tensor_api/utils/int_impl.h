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
 * \file int_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_INT_IMPL_H
#define IMPL_TENSOR_API_UTILS_INT_IMPL_H

#include "impl/utils/std/type_traits/integral_constant.h"
#include "impl/utils/std/type_traits/enable_if.h"
#include "impl/utils/std/type_traits/is_const.h"
#include "impl/utils/std/algorithm/max.h"
#include "impl/utils/std/algorithm/min.h"
#include "impl/utils/std/cmath/divide.h"
#include "impl/utils/std/cmath/ceil_division.h"
#include "impl/utils/std/cmath/ceil_align.h"
#include "impl/utils/std/cmath/sqrt.h"
#include "impl/utils/std/type_traits/is_integral.h"

namespace AscendC {
namespace Std {

#define STD_INT_NAMED_BINARY_FN(OP, CMP)                                                                               \
    template <auto t, auto u>                                                                                          \
    ASCENDC_HOST_AICORE inline constexpr Int<(CMP ? t : u)> OP(Int<t>, Int<u>)                                         \
    {                                                                                                                  \
        return {};                                                                                                     \
    }                                                                                                                  \
    template <auto t, typename Right, typename = enable_if_t<is_integral<Right>::value>>                               \
    ASCENDC_HOST_AICORE inline constexpr auto OP(Int<t>, const Right& u)                                               \
    {                                                                                                                  \
        return OP(static_cast<size_t>(t), u);                                                                          \
    }                                                                                                                  \
    template <typename Left, auto u, typename = enable_if_t<is_integral<Left>::value>>                                 \
    ASCENDC_HOST_AICORE inline constexpr auto OP(const Left& t, Int<u>)                                                \
    {                                                                                                                  \
        return OP(t, static_cast<size_t>(u));                                                                          \
    }

STD_INT_NAMED_BINARY_FN(max, t > u);
STD_INT_NAMED_BINARY_FN(min, t < u);
#undef STD_INT_NAMED_BINARY_FN

#define STD_INT_BINARY_FN(OP, EXPR)                                                                                    \
    template <auto t, auto u>                                                                                          \
    ASCENDC_HOST_AICORE inline constexpr Int<EXPR> OP(Int<t>, Int<u>)                                                  \
    {                                                                                                                  \
        return {};                                                                                                     \
    }                                                                                                                  \
    template <auto t, typename Right, typename = enable_if_t<is_integral<Right>::value>>                               \
    ASCENDC_HOST_AICORE inline constexpr auto OP(Int<t>, const Right& u)                                               \
    {                                                                                                                  \
        return OP(static_cast<size_t>(t), u);                                                                          \
    }                                                                                                                  \
    template <typename Left, auto u, typename = enable_if_t<is_integral<Left>::value>>                                 \
    ASCENDC_HOST_AICORE inline constexpr auto OP(const Left& t, Int<u>)                                                \
    {                                                                                                                  \
        return OP(t, static_cast<size_t>(u));                                                                          \
    }

STD_INT_BINARY_FN(divide, (t / u));
STD_INT_BINARY_FN(ceil_division, ((t + u - 1) / u));
STD_INT_BINARY_FN(ceil_align, (((t + u - 1) / u) * u));
#undef STD_INT_BINARY_FN

} // namespace Std
} // namespace AscendC
#endif // IMPL_TENSOR_API_UTILS_INT_IMPL_H
