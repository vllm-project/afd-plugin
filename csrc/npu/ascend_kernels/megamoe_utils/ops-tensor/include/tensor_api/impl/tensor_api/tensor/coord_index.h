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
#warning                                                                                                               \
    "impl/tensor_api/tensor/coord_index.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file coord_index.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_COORD_INDEX_H
#define IMPL_TENSOR_API_TENSOR_COORD_INDEX_H

#include "impl/tensor_api/tensor/layout_definition.h"
#include "impl/tensor_api/utils/npu_debug_check.h"

namespace asc {
namespace te {

template <typename Coord, typename Shape, typename Stride>
__aicore__ inline constexpr auto crd2idx(const Coord& coord, const Shape& shape, const Stride& stride);

template <typename Coord, typename Shape, typename Stride, size_t... Is>
__aicore__ inline constexpr auto crd2idx_ttt(const Coord& coord, const Shape& shape, const Stride& stride,
                                             Std::index_sequence<Is...>)
{
    return (... + crd2idx(Std::get<Is>(coord), Std::get<Is>(shape), Std::get<Is>(stride)));
}

template <typename Coord, typename Shape, typename Stride, size_t I0, size_t... Is>
__aicore__ inline constexpr auto crd2idx_itt(const Coord& coord, const Shape& shape, const Stride& stride,
                                             Std::index_sequence<I0, Is...>)
{
    if constexpr (sizeof...(Is) == 0) { // Avoid recursion and mod on single/last iter
        return crd2idx(coord, Std::get<I0>(shape), Std::get<I0>(stride));
    } else if constexpr (Std::is_constant<0, Coord>::value) {
        return crd2idx(_0{}, Std::get<I0>(shape), Std::get<I0>(stride))
               + (_0{} + ... + crd2idx(_0{}, Std::get<Is>(shape), Std::get<Is>(stride)));
    } else { // General case
        auto prod = product{}(Std::get<I0>(shape));
        auto div = coord / prod;
        auto mod = coord % prod;
        return crd2idx(mod, Std::get<I0>(shape), Std::get<I0>(stride))
               + crd2idx_itt(div, shape, stride, Std::index_sequence<Is...>{});
    }
}

template <typename Coord, typename Shape, typename Stride>
__aicore__ inline constexpr auto crd2idx(const Coord& coord, const Shape& shape, const Stride& stride)
{
    TENSOR_API_DEBUG_CHECK(debug_check_coord_shape, shape, coord, "crd2idx");
    if constexpr (Std::is_tuple_v<Coord>) {
        if constexpr (Std::is_tuple_v<Shape>) { // tuple tuple tuple
            static_assert(Std::tuple_size_v<Coord> == Std::tuple_size_v<Shape>, "Shape and Coord Mismatched Ranks");
            static_assert(Std::tuple_size_v<Coord> == Std::tuple_size_v<Stride>, "Stride and Coord Mismatched Ranks");
            return crd2idx_ttt(coord, shape, stride, tuple_sequence<Coord>{});
        } else { // tuple "int" "int"
            static_assert(sizeof(Coord) == 0, "Invalid parameters, Shape is not tuple!");
        }
    } else {
        if constexpr (Std::is_tuple_v<Shape>) { // "int" tuple tuple
            static_assert(Std::tuple_size_v<Shape> == Std::tuple_size_v<Stride>, "Shape and Stride Mismatched Ranks");
            return crd2idx_itt(coord, shape, stride, tuple_sequence<Shape>{});
        } else { // "int" "int" "int"
            return coord * stride;
        }
    }
}

template <typename Shape, typename Stride, typename Info>
template <typename Coord>
__aicore__ inline constexpr auto layout<Shape, Stride, Info>::operator()(const Coord& coord) const
{
    return crd2idx(coord, shape(), stride());
}

template <typename Coord, typename LayoutType, typename = Std::enable_if_t<is_layout_v<LayoutType>>>
__aicore__ inline constexpr auto crd2idx(const Coord& coord, const LayoutType& layout)
{
    TENSOR_API_DEBUG_CHECK(debug_check_coord, layout, coord, "crd2idx");
    return crd2idx(coord, layout.shape(), layout.stride());
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_COORD_INDEX_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
