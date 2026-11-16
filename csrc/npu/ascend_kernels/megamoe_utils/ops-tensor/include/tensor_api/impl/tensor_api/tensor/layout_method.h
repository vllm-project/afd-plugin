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
    "impl/tensor_api/tensor/layout_method.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_method.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_METHOD_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_METHOD_H

#include "impl/tensor_api/tensor/layout_definition.h"

namespace asc {
namespace te {

template <typename... Values>
struct has_zero_integral_constant
    : Std::bool_constant<(... || Std::is_same_v<Std::remove_cvref_t<Values>, Std::Int<0>>)> {};

template <typename FirstShape, typename... Shapes>
__aicore__ inline constexpr shape<FirstShape, Shapes...> make_shape(const FirstShape& first_shape,
                                                                    const Shapes&... shapes)
{
    static_assert(!has_zero_integral_constant<FirstShape, Shapes...>::value,
                  "MakeShape does not accept Int<0> arguments.");
    return {first_shape, shapes...};
}

template <typename FirstStride, typename... Strides>
__aicore__ inline constexpr stride<FirstStride, Strides...> make_stride(const FirstStride& first_stride,
                                                                        const Strides&... strides)
{
    return {first_stride, strides...};
}

template <typename FirstCoord, typename... Coords>
__aicore__ inline constexpr coord<FirstCoord, Coords...> make_coord(const FirstCoord& first_coord,
                                                                    const Coords&... coords)
{
    return {first_coord, coords...};
}

template <typename Shape, typename Stride>
__aicore__ inline constexpr auto make_layout(const Shape& shape, const Stride& stride)
{
    static_assert(Std::is_tuple_v<Shape> && Std::is_tuple_v<Stride>, "Shape or Stride is not tuple!");
    static_assert(nesting_depth_v<Shape> == nesting_depth_v<Stride>
                      && Std::tuple_size_v<Shape> == Std::tuple_size_v<Stride>,
                  "Shape and Stride structure are not compatible.");
    return layout<Shape, Stride>(shape, stride);
}

// shape = ((x1, x2, ..., xn), (y1, y2, ..., yn))
// stride[0][0] = 1; stride[0][i] = shape[0][i-1]*shape[1][i-1]*stride[0][i-1]; stride[1][i] = shape[0][i]*stride[0][i]
template <size_t I, typename Row, typename Col>
struct stride_row_elem {
    __aicore__ static inline constexpr auto value(const Row& row, const Col& col)
    {
        if constexpr (I == 0) {
            return _1{};
        } else {
            return Std::get<I - 1>(row) * Std::get<I - 1>(col) * stride_row_elem<I - 1, Row, Col>::value(row, col);
        }
    }
};

template <size_t I, typename Row, typename Col>
struct stride_col_elem {
    __aicore__ static inline constexpr auto value(const Row& row, const Col& col)
    {
        return Std::get<I>(row) * stride_row_elem<I, Row, Col>::value(row, col);
    }
};

template <typename Row, typename Col, size_t... Is>
__aicore__ inline constexpr auto build_stride_row_impl(const Row& row, const Col& col, Std::index_sequence<Is...>)
{
    return make_stride(stride_row_elem<Is, Row, Col>::value(row, col)...);
}

template <typename Row, typename Col, size_t... Is>
__aicore__ inline constexpr auto build_stride_col_impl(const Row& row, const Col& col, Std::index_sequence<Is...>)
{
    return make_stride(stride_col_elem<Is, Row, Col>::value(row, col)...);
}

template <typename ShapeType>
__aicore__ inline constexpr auto compute_stride(const ShapeType& shape)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::tuple_size_v<ShapeType> == 2,
                  "ShapeType must be tuple of two tuples");
    const auto& row = Std::get<0>(shape);
    const auto& col = Std::get<1>(shape);
    static_assert(Std::tuple_size_v<Std::remove_cvref_t<decltype(row)>>
                      == Std::tuple_size_v<Std::remove_cvref_t<decltype(col)>>,
                  "ShapeType rows must have same length");
    constexpr size_t n = Std::tuple_size_v<Std::remove_cvref_t<decltype(row)>>;
    using row_type = Std::remove_cvref_t<decltype(row)>;
    using col_type = Std::remove_cvref_t<decltype(col)>;
    auto stride0 = build_stride_row_impl(row, col, Std::make_index_sequence<n>{});
    auto stride1 = build_stride_col_impl(row, col, Std::make_index_sequence<n>{});
    return make_stride(stride0, stride1);
}

// shape = (x1, x2, x3, ..., xn) -> stride = (x2*x3*...*xn, ..., x_{n-1}*xn, xn, 1)
template <size_t I, typename ShapeType>
struct flat_stride_elem {
    __aicore__ static inline constexpr auto value(const ShapeType& shape)
    {
        constexpr size_t n = Std::tuple_size_v<ShapeType>;
        static_assert(n > 0, "ShapeType must not be empty");
        if constexpr (I == n - 1) {
            return _1{};
        } else {
            return flat_stride_elem<I + 1, ShapeType>::value(shape) * Std::get<I + 1>(shape);
        }
    }
};

template <typename ShapeType, size_t... Is>
__aicore__ inline constexpr auto build_flat_stride_impl(const ShapeType& shape, Std::index_sequence<Is...>)
{
    return make_stride(flat_stride_elem<Is, ShapeType>::value(shape)...);
}

template <typename ShapeType>
__aicore__ inline constexpr auto compute_flat_stride(const ShapeType& shape)
{
    static_assert(Std::is_tuple_v<ShapeType>, "ShapeType must be tuple");
    constexpr size_t n = Std::tuple_size_v<ShapeType>;
    return build_flat_stride_impl(shape, Std::make_index_sequence<n>{});
}

template <typename ShapeType>
__aicore__ inline constexpr auto make_layout(const ShapeType& shape)
{
    static_assert(Std::is_tuple_v<ShapeType>, "ShapeType is not tuple!");
    using elem_t = Std::remove_cvref_t<decltype(Std::get<0>(shape))>;
    if constexpr (Std::is_tuple_v<elem_t>) {
        return make_layout(shape, compute_stride(shape));
    } else {
        return make_layout(shape, compute_flat_stride(shape));
    }
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto get_shape(const Layout& layout)
{
    return layout.template shape<Is...>();
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto get_stride(const Layout& layout)
{
    return layout.template stride<Is...>();
}

struct coshape_sum {
    template <typename... Args>
    __aicore__ inline constexpr auto operator()(const Args&... args) const
    {
        return (_0{} + ... + args);
    }
};

struct coshape_compute {
    template <typename Shape, typename Stride>
    __aicore__ inline constexpr auto operator()(const Shape& shape, const Stride& stride) const
    {
        if constexpr (Std::is_tuple_v<Shape> && Std::is_tuple_v<Stride>) {
            static_assert(Std::tuple_size_v<Shape> == Std::tuple_size_v<Stride>, "Mismatched ranks");
            return transform_apply(shape, stride, coshape_compute{}, coshape_sum{});
        } else {
            auto m1_shape = shape - _1{};
            auto abs_stride = stride < 0 ? -stride : stride;
            return m1_shape * abs_stride;
        }
    }
};

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto coshape(const Layout& layout)
{
    auto shape = get_shape<Is...>(layout);
    auto stride = get_stride<Is...>(layout);
    auto co_coord = coshape_compute{}(shape, stride);
    return co_coord + _1{};
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto cosize(const Layout& layout)
{
    return tuple_size(coshape<Is...>(layout));
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto rank(const Layout& layout)
{
    return layout.template rank<Is...>();
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto select(const Layout& layout)
{
    return make_layout(select_tuple<Is...>(layout.shape()), select_tuple<Is...>(layout.stride()));
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto get(const Layout& layout)
{
    return make_layout(get_tuple<Is...>(layout.shape()), get_tuple<Is...>(layout.stride()));
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto size(const Layout& layout)
{
    return layout.template size<Is...>();
}

template <size_t... Is, typename Layout, typename>
__aicore__ inline constexpr auto capacity(const Layout& layout)
{
    return layout.template capacity<Is...>();
}

template <typename Tensor, typename Coord, typename Info>
__aicore__ inline decltype(auto) slice(Tensor&& tensor, const Coord& coord, const Info& info)
{
    return static_cast<Tensor&&>(tensor).slice(coord, info);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_METHOD_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
