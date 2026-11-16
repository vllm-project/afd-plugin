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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif

/*!
 * \file layout_interface.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_LAYOUT_INTERFACE_H
#define INCLUDE_TENSOR_API_TENSOR_LAYOUT_INTERFACE_H

#include "tensor_api/tensor/layout_type.h"
#include "impl/tensor_api/tensor/tuple_impl.h"
#include "impl/tensor_api/tensor/layout_definition.h"

namespace asc {
namespace te {

template <typename Layout>
using get_layout_pattern = typename get_pattern<Std::remove_cvref_t<Layout>>::type;

template <typename Coord, typename Shape, typename Stride>
__aicore__ inline constexpr auto crd2idx(const Coord& coord, const Shape& shape, const Stride& stride);

template <size_t I, size_t... Is, typename Tuple, typename Enable>
__aicore__ inline constexpr auto get(Tuple&& t);

// layout construction
template <typename FirstShape, typename... Shapes>
__aicore__ inline constexpr shape<FirstShape, Shapes...> make_shape(const FirstShape& first_shape,
                                                                    const Shapes&... shapes);

template <typename FirstStride, typename... Strides>
__aicore__ inline constexpr stride<FirstStride, Strides...> make_stride(const FirstStride& first_stride,
                                                                        const Strides&... strides);

template <typename FirstCoord, typename... Coords>
__aicore__ inline constexpr coord<FirstCoord, Coords...> make_coord(const FirstCoord& first_coord,
                                                                    const Coords&... coords);

template <typename Shape, typename Stride>
__aicore__ inline constexpr auto make_layout(const Shape& shape, const Stride& stride);

template <typename Shape>
__aicore__ inline constexpr auto make_layout(const Shape& shape);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto get_shape(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto get_stride(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto coshape(const Layout& layout);

template <typename LayoutPattern, typename Trait, typename Shape, typename Stride>
__aicore__ inline constexpr auto make_pattern_layout(const Shape& shape, const Stride& stride);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto cosize(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto rank(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto select(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto get(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto size(const Layout& layout);

template <size_t... Is, typename Layout, typename Enable = Std::enable_if_t<is_layout_v<Layout>>>
__aicore__ inline constexpr auto capacity(const Layout& layout);

template <typename Tensor, typename Coord, typename Info>
__aicore__ inline decltype(auto) slice(Tensor&& tensor, const Coord& coord, const Info& info);

template <typename LayoutPattern, typename Trait = Std::ignore_t, typename... Args>
__aicore__ inline decltype(auto) make_frame_layout(const Args&... args);

template <typename LayoutPattern, size_t C0Element, typename... Args>
__aicore__ inline decltype(auto) make_frame_layout(const Args&... args);

template <typename LayoutPattern, typename Trait = layout_trait_default<>>
struct frame_layout_format {
    template <typename... Args>
    __aicore__ inline decltype(auto) operator()(const Args&... args);
};

} // namespace te
} // namespace asc

#include "impl/tensor_api/tensor/layout_impl.h"

#endif // INCLUDE_TENSOR_API_TENSOR_LAYOUT_INTERFACE_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
