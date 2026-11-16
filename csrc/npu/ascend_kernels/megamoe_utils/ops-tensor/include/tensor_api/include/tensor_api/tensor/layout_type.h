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
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_LAYOUT_TYPE_H
#endif

/*!
 * \file layout_type.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_LAYOUT_TYPE_H
#define INCLUDE_TENSOR_API_TENSOR_LAYOUT_TYPE_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/layout_size.h"

namespace asc {
namespace te {

template <typename... Shapes>
using shape = Std::tuple<Shapes...>;

template <typename... Strides>
using stride = Std::tuple<Strides...>;

template <typename... Coords>
using coord = Std::tuple<Coords...>;

struct zn_layout_ptn {};
struct zz_layout_ptn {};
struct nn_layout_ptn {};
struct nz_layout_ptn {};
struct nd_layout_ptn {};
struct dn_layout_ptn {};
struct nd_ext_layout_ptn {};
struct dn_ext_layout_ptn {};
struct one_dim_layout_ptn {};
struct scalea_nd_layout_ptn {};
struct scalea_dn_layout_ptn {};
struct scaleb_nd_layout_ptn {};
struct scaleb_dn_layout_ptn {};
struct nchw_layout_ptn {};
struct nhwc_layout_ptn {};
struct nc1hwc0_layout_ptn {};
struct ncdhw_layout_ptn {};
struct ndc1hwc0_layout_ptn {};

template <typename DataType, typename C0>
struct layout_trait {
    static constexpr auto c0_element = C0{};
};

template <typename DataType = uint16_t, typename C0 = Std::Int<c0_element<DataType>>>
struct layout_trait_default : layout_trait<DataType, C0> {};

template <typename ShapeTuple, typename StrideTuple, typename Info = Std::ignore_t>
struct layout : private Std::tuple<ShapeTuple, StrideTuple> {
public:
    static constexpr auto depth = nesting_depth_v<ShapeTuple>;
    static constexpr auto rank_size = Std::tuple_size_v<ShapeTuple>;

    __aicore__ inline constexpr layout(const ShapeTuple& shape = {}, const StrideTuple& stride = {});

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) capacity() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) shape();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) shape() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) stride();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) stride() const;

    template <typename Coord>
    __aicore__ inline constexpr auto operator()(const Coord& coord) const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) rank() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) size() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) get();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) get() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Capacity() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Rank() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Size() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Get();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Get() const;

private:
    using tag = Info;
};

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_TENSOR_LAYOUT_TYPE_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_LAYOUT_TYPE_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_LAYOUT_TYPE_H
#endif
