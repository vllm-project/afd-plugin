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
    "impl/tensor_api/tensor/layout_definition.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_definition.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_DEFINITION_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_DEFINITION_H

namespace asc {
namespace te {

template <typename Layout>
struct get_layout_info {
    using type = Std::tuple<Std::ignore_t, Std::ignore_t>;
};

template <typename ShapeTuple, typename StrideTuple, typename LayoutPattern, typename TraitType>
struct get_layout_info<layout<ShapeTuple, StrideTuple, Std::tuple<LayoutPattern, TraitType>>> {
    using type = Std::tuple<LayoutPattern, TraitType>;
};

template <typename Layout>
using get_layout_info_t = typename get_layout_info<Layout>::type;

template <typename Layout>
struct get_pattern {
    using type = typename Std::tuple_element<0, get_layout_info_t<Layout>>::type;
};

template <typename Layout>
struct get_trait_type {
    using type = typename Std::tuple_element<1, get_layout_info_t<Layout>>::type;
};

template <typename Layout>
using get_layout_trait = typename get_trait_type<Std::remove_cvref_t<Layout>>::type;

template <typename Layout>
struct is_layout : Std::false_type {};

template <typename ShapeTuple, typename StrideTuple, typename Info>
struct is_layout<layout<ShapeTuple, StrideTuple, Info>> : Std::true_type {};

template <typename Layout>
constexpr bool is_layout_v = is_layout<Std::remove_cvref_t<Layout>>::value;

template <typename ShapeTuple, typename StrideTuple, typename Info>
__aicore__ inline constexpr layout<ShapeTuple, StrideTuple, Info>::layout(const ShapeTuple& shape,
                                                                          const StrideTuple& stride) :
    Std::tuple<ShapeTuple, StrideTuple>(shape, stride)
{
    static_assert(Std::is_tuple_v<ShapeTuple> && Std::is_tuple_v<StrideTuple>, "Shape or Stride is not tuple!");
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::capacity() const
{
    return get_capacity(shape<I...>(), stride<I...>());
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::shape()
{
    return get_value<0, I...>(static_cast<Std::tuple<ShapeTuple, StrideTuple>&>(*this));
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::shape() const
{
    return get_value<0, I...>(static_cast<const Std::tuple<ShapeTuple, StrideTuple>&>(*this));
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::stride()
{
    return get_value<1, I...>(static_cast<Std::tuple<ShapeTuple, StrideTuple>&>(*this));
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::stride() const
{
    return get_value<1, I...>(static_cast<const Std::tuple<ShapeTuple, StrideTuple>&>(*this));
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::rank() const
{
    static_assert(Std::tuple_size_v<ShapeTuple> == Std::tuple_size_v<StrideTuple>,
                  "The dimensions of the Shape and Stride are not the same.");
    return asc::te::get_rank<I...>(shape());
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::size() const
{
    return tuple_size<I...>(shape());
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::get()
{
    return asc::te::get<I...>(static_cast<Std::tuple<ShapeTuple, StrideTuple>&>(*this));
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::get() const
{
    return asc::te::get<I...>(static_cast<const Std::tuple<ShapeTuple, StrideTuple>&>(*this));
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Capacity() const
{
    return capacity<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Shape()
{
    return shape<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Shape() const
{
    return shape<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Stride()
{
    return stride<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Stride() const
{
    return stride<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Rank() const
{
    return rank<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Size() const
{
    return size<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Get()
{
    return get<I...>();
}

template <typename ShapeTuple, typename StrideTuple, typename Info>
template <size_t... I>
__aicore__ inline constexpr decltype(auto) layout<ShapeTuple, StrideTuple, Info>::Get() const
{
    return get<I...>();
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_DEFINITION_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
