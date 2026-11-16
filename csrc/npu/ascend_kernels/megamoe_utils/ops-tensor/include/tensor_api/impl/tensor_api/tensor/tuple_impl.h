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
    "impl/tensor_api/tensor/tuple_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file tuple_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_TUPLE_IMPL_H
#define IMPL_TENSOR_API_TENSOR_TUPLE_IMPL_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename Tuple, typename Transform, typename Combine, size_t... I>
__aicore__ inline constexpr auto tuple_apply(Tuple&& tuple, Transform&& transform, Combine&& combine,
                                             Std::index_sequence<I...>)
{
    return combine(transform(Std::get<I>(static_cast<Tuple&&>(tuple)))...);
}

template <typename Left, typename Right, typename Transform, typename Combine, size_t... I>
__aicore__ inline constexpr auto tuple_apply(Left&& left, Right&& right, Transform&& transform, Combine&& combine,
                                             Std::index_sequence<I...>)
{
    return combine(transform(Std::get<I>(static_cast<Left&&>(left)), Std::get<I>(static_cast<Right&&>(right)))...);
}

template <typename Input, typename Transform, typename Combine>
__aicore__ inline constexpr auto transform_apply(Input&& input, Transform&& transform, Combine&& combine)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<Input>>) {
        return tuple_apply(static_cast<Input&&>(input), transform, combine, tuple_sequence<Input>{});
    } else {
        return combine(transform(static_cast<Input&&>(input)));
    }
}

template <typename Left, typename Right, typename Transform, typename Combine>
__aicore__ inline constexpr auto transform_apply(Left&& left, Right&& right, Transform&& transform, Combine&& combine)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<Left>>) {
        return tuple_apply(static_cast<Left&&>(left), static_cast<Right&&>(right), transform, combine,
                           tuple_sequence<Left>{});
    } else {
        return combine(transform(static_cast<Left&&>(left), static_cast<Right&&>(right)));
    }
}

template <typename Left, typename Right, typename Operation>
__aicore__ inline constexpr auto transform_tuple_apply(const Left& left, const Right& right,
                                                       const Operation& operation);

template <typename Left, typename Right, typename Operation, size_t... Is>
__aicore__ inline constexpr auto transform_tuple_apply_impl(const Left& left, const Right& right,
                                                            const Operation& operation, Std::index_sequence<Is...>)
{
    return Std::make_tuple(transform_tuple_apply(Std::get<Is>(left), Std::get<Is>(right), operation)...);
}

template <typename Left, typename Right, typename Operation>
__aicore__ inline constexpr auto transform_tuple_apply(const Left& left, const Right& right, const Operation& operation)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<Left>>) {
        static_assert(Std::tuple_size_v<Std::remove_cvref_t<Left>> == Std::tuple_size_v<Std::remove_cvref_t<Right>>,
                      "Two tuple needs to be the same tuple size");
        return transform_tuple_apply_impl(left, right, operation,
                                          Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<Left>>>{});
    } else {
        static_assert(!Std::is_tuple_v<Std::remove_cvref_t<Right>>, "Two tuple needs to be the same tuple size");
        return operation(left, right);
    }
}

struct multiplies_unary_left_fold {
    template <typename... Values>
    __aicore__ inline constexpr auto operator()(Values&&... values) const
    {
        return (... * values);
    }
};

struct product {
    template <typename Value>
    __aicore__ inline constexpr auto operator()(const Value& value) const
    {
        if constexpr (Std::is_tuple_v<Value>) {
            if constexpr (Std::tuple_size_v<Value> == 0) {
                return _1{};
            } else {
                return transform_apply(value, product{}, multiplies_unary_left_fold{});
            }
        } else if constexpr (Std::is_integral<Value>::value) {
            return value;
        } else {
            static_assert(sizeof(Value) == 0, "Invalid Product parameters");
        }
    }
};

template <size_t I, size_t... Is, typename Tuple,
          typename = Std::enable_if_t<Std::is_tuple_v<Std::remove_cvref_t<Tuple>>>>
__aicore__ inline constexpr auto get(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    if constexpr (sizeof...(Is) == 0) {
        return Std::get<I>(static_cast<Tuple&&>(t));
    } else {
        return get<Is...>(get<I>(static_cast<Tuple&&>(t)));
    }
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto get_tuple(Tuple&& t)
{
    if constexpr (sizeof...(Is) == 0) {
        return static_cast<Tuple&&>(t);
    } else {
        auto element = get<Is...>(static_cast<Tuple&&>(t));
        if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(element)>>) {
            return element;
        } else {
            return Std::make_tuple(element);
        }
    }
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto get_rank(const Tuple& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple!");
    if constexpr (sizeof...(Is) == 0) {
        return Std::Int<Std::tuple_size_v<Tuple>>{};
    } else {
        return get_rank(get_tuple<Is...>(t));
    }
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto tuple_size(const Tuple& t)
{
    if constexpr (sizeof...(Is) == 0) {
        return product{}(t);
    } else {
        return tuple_size(get_tuple<Is...>(t));
    }
}

template <size_t I, typename Tuple>
__aicore__ inline constexpr auto select_tuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    auto&& tt = Std::get<I>(static_cast<Tuple&&>(t));
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(tt)>>) {
        return tt;
    } else {
        return Std::make_tuple(tt);
    }
}

template <size_t I0, size_t I1, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto select_tuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    return Std::make_tuple(Std::get<I0>(static_cast<Tuple&&>(t)), Std::get<I1>(static_cast<Tuple&&>(t)),
                           Std::get<Is>(static_cast<Tuple&&>(t))...);
}

template <typename Tuple>
__aicore__ inline constexpr auto select_tuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    return static_cast<Tuple&&>(t);
}

template <size_t index, size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr decltype(auto) get_value(const Tuple& t)
{
    decltype(auto) tuple_ele = Std::get<index>(t);
    if constexpr (sizeof...(Is) == 0) {
        return Std::get<I>(tuple_ele);
    } else {
        return Std::make_tuple(Std::get<I>(tuple_ele), Std::get<Is>(tuple_ele)...);
    }
}

template <size_t index, typename Tuple>
__aicore__ inline constexpr decltype(auto) get_value(const Tuple& t)
{
    return Std::get<index>(t);
}

template <typename FirstValue, typename... Values>
__aicore__ inline constexpr auto get_max(const FirstValue& first_value, const Values&... values)
{
    if constexpr (sizeof...(Values) == 0) {
        return first_value;
    } else {
        return Std::max(first_value, get_max(values...));
    }
}

template <typename Shape, typename Stride>
__aicore__ inline constexpr auto get_capacity(const Shape& shape, const Stride& stride);

template <typename Shape, typename Stride, size_t... Is>
__aicore__ inline constexpr auto get_capacity_impl(const Shape& shape, const Stride& stride, Std::index_sequence<Is...>)
{
    return get_max(get_capacity(Std::get<Is>(shape), Std::get<Is>(stride))...);
}

template <typename Shape, typename Stride>
__aicore__ inline constexpr auto get_capacity(const Shape& shape, const Stride& stride)
{
    if constexpr (Std::is_tuple_v<Shape> && Std::is_tuple_v<Stride>) {
        static_assert(Std::tuple_size_v<Shape> == Std::tuple_size_v<Stride>, "Mismatched ranks");
        return get_capacity_impl(shape, stride, Std::make_index_sequence<Std::tuple_size_v<Shape>>{});
    } else {
        return shape * stride;
    }
}

// ===== Squeeze helpers =====

// PrependIndex: prepend a size_t to an index_sequence.
template <size_t Idx, typename Seq>
struct prepend_index;
template <size_t Idx, size_t... Is>
struct prepend_index<Idx, Std::index_sequence<Is...>> {
    using type = Std::index_sequence<Idx, Is...>;
};

// KeepIndexSeq: produce the index_sequence of positions whose bool mask is true.
template <size_t Idx, bool... Mask>
struct keep_index_seq {
    using type = Std::index_sequence<>;
};
template <size_t Idx, bool Head, bool... Tail>
struct keep_index_seq<Idx, Head, Tail...> {
    using rest = typename keep_index_seq<Idx + 1, Tail...>::type;
    using type = Std::conditional_t<Head, typename prepend_index<Idx, rest>::type, rest>;
};
template <bool... Mask>
using keep_index_seq_t = typename keep_index_seq<0, Mask...>::type;

// KeepIndexSeqFromSeq: same as KeepIndexSeqT but the bool mask is carried by an IntegerSequence.
template <typename BoolSeq>
struct keep_index_seq_from_seq;
template <bool... Bs>
struct keep_index_seq_from_seq<Std::IntegerSequence<bool, Bs...>> {
    using type = typename keep_index_seq_t<Bs...>::type;
};

// SelectBySeq: pick tuple elements at the indices carried by an index_sequence.
template <typename Tuple, size_t... Is>
__aicore__ inline constexpr auto select_by_seq_impl(const Tuple& t, Std::index_sequence<Is...>)
{
    return Std::make_tuple(Std::get<Is>(t)...);
}
template <typename Seq, typename Tuple>
__aicore__ inline constexpr auto select_by_seq(const Tuple& t)
{
    return select_by_seq_impl(t, Seq{});
}

// IsSameStructure: two types share the same tuple nesting structure (same depth, same size per level).
template <typename Left, typename Right, typename = void>
struct is_same_structure;

template <typename Left, typename Right, typename Seq>
struct all_children_same;

template <typename Left, typename Right, size_t... Is>
struct all_children_same<Left, Right, Std::index_sequence<Is...>>
    : Std::bool_constant<(is_same_structure<typename Std::tuple_element<Is, Std::remove_cvref_t<Left>>::type,
                                            typename Std::tuple_element<Is, Std::remove_cvref_t<Right>>::type>::value
                          && ...)> {};

template <typename Left, typename Right, typename>
struct is_same_structure : Std::false_type {};

template <typename Left, typename Right>
struct is_same_structure<
    Left, Right,
    Std::enable_if_t<Std::is_tuple_v<Std::remove_cvref_t<Left>> && Std::is_tuple_v<Std::remove_cvref_t<Right>>>>
    : Std::conditional_t<Std::tuple_size_v<Std::remove_cvref_t<Left>> == Std::tuple_size_v<Std::remove_cvref_t<Right>>,
                         all_children_same<Std::remove_cvref_t<Left>, Std::remove_cvref_t<Right>,
                                           Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<Left>>>>,
                         Std::false_type> {};

template <typename Left, typename Right>
struct is_same_structure<
    Left, Right,
    Std::enable_if_t<!Std::is_tuple_v<Std::remove_cvref_t<Left>> && !Std::is_tuple_v<Std::remove_cvref_t<Right>>>>
    : Std::true_type {};

template <typename Left, typename Right>
constexpr bool is_same_structure_v = is_same_structure<Std::remove_cvref_t<Left>, Std::remove_cvref_t<Right>>::value;

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_TUPLE_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
