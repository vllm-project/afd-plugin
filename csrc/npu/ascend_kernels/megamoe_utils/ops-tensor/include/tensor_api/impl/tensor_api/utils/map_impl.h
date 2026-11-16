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
    "impl/tensor_api/utils/map_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file map_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_MAP_IMPL_H
#define IMPL_TENSOR_API_UTILS_MAP_IMPL_H

#include "impl/tensor_api/utils/extra_impl.h"

namespace asc {
namespace te {

template <typename... Pairs>
class tuple_map {
private:
    using map_type = Std::tuple<Pairs...>;
    static constexpr size_t map_size = sizeof...(Pairs);

    template <typename Pair>
    using pair_key = typename Std::tuple_element<0, Pair>::type;

    template <typename Pair>
    using pair_value = typename Std::tuple_element<1, Pair>::type;

    template <typename Pair>
    struct is_valid_pair : Std::false_type {};

    template <typename Key, typename Value>
    struct is_valid_pair<Std::tuple<Key, Value>> : Std::true_type {};

    template <typename Input, typename KeySet>
    struct contains_key : Std::false_type {};

    template <typename Input, typename... Keys>
    struct contains_key<Input, Std::tuple<Keys...>> : Std::bool_constant<(Std::is_same_v<Input, Keys> || ...)> {};

    template <typename Input, typename KeySet, bool IsTupleKey>
    struct match_key_impl : Std::is_same<KeySet, Input> {};

    template <typename Input, typename KeySet>
    struct match_key_impl<Input, KeySet, true> : contains_key<Input, KeySet> {};

    template <typename Input, typename KeySet>
    struct match_key : match_key_impl<Input, KeySet, Std::is_tuple_v<KeySet>> {};

    template <typename Pair, typename Map>
    struct prepend_pair;

    template <typename Pair, typename... ExistingPairs>
    struct prepend_pair<Pair, tuple_map<ExistingPairs...>> {
        using type = tuple_map<Pair, ExistingPairs...>;
    };

    template <typename Key, size_t Index, size_t MaxSize>
    struct get_impl {
        using current_pair = typename Std::tuple_element<Index, map_type>::type;
        static_assert(is_valid_pair<current_pair>::value, "tuple_map expects Std::tuple<Key, Value> entries.");
        using current_key = pair_key<current_pair>;
        using current_val = pair_value<current_pair>;

        using next_result = typename get_impl<Key, Index + 1, MaxSize>::type;

        using type = Std::conditional_t<Std::is_same_v<current_key, Key>, current_val, next_result>;
    };

    template <typename Key, size_t MaxSize>
    struct get_impl<Key, MaxSize, MaxSize> {
        using type = Std::ignore_t;
    };

    template <typename Input, size_t Index, size_t MaxSize>
    struct match_impl {
        using current_pair = typename Std::tuple_element<Index, map_type>::type;
        static_assert(is_valid_pair<current_pair>::value, "tuple_map expects Std::tuple<Key, Value> entries.");
        using current_key = pair_key<current_pair>;
        using current_val = pair_value<current_pair>;

        using next_result = typename match_impl<Input, Index + 1, MaxSize>::type;

        using type = Std::conditional_t<match_key<Input, current_key>::value, current_val, next_result>;
    };

    template <typename Input, size_t MaxSize>
    struct match_impl<Input, MaxSize, MaxSize> {
        using type = Std::ignore_t;
    };

    template <typename Key, typename... ExistingPairs>
    struct delete_impl;

    template <typename Key, typename Value, typename... ExistingPairs>
    struct insert_impl;

    template <typename Key, typename Value>
    struct insert_impl<Key, Value> {
        using type = tuple_map<Std::tuple<Key, Value>>;
    };

    template <typename Key, typename Value, typename FirstPair, typename... RestPairs>
    struct insert_impl<Key, Value, FirstPair, RestPairs...> {
        static_assert(is_valid_pair<FirstPair>::value, "tuple_map expects Std::tuple<Key, Value> entries.");
        using first_key = pair_key<FirstPair>;
        using new_pair = Std::tuple<Key, Value>;

        using next_map = typename insert_impl<Key, Value, RestPairs...>::type;
        using keep_head_map = typename prepend_pair<FirstPair, next_map>::type;
        using replace_tail_map = typename delete_impl<Key, RestPairs...>::type;
        using replace_head_map = typename prepend_pair<new_pair, replace_tail_map>::type;

        using type = Std::conditional_t<Std::is_same_v<first_key, Key>, replace_head_map, keep_head_map>;
    };

    template <typename Key>
    struct delete_impl<Key> {
        using type = tuple_map<>;
    };

    template <typename Key, typename FirstPair, typename... RestPairs>
    struct delete_impl<Key, FirstPair, RestPairs...> {
        static_assert(is_valid_pair<FirstPair>::value, "tuple_map expects Std::tuple<Key, Value> entries.");
        using first_key = pair_key<FirstPair>;

        using next_map = typename delete_impl<Key, RestPairs...>::type;
        using keep_head_map = typename prepend_pair<FirstPair, next_map>::type;

        using type = Std::conditional_t<Std::is_same_v<first_key, Key>, next_map, keep_head_map>;
    };

public:
    template <typename Key>
    using get = typename get_impl<Key, 0, map_size>::type;

    template <typename Input>
    using find = typename match_impl<Input, 0, map_size>::type;

    template <typename Key, typename Value>
    using insert = typename insert_impl<Key, Value, Pairs...>::type;

    template <typename Key>
    using erase = typename delete_impl<Key, Pairs...>::type;
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_UTILS_MAP_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
