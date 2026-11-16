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
    "impl/tensor_api/utils/constant_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file constant_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H
#define IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#include "tensor_api/utils/utils.h"
#include "impl/tensor_api/utils/extra_impl.h"
#include "impl/tensor_api/utils/map_impl.h"

namespace asc {
namespace te {

constexpr size_t two_dim_data = 2;
constexpr size_t three_dim_data = 3;
constexpr size_t four_dim_data = 4;
constexpr size_t five_dim_data = 5;
constexpr size_t fractal_fixed = 16;
constexpr size_t mx_scale_k0 = 2;
constexpr uint32_t block_cube = 16;
constexpr uint64_t hifloat8_mmad_ctrl_mask = 0x200000000000ULL;

struct arch_version {
    static constexpr uint32_t v3510 = 3510;
    static constexpr uint32_t v2201 = 2201;
    static constexpr uint32_t V3510 = 3510;
    static constexpr uint32_t V2201 = 2201;
};

struct get_arch_version {
    __aicore__ inline constexpr uint32_t operator()() const
    {
#ifdef __NPU_ARCH__
        return __NPU_ARCH__;
#else
        return 0;
#endif
    }
};

constexpr uint32_t current_arch_version = get_arch_version{}();

template <typename Hardware>
struct is_hardware {
private:
    template <typename Target, typename... Candidates>
    __aicore__ inline static constexpr bool is_unqualified_any_of()
    {
        return (... || Std::is_same_v<Std::remove_cvref_t<Target>, Candidates>);
    }

public:
    static constexpr bool value =
        is_unqualified_any_of<Hardware, location::invalid, location::gm, location::ub, location::l1, location::l0a,
                              location::l0b, location::l0scalea, location::l0scaleb, location::l0c, location::bias,
                              location::fixbuf, location::ssbuf>();
};

template <typename Hardware>
constexpr bool is_hardware_v = is_hardware<Hardware>::value;

template <typename TupleType>
using tuple_sequence = Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<TupleType>>>;

template <typename DataType>
struct location_attr {
    using gm_attr = __gm__ DataType*;
    using l1_attr = __cbuf__ DataType*;
    using l0a_attr = __ca__ DataType*;
    using l0b_attr = __cb__ DataType*;
    using l0c_attr = __cc__ DataType*;
    using ub_attr = __ubuf__ DataType*;
    using fixbuf_attr = __fbuf__ DataType*;
    using ssbuf_attr = __ssbuf__ DataType*;
    using bias_attr = __biasbuf__ DataType*;
    using none_attr = DataType*;

    using type =
        Std::tuple<gm_attr, l1_attr, l0a_attr, l0b_attr, l0c_attr, ub_attr, fixbuf_attr, ssbuf_attr, bias_attr>;

    using location_map = tuple_map<Std::tuple<location::gm, gm_attr>, Std::tuple<location::l1, l1_attr>,
                                   Std::tuple<location::l0a, l0a_attr>, Std::tuple<location::l0b, l0b_attr>,
                                   Std::tuple<location::l0scalea, none_attr>, Std::tuple<location::l0scaleb, none_attr>,
                                   Std::tuple<location::l0c, l0c_attr>, Std::tuple<location::ub, ub_attr>,
                                   Std::tuple<location::bias, bias_attr>, Std::tuple<location::fixbuf, fixbuf_attr>>;
};

template <typename Target, typename ProcessedTuple>
struct all_elements_same_as_a;

template <typename Target, typename First, typename... Rest>
struct all_elements_same_as_a<Target, Std::tuple<First, Rest...>> {
    static constexpr bool value = Std::is_same_v<Target, typename iter_ele<First>::type>
                                  || all_elements_same_as_a<Target, Std::tuple<Rest...>>::value;
};

template <typename Target>
struct all_elements_same_as_a<Target, Std::tuple<>> {
    static constexpr bool value = false;
};

template <typename Target, typename... Candidates>
struct check_all_same;

template <typename Target, typename Candidate, typename... RestCandidates>
struct check_all_same<Target, Candidate, RestCandidates...> {
    static constexpr bool value = Std::is_same_v<Target, Candidate>
                                  || all_elements_same_as_a<Target, typename location_attr<Candidate>::type>::value
                                  || check_all_same<Target, RestCandidates...>::value;
};

template <typename Target>
struct check_all_same<Target> {
    static constexpr bool value = false;
};

template <typename Target, typename... Candidates>
constexpr bool is_one_of_attr_v = check_all_same<Target, Candidates...>::value;

template <typename DataType>
inline constexpr bool is_data_type =
    is_one_of_attr_v<Std::remove_cvref_t<DataType>, hifloat8_t, bfloat16_t, fp4x2_e1m2_t, fp4x2_e2m1_t, fp8_e5m2_t,
                     fp8_e4m3fn_t, fp8_e8m0_t>
    || Std::is_integral_v<Std::remove_cvref_t<DataType>> || Std::is_floating_point_v<Std::remove_cvref_t<DataType>>;

template <typename DataType>
inline constexpr bool is_b4_type = is_one_of_attr_v<DataType, fp4x2_e1m2_t, fp4x2_e2m1_t>;

template <typename DataType = Std::ignore_t>
__aicore__ inline constexpr size_t get_c0_size()
{
    constexpr size_t c0_size = 32;
    if constexpr (is_b4_type<DataType>) {
        return c0_size * 2;
    } else {
        return c0_size;
    }
}

template <typename DataType = Std::ignore_t>
constexpr size_t c0_size = get_c0_size<DataType>();

template <typename DataType>
constexpr size_t c0_element = c0_size<DataType> / sizeof(DataType);

// is_integral_constant
template <typename Constant>
struct is_integral_constant : Std::false_type {};

template <size_t Value>
struct is_integral_constant<Std::Int<Value>> : Std::true_type {};

template <typename Constant>
constexpr bool is_integral_constant_v = is_integral_constant<Constant>::value;

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
