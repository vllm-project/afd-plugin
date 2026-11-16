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
    "impl/tensor_api/arch/cube/mmad/mmad.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file mmad.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_H
#define IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_H

#include "impl/tensor_api/arch/cube/mmad/routing.h"

namespace asc {
namespace te {

template <typename Trait, const Trait& traits, typename... Args>
__aicore__ inline void mmad_operation::mmad(const Args&... args)
{
    mmad_impl<traits>(args...);
}

template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor>
__aicore__ inline void mmad_operation::mmad_impl(const CTensor& c, const ATensor& a, const BTensor& b,
                                                 const mmad_params& params)
{
    using c_pos = get_mem_location<CTensor>;
    using a_pos = get_mem_location<ATensor>;
    using b_pos = get_mem_location<BTensor>;
    static_assert(Std::is_same_v<c_pos, location::l0c>, "When Mmad, c tensor must be from L0C.");
    static_assert(Std::is_same_v<a_pos, location::l0a>, "When Mmad, a tensor must be from L0A.");
    static_assert(Std::is_same_v<b_pos, location::l0b>, "When Mmad, b tensor must be from L0B.");
    using c_layout = typename CTensor::layout_type;
    using a_layout = typename ATensor::layout_type;
    using b_layout = typename BTensor::layout_type;
    using c_layout_ptn = get_layout_pattern<c_layout>;
    using a_layout_ptn = get_layout_pattern<a_layout>;
    using b_layout_ptn = get_layout_pattern<b_layout>;
    TENSOR_API_DEBUG_CHECK(debug_check_mmad_params, params, trait.disable_gemv, "mmad");
    TENSOR_API_DEBUG_CHECK(debug_check_unit_flag, params.unit_flag, "mmad");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, c.layout(), "c", "mmad");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, a.layout(), "a", "mmad");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, b.layout(), "b", "mmad");
    using mmad_impl = typename mmad_routing<current_arch_version, c_layout_ptn, a_layout_ptn, b_layout_ptn,
                                            location::invalid>::type;
    mmad_impl::template run<trait>(c, a, b, params);
}

template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename BiasTensor>
__aicore__ inline void mmad_operation::mmad_impl(const CTensor& c, const ATensor& a, const BTensor& b,
                                                 const BiasTensor& bias, const mmad_params& params)
{
    using c_pos = get_mem_location<CTensor>;
    using a_pos = get_mem_location<ATensor>;
    using b_pos = get_mem_location<BTensor>;
    using bias_pos = get_mem_location<BiasTensor>;
    static_assert(Std::is_same_v<c_pos, location::l0c>, "When Mmad, c tensor must be from L0C.");
    static_assert(Std::is_same_v<a_pos, location::l0a>, "When Mmad, a tensor must be from L0A.");
    static_assert(Std::is_same_v<b_pos, location::l0b>, "When Mmad, b tensor must be from L0B.");
    static_assert(Std::is_same_v<bias_pos, location::l0c> || Std::is_same_v<bias_pos, location::bias>,
                  "When Mmad, bias tensor must be from L0C or BIAS.");
    using c_layout = typename CTensor::layout_type;
    using a_layout = typename ATensor::layout_type;
    using b_layout = typename BTensor::layout_type;
    using c_layout_ptn = get_layout_pattern<c_layout>;
    using a_layout_ptn = get_layout_pattern<a_layout>;
    using b_layout_ptn = get_layout_pattern<b_layout>;
    TENSOR_API_DEBUG_CHECK(debug_check_mmad_params, params, trait.disable_gemv, "mmad with bias");
    TENSOR_API_DEBUG_CHECK(debug_check_unit_flag, params.unit_flag, "mmad with bias");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, c.layout(), "c", "mmad with bias");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, a.layout(), "a", "mmad with bias");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, b.layout(), "b", "mmad with bias");
    TENSOR_API_DEBUG_CHECK(debug_check_layout, bias.layout(), "bias", "mmad with bias");
    using mmad_impl =
        typename mmad_routing<current_arch_version, c_layout_ptn, a_layout_ptn, b_layout_ptn, bias_pos>::type;
    mmad_impl::template run<trait>(c, a, b, bias, params);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
