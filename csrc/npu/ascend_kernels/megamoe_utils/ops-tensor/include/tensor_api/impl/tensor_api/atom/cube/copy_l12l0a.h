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
    "impl/tensor_api/atom/cube/copy_l12l0a.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy_l12l0a.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ATOM_CUBE_COPY_L12L0A_H
#define IMPL_TENSOR_API_ATOM_CUBE_COPY_L12L0A_H

#include "impl/tensor_api/arch/cube/l1_to_l0a/copy.h"
#include "impl/tensor_api/atom/copy_traits_impl.h"

namespace asc {
namespace te {

template <typename Traits>
struct copy_traits<copy_l1_to_l0a, Traits> : public copy_traits<copy_l1_to_l0a, Traits, copy_l1_to_l0a_with, Traits> {
    using base_type = copy_traits<copy_l1_to_l0a, Traits, copy_l1_to_l0a_with, Traits>;
    using trait_type = get_trait_member_type_t<base_type>;
    using base_type::with;

    template <typename PaddingValue>
    __aicore__ inline constexpr auto with(const AscendC::Te::Img2ColParams<PaddingValue>& params) const
    {
        return copy_traits<copy_l1_to_l0a_with, Traits, PaddingValue>{AscendC::Te::normalize_img2col_params(params)};
    }

    template <typename PaddingValue>
    __aicore__ inline constexpr auto with(const img2col_params<PaddingValue>& params) const
    {
        return copy_traits<copy_l1_to_l0a_with, Traits, PaddingValue>{params};
    }
};

template <>
struct copy_traits<copy_l1_to_l0a> : public copy_traits<copy_l1_to_l0a, l1_to_l0a_trait_default> {};

template <typename TraitStruct, typename PaddingValue>
struct copy_traits<copy_l1_to_l0a_with, TraitStruct, PaddingValue> {
    using trait_type = get_trait_member_type_t<TraitStruct>;
    static constexpr const trait_type default_trait = TraitStruct::value;

    template <const trait_type& trait = default_trait, typename... Args>
    __aicore__ inline void copy_unpack(const Args&... args) const
    {
        copy_l1_to_l0a_with::copy<trait_type, trait, Args...>(args..., params);
    }
    img2col_params<PaddingValue> params;
};

template <typename TraitStruct>
struct copy_traits<copy_l1_to_l0a_with, TraitStruct> {
    using trait_type = get_trait_member_type_t<TraitStruct>;
    static constexpr const trait_type default_trait = TraitStruct::value;

    template <const trait_type& trait = default_trait, typename... Args>
    __aicore__ inline void copy_unpack(const Args&... args) const
    {
        copy_l1_to_l0a_with::copy<trait_type, trait, Args...>(args..., params);
    }
    img2col_params<int16_t> params{};
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ATOM_CUBE_COPY_L12L0A_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
