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
#warning \
    "impl/tensor_api/atom/vector/copy_gm2ub.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ATOM_VECTOR_COPY_GM2UB_H
#endif

/*!
 * \file copy_gm2ub.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ATOM_VECTOR_COPY_GM2UB_H
#define IMPL_TENSOR_API_ATOM_VECTOR_COPY_GM2UB_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/vector/gm_to_ub/copy.h"
#include "impl/tensor_api/atom/copy_traits_impl.h"

namespace asc {
namespace te {

template <typename Traits>
struct copy_traits<copy_gm_to_ub, Traits> : public copy_traits<copy_gm_to_ub, Traits, copy_gm_to_ub_with, Traits> {};

template <>
struct copy_traits<copy_gm_to_ub> : public copy_traits<copy_gm_to_ub, gm_to_ub_trait_default> {};

template <typename TraitStruct>
struct copy_traits<copy_gm_to_ub_with, TraitStruct> {
    using trait_type = get_trait_member_type_t<TraitStruct>;
    static constexpr const trait_type default_trait = TraitStruct::value;

    template <const trait_type& trait = default_trait, typename... Args>
    __aicore__ inline void copy_unpack(const Args&... args) const
    {
        copy_gm_to_ub_with::template copy<trait_type, trait, Args...>(args..., params);
    }

    gm_to_ub_params params{};
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ATOM_VECTOR_COPY_GM2UB_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ATOM_VECTOR_COPY_GM2UB_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_IMPL_TENSOR_API_ATOM_VECTOR_COPY_GM2UB_H
#endif
