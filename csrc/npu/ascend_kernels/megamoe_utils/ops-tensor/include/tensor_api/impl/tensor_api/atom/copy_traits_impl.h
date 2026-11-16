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
    "impl/tensor_api/atom/copy_traits_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy_traits_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ATOM_COPY_TRAITS_IMPL_H
#define IMPL_TENSOR_API_ATOM_COPY_TRAITS_IMPL_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename AlwaysVoid, typename CopyOperation, typename Trait, const Trait& trait, typename... Args>
struct has_pascal_copy_method : Std::false_type {};

template <typename CopyOperation, typename Trait, const Trait& trait, typename... Args>
struct has_pascal_copy_method<void_t<decltype(
    CopyOperation::template Copy<Trait, trait, Args...>(std::declval<Args>()...))>, CopyOperation, Trait, trait,
    Args...> : Std::true_type {};

template <typename AlwaysVoid, typename CopyTraitsType, typename Trait, const Trait& trait, typename... Args>
struct has_pascal_copy_unpack_method : Std::false_type {};

template <typename CopyTraitsType, typename Trait, const Trait& trait, typename... Args>
struct has_pascal_copy_unpack_method<void_t<decltype(
    std::declval<const CopyTraitsType&>().template CopyUnpack<trait, Args...>(std::declval<Args>()...))>,
    CopyTraitsType, Trait, trait, Args...> : Std::true_type {};

template <typename CopyOperation, typename CopyTrait, typename CopyOperationWith, typename CopyTraitWith>
template <typename Params>
__aicore__ inline constexpr copy_traits<CopyOperationWith, CopyTraitWith>
copy_traits<CopyOperation, CopyTrait, CopyOperationWith, CopyTraitWith>::with(const Params& params) const
{
    return {params};
}

template <typename CopyOperation, typename CopyTrait, typename CopyOperationWith, typename CopyTraitWith>
template <const typename copy_traits<CopyOperation, CopyTrait, CopyOperationWith, CopyTraitWith>::trait_type& trait,
          typename... Args>
__aicore__ inline void
copy_traits<CopyOperation, CopyTrait, CopyOperationWith, CopyTraitWith>::copy_unpack(const Args&... args) const
{
    if constexpr (has_pascal_copy_method<void, CopyOperation, trait_type, trait, Args...>::value) {
        CopyOperation::template Copy<trait_type, trait, Args...>(args...);
    } else {
        CopyOperation::template copy<trait_type, trait, Args...>(args...);
    }
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ATOM_COPY_TRAITS_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
