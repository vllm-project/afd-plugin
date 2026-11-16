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
 * \file copy_atom.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ATOM_CUBE_COPY_ATOM_H
#define INCLUDE_TENSOR_API_ATOM_CUBE_COPY_ATOM_H

#include "tensor_api/tensor/layout_interface.h"
#include "tensor_api/tensor/pointer.h"

namespace asc {
namespace te {

template <typename CopyOperation, typename... CopyOpArgs>
struct copy_traits {};

template <typename CopyOperation, typename CopyTrait, typename CopyOperationWith, typename CopyTraitWith>
struct copy_traits<CopyOperation, CopyTrait, CopyOperationWith, CopyTraitWith> {
    using trait_type = get_trait_member_type_t<CopyTrait>;
    using TraitType = trait_type;
    static constexpr const trait_type default_trait = CopyTrait::value;

    template <typename Params>
    __aicore__ inline constexpr copy_traits<CopyOperationWith, CopyTraitWith> with(const Params& params) const;

    template <const trait_type& trait = default_trait, typename... Args>
    __aicore__ inline void copy_unpack(const Args&... args) const;
};

template <typename... Args>
struct copy_atom;

template <typename CopyOperation>
struct copy_atom<CopyOperation> : copy_atom<copy_traits<CopyOperation>> {};

template <typename... Args>
struct copy_atom<copy_traits<Args...>> : copy_traits<Args...> {
    using copy_traits_type = copy_traits<Args...>;
    using trait_type = get_trait_member_type_t<copy_traits_type>;
    static constexpr const trait_type default_trait = copy_traits_type::default_trait;

    template <const trait_type& traits = default_trait, typename... Params>
    __aicore__ inline void call(const Params&... params) const;

    template <typename... TraitArgs>
    __aicore__ inline auto with(TraitArgs&&... args) const;
};

} // namespace te
} // namespace asc

#include "tensor_api/arch/cube/copy_op.h"
#include "tensor_api/arch/vector/copy_op.h"
#include "impl/tensor_api/atom/copy_atom_impl.h"

#endif // INCLUDE_TENSOR_API_ATOM_CUBE_COPY_ATOM_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
