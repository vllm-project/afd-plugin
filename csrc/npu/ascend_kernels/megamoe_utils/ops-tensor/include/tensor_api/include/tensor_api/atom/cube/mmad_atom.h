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
 * \file mmad_atom.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ATOM_CUBE_MMAD_ATOM_H
#define INCLUDE_TENSOR_API_ATOM_CUBE_MMAD_ATOM_H

#include "tensor_api/tensor/layout_interface.h"
#include "tensor_api/tensor/pointer.h"
#include "tensor_api/arch/cube/mmad_op.h"

namespace asc {
namespace te {

template <typename MmadOperation, typename... MmadOpArgs>
struct mmad_traits {};

template <typename MmadOperation, typename MmadTrait, typename MmadOperationWith, typename MmadTraitWith>
struct mmad_traits<MmadOperation, MmadTrait, MmadOperationWith, MmadTraitWith> {
    using trait_type = get_trait_member_type_t<MmadTrait>;
    static constexpr const trait_type default_trait = MmadTrait::value;

    template <typename Params>
    __aicore__ inline constexpr mmad_traits<MmadOperationWith, MmadTraitWith> with(const Params& params) const;

    template <const trait_type& trait = default_trait, typename... Args>
    __aicore__ inline void mmad_unpack(const Args&... args) const;
};

template <typename... Args>
struct mmad_atom;

template <typename MmadOperation>
struct mmad_atom<MmadOperation> : mmad_atom<mmad_traits<MmadOperation>> {};

template <typename MmadOperation, typename... Args>
struct mmad_atom<mmad_traits<MmadOperation, Args...>> : mmad_traits<MmadOperation, Args...> {
    using mmad_traits_type = mmad_traits<MmadOperation, Args...>;
    using trait_type = get_trait_member_type_t<mmad_traits_type>;
    static constexpr const trait_type default_trait = mmad_traits_type::default_trait;

    template <const trait_type& traits = default_trait, typename... Params>
    __aicore__ inline void call(const Params&... params) const;

    template <typename... TraitArgs>
    __aicore__ inline auto with(TraitArgs&&... args) const;
};

} // namespace te
} // namespace asc

#include "impl/tensor_api/atom/mmad_atom_impl.h"

#endif // INCLUDE_TENSOR_API_ATOM_CUBE_MMAD_ATOM_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
