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
    "impl/tensor_api/atom/cube/mmad.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file mmad.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ATOM_CUBE_MMAD_H
#define IMPL_TENSOR_API_ATOM_CUBE_MMAD_H

#include "impl/tensor_api/atom/mmad_traits_impl.h"
#include "impl/tensor_api/arch/cube/mmad/mmad.h"

namespace asc {
namespace te {

struct mmad_op_with : public mmad_operation {};

template <typename MmadTraitsType>
struct mmad_traits<mmad_op_with, MmadTraitsType> {
    using trait_type = get_trait_member_type_t<MmadTraitsType>;
    static constexpr const trait_type default_trait = MmadTraitsType::value;

    template <const trait_type& traits = default_trait, typename... Args>
    __aicore__ inline void mmad_unpack(const Args&... args) const
    {
        mmad_op_with::mmad<trait_type, traits, Args...>(args..., params);
    }

    mmad_params params;
};

template <typename MmadTraitsType>
struct mmad_traits<mmad_operation, MmadTraitsType>
    : public mmad_traits<mmad_operation, MmadTraitsType, mmad_op_with, MmadTraitsType> {};

template <>
struct mmad_traits<mmad_operation> : public mmad_traits<mmad_operation, mmad_trait_default> {};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ATOM_CUBE_MMAD_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
