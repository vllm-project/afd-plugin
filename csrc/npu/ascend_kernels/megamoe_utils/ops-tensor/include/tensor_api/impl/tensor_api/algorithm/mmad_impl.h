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
    "impl/tensor_api/algorithm/mmad_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file mmad_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ALGORITHM_MMAD_IMPL_H
#define IMPL_TENSOR_API_ALGORITHM_MMAD_IMPL_H

#include "impl/tensor_api/atom/mmad_atom_impl.h"

namespace asc {
namespace te {

template <typename Atom, typename CTensor, typename ATensor, typename BTensor>
__aicore__ inline void mmad(const mmad_atom<Atom>& atom, const CTensor& c, const ATensor& a, const BTensor& b)
{
    atom.call(c, a, b);
}

template <typename Atom, typename CTensor, typename ATensor, typename BTensor, typename BiasTensor,
          Std::enable_if_t<is_attr_tensor_v<BiasTensor>, int> Enable = 0>
__aicore__ inline void mmad(const mmad_atom<Atom>& atom, const CTensor& c, const ATensor& a, const BTensor& b,
                            const BiasTensor& bias)
{
    atom.call(c, a, b, bias);
}

template <typename MmadOperation>
__aicore__ inline constexpr auto make_mmad(const MmadOperation& operation)
{
    return mmad_atom<mmad_traits<MmadOperation>>{};
}

template <typename MmadOperation, typename MmadTrait>
__aicore__ inline constexpr auto make_mmad(const MmadOperation& operation, const MmadTrait& trait)
{
    return mmad_atom<mmad_traits<MmadOperation, MmadTrait>>{};
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ALGORITHM_MMAD_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
