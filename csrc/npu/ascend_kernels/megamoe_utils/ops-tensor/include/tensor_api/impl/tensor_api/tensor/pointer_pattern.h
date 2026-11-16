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
    "impl/tensor_api/tensor/pointer_pattern.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file pointer_pattern.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_H
#define IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/pointer_mem_impl.h"

namespace asc {
namespace te {

template <typename DataType = uint16_t>
struct ptr_trait {
    using type = DataType;
};

template <typename Trait, typename = void>
struct is_ptr_trait : Std::false_type {};

template <typename Trait>
struct is_ptr_trait<Trait, void_t<typename Trait::type>> : Std::true_type {};

template <typename Trait>
using mem_ptr_trait_t = typename Std::conditional<is_ptr_trait<Trait>::value, Trait, ptr_trait<Trait>>::type;

template <typename Hardware, typename PointerArg>
using enable_make_ptr_by_trait =
    Std::enable_if_t<is_hardware_v<Hardware> && !is_mem_ptr_iterator<Std::remove_cvref_t<PointerArg>>::value, int>;

template <typename Hardware, typename PointerArg>
using enable_make_hardware_ptr =
    Std::enable_if_t<is_hardware_v<Hardware> && is_mem_ptr_iterator<Std::remove_cvref_t<PointerArg>>::value, int>;

template <typename Iterator>
using enable_make_ptr_by_iter = Std::enable_if_t<is_mem_ptr_iterator<Std::remove_cvref_t<Iterator>>::value, int>;

template <typename PtrPattern, typename DataType, typename Addr, enable_make_ptr_by_trait<PtrPattern, Addr> = 0>
__aicore__ inline auto make_mem_ptr(Addr arg)
{
    using pointer = typename location_attr<DataType>::location_map::template get<PtrPattern>;
    return make_location_mem_ptr<PtrPattern>(reinterpret_cast<pointer>(asc_get_phy_buf_addr(0) + arg));
}

template <typename PtrPattern, typename Iterator, enable_make_hardware_ptr<PtrPattern, Iterator> = 0>
__aicore__ inline auto make_mem_ptr(Iterator iterator)
{
    return make_location_mem_ptr<PtrPattern>(iterator);
}

template <typename Iterator, enable_make_ptr_by_iter<Iterator> = 0>
__aicore__ inline auto make_mem_ptr(Iterator iterator)
{
    using ptr_pattern = get_attribute_location<typename iter_ele<Iterator>::type*>;
    return make_location_mem_ptr<ptr_pattern>(iterator);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
