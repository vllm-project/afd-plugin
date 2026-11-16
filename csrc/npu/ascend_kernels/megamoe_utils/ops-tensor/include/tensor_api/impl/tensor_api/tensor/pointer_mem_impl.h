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
    "impl/tensor_api/tensor/pointer_mem_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file pointer_mem_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_MEM_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_MEM_IMPL_H

#include "impl/tensor_api/tensor/pointer_adaptor_impl.h"
namespace asc {
namespace te {

template <typename Tensor>
using get_mem_location = typename Tensor::iterator::ptr_pattern;

template <typename PtrPattern, typename Pointer>
struct hardware_mem_ptr : iter_adaptor<Pointer, hardware_mem_ptr<PtrPattern, Pointer>> {
    using iter_adaptor<Pointer, hardware_mem_ptr<PtrPattern, Pointer>>::iter_adaptor;
    using ptr_pattern = PtrPattern;
};

// is hardware mem
template <typename PtrPattern, typename Pointer, typename = void>
struct is_hardware_mem : Std::false_type {};

template <typename PtrPattern, typename Pointer>
struct is_hardware_mem<PtrPattern, hardware_mem_ptr<PtrPattern, Pointer>> : Std::true_type {};

template <typename PtrPattern, typename Pointer>
struct is_hardware_mem<PtrPattern, Pointer, void_t<typename Pointer::iterator>>
    : is_hardware_mem<PtrPattern, typename Pointer::iterator> {};

template <typename PtrPattern, typename Pointer>
constexpr bool is_hardware_mem_v = is_hardware_mem<PtrPattern, Pointer>::value;

template <typename Pointer>
struct is_hardware_mem_ptr : Std::false_type {};

template <typename PtrPattern, typename Pointer>
struct is_hardware_mem_ptr<hardware_mem_ptr<PtrPattern, Pointer>> : Std::true_type {};

template <typename Pointer>
constexpr bool is_hardware_mem_ptr_v = is_hardware_mem_ptr<Std::remove_cvref_t<Pointer>>::value;

template <typename Iterator, typename = void>
struct is_mem_ptr_iterator : Std::false_type {};

template <typename Iterator>
struct is_mem_ptr_iterator<Iterator, void_t<decltype(*Std::declval<Iterator&>())>> : Std::true_type {};

template <typename PtrPattern, typename Iterator>
__aicore__ inline auto make_location_mem_ptr(Iterator iter)
{
    return hardware_mem_ptr<PtrPattern, Iterator>{iter};
}
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_POINTER_MEM_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
