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
 * \file pointer.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_POINTER_H
#define INCLUDE_TENSOR_API_TENSOR_POINTER_H

#include "impl/tensor_api/tensor/pointer_pattern.h"

namespace asc {
namespace te {

template <typename PtrPattern, typename DataType, typename Addr, enable_make_ptr_by_trait<PtrPattern, Addr> Enable>
__aicore__ inline auto make_mem_ptr(Addr address);

template <typename PtrPattern, typename Iterator, enable_make_hardware_ptr<PtrPattern, Iterator> Enable>
__aicore__ inline auto make_mem_ptr(Iterator iterator);

template <typename Iterator, enable_make_ptr_by_iter<Iterator> Enable>
__aicore__ inline auto make_mem_ptr(Iterator iterator);

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_TENSOR_POINTER_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
