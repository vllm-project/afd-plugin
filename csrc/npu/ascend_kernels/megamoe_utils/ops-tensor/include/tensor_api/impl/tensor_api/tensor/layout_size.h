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
    "impl/tensor_api/tensor/layout_size.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_size.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_SIZE_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_SIZE_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename Value>
struct nesting_depth {
    static constexpr size_t value = 1;
};

template <>
struct nesting_depth<Std::tuple<>> {
    static constexpr size_t value = 0;
};

template <typename... Args>
struct nesting_depth<Std::tuple<Args...>> {
    static constexpr size_t value = (nesting_depth<Args>::value + ...);
};

template <typename Value>
constexpr size_t nesting_depth_v = nesting_depth<Std::remove_cvref_t<Value>>::value;

template <typename Value>
constexpr size_t NestingDepthV = nesting_depth_v<Value>;

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_SIZE_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
