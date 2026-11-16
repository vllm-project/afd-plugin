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
    "impl/tensor_api/tensor/layout_fractal.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_fractal.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H

#include "impl/tensor_api/utils/utils_impl.h"

namespace asc {
namespace te {

template <typename OriginShape, typename InnerShape, size_t... Is>
__aicore__ inline constexpr decltype(auto) make_fractal_shape(OriginShape origin_shape, InnerShape inner_shape,
                                                              Std::index_sequence<Is...>)
{
    auto outer_shape = Std::make_tuple(Std::ceil_division(Std::get<Is>(origin_shape), Std::get<Is>(inner_shape))...);
    return make_shape(make_shape(Std::get<Is>(inner_shape), Std::get<Is>(outer_shape))...);
}

template <typename OriginShape, typename InnerShape>
__aicore__ inline constexpr decltype(auto) make_fractal_shape(OriginShape origin_shape, InnerShape inner_shape)
{
    static_assert(Std::tuple_size_v<OriginShape> == Std::tuple_size_v<InnerShape>,
                  "OriginShape and InnerShape must match");
    return make_fractal_shape(origin_shape, inner_shape, Std::make_index_sequence<Std::tuple_size_v<InnerShape>>{});
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
