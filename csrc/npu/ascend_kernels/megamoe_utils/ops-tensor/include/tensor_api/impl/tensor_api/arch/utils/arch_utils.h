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
    "impl/tensor_api/arch/utils/arch_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file arch_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_ARCH_UTILS_H
#define IMPL_TENSOR_API_ARCH_UTILS_ARCH_UTILS_H

#include "impl/tensor_api/arch/utils/check_data_type.h"
#include "impl/tensor_api/arch/utils/check_format.h"
#include "impl/tensor_api/arch/utils/is_format.h"

namespace asc {
namespace te {

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define SCALAR_QUANT_MODE                                                                                              \
    QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8, QuantMode_t::QS322BF16_PRE,                      \
        QuantMode_t::QF322F16_PRE, QuantMode_t::QF322BF16_PRE, QuantMode_t::QF322FP8_PRE, QuantMode_t::QF322HIF8_PRE,  \
        QuantMode_t::QF322HIF8_PRE_HYBRID, QuantMode_t::QF322F32_PRE
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define SCALAR_QUANT_MODE QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8
#else
#define SCALAR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quant_pre>
using is_scalar_quant_mode = Std::is_one_of_value<QuantMode_t, quant_pre, SCALAR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE                                                                             \
    QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8, QuantMode_t::VQS322BF16_PRE,                  \
        QuantMode_t::VQF322F16_PRE, QuantMode_t::VQF322BF16_PRE, QuantMode_t::VQF322FP8_PRE,                           \
        QuantMode_t::VQF322HIF8_PRE, QuantMode_t::VQF322HIF8_PRE_HYBRID, QuantMode_t::VQF322F32_PRE
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8
#else
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quant_pre>
using is_vector_quant_mode = Std::is_one_of_value<QuantMode_t, quant_pre, TILE_OP_INTERNAL_TENSOR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#else
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quant_pre>
using is_direct_quant_mode = Std::is_one_of_value<QuantMode_t, quant_pre, TILE_OP_INTERNAL_DIRECT_QUANT_MODE>;

template <typename Info1, typename Info2, size_t dim, typename Layout>
__aicore__ inline constexpr decltype(auto) get_element(const Layout& layout)
{
    constexpr size_t shape_or_stride = (Std::is_same_v<Info1, attr_info::shape> ? 0 : 1);
    constexpr size_t row_or_column = (Std::is_same_v<Info2, attr_info::row> ? 0 : 1);
    return layout.template get<shape_or_stride, row_or_column, dim>();
}

template <typename Info1, typename Info2, typename Layout>
__aicore__ inline constexpr decltype(auto) get_element(const Layout& layout)
{
    constexpr size_t shape_or_stride = (Std::is_same_v<Info1, attr_info::shape> ? 0 : 1);
    constexpr size_t row_or_column = (Std::is_same_v<Info2, attr_info::row> ? 0 : 1);
    return layout.template get<shape_or_stride, row_or_column>();
}

template <typename AttributeType, typename AxisType, size_t dim, typename Layout>
__aicore__ inline constexpr decltype(auto) get_matrix_element(const Layout& layout)
{
    // Shape and stride are stored at indices 0 and 1 of a layout, respectively.
    constexpr size_t attribute_index = Std::is_same_v<AttributeType, attr_info::shape> ? 0 : 1;
    // Row and column are stored at indices 0 and 1 of a matrix attribute; batched layouts skip axis 0 below.
    constexpr size_t axis_index = Std::is_same_v<AxisType, attr_info::row> ? 0 : 1;
    if constexpr (Layout::depth == three_dim_data || Layout::depth == five_dim_data) {
        return layout.template get<attribute_index, 1, axis_index, dim>();
    } else {
        return layout.template get<attribute_index, axis_index, dim>();
    }
}

template <typename LayoutType>
__aicore__ inline static constexpr uint32_t get_total_column_shape(const LayoutType& layout)
{
    if constexpr (LayoutType::depth == 2) {
        return get<1>(layout.shape());
    } else {
        return get<1, 0>(layout.shape()) * get<1, 1>(layout.shape());
    }
}

template <typename LayoutType>
__aicore__ inline static constexpr uint32_t get_total_row_shape(const LayoutType& layout)
{
    if constexpr (LayoutType::depth == 2) {
        return get<0>(layout.shape());
    } else {
        return get<0, 0>(layout.shape()) * get<0, 1>(layout.shape());
    }
}

template <typename LayoutType>
__aicore__ inline static constexpr decltype(auto) get_row_stride(const LayoutType& layout)
{
    if constexpr (LayoutType::depth == two_dim_data) {
        return get<0>(layout.stride());
    } else if constexpr (LayoutType::depth == three_dim_data) {
        return get<1, 0>(layout.stride());
    } else if constexpr (LayoutType::depth == four_dim_data) {
        return get<0, 1>(layout.stride());
    } else {
        static_assert(LayoutType::depth == five_dim_data, "Only support 2D to 5D matrix layouts");
        return get<1, 0, 1>(layout.stride());
    }
}

template <typename LayoutType>
__aicore__ inline static constexpr decltype(auto) get_column_stride(const LayoutType& layout)
{
    if constexpr (LayoutType::depth == two_dim_data) {
        return get<1>(layout.stride());
    } else if constexpr (LayoutType::depth == three_dim_data) {
        return get<1, 1>(layout.stride());
    } else if constexpr (LayoutType::depth == four_dim_data) {
        return get<1, 1>(layout.stride());
    } else {
        static_assert(LayoutType::depth == five_dim_data, "Only support 2D to 5D matrix layouts");
        return get<1, 1, 1>(layout.stride());
    }
}
template <typename Shape>
__aicore__ inline static constexpr uint32_t get_shape_columns(const Shape& copy_shape)
{
    constexpr auto depth = nesting_depth_v<Shape>;
    if constexpr (depth == two_dim_data) {
        return get<1>(copy_shape);
    } else if constexpr (depth == three_dim_data) {
        return get<1, 1>(copy_shape);
    } else if constexpr (depth == four_dim_data) {
        return get<1, 0>(copy_shape) * get<1, 1>(copy_shape);
    } else {
        static_assert(depth == five_dim_data, "Only support 2D to 5D matrix shapes");
        return get<1, 1, 0>(copy_shape) * get<1, 1, 1>(copy_shape);
    }
}

template <typename Shape>
__aicore__ inline static constexpr uint32_t get_shape_rows(const Shape& copy_shape)
{
    constexpr auto depth = nesting_depth_v<Shape>;
    if constexpr (depth == two_dim_data) {
        return get<0>(copy_shape);
    } else if constexpr (depth == three_dim_data) {
        return get<1, 0>(copy_shape);
    } else if constexpr (depth == four_dim_data) {
        return get<0, 0>(copy_shape) * get<0, 1>(copy_shape);
    } else {
        static_assert(depth == five_dim_data, "Only support 2D to 5D matrix shapes");
        return get<1, 0, 0>(copy_shape) * get<1, 0, 1>(copy_shape);
    }
}

template <typename Shape>
__aicore__ inline static constexpr uint32_t get_shape_batch_size(const Shape& copy_shape)
{
    constexpr auto depth = nesting_depth_v<Shape>;
    if constexpr (depth == three_dim_data || depth == five_dim_data) {
        return get<0>(copy_shape);
    } else {
        return 1;
    }
}
} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_UTILS_ARCH_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
