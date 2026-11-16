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
    "impl/tensor_api/arch/utils/is_format.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file is_format.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H
#define IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/pointer_pattern.h"
#include "impl/tensor_api/tensor/tensor_impl.h"

namespace asc {
namespace te {

template <typename Value, bool IsTuple>
struct to_tuple_impl;

template <typename Value>
struct to_tuple_impl<Value, true> {
    using type = Value;
};

template <typename Value>
struct to_tuple_impl<Value, false> {
    using type = Std::tuple<Value>;
};

template <typename Value>
using to_tuple = to_tuple_impl<Value, Std::is_tuple_v<Value>>;

// Selects the (row, col) shape/stride sub-tuples from a layout's top-level (rowPart, colPart) pair.
// Batched layouts carry a leading Batch axis, so the top-level pair is (Int<B>, matrixTuple): the
// real row/col live one level down, inside the matrix tuple. Non-batched layouts use the pair as-is.
template <typename ShapeRows, typename ShapeCols, typename StrideRows, typename StrideCols, bool IsBatched>
struct select_row_col_tuples {
    using rows = ShapeRows;
    using cols = ShapeCols;
    using stride_rows_t = StrideRows;
    using stride_cols_t = StrideCols;
};

template <typename ShapeRows, typename ShapeCols, typename StrideRows, typename StrideCols>
struct select_row_col_tuples<ShapeRows, ShapeCols, StrideRows, StrideCols, true> {
    using rows = typename Std::tuple_element<0, ShapeCols>::type;
    using cols = typename Std::tuple_element<1, ShapeCols>::type;
    using stride_rows_t = typename Std::tuple_element<0, StrideCols>::type;
    using stride_cols_t = typename Std::tuple_element<1, StrideCols>::type;
};

template <typename Tensor>
struct get_type_from_n_dim_trait;

template <template <typename, typename> class TensorType, typename HPos, typename Pointer, typename ShapeRows,
          typename ShapeCols, typename StrideRows, typename StrideCols, typename LayoutPattern>
struct get_type_from_n_dim_trait<
    TensorType<view_engine<hardware_mem_ptr<HPos, Pointer>>,
               layout<shape<ShapeRows, ShapeCols>, stride<StrideRows, StrideCols>, LayoutPattern>>> {
    // A batched layout has a scalar Batch axis as rowPart and the matrix tuple as colPart.
    static constexpr bool is_batched = !Std::is_tuple_v<ShapeRows> && Std::is_tuple_v<ShapeCols>;
    using selector = select_row_col_tuples<ShapeRows, ShapeCols, StrideRows, StrideCols, is_batched>;

    using shape_row_tuple = typename to_tuple<typename selector::rows>::type;
    using shape_col_tuple = typename to_tuple<typename selector::cols>::type;
    using stride_row_tuple = typename to_tuple<typename selector::stride_rows_t>::type;
    using stride_col_tuple = typename to_tuple<typename selector::stride_cols_t>::type;

    template <size_t Dim>
    using shape_row_dim = typename Std::tuple_element<Dim, shape_row_tuple>::type;

    template <size_t Dim>
    using shape_col_dim = typename Std::tuple_element<Dim, shape_col_tuple>::type;

    template <size_t Dim>
    using stride_row_dim = typename Std::tuple_element<Dim, stride_row_tuple>::type;

    template <size_t Dim>
    using stride_col_dim = typename Std::tuple_element<Dim, stride_col_tuple>::type;
};

namespace attr_info {
struct shape {};
struct stride {};
struct row {};
struct column {};
}; // namespace attr_info

template <typename Tensor, typename Info1, typename Info2, size_t dim>
struct get_n_dim_type;

template <typename Tensor, size_t dim>
struct get_n_dim_type<Tensor, attr_info::shape, attr_info::row, dim> {
    using type = Std::remove_cvref_t<
        typename get_type_from_n_dim_trait<Std::remove_cvref_t<Tensor>>::template shape_row_dim<dim>>;
};

template <typename Tensor, size_t dim>
struct get_n_dim_type<Tensor, attr_info::shape, attr_info::column, dim> {
    using type = Std::remove_cvref_t<
        typename get_type_from_n_dim_trait<Std::remove_cvref_t<Tensor>>::template shape_col_dim<dim>>;
};

template <typename Tensor, size_t dim>
struct get_n_dim_type<Tensor, attr_info::stride, attr_info::row, dim> {
    using type = Std::remove_cvref_t<
        typename get_type_from_n_dim_trait<Std::remove_cvref_t<Tensor>>::template stride_row_dim<dim>>;
};

template <typename Tensor, size_t dim>
struct get_n_dim_type<Tensor, attr_info::stride, attr_info::column, dim> {
    using type = Std::remove_cvref_t<
        typename get_type_from_n_dim_trait<Std::remove_cvref_t<Tensor>>::template stride_col_dim<dim>>;
};

template <typename TensorType, typename TargetLayoutPtn>
struct is_satisfied_ptn_format {
    using layout_pattern = get_layout_pattern<typename Std::remove_cvref_t<TensorType>::layout_type>;
    static constexpr bool value = Std::is_same_v<layout_pattern, TargetLayoutPtn>;
};

template <typename TensorType, typename TargetLayoutPtn>
inline constexpr bool is_satisfied_ptn_format_v = is_satisfied_ptn_format<TensorType, TargetLayoutPtn>::value;

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
