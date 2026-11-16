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
#warning \
    "impl/tensor_api/tensor/layout_infer.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_infer.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_INFER_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_INFER_H

#include "impl/tensor_api/tensor/layout_frame.h"

namespace asc {
namespace te {

constexpr size_t layout_infer_priority_step = 1;
constexpr size_t layout_infer_priority_begin = 0;
constexpr size_t layout_infer_priority_nd_ext = layout_infer_priority_begin;
constexpr size_t layout_infer_priority_dn_ext = layout_infer_priority_nd_ext + layout_infer_priority_step;
constexpr size_t layout_infer_priority_nn = layout_infer_priority_dn_ext + layout_infer_priority_step;
constexpr size_t layout_infer_priority_nz = layout_infer_priority_nn + layout_infer_priority_step;
constexpr size_t layout_infer_priority_zz = layout_infer_priority_nz + layout_infer_priority_step;
constexpr size_t layout_infer_priority_zn = layout_infer_priority_zz + layout_infer_priority_step;
constexpr size_t layout_infer_priority_scalea_dn = layout_infer_priority_zn + layout_infer_priority_step;
constexpr size_t layout_infer_priority_scaleb_nd = layout_infer_priority_scalea_dn + layout_infer_priority_step;
constexpr size_t layout_infer_priority_end = layout_infer_priority_scaleb_nd + layout_infer_priority_step;
constexpr size_t layout_infer_first_element = 0;
constexpr size_t layout_infer_second_element = layout_infer_first_element + layout_infer_priority_step;

template <typename RowStride, typename ColumnStride>
struct infer_flat_layout_pattern_impl {
    using type = Std::ignore_t;
};

template <typename RowStride>
struct infer_flat_layout_pattern_impl<RowStride, _1> {
    using type = nd_layout_ptn;
};

template <typename ColumnStride>
struct infer_flat_layout_pattern_impl<_1, ColumnStride> {
    using type = dn_layout_ptn;
};

template <>
struct infer_flat_layout_pattern_impl<_1, _1> {
    using type = nd_layout_ptn;
};

template <typename RowStride, typename ColumnStride>
struct infer_flat_layout_pattern
    : infer_flat_layout_pattern_impl<Std::remove_cvref_t<RowStride>, Std::remove_cvref_t<ColumnStride>> {};

template <
    typename ShapeRow0, typename ShapeRow1, typename ShapeColumn0, typename ShapeColumn1, typename StrideRow0,
    typename StrideRow1, typename StrideColumn0, typename StrideColumn1, size_t Priority = layout_infer_priority_begin,
    typename Enable = void>
struct infer_nested_layout_pattern_impl {
    using type = typename infer_nested_layout_pattern_impl<
        ShapeRow0, ShapeRow1, ShapeColumn0, ShapeColumn1, StrideRow0, StrideRow1, StrideColumn0, StrideColumn1,
        Priority + layout_infer_priority_step>::type;
};

template <
    typename ShapeRow0, typename ShapeRow1, typename ShapeColumn0, typename ShapeColumn1, typename StrideRow0,
    typename StrideRow1, typename StrideColumn0, typename StrideColumn1>
struct infer_nested_layout_pattern_impl<
    ShapeRow0, ShapeRow1, ShapeColumn0, ShapeColumn1, StrideRow0, StrideRow1, StrideColumn0, StrideColumn1,
    layout_infer_priority_end> {
    using type = Std::ignore_t;
};

template <typename ShapeRow1, typename ShapeColumn1, typename StrideRow1>
struct infer_nested_layout_pattern_impl<
    _1, ShapeRow1, _1, ShapeColumn1, _0, StrideRow1, _0, _1, layout_infer_priority_nd_ext> {
    using type = nd_ext_layout_ptn;
};

template <typename ShapeRow1, typename ShapeColumn1, typename StrideColumn1>
struct infer_nested_layout_pattern_impl<
    _1, ShapeRow1, _1, ShapeColumn1, _0, _1, _0, StrideColumn1, layout_infer_priority_dn_ext> {
    using type = dn_ext_layout_ptn;
};

template <typename ShapeRow1, typename ShapeColumn1, typename StrideColumn1>
struct infer_nested_layout_pattern_impl<
    _2, ShapeRow1, _16, ShapeColumn1, _1, _32, _2, StrideColumn1, layout_infer_priority_nn> {
    using type = nn_layout_ptn;
};

template <size_t ShapeRow0, typename ShapeRow1, size_t ShapeColumn0, typename ShapeColumn1, typename StrideColumn1>
struct infer_nested_layout_pattern_impl<
    Std::Int<ShapeRow0>, ShapeRow1, Std::Int<ShapeColumn0>, ShapeColumn1, Std::Int<ShapeColumn0>,
    Std::Int<ShapeRow0 * ShapeColumn0>, _1, StrideColumn1, layout_infer_priority_nz,
    Std::enable_if_t<ShapeRow0 == fractal_fixed>> {
    using type = nz_layout_ptn;
};

template <size_t ShapeRow0, typename ShapeRow1, size_t ShapeColumn0, typename ShapeColumn1, typename StrideRow1>
struct infer_nested_layout_pattern_impl<
    Std::Int<ShapeRow0>, ShapeRow1, Std::Int<ShapeColumn0>, ShapeColumn1, Std::Int<ShapeColumn0>, StrideRow1, _1,
    Std::Int<ShapeRow0 * ShapeColumn0>, layout_infer_priority_zz, Std::enable_if_t<ShapeRow0 == fractal_fixed>> {
    using type = zz_layout_ptn;
};

template <size_t ShapeRow0, typename ShapeRow1, size_t ShapeColumn0, typename ShapeColumn1, typename StrideRow1>
struct infer_nested_layout_pattern_impl<
    Std::Int<ShapeRow0>, ShapeRow1, Std::Int<ShapeColumn0>, ShapeColumn1, _1, StrideRow1, Std::Int<ShapeRow0>,
    Std::Int<ShapeRow0 * ShapeColumn0>, layout_infer_priority_zn, Std::enable_if_t<ShapeColumn0 == fractal_fixed>> {
    using type = zn_layout_ptn;
};

template <typename ShapeRow1, typename ShapeColumn1, typename StrideColumn1>
struct infer_nested_layout_pattern_impl<
    _1, ShapeRow1, _2, ShapeColumn1, _0, _2, _1, StrideColumn1, layout_infer_priority_scalea_dn> {
    using type = scalea_dn_layout_ptn;
};

template <typename ShapeRow1, typename ShapeColumn1, typename StrideRow1>
struct infer_nested_layout_pattern_impl<
    _2, ShapeRow1, _1, ShapeColumn1, _1, StrideRow1, _0, _2, layout_infer_priority_scaleb_nd> {
    using type = scaleb_nd_layout_ptn;
};

template <typename ShapeRows, typename ShapeColumns, typename StrideRows, typename StrideColumns>
struct infer_nested_layout_pattern {
    using type = Std::ignore_t;
};

template <
    typename ShapeRow0, typename ShapeRow1, typename ShapeColumn0, typename ShapeColumn1, typename StrideRow0,
    typename StrideRow1, typename StrideColumn0, typename StrideColumn1>
struct infer_nested_layout_pattern<
    shape<ShapeRow0, ShapeRow1>, shape<ShapeColumn0, ShapeColumn1>, stride<StrideRow0, StrideRow1>,
    stride<StrideColumn0, StrideColumn1>>
    : infer_nested_layout_pattern_impl<
          Std::remove_cvref_t<ShapeRow0>, Std::remove_cvref_t<ShapeRow1>, Std::remove_cvref_t<ShapeColumn0>,
          Std::remove_cvref_t<ShapeColumn1>, Std::remove_cvref_t<StrideRow0>, Std::remove_cvref_t<StrideRow1>,
          Std::remove_cvref_t<StrideColumn0>, Std::remove_cvref_t<StrideColumn1>> {};

template <
    typename Row, typename Column, typename RowStride, typename ColumnStride,
    bool RowIsTuple = Std::is_tuple_v<Std::remove_cvref_t<Row>>,
    bool ColumnIsTuple = Std::is_tuple_v<Std::remove_cvref_t<Column>>,
    bool RowStrideIsTuple = Std::is_tuple_v<Std::remove_cvref_t<RowStride>>,
    bool ColumnStrideIsTuple = Std::is_tuple_v<Std::remove_cvref_t<ColumnStride>>>
struct infer_two_dim_layout_pattern {
    using type = Std::ignore_t;
};

template <typename Row, typename Column, typename RowStride, typename ColumnStride>
struct infer_two_dim_layout_pattern<Row, Column, RowStride, ColumnStride, false, false, false, false>
    : infer_flat_layout_pattern<RowStride, ColumnStride> {};

template <typename Row, typename Column, typename RowStride, typename ColumnStride>
struct infer_two_dim_layout_pattern<Row, Column, RowStride, ColumnStride, true, true, true, true>
    : infer_nested_layout_pattern<Row, Column, RowStride, ColumnStride> {};

template <typename Batch, typename MatrixShape, typename BatchStride, typename MatrixStride>
struct infer_two_dim_layout_pattern<Batch, MatrixShape, BatchStride, MatrixStride, false, true, false, true>
    : infer_two_dim_layout_pattern<
          typename Std::tuple_element<layout_infer_first_element, MatrixShape>::type,
          typename Std::tuple_element<layout_infer_second_element, MatrixShape>::type,
          typename Std::tuple_element<layout_infer_first_element, MatrixStride>::type,
          typename Std::tuple_element<layout_infer_second_element, MatrixStride>::type> {};

template <typename ShapeType, typename StrideType>
struct infer_layout_pattern {
    using type = Std::ignore_t;
};

template <typename Size, typename Stride>
struct infer_layout_pattern<shape<Size>, stride<Stride>> {
    using type = one_dim_layout_ptn;
};

template <typename Row, typename Column, typename RowStride, typename ColumnStride>
struct infer_layout_pattern<shape<Row, Column>, stride<RowStride, ColumnStride>>
    : infer_two_dim_layout_pattern<Row, Column, RowStride, ColumnStride> {};

template <typename Pattern, typename C0 = _1>
struct build_inferred_layout_trait {
    using type =
        Std::conditional_t<Std::is_same_v<Pattern, Std::ignore_t>, Std::ignore_t, layout_trait<Std::ignore_t, C0>>;
};

template <typename ShapeType, typename StrideType>
struct infer_layout_trait {
    using type = Std::ignore_t;
};

template <typename Size, typename Stride>
struct infer_layout_trait<shape<Size>, stride<Stride>> {
    using type = layout_trait<Std::ignore_t, _1>;
};

template <typename Pattern, typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type {
    using type = _1;
};

template <typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type<nz_layout_ptn, ShapeRow0, ShapeColumn0> {
    using type = ShapeColumn0;
};

template <typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type<zz_layout_ptn, ShapeRow0, ShapeColumn0> {
    using type = ShapeColumn0;
};

template <typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type<zn_layout_ptn, ShapeRow0, ShapeColumn0> {
    using type = ShapeRow0;
};

template <typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type<nn_layout_ptn, ShapeRow0, ShapeColumn0> {
    using type = ShapeRow0;
};

template <typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type<scaleb_nd_layout_ptn, ShapeRow0, ShapeColumn0> {
    using type = ShapeRow0;
};

template <typename ShapeRow0, typename ShapeColumn0>
struct infer_layout_c0_type<scalea_dn_layout_ptn, ShapeRow0, ShapeColumn0> {
    using type = ShapeColumn0;
};

template <typename Row, typename Column, typename RowStride, typename ColumnStride>
struct infer_layout_trait<shape<Row, Column>, stride<RowStride, ColumnStride>> {
private:
    using pattern_type = typename infer_layout_pattern<shape<Row, Column>, stride<RowStride, ColumnStride>>::type;

public:
    using type = typename build_inferred_layout_trait<pattern_type>::type;
};

template <
    typename ShapeRow0, typename ShapeRow1, typename ShapeColumn0, typename ShapeColumn1, typename StrideRow0,
    typename StrideRow1, typename StrideColumn0, typename StrideColumn1>
struct infer_layout_trait<
    shape<shape<ShapeRow0, ShapeRow1>, shape<ShapeColumn0, ShapeColumn1>>,
    stride<stride<StrideRow0, StrideRow1>, stride<StrideColumn0, StrideColumn1>>> {
private:
    using layout_shape = shape<shape<ShapeRow0, ShapeRow1>, shape<ShapeColumn0, ShapeColumn1>>;
    using layout_stride = stride<stride<StrideRow0, StrideRow1>, stride<StrideColumn0, StrideColumn1>>;
    using pattern_type = typename infer_layout_pattern<layout_shape, layout_stride>::type;
    using c0_type = typename infer_layout_c0_type<pattern_type, ShapeRow0, ShapeColumn0>::type;

public:
    using type = typename build_inferred_layout_trait<pattern_type, c0_type>::type;
};

template <
    typename Batch, typename ShapeRow0, typename ShapeRow1, typename ShapeColumn0, typename ShapeColumn1,
    typename BatchStride, typename StrideRow0, typename StrideRow1, typename StrideColumn0, typename StrideColumn1>
struct infer_layout_trait<
    shape<Batch, shape<shape<ShapeRow0, ShapeRow1>, shape<ShapeColumn0, ShapeColumn1>>>,
    stride<BatchStride, stride<stride<StrideRow0, StrideRow1>, stride<StrideColumn0, StrideColumn1>>>>
    : infer_layout_trait<
          shape<shape<ShapeRow0, ShapeRow1>, shape<ShapeColumn0, ShapeColumn1>>,
          stride<stride<StrideRow0, StrideRow1>, stride<StrideColumn0, StrideColumn1>>> {};

template <typename Shape, typename Stride>
struct get_layout_info<layout<Shape, Stride, Std::ignore_t>> {
    using type = Std::tuple<
        typename infer_layout_pattern<Shape, Stride>::type, typename infer_layout_trait<Shape, Stride>::type>;
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_INFER_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
