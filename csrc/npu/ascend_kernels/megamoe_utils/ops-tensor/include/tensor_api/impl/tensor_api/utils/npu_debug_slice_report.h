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
    "impl/tensor_api/utils/npu_debug_slice_report.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define TENSOR_API_DEBUG_SLICE_REPORT_OWNS_INTERNAL_HEADER_ACCESS
#endif

/*!
 * \file npu_debug_slice_report.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_NPU_DEBUG_SLICE_REPORT_H
#define IMPL_TENSOR_API_UTILS_NPU_DEBUG_SLICE_REPORT_H

#include "impl/tensor_api/utils/npu_debug_assert.h"

#ifdef ASCENDC_DEBUG
namespace asc {
namespace te {

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType, size_t... source_indices, size_t... coord_indices, size_t... slice_indices>
__aicore__ inline void report_slice_error_by_format(
    debug_slice_same_format_tag, const tensor_api_assert_context& context, const LayoutType& layout,
    const CoordType& coord, const InfoType& info, const SourceShapeType& source_shape,
    const SliceShapeType& slice_shape, Std::index_sequence<source_indices...>, Std::index_sequence<coord_indices...>,
    Std::index_sequence<slice_indices...>)
{
    using view_type = debug_layout_view<LayoutType>;
    using slice_view_type = debug_slice_info_view<InfoType>;
    using source_format_type = debug_tuple_format_t<SourceShapeType>;
    if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_1>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT("(%lld)", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT("(%lld, %lld)", "(%lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT("(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT("(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_5>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT(
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_6>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT(
            "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT("(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT(
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))");
    } else {
        TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))");
    }
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType, size_t... source_indices, size_t... coord_indices, size_t... slice_indices>
__aicore__ inline void report_slice_error_by_format(
    debug_slice_flat_source_tag, const tensor_api_assert_context& context, const LayoutType& layout,
    const CoordType& coord, const InfoType& info, const SourceShapeType& source_shape,
    const SliceShapeType& slice_shape, Std::index_sequence<source_indices...>, Std::index_sequence<coord_indices...>,
    Std::index_sequence<slice_indices...>)
{
    using view_type = debug_layout_view<LayoutType>;
    using slice_view_type = debug_slice_info_view<InfoType>;
    using coord_format_type = debug_tuple_format_t<CoordType>;
    if constexpr (Std::is_same_v<coord_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "((%lld, %lld), (%lld, %lld))", "((%lld, %lld), (%lld, %lld))",
            "(%lld, (%lld, %lld), (%lld, %lld))");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, (%lld, %lld))", "(%lld, (%lld, %lld))",
            "(%lld, %lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, (%lld, %lld), (%lld, %lld))",
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))");
    } else {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, ((%lld, %lld), (%lld, %lld)))",
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))");
    }
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType, size_t... source_indices, size_t... coord_indices, size_t... slice_indices>
__aicore__ inline void report_slice_error_by_format(
    debug_slice_nested_source_tag, const tensor_api_assert_context& context, const LayoutType& layout,
    const CoordType& coord, const InfoType& info, const SourceShapeType& source_shape,
    const SliceShapeType& slice_shape, Std::index_sequence<source_indices...>, Std::index_sequence<coord_indices...>,
    Std::index_sequence<slice_indices...>)
{
    using view_type = debug_layout_view<LayoutType>;
    using slice_view_type = debug_slice_info_view<InfoType>;
    using coord_format_type = debug_tuple_format_t<CoordType>;
    if constexpr (Std::is_same_v<coord_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld)", "(%lld, %lld)",
            "(%lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld))",
            "(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))",
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))");
    } else {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))",
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, ((%lld, %lld), (%lld, %lld)))",
            "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))");
    }
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType, size_t... source_indices, size_t... coord_indices, size_t... slice_indices>
__aicore__ inline void report_slice_error_by_format(
    debug_slice_rank_expanded_tag, const tensor_api_assert_context& context, const LayoutType& layout,
    const CoordType& coord, const InfoType& info, const SourceShapeType& source_shape,
    const SliceShapeType& slice_shape, Std::index_sequence<source_indices...>, Std::index_sequence<coord_indices...>,
    Std::index_sequence<slice_indices...>)
{
    using view_type = debug_layout_view<LayoutType>;
    using slice_view_type = debug_slice_info_view<InfoType>;
    using source_format_type = debug_tuple_format_t<SourceShapeType>;
    if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)",
            "(%lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<source_format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)",
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)");
    } else {
        TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)",
            "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)",
            "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)");
    }
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType, size_t... source_indices, size_t... coord_indices, size_t... slice_indices>
__aicore__ inline void report_slice_error_by_format(
    debug_slice_unsupported_format_tag, const tensor_api_assert_context& context, const LayoutType& layout,
    const CoordType& coord, const InfoType& info, const SourceShapeType& source_shape,
    const SliceShapeType& slice_shape, Std::index_sequence<source_indices...>, Std::index_sequence<coord_indices...>,
    Std::index_sequence<slice_indices...>)
{
    report_slice_unsupported<kind>(context, layout, coord, info, source_shape, slice_shape);
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType, size_t... source_indices, size_t... coord_indices, size_t... slice_indices>
__aicore__ inline void report_slice_error_impl(
    const tensor_api_assert_context& context, const LayoutType& layout, const CoordType& coord, const InfoType& info,
    const SourceShapeType& source_shape, const SliceShapeType& slice_shape,
    Std::index_sequence<source_indices...> source_seq, Std::index_sequence<coord_indices...> coord_seq,
    Std::index_sequence<slice_indices...> slice_seq)
{
    using category_type = debug_slice_format_category_t<
        debug_tuple_format_t<SourceShapeType>, debug_tuple_format_t<CoordType>, debug_tuple_format_t<SliceShapeType>>;
    report_slice_error_by_format<kind>(
        category_type{}, context, layout, coord, info, source_shape, slice_shape, source_seq, coord_seq, slice_seq);
}

template <slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType>
__aicore__ inline void report_slice_error(
    const tensor_api_assert_context& context, const LayoutType& layout, const CoordType& coord, const InfoType& info)
{
    auto source_shape = debug_layout_view<LayoutType>::shape(layout);
    auto slice_shape = debug_slice_info_view<InfoType>::shape(info);
    report_slice_error_impl<kind>(
        context, layout, coord, info, source_shape, slice_shape,
        Std::make_index_sequence<nesting_depth_v<decltype(source_shape)>>{},
        Std::make_index_sequence<nesting_depth_v<CoordType>>{},
        Std::make_index_sequence<nesting_depth_v<decltype(slice_shape)>>{});
}

#undef TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT
#undef TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS
#undef TENSOR_API_DETAIL_SLICE_FAIL

} // namespace te
} // namespace asc
#endif // ASCENDC_DEBUG

#endif // IMPL_TENSOR_API_UTILS_NPU_DEBUG_SLICE_REPORT_H

#if defined(TENSOR_API_DEBUG_SLICE_REPORT_OWNS_INTERNAL_HEADER_ACCESS)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef TENSOR_API_DEBUG_SLICE_REPORT_OWNS_INTERNAL_HEADER_ACCESS
#endif
