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
    "impl/tensor_api/utils/npu_debug_coord_report.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define TENSOR_API_DEBUG_COORD_REPORT_OWNS_INTERNAL_HEADER_ACCESS
#endif

/*!
 * \file npu_debug_coord_report.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_NPU_DEBUG_COORD_REPORT_H
#define IMPL_TENSOR_API_UTILS_NPU_DEBUG_COORD_REPORT_H

#include "impl/tensor_api/utils/npu_debug_assert.h"

#ifdef ASCENDC_DEBUG
namespace asc {
namespace te {

template <typename ValueType>
struct is_debug_trailing_scalar_flat_2 : Std::false_type {};

template <typename Value0Type, typename Value1Type, typename TrailingType>
struct is_debug_trailing_scalar_flat_2<Std::tuple<Std::tuple<Value0Type, Value1Type>, TrailingType>>
    : Std::bool_constant<are_debug_scalar_values_v<Value0Type, Value1Type, TrailingType>> {};

template <typename ValueType>
constexpr bool is_debug_trailing_scalar_flat_2_v =
    is_debug_trailing_scalar_flat_2<Std::remove_cvref_t<ValueType>>::value;

template <typename ValueType>
struct is_debug_leading_scalar_first_nested_2 : Std::false_type {};

template <typename LeadingType, typename Value0Type, typename Value1Type, typename TrailingType>
struct is_debug_leading_scalar_first_nested_2<
    Std::tuple<LeadingType, Std::tuple<Std::tuple<Value0Type, Value1Type>, TrailingType>>>
    : Std::bool_constant<are_debug_scalar_values_v<LeadingType, Value0Type, Value1Type, TrailingType>> {};

template <typename ValueType>
constexpr bool is_debug_leading_scalar_first_nested_2_v =
    is_debug_leading_scalar_first_nested_2<Std::remove_cvref_t<ValueType>>::value;

template <typename ValueType>
struct is_debug_leading_scalar_second_nested_2 : Std::false_type {};

template <typename LeadingType, typename FirstType, typename Value0Type, typename Value1Type>
struct is_debug_leading_scalar_second_nested_2<
    Std::tuple<LeadingType, Std::tuple<FirstType, Std::tuple<Value0Type, Value1Type>>>>
    : Std::bool_constant<are_debug_scalar_values_v<LeadingType, FirstType, Value0Type, Value1Type>> {};

template <typename ValueType>
constexpr bool is_debug_leading_scalar_second_nested_2_v =
    is_debug_leading_scalar_second_nested_2<Std::remove_cvref_t<ValueType>>::value;
struct debug_coord_layout_report_tag {};
struct debug_coord_shape_report_tag {};
struct debug_coord_pruned_format_tag {};

#define TENSOR_API_DETAIL_COORD_FAIL(fields_format, ...) \
    TENSOR_API_DETAIL_ASSERT_FAIL(                       \
        context, "Failed to check coord in %s, " fields_format "; coord must be within shape.", api_name, __VA_ARGS__)

#define TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(shape_format, batch_shape_format, coord_format)               \
    do {                                                                                                          \
        if constexpr (Std::is_same_v<ReportTag, debug_coord_layout_report_tag>) {                                 \
            using view_type = debug_layout_view<SourceType>;                                                      \
            if constexpr (view_type::has_batch) {                                                                 \
                TENSOR_API_DETAIL_COORD_FAIL(                                                                     \
                    "layoutPtn=%s, shape=" batch_shape_format ", coord=" coord_format, view_type::pattern_name(), \
                    static_cast<long long>(view_type::batch(source)),                                             \
                    static_cast<long long>(get_debug_tuple_leaf<shape_indices>(shape))...,                        \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...);                       \
            } else {                                                                                              \
                TENSOR_API_DETAIL_COORD_FAIL(                                                                     \
                    "layoutPtn=%s, shape=" shape_format ", coord=" coord_format, view_type::pattern_name(),       \
                    static_cast<long long>(get_debug_tuple_leaf<shape_indices>(shape))...,                        \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...);                       \
            }                                                                                                     \
        } else {                                                                                                  \
            static_assert(                                                                                        \
                Std::is_same_v<ReportTag, debug_coord_shape_report_tag>, "Unsupported coord report source.");     \
            TENSOR_API_DETAIL_COORD_FAIL(                                                                         \
                "shape=" shape_format ", coord=" coord_format,                                                    \
                static_cast<long long>(get_debug_tuple_leaf<shape_indices>(shape))...,                            \
                static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...);                           \
        }                                                                                                         \
    } while (0)

template <typename ReportTag, typename SourceType, typename ShapeType, typename CoordType>
__aicore__ inline void report_coord_unsupported(
    ReportTag, const tensor_api_assert_context& context, const SourceType& source, const ShapeType&, const CoordType&,
    __gm__ const char* api_name)
{
    if constexpr (Std::is_same_v<ReportTag, debug_coord_layout_report_tag>) {
        using view_type = debug_layout_view<SourceType>;
        if constexpr (view_type::has_batch) {
            TENSOR_API_DETAIL_COORD_FAIL(
                "layoutPtn=%s, values=(%lld, <unsupported shape/coord tuple combination>), shapeLeafCount=%u, "
                "coordLeafCount=%u",
                view_type::pattern_name(), static_cast<long long>(view_type::batch(source)),
                static_cast<unsigned int>(nesting_depth_v<ShapeType> + 1),
                static_cast<unsigned int>(nesting_depth_v<CoordType>));
        } else {
            TENSOR_API_DETAIL_COORD_FAIL(
                "layoutPtn=%s, values=<unsupported shape/coord tuple combination>, shapeLeafCount=%u, "
                "coordLeafCount=%u",
                view_type::pattern_name(), static_cast<unsigned int>(nesting_depth_v<ShapeType>),
                static_cast<unsigned int>(nesting_depth_v<CoordType>));
        }
    } else {
        static_assert(Std::is_same_v<ReportTag, debug_coord_shape_report_tag>, "Unsupported coord report source.");
        TENSOR_API_DETAIL_COORD_FAIL(
            "values=<unsupported shape/coord tuple combination>, shapeLeafCount=%u, coordLeafCount=%u",
            static_cast<unsigned int>(nesting_depth_v<ShapeType>),
            static_cast<unsigned int>(nesting_depth_v<CoordType>));
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_by_format(
    debug_slice_same_format_tag, ReportTag, const tensor_api_assert_context& context, const SourceType& source,
    const ShapeType& shape, const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...>,
    Std::index_sequence<coord_indices...>)
{
    using shape_format_type = debug_tuple_format_t<ShapeType>;
    if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_1>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld)", "(%lld, %lld)", "(%lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_5>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_6>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)",
            "(%lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "((%lld, %lld), (%lld, %lld))");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))", "(%lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))",
            "(%lld, (%lld, %lld), (%lld, %lld))");
    } else {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))",
            "(%lld, ((%lld, %lld), (%lld, %lld)))");
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_by_format(
    debug_slice_flat_source_tag, ReportTag, const tensor_api_assert_context& context, const SourceType& source,
    const ShapeType& shape, const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...>,
    Std::index_sequence<coord_indices...>)
{
    using coord_format_type = debug_tuple_format_t<CoordType>;
    if constexpr (Std::is_same_v<coord_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "((%lld, %lld), (%lld, %lld))");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, (%lld, %lld), (%lld, %lld))");
    } else {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld)", "(%lld, %lld, %lld)", "(%lld, ((%lld, %lld), (%lld, %lld)))");
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_by_format(
    debug_slice_nested_source_tag, ReportTag, const tensor_api_assert_context& context, const SourceType& source,
    const ShapeType& shape, const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...>,
    Std::index_sequence<coord_indices...>)
{
    using coord_format_type = debug_tuple_format_t<CoordType>;
    if constexpr (Std::is_same_v<coord_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))");
    } else {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))",
            "(%lld, ((%lld, %lld), (%lld, %lld)))");
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_by_format(
    debug_slice_rank_expanded_tag, ReportTag, const tensor_api_assert_context& context, const SourceType& source,
    const ShapeType& shape, const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...>,
    Std::index_sequence<coord_indices...>)
{
    using shape_format_type = debug_tuple_format_t<ShapeType>;
    if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)");
    } else {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)",
            "(%lld, %lld, %lld, %lld, %lld, %lld)");
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_by_format(
    debug_coord_pruned_format_tag, ReportTag, const tensor_api_assert_context& context, const SourceType& source,
    const ShapeType& shape, const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...>,
    Std::index_sequence<coord_indices...>)
{
    using shape_format_type = debug_tuple_format_t<ShapeType>;
    using coord_format_type = debug_tuple_format_t<CoordType>;
    if constexpr (Std::is_same_v<shape_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "((%lld, %lld), %lld)");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<coord_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))",
            "(%lld, (%lld, %lld))");
    } else if constexpr (is_debug_leading_scalar_first_nested_2_v<CoordType>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))",
            "(%lld, ((%lld, %lld), %lld))");
    } else {
        static_assert(is_debug_leading_scalar_second_nested_2_v<CoordType>, "Unsupported pruned coord format.");
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))",
            "(%lld, (%lld, (%lld, %lld)))");
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_by_format(
    debug_slice_unsupported_format_tag, ReportTag report_tag, const tensor_api_assert_context& context,
    const SourceType& source, const ShapeType& shape, const CoordType& coord, __gm__ const char* api_name,
    Std::index_sequence<shape_indices...>, Std::index_sequence<coord_indices...>)
{
    report_coord_unsupported(report_tag, context, source, shape, coord, api_name);
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_scalar_coord_error(
    ReportTag report_tag, const tensor_api_assert_context& context, const SourceType& source, const ShapeType& shape,
    const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...>,
    Std::index_sequence<coord_indices...>)
{
    using shape_type = Std::remove_cvref_t<ShapeType>;
    using shape_format_type = debug_tuple_format_t<ShapeType>;
    if constexpr (!Std::is_tuple_v<shape_type>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("%lld", "(%lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_1>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld)", "(%lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld, %lld)", "(%lld, %lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_5>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_flat_6>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS("(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))", "%lld");
    } else if constexpr (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS(
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))", "%lld");
    } else {
        report_coord_unsupported(report_tag, context, source, shape, coord, api_name);
    }
}

template <
    typename ReportTag, typename SourceType, typename ShapeType, typename CoordType, size_t... shape_indices,
    size_t... coord_indices>
__aicore__ inline void report_coord_error_impl(
    ReportTag report_tag, const tensor_api_assert_context& context, const SourceType& source, const ShapeType& shape,
    const CoordType& coord, __gm__ const char* api_name, Std::index_sequence<shape_indices...> shape_seq,
    Std::index_sequence<coord_indices...> coord_seq)
{
    if constexpr (!Std::is_tuple_v<Std::remove_cvref_t<CoordType>>) {
        report_scalar_coord_error(report_tag, context, source, shape, coord, api_name, shape_seq, coord_seq);
    } else if constexpr (!Std::is_tuple_v<Std::remove_cvref_t<ShapeType>>) {
        report_coord_unsupported(report_tag, context, source, shape, coord, api_name);
    } else {
        using shape_format_type = debug_tuple_format_t<ShapeType>;
        using coord_format_type = debug_tuple_format_t<CoordType>;
        constexpr bool is_pruned_format =
            (Std::is_same_v<shape_format_type, debug_tuple_nested_2x2> &&
             is_debug_trailing_scalar_flat_2_v<CoordType>) ||
            (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_flat_2> &&
             Std::is_same_v<coord_format_type, debug_tuple_flat_2>) ||
            (Std::is_same_v<shape_format_type, debug_tuple_leading_scalar_nested_2x2> &&
             (is_one_of_attr_v<coord_format_type, debug_tuple_flat_2, debug_tuple_leading_scalar_flat_2> ||
              is_debug_leading_scalar_first_nested_2_v<CoordType> ||
              is_debug_leading_scalar_second_nested_2_v<CoordType>));
        using category_type = debug_slice_format_category_t<shape_format_type, coord_format_type, coord_format_type>;
        if constexpr (is_pruned_format) {
            report_coord_error_by_format(
                debug_coord_pruned_format_tag{}, report_tag, context, source, shape, coord, api_name, shape_seq,
                coord_seq);
        } else {
            report_coord_error_by_format(
                category_type{}, report_tag, context, source, shape, coord, api_name, shape_seq, coord_seq);
        }
    }
}

template <typename LayoutType, typename CoordType>
__aicore__ inline void report_coord_error(
    const tensor_api_assert_context& context, const LayoutType& layout, const CoordType& coord,
    __gm__ const char* api_name)
{
    auto shape = debug_layout_view<LayoutType>::shape(layout);
    report_coord_error_impl(
        debug_coord_layout_report_tag{}, context, layout, shape, coord, api_name,
        Std::make_index_sequence<nesting_depth_v<decltype(shape)>>{},
        Std::make_index_sequence<nesting_depth_v<CoordType>>{});
}

template <typename ShapeType, typename CoordType>
__aicore__ inline void report_coord_shape_error(
    const tensor_api_assert_context& context, const ShapeType& shape, const CoordType& coord,
    __gm__ const char* api_name)
{
    report_coord_error_impl(
        debug_coord_shape_report_tag{}, context, shape, shape, coord, api_name,
        Std::make_index_sequence<nesting_depth_v<ShapeType>>{}, Std::make_index_sequence<nesting_depth_v<CoordType>>{});
}

#undef TENSOR_API_DETAIL_REPORT_COORD_WITH_FORMATS
#undef TENSOR_API_DETAIL_COORD_FAIL

#undef TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT
#undef TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS
#undef TENSOR_API_DETAIL_SLICE_FAIL

} // namespace te
} // namespace asc
#endif // ASCENDC_DEBUG

#endif // IMPL_TENSOR_API_UTILS_NPU_DEBUG_COORD_REPORT_H

#if defined(TENSOR_API_DEBUG_COORD_REPORT_OWNS_INTERNAL_HEADER_ACCESS)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef TENSOR_API_DEBUG_COORD_REPORT_OWNS_INTERNAL_HEADER_ACCESS
#endif
