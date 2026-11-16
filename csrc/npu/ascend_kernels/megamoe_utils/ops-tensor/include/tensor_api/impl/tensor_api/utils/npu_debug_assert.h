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
    "impl/tensor_api/utils/npu_debug_assert.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define TENSOR_API_DEBUG_ASSERT_OWNS_INTERNAL_HEADER_ACCESS
#endif

/*!
 * \file npu_debug_assert.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_UTILS_NPU_DEBUG_ASSERT_H
#define IMPL_TENSOR_API_UTILS_NPU_DEBUG_ASSERT_H

#ifdef ASCENDC_DEBUG
#include "impl/tensor_api/utils/npu_debug_utils.h"
#include "utils/debug/asc_printf.h"

namespace asc {
namespace te {

struct tensor_api_assert_context {
    __gm__ const char* file;
    uint32_t line;
    __gm__ const char* function;
};

// Keep the trap intrinsic outside constexpr Tensor API functions while preserving one assert record.
template <typename... ArgTypes>
__aicore__ inline void tensor_api_debug_assert_fail(__gm__ const char* full_format, ArgTypes&&... args)
{
    __asc_aicore::printf_impl_assert(full_format, args...);
    trap();
}

} // namespace te
} // namespace asc

#define TENSOR_API_DEBUG_CONTEXT \
    ::asc::te::tensor_api_assert_context { __FILE__, static_cast<uint32_t>(__LINE__), __FUNCTION__ }

#define TENSOR_API_DETAIL_VA_ARGS_IS_EMPTY(...) (sizeof(#__VA_ARGS__) == 1)

#define TENSOR_API_DETAIL_ASSERT_FAIL(context, format, ...)                                                         \
    do {                                                                                                            \
        ::asc::te::tensor_api_debug_assert_fail(                                                                    \
            "[ASSERT] %s:%u: %s: " format "\n", (context).file, (context).line, (context).function, ##__VA_ARGS__); \
    } while (0)

#define TENSOR_API_DETAIL_DEBUG_ASSERT_AT(context, expr, ...)                                          \
    do {                                                                                               \
        if (!(expr)) {                                                                                 \
            const ::asc::te::tensor_api_assert_context tensor_api_assert_context_instance = (context); \
            if (TENSOR_API_DETAIL_VA_ARGS_IS_EMPTY(__VA_ARGS__)) {                                     \
                TENSOR_API_DETAIL_ASSERT_FAIL(tensor_api_assert_context_instance, "");                 \
            } else {                                                                                   \
                __VA_ARGS__;                                                                           \
            }                                                                                          \
        }                                                                                              \
    } while (0)

#define TENSOR_API_DETAIL_DEBUG_ASSERT(expr, ...) \
    TENSOR_API_DETAIL_DEBUG_ASSERT_AT(TENSOR_API_DEBUG_CONTEXT, expr, ##__VA_ARGS__)

#define TENSOR_API_LOG_INTERNAL(format, ...) \
    TENSOR_API_DETAIL_ASSERT_FAIL(tensor_api_assert_context_instance, format, ##__VA_ARGS__)
#define TENSOR_API_REPORT_INTERNAL(reporter, ...) reporter(tensor_api_assert_context_instance, ##__VA_ARGS__)
#define TENSOR_API_DEBUG_ASSERT_AT(context, ...) TENSOR_API_DETAIL_DEBUG_ASSERT_AT(context, __VA_ARGS__)
#define TENSOR_API_DEBUG_ASSERT(...) TENSOR_API_DETAIL_DEBUG_ASSERT(__VA_ARGS__)
#define TENSOR_API_DEBUG_CHECK(checker, ...) checker(TENSOR_API_DEBUG_CONTEXT, ##__VA_ARGS__)

namespace asc {
namespace te {

#define TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS(value_format, batch_value_format)                                      \
    do {                                                                                                               \
        if constexpr (view_type::has_batch) {                                                                          \
            TENSOR_API_DETAIL_ASSERT_FAIL(                                                                             \
                context,                                                                                               \
                "Failed to check %s tensor %s in %s, current %sLayoutPtn=%s, %s%s=" batch_value_format                 \
                "; all %s tensor %s must be %s.",                                                                      \
                tensor_name, traits_type::value_name(), api_name, tensor_name, view_type::pattern_name(), tensor_name, \
                traits_type::field_name(), static_cast<long long>(traits_type::batch(layout)),                         \
                static_cast<long long>(get_debug_tuple_leaf<indices>(value))..., tensor_name,                          \
                traits_type::value_description(), traits_type::requirement());                                         \
        } else {                                                                                                       \
            TENSOR_API_DETAIL_ASSERT_FAIL(                                                                             \
                context,                                                                                               \
                "Failed to check %s tensor %s in %s, current %sLayoutPtn=%s, %s%s=" value_format                       \
                "; all %s tensor %s must be %s.",                                                                      \
                tensor_name, traits_type::value_name(), api_name, tensor_name, view_type::pattern_name(), tensor_name, \
                traits_type::field_name(), static_cast<long long>(get_debug_tuple_leaf<indices>(value))...,            \
                tensor_name, traits_type::value_description(), traits_type::requirement());                            \
        }                                                                                                              \
    } while (0)

template <tensor_layout_error_kind kind, typename LayoutType, typename ValueType, size_t... indices>
__aicore__ inline void report_tensor_layout_error_impl(
    const tensor_api_assert_context& context, const LayoutType& layout, const ValueType& value,
    __gm__ const char* tensor_name, __gm__ const char* api_name, Std::index_sequence<indices...>)
{
    using view_type = debug_layout_view<LayoutType>;
    using traits_type = tensor_layout_error_traits<kind>;
    using format_type = debug_tuple_format_t<ValueType>;
    if constexpr (Std::is_same_v<format_type, debug_tuple_flat_1>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS("(%lld)", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS("(%lld, %lld)", "(%lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS("(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS("(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_flat_5>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_flat_6>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS(
            "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS("((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS("(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_leading_scalar_nested_2x2>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))");
    } else if constexpr (Std::is_same_v<format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS(
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))");
    } else if constexpr (view_type::has_batch) {
        TENSOR_API_DETAIL_ASSERT_FAIL(
            context,
            "Failed to check %s tensor %s in %s, current %sLayoutPtn=%s, %s%s=(%lld, <unsupported tuple structure>), "
            "%s%sLeafCount=%u; all %s tensor %s must be %s.",
            tensor_name, traits_type::value_name(), api_name, tensor_name, view_type::pattern_name(), tensor_name,
            traits_type::field_name(), static_cast<long long>(traits_type::batch(layout)), tensor_name,
            traits_type::field_name(), static_cast<unsigned int>(nesting_depth_v<ValueType> + 1), tensor_name,
            traits_type::value_description(), traits_type::requirement());
    } else {
        TENSOR_API_DETAIL_ASSERT_FAIL(
            context,
            "Failed to check %s tensor %s in %s, current %sLayoutPtn=%s, %s%s=<unsupported tuple structure>, "
            "%s%sLeafCount=%u; all %s tensor %s must be %s.",
            tensor_name, traits_type::value_name(), api_name, tensor_name, view_type::pattern_name(), tensor_name,
            traits_type::field_name(), tensor_name, traits_type::field_name(),
            static_cast<unsigned int>(nesting_depth_v<ValueType>), tensor_name, traits_type::value_description(),
            traits_type::requirement());
    }
}

#undef TENSOR_API_DETAIL_REPORT_LAYOUT_FORMATS

template <tensor_layout_error_kind kind = tensor_layout_error_kind::shape, typename LayoutType>
__aicore__ inline void report_tensor_layout_error(
    const tensor_api_assert_context& context, const LayoutType& layout, __gm__ const char* tensor_name,
    __gm__ const char* api_name)
{
    using traits_type = tensor_layout_error_traits<kind>;
    auto value = traits_type::value(layout);
    report_tensor_layout_error_impl<kind>(
        context, layout, value, tensor_name, api_name, Std::make_index_sequence<nesting_depth_v<decltype(value)>>{});
}

#define TENSOR_API_DETAIL_COPY_SIZE_FAIL(fields_format, ...)                         \
    TENSOR_API_DETAIL_ASSERT_FAIL(                                                   \
        context,                                                                     \
        "Failed to check copy data size in %s, " fields_format                       \
        "; the amount of data to copy must not exceed the destination tensor size.", \
        api_name, __VA_ARGS__)

#define TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                         \
    src_shape_format, src_batch_shape_format, dst_shape_format, dst_batch_shape_format)                          \
    do {                                                                                                         \
        if constexpr (src_view_type::has_batch && dst_view_type::has_batch) {                                    \
            TENSOR_API_DETAIL_COPY_SIZE_FAIL(                                                                    \
                "srcLayoutPtn=%s, srcShape=" src_batch_shape_format                                              \
                ", copyDataSize=%lld, dstLayoutPtn=%s, dstShape=" dst_batch_shape_format ", dstTensorSize=%lld", \
                src_view_type::pattern_name(), static_cast<long long>(src_view_type::batch(src.layout())),       \
                static_cast<long long>(get_debug_tuple_leaf<src_indices>(src_shape))...,                         \
                static_cast<long long>(src.size()), dst_view_type::pattern_name(),                               \
                static_cast<long long>(dst_view_type::batch(dst.layout())),                                      \
                static_cast<long long>(get_debug_tuple_leaf<dst_indices>(dst_shape))...,                         \
                static_cast<long long>(dst.size()));                                                             \
        } else if constexpr (src_view_type::has_batch) {                                                         \
            TENSOR_API_DETAIL_COPY_SIZE_FAIL(                                                                    \
                "srcLayoutPtn=%s, srcShape=" src_batch_shape_format                                              \
                ", copyDataSize=%lld, dstLayoutPtn=%s, dstShape=" dst_shape_format ", dstTensorSize=%lld",       \
                src_view_type::pattern_name(), static_cast<long long>(src_view_type::batch(src.layout())),       \
                static_cast<long long>(get_debug_tuple_leaf<src_indices>(src_shape))...,                         \
                static_cast<long long>(src.size()), dst_view_type::pattern_name(),                               \
                static_cast<long long>(get_debug_tuple_leaf<dst_indices>(dst_shape))...,                         \
                static_cast<long long>(dst.size()));                                                             \
        } else if constexpr (dst_view_type::has_batch) {                                                         \
            TENSOR_API_DETAIL_COPY_SIZE_FAIL(                                                                    \
                "srcLayoutPtn=%s, srcShape=" src_shape_format                                                    \
                ", copyDataSize=%lld, dstLayoutPtn=%s, dstShape=" dst_batch_shape_format ", dstTensorSize=%lld", \
                src_view_type::pattern_name(),                                                                   \
                static_cast<long long>(get_debug_tuple_leaf<src_indices>(src_shape))...,                         \
                static_cast<long long>(src.size()), dst_view_type::pattern_name(),                               \
                static_cast<long long>(dst_view_type::batch(dst.layout())),                                      \
                static_cast<long long>(get_debug_tuple_leaf<dst_indices>(dst_shape))...,                         \
                static_cast<long long>(dst.size()));                                                             \
        } else {                                                                                                 \
            TENSOR_API_DETAIL_COPY_SIZE_FAIL(                                                                    \
                "srcLayoutPtn=%s, srcShape=" src_shape_format                                                    \
                ", copyDataSize=%lld, dstLayoutPtn=%s, dstShape=" dst_shape_format ", dstTensorSize=%lld",       \
                src_view_type::pattern_name(),                                                                   \
                static_cast<long long>(get_debug_tuple_leaf<src_indices>(src_shape))...,                         \
                static_cast<long long>(src.size()), dst_view_type::pattern_name(),                               \
                static_cast<long long>(get_debug_tuple_leaf<dst_indices>(dst_shape))...,                         \
                static_cast<long long>(dst.size()));                                                             \
        }                                                                                                        \
    } while (0)

#define TENSOR_API_DETAIL_DISPATCH_COPY_DST(src_shape_format, src_batch_shape_format)                            \
    do {                                                                                                         \
        if constexpr (Std::is_same_v<dst_format_type, debug_tuple_flat_1>) {                                     \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld)", "(%lld, %lld)");                             \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_flat_2>) {                              \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, %lld)", "(%lld, %lld, %lld)");                 \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_flat_3>) {                              \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)");     \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_flat_4>) {                              \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, %lld, %lld, %lld)",                            \
                "(%lld, %lld, %lld, %lld, %lld)");                                                               \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_flat_5>) {                              \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, %lld, %lld, %lld, %lld)",                      \
                "(%lld, %lld, %lld, %lld, %lld, %lld)");                                                         \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_flat_6>) {                              \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, %lld, %lld, %lld, %lld, %lld)",                \
                "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)");                                                   \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_nested_2x2>) {                          \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "((%lld, %lld), (%lld, %lld))",                        \
                "(%lld, (%lld, %lld), (%lld, %lld))");                                                           \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_leading_scalar_flat_2>) {               \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))"); \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_leading_scalar_nested_2x2>) {           \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, ((%lld, %lld), (%lld, %lld)))",                \
                "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))");                                                   \
        } else if constexpr (Std::is_same_v<dst_format_type, debug_tuple_leading_scalar_two_flat_2>) {           \
            TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS(                                                     \
                src_shape_format, src_batch_shape_format, "(%lld, (%lld, %lld), (%lld, %lld))",                  \
                "(%lld, %lld, (%lld, %lld), (%lld, %lld))");                                                     \
        } else {                                                                                                 \
            report_copy_size_unsupported(context, src, dst, api_name);                                           \
        }                                                                                                        \
    } while (0)

template <typename SrcTensorType, typename DstTensorType>
__aicore__ inline void report_copy_size_unsupported(
    const tensor_api_assert_context& context, const SrcTensorType& src, const DstTensorType& dst,
    __gm__ const char* api_name)
{
    using src_view_type = debug_layout_view<decltype(src.layout())>;
    using dst_view_type = debug_layout_view<decltype(dst.layout())>;
    if constexpr (src_view_type::has_batch && dst_view_type::has_batch) {
        TENSOR_API_DETAIL_COPY_SIZE_FAIL(
            "srcLayoutPtn=%s, srcShape=(%lld, <unsupported tuple structure>), copyDataSize=%lld, "
            "dstLayoutPtn=%s, dstShape=(%lld, <unsupported tuple structure>), dstTensorSize=%lld",
            src_view_type::pattern_name(), static_cast<long long>(src_view_type::batch(src.layout())),
            static_cast<long long>(src.size()), dst_view_type::pattern_name(),
            static_cast<long long>(dst_view_type::batch(dst.layout())), static_cast<long long>(dst.size()));
    } else if constexpr (src_view_type::has_batch) {
        TENSOR_API_DETAIL_COPY_SIZE_FAIL(
            "srcLayoutPtn=%s, srcShape=(%lld, <unsupported tuple structure>), copyDataSize=%lld, "
            "dstLayoutPtn=%s, dstShape=<unsupported tuple structure>, dstTensorSize=%lld",
            src_view_type::pattern_name(), static_cast<long long>(src_view_type::batch(src.layout())),
            static_cast<long long>(src.size()), dst_view_type::pattern_name(), static_cast<long long>(dst.size()));
    } else if constexpr (dst_view_type::has_batch) {
        TENSOR_API_DETAIL_COPY_SIZE_FAIL(
            "srcLayoutPtn=%s, srcShape=<unsupported tuple structure>, copyDataSize=%lld, "
            "dstLayoutPtn=%s, dstShape=(%lld, <unsupported tuple structure>), dstTensorSize=%lld",
            src_view_type::pattern_name(), static_cast<long long>(src.size()), dst_view_type::pattern_name(),
            static_cast<long long>(dst_view_type::batch(dst.layout())), static_cast<long long>(dst.size()));
    } else {
        TENSOR_API_DETAIL_COPY_SIZE_FAIL(
            "srcLayoutPtn=%s, srcShape=<unsupported tuple structure>, copyDataSize=%lld, "
            "dstLayoutPtn=%s, dstShape=<unsupported tuple structure>, dstTensorSize=%lld",
            src_view_type::pattern_name(), static_cast<long long>(src.size()), dst_view_type::pattern_name(),
            static_cast<long long>(dst.size()));
    }
}

template <
    typename SrcTensorType, typename DstTensorType, typename SrcShapeType, typename DstShapeType, size_t... src_indices,
    size_t... dst_indices>
__aicore__ inline void report_copy_size_error_impl(
    const tensor_api_assert_context& context, const SrcTensorType& src, const DstTensorType& dst,
    const SrcShapeType& src_shape, const DstShapeType& dst_shape, __gm__ const char* api_name,
    Std::index_sequence<src_indices...>, Std::index_sequence<dst_indices...>)
{
    using src_view_type = debug_layout_view<decltype(src.layout())>;
    using dst_view_type = debug_layout_view<decltype(dst.layout())>;
    using src_format_type = debug_tuple_format_t<SrcShapeType>;
    using dst_format_type = debug_tuple_format_t<DstShapeType>;
    if constexpr (Std::is_same_v<src_format_type, debug_tuple_flat_1>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("(%lld)", "(%lld, %lld)");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_flat_2>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("(%lld, %lld)", "(%lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_flat_3>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("(%lld, %lld, %lld)", "(%lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_flat_4>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("(%lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_flat_5>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("(%lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_flat_6>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST(
            "(%lld, %lld, %lld, %lld, %lld, %lld)", "(%lld, %lld, %lld, %lld, %lld, %lld, %lld)");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_nested_2x2>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("((%lld, %lld), (%lld, %lld))", "(%lld, (%lld, %lld), (%lld, %lld))");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_leading_scalar_flat_2>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST("(%lld, (%lld, %lld))", "(%lld, %lld, (%lld, %lld))");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_leading_scalar_nested_2x2>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST(
            "(%lld, ((%lld, %lld), (%lld, %lld)))", "(%lld, %lld, ((%lld, %lld), (%lld, %lld)))");
    } else if constexpr (Std::is_same_v<src_format_type, debug_tuple_leading_scalar_two_flat_2>) {
        TENSOR_API_DETAIL_DISPATCH_COPY_DST(
            "(%lld, (%lld, %lld), (%lld, %lld))", "(%lld, %lld, (%lld, %lld), (%lld, %lld))");
    } else {
        report_copy_size_unsupported(context, src, dst, api_name);
    }
}

#undef TENSOR_API_DETAIL_DISPATCH_COPY_DST
#undef TENSOR_API_DETAIL_REPORT_COPY_SIZE_WITH_FORMATS

template <typename SrcTensorType, typename DstTensorType>
__aicore__ inline void report_copy_size_error(
    const tensor_api_assert_context& context, const SrcTensorType& src, const DstTensorType& dst,
    __gm__ const char* api_name)
{
    using src_view_type = debug_layout_view<decltype(src.layout())>;
    using dst_view_type = debug_layout_view<decltype(dst.layout())>;
    auto src_shape = src_view_type::shape(src.layout());
    auto dst_shape = dst_view_type::shape(dst.layout());
    report_copy_size_error_impl(
        context, src, dst, src_shape, dst_shape, api_name,
        Std::make_index_sequence<nesting_depth_v<decltype(src_shape)>>{},
        Std::make_index_sequence<nesting_depth_v<decltype(dst_shape)>>{});
}

#undef TENSOR_API_DETAIL_COPY_SIZE_FAIL

#define TENSOR_API_DETAIL_SLICE_FAIL(fields_format, ...)                                                       \
    do {                                                                                                       \
        if constexpr (kind == slice_error_kind::source_shape) {                                                \
            TENSOR_API_DETAIL_ASSERT_FAIL(                                                                     \
                context,                                                                                       \
                "Failed to check source tensor shape in slice, " fields_format                                 \
                "; sourceShape must be greater than 0.",                                                       \
                __VA_ARGS__);                                                                                  \
        } else if constexpr (kind == slice_error_kind::slice_shape) {                                          \
            TENSOR_API_DETAIL_ASSERT_FAIL(                                                                     \
                context,                                                                                       \
                "Failed to check slice shape in slice, " fields_format "; sliceShape must be greater than 0.", \
                __VA_ARGS__);                                                                                  \
        } else {                                                                                               \
            static_assert(kind == slice_error_kind::coord, "Unsupported slice error kind.");                   \
            TENSOR_API_DETAIL_ASSERT_FAIL(                                                                     \
                context, "Failed to check coord in slice, " fields_format "; coord must be within shape.",     \
                __VA_ARGS__);                                                                                  \
        }                                                                                                      \
    } while (0)

#define TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(                                                            \
    source_shape_format, source_batch_shape_format, coord_format, slice_shape_format, slice_batch_shape_format) \
    do {                                                                                                        \
        if constexpr (view_type::has_batch) {                                                                   \
            if constexpr (slice_view_type::has_layout && slice_view_type::has_batch) {                          \
                TENSOR_API_DETAIL_SLICE_FAIL(                                                                   \
                    "sourceLayoutPtn=%s, sourceShape=" source_batch_shape_format ", coord=" coord_format        \
                    ", sliceLayoutPtn=%s, sliceShape=" slice_batch_shape_format,                                \
                    view_type::pattern_name(), static_cast<long long>(view_type::batch(layout)),                \
                    static_cast<long long>(get_debug_tuple_leaf<source_indices>(source_shape))...,              \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...,                      \
                    slice_view_type::pattern_name(), static_cast<long long>(slice_view_type::batch(info)),      \
                    static_cast<long long>(get_debug_tuple_leaf<slice_indices>(slice_shape))...);               \
            } else if constexpr (slice_view_type::has_layout) {                                                 \
                TENSOR_API_DETAIL_SLICE_FAIL(                                                                   \
                    "sourceLayoutPtn=%s, sourceShape=" source_batch_shape_format ", coord=" coord_format        \
                    ", sliceLayoutPtn=%s, sliceShape=" slice_shape_format,                                      \
                    view_type::pattern_name(), static_cast<long long>(view_type::batch(layout)),                \
                    static_cast<long long>(get_debug_tuple_leaf<source_indices>(source_shape))...,              \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...,                      \
                    slice_view_type::pattern_name(),                                                            \
                    static_cast<long long>(get_debug_tuple_leaf<slice_indices>(slice_shape))...);               \
            } else {                                                                                            \
                TENSOR_API_DETAIL_SLICE_FAIL(                                                                   \
                    "sourceLayoutPtn=%s, sourceShape=" source_batch_shape_format ", coord=" coord_format        \
                    ", sliceShape=" slice_shape_format,                                                         \
                    view_type::pattern_name(), static_cast<long long>(view_type::batch(layout)),                \
                    static_cast<long long>(get_debug_tuple_leaf<source_indices>(source_shape))...,              \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...,                      \
                    static_cast<long long>(get_debug_tuple_leaf<slice_indices>(slice_shape))...);               \
            }                                                                                                   \
        } else {                                                                                                \
            if constexpr (slice_view_type::has_layout && slice_view_type::has_batch) {                          \
                TENSOR_API_DETAIL_SLICE_FAIL(                                                                   \
                    "sourceLayoutPtn=%s, sourceShape=" source_shape_format ", coord=" coord_format              \
                    ", sliceLayoutPtn=%s, sliceShape=" slice_batch_shape_format,                                \
                    view_type::pattern_name(),                                                                  \
                    static_cast<long long>(get_debug_tuple_leaf<source_indices>(source_shape))...,              \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...,                      \
                    slice_view_type::pattern_name(), static_cast<long long>(slice_view_type::batch(info)),      \
                    static_cast<long long>(get_debug_tuple_leaf<slice_indices>(slice_shape))...);               \
            } else if constexpr (slice_view_type::has_layout) {                                                 \
                TENSOR_API_DETAIL_SLICE_FAIL(                                                                   \
                    "sourceLayoutPtn=%s, sourceShape=" source_shape_format ", coord=" coord_format              \
                    ", sliceLayoutPtn=%s, sliceShape=" slice_shape_format,                                      \
                    view_type::pattern_name(),                                                                  \
                    static_cast<long long>(get_debug_tuple_leaf<source_indices>(source_shape))...,              \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...,                      \
                    slice_view_type::pattern_name(),                                                            \
                    static_cast<long long>(get_debug_tuple_leaf<slice_indices>(slice_shape))...);               \
            } else {                                                                                            \
                TENSOR_API_DETAIL_SLICE_FAIL(                                                                   \
                    "sourceLayoutPtn=%s, sourceShape=" source_shape_format ", coord=" coord_format              \
                    ", sliceShape=" slice_shape_format,                                                         \
                    view_type::pattern_name(),                                                                  \
                    static_cast<long long>(get_debug_tuple_leaf<source_indices>(source_shape))...,              \
                    static_cast<long long>(get_debug_tuple_leaf<coord_indices>(coord))...,                      \
                    static_cast<long long>(get_debug_tuple_leaf<slice_indices>(slice_shape))...);               \
            }                                                                                                   \
        }                                                                                                       \
    } while (0)

#define TENSOR_API_DETAIL_REPORT_SLICE_SAME_FORMAT(tuple_format, batch_tuple_format) \
    TENSOR_API_DETAIL_REPORT_SLICE_WITH_FORMATS(                                     \
        tuple_format, batch_tuple_format, tuple_format, tuple_format, batch_tuple_format)

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType>
__aicore__ inline void report_slice_unsupported_source_batch(
    const tensor_api_assert_context& context, const LayoutType& layout, const CoordType&, const InfoType& info,
    const SourceShapeType&, const SliceShapeType&)
{
    using view_type = debug_layout_view<LayoutType>;
    using slice_view_type = debug_slice_info_view<InfoType>;
    if constexpr (slice_view_type::has_layout && slice_view_type::has_batch) {
        TENSOR_API_DETAIL_SLICE_FAIL(
            "sourceLayoutPtn=%s, sourceShape=(%lld, <unsupported tuple combination>), "
            "sourceShapeLeafCount=%u, coordLeafCount=%u, sliceLayoutPtn=%s, "
            "sliceShape=(%lld, <unsupported tuple combination>), sliceShapeLeafCount=%u",
            view_type::pattern_name(), static_cast<long long>(view_type::batch(layout)),
            static_cast<unsigned int>(nesting_depth_v<SourceShapeType> + 1),
            static_cast<unsigned int>(nesting_depth_v<CoordType>), slice_view_type::pattern_name(),
            static_cast<long long>(slice_view_type::batch(info)),
            static_cast<unsigned int>(nesting_depth_v<SliceShapeType> + 1));
    } else if constexpr (slice_view_type::has_layout) {
        TENSOR_API_DETAIL_SLICE_FAIL(
            "sourceLayoutPtn=%s, sourceShape=(%lld, <unsupported tuple combination>), "
            "sourceShapeLeafCount=%u, coordLeafCount=%u, sliceLayoutPtn=%s, sliceShapeLeafCount=%u",
            view_type::pattern_name(), static_cast<long long>(view_type::batch(layout)),
            static_cast<unsigned int>(nesting_depth_v<SourceShapeType> + 1),
            static_cast<unsigned int>(nesting_depth_v<CoordType>), slice_view_type::pattern_name(),
            static_cast<unsigned int>(nesting_depth_v<SliceShapeType>));
    } else {
        TENSOR_API_DETAIL_SLICE_FAIL(
            "sourceLayoutPtn=%s, sourceShape=(%lld, <unsupported tuple combination>), "
            "sourceShapeLeafCount=%u, coordLeafCount=%u, sliceShapeLeafCount=%u",
            view_type::pattern_name(), static_cast<long long>(view_type::batch(layout)),
            static_cast<unsigned int>(nesting_depth_v<SourceShapeType> + 1),
            static_cast<unsigned int>(nesting_depth_v<CoordType>),
            static_cast<unsigned int>(nesting_depth_v<SliceShapeType>));
    }
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType>
__aicore__ inline void report_slice_unsupported_no_source_batch(
    const tensor_api_assert_context& context, const LayoutType&, const CoordType&, const InfoType& info,
    const SourceShapeType&, const SliceShapeType&)
{
    using view_type = debug_layout_view<LayoutType>;
    using slice_view_type = debug_slice_info_view<InfoType>;
    if constexpr (slice_view_type::has_layout && slice_view_type::has_batch) {
        TENSOR_API_DETAIL_SLICE_FAIL(
            "sourceLayoutPtn=%s, sourceShape=<unsupported tuple combination>, sourceShapeLeafCount=%u, "
            "coordLeafCount=%u, sliceLayoutPtn=%s, sliceShape=(%lld, <unsupported tuple combination>), "
            "sliceShapeLeafCount=%u",
            view_type::pattern_name(), static_cast<unsigned int>(nesting_depth_v<SourceShapeType>),
            static_cast<unsigned int>(nesting_depth_v<CoordType>), slice_view_type::pattern_name(),
            static_cast<long long>(slice_view_type::batch(info)),
            static_cast<unsigned int>(nesting_depth_v<SliceShapeType> + 1));
    } else if constexpr (slice_view_type::has_layout) {
        TENSOR_API_DETAIL_SLICE_FAIL(
            "sourceLayoutPtn=%s, sourceShape=<unsupported tuple combination>, sourceShapeLeafCount=%u, "
            "coordLeafCount=%u, sliceLayoutPtn=%s, sliceShapeLeafCount=%u",
            view_type::pattern_name(), static_cast<unsigned int>(nesting_depth_v<SourceShapeType>),
            static_cast<unsigned int>(nesting_depth_v<CoordType>), slice_view_type::pattern_name(),
            static_cast<unsigned int>(nesting_depth_v<SliceShapeType>));
    } else {
        TENSOR_API_DETAIL_SLICE_FAIL(
            "sourceLayoutPtn=%s, sourceShape=<unsupported tuple combination>, sourceShapeLeafCount=%u, "
            "coordLeafCount=%u, sliceShapeLeafCount=%u",
            view_type::pattern_name(), static_cast<unsigned int>(nesting_depth_v<SourceShapeType>),
            static_cast<unsigned int>(nesting_depth_v<CoordType>),
            static_cast<unsigned int>(nesting_depth_v<SliceShapeType>));
    }
}

template <
    slice_error_kind kind, typename LayoutType, typename CoordType, typename InfoType, typename SourceShapeType,
    typename SliceShapeType>
__aicore__ inline void report_slice_unsupported(
    const tensor_api_assert_context& context, const LayoutType& layout, const CoordType& coord, const InfoType& info,
    const SourceShapeType& source_shape, const SliceShapeType& slice_shape)
{
    using view_type = debug_layout_view<LayoutType>;
    if constexpr (view_type::has_batch) {
        report_slice_unsupported_source_batch<kind>(context, layout, coord, info, source_shape, slice_shape);
    } else {
        report_slice_unsupported_no_source_batch<kind>(context, layout, coord, info, source_shape, slice_shape);
    }
}

} // namespace te
} // namespace asc
#else
#define TENSOR_API_LOG_INTERNAL(format, ...)
#define TENSOR_API_REPORT_INTERNAL(reporter, ...)
#define TENSOR_API_DEBUG_ASSERT_AT(context, ...)
#define TENSOR_API_DEBUG_ASSERT(...)
#define TENSOR_API_DEBUG_CHECK(checker, ...)
#endif

#endif // IMPL_TENSOR_API_UTILS_NPU_DEBUG_ASSERT_H

#if defined(TENSOR_API_DEBUG_ASSERT_OWNS_INTERNAL_HEADER_ACCESS)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef TENSOR_API_DEBUG_ASSERT_OWNS_INTERNAL_HEADER_ACCESS
#endif
