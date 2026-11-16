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
    "impl/tensor_api/tensor/layout_frame.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_frame.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H

#include "impl/tensor_api/tensor/layout_pattern.h"

namespace asc {
namespace te {

using layout_format_set = tuple_map<
    Std::tuple<nz_layout_ptn, make_nz_frame_layout>, Std::tuple<nd_layout_ptn, make_nd_frame_layout>,
    Std::tuple<dn_layout_ptn, make_dn_frame_layout>, Std::tuple<nd_ext_layout_ptn, make_nd_ext_frame_layout>,
    Std::tuple<dn_ext_layout_ptn, make_dn_ext_frame_layout>, Std::tuple<nn_layout_ptn, make_nn_frame_layout>,
    Std::tuple<zz_layout_ptn, make_zz_frame_layout>, Std::tuple<zn_layout_ptn, make_zn_frame_layout>,
    Std::tuple<scalea_nd_layout_ptn, make_scalea_nd_frame_layout>,
    Std::tuple<scalea_dn_layout_ptn, make_scalea_dn_frame_layout>,
    Std::tuple<scaleb_nd_layout_ptn, make_scaleb_nd_frame_layout>,
    Std::tuple<scaleb_dn_layout_ptn, make_scaleb_dn_frame_layout>, Std::tuple<nchw_layout_ptn, make_nchw_frame_layout>,
    Std::tuple<nhwc_layout_ptn, make_nhwc_frame_layout>, Std::tuple<nc1hwc0_layout_ptn, make_nc1hwc0_frame_layout>,
    Std::tuple<ncdhw_layout_ptn, make_ncdhw_frame_layout>, Std::tuple<ndc1hwc0_layout_ptn, make_ndc1hwc0_frame_layout>>;

using format_trait_set =
    tuple_map<Std::tuple<nz_layout_ptn, layout_trait_default<>>,
              Std::tuple<nd_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<dn_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<nd_ext_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<dn_ext_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<nn_layout_ptn, layout_trait<Std::ignore_t, _2>>,
              Std::tuple<zz_layout_ptn, layout_trait_default<>>, Std::tuple<zn_layout_ptn, layout_trait_default<>>,
              Std::tuple<scalea_nd_layout_ptn, layout_trait<Std::ignore_t, _2>>,
              Std::tuple<scalea_dn_layout_ptn, layout_trait<Std::ignore_t, _2>>,
              Std::tuple<scaleb_nd_layout_ptn, layout_trait<Std::ignore_t, _2>>,
              Std::tuple<scaleb_dn_layout_ptn, layout_trait<Std::ignore_t, _2>>,
              Std::tuple<nchw_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<nhwc_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<nc1hwc0_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<ncdhw_layout_ptn, layout_trait<Std::ignore_t, _1>>,
              Std::tuple<ndc1hwc0_layout_ptn, layout_trait<Std::ignore_t, _1>>>;

template <typename Trait, typename = void>
struct is_frame_layout_trait : Std::false_type {};

template <typename Trait>
struct is_frame_layout_trait<Trait, void_t<decltype(Trait::c0_element)>> : Std::true_type {};

template <typename Trait>
constexpr bool is_frame_layout_trait_v = is_frame_layout_trait<Trait>::value;

template <typename LayoutPattern, typename TraitType>
struct trait_conversion {
private:
    using raw_trait = Std::remove_cvref_t<TraitType>;

    static_assert(Std::is_same_v<raw_trait, Std::ignore_t> || is_integral_constant_v<raw_trait>
                      || is_frame_layout_trait_v<raw_trait> || is_data_type<raw_trait>,
                  "TraitType must be ignore_t, integral constant, frame layout trait, or data type.");

    using from_pattern = typename format_trait_set::template get<LayoutPattern>;
    using from_int = layout_trait<Std::ignore_t, raw_trait>;
    using from_data_type = layout_trait_default<raw_trait>;
    using from_trait = raw_trait;

    using converted = typename Std::conditional<
        Std::is_same_v<raw_trait, Std::ignore_t>, from_pattern,
        typename Std::conditional<is_integral_constant_v<raw_trait>, from_int,
                                  typename Std::conditional<is_frame_layout_trait_v<raw_trait>, from_trait,
                                                            from_data_type>::type>::type>::type;

public:
    using type = converted;
};

// ===== Multi-batch frame layout: make_frame_layout<Ptn>(batch0, ..., batch_n, row, col) =====

// supports_row_col_make: true when the pattern's Make accepts a bare (row, column) pair, i.e. it is one
// of the matrix layouts that also offer the (batch, row, column) form. The conv feature-map patterns
// (NCHW/NHWC/NC1HWC0/NCDHW/NDC1HWC0) take fixed positional dimensions and have no 2-arg Make, so they
// never take the multi-batch path below.
template <typename Maker, typename Trait, typename = void>
struct supports_row_col_make : Std::false_type {};

template <typename Maker, typename Trait>
struct supports_row_col_make<Maker, Trait,
                             void_t<decltype(Maker::template make<Trait>(Std::declval<int>(), Std::declval<int>()))>>
    : Std::true_type {};

template <typename Maker, typename Trait>
constexpr bool supports_row_col_make_v = supports_row_col_make<Maker, Trait>::value;

// BatchStrideAt: stride of batch axis I, given the batch extents tuple and the base layout capacity.
// Batch axes are flat and row-major over the base block, so the stride of an axis is the product of
// all batch extents to its right times the base capacity:
//   stride[n-1] = capacity, stride[n-2] = batch[n-1]*capacity, ...
template <size_t I, size_t BatchNum, typename BatchTuple, typename Capacity>
__aicore__ inline constexpr auto batch_stride_at(const BatchTuple& batches, const Capacity& capacity)
{
    if constexpr (I + 1 >= BatchNum) {
        return capacity;
    } else {
        return get<I + 1>(batches) * batch_stride_at<I + 1, BatchNum, BatchTuple, Capacity>(batches, capacity);
    }
}

// Builds the flat multi-batch layout: all batch axes sit side by side in the outermost tuple and the
// base (row, col) block is appended as the last element, e.g.
//   (batch0, batch1, ((row0, row1), (col0, col1)))
// so rank is batch_num + 1 (not a chain of nested single-batch layouts).
template <typename LayoutPattern, typename Trait, typename BatchTuple, typename BaseLayout, size_t... batch_is>
__aicore__ inline constexpr auto make_flat_batch_layout(const BatchTuple& batches, const BaseLayout& base,
                                                        Std::index_sequence<batch_is...>)
{
    constexpr size_t batch_num = sizeof...(batch_is);
    auto capacity = base.capacity();
    return make_pattern_layout<LayoutPattern, Trait>(
        make_shape(get<batch_is>(batches)..., base.shape()),
        make_stride(batch_stride_at<batch_is, batch_num, BatchTuple, decltype(capacity)>(batches, capacity)...,
                    base.stride()));
}

// Peels the trailing (row, column) off the argument pack, builds the 2D base layout from them, then
// lays the leading arguments out as flat batch axes.
template <typename LayoutPattern, typename Trait, typename Maker, typename ArgsTuple, size_t... batch_is>
__aicore__ inline auto make_multi_batch_frame_layout_impl(const ArgsTuple& args, Std::index_sequence<batch_is...>)
{
    constexpr size_t arg_num = Std::tuple_size_v<ArgsTuple>;
    auto base = Maker::template make<Trait>(get<arg_num - 2>(args), get<arg_num - 1>(args));
    auto batches = Std::make_tuple(get<batch_is>(args)...);
    return make_flat_batch_layout<LayoutPattern, Trait>(batches, base, Std::index_sequence<batch_is...>{});
}

template <typename LayoutPattern, typename Trait, typename Maker, typename... Args>
__aicore__ inline auto make_multi_batch_frame_layout(const Args&... args)
{
    return make_multi_batch_frame_layout_impl<LayoutPattern, Trait, Maker>(
        Std::make_tuple(args...), Std::make_index_sequence<sizeof...(Args) - 2>{});
}

template <typename LayoutPattern, typename Trait, typename... Args>
__aicore__ inline decltype(auto) make_frame_layout(const Args&... args)
{
    using trait = typename trait_conversion<LayoutPattern, Trait>::type;
    using layout_maker = typename layout_format_set::template get<LayoutPattern>;
    static_assert(!Std::is_same_v<layout_maker, Std::ignore_t>, "Unsupported layout pattern.");
    // 2 args = (row, col) and 3 args = (batch, row, col) are handled by the pattern's own Make. Four
    // or more args on a row/col-style pattern means multiple batch axes: (batch0, ..., batch_n, row, col).
    if constexpr (sizeof...(Args) > 3 && supports_row_col_make_v<layout_maker, trait>) {
        return make_multi_batch_frame_layout<LayoutPattern, trait, layout_maker>(args...);
    } else {
        return layout_maker::template make<trait>(args...);
    }
}

template <typename LayoutPattern, size_t C0Element, typename... Args>
__aicore__ inline decltype(auto) make_frame_layout(const Args&... args)
{
    return make_frame_layout<LayoutPattern, Std::Int<C0Element>>(args...);
}

template <typename LayoutPattern, typename Trait>
template <typename... Args>
__aicore__ inline decltype(auto) frame_layout_format<LayoutPattern, Trait>::operator()(const Args&... args)
{
    return make_frame_layout<LayoutPattern, Trait>(args...);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
