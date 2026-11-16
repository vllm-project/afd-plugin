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
    "impl/tensor_api/tensor/layout_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/layout_method.h"
#include "impl/tensor_api/tensor/coord_index.h"
#include "impl/tensor_api/tensor/layout_fractal.h"
#include "impl/tensor_api/tensor/layout_frame.h"
#include "impl/tensor_api/tensor/layout_infer.h"

namespace asc {
namespace te {

struct zero_coord_type {};

inline constexpr zero_coord_type zero_coord{};

struct min_op {
    template <typename SrcValue, typename DstValue>
    __aicore__ inline constexpr auto operator()(const SrcValue& src, const DstValue& dst) const
    {
        return Std::min(src, dst);
    }
};

struct diff_op {
    template <typename Shape, typename Coord>
    __aicore__ inline constexpr auto operator()(const Shape& shape, const Coord& coord) const
    {
        return shape - coord;
    }
};

template <typename Value>
__aicore__ inline constexpr auto make_zero_coord_like(const Value& value);

template <typename Value, size_t... Is>
__aicore__ inline constexpr auto make_zero_coord_impl(const Value& value, Std::index_sequence<Is...>)
{
    return make_coord(make_zero_coord_like(Std::get<Is>(value))...);
}

template <typename Value>
__aicore__ inline constexpr auto make_zero_coord_like(const Value& value)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<Value>>) {
        return make_zero_coord_impl(value, Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<Value>>>{});
    } else {
        return _0{};
    }
}

template <typename LayoutType, typename SliceShape>
__aicore__ inline constexpr decltype(auto) resolve_copy_coord(const LayoutType& layout, const SliceShape& slice_shape,
                                                              const zero_coord_type&)
{
    using origin_shape = Std::remove_cvref_t<decltype(layout.shape())>;
    using slice_shape_type = Std::remove_cvref_t<SliceShape>;
    constexpr auto origin_shape_depth = nesting_depth_v<origin_shape>;
    constexpr auto slice_shape_depth = nesting_depth_v<slice_shape_type>;
    constexpr bool is_same_shape = origin_shape_depth == slice_shape_depth
                                   && Std::tuple_size_v<origin_shape> == Std::tuple_size_v<slice_shape_type>;
    constexpr size_t batch_num = Std::tuple_size_v<origin_shape> - 1;
    constexpr bool is_flat_batch = Std::tuple_size_v<origin_shape> == Std::tuple_size_v<slice_shape_type>
                                   && Std::tuple_size_v<origin_shape> >= two_dim_data
                                   && origin_shape_depth == batch_num + four_dim_data
                                   && slice_shape_depth == batch_num + two_dim_data;

    if constexpr (is_same_shape) {
        return make_zero_coord_like(layout.shape());
    } else if constexpr (origin_shape_depth == four_dim_data && slice_shape_depth == two_dim_data) {
        return make_coord(_0{}, _0{});
    } else if constexpr (is_flat_batch) {
        return make_zero_coord_like(slice_shape);
    } else {
        return make_zero_coord_like(layout.shape());
    }
}

template <typename LayoutType, typename SliceShape, typename Coord,
          Std::enable_if_t<!Std::is_same_v<Std::remove_cvref_t<Coord>, zero_coord_type>, int> = 0>
__aicore__ inline constexpr const Coord& resolve_copy_coord(const LayoutType&, const SliceShape&, const Coord& coord)
{
    return coord;
}

template <typename Coord, typename LayoutType>
__aicore__ inline constexpr decltype(auto) make_coord_layout(const Coord& coord, const LayoutType& layout)
{
    using shape_type = Std::remove_cvref_t<decltype(layout.shape())>;
    using coord_type = Std::remove_cvref_t<Coord>;
    static_assert(is_layout_v<LayoutType> && Std::is_tuple_v<coord_type>, "LayoutType must be Layout");
    static_assert(nesting_depth_v<shape_type> == nesting_depth_v<coord_type>
                      && Std::tuple_size_v<shape_type> == Std::tuple_size_v<coord_type>,
                  "Shape and coord must have same tuple structure");
    auto coord_shape = transform_tuple_apply(layout.shape(), coord, diff_op{});
    using trait_type = get_layout_trait<LayoutType>;
    using pattern_type = get_layout_pattern<LayoutType>;
    return make_pattern_layout<pattern_type, trait_type>(coord_shape, layout.stride());
}

template <typename LayoutType, typename CopyShape>
__aicore__ inline constexpr decltype(auto) make_slice_pattern_layout(const LayoutType& layout, const CopyShape& shape)
{
    using trait_type = get_layout_trait<LayoutType>;
    using pattern_type = get_layout_pattern<LayoutType>;
    return make_pattern_layout<pattern_type, trait_type>(shape, layout.stride());
}

template <typename Coord, typename LayoutType, typename SliceShape>
__aicore__ inline constexpr decltype(auto) make_same_slice_shape(const Coord& coord, const LayoutType& layout,
                                                                 const SliceShape& slice_shape)
{
    auto coord_layout = make_coord_layout(coord, layout);
    return transform_tuple_apply(coord_layout.shape(), slice_shape, min_op{});
}

template <typename Coord, typename LayoutType, typename SliceShape>
__aicore__ inline constexpr decltype(auto) make_4d_slice_shape(const Coord& coord, const LayoutType& layout,
                                                               const SliceShape& slice_shape)
{
    static_assert(nesting_depth_v<SliceShape> == two_dim_data, "SliceShape must be Two Dim when layout is Four Dim");
    auto inner_row = get<0, 0>(layout.shape());
    auto inner_col = get<1, 0>(layout.shape());

    auto src_row = inner_row * get<0, 1>(layout.shape()) - get<0>(coord);
    auto src_col = inner_col * get<1, 1>(layout.shape()) - get<1>(coord);

    auto real_row = Std::min(src_row, get<0>(slice_shape));
    auto real_col = Std::min(src_col, get<1>(slice_shape));
    return make_fractal_shape(make_shape(real_row, real_col), make_shape(inner_row, inner_col));
}

// Slice for the flat multi-batch layouts produced by make_frame_layout(batch0, ..., batch_n, row, col):
//   layout shape (batch0, ..., batch_n-1, ((row0, row1), (col0, col1)))   -- batch_num + 1 elements
//   slice shape  (batch0, ..., batch_n-1, (x, y))                        -- logical row/col for the block
// Every batch axis is clamped elementwise like the same-shape path, and the trailing logical (x, y) is
// refractalized against the layout's inner row/col. The batch_num == 1 case covers the classic
// five-dimensional layout and three-dimensional slice shape.
template <typename Coord, typename LayoutType, typename SliceShape, size_t... batch_is>
__aicore__ inline constexpr decltype(auto) make_flat_slice_impl(const Coord& coord, const LayoutType& layout,
                                                                const SliceShape& slice_shape,
                                                                Std::index_sequence<batch_is...>)
{
    constexpr size_t fractal_block_index = sizeof...(batch_is); // last element: the fractal block / logical (x, y)
    auto inner_row = get<fractal_block_index, 0, 0>(layout.shape());
    auto inner_col = get<fractal_block_index, 1, 0>(layout.shape());

    auto real_row =
        Std::min(inner_row * get<fractal_block_index, 0, 1>(layout.shape()) - get<fractal_block_index, 0>(coord),
                 get<fractal_block_index, 0>(slice_shape));
    auto real_col =
        Std::min(inner_col * get<fractal_block_index, 1, 1>(layout.shape()) - get<fractal_block_index, 1>(coord),
                 get<fractal_block_index, 1>(slice_shape));
    auto fractal_shape = make_fractal_shape(make_shape(real_row, real_col), make_shape(inner_row, inner_col));

    return make_shape(Std::min(get<batch_is>(layout.shape()) - get<batch_is>(coord), get<batch_is>(slice_shape))...,
                      fractal_shape);
}

template <typename Coord, typename LayoutType, typename SliceShape>
__aicore__ inline constexpr decltype(auto) make_flat_slice_shape(const Coord& coord, const LayoutType& layout,
                                                                 const SliceShape& slice_shape)
{
    constexpr size_t batch_num = Std::tuple_size_v<Std::remove_cvref_t<decltype(layout.shape())>> - 1;
    return make_flat_slice_impl(coord, layout, slice_shape, Std::make_index_sequence<batch_num>{});
}

template <typename Coord, typename LayoutType, typename SliceShape, Std::enable_if_t<!is_layout_v<SliceShape>, int> = 0>
__aicore__ inline constexpr decltype(auto) make_slice_shape(const Coord& coord, const LayoutType& layout,
                                                            const SliceShape& slice_shape)
{
    static_assert(is_layout_v<LayoutType>, "LayoutType must be Layout");
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<SliceShape>>, "SliceShape must be a tuple");
    using origin_shape = Std::remove_cvref_t<decltype(layout.shape())>;
    using slice_shape_type = Std::remove_cvref_t<SliceShape>;
    constexpr auto origin_shape_depth = nesting_depth_v<origin_shape>;
    constexpr auto slice_shape_depth = nesting_depth_v<slice_shape_type>;
    constexpr bool is_same_shape = origin_shape_depth == slice_shape_depth
                                   && Std::tuple_size_v<origin_shape> == Std::tuple_size_v<slice_shape_type>;
    // Flat multi-batch: same arity on both sides, batch axes are scalars and the last element is the
    // fractal block on the layout side vs a logical (x, y) pair on the slice side. Depth is then
    // batch_num + 4 against batch_num + 2. batch_num == 1 covers the classic five-dim/three-dim case.
    constexpr size_t batch_num = Std::tuple_size_v<origin_shape> - 1;
    constexpr bool is_flat_batch = Std::tuple_size_v<origin_shape> == Std::tuple_size_v<slice_shape_type>
                                   && Std::tuple_size_v<origin_shape> >= two_dim_data
                                   && origin_shape_depth == batch_num + four_dim_data
                                   && slice_shape_depth == batch_num + two_dim_data;

    if constexpr (is_same_shape) {
        return make_same_slice_shape(coord, layout, slice_shape);
    } else if constexpr (origin_shape_depth == four_dim_data && slice_shape_depth == two_dim_data) {
        return make_4d_slice_shape(coord, layout, slice_shape);
    } else if constexpr (is_flat_batch) {
        return make_flat_slice_shape(coord, layout, slice_shape);
    } else {
        static_assert(
            is_same_shape || (origin_shape_depth == four_dim_data && slice_shape_depth == two_dim_data)
                || is_flat_batch,
            "SliceShape must be same structure as Layout shape, or logical Two Dim Shape for Four Dim Layout, "
            "or (batch0, ..., batch_n, logical Two Dim Shape) for a flat multi-batch Layout.");
    }
}

template <typename Coord, typename LayoutType, typename SliceShape, Std::enable_if_t<!is_layout_v<SliceShape>, int> = 0>
__aicore__ inline constexpr decltype(auto) make_slice_layout(const Coord& coord, const LayoutType& layout,
                                                             const SliceShape& slice_shape)
{
    return make_slice_pattern_layout(layout, make_slice_shape(coord, layout, slice_shape));
}

template <typename Coord, typename SrcLayoutType, typename DstLayoutType,
          Std::enable_if_t<is_layout_v<DstLayoutType>, int> = 0>
__aicore__ inline constexpr decltype(auto) make_slice_layout(const Coord& coord, const SrcLayoutType& src_layout,
                                                             const DstLayoutType& dst_layout)
{
    static_assert(is_layout_v<SrcLayoutType>, "SrcLayoutType must be Layout");
    static_assert(SrcLayoutType::rank_size == DstLayoutType::rank_size,
                  "SrcLayout Rank must be equal to DstLayout Rank");

    auto slice_layout = make_coord_layout(coord, src_layout);
    auto slice_shape = transform_tuple_apply(slice_layout.shape(), dst_layout.shape(), min_op{});
    using trait_type = get_layout_trait<SrcLayoutType>;
    using pattern_type = get_layout_pattern<SrcLayoutType>;
    return make_pattern_layout<pattern_type, trait_type>(slice_shape, src_layout.stride());
}

template <typename LayoutType, typename = Std::enable_if_t<is_layout_v<LayoutType>>>
__aicore__ inline constexpr decltype(auto) remove_batch_dim(const LayoutType& layout)
{
    constexpr auto layout_depth = LayoutType::depth;
    static_assert(layout_depth == three_dim_data || layout_depth == five_dim_data,
                  "RemoveBatchDim only supports Three Dim or Five Dim Layout.");
    using trait_type = get_layout_trait<LayoutType>;
    using pattern_type = get_layout_pattern<LayoutType>;
    return make_pattern_layout<pattern_type, trait_type>(get<1>(layout.shape()), get<1>(layout.stride()));
}

// ===== Squeeze =====

// IsInPack: true if I is in the pack.
template <size_t I, size_t... Dims>
struct is_in_pack : Std::false_type {};
template <size_t I, size_t Head, size_t... Tail>
struct is_in_pack<I, Head, Tail...> : Std::conditional_t<Head == I, Std::true_type, is_in_pack<I, Tail...>> {};
template <size_t I, size_t... Dims>
constexpr bool is_in_pack_v = is_in_pack<I, Dims...>::value;

// IsSqueezeMark: pattern position marked _1.
template <typename PatternValue>
struct is_squeeze_mark : Std::is_same<Std::remove_cvref_t<PatternValue>, Std::Int<1>> {};
template <typename PatternValue>
constexpr bool is_squeeze_mark_v = is_squeeze_mark<Std::remove_cvref_t<PatternValue>>::value;

// SqueezableSize: true if axis size is Int<1> or a runtime int (trust caller).
template <typename Size, bool IsConst = is_integral_constant_v<Std::remove_cvref_t<Size>>>
struct squeezable_size : Std::true_type {};
template <typename Size>
struct squeezable_size<Size, true> : Std::bool_constant<(Std::remove_cvref_t<Size>::value == 1)> {};
template <typename Size>
constexpr bool squeezable_size_v = squeezable_size<Std::remove_cvref_t<Size>>::value;

// dims_mask: keep mask. Drop axis I iff I in SqueezeDims && SqueezableSize.
template <typename Shape, typename DimsSeq, typename IdxSeq>
struct dims_mask_impl;
template <typename Shape, size_t... Dims, size_t... Is>
struct dims_mask_impl<Shape, Std::index_sequence<Dims...>, Std::index_sequence<Is...>> {
    using type = Std::IntegerSequence<bool, (!(is_in_pack_v<Is, Dims...>
                                               && squeezable_size_v<typename Std::tuple_element<Is, Shape>::type>))...>;
};
template <typename Shape, size_t... Dims>
using dims_mask = typename dims_mask_impl<Shape, Std::index_sequence<Dims...>,
                                          Std::make_index_sequence<Std::tuple_size_v<Shape>>>::type;

// pattern_mask: keep mask. Drop axis I iff pattern[I] is _1 && SqueezableSize.
template <typename Shape, typename Pattern, typename IdxSeq>
struct pattern_mask_impl;
template <typename Shape, typename Pattern, size_t... Is>
struct pattern_mask_impl<Shape, Pattern, Std::index_sequence<Is...>> {
    using type = Std::IntegerSequence<bool, (!(is_squeeze_mark_v<typename Std::tuple_element<Is, Pattern>::type>
                                               && squeezable_size_v<typename Std::tuple_element<Is, Shape>::type>))...>;
};
template <typename Shape, typename Pattern>
using pattern_mask =
    typename pattern_mask_impl<Shape, Pattern, Std::make_index_sequence<Std::tuple_size_v<Shape>>>::type;

// Promote single nested tuple after squeeze; preserve Pattern/Trait.
template <typename Pattern, typename Trait, typename ShapeTuple, typename StrideTuple>
__aicore__ inline constexpr auto make_squeezed_layout(const ShapeTuple& shape, const StrideTuple& stride)
{
    if constexpr (Std::tuple_size_v<ShapeTuple> == 1
                  && Std::is_tuple_v<Std::remove_cvref_t<typename Std::tuple_element<0, ShapeTuple>::type>>) {
        return make_pattern_layout<Pattern, Trait>(get<0>(shape), get<0>(stride));
    } else {
        return make_pattern_layout<Pattern, Trait>(shape, stride);
    }
}

// Squeeze mode 1: drop selected axes by index (if squeezable). Preserves Pattern/Trait.
template <size_t... SqueezeDims, typename LayoutType,
          typename = Std::enable_if_t<is_layout_v<LayoutType> && (sizeof...(SqueezeDims) > 0)>>
__aicore__ inline constexpr auto squeeze_layout(const LayoutType& layout)
{
    using shape = Std::remove_cvref_t<decltype(layout.shape())>;
    using mask = dims_mask<shape, SqueezeDims...>;
    using keep_seq = typename keep_index_seq_from_seq<mask>::type;
    using pattern = get_layout_pattern<LayoutType>;
    using trait = get_layout_trait<LayoutType>;
    return make_squeezed_layout<pattern, trait>(select_by_seq<keep_seq>(layout.shape()),
                                                select_by_seq<keep_seq>(layout.stride()));
}

// ---- Recursive pattern squeeze (mode 2) ----

template <typename Shape, typename Stride, typename Pattern>
__aicore__ inline constexpr auto squeeze_rec_tuple_level(const Shape& shape, const Stride& stride,
                                                         const Pattern& pattern);

// Recurse into tuple pattern nodes; keep leaves as-is.
template <typename SNode, typename DNode, typename PNode>
__aicore__ inline constexpr auto squeeze_rec_node(const SNode& s, const DNode& d, const PNode& p)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<PNode>>) {
        return squeeze_rec_tuple_level(s, d, p);
    } else {
        return Std::make_tuple(s, d);
    }
}

// Unwrap single-element tuple pair one layer.
template <typename Shape, typename Stride>
__aicore__ inline constexpr auto squeeze_unwrap(const Shape& shape, const Stride& stride)
{
    if constexpr (Std::tuple_size_v<Std::remove_cvref_t<Shape>> == 1) {
        return Std::make_tuple(get<0>(shape), get<0>(stride));
    } else {
        return Std::make_tuple(shape, stride);
    }
}

template <typename Shape, typename Stride, typename Pattern, size_t... Is>
__aicore__ inline constexpr auto squeeze_rec_tuple_level_impl(const Shape& shape, const Stride& stride,
                                                              const Pattern& pattern, Std::index_sequence<Is...>)
{
    auto t_shape = Std::make_tuple(get<0>(squeeze_rec_node(get<Is>(shape), get<Is>(stride), get<Is>(pattern)))...);
    auto t_stride = Std::make_tuple(get<1>(squeeze_rec_node(get<Is>(shape), get<Is>(stride), get<Is>(pattern)))...);
    using mask = pattern_mask<Std::remove_cvref_t<Shape>, Std::remove_cvref_t<Pattern>>;
    using keep_seq = typename keep_index_seq_from_seq<mask>::type;
    return squeeze_unwrap(select_by_seq<keep_seq>(t_shape), select_by_seq<keep_seq>(t_stride));
}

template <typename Shape, typename Stride, typename Pattern>
__aicore__ inline constexpr auto squeeze_rec_tuple_level(const Shape& shape, const Stride& stride,
                                                         const Pattern& pattern)
{
    return squeeze_rec_tuple_level_impl(shape, stride, pattern,
                                        Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<Shape>>>{});
}

// Squeeze mode 2: recursively drop _1-marked positions (if squeezable). Preserves Pattern/Trait.
template <typename Pattern, typename LayoutType,
          typename = Std::enable_if_t<is_layout_v<LayoutType> && Std::is_tuple_v<Std::remove_cvref_t<Pattern>>>>
__aicore__ inline constexpr auto squeeze_layout(const LayoutType& layout, const Pattern& pattern)
{
    using shape = Std::remove_cvref_t<decltype(layout.shape())>;
    static_assert(is_same_structure_v<Pattern, shape>, "Squeeze pattern must be isomorphic to the layout shape.");
    using layout_pattern_type = get_layout_pattern<LayoutType>;
    using layout_trait_type = get_layout_trait<LayoutType>;
    auto pr = squeeze_rec_tuple_level(layout.shape(), layout.stride(), pattern);
    auto out_shape = get<0>(pr);
    auto out_stride = get<1>(pr);
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(out_shape)>>) {
        return make_pattern_layout<layout_pattern_type, layout_trait_type>(out_shape, out_stride);
    } else {
        return make_pattern_layout<layout_pattern_type, layout_trait_type>(Std::make_tuple(out_shape),
                                                                           Std::make_tuple(out_stride));
    }
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
