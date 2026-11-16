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
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif

/*!
 * \file tensor.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_TENSOR_H
#define INCLUDE_TENSOR_API_TENSOR_TENSOR_H

#include "tensor_api/tensor/layout_interface.h"
#include "tensor_api/tensor/pointer.h"
#include "impl/tensor_api/tensor/tuple_impl.h"
#include "tensor_api/utils/utils.h"

namespace asc {
namespace te {

template <typename EngineT, typename LayoutT>
struct base_tensor {
    using iterator = typename EngineT::iterator;
    using value_type = typename EngineT::value_type;
    using element_type = typename EngineT::element_type;
    using elementType = element_type;
    using reference = typename EngineT::reference;
    using engine_type = EngineT;
    using layout_type = LayoutT;

    static constexpr int rank = LayoutT::rank_size;

    __aicore__ inline constexpr base_tensor();
    __aicore__ inline constexpr base_tensor(const EngineT& engine, const LayoutT& layout);

    __aicore__ inline constexpr decltype(auto) tensor() const;
    __aicore__ inline constexpr decltype(auto) engine() const;
    __aicore__ inline constexpr decltype(auto) engine();
    __aicore__ inline constexpr decltype(auto) layout() const;
    __aicore__ inline constexpr decltype(auto) data() const;
    __aicore__ inline constexpr decltype(auto) data();
    __aicore__ inline constexpr decltype(auto) shape() const;
    __aicore__ inline constexpr decltype(auto) stride() const;
    __aicore__ inline constexpr auto size() const;
    __aicore__ inline constexpr auto capacity() const;

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord);

    template <typename Coord>
    __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) const;

    template <typename Coord>
    __aicore__ inline decltype(auto) operator()(const Coord& coord);

    template <typename Coord>
    __aicore__ inline decltype(auto) operator()(const Coord& coord) const;

    template <typename Coord0, typename Coord1, typename... Coords>
    __aicore__ inline decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs);

    template <typename Coord0, typename Coord1, typename... Coords>
    __aicore__ inline decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) const;

    template <typename Coord, typename Info>
    __aicore__ inline decltype(auto) slice(const Coord& coord, const Info& info);

    template <typename Coord, typename Info>
    __aicore__ inline decltype(auto) slice(const Coord& coord, const Info& info) const;

    __aicore__ inline constexpr decltype(auto) Tensor() const;
    __aicore__ inline constexpr decltype(auto) Engine() const;
    __aicore__ inline constexpr decltype(auto) Engine();
    __aicore__ inline constexpr decltype(auto) Layout() const;
    __aicore__ inline constexpr decltype(auto) Data() const;
    __aicore__ inline constexpr decltype(auto) Data();
    __aicore__ inline constexpr decltype(auto) Shape() const;
    __aicore__ inline constexpr decltype(auto) Stride() const;
    __aicore__ inline constexpr auto Size() const;
    __aicore__ inline constexpr auto Capacity() const;

    template <typename Coord, typename Info>
    __aicore__ inline decltype(auto) Slice(const Coord& coord, const Info& info);

    template <typename Coord, typename Info>
    __aicore__ inline decltype(auto) Slice(const Coord& coord, const Info& info) const;

private:
    template <typename SliceEngine, typename SliceLayout>
    __aicore__ inline static decltype(auto) make_sub_tensor(const SliceEngine& slice_engine,
                                                            const SliceLayout& slice_layout);

    Std::tuple<layout_type, engine_type> rep;
};

template <typename EngineT, typename LayoutT>
struct local_tensor : public base_tensor<EngineT, LayoutT> {
    using tensor_api_base = base_tensor<EngineT, LayoutT>;
    using element_type = typename tensor_api_base::element_type;
    using data_type = get_attribute_element_type<element_type*>;
    using layoutType = typename tensor_api_base::layout_type;
    using tensor_api_base::tensor_api_base;

    __aicore__ inline constexpr local_tensor();
};

template <typename EngineT, typename LayoutT>
struct global_tensor : public base_tensor<EngineT, LayoutT> {
    using tensor_api_base = base_tensor<EngineT, LayoutT>;
    using layoutType = typename tensor_api_base::layout_type;
    using tensor_api_base::tensor_api_base;

    __aicore__ inline constexpr global_tensor();
    __aicore__ inline constexpr cache_mode get_cache_mode() const;
    __aicore__ inline constexpr void set_l2_cache_hint(cache_mode mode);
    __aicore__ inline constexpr void SetL2CacheHint(AscendC::Te::CacheMode mode);
};

} // namespace te
} // namespace asc

#include "impl/tensor_api/tensor/tensor_impl.h"

// tensor construction
namespace asc {
namespace te {

template <typename Iterator, typename... Args>
__aicore__ inline auto make_tensor(const Iterator& iter, const Args&... args);

template <size_t... SqueezeDims, typename Input, typename Enable>
__aicore__ inline auto squeeze(const Input& x);

template <typename Pattern, typename Input, typename Enable>
__aicore__ inline auto squeeze(const Input& x, const Pattern& pattern);

} // namespace te
} // namespace asc

#endif // INCLUDE_TENSOR_API_TENSOR_TENSOR_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
