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
    "impl/tensor_api/tensor/layout_pattern.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file layout_pattern.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_PATTERN_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_PATTERN_H

#include "impl/tensor_api/tensor/layout_method.h"

namespace asc {
namespace te {

template <typename LayoutPattern, typename Trait, typename Shape, typename Stride>
__aicore__ inline constexpr auto make_pattern_layout(const Shape& shape, const Stride& stride)
{
    using layout_t = layout<Shape, Stride, Std::tuple<LayoutPattern, Trait>>;
    return layout_t(shape, stride);
}

template <typename LayoutPattern, typename TraitType, typename Batch, typename LayoutType>
__aicore__ inline constexpr auto make_batch_pattern_layout(const Batch& batch, const LayoutType& layout)
{
    return make_pattern_layout<LayoutPattern, TraitType>(make_shape(batch, layout.shape()),
                                                         make_stride(layout.capacity(), layout.stride()));
}

struct make_nz_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        auto shape = make_shape(make_shape(Std::Int<fractal_fixed>{}, Std::ceil_division(row, fractal_fixed)),
                                make_shape(c0_ele, Std::ceil_division(column, c0_ele)));
        auto stride = make_stride(make_stride(c0_ele, c0_ele * Std::Int<fractal_fixed>{}),
                                  make_stride(_1{}, c0_ele * Std::ceil_align(row, fractal_fixed)));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nz_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<nz_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_nd_ext_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        auto shape = make_shape(make_shape(_1{}, row), make_shape(_1{}, column));
        auto stride = make_stride(make_stride(_0{}, column), make_stride(_0{}, _1{}));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nd_ext_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<nd_ext_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_nd_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        auto shape = make_shape(row, column);
        auto stride = make_stride(column, _1{});
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nd_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<nd_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_zn_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        auto shape = make_shape(make_shape(c0_ele, Std::ceil_division(row, c0_ele)),
                                make_shape(Std::Int<fractal_fixed>{}, Std::ceil_division(column, fractal_fixed)));
        auto stride = make_stride(make_stride(_1{}, c0_ele * Std::ceil_align(column, fractal_fixed)),
                                  make_stride(c0_ele, c0_ele * Std::Int<fractal_fixed>{}));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<zn_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<zn_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_dn_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        auto shape = make_shape(row, column);
        auto stride = make_stride(_1{}, row);
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<dn_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<dn_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_dn_ext_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        auto shape = make_shape(make_shape(_1{}, row), make_shape(_1{}, column));
        auto stride = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, row));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<dn_ext_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<dn_ext_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_zz_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        auto shape = make_shape(make_shape(Std::Int<fractal_fixed>{}, Std::ceil_division(row, fractal_fixed)),
                                make_shape(c0_ele, Std::ceil_division(column, c0_ele)));
        auto stride = make_stride(make_stride(c0_ele, fractal_fixed * Std::ceil_align(column, c0_ele)),
                                  make_stride(_1{}, c0_ele * Std::Int<fractal_fixed>{}));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<zz_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<zz_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_nn_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        static_assert(c0_ele == 2, "NnLayoutPtn only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = make_shape(make_shape(c0_ele, row / c0_ele),
                                make_shape(Std::Int<fractal_fixed>{}, Std::ceil_division(column, fractal_fixed)));
        auto stride = make_stride(make_stride(_1{}, c0_ele * Std::Int<fractal_fixed>{}),
                                  make_stride(c0_ele, row * fractal_fixed));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nn_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<nn_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_scalea_nd_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        static_assert(c0_ele == 2, "scalea_nd_layout_ptn only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = make_shape(make_shape(_1{}, row), make_shape(_1{}, column));
        auto stride = make_stride(make_stride(_0{}, column), make_stride(_0{}, _1{}));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<scalea_nd_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<scalea_nd_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_scalea_dn_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        static_assert(c0_ele == 2, "scalea_dn_layout_ptn only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = make_shape(make_shape(_1{}, row), make_shape(c0_ele, column / c0_ele));
        auto stride = make_stride(make_stride(_0{}, c0_ele), make_stride(_1{}, c0_ele * row));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<scalea_dn_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<scalea_dn_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_scaleb_nd_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        constexpr auto c0_ele = TraitType::c0_element;
        static_assert(c0_ele == 2, "scaleb_nd_layout_ptn only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = make_shape(make_shape(c0_ele, row / c0_ele), make_shape(_1{}, column));
        auto stride = make_stride(make_stride(_1{}, c0_ele * column), make_stride(_0{}, c0_ele));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<scaleb_nd_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<scaleb_nd_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_scaleb_dn_frame_layout {
    template <typename TraitType, typename Row, typename Column>
    __aicore__ inline static auto make(Row row, Column column)
    {
        auto shape = make_shape(make_shape(_1{}, row), make_shape(_1{}, column));
        auto stride = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, row));
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<scaleb_dn_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }

    template <typename TraitType, typename Batch, typename Row, typename Column>
    __aicore__ inline static auto make(Batch batch, Row row, Column column)
    {
        return make_batch_pattern_layout<scaleb_dn_layout_ptn, TraitType>(batch, make<TraitType>(row, column));
    }
};

struct make_nchw_frame_layout {
    template <typename TraitType, typename N, typename C, typename H, typename W>
    __aicore__ inline static auto make(N n, C c, H h, W w)
    {
        auto shape = make_shape(n, c, h, w);
        auto stride = make_stride(c * h * w, h * w, w, _1{});
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nchw_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }
};

struct make_nhwc_frame_layout {
    template <typename TraitType, typename N, typename H, typename W, typename C>
    __aicore__ inline static auto make(N n, H h, W w, C c)
    {
        auto shape = make_shape(n, h, w, c);
        auto stride = make_stride(h * w * c, w * c, c, _1{});
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nhwc_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }
};

struct make_nc1hwc0_frame_layout {
    template <typename TraitType, typename N, typename C1, typename H, typename W, typename C0>
    __aicore__ inline static auto make(N n, C1 c1, H h, W w, C0 c0)
    {
        auto shape = make_shape(n, c1, h, w, c0);
        auto stride = make_stride(c1 * h * w * c0, h * w * c0, w * c0, c0, _1{});
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<nc1hwc0_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }
};

// conv3D feature map formats. NCDHW is the row-major logical layout; NDC1HWC0 is the fractal L1
// layout (C split into C1 outer / C0 inner, with the depth axis D between N and C1).
struct make_ncdhw_frame_layout { // Shape = (N, C, D, H, W)
    template <typename TraitType, typename N, typename C, typename D, typename H, typename W>
    __aicore__ inline static auto make(N n, C c, D d, H h, W w)
    {
        auto shape = make_shape(n, c, d, h, w);
        auto stride = make_stride(c * d * h * w, d * h * w, h * w, w, _1{});
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<ncdhw_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }
};

struct make_ndc1hwc0_frame_layout { // Shape = (N, D, C1, H, W, C0)
    template <typename TraitType, typename N, typename D, typename C1, typename H, typename W, typename C0>
    __aicore__ inline static auto make(N n, D d, C1 c1, H h, W w, C0 c0)
    {
        auto shape = make_shape(n, d, c1, h, w, c0);
        auto stride = make_stride(d * c1 * h * w * c0, c1 * h * w * c0, h * w * c0, w * c0, c0, _1{});
        using layout_t = layout<decltype(shape), decltype(stride), Std::tuple<ndc1hwc0_layout_ptn, TraitType>>;
        return layout_t(shape, stride);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_PATTERN_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
