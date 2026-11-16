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
    "impl/tensor_api/arch/utils/check_format.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file check_format.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_CHECK_FORMAT_H
#define IMPL_TENSOR_API_ARCH_UTILS_CHECK_FORMAT_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/pointer_pattern.h"
#include "impl/tensor_api/tensor/tensor_impl.h"
#include "impl/tensor_api/arch/utils/is_format.h"

namespace asc {
namespace te {
struct check_nz_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        constexpr auto c0_element = TraitType::c0_element;
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, Std::Int<fractal_fixed>>,
                      "Layout->Shape->Row->ZeroDim must be Int<16>!");
        static_assert(Std::is_same_v<shape_column0, Std::Int<c0_element>>,
                      "Layout->Shape->Column->ZeroDim is different from c0_element!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        using stride_row1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 1>::type;
        static_assert(Std::is_same_v<stride_row0, Std::Int<c0_element>>,
                      "Layout->Stride->Row->ZeroDim is different from c0_element!");
        static_assert(Std::is_same_v<stride_column0, _1>, "Layout->Stride->Column->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<stride_row1, Std::Int<c0_element * fractal_fixed>>,
                      "Layout->Stride->Row->OneDim is different from c0_element * fractal_fixed!");
    }
};

struct check_nd_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_column0, _1>, "Layout->Stride->Column must be Int<1>!");
    }
};

struct check_dn_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        static_assert(Std::is_same_v<stride_row0, _1>, "Src->Layout->Stride->Row must be Int<1>!");
    }
};

struct check_one_dim_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using layout_type = typename Tensor::layout_type;
        static_assert(layout_type::rank_size == 1, "One-dim layout rank must be 1!");
    }
};

struct check_nd_ext_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _1>, "Layout->Shape->Row->ZeroDim must be 1!");
        static_assert(Std::is_same_v<shape_column0, _1>, "Layout->Shape->Column->ZeroDim must be 1!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        using stride_column1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 1>::type;
        static_assert(Std::is_same_v<stride_row0, _0>, "Layout->Stride->Row->ZeroDim must be 0!");
        static_assert(Std::is_same_v<stride_column0, _0>, "Layout->Stride->Column->ZeroDim must be 0!");
        static_assert(Std::is_same_v<stride_column1, _1>, "Layout->Stride->Column->OneDim must be 1!");
    }
};

struct check_dn_ext_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _1>, "Src->Layout->Shape->Row->ZeroDim must be 1!");
        static_assert(Std::is_same_v<shape_column0, _1>, "Src->Layout->Shape->Column->ZeroDim must be 1!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_row1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 1>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_row0, _0>, "Src->Layout->Stride->Row->ZeroDim must be 0!");
        static_assert(Std::is_same_v<stride_row1, _1>, "Src->Layout->Stride->Row->OneDim must be 1!");
        static_assert(Std::is_same_v<stride_column0, _0>, "Src->Layout->Stride->Column->ZeroDim must be 0!");
    }
};

struct check_nn_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _2>, "Src->Layout->Shape->Row->ZeroDim must be Int<2>!");
        static_assert(Std::is_same_v<shape_column0, _16>, "Src->Layout->Shape->Column->ZeroDim must be Int<16>!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_row1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 1>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_row0, _1>, "Src->Layout->Stride->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<stride_row1, _32>, "Src->Layout->Stride->Row->OneDim must be Int<32>!");
        static_assert(Std::is_same_v<stride_column0, _2>, "Src->Layout->Stride->Column->ZeroDim must be Int<2>!");
    }
};

struct check_zz_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        constexpr auto c0_element = TraitType::c0_element;
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_column0, Std::Int<c0_element>>,
                      "Layout->Shape->Column->ZeroDim is different from c0_element!");
        static_assert(Std::is_same_v<shape_row0, Std::Int<fractal_fixed>>,
                      "Layout->Shape->Row->ZeroDim must be Int<16>!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_column0, _1>, "Layout->Stride->Column->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<stride_row0, Std::Int<c0_element>>,
                      "Layout->Stride->Row->ZeroDim is different from c0_element!");
    }
};

struct check_zn_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        constexpr auto c0_element = TraitType::c0_element;
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_column0, Std::Int<fractal_fixed>>,
                      "Layout->Shape->Column->ZeroDim must be Int<16>!");
        static_assert(Std::is_same_v<shape_row0, Std::Int<c0_element>>,
                      "Layout->Shape->Row->ZeroDim is different from c0_element!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_column0, Std::Int<c0_element>>,
                      "Layout->Stride->Column->ZeroDim is different from c0_element!");
        static_assert(Std::is_same_v<stride_row0, _1>, "Layout->Stride->Row->ZeroDim must be Int<1>!");
    }
};

struct check_scalea_nd_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _1>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<shape_column0, _1>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        using stride_column1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 1>::type;
        static_assert(Std::is_same_v<stride_row0, _0>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<stride_column0, _0>, "Layout->Stride->Column->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<stride_column1, _1>, "Layout->Stride->Column->OneDim must be Int<1>!");
    }
};

struct check_scalea_dn_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _1>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<shape_column0, _2>, "Layout->Shape->Column->ZeroDim must be Int<2>!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_row1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 1>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_row0, _0>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<stride_row1, _2>, "Layout->Stride->Row->OneDim must be Int<2>!");
        static_assert(Std::is_same_v<stride_column0, _1>, "Layout->Stride->Column->ZeroDim must be Int<1>!");
    }
};

struct check_scaleb_nd_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _2>, "Layout->Shape->Row->ZeroDim must be Int<2>!");
        static_assert(Std::is_same_v<shape_column0, _1>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        using stride_column1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 1>::type;
        static_assert(Std::is_same_v<stride_row0, _1>, "Layout->Stride->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<stride_column0, _0>, "Layout->Stride->Column->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<stride_column1, _2>, "Layout->Stride->Column->OneDim must be Int<2>!");
    }
};

struct check_scaleb_dn_layout_pattern {
    template <typename Tensor, typename TraitType>
    __aicore__ inline static constexpr void check()
    {
        using shape_row0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::row, 0>::type;
        using shape_column0 = typename get_n_dim_type<Tensor, attr_info::shape, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<shape_row0, _1>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<shape_column0, _1>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using stride_row0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 0>::type;
        using stride_row1 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::row, 1>::type;
        using stride_column0 = typename get_n_dim_type<Tensor, attr_info::stride, attr_info::column, 0>::type;
        static_assert(Std::is_same_v<stride_row0, _0>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<stride_row1, _1>, "Layout->Stride->Row->OneDim must be Int<1>!");
        static_assert(Std::is_same_v<stride_column0, _0>, "Layout->Stride->Column->ZeroDim must be Int<0>!");
    }
};

using layout_pattern_check_set =
    tuple_map<Std::tuple<zn_layout_ptn, check_zn_layout_pattern>, Std::tuple<zz_layout_ptn, check_zz_layout_pattern>,
              Std::tuple<nn_layout_ptn, check_nn_layout_pattern>, Std::tuple<nz_layout_ptn, check_nz_layout_pattern>,
              Std::tuple<nd_layout_ptn, check_nd_layout_pattern>, Std::tuple<dn_layout_ptn, check_dn_layout_pattern>,
              Std::tuple<one_dim_layout_ptn, check_one_dim_layout_pattern>,
              Std::tuple<nd_ext_layout_ptn, check_nd_ext_layout_pattern>,
              Std::tuple<dn_ext_layout_ptn, check_dn_ext_layout_pattern>,
              Std::tuple<scalea_nd_layout_ptn, check_scalea_nd_layout_pattern>,
              Std::tuple<scalea_dn_layout_ptn, check_scalea_dn_layout_pattern>,
              Std::tuple<scaleb_nd_layout_ptn, check_scaleb_nd_layout_pattern>,
              Std::tuple<scaleb_dn_layout_ptn, check_scaleb_dn_layout_pattern>>;

template <typename Tensor>
__aicore__ inline void constexpr check_one_layout_pattern()
{
    using layout = typename Tensor::layout_type;
    using layout_pattern = get_layout_pattern<layout>;
    using trait_type = get_layout_trait<layout>;
    using pattern_check = typename layout_pattern_check_set::template get<layout_pattern>;
    static_assert(!Std::is_same_v<pattern_check, Std::ignore_t>, "Unsupported layout pattern.");
    pattern_check::template check<Tensor, trait_type>();
}

template <typename... Args>
__aicore__ inline void constexpr check_layout_pattern()
{
    (check_one_layout_pattern<Args>(), ...);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_UTILS_CHECK_FORMAT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
