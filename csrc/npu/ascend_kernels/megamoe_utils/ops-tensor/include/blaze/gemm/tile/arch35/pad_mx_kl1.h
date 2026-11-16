/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad_mx_kl1.h
 * \brief
 */
#pragma once

#include "tensor_api/tensor.h"
#include "c_api/asc_simd.h"

using asc::te::c0_element;
using asc::te::c0_size;

namespace Blaze::Gemm::Tile {
struct PadMxKL1Base {
    template <typename T>
    __aicore__ inline static void PadZero(const T& tensorL1, uint64_t repeatTimes, uint64_t blockNum, uint64_t dstGap)
    {
        asc_fill_value_config config;
        config.repeat = repeatTimes;
        config.blk_num = blockNum;
        config.dst_gap = dstGap;
        asc_fill_l1((__cbuf__ half*)tensorL1.data().get(), half(0), config);
    }

    template <typename type>
    __aicore__ inline static constexpr bool IsMxFp4()
    {
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp4x2_e1m2_t, __cbuf__ fp4x2_e2m1_t>;
    }

    template <typename type>
    __aicore__ inline static constexpr bool IsMxFp8()
    {
        return AscendC::Std::is_one_of_v<type, __cbuf__ fp8_e5m2_t, __cbuf__ fp8_e4m3fn_t>;
    }
};

struct PadMxKAL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        using type = typename T::element_type;
        static_assert(IsMxFp4<type>() || IsMxFp8<type>(), "Only support mxfp4/mxfp8!");
        auto layoutL1 = tensorL1.layout();
        auto layoutGm = tensorGm.layout();
        auto kAxis = AscendC::Std::get<1>(AscendC::Std::get<1>(layoutGm.shape()));
        auto kAxisL1Align = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.shape())) *
                            AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.shape()));

        if constexpr (asc::te::is_satisfied_ptn_format_v<T, asc::te::nz_layout_ptn>) {
            if constexpr (IsMxFp4<type>()) {
                return;
            }

            if (kAxisL1Align - kAxis < c0_size<type>) {
                return;
            }
            auto mAlign = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.shape())) *
                          AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.shape()));
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, c0_size<type>); // K方向的坐标是ND2NZ指令对齐后的值
            auto sliceTensor = tensorL1.slice(asc::te::make_coord(0, kAxisND2NZAlign),
                                              asc::te::make_shape(mAlign, kAxisL1Align - kAxisND2NZAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
        } else if constexpr (asc::te::is_satisfied_ptn_format_v<T, asc::te::zn_layout_ptn>) {
            // ND2NZ指令只支持给最内轴补零，外轴需要自己清零
            if (kAxis == kAxisL1Align) {
                return;
            }
            // 清零迭代次数为m方向大分形个数，即m1
            auto m1 = AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.shape()));
            auto m0 = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.shape()));
            auto dstRowStride = AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.stride()));
            // A全载场景下dstGap间隔全载A矩阵的K轴；非全载场景下dstRowStride / c0_element<type> = kAxisL1Align
            auto dstGap = (dstRowStride / c0_element<type>)-kAxisL1Align + kAxis;
            auto sliceTensor = tensorL1.slice(asc::te::make_coord(0, kAxis),
                                              asc::te::make_shape(m1 * m0, kAxisL1Align - kAxis));
            PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, dstGap);
        }
    }
};

struct PadMxKBL1 : public PadMxKL1Base {
    template <typename T, typename U>
    __aicore__ inline static void PadZero(const T& tensorL1, const U& tensorGm)
    {
        using type = typename T::element_type;
        static_assert(IsMxFp4<type>() || IsMxFp8<type>(), "Only support mxfp4/mxfp8!");
        auto layoutL1 = tensorL1.layout();
        auto layoutGm = tensorGm.layout();

        // kAxis兼容weight ND和NZ的处理，stride连续的情况下得到实际的k
        auto kAxis = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutGm.shape())) *
                     AscendC::Std::get<1>(AscendC::Std::get<0>(layoutGm.shape()));
        auto kAxisL1Align = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.shape())) *
                            AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.shape()));

        if constexpr (asc::te::is_satisfied_ptn_format_v<T, asc::te::nz_layout_ptn>) {
            // ND2NZ指令只支持给最内轴补零，外轴需要自己清零
            if (kAxis == kAxisL1Align) {
                return;
            }

            // 清零迭代次数为n方向大分形个数，即n1
            auto n1 = AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.shape()));
            auto n0 = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.shape()));
            auto sliceTensor = tensorL1.slice(asc::te::make_coord(kAxis, 0),
                                              asc::te::make_shape(kAxisL1Align - kAxis, n1 * n0));
            // B矩阵不涉及全载场景，dstGap = kAxis
            PadMxKL1Base::PadZero(sliceTensor, n1, kAxisL1Align - kAxis, kAxis);
        } else if constexpr (asc::te::is_satisfied_ptn_format_v<T, asc::te::zn_layout_ptn>) {
            if constexpr (IsMxFp4<type>()) {
                return;
            }

            if (kAxisL1Align - kAxis < c0_size<type>) {
                return;
            }
            auto nAlign = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.shape())) *
                          AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.shape()));
            auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, c0_size<type>); // K方向的坐标是ND2NZ指令对齐后的值
            auto sliceTensor = tensorL1.slice(asc::te::make_coord(kAxisND2NZAlign, 0),
                                              asc::te::make_shape(kAxisL1Align - kAxisND2NZAlign, nAlign));
            PadMxKL1Base::PadZero(sliceTensor, 1, nAlign, 0);
        }
    }
};
} // namespace Blaze::Gemm::Tile
