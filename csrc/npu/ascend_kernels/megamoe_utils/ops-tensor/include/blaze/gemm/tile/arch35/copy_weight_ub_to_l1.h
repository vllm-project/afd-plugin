/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Tile {

// Custom UB-to-L1 copy for converted 8-bit weight. The source pattern describes
// the physical UB layout emitted by the corresponding W4-to-W8 path.
struct CopyUB2L1Weight8Bit {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using SrcLayoutPattern = asc::te::get_layout_pattern<typename U::layout_type>;
        using DstLayoutPattern = asc::te::get_layout_pattern<typename T::layout_type>;
        static_assert(AscendC::Std::is_same_v<DstLayoutPattern, asc::te::zn_layout_ptn>,
                      "Converted weight copy requires a standard ZN L1 destination layout");
        static_assert(sizeof(typename T::element_type) == sizeof(typename U::element_type) &&
                          sizeof(typename T::element_type) == 1,
                      "Converted weight copy requires matching 8-bit source and destination elements");
        constexpr bool IS_DN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, Weight8BitDnToZnUbLayoutPtn>;
        constexpr bool IS_NZ_UB_LAYOUT = AscendC::Std::is_same_v<SrcLayoutPattern, Weight8BitZnToZnUbLayoutPtn>;
        static_assert(IS_DN_WEIGHT || IS_NZ_UB_LAYOUT, "Converted weight copy requires a supported UB layout");

        if constexpr (IS_DN_WEIGHT) {
            CopyDnToZnWeight(dst, src);
        } else {
            CopyZnToZnWeight(dst, src);
        }
    }

private:
    template <typename T, typename U>
    __aicore__ inline static void CopyDnToZnWeight(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.layout();
        const auto& srcLayout = src.layout();
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);
        auto dstStrideTuple = asc::te::get_stride(dstLayout);
        uint16_t blockCount = static_cast<uint16_t>(AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape)));
        uint32_t blockLen = static_cast<uint32_t>(AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape)));
        int64_t srcBlockSpan = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStrideTuple)) / BLOCK_BYTE_SIZE;
        int64_t dstBlockSpan = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) / BLOCK_BYTE_SIZE;
        int64_t srcGap = srcBlockSpan - static_cast<int64_t>(blockLen);
        int64_t dstGap = dstBlockSpan - static_cast<int64_t>(blockLen);
        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(), blockCount, blockLen, srcGap,
                       dstGap);
    }

    template <typename T, typename U>
    __aicore__ inline static void CopyZnToZnWeight(const T& dst, const U& src)
    {
        using type = typename U::element_type;
        const auto& srcLayout = src.layout();

        // Get shape and stride tuples
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);

        // Extract dimensions from srcShape = ((c0, k1), (n0, n1))
        // Std::get<0>(srcShape) = (c0, k1), Std::get<1>(srcShape) = (n0, n1)
        uint16_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(srcShape));
        uint16_t k1 = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint16_t n0 = AscendC::Std::get<0>(AscendC::Std::get<1>(srcShape));
        uint16_t n1 = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));

        // Extract innerStride from srcStride = ((1, n1*InnerStride), (c0, InnerStride))
        // For UB2L1, use column stride (InnerStride) not row stride (n1*InnerStride)
        int64_t innerStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));

        // Total number of fractal blocks to copy
        uint16_t blockCount = k1 * n1;

        // Block length in 32B units
        uint32_t blockLen = (n0 * c0 * sizeof(type)) / 32;

        // Source stride in 32B units
        int64_t srcStride = (innerStride * sizeof(type)) / 32 - blockLen;

        // Destination stride in 32B units (contiguous in L1)
        int64_t dstStride = 0;

        asc_copy_ub2l1((__cbuf__ void*)dst.data().get(), (__ubuf__ void*)src.data().get(), blockCount, blockLen,
                       srcStride, dstStride);
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Blaze

namespace asc {
namespace te {

template <typename Traits>
struct copy_traits<Blaze::Gemm::Tile::CopyUB2L1Weight8Bit, Traits>
    : public copy_traits<Blaze::Gemm::Tile::CopyUB2L1Weight8Bit, Traits, Blaze::Gemm::Tile::CopyUB2L1Weight8Bit,
                         Traits> {};

template <>
struct copy_traits<Blaze::Gemm::Tile::CopyUB2L1Weight8Bit>
    : public copy_traits<Blaze::Gemm::Tile::CopyUB2L1Weight8Bit, ub_to_l1_trait_default> {};

} // namespace te
} // namespace asc
