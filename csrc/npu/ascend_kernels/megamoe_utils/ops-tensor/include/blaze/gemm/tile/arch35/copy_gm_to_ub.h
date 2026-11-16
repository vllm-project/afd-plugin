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
 * \file copy_gm_to_ub.h
 * \brief Custom copy primitive for packed B4 weight movement from GM to UB.
 */
#pragma once

#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Tile {

// Custom GM-to-UB copy for packed 4-bit weight tensors.
struct CopyGM2UBWeight {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using SrcLayoutPattern = asc::te::get_layout_pattern<typename U::layout_type>;
        using DstLayoutPattern = asc::te::get_layout_pattern<typename T::layout_type>;
        constexpr bool IS_ZN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::zn_layout_ptn>;
        constexpr bool IS_DN_WEIGHT = AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::dn_ext_layout_ptn>;
        static_assert(IS_ZN_WEIGHT || IS_DN_WEIGHT, "Packed weight copy only supports ZN and DNExt source layouts");
        static_assert(AscendC::Std::is_same_v<DstLayoutPattern, SrcLayoutPattern>,
                      "Packed weight copy source and destination layouts must match");
        static_assert(sizeof(typename T::element_type) == sizeof(typename U::element_type) &&
                          sizeof(typename T::element_type) == 1,
                      "Packed weight copy requires matching packed source and destination elements");

        if constexpr (IS_DN_WEIGHT) {
            CopyDnPackedWeight(dst, src);
        } else {
            CopyZnPackedWeight(dst, src);
        }
    }

private:
    template <typename T, typename U>
    __aicore__ inline static void CopyDnPackedWeight(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.layout();
        const auto& srcLayout = src.layout();

        asc_load_l2_cache_mode cacheMode = static_cast<asc_load_l2_cache_mode>(src.engine().get_cache_mode());
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);
        auto dstStrideTuple = asc::te::get_stride(dstLayout);

        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<1>(srcShape));
        uint32_t kLen = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));
        uint32_t srcRowStride = AscendC::Std::get<1>(AscendC::Std::get<1>(srcStrideTuple));
        uint32_t dstRowStride = AscendC::Std::get<1>(AscendC::Std::get<1>(dstStrideTuple));

        // Packed FP4 stores two logical K elements in one byte. The direct
        // intrinsic consumes the byte distance between adjacent N-row starts.
        uint32_t blockLen = kLen >> 1U;
        int64_t srcRowSpanBytes = srcRowStride >> 1U;
        int64_t dstRowSpanBytes = dstRowStride >> 1U;
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)dst.data().get(), (__gm__ uint8_t*)src.data().get(), blockCount,
                             blockLen, 0, 0, false, cacheMode, srcRowSpanBytes, dstRowSpanBytes);
    }

    template <typename T, typename U>
    __aicore__ inline static void CopyZnPackedWeight(const T& dst, const U& src)
    {
        const auto& dstLayout = dst.layout();
        const auto& srcLayout = src.layout();
        asc_load_l2_cache_mode cacheMode = static_cast<asc_load_l2_cache_mode>(src.engine().get_cache_mode());

        // Get shape and stride
        auto srcShape = asc::te::get_shape(srcLayout);
        auto srcStrideTuple = asc::te::get_stride(srcLayout);
        auto dstStrideTuple = asc::te::get_stride(dstLayout);

        // Extract k1 from shape: ((c0, k1), (n0, n1)).
        uint16_t blockCount = AscendC::Std::get<1>(AscendC::Std::get<0>(srcShape));

        // For packed B4 (2 elements per byte), convert element strides to byte strides via >> 1.
        uint32_t blockLen = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;
        int64_t srcStride = AscendC::Std::get<1>(AscendC::Std::get<0>(srcStrideTuple)) >> 1;
        int64_t dstStride = AscendC::Std::get<1>(AscendC::Std::get<0>(dstStrideTuple)) >> 1;

        asc_copy_gm2ub_align((__ubuf__ uint8_t*)dst.data().get(), (__gm__ uint8_t*)src.data().get(), blockCount,
                             blockLen, 0, 0, false, cacheMode, srcStride, dstStride);
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Blaze

namespace asc {
namespace te {

template <typename Traits>
struct copy_traits<Blaze::Gemm::Tile::CopyGM2UBWeight, Traits>
    : public copy_traits<Blaze::Gemm::Tile::CopyGM2UBWeight, Traits, Blaze::Gemm::Tile::CopyGM2UBWeight, Traits> {};

template <>
struct copy_traits<Blaze::Gemm::Tile::CopyGM2UBWeight>
    : public copy_traits<Blaze::Gemm::Tile::CopyGM2UBWeight, gm_to_ub_trait_default> {};

} // namespace te
} // namespace asc
