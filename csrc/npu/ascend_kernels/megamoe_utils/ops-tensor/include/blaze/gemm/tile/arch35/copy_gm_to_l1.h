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
 * \file copy_gm_to_l1.h
 * \brief
 */
#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"

namespace Blaze::Gemm::Tile {
using asc::te::c0_element;

struct CopyConcatGM2L1Params {
    uint64_t n;
    uint64_t k;
};

struct CopySliceGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src)
    {
        using srcType = typename U::element_type;
        auto layoutGm = src.layout(); // shape: [ndNum, [sliceM, curK]], stride: [oriM * k, [k, 1]]
        auto layoutL1 = dst.layout(); // l1 shape: [mL1, kL1] ==> NZ: ((m0, m1), (k0, k1))

        auto m0 = asc::te::get<0>(asc::te::get<MNK_M>(layoutL1.shape()));
        auto m1 = asc::te::get<1>(asc::te::get<MNK_M>(layoutL1.shape()));
        uint32_t mL1 = m1 * m0; // curML1
        uint16_t ndNum = static_cast<uint16_t>(asc::te::get<0>(layoutGm.shape()));
        uint16_t nValue = static_cast<uint16_t>(asc::te::get<0>(asc::te::get<1>(layoutGm.shape())));
        uint32_t dValue = static_cast<uint32_t>(asc::te::get<1>(asc::te::get<1>(layoutGm.shape())));
        uint64_t srcDValue = asc::te::get<0>(asc::te::get<1>(layoutGm.stride()));
        uint32_t dstNzC0Stride = AscendC::Std::ceil_align(mL1, AscendC::BLOCK_CUBE);
        uint64_t srcNdMatrixStride = asc::te::get<0>(layoutGm.stride());
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = nValue * c0_element<srcType>;
        uint64_t loop1SrcStride = srcDValue * sizeof(srcType);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(srcType);
        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride / C0_element
        uint16_t loop4DstStride = dstNzMatrixStride / c0_element<srcType>;
        uint8_t cacheMode = src.engine().get_cache_mode();

        if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ half*)(dst.data().get()), (__gm__ half*)(src.data().get()), ndNum,
                                   loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue,
                                   dValue, loop4SrcStride, false);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ float*)(dst.data().get()), (__gm__ float*)(src.data().get()), ndNum,
                                   loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue,
                                   dValue, loop4SrcStride, false);
        }
    }

private:
    template <typename T>
    __aicore__ inline static void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t ndNum,
                                                         uint16_t loop2DstStride, uint16_t loop3DstStride,
                                                         uint16_t loop4DstStride, uint64_t loop1SrcStride,
                                                         uint8_t cacheMode, uint16_t nValue, uint32_t dValue,
                                                         uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (asc::te::current_arch_version == asc::te::arch_version::v3510) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
            asc::te::set_mte2_nz_para(mte2NzPara); // CCE: store parameters for ND2NZ DMA instructions
            asc_copy_gm2l1_nd2nz(dst, src, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }
};

struct CopyConcatGM2L1 {
    template <typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src, const CopyConcatGM2L1Params& params)
    {
        using SrcLayoutPtn = asc::te::get_layout_pattern<typename U::layout_type>;
        constexpr bool isScaleB = AscendC::Std::is_one_of_v<SrcLayoutPtn, asc::te::scaleb_nd_layout_ptn,
                                                            asc::te::scaleb_dn_layout_ptn>;
        if constexpr (isScaleB) {
            CopyScaleB(dst, src, params);
        } else {
            CopyB(dst, src, params);
        }
    }

private:
    static constexpr uint16_t CONCAT_MATRIX_NUM = 2U;
    static constexpr uint64_t CONCAT_N_FACTOR = 2UL;

    template <typename T, typename U>
    __aicore__ inline static void CopyB(const T& dst, const U& src, const CopyConcatGM2L1Params& params)
    {
        using ElementType = asc::te::get_attribute_element_type<typename U::element_type*>;
        using CopyType = AscendC::Std::conditional_t<(sizeof(ElementType) == 1), int8_t, ElementType>;
        using SrcLayoutPtn = asc::te::get_layout_pattern<typename U::layout_type>;
        constexpr bool isTrans = IsTrans<SrcLayoutPtn>::value;
        auto srcLayout = src.layout();
        auto dstLayout = dst.layout();
        const uint64_t singleN = asc::te::get_total_column_shape(srcLayout);
        const uint64_t curGmBKL1 = asc::te::get_total_row_shape(srcLayout);
        const uint64_t l1K = asc::te::get_total_row_shape(dstLayout);
        const uint64_t halfN = params.n >> 1;
        const uint64_t srcMatrixStride = PackFp4Size<ElementType>(isTrans ? (halfN * params.k) : halfN);
        const uint64_t srcDValue = PackFp4Size<ElementType>(isTrans ? params.k : params.n);
        const uint16_t nValue = static_cast<uint16_t>(isTrans ? singleN : curGmBKL1);
        const uint32_t dValue = static_cast<uint32_t>(PackFp4Size<ElementType>(isTrans ? curGmBKL1 : singleN));
        constexpr uint64_t c0Size = IsFp4<ElementType>() ? C0_SIZE_B4 : C0_SIZE_B8;
        const uint16_t dstRowStride = static_cast<uint16_t>(isTrans ? (CeilAlign(singleN, c0Size) * CONCAT_N_FACTOR) :
                                                                      l1K);
        const uint16_t dstMatrixStride = static_cast<uint16_t>(isTrans ? CeilAlign(singleN, c0Size) :
                                                                         (l1K * CeilAlign(singleN, c0Size) / c0Size));

        constexpr uint16_t loop2DstStride = 1U;
        const uint16_t loop3DstStride = dstRowStride;
        const uint16_t loop4DstStride = dstMatrixStride;
        const uint64_t loop1SrcStride = srcDValue * sizeof(CopyType);
        const uint64_t loop4SrcStride = srcMatrixStride * sizeof(CopyType);
        const uint8_t cacheMode = src.engine().get_cache_mode();
        CopyGmToCbufMultiNd2nz(reinterpret_cast<__cbuf__ CopyType*>(dst.data().get()),
                               reinterpret_cast<__gm__ CopyType*>(src.data().get()), CONCAT_MATRIX_NUM, loop2DstStride,
                               loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue, dValue,
                               loop4SrcStride, false);
    }

    template <typename T, typename U>
    __aicore__ inline static void CopyScaleB(const T& dst, const U& src, const CopyConcatGM2L1Params& params)
    {
        using SrcLayoutPtn = asc::te::get_layout_pattern<typename U::layout_type>;
        constexpr bool isTrans = IsTrans<SrcLayoutPtn>::value;
        auto srcLayout = src.layout();
        auto dstLayout = dst.layout();
        const uint64_t singleN = asc::te::get_total_column_shape(srcLayout);
        const uint64_t curScaleSpan = asc::te::get_total_row_shape(srcLayout);
        const uint64_t scaleKL1Block = asc::te::get_total_row_shape(dstLayout) / MXFP_MULTI_BASE_SIZE;
        const uint64_t fullScaleKBlock = CeilDiv(params.k, MXFP_DIVISOR_SIZE);
        const uint64_t halfN = params.n >> 1;
        const uint64_t srcMatrixStride = isTrans ? (halfN * fullScaleKBlock) : halfN;
        const uint64_t srcDValue = isTrans ? fullScaleKBlock : params.n;
        const uint16_t nValue = static_cast<uint16_t>(curScaleSpan / MXFP_MULTI_BASE_SIZE);
        const uint32_t dValue = static_cast<uint32_t>(singleN);
        constexpr uint64_t halfC0Element = AscendC::ONE_BLK_SIZE / sizeof(half);
        const uint16_t dstRowStride = static_cast<uint16_t>(scaleKL1Block);
        const uint16_t dstMatrixStride = static_cast<uint16_t>(
            (isTrans ? Align16(singleN) : CeilAlign(singleN, BLOCK_CUBE)) * scaleKL1Block / halfC0Element);

        CopyWithStrides<half, isTrans>(dst, src, nValue, dValue, srcMatrixStride, srcDValue, dstMatrixStride,
                                       dstRowStride);
    }

    template <typename ElementType>
    __aicore__ inline static uint64_t PackFp4Size(uint64_t value)
    {
        if constexpr (IsFp4<ElementType>()) {
            return value >> 1;
        }
        return value;
    }

    template <typename CopyType, bool IsTrans, typename T, typename U>
    __aicore__ inline static void CopyWithStrides(const T& dst, const U& src, uint16_t nValue, uint32_t dValue,
                                                  uint64_t srcMatrixStride, uint64_t srcDValue,
                                                  uint16_t dstMatrixStride, uint16_t dstRowStride)
    {
        constexpr uint16_t loop2DstStride = 1U;
        const uint16_t loop3DstStride = dstRowStride;
        const uint16_t loop4DstStride = dstMatrixStride;
        const uint64_t loop1SrcStride = srcDValue * sizeof(CopyType);
        const uint64_t loop4SrcStride = srcMatrixStride * sizeof(CopyType);
        const uint8_t cacheMode = src.engine().get_cache_mode();
        if constexpr (IsTrans) {
            CopyGmToCbufMultiDn2nz(reinterpret_cast<__cbuf__ CopyType*>(dst.data().get()),
                                   reinterpret_cast<__gm__ CopyType*>(src.data().get()), CONCAT_MATRIX_NUM,
                                   loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue,
                                   dValue, loop4SrcStride, false);
        } else {
            CopyGmToCbufMultiNd2nz(reinterpret_cast<__cbuf__ CopyType*>(dst.data().get()),
                                   reinterpret_cast<__gm__ CopyType*>(src.data().get()), CONCAT_MATRIX_NUM,
                                   loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode, nValue,
                                   dValue, loop4SrcStride, false);
        }
    }

    template <typename T>
    __aicore__ inline static void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t ndNum,
                                                         uint16_t loop2DstStride, uint16_t loop3DstStride,
                                                         uint16_t loop4DstStride, uint64_t loop1SrcStride,
                                                         uint8_t cacheMode, uint16_t nValue, uint32_t dValue,
                                                         uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (asc::te::current_arch_version == asc::te::arch_version::v3510) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
            asc::te::set_mte2_nz_para(mte2NzPara); // CCE: store parameters for ND2NZ DMA instructions
            asc_copy_gm2l1_nd2nz(dst, src, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }

    template <typename T>
    __aicore__ inline static void CopyGmToCbufMultiDn2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t dnNum,
                                                         uint16_t loop2DstStride, uint16_t loop3DstStride,
                                                         uint16_t loop4DstStride, uint64_t loop1SrcStride,
                                                         uint8_t cacheMode, uint16_t nValue, uint32_t dValue,
                                                         uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (asc::te::current_arch_version == asc::te::arch_version::v3510) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(dnNum);                        // MTE2_NZ_PARA[15:0]
            asc::te::set_mte2_nz_para(mte2NzPara); // CCE: store parameters for DN2NZ DMA instructions
            asc_copy_gm2l1_dn2nz(dst, src, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }
};

} // namespace Blaze::Gemm::Tile

namespace asc {
namespace te {

// 特化Traits，绑定自定义GM->L1拷贝实现
template <typename Traits>
struct copy_traits<Blaze::Gemm::Tile::CopySliceGM2L1, Traits>
    : public copy_traits<Blaze::Gemm::Tile::CopySliceGM2L1, Traits, Blaze::Gemm::Tile::CopySliceGM2L1, Traits> {};

template <>
struct copy_traits<Blaze::Gemm::Tile::CopySliceGM2L1>
    : public copy_traits<Blaze::Gemm::Tile::CopySliceGM2L1, gm_to_l1_trait_default> {};

template <>
struct copy_traits<Blaze::Gemm::Tile::CopyConcatGM2L1> {
    using trait_type = typename gm_to_l1_trait_default::trait_type;
    using TraitType = trait_type;
    static constexpr const trait_type default_trait = gm_to_l1_trait_default::value;

    __aicore__ inline constexpr copy_traits with(const Blaze::Gemm::Tile::CopyConcatGM2L1Params& copyParams) const
    {
        return {copyParams};
    }

    template <const trait_type& trait = default_trait, typename T, typename U>
    __aicore__ inline void CopyUnpack(const T& dst, const U& src) const
    {
        (void)trait;
        Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(dst, src, params);
    }

    Blaze::Gemm::Tile::CopyConcatGM2L1Params params{};
};

} // namespace te
} // namespace asc
