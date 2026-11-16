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
    "impl/tensor_api/legacy/legacy_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_LEGACY_UTILS_H
#endif

#ifndef IMPL_TENSOR_API_LEGACY_LEGACY_UTILS_H
#define IMPL_TENSOR_API_LEGACY_LEGACY_UTILS_H

#include "include/tensor_api/utils/utils.h"
#include "tensor_api/arch/cube/copy_trait.h"
#include "tensor_api/arch/cube/mmad_trait.h"
#include "impl/tensor_api/utils/extra_impl.h"
#include "impl/tensor_api/utils/location_impl.h"

namespace asc {
namespace te {
template <typename PadType>
struct img2col_params;
} // namespace te
} // namespace asc

namespace AscendC {
namespace Te {

using namespace asc::te;

template <typename Pointer>
using GetAttributeElementType = asc::te::get_attribute_element_type<Pointer>;

enum class CacheMode : uint8_t {
    CACHE_MODE_NORMAL = 0,
    CACHE_MODE_DISABLE = 4,
    CACHE_MODE_LAST = 5,
    CACHE_MODE_PERSISTENT = 6
};
enum class MmadType : uint8_t { NORMAL = 0, MX = 1 };
enum class RoundMode : uint8_t { DEFAULT = 0, HYBRID = 1 };

enum DualDstMode : uint8_t { DUAL_DST_DISABLE = 0, DUAL_DST_SPLIT_M = 1, DUAL_DST_SPLIT_N = 2 };

struct CopyL0C2GMTrait {
    __aicore__ constexpr CopyL0C2GMTrait() {}

    __aicore__ constexpr CopyL0C2GMTrait(RoundMode roundMode, bool enableRelu, bool enableChannelSplit) :
        roundMode(roundMode), enableRelu(enableRelu), enableChannelSplit(enableChannelSplit)
    {}

    RoundMode roundMode = RoundMode::DEFAULT;
    bool enableRelu = false;
    bool enableChannelSplit = false;
};

struct CopyL0C2UBTrait {
    __aicore__ constexpr CopyL0C2UBTrait() {}

    __aicore__ constexpr CopyL0C2UBTrait(RoundMode roundMode, bool enableRelu, bool enableChannelSplit,
                                         DualDstMode dualDstCtl) :
        roundMode(roundMode), enableRelu(enableRelu), enableChannelSplit(enableChannelSplit), dualDstCtl(dualDstCtl)
    {}

    RoundMode roundMode = RoundMode::DEFAULT;
    bool enableRelu = false;
    bool enableChannelSplit = false;
    DualDstMode dualDstCtl = DUAL_DST_DISABLE;
};

struct CopyL0C2L1Trait {
    __aicore__ constexpr CopyL0C2L1Trait() {}

    __aicore__ constexpr CopyL0C2L1Trait(RoundMode roundMode, bool enableRelu, bool enableChannelSplit) :
        roundMode(roundMode), enableRelu(enableRelu), enableChannelSplit(enableChannelSplit)
    {}

    RoundMode roundMode = RoundMode::DEFAULT;
    bool enableRelu = false;
    bool enableChannelSplit = false;
};

struct MmadTrait {
    __aicore__ constexpr MmadTrait(){};

    __aicore__ constexpr MmadTrait(int32_t fmOffset, bool kDirectionAlign, bool cmatrixSource, bool disableGemv,
                                   MmadType mmadType) :
        fmOffset(fmOffset), kDirectionAlign(kDirectionAlign), cmatrixSource(cmatrixSource), disableGemv(disableGemv),
        mmadType(mmadType)
    {}

    int32_t fmOffset = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool disableGemv = true;
    MmadType mmadType = MmadType::NORMAL;
};

struct MmadParams {
    __aicore__ constexpr MmadParams(){};

    __aicore__ constexpr MmadParams(uint16_t m, uint16_t n, uint16_t k, uint8_t unitFlag, bool cmatrixInitVal) :
        m(m), n(n), k(k), unitFlag(unitFlag), cmatrixInitVal(cmatrixInitVal)
    {}

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    uint8_t unitFlag = 0;
    bool cmatrixInitVal = false;
};

struct FixpipeParams {
    __aicore__ constexpr FixpipeParams(){};

    __aicore__ constexpr FixpipeParams(uint8_t unitFlag, bool subBlockId = false) :
        unitFlag(unitFlag), subBlockId(subBlockId)
    {}

    uint8_t unitFlag = 0;
    bool subBlockId = false;
};

struct CopyGM2UBParams {
    __aicore__ constexpr CopyGM2UBParams() {}

    __aicore__ constexpr CopyGM2UBParams(uint8_t leftPaddingCount, uint8_t rightPaddingCount,
                                         bool enableConstantPad = true) :
        leftPaddingCount(leftPaddingCount), rightPaddingCount(rightPaddingCount), enableConstantPad(enableConstantPad)
    {}

    uint8_t leftPaddingCount = 0;
    uint8_t rightPaddingCount = 0;
    bool enableConstantPad = true;
};

template <typename PadType>
struct Img2ColParams {
    uint16_t mExtension = 0;
    uint16_t kExtension = 0;
    uint16_t mStartPt = 0;
    uint16_t kStartPt = 0;
    uint8_t padList[4] = {0, 0, 0, 0};
    uint8_t strideW = 1;
    uint8_t strideH = 1;
    uint8_t filterW = 1;
    uint8_t filterH = 1;
    uint8_t dilationFilterW = 1;
    uint8_t dilationFilterH = 1;
    bool filterSizeW = false;
    bool filterSizeH = false;
    bool transpose = false;
    bool fMatrixCtrl = false;
    PadType padValue = 0;
};

// normalize 函数: PascalCase → snake_case 类型转换
__aicore__ inline constexpr cache_mode normalize_cache_mode(cache_mode mode)
{
    return mode;
}

__aicore__ inline constexpr cache_mode normalize_cache_mode(CacheMode mode)
{
    switch (mode) {
    case CacheMode::CACHE_MODE_DISABLE:
        return cache_mode::disable;
    case CacheMode::CACHE_MODE_LAST:
        return cache_mode::last;
    case CacheMode::CACHE_MODE_PERSISTENT:
        return cache_mode::persistent;
    case CacheMode::CACHE_MODE_NORMAL:
    default:
        return cache_mode::normal;
    }
}

__aicore__ inline constexpr mmad_type normalize_mmad_type(mmad_type type)
{
    return type;
}

__aicore__ inline constexpr mmad_type normalize_mmad_type(MmadType type)
{
    switch (type) {
    case MmadType::MX:
        return mmad_type::mx;
    case MmadType::NORMAL:
    default:
        return mmad_type::normal;
    }
}

__aicore__ inline constexpr round_mode normalize_round_mode(round_mode mode)
{
    return mode;
}

__aicore__ inline constexpr round_mode normalize_round_mode(RoundMode mode)
{
    switch (mode) {
    case RoundMode::HYBRID:
        return round_mode::hybrid;
    case RoundMode::DEFAULT:
    default:
        return round_mode::default_round;
    }
}

__aicore__ inline constexpr dual_dst_mode normalize_dual_dst_mode(dual_dst_mode mode)
{
    return mode;
}

__aicore__ inline constexpr dual_dst_mode normalize_dual_dst_mode(DualDstMode mode)
{
    switch (mode) {
    case DUAL_DST_SPLIT_M:
        return dual_dst_mode::split_m;
    case DUAL_DST_SPLIT_N:
        return dual_dst_mode::split_n;
    case DUAL_DST_DISABLE:
    default:
        return dual_dst_mode::disable;
    }
}

template <typename Trait>
__aicore__ inline constexpr mmad_trait normalize_mmad_trait(const Trait& trait)
{
    using trait_type = Std::remove_cvref_t<Trait>;
    if constexpr (Std::is_same_v<trait_type, mmad_trait>) {
        return trait;
    } else {
        static_assert(Std::is_same_v<trait_type, MmadTrait>, "Trait must be mmad_trait or MmadTrait.");
        return mmad_trait{trait.fmOffset, trait.kDirectionAlign, trait.cmatrixSource, trait.disableGemv,
                          normalize_mmad_type(trait.mmadType)};
    }
}

template <typename PadType>
__aicore__ inline constexpr auto normalize_img2col_params(const Img2ColParams<PadType>& params)
{
    asc::te::img2col_params<PadType> normalized_params;
    normalized_params.m_extension = params.mExtension;
    normalized_params.k_extension = params.kExtension;
    normalized_params.m_start_pos = params.mStartPt;
    normalized_params.k_start_pos = params.kStartPt;
    for (uint32_t i = 0; i < 4; ++i) {
        normalized_params.pad_list[i] = params.padList[i];
    }
    normalized_params.stride_w = params.strideW;
    normalized_params.stride_h = params.strideH;
    normalized_params.filter_w = params.filterW;
    normalized_params.filter_h = params.filterH;
    normalized_params.dilation_filter_w = params.dilationFilterW;
    normalized_params.dilation_filter_h = params.dilationFilterH;
    normalized_params.enable_filter_w_extend = params.filterSizeW;
    normalized_params.enable_filter_h_extend = params.filterSizeH;
    normalized_params.enable_transpose = params.transpose;
    normalized_params.enable_f_matrix_ctrl = params.fMatrixCtrl;
    normalized_params.pad_value = params.padValue;
    return normalized_params;
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_LEGACY_LEGACY_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_LEGACY_UTILS_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_LEGACY_UTILS_H
#endif
