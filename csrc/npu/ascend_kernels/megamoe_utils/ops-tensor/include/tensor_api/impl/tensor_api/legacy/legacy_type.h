/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_TENSOR_API_LEGACY_LEGACY_TYPE_H
#define IMPL_TENSOR_API_LEGACY_LEGACY_TYPE_H

#include "impl/tensor_api/legacy/legacy_utils.h"

namespace AscendC {
namespace Te {

namespace location {
using INVALID = asc::te::location::invalid;
using GM = asc::te::location::gm;
using UB = asc::te::location::ub;
using L1 = asc::te::location::l1;
using L0A = asc::te::location::l0a;
using L0B = asc::te::location::l0b;
using L0ScaleA = asc::te::location::l0scalea;
using L0ScaleB = asc::te::location::l0scaleb;
using L0C = asc::te::location::l0c;
using BIAS = asc::te::location::bias;
using FIXBUF = asc::te::location::fixbuf;
using SSBUF = asc::te::location::ssbuf;
} // namespace location
namespace Location = location;

template <typename... Shapes>
using Shape = asc::te::shape<Shapes...>;

template <typename... Strides>
using Stride = asc::te::stride<Strides...>;

template <typename... Coords>
using Coord = asc::te::coord<Coords...>;

template <typename Shape, typename Stride, typename Info = Std::ignore_t>
using Layout = asc::te::layout<Shape, Stride, Info>;

template <typename DataType = uint16_t, typename C0 = Std::Int<c0_element<DataType>>>
using LayoutTraitDefault = asc::te::layout_trait_default<DataType, C0>;

template <typename Layout>
using GetLayoutPattern = asc::te::get_layout_pattern<Layout>;

using ZNLayoutPtn = asc::te::zn_layout_ptn;
using ZZLayoutPtn = asc::te::zz_layout_ptn;
using NNLayoutPtn = asc::te::nn_layout_ptn;
using NZLayoutPtn = asc::te::nz_layout_ptn;
using NDLayoutPtn = asc::te::nd_layout_ptn;
using DNLayoutPtn = asc::te::dn_layout_ptn;
using NDExtLayoutPtn = asc::te::nd_ext_layout_ptn;
using DNExtLayoutPtn = asc::te::dn_ext_layout_ptn;
using ScaleANDLayoutPtn = asc::te::scalea_nd_layout_ptn;
using ScaleADNLayoutPtn = asc::te::scalea_dn_layout_ptn;
using ScaleBNDLayoutPtn = asc::te::scaleb_nd_layout_ptn;
using ScaleBDNLayoutPtn = asc::te::scaleb_dn_layout_ptn;
using NCHWLayoutPtn = asc::te::nchw_layout_ptn;
using NHWCLayoutPtn = asc::te::nhwc_layout_ptn;
using NC1HWC0LayoutPtn = asc::te::nc1hwc0_layout_ptn;
using NCDHWLayoutPtn = asc::te::ncdhw_layout_ptn;
using NDC1HWC0LayoutPtn = asc::te::ndc1hwc0_layout_ptn;

template <typename Iterator>
using ViewEngine = asc::te::view_engine<Iterator>;

template <typename DefaultTrait>
struct legacy_trait_default {
    using TraitType = typename DefaultTrait::trait_type;
    static constexpr const TraitType value = DefaultTrait::value;
};

using CopyGM2L1 = asc::te::copy_gm_to_l1;
using CopyL12UB = asc::te::copy_l1_to_ub;
using CopyL12L0B = asc::te::copy_l1_to_l0b;
using CopyL12L0A = asc::te::copy_l1_to_l0a;
using CopyL12L0ScaleA = asc::te::copy_l1_to_l0scalea;
using CopyL12L0ScaleB = asc::te::copy_l1_to_l0scaleb;
using CopyL12FB = asc::te::copy_l1_to_fixbuf;
using CopyL12BT = asc::te::copy_l1_to_biastable;
using CopyL0C2GM = asc::te::copy_l0c_to_gm;
using CopyL0C2UB = asc::te::copy_l0c_to_ub;
using CopyL0C2L1 = asc::te::copy_l0c_to_l1;
using CopyGM2L1Trait = asc::te::gm_to_l1_trait;
using CopyGM2L1TraitDefault = legacy_trait_default<asc::te::gm_to_l1_trait_default>;
using CopyL12UBTrait = asc::te::l1_to_ub_trait;
using CopyL12UBTraitDefault = legacy_trait_default<asc::te::l1_to_ub_trait_default>;
using CopyL12L0ATrait = asc::te::l1_to_l0a_trait;
using CopyL12L0ATraitDefault = legacy_trait_default<asc::te::l1_to_l0a_trait_default>;
using CopyL12L0BTrait = asc::te::l1_to_l0b_trait;
using CopyL12L0BTraitDefault = legacy_trait_default<asc::te::l1_to_l0b_trait_default>;
using CopyL12L0ScaleATrait = asc::te::l1_to_l0scalea_trait;
using CopyL12L0ScaleATraitDefault = legacy_trait_default<asc::te::l1_to_l0scalea_trait_default>;
using CopyL12L0ScaleBTrait = asc::te::l1_to_l0scaleb_trait;
using CopyL12L0ScaleBTraitDefault = legacy_trait_default<asc::te::l1_to_l0scaleb_trait_default>;
using CopyL12FBTrait = asc::te::l1_to_fixbuf_trait;
using CopyL12FBTraitDefault = legacy_trait_default<asc::te::l1_to_fixbuf_trait_default>;
using CopyL12BTTrait = asc::te::l1_to_biastable_trait;
using CopyL12BTTraitDefault = legacy_trait_default<asc::te::l1_to_biastable_trait_default>;
using CopyL0C2GMTraitDefault = legacy_trait_default<asc::te::l0c_to_gm_trait_default>;
using CopyL0C2UBTraitDefault = legacy_trait_default<asc::te::l0c_to_ub_trait_default>;
using CopyL0C2L1TraitDefault = legacy_trait_default<asc::te::l0c_to_l1_trait_default>;

using CopyGM2UB = asc::te::copy_gm_to_ub;
using CopyUB2L1 = asc::te::copy_ub_to_l1;
using CopyUB2GM = asc::te::copy_ub_to_gm;
using CopyUB2UB = asc::te::copy_ub_to_ub;
using CopyGM2UBTrait = asc::te::gm_to_ub_trait;
using CopyGM2UBTraitDefault = legacy_trait_default<asc::te::gm_to_ub_trait_default>;
using CopyUB2L1Trait = asc::te::ub_to_l1_trait;
using CopyUB2L1TraitDefault = legacy_trait_default<asc::te::ub_to_l1_trait_default>;
using CopyUB2GMTrait = asc::te::ub_to_gm_trait;
using CopyUB2GMTraitDefault = legacy_trait_default<asc::te::ub_to_gm_trait_default>;
using CopyUB2UBTrait = asc::te::ub_to_ub_trait;
using CopyUB2UBTraitDefault = legacy_trait_default<asc::te::ub_to_ub_trait_default>;

using MmadTraitDefault = legacy_trait_default<asc::te::mmad_trait_default>;
using MmadOperation = asc::te::mmad_operation;
using MmadOpWith = asc::te::mmad_op_with;

// 兼容旧常量名（已改为 snake_case，此处保留 PascalCase 别名）
template <typename DataType = Std::ignore_t>
constexpr size_t C0_ELEMENT = asc::te::c0_element<DataType>;
template <typename DataType = Std::ignore_t>
constexpr size_t C0_SIZE = asc::te::c0_size<DataType>;
constexpr uint32_t CURRENT_ARCH_VERSION = asc::te::current_arch_version;
constexpr size_t FRACTAL_FIXED = asc::te::fractal_fixed;
constexpr size_t TWO_DIM_DATA = asc::te::two_dim_data;
constexpr size_t THREE_DIM_DATA = asc::te::three_dim_data;
constexpr size_t FOUR_DIM_DATA = asc::te::four_dim_data;
constexpr size_t FIVE_DIM_DATA = asc::te::five_dim_data;
constexpr size_t MX_SCALE_K0 = asc::te::mx_scale_k0;
constexpr uint32_t BLOCK_CUBE = asc::te::block_cube;
constexpr uint64_t HIFLOAT8_MMAD_CTRL_MASK = asc::te::hifloat8_mmad_ctrl_mask;

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_LEGACY_LEGACY_TYPE_H
