/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_TENSOR_API_LEGACY_LEGACY_ATOM_H
#define IMPL_TENSOR_API_LEGACY_LEGACY_ATOM_H

#include "impl/tensor_api/legacy/legacy_type.h"

namespace AscendC {
namespace Te {

using namespace asc::te;

template <typename Params>
__aicore__ inline constexpr mmad_params normalize_mmad_params(const Params& params)
{
    using params_type = Std::remove_cvref_t<Params>;
    if constexpr (Std::is_same_v<params_type, mmad_params>) {
        return params;
    } else {
        static_assert(Std::is_same_v<params_type, MmadParams>, "Params must be mmad_params or MmadParams.");
        return mmad_params{params.m, params.n, params.k, static_cast<unit_flag_mode>(params.unitFlag),
                           params.cmatrixInitVal};
    }
}

template <typename Params>
__aicore__ inline constexpr l0c_to_gm_params normalize_l0c_to_gm_params(const Params& params)
{
    using params_type = Std::remove_cvref_t<Params>;
    if constexpr (Std::is_same_v<params_type, l0c_to_gm_params>) {
        return params;
    } else {
        static_assert(Std::is_same_v<params_type, FixpipeParams>,
                      "Params must be l0c_to_gm_params or FixpipeParams.");
        return l0c_to_gm_params{static_cast<unit_flag_mode>(params.unitFlag)};
    }
}

template <typename Params>
__aicore__ inline constexpr l0c_to_ub_params normalize_l0c_to_ub_params(const Params& params)
{
    using params_type = Std::remove_cvref_t<Params>;
    if constexpr (Std::is_same_v<params_type, l0c_to_ub_params>) {
        return params;
    } else {
        static_assert(Std::is_same_v<params_type, FixpipeParams>,
                      "Params must be l0c_to_ub_params or FixpipeParams.");
        return l0c_to_ub_params{static_cast<unit_flag_mode>(params.unitFlag),
                               static_cast<uint8_t>(params.subBlockId)};
    }
}

template <typename Params>
__aicore__ inline constexpr l0c_to_l1_params normalize_l0c_to_l1_params(const Params& params)
{
    using params_type = Std::remove_cvref_t<Params>;
    if constexpr (Std::is_same_v<params_type, l0c_to_l1_params>) {
        return params;
    } else {
        static_assert(Std::is_same_v<params_type, FixpipeParams>,
                      "Params must be l0c_to_l1_params or FixpipeParams.");
        return l0c_to_l1_params{static_cast<unit_flag_mode>(params.unitFlag)};
    }
}

template <typename Params>
__aicore__ inline constexpr gm_to_ub_params normalize_gm_to_ub_params(const Params& params)
{
    using params_type = Std::remove_cvref_t<Params>;
    if constexpr (Std::is_same_v<params_type, gm_to_ub_params>) {
        return params;
    } else {
        static_assert(Std::is_same_v<params_type, CopyGM2UBParams>,
                      "Params must be gm_to_ub_params or CopyGM2UBParams.");
        return gm_to_ub_params{params.leftPaddingCount, params.rightPaddingCount, params.enableConstantPad};
    }
}

template <typename CopyTraitsType>
struct copy_traits_operation;

template <typename CopyOperation, typename... CopyOpArgs>
struct copy_traits_operation<asc::te::copy_traits<CopyOperation, CopyOpArgs...>> {
    using type = CopyOperation;
};

template <typename CopyTraitsType>
using copy_traits_operation_t = typename copy_traits_operation<CopyTraitsType>::type;

template <typename CopyOperation, typename Params>
__aicore__ inline constexpr decltype(auto) normalize_copy_params(const Params& params)
{
    using params_type = Std::remove_cvref_t<Params>;
    if constexpr (Std::is_same_v<CopyOperation, copy_l0c_to_gm>) {
        return normalize_l0c_to_gm_params(params);
    } else if constexpr (Std::is_same_v<CopyOperation, copy_l0c_to_ub>) {
        return normalize_l0c_to_ub_params(params);
    } else if constexpr (Std::is_same_v<CopyOperation, copy_l0c_to_l1>) {
        return normalize_l0c_to_l1_params(params);
    } else if constexpr (Std::is_same_v<params_type, CopyGM2UBParams>) {
        return normalize_gm_to_ub_params(params);
    } else {
        return params;
    }
}

template <typename CopyOperation, typename PadType>
__aicore__ inline constexpr decltype(auto) normalize_copy_params(const Img2ColParams<PadType>& params)
{
    if constexpr (Std::is_same_v<CopyOperation, copy_l1_to_l0a>) {
        return normalize_img2col_params(params);
    } else {
        return params;
    }
}

// MmadTraits: 独立的 PascalCase 模板，不依赖 mmad_traits
template <typename MmadOperationType, typename... MmadOpArgs>
struct MmadTraits {};

template <typename MmadOp, typename MmadTraitsType, typename MmadOpWith, typename MmadTraitsWith>
struct MmadTraits<MmadOp, MmadTraitsType, MmadOpWith, MmadTraitsWith> {
    using TraitType = get_trait_member_type_t<MmadTraitsType>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = MmadTraitsType::value;

    mmad_params params;

    __aicore__ inline constexpr MmadTraits() = default;
    __aicore__ inline constexpr MmadTraits(mmad_params p) : params(p) {}

    __aicore__ inline constexpr MmadTraits<MmadOpWith, MmadTraitsWith> with() const
    {
        return {};
    }

    template <typename Params>
    __aicore__ inline constexpr MmadTraits<MmadOpWith, MmadTraitsWith> with(const Params& p) const
    {
        return {AscendC::Te::normalize_mmad_params(p)};
    }

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args&... args) const
    {
        using normalized_t = Std::remove_cvref_t<decltype(AscendC::Te::normalize_mmad_trait(TraitType{}))>;
        if constexpr (Std::is_same_v<TraitType, normalized_t>) {
            MmadOp::template mmad<TraitType, trait, Args...>(args..., params);
        } else {
            constexpr static normalized_t normalized_trait = AscendC::Te::normalize_mmad_trait(trait);
            MmadOp::template mmad<normalized_t, normalized_trait, Args...>(args..., params);
        }
    }
};

// MmadTraits 2参数特化: 继承4参数
template <typename MmadOp, typename MmadTraitsType>
struct MmadTraits<MmadOp, MmadTraitsType> : public MmadTraits<MmadOp, MmadTraitsType, mmad_op_with, MmadTraitsType> {};

// MmadTraits 1参数特化: 标准操作绑定默认 trait
template <> struct MmadTraits<mmad_operation> : public MmadTraits<mmad_operation, mmad_trait_default> {};

// MmadAtom: 独立的 PascalCase 模板，不依赖 mmad_atom
template <typename... Args>
struct MmadAtom;

template <typename MmadOperationType>
struct MmadAtom<MmadOperationType> : public MmadTraits<MmadOperationType> {
    using mmad_trait_type = MmadTraits<MmadOperationType>;
    using TraitType = get_trait_member_type_t<mmad_trait_type>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = mmad_trait_type::defaultTrait;

    template <const TraitType& trait = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params&... params) const
    {
        mmad_trait_type::template MmadUnpack<trait>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const
    {
        auto traits = mmad_trait_type::with(static_cast<TraitsArgs&&>(args)...);
        return MmadAtom<decltype(traits)>{traits};
    }
};

template <typename... Args>
struct MmadAtom<MmadTraits<Args...>> : public MmadTraits<Args...> {
    using mmad_trait_type = MmadTraits<Args...>;
    using TraitType = get_trait_member_type_t<mmad_trait_type>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = mmad_trait_type::defaultTrait;

    template <const TraitType& trait = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params&... params) const
    {
        mmad_trait_type::template MmadUnpack<trait>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const
    {
        auto traits = mmad_trait_type::with(static_cast<TraitsArgs&&>(args)...);
        return MmadAtom<decltype(traits)>{traits};
    }
};

// normalize_copy_trait: PascalCase trait → snake_case trait 转换
template <typename Trait>
__aicore__ inline constexpr Trait normalize_copy_trait(const Trait& trait)
{
    return trait;
}

__aicore__ inline constexpr l0c_to_gm_trait normalize_copy_trait(const CopyL0C2GMTrait& trait)
{
    return {AscendC::Te::normalize_round_mode(trait.roundMode), trait.enableRelu, trait.enableChannelSplit};
}

__aicore__ inline constexpr l0c_to_ub_trait normalize_copy_trait(const CopyL0C2UBTrait& trait)
{
    return {AscendC::Te::normalize_round_mode(trait.roundMode), trait.enableRelu, trait.enableChannelSplit,
            AscendC::Te::normalize_dual_dst_mode(trait.dualDstCtl)};
}

__aicore__ inline constexpr l0c_to_l1_trait normalize_copy_trait(const CopyL0C2L1Trait& trait)
{
    return {AscendC::Te::normalize_round_mode(trait.roundMode), trait.enableRelu, trait.enableChannelSplit};
}

// 检测 CopyOp 是否有 snake_case copy 方法
template <typename AlwaysVoid, typename Op, typename Trait, const Trait& T, typename... Args>
struct has_copy_method : Std::false_type {};

template <typename Op, typename Trait, const Trait& T, typename... Args>
struct has_copy_method<void_t<decltype(Op::template copy<Trait, T, Args...>(std::declval<Args>()...))>,
                       Op, Trait, T, Args...> : Std::true_type {};

// CopyTraits: 独立的 PascalCase 模板，不依赖 copy_traits
template <typename CopyOperation, typename... CopyOpArgs>
struct CopyTraits {};

template <typename CopyOp, typename Traits, typename CopyOpWith, typename TraitsWith>
struct CopyTraits<CopyOp, Traits, CopyOpWith, TraitsWith> {
    using TraitType = get_trait_member_type_t<Traits>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = Traits::value;

    __aicore__ inline constexpr CopyTraits() = default;

    __aicore__ inline constexpr CopyTraits<CopyOpWith, TraitsWith> with() const
    {
        return {};
    }

    template <typename Params>
    __aicore__ inline constexpr CopyTraits<CopyOpWith, TraitsWith> with(const Params& p) const
    {
        using params_type = Std::remove_cvref_t<Params>;
        if constexpr (Std::is_same_v<CopyOp, copy_l0c_to_gm>) {
            return {AscendC::Te::normalize_l0c_to_gm_params(p)};
        } else if constexpr (Std::is_same_v<CopyOp, copy_l0c_to_ub>) {
            return {AscendC::Te::normalize_l0c_to_ub_params(p)};
        } else if constexpr (Std::is_same_v<CopyOp, copy_l0c_to_l1>) {
            return {AscendC::Te::normalize_l0c_to_l1_params(p)};
        } else if constexpr (Std::is_same_v<params_type, asc::te::gm_to_ub_params>
                             || Std::is_same_v<params_type, AscendC::Te::CopyGM2UBParams>) {
            return {AscendC::Te::normalize_gm_to_ub_params(p)};
        } else {
            return {p};
        }
    }

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void CopyUnpack(const Args&... args) const
    {
        using normalized_t = Std::remove_cvref_t<decltype(normalize_copy_trait(Traits::value))>;
        if constexpr (Std::is_same_v<TraitType, normalized_t>) {
            if constexpr (has_copy_method<void, CopyOp, normalized_t, trait, Args...>::value) {
                CopyOp::template copy<normalized_t, trait, Args...>(args...);
            } else {
                CopyOp::template Copy<normalized_t, trait, Args...>(args...);
            }
        } else {
            static constexpr normalized_t normalized_trait_val = normalize_copy_trait(trait);
            if constexpr (has_copy_method<void, CopyOp, normalized_t, normalized_trait_val, Args...>::value) {
                CopyOp::template copy<normalized_t, normalized_trait_val, Args...>(args...);
            } else {
                CopyOp::template Copy<normalized_t, normalized_trait_val, Args...>(args...);
            }
        }
    }
};

template <typename CopyOp, typename Traits, typename Params>
struct CopyTraitsWithParams {
    using TraitType = get_trait_member_type_t<Traits>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = Traits::value;

    Params params;

    __aicore__ inline constexpr CopyTraitsWithParams() = default;
    __aicore__ inline constexpr CopyTraitsWithParams(Params params) : params(params) {}

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void CopyUnpack(const Args&... args) const
    {
        using normalized_t = Std::remove_cvref_t<decltype(normalize_copy_trait(Traits::value))>;
        if constexpr (Std::is_same_v<TraitType, normalized_t>) {
            CopyOp::template copy<normalized_t, trait, Args...>(args..., params);
        } else {
            static constexpr normalized_t normalized_trait = normalize_copy_trait(trait);
            CopyOp::template copy<normalized_t, normalized_trait, Args...>(args..., params);
        }
    }
};

template <typename Traits>
struct CopyTraits<copy_l0c_to_gm_with, Traits>
    : public CopyTraitsWithParams<copy_l0c_to_gm_with, Traits, l0c_to_gm_params> {
    using CopyTraitsWithParams<copy_l0c_to_gm_with, Traits, l0c_to_gm_params>::CopyTraitsWithParams;
};

template <typename Traits>
struct CopyTraits<copy_l0c_to_ub_with, Traits>
    : public CopyTraitsWithParams<copy_l0c_to_ub_with, Traits, l0c_to_ub_params> {
    using CopyTraitsWithParams<copy_l0c_to_ub_with, Traits, l0c_to_ub_params>::CopyTraitsWithParams;
};

template <typename Traits>
struct CopyTraits<copy_l0c_to_l1_with, Traits>
    : public CopyTraitsWithParams<copy_l0c_to_l1_with, Traits, l0c_to_l1_params> {
    using CopyTraitsWithParams<copy_l0c_to_l1_with, Traits, l0c_to_l1_params>::CopyTraitsWithParams;
};

template <typename Traits, typename PadType>
struct CopyTraits<copy_l1_to_l0a_with, Traits, PadType>
    : public CopyTraitsWithParams<copy_l1_to_l0a_with, Traits, img2col_params<PadType>> {
    using CopyTraitsWithParams<copy_l1_to_l0a_with, Traits,
                               img2col_params<PadType>>::CopyTraitsWithParams;
};

// CopyTraits 2参数特化: 继承4参数
template <typename CopyOp, typename Traits>
struct CopyTraits<CopyOp, Traits> : public CopyTraits<CopyOp, Traits, CopyOp, Traits> {};

template <typename Traits>
struct CopyTraits<copy_l0c_to_gm, Traits>
    : public CopyTraits<copy_l0c_to_gm, Traits, copy_l0c_to_gm_with, Traits> {};

template <typename Traits>
struct CopyTraits<copy_l0c_to_ub, Traits>
    : public CopyTraits<copy_l0c_to_ub, Traits, copy_l0c_to_ub_with, Traits> {};

template <typename Traits>
struct CopyTraits<copy_l0c_to_l1, Traits>
    : public CopyTraits<copy_l0c_to_l1, Traits, copy_l0c_to_l1_with, Traits> {};

template <typename Traits>
struct CopyTraits<copy_l1_to_l0a, Traits>
    : public CopyTraits<copy_l1_to_l0a, Traits, copy_l1_to_l0a_with, Traits> {
    using base_type = CopyTraits<copy_l1_to_l0a, Traits, copy_l1_to_l0a_with, Traits>;
    using base_type::with;

    template <typename PadType>
    __aicore__ inline constexpr auto with(const Img2ColParams<PadType>& params) const
    {
        return CopyTraits<copy_l1_to_l0a_with, Traits, PadType>{normalize_img2col_params(params)};
    }

    template <typename PadType>
    __aicore__ inline constexpr auto with(const img2col_params<PadType>& params) const
    {
        return CopyTraits<copy_l1_to_l0a_with, Traits, PadType>{params};
    }
};

// CopyTraits 1参数特化: 标准操作绑定默认 trait
template <> struct CopyTraits<copy_gm_to_l1> : public CopyTraits<copy_gm_to_l1, gm_to_l1_trait_default> {};
template <> struct CopyTraits<copy_gm_to_ub> : public CopyTraits<copy_gm_to_ub, gm_to_ub_trait_default> {};
template <> struct CopyTraits<copy_ub_to_gm> : public CopyTraits<copy_ub_to_gm, ub_to_gm_trait_default> {};
template <> struct CopyTraits<copy_ub_to_l1> : public CopyTraits<copy_ub_to_l1, ub_to_l1_trait_default> {};
template <> struct CopyTraits<copy_ub_to_ub> : public CopyTraits<copy_ub_to_ub, ub_to_ub_trait_default> {};
template <> struct CopyTraits<copy_l1_to_ub> : public CopyTraits<copy_l1_to_ub, l1_to_ub_trait_default> {};
template <> struct CopyTraits<copy_l0c_to_gm> : public CopyTraits<copy_l0c_to_gm, l0c_to_gm_trait_default> {};
template <> struct CopyTraits<copy_l0c_to_ub> : public CopyTraits<copy_l0c_to_ub, l0c_to_ub_trait_default> {};
template <> struct CopyTraits<copy_l0c_to_l1> : public CopyTraits<copy_l0c_to_l1, l0c_to_l1_trait_default> {};
template <> struct CopyTraits<copy_l1_to_l0a> : public CopyTraits<copy_l1_to_l0a, l1_to_l0a_trait_default> {};
template <> struct CopyTraits<copy_l1_to_l0b> : public CopyTraits<copy_l1_to_l0b, l1_to_l0b_trait_default> {};
template <> struct CopyTraits<copy_l1_to_l0scalea> : public CopyTraits<copy_l1_to_l0scalea, l1_to_l0scalea_trait_default> {};
template <> struct CopyTraits<copy_l1_to_l0scaleb> : public CopyTraits<copy_l1_to_l0scaleb, l1_to_l0scaleb_trait_default> {};
template <> struct CopyTraits<copy_l1_to_fixbuf> : public CopyTraits<copy_l1_to_fixbuf, l1_to_fixbuf_trait_default> {};
template <> struct CopyTraits<copy_l1_to_biastable> : public CopyTraits<copy_l1_to_biastable, l1_to_biastable_trait_default> {};

template <typename AlwaysVoid, typename... Args>
struct has_copy_traits_impl : Std::false_type {};

template <typename CopyOperation, typename... CopyOpArgs>
struct has_copy_traits_impl<
    void_t<typename copy_traits<CopyOperation>::trait_type, typename copy_traits<CopyOperation, CopyOpArgs...>::trait_type,
           decltype(copy_traits<CopyOperation, CopyOpArgs...>::default_trait)>,
    CopyOperation, CopyOpArgs...>
    : Std::bool_constant<Std::is_same_v<typename copy_traits<CopyOperation>::trait_type,
                                        typename copy_traits<CopyOperation, CopyOpArgs...>::trait_type>> {};

template <typename... Args>
struct has_copy_traits : has_copy_traits_impl<void, Args...> {};

// CopyAtom: 独立的 PascalCase 模板，不依赖 copy_atom
template <typename... Args>
struct CopyAtom;

template <typename CopyOperation>
struct CopyAtom<CopyOperation> : public CopyTraits<CopyOperation> {
    using copy_trait_type = CopyTraits<CopyOperation>;
    using TraitType = get_trait_member_type_t<copy_trait_type>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = copy_trait_type::defaultTrait;

    template <const TraitType& trait = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params&... params) const
    {
        using traits_type = copy_trait_type;
        static_cast<const traits_type&>(*this).template CopyUnpack<trait, Params...>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const
    {
        auto traits = copy_trait_type::with(static_cast<TraitsArgs&&>(args)...);
        return CopyAtom<decltype(traits)>{traits};
    }
};

template <typename... Args>
struct CopyAtom<CopyTraits<Args...>> : public CopyTraits<Args...> {
    using copy_trait_type = CopyTraits<Args...>;
    using TraitType = get_trait_member_type_t<copy_trait_type>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = copy_trait_type::defaultTrait;

    template <const TraitType& trait = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params&... params) const
    {
        using traits_type = copy_trait_type;
        static_cast<const traits_type&>(*this).template CopyUnpack<trait, Params...>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const
    {
        auto traits = copy_trait_type::with(static_cast<TraitsArgs&&>(args)...);
        return CopyAtom<decltype(traits)>{traits};
    }
};

template <typename CopyOperation>
struct CopyAtom<copy_traits<CopyOperation>> : public copy_traits<CopyOperation> {
    using copy_trait_type = copy_traits<CopyOperation>;
    using copy_operation_type = CopyOperation;
    using TraitType = get_trait_member_type_t<copy_trait_type>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = copy_trait_type::default_trait;

    template <const TraitType& trait = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params&... params) const
    {
        static_cast<const copy_trait_type&>(*this).template copy_unpack<trait, Params...>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const
    {
        auto traits = copy_trait_type::with(
            normalize_copy_params<copy_operation_type>(static_cast<TraitsArgs&&>(args))...);
        return CopyAtom<decltype(traits)>{traits};
    }
};

template <typename... Args>
struct CopyAtom<copy_traits<Args...>> : public copy_traits<Args...> {
    using copy_trait_type = copy_traits<Args...>;
    using copy_operation_type = copy_traits_operation_t<copy_trait_type>;
    using TraitType = get_trait_member_type_t<copy_trait_type>;
    using trait_type = TraitType;
    static constexpr const TraitType defaultTrait = copy_trait_type::default_trait;

    template <const TraitType& trait = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params&... params) const
    {
        static_cast<const copy_trait_type&>(*this).template copy_unpack<trait, Params...>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const
    {
        auto traits = copy_trait_type::with(
            normalize_copy_params<copy_operation_type>(static_cast<TraitsArgs&&>(args))...);
        return CopyAtom<decltype(traits)>{traits};
    }
};

// Copy / MakeCopy: PascalCase 接口
template <typename AtomType, typename DstTensor, typename SrcTensor>
__aicore__ inline void Copy(const AtomType& atom, const DstTensor& dst, const SrcTensor& src)
{
    atom.Call(dst, src);
}

template <typename AtomType, typename DstTensor, typename SrcTensor, typename Quant,
          Std::enable_if_t<is_valid_quant_v<Quant>, int> = 0>
__aicore__ inline void Copy(const AtomType& atom, const DstTensor& dst, const SrcTensor& src,
                            const Quant& quant)
{
    atom.Call(dst, src, quant);
}

template <typename CopyOperationType>
__aicore__ inline constexpr auto MakeCopy(const CopyOperationType& copy_operation)
{
    if constexpr (has_copy_traits<CopyOperationType>::value) {
        return CopyAtom<copy_traits<CopyOperationType>>{};
    } else {
        return CopyAtom<CopyOperationType>{};
    }
}

template <typename CopyOperationType, typename CopyTraitType>
__aicore__ inline constexpr auto MakeCopy(const CopyOperationType& copy_operation, const CopyTraitType& copy_trait)
{
    if constexpr (has_copy_traits<CopyOperationType, CopyTraitType>::value) {
        return CopyAtom<copy_traits<CopyOperationType, CopyTraitType>>{};
    } else {
        return CopyAtom<CopyTraits<CopyOperationType, CopyTraitType>>{};
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_LEGACY_LEGACY_ATOM_H
