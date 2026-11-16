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
    "impl/tensor_api/arch/cube/mmad/mmad_impl/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_INSTRUCTION_H

#include "impl/tensor_api/arch/utils/arch_utils.h"
#include "impl/tensor_api/tensor/pointer_pattern.h"
#include "impl/tensor_api/tensor/tensor_impl.h"

namespace asc {
namespace te {

__aicore__ inline static void set_ctrl_for_hifloat8()
{
    uint64_t ori_config = asc_get_ctrl();
    uint64_t config = ori_config | hifloat8_mmad_ctrl_mask;
    asc_set_ctrl(config);
}

__aicore__ inline static void reset_ctrl_for_hifloat8()
{
    uint64_t ori_config = asc_get_ctrl();
    uint64_t config = ori_config & ~hifloat8_mmad_ctrl_mask;
    asc_set_ctrl(config);
}

class mmad_instr {
public:
    template <typename CTensor, typename ATensor, typename BTensor, typename... Params>
    __aicore__ inline static void mmad(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                       const Params&... params)
    {
        mmad_impl(dst.data().get(), fm.data().get(), filter.data().get(), params...);
    }

private:
    template <typename CType, typename AType, typename BType>
    __aicore__ inline static void mmad_impl(__cc__ CType* dst, __ca__ AType* fm, __cb__ BType* filter, uint16_t m,
                                            uint16_t k, uint16_t n, uint8_t unit_flag, bool disable_gemv,
                                            bool init_with_btbuf, bool init_with_zero)
    {
        if constexpr (Std::is_same_v<AType, hifloat8_t> && Std::is_same_v<BType, hifloat8_t>) {
            set_ctrl_for_hifloat8();
            asc_mmad(dst, reinterpret_cast<__ca__ fp8_e4m3fn_t*>(fm), reinterpret_cast<__cb__ fp8_e4m3fn_t*>(filter), m,
                     k, n, unit_flag, disable_gemv, init_with_btbuf, init_with_zero);
            reset_ctrl_for_hifloat8();
        } else {
            asc_mmad(dst, fm, filter, m, k, n, unit_flag, disable_gemv, init_with_btbuf, init_with_zero);
        }
    }
};

class mmad_bias_instr {
public:
    template <typename CTensor, typename ATensor, typename BTensor, typename BiasTensor, typename... Params>
    __aicore__ inline static void mmad(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                       const BiasTensor& bias, const Params&... params)
    {
        mmad_impl(dst.data().get(), fm.data().get(), filter.data().get(), reinterpret_cast<uint64_t>(bias.data().get()),
                  params...);
    }

private:
    template <typename CType, typename AType, typename BType>
    __aicore__ inline static void mmad_impl(__cc__ CType* dst, __ca__ AType* fm, __cb__ BType* filter, uint64_t bias,
                                            uint16_t m, uint16_t k, uint16_t n, int8_t unit_flag, bool disable_gemv,
                                            bool init_with_btbuf, bool init_with_zero)
    {
        uint64_t xd = reinterpret_cast<uint64_t>(dst) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
        if constexpr (Std::is_same_v<AType, hifloat8_t> && Std::is_same_v<BType, hifloat8_t>) {
            set_ctrl_for_hifloat8();
            asc_mmad(reinterpret_cast<__cc__ CType*>(xd), reinterpret_cast<__ca__ fp8_e4m3fn_t*>(fm),
                     reinterpret_cast<__cb__ fp8_e4m3fn_t*>(filter), m, k, n, unit_flag, disable_gemv, init_with_btbuf,
                     init_with_zero);
            reset_ctrl_for_hifloat8();
        } else {
            asc_mmad(reinterpret_cast<__cc__ CType*>(xd), fm, filter, m, k, n, unit_flag, disable_gemv,
                     init_with_btbuf, init_with_zero);
        }
    }
};

class mmad_mx_instr {
public:
    template <typename CTensor, typename ATensor, typename BTensor, typename... Params>
    __aicore__ inline static void mmad(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                       const Params&... params)
    {
        mmad_impl(dst.data().get(), fm.data().get(), filter.data().get(), params...);
    }

private:
    template <typename CType, typename AType, typename BType>
    __aicore__ inline static void mmad_impl(__cc__ CType* dst, __ca__ AType* fm, __cb__ BType* filter, uint16_t m,
                                            uint16_t k, uint16_t n, uint8_t unit_flag, bool disable_gemv,
                                            bool init_with_btbuf, bool init_with_zero)
    {
        asc_mmad_mx(dst, fm, filter, m, k, n, unit_flag, disable_gemv, init_with_btbuf, init_with_zero);
    }
};

class mmad_mx_bias_instr {
public:
    template <typename CTensor, typename ATensor, typename BTensor, typename BiasTensor, typename... Params>
    __aicore__ inline static void mmad(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                       const BiasTensor& bias, const Params&... params)
    {
        mmad_impl(dst.data().get(), fm.data().get(), filter.data().get(), reinterpret_cast<uint64_t>(bias.data().get()),
                  params...);
    }

private:
    template <typename CType, typename AType, typename BType>
    __aicore__ inline static void mmad_impl(__cc__ CType* dst, __ca__ AType* fm, __cb__ BType* filter, uint64_t bias,
                                            uint16_t m, uint16_t k, uint16_t n, int8_t unit_flag, bool disable_gemv,
                                            bool init_with_btbuf, bool init_with_zero)
    {
        uint64_t xd = reinterpret_cast<uint64_t>(dst) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
        asc_mmad_mx(reinterpret_cast<__cc__ CType*>(xd), fm, filter, m, k, n, unit_flag, disable_gemv,
                    init_with_btbuf, init_with_zero);
    }
};

} // namespace te
} // namespace asc

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
