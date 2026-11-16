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
    "impl/tensor_api/arch/cube/mmad/mmad_impl/mmad_with_bias.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file mmad_with_bias.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_MMAD_WITH_BIAS_H
#define IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_MMAD_WITH_BIAS_H

#include "impl/tensor_api/arch/cube/mmad/mmad_impl/instruction.h"

namespace asc {
namespace te {

class mmad_with_bias {
public:
    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename BiasTensor,
              typename Params>
    __aicore__ inline static void run(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                      const BiasTensor& bias, const Params& params)
    {
        mmad_impl<trait, CTensor, ATensor, BTensor, BiasTensor>(dst, fm, filter, bias, params);
    }

private:
    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename BiasTensor>
    __aicore__ inline static constexpr void check_template_for_normal()
    {
        check_layout_pattern<CTensor, ATensor, BTensor, BiasTensor>();
        check_data_type::check_mmad_bias_data_type<CTensor, ATensor, BTensor, BiasTensor>();
    }

    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename BiasTensor>
    __aicore__ inline static constexpr void check_template_for_mx()
    {
        check_layout_pattern<CTensor, ATensor, BTensor, BiasTensor>();
        check_data_type::check_mx_mmad_bias_data_type<CTensor, ATensor, BTensor, BiasTensor>();
    }

    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename BiasTensor,
              typename Params>
    __aicore__ inline static void mmad_impl(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                            const BiasTensor& bias, const Params& params)
    {
        if constexpr (trait.mmad_type == mmad_type::normal) {
            check_template_for_normal<trait, CTensor, ATensor, BTensor, BiasTensor>();
        } else if constexpr (trait.mmad_type == mmad_type::mx) {
            check_template_for_mx<trait, CTensor, ATensor, BTensor, BiasTensor>();
        }

        bool init_with_btbuf = false;
        if constexpr (Std::is_same_v<get_mem_location<BiasTensor>, location::bias>) {
            init_with_btbuf = true;
        }

        if constexpr (trait.mmad_type == mmad_type::normal) {
            mmad_bias_instr::mmad(dst, fm, filter, bias, params.m, params.k, params.n,
                                  static_cast<uint8_t>(params.unit_flag), trait.disable_gemv, init_with_btbuf, false);
        } else if constexpr (trait.mmad_type == mmad_type::mx) {
            mmad_mx_bias_instr::mmad(dst, fm, filter, bias, params.m, params.k, params.n,
                                     static_cast<uint8_t>(params.unit_flag), trait.disable_gemv, init_with_btbuf,
                                     false);
        }
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_MMAD_WITH_BIAS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
