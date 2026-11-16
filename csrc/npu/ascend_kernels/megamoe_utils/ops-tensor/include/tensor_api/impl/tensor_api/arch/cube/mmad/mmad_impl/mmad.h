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
    "impl/tensor_api/arch/cube/mmad/mmad_impl/mmad.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file mmad.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_MMAD_H
#define IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_MMAD_H

#include "impl/tensor_api/arch/cube/mmad/mmad_impl/instruction.h"

namespace asc {
namespace te {

class mmad_executor {
public:
    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename Params>
    __aicore__ inline static void run(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                      const Params& params)
    {
        mmad_impl<trait, CTensor, ATensor, BTensor>(dst, fm, filter, params);
    }

private:
    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor>
    __aicore__ inline static constexpr void check_template_for_normal()
    {
        check_layout_pattern<CTensor, ATensor, BTensor>();
        check_data_type::check_mmad_data_type<CTensor, ATensor, BTensor>();
    }

    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor>
    __aicore__ inline static constexpr void check_template_for_mx()
    {
        check_layout_pattern<CTensor, ATensor, BTensor>();
        check_data_type::check_mx_mmad_data_type<CTensor, ATensor, BTensor>();
    }

    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor, typename Params>
    __aicore__ inline static void mmad_impl(const CTensor& dst, const ATensor& fm, const BTensor& filter,
                                            const Params& params)
    {
        if constexpr (trait.mmad_type == mmad_type::normal) {
            check_template_for_normal<trait, CTensor, ATensor, BTensor>();
            mmad_instr::mmad(dst, fm, filter, params.m, params.k, params.n, static_cast<uint8_t>(params.unit_flag),
                             trait.disable_gemv, trait.init_with_btbuf, params.init_with_zero);
        } else if constexpr (trait.mmad_type == mmad_type::mx) {
            check_template_for_mx<trait, CTensor, ATensor, BTensor>();
            mmad_mx_instr::mmad(dst, fm, filter, params.m, params.k, params.n, static_cast<uint8_t>(params.unit_flag),
                                trait.disable_gemv, trait.init_with_btbuf, params.init_with_zero);
        }
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_MMAD_MMAD_IMPL_MMAD_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
