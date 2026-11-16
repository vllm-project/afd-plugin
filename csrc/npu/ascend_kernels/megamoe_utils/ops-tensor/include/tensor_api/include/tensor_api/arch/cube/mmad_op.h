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
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif

/*!
 * \file mmad_op.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_CUBE_MMAD_OP_H
#define INCLUDE_TENSOR_API_ARCH_CUBE_MMAD_OP_H

#include "tensor_api/arch/cube/mmad_trait.h"

namespace asc {
namespace te {

struct mmad_params {
    __aicore__ constexpr mmad_params(){};

    __aicore__ constexpr mmad_params(uint16_t m, uint16_t n, uint16_t k, unit_flag_mode unit_flag,
                                      bool init_with_zero) :
        m(m), n(n), k(k), unit_flag(unit_flag), init_with_zero(init_with_zero){};

    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    unit_flag_mode unit_flag = unit_flag_mode::disable;
    bool init_with_zero = false;
};

struct mmad_operation {
public:
    template <typename Trait, const Trait& traits, typename... Args>
    __aicore__ inline static void mmad(const Args&... args);

private:
    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor>
    __aicore__ inline static void mmad_impl(const CTensor& c, const ATensor& a, const BTensor& b,
                                            const mmad_params& params);

    template <const mmad_trait& trait, typename CTensor, typename ATensor, typename BTensor,
              typename BiasTensor>
    __aicore__ inline static void mmad_impl(const CTensor& c, const ATensor& a, const BTensor& b,
                                            const BiasTensor& bias, const mmad_params& params);
};

} // namespace te
} // namespace asc

#include "impl/tensor_api/arch/cube/mmad/mmad.h"

#endif // INCLUDE_TENSOR_API_ARCH_CUBE_MMAD_OP_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
