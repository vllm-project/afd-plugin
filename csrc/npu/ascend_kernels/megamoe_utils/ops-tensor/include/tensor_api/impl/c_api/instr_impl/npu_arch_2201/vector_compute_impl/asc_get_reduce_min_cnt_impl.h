/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_VECTOR_COMPUTE_IMPL_ASC_GET_REDUCE_MIN_CNT_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_VECTOR_COMPUTE_IMPL_ASC_GET_REDUCE_MIN_CNT_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

__aicore__ inline void asc_get_reduce_min_cnt_impl(half& val, uint32_t& index)
{
    if ASC_IS_AIV {
        constexpr uint8_t offset = 32;
        int64_t max_min_cnt = get_max_min_cnt();
        union {
            half h;
            uint16_t u;
        } u162half = {.u = static_cast<uint16_t>(0xffff & max_min_cnt)};
        val = u162half.h;
        index = 0xffffffff & (max_min_cnt >> offset);
    }
}

__aicore__ inline void asc_get_reduce_min_cnt_impl(float& val, uint32_t& index)
{
    if ASC_IS_AIV {
        constexpr uint8_t offset = 32;
        int64_t max_min_cnt = get_max_min_cnt();
        union {
            float f;
            uint32_t u;
        } u322float = {.u = static_cast<uint32_t>(0xffffffff & max_min_cnt)};
        val = u322float.f;
        index = 0xffffffff & (max_min_cnt >> offset);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
