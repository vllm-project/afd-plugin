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

#ifndef IMPL_C_API_INSTR_IMPL_VECTOR_COMPUTE_IMPL_ASC_SET_VA_REG_IMPL_H
#define IMPL_C_API_INSTR_IMPL_VECTOR_COMPUTE_IMPL_ASC_SET_VA_REG_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl/utils_impl.h"

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ int8_t** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ uint8_t** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ int16_t** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ uint16_t** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ half** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ int32_t** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ uint32_t** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

__aicore__ inline void asc_set_va_reg_impl(ub_addr8_t addr, __ubuf__ float** src_array)
{
    if ASC_IS_AIV {
        uint64_t vaRegArray[8];
        for (int32_t i = 0; i < 8; ++i) {
            vaRegArray[i] = (uint64_t)src_array[i];
        }
        set_va_reg_sb(addr, vaRegArray);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
