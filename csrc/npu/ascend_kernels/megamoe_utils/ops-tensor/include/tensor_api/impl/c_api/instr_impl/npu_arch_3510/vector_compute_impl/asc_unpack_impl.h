/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_unpack_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_UNPACK_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_UNPACK_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_unpack_upper_impl(vector_uint16_t& dst, vector_uint8_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, HIGHER);
    }
}

__simd_callee__ inline void asc_unpack_lower_impl(vector_uint16_t& dst, vector_uint8_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, LOWER);
    }
}

__simd_callee__ inline void asc_unpack_upper_impl(vector_int16_t& dst, vector_int8_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, HIGHER);
    }
}

__simd_callee__ inline void asc_unpack_lower_impl(vector_int16_t& dst, vector_int8_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, LOWER);
    }
}

__simd_callee__ inline void asc_unpack_upper_impl(vector_uint32_t& dst, vector_uint16_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, HIGHER);
    }
}

__simd_callee__ inline void asc_unpack_lower_impl(vector_uint32_t& dst, vector_uint16_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, LOWER);
    }
}

__simd_callee__ inline void asc_unpack_upper_impl(vector_int32_t& dst, vector_int16_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, HIGHER);
    }
}

__simd_callee__ inline void asc_unpack_lower_impl(vector_int32_t& dst, vector_int16_t src)
{
    if ASC_IS_AIV {
        vunpack(dst, src, LOWER);
    }
}

__simd_callee__ inline void asc_unpack_upper_impl(vector_bool& dst, vector_bool src)
{
    if ASC_IS_AIV {
        punpack(dst, src, HIGHER);
    }
}

__simd_callee__ inline void asc_unpack_lower_impl(vector_bool& dst, vector_bool src)
{
    if ASC_IS_AIV {
        punpack(dst, src, LOWER);
    }
}

#endif
