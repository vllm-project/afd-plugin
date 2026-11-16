/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_scatter_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CUBE_DMAMOVE_ASC_SCATTER_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CUBE_DMAMOVE_ASC_SCATTER_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// asc_scatter int8_t
__simd_callee__ inline void asc_scatter_impl(
    vector_int8_t& src, __ubuf__ int8_t* dst, vector_uint16_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter uint8_t
__simd_callee__ inline void asc_scatter_impl(
    vector_uint8_t& src, __ubuf__ uint8_t* dst, vector_uint16_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter int16_t
__simd_callee__ inline void asc_scatter_impl(
    vector_int16_t& src, __ubuf__ int16_t* dst, vector_uint16_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter uint16_t
__simd_callee__ inline void asc_scatter_impl(
    vector_uint16_t& src, __ubuf__ uint16_t* dst, vector_uint16_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter int32_t
__simd_callee__ inline void asc_scatter_impl(
    vector_int32_t& src, __ubuf__ int32_t* dst, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter uint32_t
__simd_callee__ inline void asc_scatter_impl(
    vector_uint32_t& src, __ubuf__ uint32_t* dst, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter bfloat16_t
__simd_callee__ inline void asc_scatter_impl(
    vector_bfloat16_t& src, __ubuf__ bfloat16_t* dst, vector_uint16_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter float16_t
__simd_callee__ inline void asc_scatter_impl(
    vector_half& src, __ubuf__ half* dst, vector_uint16_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

// asc_scatter float32_t
__simd_callee__ inline void asc_scatter_impl(
    vector_float& src, __ubuf__ float* dst, vector_uint32_t index, vector_bool mask)
{
    if ASC_IS_AIV {
        vscatter(src, dst, index, mask);
    }
}

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
