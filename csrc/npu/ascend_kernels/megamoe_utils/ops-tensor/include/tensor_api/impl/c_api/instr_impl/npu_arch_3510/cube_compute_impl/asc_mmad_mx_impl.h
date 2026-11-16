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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CUBE_COMPUTE_IMPL_ASC_MMAD_MX_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CUBE_COMPUTE_IMPL_ASC_MMAD_MX_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl/utils_impl.h"

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e1m2_t* a_matrix, __cb__ fp4x2_e1m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float4_e1m2x2_t* a_matrix_tmp = reinterpret_cast<__ca__ float4_e1m2x2_t*>(a_matrix);
        __cb__ float4_e1m2x2_t* b_matrix_tmp = reinterpret_cast<__cb__ float4_e1m2x2_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e1m2_t* a_matrix, __cb__ fp4x2_e1m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e1m2_t* a_matrix, __cb__ fp4x2_e2m1_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float4_e1m2x2_t* a_matrix_tmp = reinterpret_cast<__ca__ float4_e1m2x2_t*>(a_matrix);
        __cb__ float4_e2m1x2_t* b_matrix_tmp = reinterpret_cast<__cb__ float4_e2m1x2_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e1m2_t* a_matrix, __cb__ fp4x2_e2m1_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e2m1_t* a_matrix, __cb__ fp4x2_e1m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float4_e2m1x2_t* a_matrix_tmp = reinterpret_cast<__ca__ float4_e2m1x2_t*>(a_matrix);
        __cb__ float4_e1m2x2_t* b_matrix_tmp = reinterpret_cast<__cb__ float4_e1m2x2_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e2m1_t* a_matrix, __cb__ fp4x2_e1m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e2m1_t* a_matrix, __cb__ fp4x2_e2m1_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float4_e2m1x2_t* a_matrix_tmp = reinterpret_cast<__ca__ float4_e2m1x2_t*>(a_matrix);
        __cb__ float4_e2m1x2_t* b_matrix_tmp = reinterpret_cast<__cb__ float4_e2m1x2_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp4x2_e2m1_t* a_matrix, __cb__ fp4x2_e2m1_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp8_e4m3fn_t* a_matrix, __cb__ fp8_e4m3fn_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float8_e4m3_t* a_matrix_tmp = reinterpret_cast<__ca__ float8_e4m3_t*>(a_matrix);
        __cb__ float8_e4m3_t* b_matrix_tmp = reinterpret_cast<__cb__ float8_e4m3_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp8_e4m3fn_t* a_matrix, __cb__ fp8_e4m3fn_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp8_e4m3fn_t* a_matrix, __cb__ fp8_e5m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float8_e4m3_t* a_matrix_tmp = reinterpret_cast<__ca__ float8_e4m3_t*>(a_matrix);
        __cb__ float8_e5m2_t* b_matrix_tmp = reinterpret_cast<__cb__ float8_e5m2_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp8_e4m3fn_t* a_matrix, __cb__ fp8_e5m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp8_e5m2_t* a_matrix, __cb__ fp8_e4m3fn_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float8_e5m2_t* a_matrix_tmp = reinterpret_cast<__ca__ float8_e5m2_t*>(a_matrix);
        __cb__ float8_e4m3_t* b_matrix_tmp = reinterpret_cast<__cb__ float8_e4m3_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp8_e5m2_t* a_matrix, __cb__ fp8_e4m3fn_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}

__aicore__ inline void asc_mmad_mx_impl(
    __cc__ float* c_matrix, __ca__ fp8_e5m2_t* a_matrix, __cb__ fp8_e5m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    if ASC_IS_AIC {
        __ca__ float8_e5m2_t* a_matrix_tmp = reinterpret_cast<__ca__ float8_e5m2_t*>(a_matrix);
        __cb__ float8_e5m2_t* b_matrix_tmp = reinterpret_cast<__cb__ float8_e5m2_t*>(b_matrix);
        mad_mx(
            c_matrix, a_matrix_tmp, b_matrix_tmp, left_height, n_dim, right_width, unit_flag, disable_gemv,
            c_matrix_source, c_matrix_init_val);
    }
}

__aicore__ inline void asc_mmad_mx_sync_impl(
    __cc__ float* c_matrix, __ca__ fp8_e5m2_t* a_matrix, __cb__ fp8_e5m2_t* b_matrix, uint16_t left_height,
    uint16_t n_dim, uint16_t right_width, uint8_t unit_flag, bool disable_gemv, bool c_matrix_source,
    bool c_matrix_init_val)
{
    asc_mmad_mx_impl(
        c_matrix, a_matrix, b_matrix, left_height, n_dim, right_width, unit_flag, disable_gemv, c_matrix_source,
        c_matrix_init_val);
    asc_sync_post_process();
}
#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
