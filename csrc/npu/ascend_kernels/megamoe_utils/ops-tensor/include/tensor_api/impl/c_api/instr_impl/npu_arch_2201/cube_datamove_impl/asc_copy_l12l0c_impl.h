/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_COPY_L12L0C_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_2201_CUBE_DATAMOVE_IMPL_ASC_COPY_L12L0C_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ half* dst, __cbuf__ half* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ half* dst, __cbuf__ half* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}
__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ half* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ half* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}
__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ bfloat16_t* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ bfloat16_t* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ float* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ float* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ int32_t* dst, __cbuf__ int32_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ int32_t* dst, __cbuf__ int32_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}

__aicore__ inline void asc_copy_l12l0c_impl(
    __cc__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    if ASC_IS_AIC {
        copy_matrix_cbuf_to_cc(dst, src, n_burst, len_burst, src_gap, dst_gap);
    }
}

__aicore__ inline void asc_copy_l12l0c_sync_impl(
    __cc__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap,
    uint16_t dst_gap)
{
    asc_copy_l12l0c_impl(dst, src, n_burst, len_burst, src_gap, dst_gap);
    asc_sync_post_process();
}
#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
