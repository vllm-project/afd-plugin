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
 * \file asc_storeunalign_post_postupdate_impl.h
 * \brief
 */

#ifndef IMPL_C_API_INSTR_IMPL_VECTOR_DATAMOVE_IMPL_NPU_ARCH_3510_ASC_STOREUNALIGN_POST_POSTUPDATE_IMPL_H
#define IMPL_C_API_INSTR_IMPL_VECTOR_DATAMOVE_IMPL_NPU_ARCH_3510_ASC_STOREUNALIGN_POST_POSTUPDATE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ int8_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ uint8_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ int16_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ uint16_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ int32_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ int64_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ uint32_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ half*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ float*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ bfloat16_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, dst, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ hifloat8_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        vstas(src, reinterpret_cast<__ubuf__ uint8_t*&>(dst), offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ fp8_e4m3fn_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e4m3_t* dst_tmp = reinterpret_cast<__ubuf__ float8_e4m3_t*>(dst);
        vstas(src, dst_tmp, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ fp8_e5m2_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e5m2_t* dst_tmp = reinterpret_cast<__ubuf__ float8_e5m2_t*>(dst);
        vstas(src, dst_tmp, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ fp8_e8m0_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float8_e8m0_t* dst_tmp = reinterpret_cast<__ubuf__ float8_e8m0_t*>(dst);
        vstas(src, dst_tmp, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ fp4x2_e2m1_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e2m1x2_t* dst_tmp = reinterpret_cast<__ubuf__ float4_e2m1x2_t*>(dst);
        vstas(src, dst_tmp, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ fp4x2_e1m2_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t* dst_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*>(dst);
        vstas(src, dst_tmp, offset, POST_UPDATE);
    }
}

__simd_callee__ inline void asc_storeunalign_post_postupdate_impl(
    __ubuf__ int4b_t*& dst, vector_store_unalign src, int32_t offset)
{
    if ASC_IS_AIV {
        __ubuf__ float4_e1m2x2_t*& dst_tmp = reinterpret_cast<__ubuf__ float4_e1m2x2_t*&>(dst);
        vstas(src, dst_tmp, offset, POST_UPDATE);
    }
}

#endif
