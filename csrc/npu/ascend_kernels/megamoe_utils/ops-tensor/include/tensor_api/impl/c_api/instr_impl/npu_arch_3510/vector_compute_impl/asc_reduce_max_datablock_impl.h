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
 * \file asc_reduce_max_datablock_impl.h
 * \brief
 */

#ifndef IMPL_C_API_INSTR_IMPL_VECTOR_COMPUTE_IMPL_NPU_ARCH_3510_ASC_REDUCE_MAX_DATABLOCK_IMPL_H
#define IMPL_C_API_INSTR_IMPL_VECTOR_COMPUTE_IMPL_NPU_ARCH_3510_ASC_REDUCE_MAX_DATABLOCK_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

// asc_reduce_max_datablock half
__simd_callee__ inline void asc_reduce_max_datablock_impl(vector_half& dst, vector_half src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcgmax(dst, src, mask, MODE_ZEROING);
    }
}

// asc_reduce_max_datablock float
__simd_callee__ inline void asc_reduce_max_datablock_impl(vector_float& dst, vector_float src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcgmax(dst, src, mask, MODE_ZEROING);
    }
}

// asc_reduce_max_datablock uint16_t
__simd_callee__ inline void asc_reduce_max_datablock_impl(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcgmax(dst, src, mask, MODE_ZEROING);
    }
}

// asc_reduce_max_datablock int16_t
__simd_callee__ inline void asc_reduce_max_datablock_impl(vector_int16_t& dst, vector_int16_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcgmax(dst, src, mask, MODE_ZEROING);
    }
}

// asc_reduce_max_datablock uint32_t
__simd_callee__ inline void asc_reduce_max_datablock_impl(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcgmax(dst, src, mask, MODE_ZEROING);
    }
}

// asc_reduce_max_datablock int32_t
__simd_callee__ inline void asc_reduce_max_datablock_impl(vector_int32_t& dst, vector_int32_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcgmax(dst, src, mask, MODE_ZEROING);
    }
}

#endif
