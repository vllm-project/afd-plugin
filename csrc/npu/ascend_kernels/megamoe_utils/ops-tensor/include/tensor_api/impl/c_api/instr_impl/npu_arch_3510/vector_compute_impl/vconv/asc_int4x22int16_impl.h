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
 * \file asc_int4x22int16_impl.h
 * \brief
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_VCONV_ASC_INT4X22INT16_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_VCONV_ASC_INT4X22INT16_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__simd_callee__ inline void asc_int4x22int16_impl(vector_int16_t& dst, vector_int4x2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_s42s16(dst, src, mask, PART_P0, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_int4x22int16_v2_impl(vector_int16_t& dst, vector_int4x2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_s42s16(dst, src, mask, PART_P1, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_int4x22int16_v3_impl(vector_int16_t& dst, vector_int4x2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_s42s16(dst, src, mask, PART_P2, MODE_ZEROING);
    }
}

__simd_callee__ inline void asc_int4x22int16_v4_impl(vector_int16_t& dst, vector_int4x2_t src, vector_bool mask)
{
    if ASC_IS_AIV {
        vcvt_s42s16(dst, src, mask, PART_P3, MODE_ZEROING);
    }
}

#endif
