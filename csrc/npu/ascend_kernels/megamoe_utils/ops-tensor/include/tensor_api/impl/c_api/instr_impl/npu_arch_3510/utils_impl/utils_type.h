/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_UTILS_IMPL_UTILS_C_API_TYPE_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_UTILS_IMPL_UTILS_C_API_TYPE_H

#include "impl/utils/common_types.h"

#ifndef INT4X2_T_STRUCT
#define INT4X2_T_STRUCT

#endif

using vector_uint8_t = vector_u8;
using vector_uint16_t = vector_u16;
using vector_uint32_t = vector_u32;
using vector_uint64_t = vector_u64;
using vector_int8_t = vector_s8;
using vector_int16_t = vector_s16;
using vector_int32_t = vector_s32;
using vector_int64_t = vector_s64;
using vector_half = vector_f16;
using vector_float = vector_f32;
using vector_store_unalign = vector_align;
using vector_load_unalign = vector_align;
using addr_reg = vector_address;
// iter_reg has been deprecated, please use addr_reg instead.
using iter_reg [[deprecated("iter_reg has been deprecated, please use addr_reg instead.")]] = addr_reg;
using vector_bfloat16_t = vector_bf16;
using vector_hifloat8_t = vector_hif8;
using vector_fp8_e4m3fn_t = vector_f8e4m3;
using vector_fp8_e5m2_t = vector_f8e5m2;
using vector_fp8_e8m0_t = vector_f8e8m0;
using vector_int4x2_t = vector_s4x2;
using vector_fp4x2_e2m1_t = vector_f4e2m1x2;
using vector_fp4x2_e1m2_t = vector_f4e1m2x2;

#endif
