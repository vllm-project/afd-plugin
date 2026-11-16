/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SCALAR_COMPUTE_IMPL_ASC_FLOAT2INT32_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SCALAR_COMPUTE_IMPL_ASC_FLOAT2INT32_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__aicore__ inline int32_t asc_float2int32_rd_impl(float value) { return conv_f322s32f(value); }

__aicore__ inline int32_t asc_float2int32_ru_impl(float value) { return conv_f322s32c(value); }

__aicore__ inline int32_t asc_float2int32_rn_impl(float value) { return conv_f322s32r(value); }

__aicore__ inline int32_t asc_float2int32_rna_impl(float value) { return conv_f322s32a(value); }

#endif
