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

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SCALAR_COMPUTE_IMPL_ASC_ATOMIC_ADD_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SCALAR_COMPUTE_IMPL_ASC_ATOMIC_ADD_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl/utils_impl.h"

namespace __asc_aicore {
__aicore__ inline int32_t asc_atomic_add_impl(__gm__ int32_t* address, int32_t val) { return atomicAdd(address, val); }

__aicore__ inline uint32_t asc_atomic_add_impl(__gm__ uint32_t* address, uint32_t val)
{
    return atomicAdd(address, val);
}

__aicore__ inline float asc_atomic_add_impl(__gm__ float* address, float val) { return atomicAdd(address, val); }

__aicore__ inline int64_t asc_atomic_add_impl(__gm__ int64_t* address, int64_t val) { return atomicAdd(address, val); }

__aicore__ inline uint64_t asc_atomic_add_impl(__gm__ uint64_t* address, uint64_t val)
{
    return atomicAdd(address, val);
}

} // namespace __asc_aicore

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
