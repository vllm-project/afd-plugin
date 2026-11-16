/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
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

#ifndef IMPL_C_API_C_API_INSTR_IMPL_SYNC_C_API_IMPL_SYNC_C_API_IMPL_H
#define IMPL_C_API_C_API_INSTR_IMPL_SYNC_C_API_IMPL_SYNC_C_API_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h"

#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_set_flag_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_wait_flag_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_pipe_barrier_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_sync_data_barrier_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_sync_block_arrive_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_sync_inter_arrive_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sync_impl/asc_sync_subblock_arrive_impl.h"

__aicore__ inline void asc_sync_vec(int id) { asc_sync_vec_impl(id); }

[[deprecated("NOTICE: asc_sync_vec() is deprecated. "
             "Please use asc_sync() instead")]]
__aicore__ inline void asc_sync_vec()
{
    asc_sync_vec_impl();
}

__aicore__ inline void asc_sync_mte3(int id) { asc_sync_mte3_impl(id); }

__aicore__ inline void asc_sync_mte2(int id) { asc_sync_mte2_impl(id); }

__aicore__ inline void asc_sync() { asc_sync_impl(); }

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
