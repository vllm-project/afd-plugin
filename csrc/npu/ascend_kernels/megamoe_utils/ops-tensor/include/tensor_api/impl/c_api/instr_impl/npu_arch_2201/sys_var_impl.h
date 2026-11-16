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

#ifndef IMPL_C_API_C_API_INSTR_IMPL_SYS_VAR_C_API_IMPL_SYS_VAR_C_API_IMPL_H
#define IMPL_C_API_C_API_INSTR_IMPL_SYS_VAR_C_API_IMPL_SYS_VAR_C_API_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_core_id_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_ctrl_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_phy_buf_addr_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_overflow_status_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_sub_block_id_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_sub_block_num_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_sys_cnt_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_set_ctrl_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_block_idx_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_block_num_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_arch_ver_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/get_program_counter_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_ffts_base_addr_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_set_ffts_base_addr_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_smmu_tag_version_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_phy_stack_base_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_status_impl.h"
#include "impl/c_api/instr_impl/npu_arch_2201/sys_var_impl/asc_get_sys_virtual_base_impl.h"

__aicore__ inline int64_t asc_get_core_id() { return asc_get_core_id_impl(); }

__aicore__ inline int64_t asc_get_block_idx() { return asc_get_block_idx_impl(); }

__aicore__ inline int64_t asc_get_block_num() { return asc_get_block_num_impl(); }

__aicore__ inline int64_t asc_get_ctrl() { return asc_get_ctrl_impl(); }

__aicore__ inline uint64_t asc_get_phy_buf_addr(uint64_t offset) { return asc_get_phy_buf_addr_impl(offset); }

__aicore__ inline uint64_t asc_get_overflow_status() { return asc_get_overflow_status_impl(); }

__aicore__ inline int64_t asc_get_sub_block_id() { return asc_get_sub_block_id_impl(); }

__aicore__ inline int64_t asc_get_sub_block_num() { return asc_get_sub_block_num_impl(); }

__aicore__ inline int64_t asc_get_system_cycle() { return asc_get_system_cycle_impl(); }

__aicore__ inline void asc_set_ctrl(uint64_t config) { asc_set_ctrl_impl(config); }

__aicore__ inline void asc_get_arch_ver(uint32_t& core_version) { asc_get_arch_ver_impl(core_version); }

__aicore__ inline int64_t asc_get_program_counter() { return asc_get_program_counter_impl(); }

__aicore__ inline int64_t asc_get_ffts_base_addr() { return asc_get_ffts_base_addr_impl(); }

__aicore__ inline void asc_set_ffts_base_addr(uint64_t config) { asc_set_ffts_base_addr_impl(config); }

__aicore__ inline int64_t asc_get_smmu_tag_version() { return asc_get_smmu_tag_version_impl(); }

__aicore__ inline int64_t asc_get_phy_stack_base() { return asc_get_phy_stack_base_impl(); }

__aicore__ inline int64_t asc_get_status() { return asc_get_status_impl(); }

__aicore__ inline int64_t asc_get_sys_virtual_base() { return asc_get_sys_virtual_base_impl(); }

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
