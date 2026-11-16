/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file asc_sync_pipe_impl.h
 * \brief
 */

#if !defined(ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS)

#define ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SYNC_IMPL_ASC_SYNC_PIPE_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_SYNC_IMPL_ASC_SYNC_PIPE_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

__aicore__ inline void asc_sync_pipe_impl(pipe_t pipe)
{
    if ASC_IS_AIC {
        if (pipe == pipe_t::PIPE_M) {
            pipe_barrier(pipe_t::PIPE_M);
        } else if (pipe == pipe_t::PIPE_MTE1) {
            pipe_barrier(pipe_t::PIPE_MTE1);
        } else if (pipe == pipe_t::PIPE_MTE2) {
            pipe_barrier(pipe_t::PIPE_MTE2);
        } else if (pipe == pipe_t::PIPE_ALL) {
            pipe_barrier(pipe_t::PIPE_ALL);
        } else if (pipe == pipe_t::PIPE_FIX) {
            pipe_barrier(pipe_t::PIPE_FIX);
        }
    } else if ASC_IS_AIV {
        if (pipe == pipe_t::PIPE_MTE2) {
            pipe_barrier(pipe_t::PIPE_MTE2);
        } else if (pipe == pipe_t::PIPE_MTE3) {
            pipe_barrier(pipe_t::PIPE_MTE3);
        } else if (pipe == pipe_t::PIPE_ALL) {
            pipe_barrier(pipe_t::PIPE_ALL);
        }
    }
}

__aicore__ inline void asc_sync_vec_impl() { pipe_barrier(pipe_t::PIPE_ALL); }

__aicore__ inline void asc_sync_vec_impl(int id)
{
    set_flag(pipe_t::PIPE_V, pipe_t::PIPE_MTE2, static_cast<event_t>(id));
    set_flag(pipe_t::PIPE_V, pipe_t::PIPE_MTE3, static_cast<event_t>(id));
    set_flag(pipe_t::PIPE_V, pipe_t::PIPE_S, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_V, pipe_t::PIPE_MTE2, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_V, pipe_t::PIPE_MTE3, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_V, pipe_t::PIPE_S, static_cast<event_t>(id));
}

__aicore__ inline void asc_sync_mte2_impl(int id)
{
    set_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_MTE3, static_cast<event_t>(id));
    set_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_V, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_MTE3, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_V, static_cast<event_t>(id));
    pipe_barrier(pipe_t::PIPE_MTE2);
}

__aicore__ inline void asc_sync_mte3_impl(int id)
{
    set_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_MTE2, static_cast<event_t>(id));
    set_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_V, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_MTE2, static_cast<event_t>(id));
    wait_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_V, static_cast<event_t>(id));
    pipe_barrier(pipe_t::PIPE_MTE3);
}

__aicore__ inline void asc_sync_impl() { pipe_barrier(pipe_t::PIPE_ALL); }

#endif

#if defined(UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_C_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
