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
 * \file asc_update_addr_reg_impl.h
 * \brief asc_update_addr_reg_{b8,b16,b32}_impl：同名重载 1~4 参，原样转发 vag_{b8,b16,b32}
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_UPDATE_ADDR_REG_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_VECTOR_COMPUTE_IMPL_ASC_UPDATE_ADDR_REG_IMPL_H

#include "impl/c_api/instr_impl/npu_arch_3510/utils_impl.h"

#define ASC_DEFINE_UPDATE_ADDR_REG_IMPL(bxx)                                                                    \
    __simd_callee__ inline addr_reg asc_update_addr_reg_##bxx##_impl(uint32_t offset)                    \
    {                                                                                                    \
        if ASC_IS_AIV {                                                                                  \
            return vag_##bxx(offset);                                                                    \
        }                                                                                                \
        return addr_reg{};                                                                               \
    }                                                                                                    \
    __simd_callee__ inline addr_reg asc_update_addr_reg_##bxx##_impl(uint32_t offset0, uint32_t offset1) \
    {                                                                                                    \
        if ASC_IS_AIV {                                                                                  \
            return vag_##bxx(offset0, offset1);                                                          \
        }                                                                                                \
        return addr_reg{};                                                                               \
    }                                                                                                    \
    __simd_callee__ inline addr_reg asc_update_addr_reg_##bxx##_impl(                                    \
        uint32_t offset0, uint32_t offset1, uint32_t offset2)                                            \
    {                                                                                                    \
        if ASC_IS_AIV {                                                                                  \
            return vag_##bxx(offset0, offset1, offset2);                                                 \
        }                                                                                                \
        return addr_reg{};                                                                               \
    }                                                                                                    \
    __simd_callee__ inline addr_reg asc_update_addr_reg_##bxx##_impl(                                    \
        uint32_t offset0, uint32_t offset1, uint32_t offset2, uint32_t offset3)                          \
    {                                                                                                    \
        if ASC_IS_AIV {                                                                                  \
            return vag_##bxx(offset0, offset1, offset2, offset3);                                        \
        }                                                                                                \
        return addr_reg{};                                                                               \
    }

ASC_DEFINE_UPDATE_ADDR_REG_IMPL(b32)
ASC_DEFINE_UPDATE_ADDR_REG_IMPL(b16)
ASC_DEFINE_UPDATE_ADDR_REG_IMPL(b8)

#undef ASC_DEFINE_UPDATE_ADDR_REG_IMPL

#endif
