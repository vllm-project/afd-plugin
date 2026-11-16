/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/cube/l0c_to_gm/copy_impl/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_COPY_IMPL_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_COPY_IMPL_INSTRUCTION_H

#include "impl/tensor_api/arch/utils/arch_utils.h"

namespace asc {
namespace te {

class copy_l0c_to_gm_instr {
public:
    template <QuantMode_t quant_pre, typename DstTensor, typename SrcTensor, typename DstOffset, typename SrcOffset,
              typename... Params>
    __aicore__ inline static void data_copy_with_offset(const DstTensor& dst, const SrcTensor& src,
                                                        const DstOffset& dst_offset, const SrcOffset& src_offset,
                                                        const Params&... params)
    {
        auto dst_data = dst.data() + dst_offset;
        auto src_data = src.data() + src_offset;
        data_copy<quant_pre>(dst_data.get(), src_data.get(), params...);
    }

    template <QuantMode_t quant_pre, typename DstType, typename SrcType>
    __aicore__ inline static void data_copy(__gm__ DstType* dst, __cc__ SrcType* src, uint32_t n_size, uint32_t m_size,
                                            uint32_t src_stride, uint32_t dst_stride, uint8_t cache_mode, bool relu_en,
                                            uint8_t unit_flag, bool is_channel_split, bool nz2nd_en, bool nz2dn_en)
    {
        TENSOR_API_DEBUG_CHECK(debug_check_block_count, n_size, "n_size", "copy_l0c_to_gm instruction");
        TENSOR_API_DEBUG_CHECK(debug_check_fixpipe_n, n_size, is_channel_split, nz2nd_en, nz2dn_en, "copy_l0c_to_gm");
        TENSOR_API_DEBUG_CHECK(debug_check_fixpipe_m, m_size, nz2nd_en, "copy_l0c_to_gm instruction");
        TENSOR_API_DEBUG_CHECK(debug_check_fixpipe_stride, src_stride, dst_stride, "copy_l0c_to_gm instruction");
        TENSOR_API_DEBUG_CHECK(debug_check_unit_flag, unit_flag, "copy_l0c_to_gm instruction");

        asc_copy_l0c2gm(dst, src, static_cast<uint16_t>(n_size), static_cast<uint16_t>(m_size), dst_stride,
                        static_cast<uint16_t>(src_stride), cache_mode, 0, unit_flag, static_cast<uint64_t>(quant_pre),
                        static_cast<uint8_t>(relu_en), is_channel_split, nz2nd_en,
                        static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, 0, false, false, false, nz2dn_en);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_GM_COPY_IMPL_INSTRUCTION_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
