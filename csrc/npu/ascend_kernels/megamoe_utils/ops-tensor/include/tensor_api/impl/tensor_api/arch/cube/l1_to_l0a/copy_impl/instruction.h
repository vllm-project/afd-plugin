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
    "impl/tensor_api/arch/cube/l1_to_l0a/copy_impl/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_INSTRUCTION_H

#include "impl/tensor_api/arch/utils/arch_utils.h"

namespace asc {
namespace te {

constexpr img2col_params<int16_t> default_img2col_params{};

class load_l1_to_l0a_instr {
public:
    template <bool transpose, typename DstTensor, typename SrcTensor, typename DstOffset, typename SrcOffset,
              typename... Params>
    __aicore__ inline static void load_data_with_offset(const DstTensor& dst, const SrcTensor& src,
                                                        const DstOffset& dst_offset, const SrcOffset& src_offset,
                                                        const Params&... params)
    {
        auto src_data = src.data() + src_offset;
        load_data<transpose>((dst.data() + dst_offset).get(), src_data.get(), params...);
    }

    template <bool transpose, typename DataType>
    __aicore__ inline static void load_data(__ca__ DataType* dst, __cbuf__ DataType* src, uint16_t m_start_position,
                                            uint16_t k_start_position, uint8_t m_step, uint8_t k_step,
                                            int16_t src_stride, uint16_t dst_stride)
    {
        TENSOR_API_DEBUG_CHECK(debug_check_l0_transpose, transpose, sizeof(DataType), k_step, "copy_l1_to_l0a");
        if constexpr (transpose) {
            asc_copy_l12l0a_transpose(dst, src, m_start_position, k_start_position, m_step, k_step, src_stride,
                                      dst_stride);
        } else {
            asc_copy_l12l0a(dst, src, m_start_position, k_start_position, m_step, k_step, src_stride, dst_stride);
        }
    }
};

class load_l1_to_l0a_img2col_instr {
private:
    static constexpr uint32_t dst_stride_bit_offset = 32;

public:
    __aicore__ inline static void set_f_matrix(uint16_t l1_h, uint16_t l1_w, const uint8_t pad_list[4])
    {
        asc_l13d_fmatrix_config config;
        config.l1_height = l1_h;
        config.l1_width = l1_w;
        config.padding_left_size = pad_list[0];
        config.padding_right_size = pad_list[1];
        config.padding_top_size = pad_list[2];
        config.padding_bottom_size = pad_list[3];
        asc_set_l13d_fmatrix(config);
    }

    template <typename PadType>
    __aicore__ inline static void set_padding(PadType pad_value)
    {
        asc_set_l13d_padding(pad_value);
    }

    __aicore__ inline static void set_repeat(uint16_t dst_stride)
    {
        asc_load3d_v2_config config;
        config.rpt_stride = 0;
        config.rpt_time = 1;
        config.rpt_mode = 0;
        config.config |= static_cast<uint64_t>(dst_stride) << dst_stride_bit_offset;
        asc_set_l13d_rpt(config);
    }

    template <typename DstTensor, typename SrcTensor, typename DstOffset, typename SrcOffset, typename... Params>
    __aicore__ inline static void load_data_with_offset(const DstTensor& dst, const SrcTensor& src,
                                                        const DstOffset& dst_offset, const SrcOffset& src_offset,
                                                        const Params&... params)
    {
        auto src_data = src.data() + src_offset;
        load_data((dst.data() + dst_offset).get(), src_data.get(), params...);
    }

    template <typename DataType>
    __aicore__ inline static void
    load_data(__ca__ DataType* dst, __cbuf__ DataType* src, uint16_t k_extension, uint16_t m_extension,
              uint16_t k_start_pt, uint16_t m_start_pt, uint8_t stride_w, uint8_t stride_h, uint8_t filter_w,
              uint8_t filter_h, uint8_t dilation_filter_w, uint8_t dilation_filter_h, bool filter_size_w,
              bool filter_size_h, bool transpose, bool f_matrix_ctrl, uint16_t channel_size)
    {
        asc_copy_l12l0a(dst, src, k_extension, m_extension, k_start_pt, m_start_pt, stride_w, stride_h, filter_w,
                        filter_h, dilation_filter_w, dilation_filter_h, filter_size_w, filter_size_h, transpose,
                        f_matrix_ctrl, channel_size);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_L0A_COPY_IMPL_INSTRUCTION_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
