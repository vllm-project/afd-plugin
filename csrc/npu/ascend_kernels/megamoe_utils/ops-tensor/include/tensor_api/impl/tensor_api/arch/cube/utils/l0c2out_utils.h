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
    "impl/tensor_api/arch/cube/utils/l0c2out_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file l0c2out_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_UTILS_L0C2OUT_UTILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_UTILS_L0C2OUT_UTILS_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/layout_method.h"

namespace asc {
namespace te {

constexpr uint32_t main_loop_n_size = 512;
constexpr uint32_t cburst_num = main_loop_n_size / block_cube;

constexpr l0c_to_gm_params default_l0c_to_gm_params = {};
constexpr l0c_to_ub_params default_l0c_to_ub_params = {};
constexpr l0c_to_l1_params default_l0c_to_l1_params = {};

template <typename TensorType>
inline constexpr bool is_l0c_out_src_batch_layout_v = TensorType::layout_type::depth == five_dim_data;

// Generic batched-layout predicate for L0C->out tensors that may use ND/DN/NDExt/DNExt or quant
// shapes: non-batch depth is 2/4, batch depth is 3/5. Use is_l0c_out_src_batch_layout_v for src (NZ).
template <typename TensorType>
inline constexpr bool is_l0c_out_batched_layout_v =
    TensorType::layout_type::depth == three_dim_data || TensorType::layout_type::depth == five_dim_data;

template <typename TensorType>
inline constexpr bool is_l0c_out_nd_format_v =
    is_satisfied_ptn_format_v<TensorType, nd_ext_layout_ptn> || is_satisfied_ptn_format_v<TensorType, nd_layout_ptn>;

template <typename TensorType>
inline constexpr bool is_l0c_out_dn_format_v =
    is_satisfied_ptn_format_v<TensorType, dn_ext_layout_ptn> || is_satisfied_ptn_format_v<TensorType, dn_layout_ptn>;

template <typename TensorType>
inline constexpr bool is_l0c_out_nz_format_v = is_satisfied_ptn_format_v<TensorType, nz_layout_ptn>;

// NC1HWC0 (N, C1, H, W, C0): a flat 5-dim layout with no pattern tag (depth 5 AND rank 5). Batched
// NZ/ND/DN dst layouts are also depth 5 but nest to rank 2, so the rank check keeps them apart.
template <typename TensorType>
inline constexpr bool is_l0c_out_nc1hwc0_v =
    TensorType::layout_type::depth == five_dim_data && TensorType::layout_type::rank_size == five_dim_data;

template <typename DstTensorType, typename SrcTensorType>
inline constexpr bool is_l0c_out_batch_nz2nz_v =
    is_l0c_out_nz_format_v<DstTensorType> && is_l0c_out_nz_format_v<SrcTensorType>
    && DstTensorType::layout_type::depth == five_dim_data && SrcTensorType::layout_type::depth == five_dim_data;

template <typename DstTensor, typename LayoutType>
__aicore__ inline static constexpr uint32_t get_l0c_out_nd_stride(const LayoutType& layout)
{
    if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
        return get_element<attr_info::stride, attr_info::row>(layout);
    } else {
        return get_element<attr_info::stride, attr_info::row, 1>(layout);
    }
}

template <typename DstTensor, typename LayoutType>
__aicore__ inline static constexpr uint32_t get_l0c_out_dn_stride(const LayoutType& layout)
{
    if constexpr (is_satisfied_ptn_format_v<DstTensor, dn_layout_ptn>) {
        return get_element<attr_info::stride, attr_info::column>(layout);
    } else {
        return get_element<attr_info::stride, attr_info::column, 1>(layout);
    }
}

struct l0c_out_copy_params {
    uint32_t n_size;
    uint32_t m_size;
    uint32_t src_stride;
    uint32_t dst_stride;
};

template <typename DstTensor, typename SrcTensor, typename DstLayout, typename SrcLayout>
__aicore__ inline static constexpr l0c_out_copy_params make_l0c_out_copy_params(const DstLayout& dst_layout,
                                                                                const SrcLayout& src_layout)
{
    if constexpr (is_l0c_out_nc1hwc0_v<DstTensor>) {
        // dst NC1HWC0 (N, C1, H, W, C0), N == 1: nSize = Cout = C1*C0, mSize = Ho*Wo = H*W.
        // dst_stride is the C1-axis stride (= H*W*C0), i.e. the C0-block step without the NZ 16-row
        // alignment padding. src is the non-batch NZ, so src_stride follows the plain NZ path.
        auto dst_shape = dst_layout.shape();
        return {static_cast<uint32_t>(get<1>(dst_shape) * get<4>(dst_shape)),
                static_cast<uint32_t>(get<2>(dst_shape) * get<3>(dst_shape)),
                static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(src_layout) / fractal_fixed),
                static_cast<uint32_t>(get<1>(dst_layout.stride()))};
    } else if constexpr (is_l0c_out_batch_nz2nz_v<DstTensor, SrcTensor>) {
        auto src_no_batch_layout = remove_batch_dim(src_layout);
        auto dst_no_batch_layout = remove_batch_dim(dst_layout);
        return {
            static_cast<uint32_t>(Std::min(get<0>(src_layout.shape()) * get_total_column_shape(src_no_batch_layout),
                                           get<0>(dst_layout.shape()) * get_total_column_shape(dst_no_batch_layout))),
            static_cast<uint32_t>(
                Std::min(get_total_row_shape(src_no_batch_layout), get_total_row_shape(dst_no_batch_layout))),
            static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(src_no_batch_layout)
                                  / fractal_fixed),
            static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(dst_no_batch_layout))};
    } else {
        const uint32_t n_size =
            static_cast<uint32_t>(Std::min(get_total_column_shape(src_layout), get_total_column_shape(dst_layout)));
        const uint32_t m_size =
            static_cast<uint32_t>(Std::min(get_total_row_shape(src_layout), get_total_row_shape(dst_layout)));
        const uint32_t src_stride =
            static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(src_layout) / fractal_fixed);

        if constexpr (is_l0c_out_nd_format_v<DstTensor>) {
            return {n_size, m_size, src_stride, get_l0c_out_nd_stride<DstTensor>(dst_layout)};
        } else if constexpr (is_l0c_out_dn_format_v<DstTensor>) {
            return {n_size, m_size, src_stride, get_l0c_out_dn_stride<DstTensor>(dst_layout)};
        } else {
            return {n_size, m_size, src_stride,
                    static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(dst_layout))};
        }
    }
}

template <typename DstTensor, typename SrcTensor, typename DstLayout, typename SrcLayout, typename DstShape>
__aicore__ inline static constexpr l0c_out_copy_params
make_l0c_out_copy_params(const DstLayout& dst_layout, const SrcLayout& src_layout, const DstShape& dst_shape)
{
    const uint32_t n_size =
        static_cast<uint32_t>(Std::min(get_total_column_shape(src_layout), get_shape_columns(dst_shape)));
    const uint32_t m_size = static_cast<uint32_t>(Std::min(get_total_row_shape(src_layout), get_shape_rows(dst_shape)));
    const uint32_t src_stride =
        static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(src_layout) / fractal_fixed);
    if constexpr (is_l0c_out_nd_format_v<DstTensor>) {
        return {n_size, m_size, src_stride, get_l0c_out_nd_stride<DstTensor>(dst_layout)};
    } else if constexpr (is_l0c_out_dn_format_v<DstTensor>) {
        return {n_size, m_size, src_stride, get_l0c_out_dn_stride<DstTensor>(dst_layout)};
    } else {
        return {n_size, m_size, src_stride,
                static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(dst_layout))};
    }
}

template <typename DstLayout, typename SrcLayout, typename SrcShape>
__aicore__ inline static constexpr l0c_out_copy_params
make_nc1hwc0_params(const DstLayout& dst_layout, const SrcLayout& src_layout, const SrcShape& src_shape)
{
    return {get_shape_columns(src_shape), get_shape_rows(src_shape),
            static_cast<uint32_t>(get_element<attr_info::stride, attr_info::column, 1>(src_layout) / fractal_fixed),
            static_cast<uint32_t>(get<1>(dst_layout.stride()))};
}

template <round_mode mode, typename DstType, typename SrcType>
__aicore__ inline constexpr QuantMode_t get_vector_quant_mode()
{
    if constexpr (is_one_of_attr_v<SrcType, int32_t> && is_one_of_attr_v<DstType, half>) {
        return QuantMode_t::VDEQF16;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, uint8_t, int8_t>) {
        return QuantMode_t::VQF322B8_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, int32_t> && is_one_of_attr_v<DstType, uint8_t, int8_t>) {
        return QuantMode_t::VREQ8;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, fp8_e4m3fn_t>) {
        return QuantMode_t::VQF322FP8_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, hifloat8_t>) {
        if constexpr (mode == round_mode::hybrid) {
            return QuantMode_t::VQF322HIF8_PRE_HYBRID;
        } else {
            return QuantMode_t::VQF322HIF8_PRE;
        }
    } else if constexpr (is_one_of_attr_v<SrcType, int32_t> && is_one_of_attr_v<DstType, bfloat16_t>) {
        return QuantMode_t::VQS322BF16_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, half>) {
        return QuantMode_t::VQF322F16_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, bfloat16_t>) {
        return QuantMode_t::VQF322BF16_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, float>) {
        return QuantMode_t::VQF322F32_PRE;
    }
}

template <round_mode mode, typename DstType, typename SrcType>
__aicore__ inline constexpr QuantMode_t get_scalar_quant_mode()
{
    if constexpr (is_one_of_attr_v<SrcType, int32_t> && is_one_of_attr_v<DstType, half>) {
        return QuantMode_t::DEQF16;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, uint8_t, int8_t>) {
        return QuantMode_t::QF322B8_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, int32_t> && is_one_of_attr_v<DstType, uint8_t, int8_t>) {
        return QuantMode_t::REQ8;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, fp8_e4m3fn_t>) {
        return QuantMode_t::QF322FP8_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, hifloat8_t>) {
        if constexpr (mode == round_mode::hybrid) {
            return QuantMode_t::QF322HIF8_PRE_HYBRID;
        } else {
            return QuantMode_t::QF322HIF8_PRE;
        }
    } else if constexpr (is_one_of_attr_v<SrcType, int32_t> && is_one_of_attr_v<DstType, bfloat16_t>) {
        return QuantMode_t::QS322BF16_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, half>) {
        return QuantMode_t::QF322F16_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, bfloat16_t>) {
        return QuantMode_t::QF322BF16_PRE;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, float>) {
        return QuantMode_t::QF322F32_PRE;
    }
}

template <round_mode mode, typename DstType, typename SrcType>
__aicore__ inline constexpr QuantMode_t get_cast_quant_mode()
{
    if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, half>) {
        return QuantMode_t::F322F16;
    } else if constexpr (is_one_of_attr_v<SrcType, float> && is_one_of_attr_v<DstType, bfloat16_t>) {
        return QuantMode_t::F322BF16;
    } else {
        return QuantMode_t::NoQuant;
    }
}

template <round_mode mode, typename DstTensor, typename SrcTensor, typename Quant = void>
__aicore__ inline constexpr QuantMode_t get_quant_mode()
{
    using src_type = typename SrcTensor::element_type;
    using dst_type = typename DstTensor::element_type;
    constexpr bool is_tensor = is_attr_tensor_v<Quant>;
    constexpr bool is_scalar = Std::is_same_v<Quant, uint64_t>;

    if constexpr (mode == round_mode::hybrid) {
        static_assert((is_one_of_attr_v<src_type, float> && is_one_of_attr_v<dst_type, hifloat8_t>),
                      "Only when L0CType is float and output Type is hifloat8_t support round_mode::hybrid in Fixpipe");
    }
    if constexpr (is_tensor) {
        return get_vector_quant_mode<mode, dst_type, src_type>();
    } else if constexpr (is_scalar) {
        return get_scalar_quant_mode<mode, dst_type, src_type>();
    } else {
        return get_cast_quant_mode<mode, dst_type, src_type>();
    }
}

class set_register_instr {
public:
    __aicore__ inline static void set_register(uint64_t quant, uint32_t nd_num, uint32_t dst_n_d_stride,
                                               uint32_t src_n_d_stride)
    {
        set_quant_pre(quant);
        set_loop3_para<uint64_t>(nd_num, dst_n_d_stride, src_n_d_stride);
    }

    __aicore__ inline static void set_register(uint64_t quant, uint32_t dn_num, uint32_t dst_d_n_stride,
                                               uint32_t src_n_z_matrix_stride, uint32_t src_n_z_c0_stride)
    {
        set_quant_pre(quant);
        set_loop3_para<uint64_t>(dn_num, dst_d_n_stride, src_n_z_matrix_stride);
        set_channel_para<uint64_t>(src_n_z_c0_stride);
    }

    __aicore__ inline static void set_register(uint32_t nd_num, uint32_t dst_n_d_stride, uint32_t src_n_d_stride)
    {
        set_loop3_para<uint64_t>(nd_num, dst_n_d_stride, src_n_d_stride);
    }

    __aicore__ inline static void set_register(uint32_t dn_num, uint32_t dst_d_n_stride, uint32_t src_n_z_matrix_stride,
                                               uint32_t src_n_z_c0_stride)
    {
        set_loop3_para<uint64_t>(dn_num, dst_d_n_stride, src_n_z_matrix_stride);
        set_channel_para<uint64_t>(src_n_z_c0_stride);
    }

private:
    static constexpr uint32_t shift_loop3_dst_stride = 32;
    static constexpr uint32_t shift_loop3_src_matrix = 16;
    static constexpr uint32_t shift_channel_c0_stride = 48;

    __aicore__ inline static void set_quant_pre(uint64_t quant)
    {
        asc_set_l0c_copy_prequant(quant);
    }

    template <typename RegisterType>
    __aicore__ inline static void set_loop3_para(uint32_t num, uint32_t dst_stride, uint32_t src_stride)
    {
        asc_set_l0c2gm_nz2nd(static_cast<RegisterType>(num), static_cast<RegisterType>(src_stride),
                             static_cast<RegisterType>(dst_stride));
    }

    template <typename RegisterType>
    __aicore__ inline static void set_channel_para(uint32_t src_n_z_c0_stride)
    {
        RegisterType channel_para = 0;
        channel_para |= static_cast<RegisterType>(src_n_z_c0_stride) << shift_channel_c0_stride;
        asc_set_l0c2gm_channel_para(channel_para);
    }
};

__aicore__ inline auto alloc_fixbuf_temp_buf(const uint16_t& /* cal_n_size */)
{
    return reinterpret_cast<uint64_t>(asc_get_phy_buf_addr(0));
}

template <typename DataType>
__aicore__ inline void set_qua(const __fbuf__ DataType* deq_tensor_temp_buf)
{
    uint64_t deq_tensor_addr = (reinterpret_cast<uint64_t>(deq_tensor_temp_buf) >> 7) << 8;
    asc_set_l0c_copy_prequant(deq_tensor_addr);
}

__aicore__ inline void insert_sync()
{
    asc_sync_pipe(PIPE_FIX);
}

template <typename DstTensor, typename SrcLayout>
__aicore__ inline static void emit_set_register(const SrcLayout& src_layout, uint32_t batch_num,
                                                uint32_t dst_batch_stride, uint32_t src_batch_stride)
{
    if constexpr (is_l0c_out_nd_format_v<DstTensor>) {
        set_register_instr::set_register(batch_num, dst_batch_stride, src_batch_stride);
    } else {
        set_register_instr::set_register(batch_num, dst_batch_stride, src_batch_stride,
                                         get_element<attr_info::stride, attr_info::column, 0>(src_layout));
    }
}

template <typename DstTensor, typename SrcLayout>
__aicore__ inline static void emit_set_register(const SrcLayout& src_layout, uint64_t quant, uint32_t batch_num,
                                                uint32_t dst_batch_stride, uint32_t src_batch_stride)
{
    if constexpr (is_l0c_out_nd_format_v<DstTensor>) {
        set_register_instr::set_register(quant, batch_num, dst_batch_stride, src_batch_stride);
    } else {
        set_register_instr::set_register(quant, batch_num, dst_batch_stride, src_batch_stride,
                                         get_element<attr_info::stride, attr_info::column, 0>(src_layout));
    }
}

template <typename DstTensor, typename SrcTensor>
__aicore__ inline static void set_register_impl(const DstTensor& dst, const SrcTensor& src)
{
    if constexpr (is_l0c_out_nd_format_v<DstTensor> || is_l0c_out_dn_format_v<DstTensor>) {
        if constexpr (is_l0c_out_src_batch_layout_v<SrcTensor>) {
            auto src_layout = src.layout();
            auto dst_layout = dst.layout();
            emit_set_register<DstTensor>(get<1>(src_layout), get<0>(src_layout.shape()), get<0>(dst_layout.stride()),
                                         get<0>(src_layout.stride()) / fractal_fixed);
        } else {
            emit_set_register<DstTensor>(src.layout(), 1, 0, 0);
        }
    } else {
        set_register_instr::set_register(1, 0, 0);
    }
}

template <typename DstTensor, typename SrcTensor>
__aicore__ inline static void set_register_impl(const DstTensor& dst, const SrcTensor& src, uint64_t quant)
{
    if constexpr (is_l0c_out_nd_format_v<DstTensor> || is_l0c_out_dn_format_v<DstTensor>) {
        if constexpr (is_l0c_out_src_batch_layout_v<SrcTensor>) {
            auto src_layout = src.layout();
            auto dst_layout = dst.layout();
            emit_set_register<DstTensor>(get<1>(src_layout), quant, get<0>(src_layout.shape()),
                                         get<0>(dst_layout.stride()), get<0>(src_layout.stride()) / fractal_fixed);
        } else {
            emit_set_register<DstTensor>(src.layout(), quant, 1, 0, 0);
        }
    } else {
        set_register_instr::set_register(quant, 1, 0, 0);
    }
}

// Validate the batch-axis consistency between dst/src/quant tensors used by L0C->GM/UB vector
// quant copy. Returns whether the quant tensor is batched so the caller can use the result for
// `if constexpr` dispatch without recomputing it.
template <typename DstTensor, typename SrcTensor, typename Quant>
__aicore__ inline static constexpr bool check_vector_quant_batch_consistency()
{
    constexpr bool src_batched = is_l0c_out_src_batch_layout_v<SrcTensor>;
    constexpr bool dst_batched = is_l0c_out_batched_layout_v<DstTensor>;
    constexpr bool quant_batched = is_l0c_out_batched_layout_v<Quant>;
    static_assert(src_batched == dst_batched, "src and dst tensors must both carry a batch axis or both omit it.");
    static_assert(!quant_batched || src_batched,
                  "Vector quant with batched quant tensor requires batched src and dst tensors as well.");
    return quant_batched;
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_UTILS_L0C2OUT_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
