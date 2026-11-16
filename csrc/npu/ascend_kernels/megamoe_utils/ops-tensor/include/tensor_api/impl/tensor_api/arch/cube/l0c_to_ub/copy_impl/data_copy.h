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
    "impl/tensor_api/arch/cube/l0c_to_ub/copy_impl/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_UB_COPY_IMPL_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_UB_COPY_IMPL_DATA_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/cube/l1_to_fb/copy.h"
#include "impl/tensor_api/arch/cube/utils/l0c2out_utils.h"
#include "impl/tensor_api/arch/cube/l0c_to_ub/copy_impl/instruction.h"

namespace asc {
namespace te {

__aicore__ constexpr l0c_to_ub_trait::l0c_to_ub_trait(asc::te::round_mode round_mode, bool enable_relu,
                                                      bool enable_channel_split, dual_dst_mode dual_dst_ctl) :
    round_mode(round_mode), enable_relu(enable_relu), enable_channel_split(enable_channel_split),
    dual_dst_ctl(dual_dst_ctl)
{}

class data_copy_l0c_to_ub_no_vector_quant {
public:
    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src,
                                                 const l0c_to_ub_params& params)
    {
        if constexpr (is_l0c_out_src_batch_layout_v<SrcTensor> && !is_l0c_out_batch_nz2nz_v<DstTensor, SrcTensor>) {
            emit_data_copy<trait, quant_pre>(dst, src, get<1>(dst.layout()), get<1>(src.layout()), params);
        } else {
            emit_data_copy<trait, quant_pre>(dst, src, dst.layout(), src.layout(), params);
        }
    }

    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename DstCoord, typename SrcCoord, typename CopyShape>
    __aicore__ inline static void data_copy_with_offset(const DstTensor& dst, const SrcTensor& src,
                                                        const DstCoord& dst_coord, const SrcCoord& src_coord,
                                                        const CopyShape& copy_shape, const l0c_to_ub_params& params)
    {
        auto dst_shape = make_slice_shape(dst_coord, dst.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        auto copy_params = make_l0c_out_copy_params<DstTensor, SrcTensor>(dst.layout(), src.layout(), dst_shape);
        constexpr bool is_nd_format = is_l0c_out_nd_format_v<DstTensor>;
        constexpr bool is_dn_format = is_l0c_out_dn_format_v<DstTensor>;
        copy_l0c_to_ub_instr::data_copy_with_offset<quant_pre, DstTensor, SrcTensor>(
            dst, src, dst_offset, src_offset, copy_params.n_size, copy_params.m_size, copy_params.src_stride,
            copy_params.dst_stride, static_cast<uint8_t>(trait.dual_dst_ctl), trait.enable_relu,
            static_cast<uint8_t>(params.unit_flag), params.sub_block_id, trait.enable_channel_split, is_nd_format,
            is_dn_format);
    }

private:
    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename DstLayout, typename SrcLayout>
    __aicore__ inline static void emit_data_copy(const DstTensor& dst, const SrcTensor& src,
                                                 const DstLayout& dst_layout, const SrcLayout& src_layout,
                                                 const l0c_to_ub_params& params)
    {
        constexpr bool is_nd_format = is_l0c_out_nd_format_v<DstTensor>;
        constexpr bool is_dn_format = is_l0c_out_dn_format_v<DstTensor>;
        auto copy_params = make_l0c_out_copy_params<DstTensor, SrcTensor>(dst_layout, src_layout);

        bool relu_en = trait.enable_relu;
        uint8_t unit_flag = static_cast<uint8_t>(params.unit_flag);
        bool nz2nd_en = is_nd_format;
        bool nz2dn_en = is_dn_format;

        bool is_channel_split = trait.enable_channel_split;

        uint8_t dual_dst_ctl = static_cast<uint8_t>(trait.dual_dst_ctl);
        uint8_t sub_block_id = params.sub_block_id;
        copy_l0c_to_ub_instr::data_copy<quant_pre>(dst.data().get(), src.data().get(), copy_params.n_size,
                                                   copy_params.m_size, copy_params.src_stride, copy_params.dst_stride,
                                                   dual_dst_ctl, relu_en, unit_flag, sub_block_id, is_channel_split,
                                                   nz2nd_en, nz2dn_en);
    }
};

class data_copy_l0c_to_ub_vector_quant {
public:
    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename Quant>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src, const Quant& quant,
                                                 const l0c_to_ub_params& params)
    {
        if constexpr (is_l0c_out_batch_nz2nz_v<DstTensor, SrcTensor>) {
            emit_batch_nz2_nz_data_copy<trait, quant_pre>(dst, src, quant, params);
        } else if constexpr (is_l0c_out_src_batch_layout_v<SrcTensor>) {
            emit_data_copy<trait, quant_pre>(dst, src, quant, get<1>(dst.layout()), get<1>(src.layout()), params);
        } else {
            emit_data_copy<trait, quant_pre>(dst, src, quant, dst.layout(), src.layout(), params);
        }
    }

    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename Quant, typename DstCoord, typename SrcCoord, typename CopyShape>
    __aicore__ inline static void data_copy_with_offset(const DstTensor& dst, const SrcTensor& src, const Quant& quant,
                                                        const DstCoord& dst_coord, const SrcCoord& src_coord,
                                                        const CopyShape& copy_shape, const l0c_to_ub_params& params)
    {
        auto dst_shape = make_slice_shape(dst_coord, dst.layout(), copy_shape);
        auto base_dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        auto copy_params = make_l0c_out_copy_params<DstTensor, SrcTensor>(dst.layout(), src.layout(), dst_shape);
        uint16_t n_iter_num = Std::ceil_division(copy_params.n_size, main_loop_n_size);
        for (uint16_t i = 0; i < n_iter_num; ++i) {
            uint32_t n_offset = i * main_loop_n_size;
            uint32_t cal_n_size = Std::min(copy_params.n_size - n_offset, main_loop_n_size);
            copy_l1_to_fixbuf(quant, cal_n_size, i);
            insert_sync();
            auto src_coord = make_src_coord<SrcTensor>(i * cburst_num);
            auto dst_coord = make_dst_coord<DstTensor>(n_offset);
            auto dst_offset = base_dst_offset + dst.layout()(dst_coord);
            copy_l0c_to_ub_instr::data_copy_with_offset<quant_pre>(
                dst, src(src_coord), dst_offset, src_offset, cal_n_size, copy_params.m_size, copy_params.src_stride,
                copy_params.dst_stride, static_cast<uint8_t>(trait.dual_dst_ctl), trait.enable_relu,
                static_cast<uint8_t>(params.unit_flag), params.sub_block_id, trait.enable_channel_split,
                is_l0c_out_nd_format_v<DstTensor>, is_l0c_out_dn_format_v<DstTensor>);
        }
    }

private:
    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename Quant>
    __aicore__ inline static void emit_batch_nz2_nz_data_copy(const DstTensor& dst, const SrcTensor& src,
                                                              const Quant& quant, const l0c_to_ub_params& params)
    {
        auto dst_layout = dst.layout();
        auto src_layout = src.layout();
        for (uint32_t batch_index = 0; batch_index < get<0>(src_layout.shape()); ++batch_index) {
            emit_data_copy<trait, quant_pre>(
                dst(make_coord(batch_index, make_coord(make_coord(0, 0), make_coord(0, 0)))),
                src(make_coord(batch_index, make_coord(make_coord(0, 0), make_coord(0, 0)))), quant, get<1>(dst_layout),
                get<1>(src_layout), params);
        }
    }

    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename Quant, typename DstLayout, typename SrcLayout>
    __aicore__ inline static void emit_data_copy(const DstTensor& dst, const SrcTensor& src, const Quant& quant,
                                                 const DstLayout& dst_layout, const SrcLayout& src_layout,
                                                 const l0c_to_ub_params& params)
    {
        uint32_t n_size = Std::min(get_total_column_shape(src_layout), get_total_column_shape(dst_layout));
        uint16_t n_iter_num = 1;
        uint32_t cal_n_size = n_size;
        uint32_t tail_n_size = 0;
        if (cal_n_size > main_loop_n_size) {
            n_iter_num = n_size / main_loop_n_size;
            tail_n_size = n_size % main_loop_n_size;
            cal_n_size = main_loop_n_size;
        }
        execute_data_copy<trait, quant_pre>(dst, src, quant, n_iter_num, cal_n_size, tail_n_size, dst_layout,
                                            src_layout, params);
    }

    template <const l0c_to_ub_trait& trait, typename DstTensor, bool is_tail, typename DstLayout, typename SrcLayout>
    __aicore__ inline static auto generate_params(const DstLayout& dst_layout, const SrcLayout& src_layout,
                                                  const l0c_to_ub_params& params)
    {
        constexpr bool is_nd_format = is_l0c_out_nd_format_v<DstTensor>;
        constexpr bool is_dn_format = is_l0c_out_dn_format_v<DstTensor>;

        uint32_t n_size = Std::min(get_total_column_shape(src_layout), get_total_column_shape(dst_layout));
        uint32_t m_size = Std::min(get_total_row_shape(src_layout), get_total_row_shape(dst_layout));
        if constexpr (is_tail) {
            n_size %= main_loop_n_size;
        } else {
            if (n_size > main_loop_n_size) {
                n_size = main_loop_n_size;
            }
        }

        const uint32_t src_stride = get_element<attr_info::stride, attr_info::column, 1>(src_layout) / fractal_fixed;
        uint32_t dst_stride = 0;
        if constexpr (is_nd_format) {
            dst_stride = get_l0c_out_nd_stride<DstTensor>(dst_layout);
        } else if constexpr (is_dn_format) {
            dst_stride = get_l0c_out_dn_stride<DstTensor>(dst_layout);
        } else {
            dst_stride = get_element<attr_info::stride, attr_info::column, 1>(dst_layout);
        }

        const bool relu_en = trait.enable_relu;
        const uint8_t unit_flag = static_cast<uint8_t>(params.unit_flag);

        constexpr bool nz2nd_en = is_nd_format;
        constexpr bool nz2dn_en = is_dn_format;

        const bool channel_split = trait.enable_channel_split;
        uint8_t sub_block_id = params.sub_block_id;
        uint8_t dual_dst_ctl = static_cast<uint8_t>(trait.dual_dst_ctl);

        return Std::make_tuple(n_size, m_size, src_stride, dst_stride, dual_dst_ctl, relu_en, unit_flag, sub_block_id,
                               channel_split, nz2nd_en, nz2dn_en);
    }

    template <typename SrcTensor>
    __aicore__ inline static void copy_l1_to_fixbuf(const SrcTensor& src, uint16_t cal_n_size, uint16_t n_iter_index)
    {
        auto dst_addr = reinterpret_cast<__fbuf__ uint64_t*>(alloc_fixbuf_temp_buf(cal_n_size));
        auto dst = make_tensor(make_mem_ptr<location::fixbuf>(dst_addr), src.layout());
        auto coord = make_coord(_0{}, n_iter_index * main_loop_n_size);
        auto shape = make_shape(_1{}, cal_n_size);
        auto tile_src = src.slice(coord, shape);
        copy_l1_to_fixbuf_nd::run<l1_to_fixbuf_trait_default::value>(dst, tile_src);
        set_qua(dst_addr);
    }

    template <QuantMode_t quant_pre, typename DstTensor, typename SrcTensor, typename ParamTuple, size_t... Is>
    __aicore__ inline static void data_copy_wrapper(const DstTensor& dst, const SrcTensor& src,
                                                    const ParamTuple& param_tuple, Std::index_sequence<Is...>)
    {
        copy_l0c_to_ub_instr::data_copy<quant_pre>(dst.data().get(), src.data().get(), Std::get<Is>(param_tuple)...);
    }

    template <typename DstTensor>
    __aicore__ inline static constexpr auto make_dst_coord(uint32_t n_offset)
    {
        using layout_type = typename DstTensor::layout_type;
        if constexpr (layout_type::depth == five_dim_data) {
            return make_coord(_0{}, make_coord(make_coord(0, 0), make_coord(0, n_offset)));
        } else if constexpr (layout_type::depth == four_dim_data) {
            return make_coord(make_coord(0, 0), make_coord(0, n_offset));
        } else if constexpr (layout_type::depth == three_dim_data) {
            return make_coord(_0{}, make_coord(0, n_offset));
        } else {
            static_assert(layout_type::depth == two_dim_data, "Only support two-dim or four-dim dst tensor.");
            return make_coord(0, n_offset);
        }
    }

    template <typename SrcTensor>
    __aicore__ inline static constexpr auto make_src_coord(uint32_t n_offset)
    {
        using layout_type = typename SrcTensor::layout_type;
        if constexpr (layout_type::depth == five_dim_data) {
            return make_coord(_0{}, make_coord(make_coord(0, 0), make_coord(0, n_offset)));
        } else {
            static_assert(layout_type::depth == four_dim_data, "Only support four-dim or five-dim src tensor.");
            return make_coord(make_coord(0, 0), make_coord(0, n_offset));
        }
    }

    template <const l0c_to_ub_trait& trait, QuantMode_t quant_pre, typename DstTensor, typename SrcTensor,
              typename Quant, typename DstLayout, typename SrcLayout>
    __aicore__ inline static void execute_data_copy(const DstTensor& dst, const SrcTensor& src, const Quant& quant,
                                                    uint16_t n_iter_num, uint32_t cal_n_size, uint32_t tail_n_size,
                                                    const DstLayout& dst_layout, const SrcLayout& src_layout,
                                                    const l0c_to_ub_params& params)
    {
        const auto main_loop_param = generate_params<trait, DstTensor, false>(dst_layout, src_layout, params);

        for (uint16_t i = 0; i < n_iter_num; ++i) {
            copy_l1_to_fixbuf(quant, cal_n_size, i);
            insert_sync();

            const auto src_coord = make_src_coord<SrcTensor>(i * cburst_num);
            const auto dst_coord = make_dst_coord<DstTensor>(i * main_loop_n_size);

            data_copy_wrapper<quant_pre>(dst(dst_coord), src(src_coord), main_loop_param,
                                         Std::make_index_sequence<Std::tuple_size_v<decltype(main_loop_param)>>{});
        }

        if (tail_n_size) {
            const auto tail_param = generate_params<trait, DstTensor, true>(dst_layout, src_layout, params);

            copy_l1_to_fixbuf(quant, tail_n_size, n_iter_num);
            insert_sync();

            const auto src_coord = make_src_coord<SrcTensor>(n_iter_num * cburst_num);
            const auto dst_coord = make_dst_coord<DstTensor>(n_iter_num * main_loop_n_size);

            data_copy_wrapper<quant_pre>(dst(dst_coord), src(src_coord), tail_param,
                                         Std::make_index_sequence<Std::tuple_size_v<decltype(tail_param)>>{});
        }
    }
};

class data_copy_l0c_to_ub {
public:
    template <const l0c_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const l0c_to_ub_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor>();
        check_data_type::check_l0c_to_ub_data_type<quant_pre, DstTensor, SrcTensor>();
        set_register_impl<DstTensor, SrcTensor>(dst, src);
        data_copy_l0c_to_ub_no_vector_quant::data_copy_impl<trait, quant_pre, DstTensor, SrcTensor>(dst, src, params);
    }

    template <const l0c_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape,
                                      const l0c_to_ub_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor>();
        check_data_type::check_l0c_to_ub_data_type<quant_pre, DstTensor, SrcTensor>();
        set_register_impl<DstTensor, SrcTensor>(dst, src);
        data_copy_l0c_to_ub_no_vector_quant::data_copy_with_offset<trait, quant_pre>(dst, src, dst_coord, src_coord,
                                                                                     copy_shape, params);
    }

    template <const l0c_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, uint64_t quant,
                                      const l0c_to_ub_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor, uint64_t>();
        set_register_impl<DstTensor, SrcTensor>(dst, src, quant);
        data_copy_l0c_to_ub_no_vector_quant::data_copy_impl<trait, quant_pre, DstTensor, SrcTensor>(dst, src, params);
    }

    template <const l0c_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, uint64_t quant,
                                      const DstCoord& dst_coord, const SrcCoord& src_coord, const CopyShape& copy_shape,
                                      const l0c_to_ub_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor, uint64_t>();
        set_register_impl<DstTensor, SrcTensor>(dst, src, quant);
        data_copy_l0c_to_ub_no_vector_quant::data_copy_with_offset<trait, quant_pre>(dst, src, dst_coord, src_coord,
                                                                                     copy_shape, params);
    }

    template <const l0c_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename Quant>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const Quant& quant,
                                      const l0c_to_ub_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor, Quant>();
        constexpr bool quant_batched = check_vector_quant_batch_consistency<DstTensor, SrcTensor, Quant>();

        if constexpr (quant_batched) {
            // Fixpipe hardware reads a single quant slice from FB per instruction and reuses it
            // across the nd_num batch repetition; the FB quant address does not auto-step per
            // batch. To honor per-batch quant we must split on the host: each batch issues its
            // own fixpipe instruction with its own L1->FB quant copy.
            uint32_t batch_size = get<0>(src.layout().shape());
            for (uint32_t b = 0; b < batch_size; ++b) {
                auto sub_dst = make_single_batch_sub_tensor(dst, b);
                auto sub_src = make_single_batch_sub_tensor(src, b);
                auto sub_quant = make_single_batch_sub_tensor(quant, b);
                set_register_impl<decltype(sub_dst), decltype(sub_src)>(sub_dst, sub_src);
                data_copy_l0c_to_ub_vector_quant::data_copy_impl<trait, quant_pre, decltype(sub_dst), decltype(sub_src),
                                                                 decltype(sub_quant)>(sub_dst, sub_src, sub_quant,
                                                                                      params);
            }
        } else {
            set_register_impl<DstTensor, SrcTensor>(dst, src);
            data_copy_l0c_to_ub_vector_quant::data_copy_impl<trait, quant_pre, DstTensor, SrcTensor, Quant>(
                dst, src, quant, params);
        }
    }

    template <const l0c_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename Quant, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const Quant& quant,
                                      const DstCoord& dst_coord, const SrcCoord& src_coord, const CopyShape& copy_shape,
                                      const l0c_to_ub_params& params)
    {
        constexpr QuantMode_t quant_pre = get_quant_mode<trait.round_mode, DstTensor, SrcTensor, Quant>();
        constexpr bool quant_batched = check_vector_quant_batch_consistency<DstTensor, SrcTensor, Quant>();
        if constexpr (quant_batched) {
            auto dst_shape = make_slice_shape(dst_coord, dst.layout(), copy_shape);
            uint32_t batch_size = get_shape_batch_size(dst_shape);
            for (uint32_t b = 0; b < batch_size; ++b) {
                auto sub_dst = make_single_batch_sub_tensor(dst, get<0>(dst_coord) + b);
                auto sub_src = make_single_batch_sub_tensor(src, get<0>(src_coord) + b);
                auto sub_quant = make_single_batch_sub_tensor(quant, b);
                set_register_impl<decltype(sub_dst), decltype(sub_src)>(sub_dst, sub_src);
                data_copy_l0c_to_ub_vector_quant::data_copy_with_offset<trait, quant_pre>(
                    sub_dst, sub_src, sub_quant, get<1>(dst_coord), get<1>(src_coord), get<1>(copy_shape), params);
            }
        } else {
            set_register_impl<DstTensor, SrcTensor>(dst, src);
            data_copy_l0c_to_ub_vector_quant::data_copy_with_offset<trait, quant_pre>(dst, src, quant, dst_coord,
                                                                                      src_coord, copy_shape, params);
        }
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L0C_TO_UB_COPY_IMPL_DATA_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
