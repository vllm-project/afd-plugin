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
    "impl/tensor_api/arch/cube/l1_to_ub/copy_impl/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_L1_TO_UB_COPY_IMPL_DATA_COPY_H
#define IMPL_TENSOR_API_ARCH_CUBE_L1_TO_UB_COPY_IMPL_DATA_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/cube/l1_to_ub/copy_impl/instruction.h"

namespace asc {
namespace te {

class copy_l1_to_ub_nd {
public:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        execute<trait>(dst, src);
    }

    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint16_t block_count = get_shape_rows(src_shape);
        auto columns = get_shape_columns(src_shape);
        uint32_t block_len = Std::ceil_division(columns, c0_element<src_type>);
        uint32_t src_row_stride;
        uint32_t dst_row_stride;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, nd_layout_ptn>) {
            src_row_stride = get_element<attr_info::stride, attr_info::row>(src.layout());
            dst_row_stride = get_element<attr_info::stride, attr_info::row>(dst.layout());
        } else {
            src_row_stride = get_element<attr_info::stride, attr_info::row, 1>(src.layout());
            dst_row_stride = get_element<attr_info::stride, attr_info::row, 1>(dst.layout());
        }
        int64_t src_stride = Std::ceil_division(src_row_stride - columns, c0_element<src_type>);
        int64_t dst_stride = Std::ceil_division(dst_row_stride - columns, c0_element<dst_type>);
        copy_l1_to_ub_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, block_count, block_len, src_stride,
                                                   dst_stride);
    }

private:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void execute(const DstTensor& dst, const SrcTensor& src)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint16_t block_count = get_total_row_shape(src_layout);
        uint32_t block_len = Std::ceil_division(get_total_column_shape(src_layout), c0_element<src_type>);

        uint32_t src_row_stride;
        uint32_t dst_row_stride;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, nd_layout_ptn>) {
            src_row_stride = get_element<attr_info::stride, attr_info::row>(src_layout);
        } else {
            src_row_stride = get_element<attr_info::stride, attr_info::row, 1>(src_layout);
        }
        if constexpr (is_satisfied_ptn_format_v<DstTensor, nd_layout_ptn>) {
            dst_row_stride = get_element<attr_info::stride, attr_info::row>(dst_layout);
        } else {
            dst_row_stride = get_element<attr_info::stride, attr_info::row, 1>(dst_layout);
        }
        int64_t src_stride =
            Std::ceil_division(src_row_stride - get_total_column_shape(src_layout), c0_element<src_type>);
        int64_t dst_stride =
            Std::ceil_division(dst_row_stride - get_total_column_shape(src_layout), c0_element<dst_type>);

        copy_l1_to_ub_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, src_stride,
                                       dst_stride);
    }
};

class copy_l1_to_ub_dn {
public:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        execute<trait>(dst, src);
    }

    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        auto rows = get_shape_rows(src_shape);
        uint16_t block_count = get_shape_columns(src_shape);
        uint32_t block_len = Std::ceil_division(rows, c0_element<src_type>);
        uint32_t src_column_stride;
        uint32_t dst_column_stride;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, dn_layout_ptn>) {
            src_column_stride = get_element<attr_info::stride, attr_info::column>(src.layout());
            dst_column_stride = get_element<attr_info::stride, attr_info::column>(dst.layout());
        } else {
            src_column_stride = get_element<attr_info::stride, attr_info::column, 1>(src.layout());
            dst_column_stride = get_element<attr_info::stride, attr_info::column, 1>(dst.layout());
        }
        int64_t src_stride = Std::ceil_division(src_column_stride - rows, c0_element<src_type>);
        int64_t dst_stride = Std::ceil_division(dst_column_stride - rows, c0_element<dst_type>);
        copy_l1_to_ub_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, block_count, block_len, src_stride,
                                                   dst_stride);
    }

private:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void execute(const DstTensor& dst, const SrcTensor& src)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint16_t block_count = get_total_column_shape(src_layout);
        uint32_t block_len = Std::ceil_division(get_total_row_shape(src_layout), c0_element<src_type>);

        uint32_t src_column_stride;
        uint32_t dst_column_stride;
        if constexpr (is_satisfied_ptn_format_v<SrcTensor, dn_layout_ptn>) {
            src_column_stride = get_element<attr_info::stride, attr_info::column>(src_layout);
        } else {
            src_column_stride = get_element<attr_info::stride, attr_info::column, 1>(src_layout);
        }
        if constexpr (is_satisfied_ptn_format_v<DstTensor, dn_layout_ptn>) {
            dst_column_stride = get_element<attr_info::stride, attr_info::column>(dst_layout);
        } else {
            dst_column_stride = get_element<attr_info::stride, attr_info::column, 1>(dst_layout);
        }
        int64_t src_stride =
            Std::ceil_division(src_column_stride - get_total_row_shape(src_layout), c0_element<src_type>);
        int64_t dst_stride =
            Std::ceil_division(dst_column_stride - get_total_row_shape(src_layout), c0_element<dst_type>);

        copy_l1_to_ub_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, src_stride,
                                       dst_stride);
    }
};

class copy_l1_to_ub_nz {
public:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src)
    {
        execute<trait>(dst, src);
    }

    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void run(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                      const SrcCoord& src_coord, const CopyShape& copy_shape)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
        auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
        auto dst_offset = dst.layout()(dst_coord);
        auto src_offset = src.layout()(src_coord);
        uint16_t block_count = get<1, 1>(src_shape);
        uint32_t block_len = get_shape_rows(src_shape);
        int64_t src_stride =
            get_element<attr_info::stride, attr_info::column, 1>(src.layout()) / c0_element<src_type> - block_len;
        int64_t dst_stride =
            get_element<attr_info::stride, attr_info::column, 1>(dst.layout()) / c0_element<dst_type> - block_len;
        copy_l1_to_ub_instr::data_copy_with_offset(dst, src, dst_offset, src_offset, block_count, block_len, src_stride,
                                                   dst_stride);
    }

private:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void execute(const DstTensor& dst, const SrcTensor& src)
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;

        auto dst_layout = dst.layout();
        auto src_layout = src.layout();

        uint16_t block_count = get_element<attr_info::shape, attr_info::column, 1>(src_layout);
        uint32_t block_len = get_total_row_shape(src_layout);
        int64_t src_stride =
            get_element<attr_info::stride, attr_info::column, 1>(src_layout) / c0_element<src_type> - block_len;
        int64_t dst_stride =
            get_element<attr_info::stride, attr_info::column, 1>(dst_layout) / c0_element<dst_type> - block_len;

        copy_l1_to_ub_instr::data_copy(dst.data().get(), src.data().get(), block_count, block_len, src_stride,
                                       dst_stride);
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_L1_TO_UB_COPY_IMPL_DATA_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
