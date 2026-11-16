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
    "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/copy_common.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy_common.h
 * \brief Shared helpers for the gm_to_l1 copy implementations.
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_COPY_COMMON_H
#define IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_COPY_COMMON_H

#include "impl/tensor_api/arch/cube/gm_to_l1/copy_impl/instruction.h"

namespace asc {
namespace te {

// Shared GM->L1 batch dispatch for the four copy ops (ND2Nz/DN2Nz/ND2Zn/DN2Zn). The op-specific
// single-matrix parameter extraction lives in CopyOp::EmitCopy; this routine handles the parts that
// are identical across ops: depth-based batch detection, stripping the leading B axis, and reading
// the per-batch counts/strides. CopyOp must provide static check_template<trait,DstTensor,SrcTensor>() and
// emit_copy(dst, src, src_layout, dst_layout, matrix_num, src_matrix_stride, dst_matrix_stride).
//   - Non-batch (depth 2/4): pass the full layouts, matrix_num=1, strides=0.
//   - Batch (depth 3/5): strip B with remove_batch_dim (keeps pattern/trait), matrix_num/strides from
//     the B axis. The GM stride comes straight from the layout, so both bmk-contiguous and
//     mbk-non-contiguous memory are covered.
template <const gm_to_l1_trait& trait, typename CopyOp, typename DstTensor, typename SrcTensor>
__aicore__ inline void run_gm_to_l1_batched(const DstTensor& dst, const SrcTensor& src)
{
    CopyOp::template check_template<trait, DstTensor, SrcTensor>();
    constexpr auto src_depth = nesting_depth_v<decltype(src.layout().shape())>;
    if constexpr (src_depth == three_dim_data || src_depth == five_dim_data) {
        auto src_layout = src.layout();
        auto dst_layout = dst.layout();
        uint16_t matrix_num = get<0>(src_layout.shape());
        uint64_t src_matrix_stride = get<0>(src_layout.stride());
        uint32_t dst_matrix_stride = get<0>(dst_layout.stride());
        CopyOp::emit_copy(dst, src, remove_batch_dim(src_layout), remove_batch_dim(dst_layout), matrix_num,
                          src_matrix_stride, dst_matrix_stride);
    } else {
        CopyOp::emit_copy(dst, src, src.layout(), dst.layout(), 1, 0, 0);
    }
}

template <const gm_to_l1_trait& trait, typename CopyOp, typename DstTensor, typename SrcTensor, typename DstCoord,
          typename SrcCoord, typename CopyShape>
__aicore__ inline void run_gm_to_l1_batched(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                            const SrcCoord& src_coord, const CopyShape& copy_shape)
{
    CopyOp::template check_template<trait, DstTensor, SrcTensor>();
    auto src_shape = make_slice_shape(src_coord, src.layout(), copy_shape);
    auto dst_offset = dst.layout()(dst_coord);
    auto src_offset = src.layout()(src_coord);
    uint16_t matrix_num = get_shape_batch_size(src_shape);
    uint64_t src_matrix_stride = 0;
    uint32_t dst_matrix_stride = 0;
    constexpr auto src_depth = nesting_depth_v<decltype(src.layout().shape())>;
    if constexpr (src_depth == three_dim_data || src_depth == five_dim_data) {
        src_matrix_stride = get<0>(src.layout().stride());
        dst_matrix_stride = get<0>(dst.layout().stride());
    }
    // emit_copy receives matrix strides in element units and performs instruction-specific conversion.
    CopyOp::emit_copy(dst, src, src_shape, matrix_num, src_matrix_stride, dst_matrix_stride, dst_offset, src_offset);
}

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_CUBE_GM_TO_L1_COPY_IMPL_COPY_COMMON_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
