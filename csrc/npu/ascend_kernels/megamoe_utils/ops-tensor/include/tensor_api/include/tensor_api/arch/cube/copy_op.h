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
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif

/*!
 * \file copy_op.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_CUBE_COPY_OP_H
#define INCLUDE_TENSOR_API_ARCH_CUBE_COPY_OP_H

#include "tensor_api/arch/cube/copy_trait.h"

namespace asc {
namespace te {

struct l0c_to_gm_params {
    __aicore__ constexpr l0c_to_gm_params() = default;

    __aicore__ constexpr l0c_to_gm_params(unit_flag_mode unit_flag) : unit_flag(unit_flag) {}

    unit_flag_mode unit_flag = unit_flag_mode::disable;
};

struct l0c_to_ub_params {
    __aicore__ constexpr l0c_to_ub_params() = default;

    __aicore__ constexpr l0c_to_ub_params(unit_flag_mode unit_flag, uint8_t sub_block_id = 0) :
        unit_flag(unit_flag), sub_block_id(sub_block_id)
    {}

    unit_flag_mode unit_flag = unit_flag_mode::disable;
    uint8_t sub_block_id = 0;
};

struct l0c_to_l1_params {
    __aicore__ constexpr l0c_to_l1_params() = default;

    __aicore__ constexpr l0c_to_l1_params(unit_flag_mode unit_flag) : unit_flag(unit_flag) {}

    unit_flag_mode unit_flag = unit_flag_mode::disable;
};

template <typename PadType>
struct img2col_params {
    uint16_t m_extension = 0;
    uint16_t k_extension = 0;
    uint16_t m_start_pos = 0;
    uint16_t k_start_pos = 0;
    uint8_t pad_list[4] = {0, 0, 0, 0};
    uint8_t stride_w = 1;
    uint8_t stride_h = 1;
    uint8_t filter_w = 1;
    uint8_t filter_h = 1;
    uint8_t dilation_filter_w = 1;
    uint8_t dilation_filter_h = 1;
    bool enable_filter_w_extend = false;
    bool enable_filter_h_extend = false;
    bool enable_transpose = false;
    bool enable_f_matrix_ctrl = false;
    PadType pad_value = 0;
};

struct copy_gm_to_l1 {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <const gm_to_l1_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                                 const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l1_to_ub {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <const l1_to_ub_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord, typename SrcCoord,
              typename CopyShape>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                                 const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l1_to_l0a {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);
};

struct copy_l1_to_l0b {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const l1_to_l0b_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void load_data(const DstTensor& dst, const SrcTensor& src);

    template <const l1_to_l0b_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void load_data(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                            const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l1_to_l0scalea {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const l1_to_l0scalea_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void load_data(const DstTensor& dst, const SrcTensor& src);

    template <const l1_to_l0scalea_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void load_data(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                            const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l1_to_l0scaleb {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void load_data(const DstTensor& dst, const SrcTensor& src);

    template <const l1_to_l0scaleb_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void load_data(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                            const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l1_to_fixbuf {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <const l1_to_fixbuf_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                                 const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l1_to_biastable {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);

private:
    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src);

    template <const l1_to_biastable_trait& trait, typename DstTensor, typename SrcTensor, typename DstCoord,
              typename SrcCoord, typename CopyShape>
    __aicore__ inline static void data_copy_impl(const DstTensor& dst, const SrcTensor& src, const DstCoord& dst_coord,
                                                 const SrcCoord& src_coord, const CopyShape& copy_shape);
};

struct copy_l0c_to_gm {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);
};

struct copy_l0c_to_ub {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);
};

struct copy_l0c_to_l1 {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);
};

} // namespace te
} // namespace asc

#include "impl/tensor_api/arch/cube/gm_to_l1/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_ub/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_l0a/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_l0b/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_l0scalea/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_l0scaleb/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_fb/copy.h"
#include "impl/tensor_api/arch/cube/l1_to_bt/copy.h"
#include "impl/tensor_api/arch/cube/l0c_to_gm/copy.h"
#include "impl/tensor_api/arch/cube/l0c_to_ub/copy.h"
#include "impl/tensor_api/arch/cube/l0c_to_l1/copy.h"

#endif // INCLUDE_TENSOR_API_ARCH_CUBE_COPY_OP_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC_TENSOR_API_H
#endif
