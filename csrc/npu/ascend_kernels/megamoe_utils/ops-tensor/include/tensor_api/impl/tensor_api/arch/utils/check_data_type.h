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
    "impl/tensor_api/arch/utils/check_data_type.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file check_data_type.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_CHECK_DATA_TYPE_H
#define IMPL_TENSOR_API_ARCH_UTILS_CHECK_DATA_TYPE_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/utils/is_format.h"

namespace asc {
namespace te {

class check_data_type {
public:
    template <typename DstTensor, typename FmTensor, typename FilterTensor>
    __aicore__ inline static constexpr void check_mx_mmad_data_type()
    {
        using dst_data_type = typename DstTensor::element_type;
        using fm_data_type = typename FmTensor::element_type;
        using filter_data_type = typename FilterTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dst_data_type, fm_data_type, filter_data_type>,
                                       Std::tuple<__cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
                                       Std::tuple<__cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
                                       Std::tuple<__cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
                                       Std::tuple<__cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>,
                      "The data type is not supported for L0C position.");
#endif
    }

    template <typename DstTensor, typename FmTensor, typename FilterTensor, typename BiasTensor>
    __aicore__ inline static constexpr void check_mx_mmad_bias_data_type()
    {
        using dst_data_type = typename DstTensor::element_type;
        using bias_data_type = typename BiasTensor::element_type;
        using fm_data_type = typename FmTensor::element_type;
        using filter_data_type = typename FilterTensor::element_type;
        using bias_pos = get_mem_location<BiasTensor>;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if constexpr (Std::is_same_v<bias_pos, location::bias>) {
            static_assert(
                Std::is_one_of_v<Std::tuple<bias_data_type, dst_data_type, fm_data_type, filter_data_type>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>,
                "The data type is not supported for BIAS position.");
        } else if constexpr (Std::is_same_v<bias_pos, location::l0c>) {
            static_assert(
                Std::is_one_of_v<Std::tuple<bias_data_type, dst_data_type, fm_data_type, filter_data_type>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>,
                "The data type is not supported for L0C position.");
        }
#endif
    }

    template <typename DstTensor, typename FmTensor, typename FilterTensor>
    __aicore__ inline static constexpr void check_mmad_data_type()
    {
        using dst_data_type = typename DstTensor::element_type;
        using fm_data_type = typename FmTensor::element_type;
        using filter_data_type = typename FilterTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dst_data_type, fm_data_type, filter_data_type>,
                                       Std::tuple<__cc__ int32_t, __ca__ int8_t, __cb__ int8_t>,
                                       Std::tuple<__cc__ float, __ca__ half, __cb__ half>,
                                       Std::tuple<__cc__ float, __ca__ float, __cb__ float>,
                                       Std::tuple<__cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                                       Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>,
                                       Std::tuple<__cc__ float, __ca__ hifloat8_t, __cb__ hifloat8_t>>,
                      "The data type is not supported for L0C position.");
#endif
    }

    template <typename DstTensor, typename FmTensor, typename FilterTensor, typename BiasTensor>
    __aicore__ inline static constexpr void check_mmad_bias_data_type()
    {
        using dst_data_type = typename DstTensor::element_type;
        using fm_data_type = typename FmTensor::element_type;
        using filter_data_type = typename FilterTensor::element_type;
        using bias_data_type = typename BiasTensor::element_type;
        using bias_pos = get_mem_location<BiasTensor>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if constexpr (Std::is_same_v<bias_pos, location::bias>) {
            static_assert(
                Std::is_one_of_v<Std::tuple<bias_data_type, dst_data_type, fm_data_type, filter_data_type>,
                                 Std::tuple<__biasbuf__ int32_t, __cc__ int32_t, __ca__ int8_t, __cb__ int8_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ half, __cb__ half>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ float, __cb__ float>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>,
                                 Std::tuple<__biasbuf__ float, __cc__ float, __ca__ hifloat8_t, __cb__ hifloat8_t>>,
                "The data type is not supported for BIAS position.");
        } else if constexpr (Std::is_same_v<bias_pos, location::l0c>) {
            static_assert(
                Std::is_one_of_v<Std::tuple<bias_data_type, dst_data_type, fm_data_type, filter_data_type>,
                                 Std::tuple<__cc__ int32_t, __cc__ int32_t, __ca__ int8_t, __cb__ int8_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ half, __cb__ half>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ float, __cb__ float>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>,
                                 Std::tuple<__cc__ float, __cc__ float, __ca__ hifloat8_t, __cb__ hifloat8_t>>,
                "The data type is not supported for L0C position.");
        }
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_gm_to_l1_data_type()
    {
        using dst_data_type = typename DstTensor::element_type;
        using src_data_type = typename SrcTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<
                          Std::tuple<dst_data_type, src_data_type>, Std::tuple<__cbuf__ half, __gm__ half>,
                          Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ float, __gm__ float>,
                          Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>,
                          Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>,
                          Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>,
                          Std::tuple<__cbuf__ fp8_e5m2_t, __gm__ fp8_e5m2_t>,
                          Std::tuple<__cbuf__ fp8_e4m3fn_t, __gm__ fp8_e4m3fn_t>,
                          Std::tuple<__cbuf__ hifloat8_t, __gm__ hifloat8_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_gm_to_l1_fp4_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<
                          Std::tuple<dst_data_type, src_data_type>, Std::tuple<__cbuf__ half, __gm__ half>,
                          Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ float, __gm__ float>,
                          Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>,
                          Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>,
                          Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>,
                          Std::tuple<__cbuf__ fp4x2_e1m2_t, __gm__ fp4x2_e1m2_t>,
                          Std::tuple<__cbuf__ fp4x2_e2m1_t, __gm__ fp4x2_e2m1_t>,
                          Std::tuple<__cbuf__ fp8_e5m2_t, __gm__ fp8_e5m2_t>,
                          Std::tuple<__cbuf__ fp8_e4m3fn_t, __gm__ fp8_e4m3fn_t>,
                          Std::tuple<__cbuf__ hifloat8_t, __gm__ hifloat8_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_gm_to_l1_scale_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dst_data_type, src_data_type>,
                                       Std::tuple<__cbuf__ fp8_e8m0_t, __gm__ fp8_e8m0_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_gm_to_l1_align_v2_nd_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<
                          Std::tuple<dst_data_type, src_data_type>, Std::tuple<__cbuf__ half, __gm__ half>,
                          Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ float, __gm__ float>,
                          Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>,
                          Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>,
                          Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>,
                          Std::tuple<__cbuf__ int64_t, __gm__ int64_t>, Std::tuple<__cbuf__ uint64_t, __gm__ uint64_t>,
                          Std::tuple<__cbuf__ fp4x2_e1m2_t, __gm__ fp4x2_e1m2_t>,
                          Std::tuple<__cbuf__ fp4x2_e2m1_t, __gm__ fp4x2_e2m1_t>,
                          Std::tuple<__cbuf__ fp8_e5m2_t, __gm__ fp8_e5m2_t>,
                          Std::tuple<__cbuf__ fp8_e4m3fn_t, __gm__ fp8_e4m3fn_t>,
                          Std::tuple<__cbuf__ hifloat8_t, __gm__ hifloat8_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_gm_to_ub_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<
                          Std::tuple<dst_data_type, src_data_type>, Std::tuple<__ubuf__ half, __gm__ half>,
                          Std::tuple<__ubuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__ubuf__ float, __gm__ float>,
                          Std::tuple<__ubuf__ int8_t, __gm__ int8_t>, Std::tuple<__ubuf__ uint8_t, __gm__ uint8_t>,
                          Std::tuple<__ubuf__ int16_t, __gm__ int16_t>, Std::tuple<__ubuf__ uint16_t, __gm__ uint16_t>,
                          Std::tuple<__ubuf__ int32_t, __gm__ int32_t>, Std::tuple<__ubuf__ uint32_t, __gm__ uint32_t>,
                          Std::tuple<__ubuf__ int64_t, __gm__ int64_t>, Std::tuple<__ubuf__ uint64_t, __gm__ uint64_t>,
                          Std::tuple<__ubuf__ fp4x2_e1m2_t, __gm__ fp4x2_e1m2_t>,
                          Std::tuple<__ubuf__ fp4x2_e2m1_t, __gm__ fp4x2_e2m1_t>,
                          Std::tuple<__ubuf__ fp8_e5m2_t, __gm__ fp8_e5m2_t>,
                          Std::tuple<__ubuf__ fp8_e4m3fn_t, __gm__ fp8_e4m3fn_t>,
                          Std::tuple<__ubuf__ hifloat8_t, __gm__ hifloat8_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_ub_to_ub_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_one_of_v<
                Std::tuple<dst_data_type, src_data_type>, Std::tuple<__ubuf__ half, __ubuf__ half>,
                Std::tuple<__ubuf__ bfloat16_t, __ubuf__ bfloat16_t>, Std::tuple<__ubuf__ float, __ubuf__ float>,
                Std::tuple<__ubuf__ int8_t, __ubuf__ int8_t>, Std::tuple<__ubuf__ uint8_t, __ubuf__ uint8_t>,
                Std::tuple<__ubuf__ int16_t, __ubuf__ int16_t>, Std::tuple<__ubuf__ uint16_t, __ubuf__ uint16_t>,
                Std::tuple<__ubuf__ int32_t, __ubuf__ int32_t>, Std::tuple<__ubuf__ uint32_t, __ubuf__ uint32_t>,
                Std::tuple<__ubuf__ int64_t, __ubuf__ int64_t>, Std::tuple<__ubuf__ uint64_t, __ubuf__ uint64_t>,
                Std::tuple<__ubuf__ fp4x2_e1m2_t, __ubuf__ fp4x2_e1m2_t>,
                Std::tuple<__ubuf__ fp4x2_e2m1_t, __ubuf__ fp4x2_e2m1_t>,
                Std::tuple<__ubuf__ fp8_e5m2_t, __ubuf__ fp8_e5m2_t>,
                Std::tuple<__ubuf__ fp8_e4m3fn_t, __ubuf__ fp8_e4m3fn_t>,
                Std::tuple<__ubuf__ hifloat8_t, __ubuf__ hifloat8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_ub_to_gm_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<
                          Std::tuple<dst_data_type, src_data_type>, Std::tuple<__gm__ half, __ubuf__ half>,
                          Std::tuple<__gm__ bfloat16_t, __ubuf__ bfloat16_t>, Std::tuple<__gm__ float, __ubuf__ float>,
                          Std::tuple<__gm__ int8_t, __ubuf__ int8_t>, Std::tuple<__gm__ uint8_t, __ubuf__ uint8_t>,
                          Std::tuple<__gm__ int16_t, __ubuf__ int16_t>, Std::tuple<__gm__ uint16_t, __ubuf__ uint16_t>,
                          Std::tuple<__gm__ int32_t, __ubuf__ int32_t>, Std::tuple<__gm__ uint32_t, __ubuf__ uint32_t>,
                          Std::tuple<__gm__ int64_t, __ubuf__ int64_t>, Std::tuple<__gm__ uint64_t, __ubuf__ uint64_t>,
                          Std::tuple<__gm__ fp4x2_e1m2_t, __ubuf__ fp4x2_e1m2_t>,
                          Std::tuple<__gm__ fp4x2_e2m1_t, __ubuf__ fp4x2_e2m1_t>,
                          Std::tuple<__gm__ fp8_e5m2_t, __ubuf__ fp8_e5m2_t>,
                          Std::tuple<__gm__ fp8_e4m3fn_t, __ubuf__ fp8_e4m3fn_t>,
                          Std::tuple<__gm__ hifloat8_t, __ubuf__ hifloat8_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_ub_to_l1_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_one_of_v<
                Std::tuple<dst_data_type, src_data_type>, Std::tuple<__cbuf__ half, __ubuf__ half>,
                Std::tuple<__cbuf__ bfloat16_t, __ubuf__ bfloat16_t>, Std::tuple<__cbuf__ float, __ubuf__ float>,
                Std::tuple<__cbuf__ int8_t, __ubuf__ int8_t>, Std::tuple<__cbuf__ uint8_t, __ubuf__ uint8_t>,
                Std::tuple<__cbuf__ int16_t, __ubuf__ int16_t>, Std::tuple<__cbuf__ uint16_t, __ubuf__ uint16_t>,
                Std::tuple<__cbuf__ int32_t, __ubuf__ int32_t>, Std::tuple<__cbuf__ uint32_t, __ubuf__ uint32_t>,
                Std::tuple<__cbuf__ int64_t, __ubuf__ int64_t>, Std::tuple<__cbuf__ uint64_t, __ubuf__ uint64_t>,
                Std::tuple<__cbuf__ fp4x2_e1m2_t, __ubuf__ fp4x2_e1m2_t>,
                Std::tuple<__cbuf__ fp4x2_e2m1_t, __ubuf__ fp4x2_e2m1_t>,
                Std::tuple<__cbuf__ fp8_e5m2_t, __ubuf__ fp8_e5m2_t>,
                Std::tuple<__cbuf__ fp8_e4m3fn_t, __ubuf__ fp8_e4m3fn_t>,
                Std::tuple<__cbuf__ hifloat8_t, __ubuf__ hifloat8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename SrcTensor>
    __aicore__ inline static constexpr void check_gm_to_l1_nd2_nd_src_one_dim()
    {
        using shape_row1 = typename get_n_dim_type<SrcTensor, attr_info::shape, attr_info::row, 1>::type;
        using shape_col1 = typename get_n_dim_type<SrcTensor, attr_info::shape, attr_info::column, 1>::type;
        static_assert(Std::is_constant<1, shape_row1>::value || Std::is_constant<1, shape_col1>::value,
                      "The src only support 1D tensor");
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l1_to_biastable_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<
                          Std::tuple<dst_data_type, src_data_type>, Std::tuple<__biasbuf__ float, __cbuf__ bfloat16_t>,
                          Std::tuple<__biasbuf__ float, __cbuf__ half>, Std::tuple<__biasbuf__ float, __cbuf__ float>,
                          Std::tuple<__biasbuf__ int32_t, __cbuf__ int32_t>>,
                      "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l1_to_fixbuf_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_same_v<Std::tuple<dst_data_type, src_data_type>, Std::tuple<__fbuf__ uint64_t, __cbuf__ uint64_t>>,
            "The data type is not supported.");
#endif
    }

    template <QuantMode_t quant_pre, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l0c_to_gm_data_type()
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            (quant_pre == QuantMode_t::NoQuant
             && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__gm__ float, __cc__ float>,
                                 Std::tuple<__gm__ int32_t, __cc__ int32_t>>)
                || (quant_pre == QuantMode_t::F322F16
                    && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__gm__ half, __cc__ float>>)
                || (quant_pre == QuantMode_t::F322BF16
                    && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__gm__ bfloat16_t, __cc__ float>>),
            "The data type is not supported.");
#endif
    }

    template <QuantMode_t quant_pre, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l0c_to_l1_data_type()
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            (quant_pre == QuantMode_t::NoQuant
             && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__cbuf__ float, __cc__ float>,
                                 Std::tuple<__cbuf__ int32_t, __cc__ int32_t>>)
                || (quant_pre == QuantMode_t::F322F16
                    && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__cbuf__ half, __cc__ float>>)
                || (quant_pre == QuantMode_t::F322BF16
                    && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__cbuf__ bfloat16_t, __cc__ float>>),
            "The data type is not supported.");
#endif
    }

    template <QuantMode_t quant_pre, typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l0c_to_ub_data_type()
    {
        using src_type = typename SrcTensor::element_type;
        using dst_type = typename DstTensor::element_type;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            (quant_pre == QuantMode_t::NoQuant
             && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__ubuf__ float, __cc__ float>,
                                 Std::tuple<__ubuf__ int32_t, __cc__ int32_t>>)
                || (quant_pre == QuantMode_t::F322F16
                    && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__ubuf__ half, __cc__ float>>)
                || (quant_pre == QuantMode_t::F322BF16
                    && Std::is_one_of_v<Std::tuple<dst_type, src_type>, Std::tuple<__ubuf__ bfloat16_t, __cc__ float>>),
            "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l1_to_l0a_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_one_of_v<
                Std::tuple<dst_data_type, src_data_type>, Std::tuple<__ca__ half, __cbuf__ half>,
                Std::tuple<__ca__ int16_t, __cbuf__ int16_t>, Std::tuple<__ca__ uint16_t, __cbuf__ uint16_t>,
                Std::tuple<__ca__ bfloat16_t, __cbuf__ bfloat16_t>, Std::tuple<__ca__ uint32_t, __cbuf__ uint32_t>,
                Std::tuple<__ca__ int32_t, __cbuf__ int32_t>, Std::tuple<__ca__ float, __cbuf__ float>,
                Std::tuple<__ca__ uint8_t, __cbuf__ uint8_t>, Std::tuple<__ca__ int8_t, __cbuf__ int8_t>,
                Std::tuple<__ca__ fp8_e4m3fn_t, __cbuf__ fp8_e4m3fn_t>,
                Std::tuple<__ca__ fp8_e5m2_t, __cbuf__ fp8_e5m2_t>,
                Std::tuple<__ca__ fp4x2_e2m1_t, __cbuf__ fp4x2_e2m1_t>,
                Std::tuple<__ca__ fp4x2_e1m2_t, __cbuf__ fp4x2_e1m2_t>,
                Std::tuple<__ca__ hifloat8_t, __cbuf__ hifloat8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l1_to_l0scalea_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_one_of_v<Std::tuple<dst_data_type, src_data_type>, Std::tuple<fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l1_to_l0b_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_one_of_v<
                Std::tuple<dst_data_type, src_data_type>, Std::tuple<__cb__ half, __cbuf__ half>,
                Std::tuple<__cb__ int16_t, __cbuf__ int16_t>, Std::tuple<__cb__ uint16_t, __cbuf__ uint16_t>,
                Std::tuple<__cb__ bfloat16_t, __cbuf__ bfloat16_t>, Std::tuple<__cb__ uint32_t, __cbuf__ uint32_t>,
                Std::tuple<__cb__ int32_t, __cbuf__ int32_t>, Std::tuple<__cb__ float, __cbuf__ float>,
                Std::tuple<__cb__ uint8_t, __cbuf__ uint8_t>, Std::tuple<__cb__ int8_t, __cbuf__ int8_t>,
                Std::tuple<__cb__ fp8_e4m3fn_t, __cbuf__ fp8_e4m3fn_t>,
                Std::tuple<__cb__ fp8_e5m2_t, __cbuf__ fp8_e5m2_t>,
                Std::tuple<__cb__ fp4x2_e2m1_t, __cbuf__ fp4x2_e2m1_t>,
                Std::tuple<__cb__ fp4x2_e1m2_t, __cbuf__ fp4x2_e1m2_t>,
                Std::tuple<__cb__ hifloat8_t, __cbuf__ hifloat8_t>>,
            "The data type is not supported.");
#endif
    }

    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline static constexpr void check_l1_to_l0scaleb_data_type()
    {
        using src_data_type = typename SrcTensor::element_type;
        using dst_data_type = typename DstTensor::element_type;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(
            Std::is_one_of_v<Std::tuple<dst_data_type, src_data_type>, Std::tuple<fp8_e8m0_t, __cbuf__ fp8_e8m0_t>>,
            "The data type is not supported.");
#endif
    }
};

} // namespace te
} // namespace asc

#endif // IMPL_TENSOR_API_ARCH_UTILS_CHECK_DATA_TYPE_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
