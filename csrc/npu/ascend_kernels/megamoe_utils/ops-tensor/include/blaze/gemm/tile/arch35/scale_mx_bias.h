/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scale_mx_bias.h
 * \brief Scale bias for MX MMAD accumulation.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Gemm::Tile {

template <typename BiasType_>
class ScaleMxBias {
public:
    template <typename BiasInTensor, typename BiasOutTensor>
    __aicore__ inline ScaleMxBias(const BiasInTensor& biasInTensor, const BiasOutTensor& biasOutTensor)
    {
        // This specialized tile consumes a contiguous bias vector. Reject other layouts
        // at instantiation time instead of silently treating their data as contiguous.
        using BiasInLayoutPattern = asc::te::get_layout_pattern<typename BiasInTensor::layout_type>;
        using BiasOutLayoutPattern = asc::te::get_layout_pattern<typename BiasOutTensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<BiasInLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                          AscendC::Std::is_same_v<BiasOutLayoutPattern, asc::te::nd_ext_layout_ptn>,
                      "ScaleMxBias requires contiguous NDExt bias tensors");

        constexpr uint64_t VECTOR_REG_BYTE_SIZE = static_cast<uint64_t>(asc_get_vf_len());
        constexpr uint64_t VECTOR_ELEMENTS = VECTOR_REG_BYTE_SIZE / sizeof(BiasType);
        uint64_t elementCount = static_cast<uint64_t>(asc::te::get_total_column_shape(biasInTensor.layout()));
        ScaleMxBiasParams params{static_cast<uint16_t>(CeilDiv(elementCount, VECTOR_ELEMENTS)),
                                 (__ubuf__ BiasType*)biasInTensor.data().get(),
                                 (__ubuf__ BiasType*)biasOutTensor.data().get()};
        asc_vf_call<ScaleMxBiasVf>(params);
    }

private:
    using BiasType = BiasType_;

    struct ScaleMxBiasParams {
        uint16_t loopNum;
        __ubuf__ BiasType_* biasInAddr;
        __ubuf__ BiasType_* biasOutAddr;
    };

    static __simd_vf__ inline void ScaleMxBiasVf(ScaleMxBiasParams params) { ScaleMxBiasCallee(params); }

    // Kept as a SIMD callee so the weight-conversion tile can compose bias
    // scaling into the same compile-time VF entry.  This does not introduce the
    // asc_vf_call barrier that would serialize the vector and MTE pipelines.
    static __simd_callee__ inline void ScaleMxBiasCallee(ScaleMxBiasParams params)
    {
        namespace MicroAPI = AscendC::MicroAPI;
        constexpr uint64_t VECTOR_REG_BYTE_SIZE = static_cast<uint64_t>(asc_get_vf_len());
        constexpr uint64_t VECTOR_ELEMENTS = VECTOR_REG_BYTE_SIZE / sizeof(BiasType);
        constexpr BiasType MX_BIAS_FACTOR = static_cast<BiasType>(0.015625f);

        MicroAPI::RegTensor<BiasType> biasReg;
        MicroAPI::RegTensor<BiasType> factorReg;
        MicroAPI::MaskReg mask = MicroAPI::CreateMask<BiasType, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate<BiasType, MicroAPI::MaskMergeMode::ZEROING>(factorReg, MX_BIAS_FACTOR, mask);
        for (uint16_t loopIdx = 0; loopIdx < params.loopNum; ++loopIdx) {
            MicroAPI::AddrReg addr = MicroAPI::CreateAddrReg<BiasType>(loopIdx, VECTOR_ELEMENTS);
            MicroAPI::LoadAlign<BiasType, MicroAPI::LoadDist::DIST_NORM>(biasReg, params.biasInAddr, addr);
            MicroAPI::Mul(biasReg, biasReg, factorReg, mask);
            MicroAPI::StoreAlign<BiasType, MicroAPI::StoreDist::DIST_NORM_B16>(params.biasOutAddr, biasReg, addr, mask);
        }
    }

    template <typename, typename>
    friend class ShiftW4ToW8;
};

} // namespace Blaze::Gemm::Tile
