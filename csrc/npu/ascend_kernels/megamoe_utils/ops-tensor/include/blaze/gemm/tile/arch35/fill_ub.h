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
 * \file fill_ub.h
 * \brief Tile primitive for filling a contiguous UB tensor with a provided value on DAV_3510.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze::Gemm::Tile {

template <typename T>
class FillUb {
public:
    template <typename Tensor>
    __aicore__ inline static void FillWithValue(const Tensor& dstTensor, T fillValue)
    {
        using TensorElementType = asc::te::get_attribute_element_type<typename Tensor::element_type*>;
        using LayoutPattern = asc::te::get_layout_pattern<typename Tensor::layout_type>;
        static_assert(AscendC::Std::is_same_v<TensorElementType, T>,
                      "FillUb requires the tensor element type to match the fill value type");
        static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<Tensor>, asc::te::location::ub>,
                      "FillUb only supports UB tensors");
        static_assert(AscendC::Std::is_same_v<LayoutPattern, asc::te::nd_ext_layout_ptn>,
                      "FillUb requires a contiguous NDExt UB tensor");

        auto rowCount = static_cast<uint64_t>(asc::te::get_total_row_shape(dstTensor.layout()));
        auto columnCount = static_cast<uint64_t>(asc::te::get_total_column_shape(dstTensor.layout()));
        auto elementCount = rowCount * columnCount;
        if (elementCount == 0) {
            return;
        }

        constexpr uint32_t elementsPerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        auto repeatTimes = CeilDiv(elementCount, static_cast<uint64_t>(elementsPerRepeat));
        asc_vf_call<FillWithValueVf>((__ubuf__ T*)dstTensor.data().get(), fillValue,
                                     static_cast<uint32_t>(elementCount), static_cast<uint16_t>(repeatTimes));
    }

private:
    static __simd_vf__ inline void FillWithValueVf(__ubuf__ T* dstUbAddr, T fillValue, uint32_t elementCount,
                                                   uint16_t repeatTimes)
    {
        constexpr uint32_t elementsPerRepeat = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        AscendC::Reg::RegTensor<T> fillReg;
        AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate<T, AscendC::Reg::MaskMergeMode::ZEROING>(fillReg, fillValue, fullMask);

        uint32_t remainingElements = elementCount;
        for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<T>(remainingElements);
            AscendC::Reg::StoreAlign<T>(dstUbAddr + static_cast<uint32_t>(repeatIdx) * elementsPerRepeat, fillReg,
                                        mask);
        }
    }
};
} // namespace Blaze::Gemm::Tile
