/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Apply fused activation and quantization in the GEMM epilogue.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

/*!
 * \file activation_common.h
 * \brief 门控激活公共数据结构、参数和常量。
 */

#ifndef MEGA_DISPATCH_SEND_ARCH35_ACTIVATION_COMMON_H
#define MEGA_DISPATCH_SEND_ARCH35_ACTIVATION_COMMON_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/atten_send_constants.h"

namespace MegaDispatchSendImpl {
namespace Activation {

// 具体激活实现按 FP32 寄存器宽度划分主循环。
constexpr uint32_t VECTOR_LENGTH_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);

// 输入统一转换为 FP32，供各激活的寄存器计算流程使用。
constexpr AscendC::Reg::CastTrait CAST_INPUT_TO_FP32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::UNKNOWN};

// 当前激活结果统一写入 BF16 UB；量化流程在激活之后独立执行。
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_BF16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};

/*
 * 单个门控激活 tile 的完整执行上下文。
 *
 * Block Epilogue 在分派激活前直接计算地址、物理行距、循环次数和尾部 Mask，具体激活仅消费这些
 * 已物化的数据。此结构不保存 RegTensor、MaskReg 等 VF 内对象。
 */
template <typename InputType>
struct GatedActivationTileContext {
    __ubuf__ InputType *gate;
    __ubuf__ InputType *up;
    __ubuf__ bfloat16_t *output;
    __ubuf__ float *topkWeights;

    __ubuf__ InputType *gateTail;
    __ubuf__ InputType *upTail;
    __ubuf__ bfloat16_t *outputTail;
    __ubuf__ bfloat16_t *additionalPaddingOutput;

    uint32_t inputRowStrideElements;
    uint32_t outputRowStrideElements;
    uint16_t rowLoopCount;
    uint16_t fullVectorLoopCount;
    // 0/1 标志：直接作为尾 Vector 计算的 VF 循环上界。
    uint16_t needTailVectorCompute;
    // 0/1 标志：直接作为额外补零写回的 VF 循环上界。
    uint16_t needAdditionalPaddingStore;

    // UpdateMask 接收可写引用，具体激活需将以下计数复制到局部变量后再调用。
    uint32_t tailComputeMaskElementCount;
    uint32_t tailStoreMaskElementCount;
    uint32_t additionalPaddingStoreMaskElementCount;
};

} // namespace Activation
} // namespace MegaDispatchSendImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_DISPATCH_SEND_ARCH35_ACTIVATION_COMMON_H
