/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: grouped_matmul_swiglu_quant_v2_layered header file
 * Create: 2026-09-20
 * Note:
 * History: 2026-09-20 port from cam_async repository
 */
#ifndef OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_H
#define OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_H

#include "opdev/op_executor.h"

namespace l0op {

const std::tuple<aclTensor *, aclTensor *> GroupedMatmulSwigluQuantV2Layered(
    const aclTensor *x, const aclTensorList *weight, const aclTensorList *weightScale, const aclTensor *xScale,
    const aclTensorList *weightAssistanceMatrix, const aclTensor *bias, const aclTensor *smoothScale,
    const aclTensor *groupList, const aclTensor *layerIndex, int64_t dequantMode, int64_t dequantDtype,
    int64_t quantMode, int64_t quantDtype, bool transposeWeight, int64_t groupListType,
    const aclIntArray *tuningConfigOptional, aclOpExecutor *executor);
}

#endif