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
 * Description: aclnn_grouped_matmul_swiglu_quant_v2_layered header file
 * Create: 2026-09-20
 * Note:
 * History: 2026-09-20 port from cam_async repository
 */
#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_LAYERED_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First stage interface of aclnnGroupedMatmulSwigluQuantV2Layered, which computes the workspace size
 * according to the concrete computation flow.
 * @domain aclnn_ops_infer
 *
 * @param [in] x:
 * Represents x in the formula. Supported data types include INT8, FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2, and
 * HIFLOAT8; the data format supports ND.
 * @param [in] allWeight:
 * The all-layer weight list (each element is the weight of all experts in that layer, selected by layer_index).
 * The supported data type is INT4.
 * @param [in] allWeightScale:
 * The all-layer weight scale list (corresponding to allWeight layer by layer). The supported data type is UINT64;
 * the data format supports ND.
 * @param [in] allWeightAssistMatrix:
 * The all-layer weight auxiliary matrix list (bias semantics, corresponding to allWeight layer by layer).
 * The supported data type is FLOAT32.
 * @param [in] bias: Reserved parameter; must be nullptr in the current version.
 * @param [in] xScale:
 * Represents the perToken quantization parameter. Supported data types are FLOAT8_E8M0 and FLOAT32; the data format
 * supports ND.
 * @param [in] smoothScale:
 * The quantization factor of the left matrix. The supported data type is FLOAT32; the data format supports ND.
 * @param [in] groupList: Required parameter, represents the number of tokens participating in the computation for each
 * group. The supported data type is INT64.
 * @param [in] layerIndex: Required parameter, the current layer number (a device-side tensor with shape [1]).
 * The supported data type is INT64.
 * @param [in] dequantMode: Represents the dequantization computation type, used to determine the dequantization
 * method for the activation matrix and the weight matrix.
 * @param [in] dequantDtype: Represents the result data type of the intermediate GroupedMatmul.
 * @param [in] quantMode: Represents the quantization computation type, used to determine the quantization mode of the
 * swiglu result.
 * @param [in] groupListType: Represents the interpretation of the specified grouping, used to determine the semantics
 * of groupList.
 * @param [in] tuningConfig: Used to estimate the sizes of m/e for the operator, selecting different operator templates
 * to suit performance requirements of different scenarios.
 * @param [out] quantOutput:
 * Represents out in the formula. Supported data types include INT8, FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2, and
 * HIFLOAT8; the data format supports ND.
 * @param [out] quantScaleOutput: Represents outQuantScale in the formula. Supported data types are FLOAT32 and
 * FLOAT8_E8M0.
 * @param [out] workspaceSize: Returns the workspace size the user needs to allocate on the npu device side.
 * @param [out] executor: Returns the op executor, which contains the operator computation flow.
 * @return aclnnStatus: Returns the status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantV2LayeredGetWorkspaceSize(
    const aclTensor *x, const aclTensorList *allWeight, const aclTensorList *allWeightScale,
    const aclTensorList *allWeightAssistMatrix, const aclTensor *bias, const aclTensor *xScale,
    const aclTensor *smoothScale, const aclTensor *groupList, const aclTensor *layerIndex, int64_t dequantMode,
    int64_t dequantDtype, int64_t quantMode, int64_t groupListType, const aclIntArray *tuningConfigOptional,
    aclTensor *output, aclTensor *outputScale, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Second stage interface of aclnnGroupedMatmulSwigluQuantV2Layered, used to execute the computation.
 * @param [in] workspace: The starting address of the workspace allocated on the npu device side.
 * @param [in] workspaceSize: The workspace size allocated on the npu
 * device side, obtained from the first stage interface aclnnGroupedMatmulSwigluQuantV2LayeredGetWorkspaceSize.
 * @param [in] stream: The acl stream.
 * @param [in] executor: The op executor, which contains the operator computation flow.
 * @return aclnnStatus: Returns the status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantV2Layered(void *workspace,
                                                                                          uint64_t workspaceSize,
                                                                                          aclOpExecutor *executor,
                                                                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif