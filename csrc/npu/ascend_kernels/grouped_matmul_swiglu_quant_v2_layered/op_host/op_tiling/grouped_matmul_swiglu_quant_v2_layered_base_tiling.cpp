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
 * Description: grouped_matmul_swiglu_quant_v2_layered_base_tiling source file
 * Create: 2026-09-20
 * Note:
 * History: 2026-09-20 port from cam_async repository
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_base_tiling.cpp
 * \brief
 */
#include "grouped_matmul_swiglu_quant_v2_layered_base_tiling.h"
#include "util/math_util.h"
#include "err/ops_err.h"

using namespace matmul_tiling;

namespace optiling {
namespace GroupedMatmulSwigluQuantV2LayeredTiling {

constexpr int64_t ND_WEIGHT_MULTI_TENSOR_DIM = 2;
constexpr int64_t NZ_WEIGHT_MULTI_TENSOR_DIM = 4;
constexpr float EFFECTIVE_TASK_RATIO = 0.95f;
constexpr int32_t MIN_BASE_M = 16;

template <typename T> static inline auto AlignUp(T a, T base) -> T
{
    if (base == 0) {
        return 0;
    }
    return (a + base - 1) / base * base;
}

template <typename T1, typename T2> auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U;
    }
    return a & ~15U;
}

int64_t GroupedMatmulSwigluQuantV2LayeredBaseTiling::CalMaxRowInUbA8W4(const uint64_t ubSize, const uint64_t n) const
{
    const uint64_t ALIGNMENT = 8;
    const float WEIGHT_FACTOR = isA4W4_ ? 4.5f : 8.5f;
    const uint64_t ALIGNMENT_TERM_FACTOR = 4;
    const uint64_t LINEAR_TERM_FACTOR = 6;
    const uint64_t CONSTANT_TERM = 64;
    const int64_t MIN_ROW_THRESHOLD = 1;

    // A8W4 expression: 8.5 * row * n + 4 * alignUp(row, 8) + 6n + 64 <= ubSize
    // A4W4 expression: 4.5 * row * n + 4 * alignUp(row, 8) + 6n + 64 <= ubSize

    // Initial estimate ignoring the alignment term
    int64_t maxRowEstimate =
        (ubSize - CONSTANT_TERM - LINEAR_TERM_FACTOR * n) / static_cast<int64_t>(WEIGHT_FACTOR * n);

    // Consider the impact of alignment
    uint64_t alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    uint64_t totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) +
                         ALIGNMENT_TERM_FACTOR * alignedRow + LINEAR_TERM_FACTOR * n + CONSTANT_TERM;

    // If the UB size is exceeded, gradually decrease row until the condition is satisfied
    while (totalSize > ubSize && maxRowEstimate > 0) {
        maxRowEstimate--;
        alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) + ALIGNMENT_TERM_FACTOR * alignedRow +
                    LINEAR_TERM_FACTOR * n + CONSTANT_TERM;
    }

    if (maxRowEstimate < MIN_ROW_THRESHOLD) {
        OP_LOGE(context_->GetNodeName(), "GMM_SWIGLU_QUANT TILING: No valid row found for n = %lu, ubSize = %lu\n", n,
                ubSize);
        return 0;
    }
    return maxRowEstimate;
}

int64_t GroupedMatmulSwigluQuantV2LayeredBaseTiling::CalMaxRowInUb(const uint64_t ubSize, const uint64_t n) const
{
    uint64_t tmpBufSize = (n / SWIGLU_REDUCE_FACTOR) * FP32_DTYPE_SIZE;
    uint64_t perchannleBufSize = n * FP32_DTYPE_SIZE * DOUBLE_BUFFER;
    uint64_t reduceMaxResBufSize = BLOCK_BYTE;
    uint64_t reduceMaxTmpBufSize = BLOCK_BYTE;
    const uint64_t CONSTANT_TERM = 64;
    int64_t remainUbSize = ubSize - tmpBufSize - perchannleBufSize - reduceMaxResBufSize - reduceMaxTmpBufSize;
    int64_t maxRowInUb =
        remainUbSize / (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    int64_t curUb = DOUBLE_BUFFER * (maxRowInUb * (INT32_DTYPE_SIZE * n + n / SWIGLU_REDUCE_FACTOR) +
                                     AlignUp(maxRowInUb, FP32_BLOCK_SIZE) * FP32_DTYPE_SIZE);
    if (curUb > remainUbSize) {
        // 64 : make sure ub does not excceed maxUbSize after align up to 8
        maxRowInUb = (remainUbSize - CONSTANT_TERM) /
                     (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    }
    if (maxRowInUb < 1) {
        // when n > (ubSize - 72) / 19 = 10330, maxRowInUb < 1
        OP_LOGE(context_->GetNodeName(), "GMM_SWIGLU_QUANT TILING: n should not be greater than 10240, now is %lu\n",
                n);
    }
    return maxRowInUb;
}

bool GroupedMatmulSwigluQuantV2LayeredBaseTiling::IsCapable()
{
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    ge::DataType weightDType = weightDesc->GetDataType();
    if (weightDType != ge::DataType::DT_INT4) {
        return false;
    }

    // layered: each list element is one layer's full-expert tensor (single-tensor semantics),
    // so only the single-tensor dim layouts are accepted.
    auto wTensor = context_->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wTensor);
    if (!(wTensor->GetStorageShape().GetDimNum() == ND_WEIGHT_DIM_LIMIT ||
          wTensor->GetStorageShape().GetDimNum() == NZ_WEIGHT_DIM_LIMIT)) {
        return false;
    }

    return true;
}

ge::graphStatus GroupedMatmulSwigluQuantV2LayeredBaseTiling::ParseInputAndAttr()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto weightDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    auto wTensor = context_->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wTensor);
    auto xTensor = context_->GetDynamicInputTensor(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto wScaleTensor = context_->GetDynamicInputTensor(WEIGHT_SCALE_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wScaleTensor);
    auto groupListTensor = context_->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, groupListTensor);

    // layered: layer_index must be INT64 with shape [1]
    auto layerTensor = context_->GetDynamicInputTensor(LAYER_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layerTensor);
    OP_CHECK_IF(layerTensor->GetDataType() != ge::DataType::DT_INT64,
                OP_LOGE(context_->GetNodeName(), "layer_index must be INT64."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(layerTensor->GetStorageShape().GetDimNum() != 1 || layerTensor->GetStorageShape().GetDim(0) != 1,
                OP_LOGE(context_->GetNodeName(), "layer_index must be a 1-element tensor."), return ge::GRAPH_FAILED);

    // layered: all_weight / all_weight_scale / all_weight_assist_matrix lists must be the same
    // length (one element per layer). The value of layer_index lives on device and cannot be
    // range-checked here; the kernel trusts it.
    uint32_t weightListLen = 0;
    while (context_->GetDynamicInputTensor(WEIGHT_INDEX, weightListLen) != nullptr) {
        weightListLen++;
    }
    uint32_t wScaleListLen = 0;
    while (context_->GetDynamicInputTensor(WEIGHT_SCALE_INDEX, wScaleListLen) != nullptr) {
        wScaleListLen++;
    }
    uint32_t wAssistListLen = 0;
    while (context_->GetDynamicInputTensor(WEIGHT_ASSIST_INDEX, wAssistListLen) != nullptr) {
        wAssistListLen++;
    }
    OP_CHECK_IF(weightListLen == 0 || weightListLen != wScaleListLen,
                OP_LOGE(context_->GetNodeName(), "all_weight and all_weight_scale list lengths differ: %u vs %u.",
                        weightListLen, wScaleListLen),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wAssistListLen != 0 && wAssistListLen != weightListLen,
                OP_LOGE(context_->GetNodeName(), "all_weight_assist_matrix list length %u differs from all_weight %u.",
                        wAssistListLen, weightListLen),
                return ge::GRAPH_FAILED);

    // layered: every layer element carries all experts (single-tensor semantics per layer)
    isSingleTensor_ = 1;

    auto attr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attr); // check attr is not null
    const int64_t *dequantModePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    auto dequantMode = dequantModePtr != nullptr ? *dequantModePtr : 0;
    OP_CHECK_IF(!(dequantMode == 0 || dequantMode == 1),
                OP_LOGE(context_->GetNodeName(), "dequantMode must be 0 or 1, but actual value is %ld.", dequantMode),
                return ge::GRAPH_FAILED);
    const int64_t *groupListTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE);
    groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;
    OP_CHECK_IF(
        !(groupListType_ == 0 || groupListType_ == 1),
        OP_LOGE(context_->GetNodeName(), "GroupListType must be 0 or 1, but actual value is %ld.", groupListType_),
        return ge::GRAPH_FAILED);

    ge::DataType xDType = xDesc->GetDataType();
    ge::DataType weightDType = weightDesc->GetDataType();

    isA8W4MSD_ = (xDType == ge::DataType::DT_INT8 && weightDType == ge::DataType::DT_INT4);
    isA4W4_ = (xDType == ge::DataType::DT_INT4 && weightDType == ge::DataType::DT_INT4);
    if (isA4W4_) {
        auto smoothScaleTensor = context_->GetDynamicInputTensor(SMOOTH_SCALE_INDEX, 0);
        if (smoothScaleTensor == nullptr) {
            smoothScaleDimNum_ = 0;
        } else {
            smoothScaleDimNum_ = smoothScaleTensor->GetStorageShape().GetDimNum();
        }
    }

    auto compileInfoPtr = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr"),
                return ge::GRAPH_FAILED);

    m_ = xTensor->GetStorageShape().GetDim(0);
    k_ = xTensor->GetStorageShape().GetDim(1);
    auto wScaleDimNum = wScaleTensor->GetStorageShape().GetDimNum();
    isWeightTrans_ = false;
    if (wTensor->GetStorageShape().GetDimNum() == NZ_WEIGHT_DIM_LIMIT) {
        isNz_ = true;
    }
    const auto tuningConfigPtr = attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG);
    tuningConfig_ = tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 1
                        ? (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[0]
                        : 0;

    if (isA4W4_) {
        n_ = wScaleTensor->GetStorageShape().GetDim(wScaleDimNum - DIM_1);
    } else {
        if (wTensor->GetStorageShape().GetDimNum() == ND_WEIGHT_DIM_LIMIT) {
            // ND per-layer single tensor [E, K, N]
            n_ = wTensor->GetStorageShape().GetDim(DIM_2);
        } else {
            // NZ per-layer single tensor [E, N // 64, K // 16, 16, 64]
            n_ = wTensor->GetStorageShape().GetDim(DIM_1) * wTensor->GetStorageShape().GetDim(DIM_4);
        }
    }

    isWeightTrans_ = *attr->GetAttrPointer<int64_t>(ATTR_INDEX_TRANSPOSE_WEIGHT);

    if (dequantMode == 1) { // perGroup quantization mode: single-tensor scenario [E, KGroupCount, N], multi-tensor
                            // scenario [KGroupCount, N]
        quantGroupNum_ = wScaleTensor->GetStorageShape().GetDim(wScaleDimNum - DIM_2);
    } else { // perChannel quantization mode
        quantGroupNum_ = 1;
    }

    groupNum_ = groupListTensor->GetStorageShape().GetDim(0);

    if (isA8W4MSD_ || isA4W4_) {
        maxProcessRowNum_ = CalMaxRowInUbA8W4(compileInfoPtr->ubSize_, n_);
    } else {
        maxProcessRowNum_ = CalMaxRowInUb(compileInfoPtr->ubSize_, n_);
    }

    blockDim_ = compileInfoPtr->aicNum_;
    return ge::GRAPH_SUCCESS;
}

int32_t GroupedMatmulSwigluQuantV2LayeredBaseTiling::FindBestSingleN(const uint32_t &aicNum, int64_t baseM,
                                                                     int64_t baseN) const
{
    uint64_t quantGroupNum = quantGroupNum_;
    // baseN is used as a divisor below, so reject the degenerate zero case early.
    if (baseN <= 0 || n_ < baseN || tuningConfig_ <= 0 || !(quantGroupNum == 1)) {
        return baseN;
    }
    int32_t mDim = CeilDiv(tuningConfig_, baseM);
    int32_t nDim = CeilDiv(n_, baseN);
    int32_t taskNum = mDim * nDim * static_cast<int32_t>(groupNum_);
    int32_t taskNumPerCore = CeilDiv(taskNum, aicNum);
    // When each core only needs to process 1 basic block, the task volume is too small and no handling is needed
    if (taskNumPerCore <= 1) {
        return baseN;
    }
    int32_t curNDim = 0;
    int32_t curTaskNum = 0;
    int32_t bestSingleN = baseN;
    float ratio = 0;
    for (uint32_t i = 1; i <= aicNum; ++i) {
        if (isNz_) {
            bestSingleN = CeilDiv(static_cast<int32_t>(n_), i);
            if (bestSingleN != n_ && bestSingleN % baseN != 0) {
                continue;
            }
        } else {
            // For now, dynamic blocking is only enabled for the NZ format
            return baseN;
        }
        curNDim = CeilDiv(n_, bestSingleN);
        curTaskNum = mDim * curNDim * static_cast<int32_t>(groupNum_);
        ratio = static_cast<float>(curTaskNum) / AlignUp(static_cast<uint32_t>(curTaskNum), aicNum);
        if (ratio >= EFFECTIVE_TASK_RATIO) {
            return bestSingleN;
        }
    }
    return baseN;
}

bool GroupedMatmulSwigluQuantV2LayeredBaseTiling::TryFullLoadA(int32_t baseM, int64_t baseN, int64_t baseK,
                                                               uint64_t l1Size)
{
    // For now only A4W4 is supported
    float sizeofweightDtype = 0.5f;
    float sizeofxDtype = 0.5f;
    auto matBl1Size = static_cast<int32_t>(tilingData_.mmTilingData.get_depthB1() * baseN * baseK * sizeofweightDtype);
    auto remainL1Size = l1Size - matBl1Size - 8 * baseN;
    int32_t newDepthA1 = CeilDiv(k_, baseK);
    if (static_cast<int32_t>(newDepthA1 * baseM * baseK * sizeofxDtype) < static_cast<int32_t>(remainL1Size)) {
        tilingData_.mmTilingData.set_stepKa(newDepthA1);
        tilingData_.mmTilingData.set_depthA1(newDepthA1);
        return true;
    }
    return false;
}

ge::graphStatus GroupedMatmulSwigluQuantV2LayeredBaseTiling::DynamicTilingSingleN(gert::TilingContext *context,
                                                                                  const uint32_t &aicNum, int64_t baseM,
                                                                                  int64_t baseN, int64_t baseK)
{
    // get info
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t l1Size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    tilingData_.gmmSwigluQuantV2BaseParams.set_singleN(0);

    if (n_ < baseN || tuningConfig_ <= 0 || !isA4W4_) {
        return ge::GRAPH_SUCCESS;
    }
    int32_t bestSingleN = FindBestSingleN(aicNum, baseM, baseN);
    if (bestSingleN == baseN) { // No better singleN found
        return ge::GRAPH_SUCCESS;
    }
    tilingData_.gmmSwigluQuantV2BaseParams.set_singleN(bestSingleN);
    // First, without changing anything, check whether baseM can fully load the left matrix
    if (TryFullLoadA(baseM, baseN, baseK, l1Size)) {
        return ge::GRAPH_SUCCESS;
    }
    // We can try reducing baseM to fully load the left matrix
    int32_t newBaseM = static_cast<int32_t>(SixteenAlign(tuningConfig_, true));
    // Avoid an uneven distribution
    newBaseM += MIN_BASE_M;
    // Check again whether the left matrix can be fully loaded
    if (newBaseM < baseM && TryFullLoadA(newBaseM, baseN, baseK, l1Size)) {
        tilingData_.mmTilingData.set_baseM(newBaseM);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2LayeredBaseTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "Begin Run GMM Swiglu Tiling .");

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    MatmulApiTiling tiling(ascendcPlatform);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBias(false);
    tiling.SetShape(A8W4_BASEM, A8W4_BASEN, k_);
    tiling.SetFixSplit(A8W4_BASEM, A8W4_BASEN, A8W4_BASEK);
    tiling.SetOrgShape(m_, n_, k_);
    tiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(tiling.GetTiling(tilingData_.mmTilingData) == -1,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "grouped_matmul_swiglu_quant_base_tiling, get tiling failed"),
                return ge::GRAPH_FAILED);
    if (isA8W4MSD_ || isA4W4_) {
        tilingData_.mmTilingData.set_baseM(A8W4_BASEM);
        tilingData_.mmTilingData.set_baseN(A8W4_BASEN);
        tilingData_.mmTilingData.set_baseK(A8W4_BASEK);
        tilingData_.mmTilingData.set_dbL0B(DOUBLE_BUFFER);
        tilingData_.mmTilingData.set_stepKa(NUM_FOUR);
        tilingData_.mmTilingData.set_stepKb(NUM_FOUR);
        tilingData_.mmTilingData.set_depthA1(NUM_EIGHT);
        tilingData_.mmTilingData.set_depthB1(NUM_EIGHT);
        tilingData_.mmTilingData.set_stepM(1);
        tilingData_.mmTilingData.set_stepN(1);
    }

    usrWorkspaceLimit_ = USER_WORKSPACE_LIMIT;
    mLimit_ = 0;
    if (isA8W4MSD_) {
        mLimit_ =
            ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / (k_ * sizeof(int8_t) + DOUBLE_ROW * n_ * SIZE_OF_HALF_2));
    } else if (isA4W4_) {
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / (n_ * SIZE_OF_HALF_2));
    } else {
        mLimit_ = ((usrWorkspaceLimit_ / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n_;
    }

    OP_CHECK_IF(mLimit_ <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "mLimit_ is %ld must over then 0.", mLimit_),
                return ge::GRAPH_FAILED);
    tilingData_.gmmSwigluQuantV2BaseParams.set_mLimit(mLimit_);

    DynamicTilingSingleN(context_, blockDim_, A8W4_BASEM, A8W4_BASEN, A8W4_BASEK);

    if (isA8W4MSD_) {
        int workSpaceMTemp = mLimit_ * DOUBLE_WORKSPACE_SPLIT;
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset1(workSpaceMTemp * k_ * sizeof(int8_t));
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset2(DOUBLE_ROW * workSpaceMTemp * n_ * SIZE_OF_HALF_2);
        workspaceSize_ =
            SYS_WORKSPACE_SIZE + // 16MB reserved by the system
            (workSpaceMTemp * k_ *
             sizeof(int8_t)) + // Phase 1: preprocess the left matrix (mLimit_, K) * int8 * 2 (double WorkSpace)
            (DOUBLE_ROW * workSpaceMTemp * n_ *
             SIZE_OF_HALF_2); // Phase 2: matrix multiplication result (2 * mLimit_, N) * fp16 * 2 (double WorkSpace)
    } else if (isA4W4_) {
        int workSpaceMTemp = mLimit_ * DOUBLE_WORKSPACE_SPLIT;
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset1(mLimit_ * n_ * SIZE_OF_HALF_2);
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset2(0);
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * SIZE_OF_HALF_2);
    } else {
        int workSpaceMTemp = (mLimit_ * DOUBLE_WORKSPACE_SPLIT > m_ ? m_ : mLimit_ * DOUBLE_WORKSPACE_SPLIT);
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset1(0);
        tilingData_.gmmSwigluQuantV2BaseParams.set_workSpaceOffset2(0);
        workspaceSize_ = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n_ * sizeof(int32_t));
    }

    isSplitWorkSpace_ = m_ > mLimit_ * DOUBLE_WORKSPACE_SPLIT;
    SetTilingKeyAndScheMode();
    FillTilingData();
    PrintTilingData();
    OP_LOGD(context_->GetNodeName(), "End Run GMM Swiglu Tiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2LayeredBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

void GroupedMatmulSwigluQuantV2LayeredBaseTiling::FillTilingData()
{
    tilingData_.gmmSwigluQuantV2BaseParams.set_groupNum(groupNum_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_coreNum(blockDim_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_K(k_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_N(n_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_M(m_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_baseM(A8W4_BASEM);
    tilingData_.gmmSwigluQuantV2BaseParams.set_baseN(A8W4_BASEN);
    tilingData_.gmmSwigluQuantV2BaseParams.set_quantGroupNum(quantGroupNum_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_isSingleTensor(isSingleTensor_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_groupListType(groupListType_);
    tilingData_.gmmSwigluQuantV2BaseParams.set_smoothScaleDimNum(smoothScaleDimNum_);
    tilingData_.gmmSwigluQuantV2.set_maxProcessRowNum(maxProcessRowNum_);
    tilingData_.gmmSwigluQuantV2.set_groupListLen(groupNum_);
    tilingData_.gmmSwigluQuantV2.set_tokenLen(n_);
}

void GroupedMatmulSwigluQuantV2LayeredBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "grouped_matmul_swiglu_quant_base_tiling.");
    OP_LOGD(context_->GetNodeName(), "groupNum:      %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_groupNum());
    OP_LOGD(context_->GetNodeName(), "coreNum:       %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "M:             %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_M());
    OP_LOGD(context_->GetNodeName(), "K:             %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_K());
    OP_LOGD(context_->GetNodeName(), "N:             %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_N());
    OP_LOGD(context_->GetNodeName(), "baseM:         %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_baseM());
    OP_LOGD(context_->GetNodeName(), "baseN:         %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_baseN());
    OP_LOGD(context_->GetNodeName(), "mLimit:        %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_mLimit());
    OP_LOGD(context_->GetNodeName(), "quantGroupNum: %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_quantGroupNum());
    OP_LOGD(context_->GetNodeName(), "isSingleTensor:%ld", tilingData_.gmmSwigluQuantV2BaseParams.get_isSingleTensor());
    OP_LOGD(context_->GetNodeName(), "groupListType: %ld", tilingData_.gmmSwigluQuantV2BaseParams.get_groupListType());
    OP_LOGD(context_->GetNodeName(), "smoothScaleDimNum: %ld",
            tilingData_.gmmSwigluQuantV2BaseParams.get_smoothScaleDimNum());
    OP_LOGD(context_->GetNodeName(), "maxProcessRowNum:      %ld", tilingData_.gmmSwigluQuantV2.get_maxProcessRowNum());
    OP_LOGD(context_->GetNodeName(), "groupListLen:          %ld", tilingData_.gmmSwigluQuantV2.get_groupListLen());
    OP_LOGD(context_->GetNodeName(), "tokenLen:              %ld", tilingData_.gmmSwigluQuantV2.get_tokenLen());
    OP_LOGD(context_->GetNodeName(), "USER_WORKSPACE_LIMIT:  %ld", usrWorkspaceLimit_);
    OP_LOGD(context_->GetNodeName(), "workspaceSizes:        %lu", workspaceSize_);
    OP_LOGD(context_->GetNodeName(), "isSplitWorkSpace:      %s", isSplitWorkSpace_ ? "true" : "false");
}

void GroupedMatmulSwigluQuantV2LayeredBaseTiling::SetTilingKeyAndScheMode()
{
    if (isA8W4MSD_) { // A8W4 MSD tiling_key
        tilingKey_ = A8W4_MSD_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else if (isA4W4_ && !isWeightTrans_) {
        tilingKey_ = A4W4_WEIGHT_NOTRANS_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else if (isA4W4_ && isWeightTrans_) {
        tilingKey_ = A4W4_WEIGHT_TRANS_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else if (isSplitWorkSpace_) {
        tilingKey_ = SPLITWORKSPACE_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else {
        tilingKey_ = COMMON_TILING_KEY_MODE;
        context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    }
}

ge::graphStatus GroupedMatmulSwigluQuantV2LayeredBaseTiling::PostTiling()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(blockDim_);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    return ge::GRAPH_SUCCESS;
}

} // namespace GroupedMatmulSwigluQuantV2LayeredTiling
} // namespace optiling
