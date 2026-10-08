/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling.cpp
 * \brief
 */
#include "grouped_matmul_tiling.h"

#include <climits>
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../../op_kernel/grouped_matmul_tiling_key.h"
using namespace ge;
using namespace AscendC;
using namespace GroupedMatmul;

namespace optiling {
static inline uint32_t SixteenAlign(uint32_t a, bool up = false) {
    if (up) {
        a += 15U;  // 15: 16 bytes up-align
    }
    return a & ~15U;  // ~15: 16 bytes down-align
}

static inline int64_t SixteenAlign(int64_t a, bool up = false) {
    if (up) {
        a += 15;  // 15: 16 bytes up-align
    }
    return a & ~15;  // ~15: 16 bytes down-align
}

template <typename T>
static inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

constexpr uint32_t GROUP_LIST_SPARSE_M = 2U;
// EFFECTIVE_TASK_RATIO表示动态分块后的任务数量占可处理任务数量的比例
// 通常情况下阈值越高，优化效果越好，但是阈值调高会导致可能找不到更优分块
// 基于当前模型case 测试，整体有优化
// DSKV3,k/n=7168/4096和2048/7168
// A2双机e=16/17， m/e平均32/48/64/96
// A2大EP/A3四机 e=4/5， m/e平均32/48/64/96
// A3八机e=2/3，m/e平均192
// Qwen3 k/n=2048/1536和768/2048
// 30B单机 e=128，m/e平均 8/16
constexpr float EFFECTIVE_TASK_RATIO = 0.95f;
// 小K时,由于vector bound,所以开启双vector会有优化,
// 中K时,开启两个vector时会导致MTE2带宽争抢严重,可能会劣化
// 大K时,开启两个vector会提升部分带宽利用率，此时也会有提升
// 实测数据表明,当K小于等于1024或者K大于2048时开启双Vector会有优化
constexpr int64_t DOUBLE_VECTOT_THRESHOLD_K_LOWER = 1024L;
constexpr int64_t DOUBLE_VECTOT_THRESHOLD_K_UPPER = 2048L;
// 实测当单专家token数低于128时cube算力不能完全发挥，导致开启2个vector核可能会劣化
constexpr int32_t SMALL_TUNING_CONFIG_THRESHOLD = 128;
constexpr int32_t BIAS_REMAIN_SPACE = 2 * 1024;
constexpr int32_t MIN_BASE_M = 16;
// 定轴搬移算法K的范围
constexpr int64_t FIXAXISMOVE_K1 = 2048L;
constexpr int64_t FIXAXISMOVE_K2 = 7168L;
// 定轴搬移算法N的范围
constexpr int64_t FIXAXISMOVE_N1 = 7168L;
constexpr int64_t FIXAXISMOVE_N2 = 4096L;
// 定轴搬移算法group_num的范围
constexpr int32_t FIXAXISMOVE_GROUP_NUM = 4;
// 定轴搬移算法每个专家M的范围
constexpr int64_t FIXAXISMOVE_PERM_LOWER = 128L;
constexpr int64_t FIXAXISMOVE_PERM_UPPER = 512L;
// 定轴搬移算法split_item的范围
constexpr int64_t FIXAXISMOVE_SPLIT_ITEM2 = 2L;
constexpr int64_t FIXAXISMOVE_SPLIT_ITEM3 = 3L;
// 定轴搬移算法group_list_type的范围
constexpr int64_t FIXAXISMOVE_GROUP_LIST_TYPE = 0L;
// 定轴搬移算法group_type的范围
constexpr int32_t FIXAXISMOVE_GROUP_TYPE = 0;
constexpr size_t TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX = 0;
constexpr size_t TUNING_CONFIG_A8W4_SPEC_SCENARIO_INDEX = 1;
constexpr size_t TUNING_CONFIG_ALLOW_WORKSPACE_INDEX = 2;
constexpr int64_t SPLITK_M_N_RATIO_THRESHOLD_2 = 2L;


static inline uint32_t FindBestSingleNA8W4(uint32_t baseM_, uint32_t baseN_, uint32_t avg_m, uint32_t maxN_, uint32_t groupNum_, const uint32_t& aicNum) {
  int32_t mDim = CeilDiv(avg_m, baseM_);
  int32_t nDim = CeilDiv(maxN_, baseN_);
  int32_t taskNum = mDim * nDim * static_cast<int32_t>(groupNum_);
  int32_t taskNumPerCore = CeilDiv(taskNum, aicNum);
  //每个核只需要做1个基本块的时候，任务量太少，无需处理
  if(taskNumPerCore <= 1){
    return baseN_;
  }
  int32_t curNDim = 0;
  int32_t curTaskNum = 0;
  int32_t bestSingleN = baseN_;
  float ratio = 0;
  for (uint32_t i = 1; i <= aicNum; ++i){
    bestSingleN = CeilDiv(static_cast<int32_t>(maxN_), i);
    if(baseN_ != 0 && static_cast<int64_t>(bestSingleN) != maxN_ && bestSingleN % baseN_ != 0) {
      continue;
    }
    curNDim = CeilDiv(maxN_, bestSingleN);
    curTaskNum = mDim * curNDim * static_cast<int32_t>(groupNum_);
    ratio = static_cast<float>(curTaskNum) / AlignUp(static_cast<uint32_t>(curTaskNum), aicNum);
    if(ratio >= EFFECTIVE_TASK_RATIO) {
      return bestSingleN;
    }
  }
  return baseN_;
}


ge::graphStatus GMMTiling::CheckWeightNZShape(const gert::TilingContext* context, int64_t numInOneBlk) const {
  OP_CHECK_IF(numInOneBlk <= 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "numInOneBlk, the "
             "input of CheckWeightNZShape has an invalid value %ld", numInOneBlk), return ge::GRAPH_FAILED);
  size_t i = 0;
  while (true) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i++);
    if (wTensor == nullptr) { break; }
    gert::Shape wOriginShape = wTensor->GetOriginShape();
    int64_t lastDimValue = wOriginShape.GetDim(wOriginShape.GetDimNum() - 1);  // inner axis
    OP_CHECK_IF(lastDimValue % numInOneBlk != 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
               "the inner axis size of nz weight is expected to be a multiple of 32B, "
               "but now the inner axis size is %ld.", lastDimValue),
               return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::CheckMKN(const gert::TilingContext* context) {
  mmDataTypeSize_ = GetSizeByDataType(mmDType_);
  OP_CHECK_IF(mmDataTypeSize_ == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMM get mm dtype[%s] size is 0.", TypeUtils::DataTypeToAscendString(mmDType_).GetString()),
             return ge::GRAPH_FAILED);
  uint32_t numInOneBlk = 0;
  if (isA4W4_) {
    numInOneBlk = static_cast<uint32_t>(ONE_BLK_SIZE / INT4_DATA_TYPE_SIZE);
  } else {
    numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize_;
  }
  OP_CHECK_IF(numInOneBlk == 0, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);
  int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
  OP_CHECK_IF(maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "32B-aligned m, n or k axis is out of range int32!"),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::SetTilingDataIsSingleTensor() {
  tilingData.gmmBaseParams.set_singleWeight(static_cast<uint32_t>(isSingleWeight_));
  tilingData.gmmBaseParams.set_singleX(static_cast<uint32_t>(isSingleX_));
  tilingData.gmmBaseParams.set_singleY(static_cast<uint32_t>(isSingleY_));
}

bool GMMTiling::CheckTensorListLength(const gert::TilingContext *context)
{
  auto xTensor = context->GetDynamicInputTensor(X_INDEX, MAX_TENSOR_CONT);
  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, MAX_TENSOR_CONT);
  OP_CHECK_IF(xTensor != nullptr || wTensor != nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM tensorList length should between [1, %d]",
                                          MAX_TENSOR_CONT),
              return false);
  return true;
}

ge::graphStatus GMMTiling::PrepareTilingData(const gert::TilingContext* context) {
  // get transpose and groupType
  OP_CHECK_IF(GMMGetAttrs(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMMGetAttrs failed"),
             return ge::GRAPH_FAILED);
  // get the first tensor's shape of weight and x
  auto xTensor = context->GetDynamicInputTensor(X_INDEX, 0);  // 0: get first tensor
  OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
  gert::Shape xShape = xTensor->GetStorageShape();
  xDimNum_ = static_cast<uint32_t>(xShape.GetDimNum());
  xKDim_ = transposeX_ ? 0U : xDimNum_ - 1U;  // 0: when x is transposed, the first dim is k; -1：otherwise, the last dim is k

  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, wTensor);
  gert::Shape wShape = wTensor->GetOriginShape();
  uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
  weightNDim_ = transposeWeight_ ? wDimNum - 2U : wDimNum - 1U;  // -2: when w is transposed, the last 2 dim is n; -1: otherwise, the last dim is n
  weightKDim_ = transposeWeight_ ? wDimNum - 1U : wDimNum - 2U;  // -2: when w is transposed, the last 1 dim is k; -1: otherwise, the last 2 dim is k
  nzFactor_ = 1;  // init
  if (wFormat_ == matmul_tiling::CubeFormat::NZ) {
    uint32_t numInOneBlk = isA4W4_ ? static_cast<uint32_t>(UB_BLOCK_UNIT_SIZE / INT4_DATA_TYPE_SIZE) :
                           UB_BLOCK_UNIT_SIZE / std::max(1, GetSizeByDataType(weightDtype_));
    if (isA8W4FakeA8W8_) {
      numInOneBlk = UB_BLOCK_UNIT_SIZE;
    }
    if (wDimNum >= 4U) {  // 4: least dim num of nz format tensor
      weightNDim_ = transposeWeight_ ? wDimNum - 3U : wDimNum - 4U;  // -3: when w is transposed, the last 3 dim is n/nzFactor; -4: when w has nz format, the last 4 dim is n/nzFactor
      // nzFactor_ is a factor used to compute n axis size. If weight is transposed, nzFactor_ is 16; otherwise nzFactor_ is 16 for bf16, 32 for int8
      nzFactor_ = transposeWeight_ ? 16 : static_cast<int32_t>(numInOneBlk);
    } else {
      OP_CHECK_IF(CheckWeightNZShape(context, static_cast<int64_t>(numInOneBlk)) != ge::GRAPH_SUCCESS,
                 OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "the shape of nz weight is invalid."),
                 return ge::GRAPH_FAILED);
    }
  }
  // layered: all_weight is a per-layer full-expert tensor list; the weight list length is the
  // layer count, so singleWeight semantics is fixed to 1 (expert addressing = element offsets
  // within the layer) regardless of how many layers are passed.
  isSingleWeight_ = true;
  isSingleX_ = (context->GetDynamicInputTensor(X_INDEX, 1) == nullptr);
  isSingleY_ = (splitItem_ == 2 || splitItem_ == 3);  // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
  SetTilingDataIsSingleTensor();

  if (groupType_ == SPLIT_M) {
    return GMMGetTensorShapeSplitM(context, xShape, wShape);
  }
  if (groupType_ == SPLIT_K) {
    return GMMGetTensorShapeSplitK(context, xShape, wShape);
  }
  if (groupType_ == NO_SPLIT) {  // not split any axis
    if (isSingleWeight_ && wDimNum > 2U) {  // 2: dim of split weight tensor
      return SeparatedXSingleWeight(context, wShape);
    }
    return SeparatedXSeparatedWeight(context);
  }
  OP_LOGE(context->GetNodeName(), "GMM_tiling: not support groupType_=%ld, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d",
            groupType_, isSingleWeight_, isSingleX_, isSingleY_);
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape &xShape,
    const gert::Shape &wShape) {
    if (isSingleX_ && isSingleWeight_ && isSingleY_) {  // split M, s-s-s
      return SplitMSingleXSingleWeightSingleY(xShape, wShape);
    }
    if (isSingleX_ && !isSingleWeight_ && isSingleY_) {  // split M, s-m-s
      return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (isSingleX_ && !isSingleWeight_ && !isSingleY_) {  // splitM, s-m-m
      return SplitMSingleXSeparatedWeight(context, xShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && isSingleY_) {  // split M, m-m-s
      return SeparatedXSeparatedWeight(context);
    }
    if (!isSingleX_ && isSingleWeight_) {  // split M, m-s-m/m-s-s
      return SeparatedXSingleWeight(context, wShape);
    }
    if (!isSingleX_ && !isSingleWeight_ && !isSingleY_) {  // split M, m-m-m
      return SeparatedXSeparatedWeight(context);
    }
    OP_LOGE(context->GetNodeName(), "GMM_tiling: not support groupType_=%ld, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d",
              groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape &xShape,
    const gert::Shape &wShape) {
    if (isSingleX_ && isSingleWeight_ && isSingleY_) {  // splitK, s-s-s
      return SplitKSingleXSingleWeightSingleY(context, xShape, wShape);
    }
    if (isSingleX_ && !isSingleWeight_ && !isSingleY_) {  // splitK, s-m-s
      return SplitKSingleXSeparatedWeight(context, xShape, wShape);
    }
    if (!isSingleX_ && isSingleWeight_) {  // splitK, m-s-m/m-s-s
      return SeparatedXSingleWeight(context, wShape);
    }
    OP_LOGE(context->GetNodeName(), "GMM_tiling: not support groupType_=%ld, isSingleWeight_=%d, isSingleX_=%d, isSingleY_=%d",
              groupType_, isSingleWeight_, isSingleX_, isSingleY_);
    return ge::GRAPH_FAILED;
}

/** @brief split M：single-single-single(s-s-s)
*/
ge::graphStatus GMMTiling::SplitMSingleXSingleWeightSingleY(const gert::Shape &xShape, const gert::Shape &wShape) {
  groupNum_ = static_cast<int32_t>(wShape.GetDim(0));
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * static_cast<int64_t>(nzFactor_);
  kList_[0] = static_cast<int32_t>(k);  // if split M axis, the K axis values of x tensorList are all the same.
  nList_[0] = static_cast<int32_t>(n);
  mList_[0] = -1;
  maxM_ = m;
  maxK_ = k;
  maxN_ = n;
  totalM_ = static_cast<uint32_t>(m);
  FixedAxisMoveWorkspace_ = maxM_ * maxN_ * sizeof(int32_t);
  return ge::GRAPH_SUCCESS;
}

/** @brief split M：single-multi-single(s-m-s)/single-multi-multi(s-m-m), share the same function.
*/
ge::graphStatus GMMTiling::SplitMSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape &xShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    if (wTensor == nullptr) { break; }  // when x has multi tensors, xTensor is allowed to be empty
    auto wShape = wTensor->GetOriginShape();

    groupNum_ += 1U;
    kList_[i] = static_cast<int32_t>(k);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    nList_[i] = static_cast<int32_t>(n);
    maxN_ = std::max(maxN_, n);
  }
  mList_[0] = -1;  // mList is unknown right now
  maxM_ = m;
  maxK_ = k;
  totalM_ = static_cast<uint32_t>(m);

  return ge::GRAPH_SUCCESS;
}

/** @brief split M：multi-multi-single(m-m-s); no split: multi-multi-multi(m-m-m), share the same function
*/
ge::graphStatus GMMTiling::SeparatedXSeparatedWeight(const gert::TilingContext* context) {
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
    if (wTensor == nullptr || xTensor == nullptr) { break; }
    auto wShape = wTensor->GetOriginShape();
    auto xShape = xTensor->GetStorageShape();
    groupNum_ += 1U;
    int64_t m = GMMGetBS(xShape);
    int64_t k = xShape.GetDim(xKDim_);
    int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
    mList_[i] = static_cast<int32_t>(m);
    kList_[i] = static_cast<int32_t>(k);
    nList_[i] = static_cast<int32_t>(n);
    maxM_ = std::max(maxM_, m);
    maxK_ = std::max(maxK_, k);
    maxN_ = std::max(maxN_, n);
    totalM_ += static_cast<uint32_t>(m);
  }
  groupType_ = NO_SPLIT;
  return ge::GRAPH_SUCCESS;
}

/** @brief split M : multi-single-multi(m-s-m), split K : multi-single-multi(m-s-m), share the same function
*/
ge::graphStatus GMMTiling::SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape &wShape) {
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto xTensor = context->GetDynamicInputTensor(X_INDEX, i);
    if (xTensor == nullptr) { break; }  // when x has multi tensors, xTensor is allowed to be empty
    auto xShape = xTensor->GetStorageShape();
    groupNum_ += 1U;
    int64_t m = GMMGetBS(xShape);
    int64_t k = xShape.GetDim(xKDim_);
    mList_[i] = static_cast<int32_t>(m);
    kList_[i] = static_cast<int32_t>(k);
    nList_[i] = static_cast<int32_t>(n);
    maxM_ = std::max(maxM_, m);
    maxK_ = std::max(maxK_, k);
    totalM_ += static_cast<uint32_t>(m);
  }
  maxN_ = n;
  groupType_ = NO_SPLIT;
  return ge::GRAPH_SUCCESS;
}

/** @brief split K single-single-single
*/
ge::graphStatus GMMTiling::SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context,
    const gert::Shape &xShape, const gert::Shape &wShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;

  auto groupListTensor = context->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
  if (groupListTensor == nullptr) {
    OP_LOGE(context->GetNodeName(), "groupListTensor is nullptr");
    return ge::GRAPH_FAILED;
  }
  gert::Shape groupListShape = groupListTensor->GetStorageShape();
  groupNum_ = static_cast<int32_t>(groupListShape.GetDim(0));  // 0: the first dim of groupList is groupNum
  mList_[0] = static_cast<int32_t>(m);
  nList_[0] = static_cast<int32_t>(n);
  kList_[0] = -1;
  maxM_ = m;
  maxN_ = n;
  maxK_ = k;
  totalM_ = static_cast<uint32_t>(m);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::SplitKSingleXSeparatedWeight(const gert::TilingContext* context,
  const gert::Shape &xShape, const gert::Shape &wShape) {
  int64_t m = GMMGetBS(xShape);
  int64_t k = xShape.GetDim(xKDim_);
  int64_t n = wShape.GetDim(weightNDim_) * nzFactor_;
  for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, i);
    if (wTensor == nullptr) { break; }
    auto wTensorShape = wTensor->GetOriginShape();
    groupNum_ += 1;
    mList_[i] = static_cast<int32_t>(m);
    k = wTensorShape.GetDim(weightKDim_);
    kList_[i] = static_cast<int32_t>(k);
    maxK_ = std::max(maxK_, k);
    n = wTensorShape.GetDim(weightNDim_);
    nList_[i] = static_cast<int32_t>(n);
    maxN_ = std::max(maxN_, n);
  }
  maxM_ = m;
  totalM_ = static_cast<uint32_t>(m);
  groupType_ = NO_SPLIT;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::Init(const gert::TilingContext* context) {
  OP_CHECK_IF(!CheckTensorListLength(context),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Check tensorList length failed."),
             return ge::GRAPH_FAILED);
  // layered: layer_index carries the current layer id (INT64, one element); all_weight /
  // all_bias / all_scale lists must share the same length (one element per layer). The layer
  // value itself lives on device and cannot be range-checked here; layerNum is shipped in the
  // tiling so the kernel can bounds-check the device-side index before dereferencing the
  // pointer array.
  {
    auto layerTensor = context->GetDynamicInputTensor(LAYER_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, layerTensor);
    OP_CHECK_IF(layerTensor->GetDataType() != ge::DT_INT64,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "layer_index must be INT64."),
                return ge::GRAPH_FAILED);
    // No shape assertion here, deliberately. layer_index is a REQUIRED input and the storage shape
    // this host context exposes for it is not the caller's logical [1]: a rank-1 [1] int64 tensor
    // arrives here normalized to rank 2 [1, 1]. An unconditional `dimNum == 1 && dim(0) == 1` test
    // therefore rejected a well-formed call before anything reached the device. The shape is
    // validated instead where the caller's real logical shape is available - GetViewShape() at the
    // aclnn entry (CheckLayerIndex in op_api/aclnn_grouped_matmul_layered.cpp) - and the value is
    // consumed on device by LayeredReadLayerIndex, which reads element 0 from GM without consulting
    // any shape. The dtype check above is still meaningful and stays.
    OP_LOGD(context->GetNodeName(),
            "GMM_tiling: layer_index host storage rank=%zu dim0=%ld (shape not asserted; the layer id "
            "is validated at the aclnn entry and clamped against layerNum on device).",
            layerTensor->GetStorageShape().GetDimNum(),
            layerTensor->GetStorageShape().GetDimNum() >= 1 ? layerTensor->GetStorageShape().GetDim(0) : -1L);
    uint32_t weightListLen = 0;
    while (context->GetDynamicInputTensor(WEIGHT_INDEX, weightListLen) != nullptr) {
      weightListLen++;
    }
    uint32_t biasListLen = 0;
    while (context->GetDynamicInputTensor(BIAS_INDEX, biasListLen) != nullptr) {
      biasListLen++;
    }
    uint32_t scaleListLen = 0;
    while (context->GetDynamicInputTensor(SCALE_INDEX, scaleListLen) != nullptr) {
      scaleListLen++;
    }
    OP_CHECK_IF(weightListLen == 0 || biasListLen != weightListLen || scaleListLen != weightListLen,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "all_weight/all_bias/all_scale list lengths differ: %u/%u/%u.",
                                            weightListLen, biasListLen, scaleListLen),
                return ge::GRAPH_FAILED);
    tilingData.gmmBaseParams.set_layerNum(weightListLen);
  }
  OP_CHECK_IF(PrepareTilingData(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM PrepareTilingData failed."),
             return ge::GRAPH_FAILED);
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);  // check compileInfoPtr is not null

  // check tuningConfig_
  if (tuningConfig_ < 0 || tuningConfig_ > maxM_) {
    OP_LOGE(context->GetNodeName(), "Invalid tuningConfig_: %ld. Valid range: [0, maxM]", tuningConfig_);
    return ge::GRAPH_FAILED;
  }
  // check whether x, weight and y are all single tensor
  isAllSingleTensor_ = isSingleX_ && isSingleWeight_ && isSingleY_;
  bool isA16W8 = (xDType_ == ge::DT_FLOAT16 || xDType_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT8;
  // check whether k and n are supported in msd
  bool isKNForA16W8MSD = maxN_ % static_cast<int64_t>(A16W8_MSD_KN_BASE_BLOCK) == 0L &&
                         maxK_ % static_cast<int64_t>(A16W8_MSD_KN_BASE_BLOCK) == 0L &&
                         maxK_ <= static_cast<int64_t>(A16W8_MSD_MAX_K) &&
                         maxN_ >= static_cast<int64_t>(A16W8_MSD_MIN_N);
  // check whether total token num and average token num are supported in msd
  bool isMForA16W8MSD = totalM_ <= A16W8_MSD_AVERAGE_TOKEN_NUM * groupNum_;
  isA16W8Msd_ = isAllSingleTensor_ && groupType_ == SPLIT_M && isA16W8 && isKNForA16W8MSD && isMForA16W8MSD;
  mmDType_ = isA16W8Msd_ ? ge::DT_INT8 : xDType_;
  OP_CHECK_IF(CheckMKN(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM CheckMKN failed."),
             return ge::GRAPH_FAILED);
  auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);  // 0: obtain the first tensor of the tensorList
  hasBias_ = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
  if (isA4W4_) {
    uint64_t quantGroupNum = 0;
    uint32_t scaleDimNum = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDimNum();
    // 3: pergroup scale shape is [e,g,n]
    if (scaleDimNum == 3U) {
      quantGroupNum = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(1);
    // 2: perchannel scale shape is [e, n]
    } else if (scaleDimNum == 2U) {
      quantGroupNum = 1UL;
    } else {
      OP_LOGE(context->GetNodeName(), "GMM A4W4: scale dim should be 2 or 3, but now is %u", scaleDimNum);
      return ge::GRAPH_FAILED;
    }
    tilingData.gmmBaseParams.set_k(maxK_);
    tilingData.gmmBaseParams.set_n(maxN_);
    tilingData.gmmBaseParams.set_quantGroupNum(quantGroupNum);
  }
  if (isA8W4FakeA8W8_) {
    hasBias_ = false;
  }
  tilingData.gmmArray.set_mList(mList_);
  tilingData.gmmArray.set_kList(kList_);
  tilingData.gmmArray.set_nList(nList_);
  tilingData.gmmBaseParams.set_groupNum(groupNum_);
  tilingData.gmmBaseParams.set_m(totalM_);
  tilingData.gmmBaseParams.set_hasBias(static_cast<uint32_t>(hasBias_));
  tilingData.gmmBaseParams.set_groupType(static_cast<int32_t>(groupType_));
  tilingData.gmmBaseParams.set_activeType(actType_);
  tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  tilingData.gmmBaseParams.set_groupListType(groupListType_);
  tilingData.gmmBaseParams.set_k(maxK_);
  tilingData.gmmBaseParams.set_n(maxN_);
  OP_LOGI(context->GetNodeName(), "GMM_tiling: groupNum_ is %u, maxM_ is %ld, maxK_ is %ld, maxN_ is %ld.",
            groupNum_, maxM_, maxK_, maxN_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GetPerGroupNum(const gert::TilingContext* context) {
  const gert::Tensor *antiquantScale = nullptr; /* antiquant_scale dropped */
  OP_CHECK_NULL_WITH_CONTEXT(context, antiquantScale);
  auto antiquantScaleShape = antiquantScale->GetStorageShape();
  int64_t dimNum = antiquantScaleShape.GetDimNum();
  auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, wTensor);
  gert::Shape wShape = wTensor->GetOriginShape();
  size_t wDimNum = wShape.GetDimNum();
  if ((isSingleWeight_ && wDimNum > 2UL && dimNum == 3L) || (!isSingleWeight_ && dimNum == 2L)) {  // 2 and 3: dim threshold
    int64_t g = antiquantScaleShape.GetDim(dimNum - 2L);
    perTokenOrPerGroupSize_ = g > 1L ? static_cast<uint32_t>(kList_[0] / g) : 0U;
    tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  }
  if (isSingleWeight_) {
      tilingData.gmmBaseParams.set_withOffset(GetWithOffset(context));
  } else {
      tilingData.gmmBaseParams.set_withOffset(1);  // 非单场景，offset默认存在
  }
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::DivideUbAndSetWorkspaceAntiquant(size_t* workspaces, const uint32_t& aicNum, uint32_t &ubSize) {
    if (isA16W8Msd_) {
      // whole workspace for a16w8 msd scene is combined by workspace for global max of each row (m * 8),
      // workspace for local reduce sum of each row (m * aicNum), workspace for data after prerpocessing x (2 * m * k)
      // and workspace for matmul output (2 * m * n)
      // 32: need 32 byte to store global max
      workspaces[0] += totalM_ * (aicNum * sizeof(float) + 32UL +
                                  A16W8_MSD_STEP * (static_cast<uint64_t>(maxK_) * sizeof(int8_t) +
                                                    static_cast<uint64_t>(maxN_) * sizeof(int32_t)));
      // 7: make aicnum align up to 8
      uint32_t alignedAicNum = (aicNum + 7U) & (~7U);
      ubSize = static_cast<uint32_t>(ubSize_ - (static_cast<uint64_t>(baseM_) / A16W8_MSD_STEP) *
                                               alignedAicNum * sizeof(float));
      // workspacesSize in GMMBaseParams is size of matmul left input matrix of matmul.
      workspacesSize_ += A16W8_MSD_STEP * static_cast<uint64_t>(maxK_) * static_cast<uint64_t>(maxM_);
    } else {
      for (uint32_t i = 0; i < groupNum_; i++) {
        bool isAllSingleTensor = isSingleX_ && isSingleWeight_ && isSingleY_;
        int32_t kInList = isAllSingleTensor ? kList_[0] : kList_[i];  // in s-s-s case，k only exits in the first of the list
        int32_t nInList = isAllSingleTensor ? nList_[0] : nList_[i];  // in s-s-s case，n only exits in the first of the list
        int32_t k = kList_[0] == -1 ? static_cast<int32_t>(maxK_) : kInList;
        int32_t n = nList_[0] == -1 ? static_cast<int32_t>(maxN_) : nInList;
        minK_ = std::min(minK_, k);
        workspacesSize_ += static_cast<uint64_t>(k) * static_cast<uint64_t>(n);
      }
      // when minK * baseN * coreNum * sizeof(float16) > 12M, it goes into antiquantPerformance branch (12M is obtained by test).
      int32_t dimMN =
        CeilDiv(CeilDiv(maxM_, groupNum_), baseM_) * CeilDiv(maxN_, baseN_);
      bool goodCubeUtility = dimMN * (xDType_ == ge::DT_BF16 ? 2 : 1) >= static_cast<int32_t>(aicNum * 0.4);  // 0.4: a factor, in practice.
      antiquantPerformance_ =
        goodCubeUtility && static_cast<int64_t>(minK_) * baseN_ * static_cast<int64_t>(aicNum) >= ANTIQUANT_PERFORMANCE_THRESHOLD && !transposeWeight_;
      uint32_t maxUbBaseN = static_cast<uint32_t>(BEST_UB_BASEN);
      if (transposeWeight_) {
        maxUbBaseN = baseN_;
      } else if (antiquantPerformance_) {
        // 2: use 2 pieces of workspace in antiquantPerformance branch
        workspacesSize_ = static_cast<uint64_t>(maxN_) * static_cast<uint64_t>(maxK_) * 2UL;
      }
      // 2: 2 InQueue(antiquant_scale,antiquant_offset)
      ubSize = static_cast<uint32_t>(ubSize_ - 2U * maxUbBaseN * mmDataTypeSize_ * QUEUE_DOUBLE_BUFFER);
      workspaces[0] += workspacesSize_ * mmDataTypeSize_;
    }
}

int32_t GMMTiling::FindBestSingleN(const uint32_t& aicNum) {
  uint64_t quantGroupNum = tilingData.gmmBaseParams.get_quantGroupNum();
  // A8W8模式以及A4W4 Perchannel模式支持动态分块
  if(maxN_ < baseN_ || tuningConfig_ <= 0|| !(isA8W8_ || (isA4W4_ && quantGroupNum == 1)) ) {
    return baseN_;
  }
  int32_t mDim = CeilDiv(tuningConfig_ , baseM_);
  int32_t nDim = CeilDiv(maxN_, baseN_);
  int32_t taskNum = mDim * nDim * static_cast<int32_t>(groupNum_);
  int32_t taskNumPerCore = CeilDiv(taskNum, aicNum);
  // 每个核只需要做1个基本块的时候，任务量太少，无需处理
  if(taskNumPerCore <= 1) {
    return baseN_;
  }
  int32_t curNDim = 0;
  int32_t curTaskNum = 0;
  int32_t bestSingleN = baseN_;
  float ratio = 0;
  for (uint32_t i = 1; i <= aicNum; ++i) {
    if(wFormat_ == matmul_tiling::CubeFormat::NZ) {
      bestSingleN = CeilDiv(static_cast<int32_t>(maxN_), i);
      if(bestSingleN != maxN_ && bestSingleN % baseN_ != 0) {
        continue;
      }
    } else {
      //暂时只NZ格式开启动态分块
      return baseN_;
    }
    curNDim = CeilDiv(maxN_, bestSingleN);
    curTaskNum = mDim * curNDim * static_cast<int32_t>(groupNum_);
    ratio = static_cast<float>(curTaskNum) / AlignUp(static_cast<uint32_t>(curTaskNum), aicNum);
    if (ratio >= EFFECTIVE_TASK_RATIO) {
      return bestSingleN;
    }
  }
  return baseN_;
}

bool GMMTiling::TryFullLoadA(int32_t baseM,const GMMCompileInfo *compileInfoPtr) {
  auto l1Size = compileInfoPtr->l1Size;
  //暂时只支持A8W8和A4W4Perchannel模式
  float sizeofweightDtype = weightDtype_ == ge::DT_INT4 ? 0.5f : 1.0f;
  float sizeofxDtype = xDType_ == ge::DT_INT4 ? 0.5f : 1.0f;

  auto matBl1Size = static_cast<int32_t>(tilingData.mmTilingData.get_depthB1() * baseN_ * baseK_ * sizeofweightDtype);
  auto remainL1Size = l1Size - matBl1Size;
  if(hasBias_) {
    remainL1Size -= BIAS_REMAIN_SPACE;
  }
  int32_t newDepthA1 = CeilDiv(maxK_, baseK_);
  if(static_cast<int32_t>(newDepthA1 * baseM * baseK_ * sizeofxDtype) < static_cast<int32_t>(remainL1Size)) {
    tilingData.mmTilingData.set_stepKa(newDepthA1);
    tilingData.mmTilingData.set_depthA1(newDepthA1);
    return true;
  }
  return false;
}

/*
1、输出默认会存储在L2Cache中，可以考虑设置输出不走L2Cache从而提高输入的L2Cache命中率
2、后续算子可能会用到当前算子的输出，若当前输出不在L2Cache中可能会影响后续算子效率
3、当前算子输出如果超过L2Cache大小时，实测不会对后续算子造成影响，此时可以跳过L2Cache
*/
ge::graphStatus GMMTiling::IsOutputDisableL2Cache(gert::TilingContext* context, const GMMCompileInfo *compileInfoPtr) {
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(
                context->GetNodeName(), "compileInfoPtr is nullptr."), return ge::GRAPH_FAILED);
    //仅切K单x单weight场景可以Tiling侧知道输出矩阵大小
    if(groupType_ != SPLIT_K || !isSingleX_ || !isSingleWeight_) {
        tilingData.gmmBaseParams.set_isOutputDisableL2Cache(0);
        return ge::GRAPH_SUCCESS;
    }
    auto outputSize = static_cast<uint64_t>(GetSizeByDataType(yDtype_) * maxM_ * maxN_ * groupNum_);
    if(outputSize > compileInfoPtr->l2Size) {
        tilingData.gmmBaseParams.set_isOutputDisableL2Cache(1);
    } else {
        tilingData.gmmBaseParams.set_isOutputDisableL2Cache(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::DynamicTilingSingleN(gert::TilingContext* context, const uint32_t& aicNum, const GMMCompileInfo *compileInfoPtr) {
  OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(
               context->GetNodeName(), "compileInfoPtr is nullptr."), return ge::GRAPH_FAILED);
  if (maxN_ < baseN_ || tuningConfig_ <= 0 || wFormat_ == matmul_tiling::CubeFormat::ND) {
    return ge::GRAPH_SUCCESS;
  }
  int32_t bestSingleN = FindBestSingleN(aicNum);
  if(bestSingleN == baseN_) {//没找到更优的singleN
    return ge::GRAPH_SUCCESS;
  }
  tilingData.gmmBaseParams.set_singleN(bestSingleN);
  //先不改看看baseM能否全载左矩阵
  if(TryFullLoadA(baseM_, compileInfoPtr)) {
    return ge::GRAPH_SUCCESS;
  }
  //可以尝试减小baseM来全载左矩阵
  int32_t newBaseM = static_cast<int32_t>(SixteenAlign(tuningConfig_, true));
  //防止不均匀情况
  newBaseM += MIN_BASE_M;
  //再看看能否全载左矩阵
  if(newBaseM < baseM_ && TryFullLoadA(newBaseM, compileInfoPtr)) {
    tilingData.mmTilingData.set_baseM(newBaseM);
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus GMMTiling::DivideUbAndSetWorkspace(gert::TilingContext* context, const uint32_t& aicNum) {
  size_t* workspaces = context->GetWorkspaceSizes(1);  // get second variable
  OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);  // check workspaces is not null
  workspaces[0] = SYS_WORKSPACE_SIZE;  // default size
  if (weightDtype_ != ge::DT_INT8 && weightDtype_ != ge::DT_INT4) {
    return ge::GRAPH_SUCCESS;
  }
  uint32_t ubSize = static_cast<uint32_t>(ubSize_);
  if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16)) {
    DivideUbAndSetWorkspaceAntiquant(workspaces, aicNum, ubSize);
    OP_CHECK_IF(GetPerGroupNum(context) != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(
               context->GetNodeName(), "GetPerGroupNum failed."), return ge::GRAPH_FAILED);
  } else if (xDType_ == ge::DT_INT8) {
    isFixedAxisMove_ = IsFixedAxisMoveCondition();
    if (isFixedAxisMove_) {
          workspaces[0] += FixedAxisMoveWorkspace_;
    } else {
      // if tuningConfig_ in [1,256], recompute coreNum
      constexpr int32_t tuningConfigLowerLimit = 1;
      constexpr int32_t tuningConfigUpperLimit = 256;
      if (tuningConfig_ >= tuningConfigLowerLimit && tuningConfig_ <= tuningConfigUpperLimit) {
        FindBestUsedCoreNumOneGroup(aicNum);
      }
      if (yDtype_ == ge::DT_INT32) {
        return ge::GRAPH_SUCCESS;
      }
      uint32_t scaleDataTypeSize = GetSizeByDataType(scaleDtype_);
      ubSize = perTokenOrPerGroupSize_ == 1U ?  // is perToken
        static_cast<uint32_t>(ubSize_ -
                              (static_cast<uint64_t>(baseN_) * scaleDataTypeSize +
                              static_cast<uint64_t>(baseM_) * sizeof(float)) * QUEUE_DOUBLE_BUFFER) :
        static_cast<uint32_t>(ubSize_ - baseN_ * scaleDataTypeSize * QUEUE_DOUBLE_BUFFER);
      OP_CHECK_IF(SetWorkspscesPerTokenQuant(aicNum, workspaces) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetWorkspscesPerTokenQuant failed."),
                return ge::GRAPH_FAILED);
      if (isA8W4FakeA8W8_) {
        workspaces[0] += A8W4noMsdSpace_;
      }
    }
  } else if (xDType_ == ge::DT_INT4) {
    ubSize = perTokenOrPerGroupSize_ == 1U ?  // is perToken
      static_cast<uint32_t>(ubSize_ - (static_cast<uint32_t>(baseM_) * sizeof(float)) * QUEUE_DOUBLE_BUFFER) : ubSize_;
    OP_CHECK_IF(SetWorkspscesPerTokenQuant(aicNum, workspaces) != ge::GRAPH_SUCCESS,
      OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetWorkspscesPerTokenQuant failed."),
      return ge::GRAPH_FAILED);
  }
  OP_CHECK_IF(GMMSetUbDivideBlk() != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMMSetUbDivideBlk failed."),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(GMMCalUbSize(context, ubSize) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "GMMCalUbSize failed."),
             return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

int32_t GMMTiling::FindBestSingleNPertoken(const uint32_t aicNum) const {
  if (CeilDiv(maxN_, baseN_) * groupNum_ <= aicNum) {  // if all matmuls only occupy a part of cores
    return baseN_;
  }
  if (maxN_  >= 2048) {  // 2048: a threshold
    return 1024;  // 1024: max singleN
  }
  int32_t bestSingleN = baseN_;  // init bestSingleN
  uint32_t bestLastCycleCoreNum = (groupNum_ * CeilDiv(maxN_, bestSingleN)) % aicNum;  // init lastCycleCoreNum
  // 1024: max singleN
  for (int32_t tempSingleN = 1024 / baseN_ * baseN_; tempSingleN > baseN_; tempSingleN -= baseN_) {
    uint32_t lastCycleCoreNum = (groupNum_ * CeilDiv(maxN_, tempSingleN)) % aicNum;
    if (lastCycleCoreNum == 0U) {
      bestSingleN = tempSingleN;
      break;
    }
    if (lastCycleCoreNum > bestLastCycleCoreNum ||
      (lastCycleCoreNum == bestLastCycleCoreNum && maxN_ % tempSingleN == 0)) {
      bestSingleN = tempSingleN;
      bestLastCycleCoreNum = lastCycleCoreNum;
    }
  }
  return bestSingleN;
}

void GMMTiling::FindBestUsedCoreNumOneGroup(const uint32_t aicNum) {
  if (groupNum_ > 1U) {
    return;
  }
  uint32_t totalCoreNums = CeilDiv(maxN_, baseN_);
  // 3: if cube iterNum less or equal to 3, and more than half cores are unused in last iter, use less cores each iter
  if ((aicNum * 3U > totalCoreNums && totalCoreNums % aicNum <= aicNum / 2U) || totalCoreNums < aicNum) {  // 2: half of aicNum
    uint32_t  cubeIterNum = CeilDiv(totalCoreNums, aicNum);
    usedCoreNum_ = CeilDiv(totalCoreNums, cubeIterNum);
  }
}


ge::graphStatus GMMTiling::SetWorkspscesPerTokenQuant(const uint32_t aicNum, size_t* workspaces) {
  if (aicNum == 0U) {  // invalid value
    return ge::GRAPH_FAILED;
  }
  bool opt = (maxM_ <= 32 * groupNum_ && wFormat_ == matmul_tiling::CubeFormat::NZ) &&
             (!transposeWeight_ || maxN_ >= 2048);  // 32: a factor, 2048: a threshold.
  if (opt) {  // non-basic strategy. matmul output in non-continugous mode with singleN >= baseN
    int32_t bestSingleN = FindBestSingleNPertoken(aicNum);
    tilingData.gmmBaseParams.set_singleN(bestSingleN);
  }
  if (isA4W4_) {
    // 4： when do cv parallelism, four pieces of workspace are used for storing four cycles of matmul output
    workspaces[0] += 4UL * baseM_ * baseN_ * usedCoreNum_ * sizeof(short); // a4w4 mmout dtype is half
  } else {
    // 4： when do cv parallelism, four pieces of workspace are used for storing four cycles of matmul output
    workspaces[0] += 4UL * baseM_ * baseN_ * usedCoreNum_ * sizeof(int32_t);
  }

  return ge::GRAPH_SUCCESS;
}

bool GMMTiling::StaticTilingProcess(gert::TilingContext *context) {
  // cond.1 A8W8
  // cond.2 singleX-singleW-singleY scenario
  // cond.3 without bias
  // cond.4 without activation
  // cond.5 no pretiling
  // cond.6 only support typeM
  // cond.7 tilingdata corresponds to expected value
  if ((xDType_ != ge::DT_INT8 || weightDtype_ != ge::DT_INT8) ||
      (isSingleX_ == 0 || isSingleWeight_ == 0 || isSingleY_ == 0) ||
      hasBias_ ||
      actType_ != 0U ||
      tilingData.gmmBaseParams.get_isPreTiling() != 0 ||
      tilingData.gmmBaseParams.get_groupType() != 0 ||
      !CheckTilingMatchStaticValue()) {
    return false;
  }
  // cond.8 only support milan platform
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "CompileInfoPtr is nullptr."), return false);
  if (!(compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93)) {
    return false;
  }

  tilingData.gmmBaseParams.set_m(tilingData.mmTilingData.get_M());
  tilingData.gmmBaseParams.set_n(tilingData.mmTilingData.get_N());
  tilingData.gmmBaseParams.set_k(tilingData.mmTilingData.get_Ka());
  return true;
}

bool GMMTiling::CheckTilingMatchStaticValue() {
  if (tilingData.mmTilingData.get_depthA1() != STATIC_TILING_DEPTH_A1_B1 ||
      tilingData.mmTilingData.get_depthB1() != STATIC_TILING_DEPTH_A1_B1 ||
      tilingData.mmTilingData.get_stepM() != 1 ||
      tilingData.mmTilingData.get_stepN() != 1 ||
      tilingData.mmTilingData.get_stepKa() != STATIC_TILING_STEP_KA_KB ||
      tilingData.mmTilingData.get_stepKb() != STATIC_TILING_STEP_KA_KB ||
      tilingData.mmTilingData.get_dbL0A() != DOUBLE_BUFFER_L0A_L0B ||
      tilingData.mmTilingData.get_dbL0B() != DOUBLE_BUFFER_L0A_L0B ||
      tilingData.mmTilingData.get_dbL0C() != 1 ||
      maxK_ > STATIC_TILING_MAX_K) {
    return false;
  }
  if (tilingData.mmTilingData.get_baseM() == BASIC_BLOCK_SIZE_128 &&
      tilingData.mmTilingData.get_baseN() == BASIC_BLOCK_SIZE_256 &&
      tilingData.mmTilingData.get_baseK() == BASIC_BLOCK_SIZE_128) {
        return true;
  }
  return false;
}

ge::graphStatus GMMTiling::RunFusionKernelTiling(gert::TilingContext* context) {
  OP_LOGI(context->GetNodeName(), "Begin Run GMM Tiling");
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);  // check compileInfoPtr is not null

  ubSize_ = compileInfoPtr->ubSize;  // get ubSize from compileInfo
  const uint32_t& aicNum = compileInfoPtr->aicNum;  // get aicNum from compileInfo
  if (aicNum == 0U) {  // invalid value
    return ge::GRAPH_FAILED;
  }
  usedCoreNum_ = aicNum;

  OP_CHECK_IF(CalMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM CalMMTiling failed"), return ge::GRAPH_FAILED);

  OP_CHECK_IF(GMMSetMMTiling(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM GMMSetMMTiling failed"),
             return ge::GRAPH_FAILED);
  tilingData.gmmBaseParams.set_singleN(0);  // 0 is the default value
  FullLoadK(compileInfoPtr);
  OP_CHECK_IF(DivideUbAndSetWorkspace(context, aicNum) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM DivideUbAndSetWorkspace failed"),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF(DynamicTilingSingleN(context, usedCoreNum_, compileInfoPtr) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM DynamicTilingSingleN failed"),
             return ge::GRAPH_FAILED);
  OP_CHECK_IF(IsOutputDisableL2Cache(context, compileInfoPtr) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM IsOutputDisableL2Cache failed"),
             return ge::GRAPH_FAILED);
  tilingData.gmmBaseParams.set_workspaceSize(workspacesSize_);
  tilingData.mmTilingData.set_usedCoreNum(usedCoreNum_);  // usedCoreNum is ai_core num
  tilingData.gmmBaseParams.set_coreNum(usedCoreNum_);  // ai cube number
  GMMSetTplTilingKey(context);
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->SetBlockDim(usedCoreNum_);  // block dim is the number of aicube
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  PrintTilingInfo(context);
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::PrintTilingInfo(gert::TilingContext *context) {
  OP_LOGD(context->GetNodeName(), "End Run GMM Tiling");
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: usedCoreNum is %d.", tilingData.mmTilingData.get_usedCoreNum());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: bestSingleN is %u.", tilingData.gmmBaseParams.get_singleN());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: with_offset is %d.", tilingData.gmmBaseParams.get_withOffset());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: tuning_config is %ld.", tuningConfig_);
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: Ka is %d.", tilingData.mmTilingData.get_Ka());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: Kb is %d.", tilingData.mmTilingData.get_Kb());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: baseM is %d.", tilingData.mmTilingData.get_baseM());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: baseN is %d.", tilingData.mmTilingData.get_baseN());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: baseK is %d.", tilingData.mmTilingData.get_baseK());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: depthA1 is %d.", tilingData.mmTilingData.get_depthA1());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: depthB1 is %d.", tilingData.mmTilingData.get_depthB1());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepKa is %d.", tilingData.mmTilingData.get_stepKa());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepKb is %d.", tilingData.mmTilingData.get_stepKb());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepM is %d.", tilingData.mmTilingData.get_stepM());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: stepN is %d.", tilingData.mmTilingData.get_stepN());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: isBias is %d.", tilingData.mmTilingData.get_isBias());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: transLength is %d.", tilingData.mmTilingData.get_transLength());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: iterateOrder is %d.", tilingData.mmTilingData.get_iterateOrder());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: dbL0A is %d.", tilingData.mmTilingData.get_dbL0A());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: dbL0B is %d.", tilingData.mmTilingData.get_dbL0B());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: dbL0C is %d.", tilingData.mmTilingData.get_dbL0C());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: usedL1Size is %d.", tilingData.mmTilingData.get_shareL1Size());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: usedUBSize is %d.", tilingData.mmTilingData.get_shareUbSize());
  OP_LOGD(context->GetNodeName(), "GMM_tiling_new: isOutputDisableL2Cache is %d.", tilingData.gmmBaseParams.get_isOutputDisableL2Cache());
  auto buf = (uint32_t *)context->GetRawTilingData()->GetData();
  auto bufLen = context->GetRawTilingData()->GetDataSize();
  std::ostringstream oss;
  oss << "Start to dump tiling info. tilingkey:" << context->GetTilingKey() << ", tiling data size:" << bufLen
      << ", content:";
  for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
      oss << *(buf + i) << ",";
      if (oss.str().length() > 640) { // Split according to 640 to avoid truncation
          OP_LOGD(context, "%s", oss.str().c_str());
          oss.str("");
      }
  }
  OP_LOGD(context, "%s", oss.str().c_str());
}

ge::graphStatus GMMTiling::GMMCalUbSize(const gert::TilingContext* context, uint32_t ubSize) {
  OP_CHECK_IF((ubDivideBlkNum_ == 0 || ubBlockAlign_ == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubDivideBlkNum and ubBlockAlign cannot be 0"),
             return ge::GRAPH_FAILED);
  uint32_t ubCalSize = ubSize / ubDivideBlkNum_;  // divide the UB into ubDivideBlkNum_ pieces
  ubCalSize = ubCalSize / ubBlockAlign_ * ubBlockAlign_;  // 16k/8k/4k align.
  uint32_t ubRestBytes = ubSize - ubCalSize * ubIoBlkNum_;  // compute the rest memory in UB space
  ubRestBytes = ubRestBytes / UB_BLOCK_UNIT_SIZE * UB_BLOCK_UNIT_SIZE;  // 32B align.
  OP_CHECK_IF((ubCalSize == 0 || ubRestBytes == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubCalSize and ubRestBytes cannot be 0"),
             return ge::GRAPH_FAILED);
  uint32_t ubBaseN = 0;  // init
  uint32_t ubBaseK = 0;  // init
  uint32_t ubBaseM = 0;  // init
  if (transposeWeight_) {
    ubBaseK = static_cast<uint32_t>(BEST_UB_BASEK);
    ubBaseN = ubCalSize / ubBaseK;
    uint32_t alignFactor = UB_BLOCK_UNIT_SIZE;
    if (weightDtype_ == ge::DT_INT4) {
      alignFactor <<= 1U;  // int4 need 64 elements align.
    }
    ubBaseN = ubBaseN / alignFactor * alignFactor;
  } else {
    if ((xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT16) &&
        (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
      if (perTokenOrPerGroupSize_ > 0U) {
        ubBaseK = perTokenOrPerGroupSize_;
        ubBaseN = std::min<uint32_t>(BEST_UB_BASEN, std::max<uint32_t>(MIN_UB_BASEN, (ubCalSize / ubBaseK + MIN_UB_BASEN - 1) / MIN_UB_BASEN * MIN_UB_BASEN));
      } else if (antiquantPerformance_) {
        ubBaseN = static_cast<uint32_t>(BEST_UB_BASEN);
      } else {
        ubBaseN = static_cast<uint32_t>(baseN_);
      }
    } else {
      ubBaseN = static_cast<uint32_t>(baseN_);
    }
    ubBaseK = ubCalSize / ubBaseN;  // ubCalSize is the number of elements, not in bytes unit.
    ubBaseM = ubCalSize / ubBaseN;
  }
  if (xDType_ == ge::DT_BF16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4) && !isA16W8Msd_) {
    OP_CHECK_IF(ubBaseK == 0 || ubBaseN == 0,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "ubBaseK or ubBaseN cannot be 0"),
               return ge::GRAPH_FAILED);
  }
  tilingData.gmmBaseParams.set_ubCalSize(ubCalSize);
  tilingData.gmmBaseParams.set_ubRestBytes(ubRestBytes);  // in byte unit
  tilingData.gmmBaseParams.set_ubBaseK(ubBaseK);
  tilingData.gmmBaseParams.set_ubBaseN(ubBaseN);
  tilingData.gmmBaseParams.set_vBaseM(ubBaseM);
  return ge::GRAPH_SUCCESS;
}

int64_t GMMTiling::GMMGetBS(const gert::Shape &xShape) const {
    int64_t bs = 0;  // init bs
    if (transposeX_) {
      bs = xShape.GetDim(1);  // x shape is [k, m] if x is transpose_
    } else {
      if (groupType_ == -1) {  // -1: no group case, may exits a situation that multi dims product equals to bs.
        bs = xShape.GetDim(0);  // 0: x first dim
        size_t bsDimNum = xDimNum_ >= 1U ? xDimNum_ - 1UL : 0UL;  // 1: x last dim k, the other dimensions are bs
        for (size_t i = 1; i < bsDimNum; i++) {
            bs *= xShape.GetDim(i);
        }
      } else {
        bs = xShape.GetDim(0);  // in group case，x's shapeis [m,k], 0 is the m axis.
      }
    }
    return bs;
}

uint32_t GMMTiling::GetTplDataType(const ge::DataType &dtype) {
  static const std::map<ge::DataType, uint32_t> dtype_map = {
    {ge::DT_INT4, GMM_TPL_INT4},
    {ge::DT_INT8, GMM_TPL_INT8},
    {ge::DT_FLOAT16, GMM_TPL_FLOAT16},
    {ge::DT_BF16, GMM_TPL_BF16},
    {ge::DT_INT32, GMM_TPL_INT32},
    {ge::DT_FLOAT, GMM_TPL_FLOAT}
  };
  auto it = dtype_map.find(dtype);
  if (it != dtype_map.end()) {
    return it->second;
  }
  return GMM_TPL_INVALID;
}

bool GMMTiling::IsAivAicRatioTwoRequired() {
    // Must be valid group type (not 2)
    if (groupListType_ == GROUP_LIST_SPARSE_M) {
      return false;
    }

    // Condition 1: GELU activation (immediate match)
    if (actType_ == ACT_TYPE_GELU) {
      return true;
    }

    // Condition 2: Complex tuning configuration requiring:
    // - K dimension outside normal vectorization range
    // - Minimum tuning configuration threshold
    // - Valid token/group size
    const bool needs_double_vector = (maxK_ <= DOUBLE_VECTOT_THRESHOLD_K_LOWER) ||
                                    (maxK_ >= DOUBLE_VECTOT_THRESHOLD_K_UPPER);
    const bool has_sufficient_tuning = (tuningConfig_ >= SMALL_TUNING_CONFIG_THRESHOLD);
    const bool has_valid_workload = (perTokenOrPerGroupSize_ > 0U);

    return needs_double_vector && has_sufficient_tuning && has_valid_workload;
}

bool GMMTiling::IsFixedAxisMoveCondition() {
    bool isCorrectShape = (maxK_ == FIXAXISMOVE_K1 && maxN_ == FIXAXISMOVE_N1) ||
                          (maxK_ == FIXAXISMOVE_K2 && maxN_ == FIXAXISMOVE_N2);
    bool isGroupCorrect = (groupNum_ == FIXAXISMOVE_GROUP_NUM);
    bool isTuningInRange = (tuningConfig_ >= FIXAXISMOVE_PERM_LOWER) &&
                          (tuningConfig_ <= FIXAXISMOVE_PERM_UPPER);
    bool isDataTypeCorrect = yDtype_ == ge::DT_FLOAT16 && scaleDtype_ == ge::DT_FLOAT && perTokenScaleDtype_ == ge::DT_FLOAT;
    bool isConfigCorrect = !transposeX_ && (splitItem_ == FIXAXISMOVE_SPLIT_ITEM2 || splitItem_ == FIXAXISMOVE_SPLIT_ITEM3)
                          && (groupListType_ == FIXAXISMOVE_GROUP_LIST_TYPE)
                          && (groupType_ == FIXAXISMOVE_GROUP_TYPE) && (actType_ == 0)
                          && !transposeWeight_;
    bool isWorkspaceValid = (static_cast<int64_t>(FixedAxisMoveWorkspace_) <= tuningConfigWorkspace_) ||
                           (tuningConfigWorkspace_ == -1);
    bool isFormatValid = (wFormat_ == matmul_tiling::CubeFormat::NZ);

    return isCorrectShape && isTuningInRange && isGroupCorrect && isA8W8_ &&
           isDataTypeCorrect && isConfigCorrect && isWorkspaceValid && !hasBias_ && isFormatValid;
}

bool GMMTiling::IsIntDataType() {
    return yDtype_ == ge::DT_INT8 || yDtype_ == ge::DT_INT32;
}

void GMMTiling::GMMSetTplTilingKey(gert::TilingContext* context) {
  uint32_t isStaticTilingApi = 0;
  uint32_t a8w4KernelTemplate = GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE;
  uint32_t a16w8KernelTemplate = GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE;
  uint32_t aivAicRatio = GROUPED_MATMUL_AIV_AIC_RATIO_1;
  uint32_t isEnableFixedAxis = 0;

  if (isA8W4FakeA8W8_) {
    a8w4KernelTemplate = static_cast<uint32_t>(GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERCHANNEL_ANTIQUANT);
  }

  if ((xDType_ == ge::DT_FLOAT16 || xDType_ == ge::DT_BF16) && weightDtype_ == ge::DT_INT8) {
    a16w8KernelTemplate = static_cast<uint32_t>(GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_ANTIQUANT);
    if (isA16W8Msd_) {
      a16w8KernelTemplate = static_cast<uint32_t>(GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_MSD);
    }
  }

  if (isA4W4_ || antiquantPerformance_) {
    aivAicRatio = static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_2);
  } else if (isA8W8_) {
    if (isFixedAxisMove_) {
      aivAicRatio = static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_1);
      isEnableFixedAxis = 1;
    } else if (IsAivAicRatioTwoRequired()) {
      aivAicRatio = static_cast<uint32_t>(GROUPED_MATMUL_AIV_AIC_RATIO_2);
    } else if (IsIntDataType()) {
      aivAicRatio = static_cast<uint32_t>(GROUPED_MATMUL_CUBE_ONLY);
    }
  } else if (!transposeX_ && xDType_ == weightDtype_ && (xDType_ == ge::DT_FLOAT16 || xDType_ == ge::DT_BF16 || xDType_ == ge::DT_FLOAT)) {
    aivAicRatio = static_cast<uint32_t>(GROUPED_MATMUL_CUBE_ONLY);
  }

  if (a8w4KernelTemplate == GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE &&
      a16w8KernelTemplate == GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE &&
      aivAicRatio != GROUPED_MATMUL_AIV_AIC_RATIO_2 &&
      !isFixedAxisMove_ &&
      StaticTilingProcess(context)) {
    isStaticTilingApi = 1U;
  }

  const uint64_t tilingKey = GET_TPL_TILING_KEY(GetTplDataType(xDType_),
                                                GetTplDataType(weightDtype_),
                                                GetTplDataType(yDtype_),
                                                transposeX_? 1UL : 0UL,
                                                transposeWeight_? 1UL : 0UL,
                                                groupListType_,
                                                isStaticTilingApi,
                                                a8w4KernelTemplate,
                                                a16w8KernelTemplate,
                                                aivAicRatio,
                                                isEnableFixedAxis);
  context->SetTilingKey(tilingKey);

  if (isA16W8Msd_ || antiquantPerformance_) {
    context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
  }
}

ge::graphStatus GMMTiling::GMMGetAttrs(const gert::TilingContext* context) {
  auto attr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attr);  // check attr is not null
  const bool* transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
  const bool* transposeXPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
  const int64_t* groupTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
  const int64_t* splitItemPtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
  const int64_t* actTypePtr = attr->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);
  const uint32_t* groupListTypePtr = attr->GetAttrPointer<uint32_t>(ATTR_INDEX_GROUP_LIST_TYPE);
  const auto tuningConfigPtr = attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG);
  transposeWeight_ = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
  transposeX_ = transposeXPtr != nullptr ? *transposeXPtr : false;
  groupType_ = groupTypePtr != nullptr ? *groupTypePtr : NO_SPLIT;
  splitItem_ = splitItemPtr != nullptr ? *splitItemPtr : 0U;  // 0: 默认split_item
  actType_ = actTypePtr != nullptr ? *actTypePtr : 0;
  groupListType_ = groupListTypePtr != nullptr ? *groupListTypePtr : 0;

  auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);  // check xDesc is not null
  xDType_ = xDesc->GetDataType();
  auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
  weightDtype_ = w0Desc->GetDataType();
  if (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT4) {
    // scale layout: pergroup [e, G, n] (G = quantGroupNum) or perchannel [e, n]
    // (= [e, 1, n] in memory, quantGroupNum degrades to 1); the kernel's scale
    // addressing is identical in both cases.
    auto scale0Shape = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape();
    uint32_t scaleDimNum = scale0Shape.GetDimNum();
    uint64_t n = 0;
    uint64_t quantGroupNum = 1;
    if (scaleDimNum == 3U) {
      n = scale0Shape.GetDim(2);
      quantGroupNum = scale0Shape.GetDim(1);
    } else if (scaleDimNum == 2U) {
      n = scale0Shape.GetDim(1);
    } else {
      OP_LOGE(context->GetNodeName(), "GMM A8W4: scale dim should be 2 (perchannel [e, n]) or 3 "
              "(pergroup [e, G, n]), but now is %u", scaleDimNum);
      return ge::GRAPH_FAILED;
    }
    const uint64_t k = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
    const uint64_t groupNum = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(0);
    isA8W4FakeA8W8_ = true;
    A8W4noMsdSpace_ = groupNum * k * n * sizeof(int8_t) + groupNum * n * sizeof(float);
    tilingData.gmmBaseParams.set_groupNum(groupNum);
    tilingData.gmmBaseParams.set_n(n);
    tilingData.gmmBaseParams.set_k(k);
    tilingData.gmmBaseParams.set_quantGroupNum(quantGroupNum);
  }
  isA8W8_ = (xDType_ == ge::DT_INT8 && weightDtype_ == ge::DT_INT8);
  isA4W4_ = xDType_ == ge::DT_INT4 && weightDtype_ == ge::DT_INT4;
  isA16W16_ = (xDType_ == ge::DT_FLOAT16 && weightDtype_ == ge::DT_FLOAT16) || (xDType_ == ge::DT_BF16 && weightDtype_ == ge::DT_BF16);
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);  // check compileInfoPtr is not null
  if (groupListType_ == GROUP_LIST_SPARSE_M) {
    OP_CHECK_IF((!(compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
                  compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93)),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "This platform not support groupListType is 2"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!isA8W8_,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "Only both x and weight dtype are int8 support groupListType is 2"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupType_ != SPLIT_M,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "When groupListType is 2 only support groupType 0, but get groupType %ld",
                                           groupType_),
               return ge::GRAPH_FAILED);
  }

  auto perTokenScalePtr = context->GetOptionalInputTensor(PER_TOKEN_SCALE_INDEX);
  if (perTokenScalePtr != nullptr && perTokenScalePtr->GetStorageShape().GetShapeSize() != 0) {
    perTokenOrPerGroupSize_ = 1U;
  }
  auto pertokenScaleDesc = context->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
  if (pertokenScaleDesc != nullptr) {
    perTokenScaleDtype_ = pertokenScaleDesc->GetDataType();
  }
  tilingData.gmmBaseParams.set_quantParam(perTokenOrPerGroupSize_);
  auto yDesc = context->GetOutputDesc(Y_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
  yDtype_ = yDesc->GetDataType();
  if ((weightDtype_ == ge::DT_INT8 && xDType_ == ge::DT_INT8 && yDtype_ != ge::DT_INT32) || isA8W4FakeA8W8_ ||
      (xDType_ == ge::DT_FLOAT8_E4M3FN) || (xDType_ == ge::DT_FLOAT8_E5M2)) {
      auto scale0Desc = context->GetDynamicInputDesc(SCALE_INDEX, 0);
      OP_CHECK_NULL_WITH_CONTEXT(context, scale0Desc);
      scaleDtype_ = scale0Desc->GetDataType();
  }

  auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
  wFormat_ = wFormat0 == ge::FORMAT_FRACTAL_NZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND;
  tuningConfig_ = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX) ?
                  (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX] : 0;
  tuningConfigWorkspace_ = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_ALLOW_WORKSPACE_INDEX) ?
                  (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_ALLOW_WORKSPACE_INDEX] : 0;

  if (((compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
        compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93)) &&
    isA4W4_ ){
      OP_CHECK_IF( !((wFormat0 == ge::FORMAT_FRACTAL_NZ) || (wFormat0 != FORMAT_FRACTAL_NZ && transposeWeight_ == 0)),
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                           "A4W4 GMM currently supports only weight tensor nz transpose/untranspose input \
or nd format untranspose input."
                                           ),
                return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkAntiquant() {
  if (isA16W8Msd_) {
    ubDivideBlkNum_ = UB_A16W8_MSD_BLOCK_NUM;
    ubIoBlkNum_ = UB_A16W8_MSD_IO_USED_BLOCK;
    ubBlockAlign_ = UB_A16W8_MSD_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (xDType_ == ge::DT_FLOAT16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
    if (weightDtype_ == ge::DT_INT8) {
      ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_FP16;
      ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_FP16;
    } else {  // int4
      ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_FP16;
      ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_FP16;
    }
    ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if (xDType_ == ge::DT_BF16 && (weightDtype_ == ge::DT_INT8 || weightDtype_ == ge::DT_INT4)) {
    if (weightDtype_ == ge::DT_INT8) {
      ubDivideBlkNum_ = UB_A16W8_BLOCK_NUM_BF16;
      ubIoBlkNum_ = UB_A16W8_IO_USED_BLOCK_BF16;
    } else {
      ubDivideBlkNum_ = UB_A16W4_BLOCK_NUM_BF16;
      ubIoBlkNum_ = UB_A16W4_IO_USED_BLOCK_BF16;
    }
    ubBlockAlign_ = UB_ANTIQUANT_PER_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkQuant() {
  if ((weightDtype_ == ge::DT_INT8 || isA8W4FakeA8W8_) && (perTokenOrPerGroupSize_ == 1 || actType_ != 0)) {
    // include case per-token without activation, per-token with activation and per-tensor with activation
    ubDivideBlkNum_ = UB_DYNAMIC_QUANT_BLOCK_NUM;
    ubIoBlkNum_ = UB_DUNAMIC_QUANT_IO_USED_BLOCK;
    ubBlockAlign_ = UB_QUANT_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  if ((weightDtype_ == ge::DT_INT8 || isA8W4FakeA8W8_) && perTokenOrPerGroupSize_ != 1) {
    // include case per-tensor without activation
    if (yDtype_ == ge::DT_FLOAT16) {
      ubDivideBlkNum_ = UB_STATIC_QUANT_BLOCK_NUM_FP16;
    } else {
      ubDivideBlkNum_ = UB_STATIC_QUANT_BLOCK_NUM_BF16;
    }
    ubIoBlkNum_ = UB_STATIC_QUANT_IO_USED_BLOCK;
    ubBlockAlign_ = UB_QUANT_BLOCK_ALIGN;
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlkA4W4() {
  ubDivideBlkNum_ = UB_A4W4_BLOCK_NUM;
  ubIoBlkNum_ = UB_A4W4_IO_USED_BLOCK_HALF;
  ubBlockAlign_ = UB_A4W4_PER_BLOCK_ALIGN;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetUbDivideBlk() {
  ubDivideBlkNum_ = 0U;  // init ubDivideBlkNum_
  ubIoBlkNum_ = 0U;  // init ubIoBlkNum_
  ubBlockAlign_ = 0U;  // init ubBlockAlign_
  if (xDType_ == ge::DT_INT8) {
    return GMMSetUbDivideBlkQuant();
  } else if (isA4W4_) {
    return GMMSetUbDivideBlkA4W4();
  } else {
    return GMMSetUbDivideBlkAntiquant();
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus GMMTiling::SetBias(const gert::TilingContext* context, matmul_tiling::MultiCoreMatmulTiling& mm) const {
  if (!hasBias_ || isA16W8Msd_ || isA4W4_) {
    mm.SetBias(false);
  } else {
    mm.SetBias(true);
    auto biasTensor = context->GetDynamicInputTensor(BIAS_INDEX, 0);
    OP_CHECK_IF(biasTensor == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "Get bias tensor failed."),
               return ge::GRAPH_FAILED);
    mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   static_cast<matmul_tiling::DataType>(biasTensor->GetDataType()));
  }
  return ge::GRAPH_SUCCESS;
}

static void InitPlatformInfo(const GMMCompileInfo* compileInfoPtr, matmul_tiling::PlatformInfo& platformInfo) {
  platformInfo.socVersion = compileInfoPtr->socVersion;
  platformInfo.l1Size = compileInfoPtr->l1Size;
  platformInfo.l0CSize = compileInfoPtr->l0CSize;
  platformInfo.ubSize = compileInfoPtr->ubSize;
  platformInfo.l0ASize = compileInfoPtr->l0ASize;
  platformInfo.l0BSize = compileInfoPtr->l0BSize;
}

void GMMTiling::FullLoadK(const GMMCompileInfo* compileInfoPtr) {
  if (wFormat_ == matmul_tiling::CubeFormat::ND && !transposeWeight_ && !transposeX_ && xDType_ == weightDtype_ &&
      (weightDtype_ == ge::DT_FLOAT16 || weightDtype_ == ge::DT_BF16) &&
      maxM_ >= FULL_K_M_E_THRESHOLD * groupNum_ && maxN_ == FULL_K_N_THRESHOLD &&
      maxK_ <= FULL_K_MAX_K_THRESHOLD &&
      maxK_ >= FULL_K_MIN_K_THRESHOLD) {
    int64_t fullLoadStepKa = CeilDiv(maxK_ , baseK_);
    int64_t fullLoadStepKb = fullLoadStepKa / static_cast<int64_t>(QUEUE_DOUBLE_BUFFER);
    int64_t fullLoadDepthKa = fullLoadStepKa * static_cast<int64_t>(QUEUE_DOUBLE_BUFFER);
    int64_t fullLoadDepthKb = fullLoadStepKb * static_cast<int64_t>(QUEUE_DOUBLE_BUFFER);
    bool ifFullLoad = (maxM_ > FULL_K_M_THRESHOLD) && isAllSingleTensor_ && groupType_ == SPLIT_M &&
                      (((baseM_ * baseK_ * static_cast<int64_t>(mmDataTypeSize_)) * fullLoadDepthKa +
                        (baseN_ * baseK_ * static_cast<int64_t>(mmDataTypeSize_)) * fullLoadDepthKb) <=
                      static_cast<int64_t>(compileInfoPtr->l1Size));
    if (ifFullLoad) {
      tilingData.mmTilingData.set_stepKa(fullLoadStepKa);  // set precomputed mmStepKa
      tilingData.mmTilingData.set_depthA1(fullLoadDepthKa);  // set precomputed mmDepthA1
      tilingData.mmTilingData.set_stepKb(fullLoadStepKb);  // set precomputed mmStepKb
      tilingData.mmTilingData.set_depthB1(fullLoadDepthKb);  // set precomputed mmDepthB1
      tilingData.mmTilingData.set_iterateOrder(1);  // set precomputed stepN
      tilingData.gmmBaseParams.set_singleN(FULL_K_SINGLE_N);  // 0 is the default value
    }
  }
}

ge::graphStatus GMMTiling::CalcStepKaKb(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr,
                                        int64_t mInMM, uint32_t& mmStepKa, uint32_t& mmStepKb) {
  uint64_t availableL1Size = compileInfoPtr->l1Size;
  if (isA8W8_ || isA8W4FakeA8W8_) {
    availableL1Size -= static_cast<uint64_t>(baseN_) * sizeof(uint64_t);
  }
  if (hasBias_) {
    availableL1Size -= static_cast<uint64_t>(baseN_) * static_cast<uint64_t>(4);  // 4: size of float32 or int32
  }
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
    availableL1Size = BEST_L1_PARTA + BEST_L1_PARTB;
  }
  OP_CHECK_IF(availableL1Size < L1_PARTA_SIZE,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "availableL1Size is less than 256k"),
             return ge::GRAPH_FAILED);
  // according to double buffer, recompute the params used for data movement from GM to L1
  uint64_t l1ASize = baseM_ > baseN_ ? L1_PARTA_SIZE : availableL1Size - L1_PARTA_SIZE;
  uint64_t l1BSize = availableL1Size - l1ASize;
  if (isA4W4_) {
    // 2: double buffer
    mmStepKa = static_cast<uint32_t>((l1ASize / 2UL) / static_cast<uint64_t>(INT4_DATA_TYPE_SIZE *
                                                                             static_cast<float>(baseM_) *
                                                                             static_cast<float>(baseK_)));
    // 2: double buffer
    mmStepKb = static_cast<uint32_t>((l1BSize / 2UL) / static_cast<uint64_t>(INT4_DATA_TYPE_SIZE *
                                                                             static_cast<float>(baseN_) *
                                                                             static_cast<float>(baseK_)));
  } else {
    // 2: double buffer
    mmStepKa = (l1ASize / 2UL) / (static_cast<uint64_t>(baseM_) * baseK_ * mmDataTypeSize_);
    // 2: double buffer
    mmStepKb = (l1BSize / 2UL) / (static_cast<uint64_t>(baseN_) * baseK_ * mmDataTypeSize_);
  }
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P &&
                                    wFormat_ == matmul_tiling::CubeFormat::NZ && mInMM <= baseM_) {
    mmStepKa = std::min<uint32_t>(mmStepKa, std::max(1, 128 / baseK_));  // 128: nz inner block size. In practice, baseK_*mmStepKa=128 makes performance better.
  }

  OP_CHECK_IF(mmStepKa == 0 || mmStepKb == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "stepka or stepkb cannot be 0"),
             return ge::GRAPH_FAILED);

  if (mmStepKa > mmStepKb) {
    mmStepKa = mmStepKa / mmStepKb * mmStepKb;
  } else if (mmStepKa < mmStepKb) {
    mmStepKb = mmStepKb / mmStepKa * mmStepKa;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::GMMSetMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  matmul_tiling::DataType matmulDtype = static_cast<matmul_tiling::DataType>(mmDType_);
  matmul_tiling::PlatformInfo platformInfo;
  InitPlatformInfo(compileInfoPtr, platformInfo);
  matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
  int64_t mInMM = isA16W8Msd_ ? static_cast<int64_t>(A16W8_MSD_STEP) * maxM_ : maxM_;  // if msd, m in matmul should mul steps
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
  mm.SetBType(matmul_tiling::TPosition::GM, wFormat_, matmulDtype, false);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT16);
  OP_CHECK_IF(SetBias(context, mm) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetBias failed."), return ge::GRAPH_FAILED);
  mm.SetOrgShape(mInMM, maxN_, maxK_);
  mm.SetShape(mInMM, baseN_, maxK_);
  mm.SetFixSplit(baseM_, baseN_, baseK_);
  mm.SetBufferSpace(compileInfoPtr->l1Size, compileInfoPtr->l0CSize, ubSize_);
  if(groupType_ == SPLIT_K && maxK_ == 0) {
    //切K场景K轴为0时不需要使用高阶API，无需tiling，kernel需要做清零处理，此处无需tiling
    OP_LOGD("GMM SplitK,and K == 0, no need tiling");
  } else {
    OP_CHECK_IF(mm.GetTiling(tilingData.mmTilingData) == -1,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "matmul getTiling failed."),
            return ge::GRAPH_FAILED);
  }


  uint32_t mmStepKa = 1;
  uint32_t mmStepKb = 1;
  OP_CHECK_IF(CalcStepKaKb(context, compileInfoPtr, mInMM, mmStepKa, mmStepKb) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "matmul calc stepka or stepkb failed."),
             return ge::GRAPH_FAILED);

  constexpr uint32_t stepM = 1;  // 1: stepM set fixed value 1
  constexpr uint32_t stepN = 1;  // 1: stepN set fixed value 1
  uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
  uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
  tilingData.mmTilingData.set_shareMode(0);
  if (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
    tilingData.mmTilingData.set_shareUbSize(0);
    tilingData.mmTilingData.set_transLength(131072);  // 131072: 128KB size
  }
  tilingData.mmTilingData.set_dbL0C(1);  // disable double buffer for LOC
  tilingData.mmTilingData.set_baseM(baseM_);  // set precomputed baseM
  tilingData.mmTilingData.set_baseN(baseN_);  // set precomputed baseN
  tilingData.mmTilingData.set_baseK(baseK_);  // set precomputed baseK
  tilingData.mmTilingData.set_stepKa(mmStepKa);  // set precomputed mmStepKa
  tilingData.mmTilingData.set_depthA1(mmDepthA1);  // set precomputed mmDepthA1
  tilingData.mmTilingData.set_stepKb(mmStepKb);  // set precomputed mmStepKb
  tilingData.mmTilingData.set_depthB1(mmDepthB1);  // set precomputed mmDepthB1
  tilingData.mmTilingData.set_stepM(stepM);  // set precomputed stepM
  tilingData.mmTilingData.set_stepN(stepN);  // set precomputed stepN
  SetMMPreTiling();
  OP_LOGI(context->GetNodeName(), "GMM_tiling: baseM is %d, baseK is %d, baseN is %d.", baseM_, baseK_, baseN_);
  return ge::GRAPH_SUCCESS;
}

void GMMTiling::SetMMPreTiling() {
  uint64_t ispreTiling = 0;
  int64_t isNz = wFormat_ ==  matmul_tiling::CubeFormat::NZ ? 1 : 0;
  if (tuningConfig_ == 0L && (isA8W8_ || isA8W4FakeA8W8_) && groupNum_ == 1U && usedCoreNum_ == A3_AIC_NUM) {
    ispreTiling = static_cast<uint64_t>(1); // 1: pretiling key
  } else {
    tilingData.gmmBaseParams.set_isPreTiling(ispreTiling);
    return;
  }
  std::array<int64_t, 4> mKNList = {maxM_, maxK_, maxN_, isNz}; // 4: input shape info size
  if (A8W8_PRETILING_WHITE_LIST.count(mKNList)) {
    int64_t newBaseM = A8W8_PRETILING_WHITE_LIST.at(mKNList)[0];
    int64_t bestSingleN = A8W8_PRETILING_WHITE_LIST.at(mKNList)[1];
    tilingData.mmTilingData.set_baseM(newBaseM);  // set pretiling baseM
    tilingData.mmTilingData.set_singleCoreN(bestSingleN);  // set pretiling singleN
    ispreTiling = static_cast<uint64_t>(2); // 2: white list pretiling key
  }
  tilingData.gmmBaseParams.set_isPreTiling(ispreTiling);
  return;
}

ge::graphStatus GMMTiling::CalMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  // if tuningConfig_ in (128, 256], recompute tiling.
  // or y:int32 and tuningConfig_ in (0, 128], k,n:(7168, 2048) or k,n:(7680, 2048), which got by actual measurement)
  constexpr int32_t tuningConfigBaseLowerLimit = 128;
  constexpr int32_t tuningConfigBaseUpperLimit = 256;
  constexpr int32_t maxKLimit = 7168;
  constexpr int32_t altKLimit = 7680;
  constexpr int32_t maxNLimit = 2048;

  bool tuningConfigFlag = (tuningConfig_ > tuningConfigBaseLowerLimit && tuningConfig_ <= tuningConfigBaseUpperLimit)
    || (yDtype_ == ge::DT_INT32 && tuningConfig_ > 0 && tuningConfig_ <= tuningConfigBaseLowerLimit && (maxK_ == maxKLimit || maxK_ == altKLimit) && maxN_ == maxNLimit);

  baseN_ = BEST_BASEN; // init
  // 2048: min n for a16w8 msd to set baseN 512
  if (isA16W8Msd_ && maxN_ >= 2048 && !transposeWeight_) {
    baseN_ = BEST_BASEN_MSD;
  } else if ((isA8W8_ || isA8W4FakeA8W8_) && tuningConfigFlag){
    baseN_ = BEST_BASEN_QUANT_ONE_GROUP;
    baseM_ = BEST_BASEM_QUANT_ONE_GROUP;
    baseK_ = BEST_BASEK_QUANT_ONE_GROUP;
    baseM_ = baseM_ > maxM_ ? static_cast<int32_t>(SixteenAlign(maxM_, true)) : baseM_;
    return ge::GRAPH_SUCCESS;
  } else if (isA4W4_) {
    baseN_ = tuningConfig_ > 64 ? BEST_BASEN : BEST_BASEN_A4W4; // 64 : when token in each group > 64, set baseN to 256
  } else {
    baseN_ = BEST_BASEN;
  }
  if (groupType_ == SPLIT_K && isSingleX_ && isSingleWeight_ &&
             isA16W16_ &&
             (compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910B ||
             compileInfoPtr->socVersion == platform_ascendc::SocVersion::ASCEND910_93) &&
             maxM_ > (maxN_ * SPLITK_M_N_RATIO_THRESHOLD_2)) {
    //切K场景左矩阵转置，尾轴为M轴，右矩阵不转置。
    //左矩阵搬运也希望尾轴512B对齐，此时如果M较大，可以考虑将baseM:256,baseN:128,baseK:64
    baseM_ = SPLITK_BASEM_256;
    baseN_ = SPLITK_BASEN_128;
    baseK_ = SPLITK_BASEK_64;
    baseM_ = baseM_ > maxM_ ? static_cast<int32_t>(SixteenAlign(maxM_, true)) : baseM_;
    return ge::GRAPH_SUCCESS;
  }
  // according to the double buffer enabled L0B, compute baseK
  baseK_ = isA4W4_ ? static_cast<int32_t>((compileInfoPtr->l0BSize / DOUBLE_BUFFER_L0A_L0B) /
                                          static_cast<uint32_t>(baseN_ * INT4_DATA_TYPE_SIZE)) :
                     static_cast<int32_t>((compileInfoPtr->l0BSize / DOUBLE_BUFFER_L0A_L0B) /
                                          (static_cast<uint32_t>(baseN_) * mmDataTypeSize_));
  baseK_ = static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(baseK_)));
  OP_CHECK_IF(baseK_ == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "baseK_ cannot be 0."),
             return ge::GRAPH_FAILED);
  // according to the double buffer enabled L0A/L0C, compute baseM(cube)
  uint32_t maxBaseM = static_cast<uint32_t>(compileInfoPtr->l0CSize /
                                            (static_cast<uint32_t>(baseN_) * FP32_DATATYPE_SIZE));
  baseM_ = isA4W4_ ?
           std::min<uint32_t>((compileInfoPtr->l0ASize / DOUBLE_BUFFER_L0A_L0B) /
           static_cast<uint32_t>(baseK_ * INT4_DATA_TYPE_SIZE), maxBaseM) :
           std::min<uint32_t>((compileInfoPtr->l0ASize / DOUBLE_BUFFER_L0A_L0B) /
           (static_cast<uint32_t>(baseK_) * mmDataTypeSize_), maxBaseM);

  if (!isA16W8Msd_) {
    baseM_ = baseM_ > maxM_ ? SixteenAlign(maxM_, true) : SixteenAlign(static_cast<uint32_t>(baseM_));
  } else {
    baseM_ = baseM_ > A16W8_MSD_STEP * maxM_ ? static_cast<int32_t>(SixteenAlign(static_cast<int64_t>(A16W8_MSD_STEP) * maxM_, true)) :
                                               SixteenAlign(static_cast<int64_t>(baseM_));
  }
  if (baseM_ > MAX_BASEM) {
    baseM_ = MAX_BASEM;
  }
  OP_CHECK_IF(baseM_ == 0,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "baseM_ cannot be 0."),
             return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

static void SetA8W4HPTiling(A8W4HPTiling *tiling_data, uint32_t aicNum)
{
  constexpr int SIZE_TWO = 2;
  constexpr int SIZE_THREE = 3;
  constexpr int IDX_ZERO = 0;
  constexpr int IDX_ONE = 1;
  constexpr int IDX_TWO = 2;
  constexpr uint32_t SINGLE_CORE_TILING_0 = 128;
  constexpr uint32_t SINGLE_CORE_TILING_1 = 256;
  constexpr uint32_t SINGLE_CORE_BASE_TILING_0 = 128;
  constexpr uint32_t SINGLE_CORE_BASE_TILING_1 = 256;
  constexpr uint32_t TOTAL_K_THRESHOLD_6656 = 6656;

  uint32_t *ori_in0_shape = tiling_data->get_ori_in0_shape();
  uint32_t total_K = ori_in0_shape[IDX_ONE];
  uint32_t core_num = aicNum;
  uint32_t splitRecord[SIZE_THREE] = {1, 1, 1};
  uint32_t single_core_tiling[SIZE_THREE] = {SINGLE_CORE_TILING_0, SINGLE_CORE_TILING_1, total_K};
  uint32_t single_core_base_tiling[SIZE_TWO] = {SINGLE_CORE_BASE_TILING_0, SINGLE_CORE_BASE_TILING_1};

  tiling_data->set_required_core_num(core_num);
  tiling_data->set_kernel_index(0);
  tiling_data->set_splitTimes(0);

  if (total_K > TOTAL_K_THRESHOLD_6656) {
    splitRecord[IDX_ZERO] = CeilDiv(total_K, TOTAL_K_THRESHOLD_6656);
    single_core_tiling[IDX_TWO] = TOTAL_K_THRESHOLD_6656;
  }

  tiling_data->set_splitRecord(splitRecord);
  tiling_data->set_single_core_tiling(single_core_tiling);
  tiling_data->set_single_core_base_tiling(single_core_base_tiling);
}

static void PrintA8W4HPTiling(gert::TilingContext* context, A8W4HPTiling *data)
{
  constexpr int TWO = 2;
  OP_LOGD(context->GetNodeName(), "Tiling data strategy: ");
  OP_LOGD(context->GetNodeName(), "  group_num=%u", data->get_group_num());
  OP_LOGD(context->GetNodeName(), "  group_type=%hhd", data->get_group_type());
  OP_LOGD(context->GetNodeName(), "  required_core_num=%u", data->get_required_core_num());
  OP_LOGD(context->GetNodeName(), "  format_in=%f", data->get_format_in());
  OP_LOGD(context->GetNodeName(), "  format_out=%f", data->get_format_out());
  OP_LOGD(context->GetNodeName(), "  numAic=%u", data->get_numAic());
  OP_LOGD(context->GetNodeName(), "  numAiv=%u", data->get_numAiv());
  OP_LOGD(context->GetNodeName(), "  szUb=%llu", static_cast<long long unsigned int>(data->get_szUb()));
  OP_LOGD(context->GetNodeName(), "  szL0A=%llu", static_cast<long long unsigned int>(data->get_szL0A()));
  OP_LOGD(context->GetNodeName(), "  szL0C=%llu", static_cast<long long unsigned int>(data->get_szL0C()));
  OP_LOGD(context->GetNodeName(), "  pattern=%hhu", data->get_pattern());
  OP_LOGD(context->GetNodeName(), "  kernel_index=%hhu", data->get_kernel_index());
  OP_LOGD(context->GetNodeName(), "  splitTimes=%u", data->get_splitTimes());
  OP_LOGD(context->GetNodeName(), "  output_type=%hhd", data->get_output_type());
  OP_LOGD(context->GetNodeName(), "  ori_in0_shape=[%u,%u]", data->get_ori_in0_shape()[0],
            data->get_ori_in0_shape()[1]);
  OP_LOGD(context->GetNodeName(), "  ori_in1_shape=[%u,%u]", data->get_ori_in1_shape()[0],
            data->get_ori_in1_shape()[1]);
  OP_LOGD(context->GetNodeName(), "  ori_out_shape=[%u,%u]", data->get_ori_out_shape()[0],
            data->get_ori_out_shape()[1]);
  OP_LOGD(context->GetNodeName(), "  single_core_tiling=[%u,%u,%u]", data->get_single_core_tiling()[0],
            data->get_single_core_tiling()[1], data->get_single_core_tiling()[TWO]);
  OP_LOGD(context->GetNodeName(), "  single_core_base_tiling=[%u,%u]", data->get_single_core_base_tiling()[0],
            data->get_single_core_base_tiling()[1]);
  OP_LOGD(context->GetNodeName(), "  splitRecord=[%u,%u,%u]", data->get_splitRecord()[0],
            data->get_splitRecord()[1], data->get_splitRecord()[TWO]);
  OP_LOGD(context->GetNodeName(), "  workspaceOffset=%llu", static_cast<long long unsigned int>(data->get_workspaceOffset()));
}

ge::graphStatus GMMTiling::CheckA16W4MsdEnable(uint64_t mSize, uint64_t antiquantGroupNum,
                                               const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr)
{
    if (!isSingleWeight_ || !isSingleX_ || !isSingleY_) {
        OP_LOGD(context->GetNodeName(),
                "A16W4 only support SingleWeight/SingleX/SingleY. current  isSingleWeight_: %d, isSingleX_: %d, "
                "isSingleY_: %d.",
                isSingleWeight_, isSingleX_, isSingleY_);
        return GRAPH_PARAM_INVALID;
    }
    // n轴在scale shape的第2维
    const gert::Tensor *antiquantScaleDropped = nullptr; /* antiquant_scale dropped */
    uint64_t nSize = antiquantScaleDropped->GetStorageShape().GetDim(2);
    uint64_t kSize = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
    // 2: input shape info size
    std::array<int64_t, 2> kNList = {static_cast<int64_t>(kSize), static_cast<int64_t>(nSize)};
    int isInA16W4MsdWhiteList = A16W4_MSD_WHITE_LIST.count(kNList);
    uint64_t avgMSize = groupNum_ != 0 ? mSize / groupNum_ : 0;

    auto attr = context->GetAttrs();
    auto tuningConfigPtr =
        attr != nullptr ? (attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG)) : nullptr;
    int64_t tuningConfig = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0)
                               ? (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[0]
                               : 0;
    if (tuningConfig != 0L) {
        avgMSize = static_cast<uint32_t>(tuningConfig);
    }

    uint64_t withOffset = GetWithOffset(context);

    OP_LOGD(context->GetNodeName(),
            "A16W4 try to enable msd. antiquantGroupNum: %llu, whiteListFlag: %d, withOffset: %d avgM: %d. ",
            antiquantGroupNum, isInA16W4MsdWhiteList, withOffset, avgMSize);
    const auto transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    auto transposeWeight = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);
    bool hasBias = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
    // msd当前仅支持group size为32, 方案的m轴收益上限是16
    if (!(isInA16W4MsdWhiteList && antiquantGroupNum != 0 && kSize / antiquantGroupNum == 32 && !withOffset &&
          avgMSize <= 16 && context->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_BF16 && transposeWeight &&
          !hasBias)) {
        return GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GMMTiling::GetWithOffset(const gert::TilingContext *context)
{
    auto offset = (const gert::Tensor *)nullptr /* antiquant_offset dropped */;
    uint64_t withOffset = 0;
    if (offset != nullptr) {
        auto &offsetShape = offset->GetStorageShape();
        if (offsetShape.GetDimNum() > 1 || offsetShape.GetDim(0) > 0) {
            withOffset = 1U;
        }
    }
    return withOffset;
}

ge::graphStatus GMMTiling::A16W4MsdTiling(gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr)
{
    uint64_t mSize = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(0);
    uint64_t kSize = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
    const gert::Tensor *antiquantScaleDropped2 = nullptr; /* antiquant_scale dropped */
    uint64_t antiquantGroupNum = antiquantScaleDropped2->GetStorageShape().GetDim(1);
    if (CheckA16W4MsdEnable(mSize, antiquantGroupNum, context, compileInfoPtr) != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }
    OP_LOGD(context->GetNodeName(), "A16W4 enable msd. groupListType: %u. ", groupListType_);
    GMMTilingData tilingDataA16W4;
    tilingDataA16W4.mmTilingData.set_dbL0B(1);
    tilingDataA16W4.mmTilingData.set_dbL0C(1);
    tilingDataA16W4.mmTilingData.set_stepKa(1);
    tilingDataA16W4.mmTilingData.set_stepKb(1);
    tilingDataA16W4.mmTilingData.set_depthA1(1);
    tilingDataA16W4.mmTilingData.set_depthB1(1);
    tilingDataA16W4.mmTilingData.set_baseK(512);    // MM tiling设置初始值512
    tilingDataA16W4.mmTilingData.set_baseN(256);    // MM tiling设置初始值256
    tilingDataA16W4.mmTilingData.set_stepM(1);
    tilingDataA16W4.mmTilingData.set_stepN(1);

    tilingDataA16W4.gmmBaseParams.set_coreNum(compileInfoPtr->aicNum);
    tilingDataA16W4.gmmBaseParams.set_groupNum(groupNum_);
    tilingDataA16W4.gmmBaseParams.set_totalInGroup(mSize);
    tilingDataA16W4.gmmBaseParams.set_k(kSize);
    tilingDataA16W4.gmmBaseParams.set_n(
        antiquantScaleDropped2->GetStorageShape().GetDim(2));  // dim 2 is the n axis
    tilingDataA16W4.gmmBaseParams.set_vBaseM(0);
    tilingDataA16W4.gmmBaseParams.set_ubCalSize(0);
    tilingDataA16W4.gmmBaseParams.set_ubRestBytes(0);
    tilingDataA16W4.gmmBaseParams.set_parallNum(1);
    tilingDataA16W4.gmmBaseParams.set_quantGroupNum(antiquantGroupNum);
    tilingDataA16W4.gmmBaseParams.set_m(mSize);
    tilingDataA16W4.gmmBaseParams.set_withOffset(0);
    context->SetBlockDim(compileInfoPtr->aicNum);
    context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
    context->SetTilingKey(GET_TPL_TILING_KEY(
        GMM_TPL_BF16, GMM_TPL_INT4, GMM_TPL_BF16, 0, 1, groupListType_, 0, GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE,
        GROUPED_MATMUL_A16W4_KERNEL_TEMPLATE_MSD_ANTIQUANT_GS32, GROUPED_MATMUL_AIV_AIC_RATIO_2, 0));
    tilingDataA16W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingDataA16W4.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);  // get second variable
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);     // check workspaces is not null
    workspaces[0] = SYS_WORKSPACE_SIZE;                  // default size
    // A矩阵原始空间需要占用2Byte的m*k
    uint64_t aUnfoldSize = 2 * mSize * kSize;
    // max需要再ub上做32B对齐。cache line为512B, 整体占用空间需要做cache line对齐
    uint64_t aMaxSize = CeilDiv(mSize * 32, 512) * 512;
    // 每核预留weight的空间大小为512*1024Byte，同时开2 db，保证流水
    uint64_t weightS8Size = compileInfoPtr->aicNum * 512 * 1024 * 2;
    // 每核预留C矩阵的shape为32 * 320。
    uint64_t singeCF32Size = compileInfoPtr->aicNum * 32 * 320 * sizeof(float);
    // 一轮计算k轴会计算1024的长度，groupsize = 32时，需要预留32块C矩阵空间，同时开db保证流水
    uint64_t totalCF32Size = singeCF32Size * 1024 / 32 * 2;
    workspaces[0] += static_cast<size_t>(aUnfoldSize + aMaxSize + weightS8Size + totalCF32Size);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GMMTiling::A8W4Tiling(gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
      auto yDesc = context->GetOutputDesc(Y_INDEX);
      OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
      uint32_t yDtype = GMM_TPL_FLOAT16;
      if (yDesc->GetDataType() == ge::DT_BF16) {
        yDtype = static_cast<uint32_t>(GMM_TPL_BF16);
      }
      GMMTilingData tilingDataA8W4;
      // layered: the all_weight list length is the layer count. It is shipped in both tiling
      // structs (hpTilingData for the autotiling kernel, gmmBaseParams for the MSD/pergroup
      // kernels) so the device-side layer_index can be bounds-checked before dereferencing the
      // descriptor's pointer array. The index value itself lives on device.
      uint32_t weightListLen = 0;
      while (context->GetDynamicInputTensor(WEIGHT_INDEX, weightListLen) != nullptr) {
        weightListLen++;
      }
      auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
      auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
      bool wNZ = wFormat0 == ge::FORMAT_FRACTAL_NZ;

      constexpr uint32_t cvParallNum = 4; // for cv collaboration
      constexpr uint32_t THIRTY_TWO = 32;
      constexpr uint32_t UBCALSIZE = 32U * 256U;  // for vector compute
      constexpr uint32_t UBRESTBYTES = 9U * 32U * 256U;  // for vector compute
      constexpr uint32_t TWO = 2;
      constexpr uint32_t EIGHT = 8;
      constexpr uint32_t FIVE = 5;
      constexpr uint32_t FOUR = 4;
      constexpr uint32_t ONE = 1;

      uint32_t singleN = 256;
      uint32_t singleM = 128;
      OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);  // check compileInfoPtr is not null
      const uint32_t& aicNum = compileInfoPtr->aicNum;  // get aicNum from compileInfo
      if (aicNum == 0U) {  // invalid value
        return ge::GRAPH_FAILED;
      }

      auto attr = context->GetAttrs();
      const auto tuningConfigPtr = attr != nullptr ? (attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG)) : nullptr;

      bool useHighPerf = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_A8W4_SPEC_SCENARIO_INDEX) ?
                    ((reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_A8W4_SPEC_SCENARIO_INDEX] == 1) : false;
      if (useHighPerf) {
        OP_LOGD(context->GetNodeName(), "Enter GMM A8W4 MSD high performance path...");
        constexpr int CASE_ZERO = 0;
        constexpr int CASE_ONE = 1;
        constexpr int CASE_TWO = 2;
        constexpr int CASE_THREE = 3;
        constexpr float INT4_TYPE_COUNT  = 0.5f;
        constexpr float FP16_TYPE_COUNT  = 2.0f;
        uint32_t groupNum = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(0);
        uint32_t N = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(TWO);
        uint32_t K = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
        uint32_t M = TWO * context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(0);

        const auto transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
        const auto transposeXPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
        auto transposeWeight = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
        auto transposeX = transposeXPtr != nullptr ? *transposeXPtr : false;
        // | transposeX | transposeWeight | pattern |
        // |       true |            true |       2 |
        // |       true |           false |       3 |
        // |      false |            true |       0 |
        // |      false |           false |       1 |
        int pattern = transposeX ? (transposeWeight ? CASE_TWO: CASE_THREE) : (transposeWeight ? CASE_ZERO : CASE_ONE);
        pattern = CASE_ZERO;

        tilingDataA8W4.hpTilingData.set_pattern(static_cast<uint8_t>(pattern));
        tilingDataA8W4.hpTilingData.set_kernel_index(0);
        tilingDataA8W4.hpTilingData.set_format_in(INT4_TYPE_COUNT);   // int4
        tilingDataA8W4.hpTilingData.set_format_out(FP16_TYPE_COUNT);  // fp16

        std::vector<uint32_t> ori_in0_shape;
        std::vector<uint32_t> ori_in1_shape;
        std::vector<uint32_t> ori_out_shape;
        switch (pattern) {
          case CASE_ZERO:
            ori_in0_shape = {M, K};
            ori_in1_shape = {N, K};
            ori_out_shape = {M, N};
            break;
          case CASE_ONE:
            ori_in0_shape = {M, K};
            ori_in1_shape = {K, N};
            ori_out_shape = {M, N};
            break;
          case CASE_TWO:
            ori_in0_shape = {K, M};
            ori_in1_shape = {N, K};
            ori_out_shape = {M, N};
            break;
          case CASE_THREE:
            ori_in0_shape = {K, M};
            ori_in1_shape = {K, N};
            ori_out_shape = {M, N};
            break;
          default:
            // unreachable
            return ge::GRAPH_FAILED;
        }
        tilingDataA8W4.hpTilingData.set_ori_in0_shape(ori_in0_shape.data());
        tilingDataA8W4.hpTilingData.set_ori_in1_shape(ori_in1_shape.data());
        tilingDataA8W4.hpTilingData.set_ori_out_shape(ori_out_shape.data());
        uint32_t aic = aicNum;
        uint32_t aiv = compileInfoPtr->aivNum;
        uint64_t szUB = compileInfoPtr->ubSize;
        uint64_t szL0A = compileInfoPtr->l0ASize;
        uint64_t szL0C = compileInfoPtr->l0CSize;
        SetA8W4HPTiling(&tilingDataA8W4.hpTilingData, aic);

        // autotiling parameters
        tilingDataA8W4.hpTilingData.set_group_num(groupNum);
        tilingDataA8W4.hpTilingData.set_group_type(0);
        tilingDataA8W4.hpTilingData.set_layerNum(weightListLen);
        aic = tilingDataA8W4.hpTilingData.get_required_core_num();
        if (aic == 0U) { // invalid value
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(aic);
        tilingDataA8W4.hpTilingData.set_numAic(aic);
        tilingDataA8W4.hpTilingData.set_numAiv(aiv);
        tilingDataA8W4.hpTilingData.set_szUb(szUB);
        tilingDataA8W4.hpTilingData.set_szL0A(szL0A);
        tilingDataA8W4.hpTilingData.set_szL0C(szL0C);
        auto yDtypeLocal = yDesc->GetDataType();
        if (yDtypeLocal == ge::DT_FLOAT16) {
          tilingDataA8W4.hpTilingData.set_output_type(0);
        } else {
          tilingDataA8W4.hpTilingData.set_output_type(1);
        }

        size_t workspaceSize = M * N * sizeof(int16_t) +
                               (static_cast<size_t>(SixteenAlign(M, true)) * K / TWO * sizeof(uint8_t));
        context->SetScheduleMode(1); // set as batchmod for template using SyncAll
        context->SetTilingKey(GET_TPL_TILING_KEY(GMM_TPL_INT8, GMM_TPL_INT4, yDtype, 0, 0,
                                                 GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, 0,
                                                 GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING,
                                                 GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE,
                                                 GROUPED_MATMUL_AIV_AIC_RATIO_2, 0));

        size_t *workspaces = context->GetWorkspaceSizes(1); // get second variable
        workspaces[0] = SYS_WORKSPACE_SIZE;                 // default size
        workspaces[0] += workspaceSize;

        tilingDataA8W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingDataA8W4.GetDataSize());
        PrintA8W4HPTiling(context, &tilingDataA8W4.hpTilingData);
        return ge::GRAPH_SUCCESS;
      } else {
        // scale layout: pergroup [e, G, n] or perchannel [e, n] (memory-identical
        // to [e, 1, n]; quantGroupNum degrades to 1 and the kernel addressing is
        // unchanged).
        auto scale0ShapeA8W4 = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape();
        uint32_t scaleDimNumA8W4 = scale0ShapeA8W4.GetDimNum();
        uint32_t n = 0;
        uint32_t quantGroupNum = 1;
        if (scaleDimNumA8W4 == 3U) {
          n = scale0ShapeA8W4.GetDim(TWO);
          quantGroupNum = scale0ShapeA8W4.GetDim(1);
        } else if (scaleDimNumA8W4 == 2U) {
          n = scale0ShapeA8W4.GetDim(1);
        } else {
          OP_LOGE(context->GetNodeName(), "GMM A8W4: scale dim should be 2 (perchannel [e, n]) or 3 "
                  "(pergroup [e, G, n]), but now is %u", scaleDimNumA8W4);
          return ge::GRAPH_FAILED;
        }
        const uint32_t k = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
        const uint32_t m = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(0);
        const uint32_t groupNum = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(0);
        std::array<int64_t, FIVE> mKNList = {groupNum, m, k, n, wNZ}; // 5: input shape info size

        const gert::Tensor *offset = nullptr; /* offset dropped */
        uint32_t withOffset = 0;
        if (offset != nullptr) {
          auto &offsetShape = offset->GetStorageShape();
          const size_t offsetDimNum = offsetShape.GetDimNum();
          auto offsetDim0 = offsetShape.GetDim(0);
          auto offsetDim1 = offsetShape.GetDim(1);
          auto offsetDim2 = offsetShape.GetDim(TWO);
          if (offsetDimNum == OFFSET_DIM_A8W4 && offsetDim0 == groupNum && offsetDim1 == 1 && offsetDim2 == n) {
              withOffset = 1U;
              OP_LOGD(context->GetNodeName(), "GMM A8W4: offset is enable .");
          } else {
              OP_LOGW(context->GetNodeName(), "GMM A8W4: offset's shape is invalid, If you want to enable offset, the expected shape is (%u,1,%u), and the current shape is (%ld,%ld,%ld).",
                  groupNum, n, offsetDim0, offsetDim1, offsetDim2);
          }
        }
        //GEMM Tiling
        int64_t tuningConfig = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX) ?
                    (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX] : 0;

        const int is_in_a8w4_white_list = ((tuningConfig !=0 &&tuningConfig <= 64) || (A8W4_PRETILING_WHITE_LIST.count(mKNList)))
              && quantGroupNum != 0 && k / quantGroupNum == 256 && k % quantGroupNum == 0
              && withOffset == 0 && k%64==0 && n%64==0; // 256: 新方案只支持256 pergroup

        tilingDataA8W4.gmmBaseParams.set_coreNum(aicNum);
        tilingDataA8W4.gmmBaseParams.set_groupNum(groupNum);
        tilingDataA8W4.gmmBaseParams.set_totalInGroup(m);
        tilingDataA8W4.gmmBaseParams.set_k(k);
        tilingDataA8W4.gmmBaseParams.set_n(n);
        tilingDataA8W4.gmmBaseParams.set_vBaseM(THIRTY_TWO);
        tilingDataA8W4.gmmBaseParams.set_ubCalSize(UBCALSIZE);
        tilingDataA8W4.gmmBaseParams.set_ubRestBytes(UBRESTBYTES);
        tilingDataA8W4.gmmBaseParams.set_parallNum(cvParallNum);
        tilingDataA8W4.gmmBaseParams.set_quantGroupNum(quantGroupNum);
        tilingDataA8W4.gmmBaseParams.set_m(m);
        tilingDataA8W4.gmmBaseParams.set_withOffset(withOffset);
        // layered: the MSD / MSD_NEW / pergroup kernels resolve the current layer through
        // gmmBaseParams, so this must be set here as well - A8W4Tiling returns before Init(),
        // which is the only other place that fills the field. Left unset it defaults to 0,
        // and GetLayerTensorAddr's range guard would then clamp every layer index to 0.
        tilingDataA8W4.gmmBaseParams.set_layerNum(weightListLen);
        context->SetBlockDim(aicNum);

        if (quantGroupNum == 0U || k % quantGroupNum != 0U) {
          OP_LOGE(context->GetNodeName(), "GMM_tiling: k should be divisible by quantGroupNum, but now k=%u and quantGroupNum=%u",
              k, quantGroupNum);
          return ge::GRAPH_FAILED;
        }
        const uint32_t K_UNIT = 64;  // 64: int4 in 32B
        if (k % K_UNIT != 0) {
          OP_LOGE(context->GetNodeName(), "GMM_tiling: k should be divisible by 64, but now k=%u", k);
          return ge::GRAPH_FAILED;
        }
        const uint32_t MAX_K_A8W4_MSD = 18432;  // k is limited by pre process, a line of X should be able to put in UB
        if (k > MAX_K_A8W4_MSD) {
          OP_LOGE(context->GetNodeName(), "GMM_tiling: K should be less than 18432 on the A8W4 scenario, but now is %u",
              k);
          return ge::GRAPH_FAILED;
        }
        matmul_tiling::PlatformInfo platformInfo;
        InitPlatformInfo(compileInfoPtr, platformInfo);
        matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);

        uint32_t calc_m = 1U;
        if (groupNum != 0U) {
          calc_m = m / groupNum;
        }
        const uint32_t avg_m = tuningConfig != 0L ? static_cast<uint32_t>(tuningConfig) : calc_m;
        const bool isPerchannel = quantGroupNum == 1U;
        auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);
        const bool isMSD = (tuningConfig == 0L || avg_m == 0U || n / avg_m > 4U || withOffset == true) &&
                    !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
        if (withOffset && !isMSD) {
            OP_LOGW(context->GetNodeName(), "GMM W4A8: with-offset mode requires "
                "non-empty bias, but got an empty tensor.");
        }
        if (!isMSD) {
            constexpr uint32_t A8W4_BASE_M = 128;
            constexpr uint32_t A8W4_BASE_K = 64;
            constexpr uint32_t A8W4_BASE_N = 128;
            mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
            if (wNZ) {
              mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, false);
            } else {
              mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
            }
            mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
            mm.SetBias(false);
            mm.SetOrgShape(A8W4_BASE_M, n, k);
            mm.SetShape(A8W4_BASE_M, A8W4_BASE_N, k);
            mm.SetFixSplit(A8W4_BASE_M, A8W4_BASE_N, A8W4_BASE_K);
            if (mm.GetTiling(tilingDataA8W4.mmTilingData) == -1) {
              return ge::GRAPH_FAILED;
            }
            context->SetTilingKey(GET_TPL_TILING_KEY(GMM_TPL_INT8, GMM_TPL_INT4, yDtype, 0, 0,
                                                     GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, 0,
                                                     GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT,
                                                     GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE,
                                                     GROUPED_MATMUL_AIV_AIC_RATIO_2, 0));
            tilingDataA8W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tilingDataA8W4.GetDataSize());

            size_t* workspaces = context->GetWorkspaceSizes(1);  // get second variable
            OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);  // check workspaces is not null

            workspaces[0] = SYS_WORKSPACE_SIZE;  // default size
            workspaces[0] += static_cast<size_t>(groupNum * k * n * static_cast<uint32_t>(sizeof(int8_t)) + (cvParallNum * aicNum * singleN * singleM * static_cast<uint32_t>(sizeof(int32_t)) * EIGHT));
            if (isPerchannel) {
              return ge::GRAPH_PARAM_INVALID; // continue A8W8
            } else {
              return ge::GRAPH_SUCCESS;
            }
        } else {
          const bool isShortM = avg_m < 32U;
          uint32_t A8W4_MSD_BASE_M = isShortM ? 64U : 128U;
          constexpr uint32_t A8W4_MSD_BASE_M_NEW = 64;
          constexpr uint32_t A8W4_MSD_BASE_K = 256;
          uint32_t A8W4_MSD_BASE_N = isShortM ? 512U : 256U;
          constexpr uint32_t A8W4_MSD_BASE_N_NEW = 512;
          constexpr uint32_t A8W4_MSD_BASE_K_NEW = 256;
          //增加动态Tiling部分
          uint32_t A8W4_MSD_SINGLE_N = FindBestSingleNA8W4(A8W4_MSD_BASE_M_NEW, A8W4_MSD_BASE_N_NEW, avg_m, n, groupNum, aicNum);
          mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
          if (wNZ) {
            mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4, false);
          } else {
            mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
          }
          mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
          mm.SetBias(false);
          if (is_in_a8w4_white_list) {
            mm.SetOrgShape(A8W4_MSD_BASE_M_NEW, n, k);
                  mm.SetShape(A8W4_MSD_BASE_M_NEW, A8W4_MSD_SINGLE_N, k);
            mm.SetFixSplit(A8W4_MSD_BASE_M_NEW, A8W4_MSD_BASE_N_NEW, A8W4_MSD_BASE_K_NEW);
            OP_LOGI(context->GetNodeName(), "GMM A8W4 tiling: baseM is %u, baseN is %u, baseK is %u, tuningConfig is %ld.", A8W4_MSD_BASE_M_NEW, A8W4_MSD_BASE_N_NEW, A8W4_MSD_BASE_K_NEW, tuningConfig);
          } else {
            mm.SetOrgShape(A8W4_MSD_BASE_M, n, k);
            mm.SetShape(A8W4_MSD_BASE_M, A8W4_MSD_BASE_N, k);
            mm.SetFixSplit(A8W4_MSD_BASE_M, A8W4_MSD_BASE_N, A8W4_MSD_BASE_K);
            OP_LOGI(context->GetNodeName(), "GMM A8W4 tiling: baseM is %u, baseN is %u, baseK is %u, tuningConfig is %ld.", A8W4_MSD_BASE_M, A8W4_MSD_BASE_N, A8W4_MSD_BASE_K, tuningConfig);
          }
          if (mm.GetTiling(tilingDataA8W4.mmTilingData) == -1){
            return ge::GRAPH_FAILED;
          }

          if (is_in_a8w4_white_list) {
            tilingDataA8W4.mmTilingData.set_dbL0A(ONE);  // disable double buffer for LOA
            tilingDataA8W4.mmTilingData.set_dbL0B(ONE);  // disable double buffer for LOB
            tilingDataA8W4.mmTilingData.set_dbL0C(ONE);  // disable double buffer for LOC
            tilingDataA8W4.mmTilingData.set_stepKa(ONE);  // set precomputed mmStepKa
            tilingDataA8W4.mmTilingData.set_stepKb(ONE);  // set precomputed mmStepKb
            tilingDataA8W4.mmTilingData.set_depthA1(TWO);  // set precomputed mmDepthA1
            tilingDataA8W4.mmTilingData.set_depthB1(TWO);  // set precomputed mmDepthB1
            tilingDataA8W4.mmTilingData.set_baseK(A8W4_MSD_BASE_K);
            tilingDataA8W4.mmTilingData.set_stepM(ONE);  // set precomputed stepM
            tilingDataA8W4.mmTilingData.set_stepN(ONE);  // set precomputed stepN
          } else {
            tilingDataA8W4.mmTilingData.set_dbL0C(1);  // disable double buffer for LOC
            tilingDataA8W4.mmTilingData.set_stepKa(FOUR);  // set precomputed mmStepKa
            tilingDataA8W4.mmTilingData.set_stepKb(FOUR);  // set precomputed mmStepKb
            tilingDataA8W4.mmTilingData.set_depthA1(EIGHT);  // set precomputed mmDepthA1
            tilingDataA8W4.mmTilingData.set_depthB1(EIGHT);  // set precomputed mmDepthB1
            tilingDataA8W4.mmTilingData.set_stepM(1);  // set precomputed step
            tilingDataA8W4.mmTilingData.set_stepN(1);  // set precomputed stepN
          }
          OP_LOGI(context->GetNodeName(), "GMM_tiling: baseM is %u, baseK is %u, baseN is %u.", A8W4_MSD_BASE_M, A8W4_MSD_BASE_K, A8W4_MSD_BASE_N);
          context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
          uint32_t a8w4KernelTemplate = GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT;
          if (is_in_a8w4_white_list) {
            a8w4KernelTemplate = static_cast<uint32_t>(GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT);
          }
          context->SetTilingKey(GET_TPL_TILING_KEY(GMM_TPL_INT8, GMM_TPL_INT4, yDtype, 0, 0,
                                                   GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, 0,
                                                   a8w4KernelTemplate,
                                                   GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE,
                                                   GROUPED_MATMUL_AIV_AIC_RATIO_2, 0));
          tilingDataA8W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
          context->GetRawTilingData()->SetDataSize(tilingDataA8W4.GetDataSize());

          size_t* workspaces = context->GetWorkspaceSizes(1);  // get second variable
          OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);  // check workspaces is not null
          workspaces[0] = SYS_WORKSPACE_SIZE;  // default size
          if (is_in_a8w4_white_list) {
            workspaces[0] += static_cast<size_t>((cvParallNum * aicNum * A8W4_MSD_BASE_M * A8W4_MSD_BASE_N * static_cast<uint32_t>(sizeof(short))) * TWO);
          } else {
            workspaces[0] += static_cast<size_t>((cvParallNum * aicNum * singleN * singleM * static_cast<uint32_t>(sizeof(int32_t)) * EIGHT));
          }
          return ge::GRAPH_SUCCESS;
        }
      }
}

ASCENDC_EXTERN_C ge::graphStatus TilingGMM(gert::TilingContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);  // check xDesc is not null
  ge::DataType xDType = xDesc->GetDataType();
  auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
  ge::DataType weightDtype = w0Desc->GetDataType();
  auto compileInfoPtr = context->GetCompileInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
  // The DAV_3510 (ascend950) dispatch that upstream ops-transformer has here is
  // not carried: it routes into the arch35 tiling templates, which ship with the
  // ascend950 kernel entry (grouped_matmul_apt.cpp) that this port does not
  // include, and which register themselves under the upstream operator name
  // "GroupedMatmul" rather than this operator's name. The op is registered for
  // ascend910_93 only, where npuArch is DAV_2201, so the branch could not be
  // taken here regardless.
  GMMTiling tiling;
  if(xDType == ge::DT_INT8 && weightDtype == ge::DT_INT4) {     // A8W4 Tiling
    ge::graphStatus A8W4TilingResult = tiling.A8W4Tiling(context, compileInfoPtr);
    if (A8W4TilingResult != ge::GRAPH_PARAM_INVALID) {
      return A8W4TilingResult;
    }
  }

  OP_CHECK_IF(tiling.Init(context) != ge::GRAPH_SUCCESS,
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GMM tiling init failed"),
             return ge::GRAPH_FAILED);

  if(xDType == ge::DT_BF16 && weightDtype == ge::DT_INT4) {     // A16W4 Msd Tiling
    ge::graphStatus A16W4MsdTilingResult = tiling.A16W4MsdTiling(context, compileInfoPtr);
    if (A16W4MsdTilingResult == ge::GRAPH_SUCCESS) {
      return A16W4MsdTilingResult;
    }
  }
  return tiling.RunFusionKernelTiling(context);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGMM(gert::TilingParseContext* context) {
  OP_CHECK_NULL_WITH_CONTEXT(context, context);
  fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
  auto compileInfoPtr = context->GetCompiledInfo<GMMCompileInfo>();
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
  compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
  compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
  compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

  OP_CHECK_IF((compileInfoPtr->npuArch != NpuArch::DAV_3510 && compileInfoPtr->aivNum == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
             compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
             compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
             return ge::GRAPH_FAILED);

  OP_CHECK_IF((compileInfoPtr->aicNum == 0 || compileInfoPtr->ubSize == 0 || \
             compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 || \
             compileInfoPtr->l0BSize == 0),
             OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
             "platform info is invalid, aicNum=%u, aivNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
             compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size,
             compileInfoPtr->l0CSize, compileInfoPtr->l0ASize, compileInfoPtr->l0BSize),
             return ge::GRAPH_FAILED);

  OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulLayered)
.Tiling(TilingGMM)
.TilingParse<GMMCompileInfo>(TilingPrepareForGMM);  // register into the framework
}  // namespace optiling
