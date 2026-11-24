/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
 * Description: Host tiling and registration for afd_async_combine_recv.
 * Create: 2026-11-23
 * Note: The host selects the AIV launch dimensions and configures the MC2
 *       communication group.
 * History: 2026-11-23 add to the Ascend 950 port
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/afd_async_combine_recv_tiling.h"

#include "afd_async_host_tiling.h"
#include "ops_log.h"
#include "ops_error.h"

using namespace ge;
using namespace afd_async;

constexpr static int ONE_DIM = 1;
constexpr static int TWO_DIMS = 2;

constexpr static int INPUT_X_INDEX = 0;
constexpr static int INPUT_EXPERT_IDS_INDEX = 1;
constexpr static int INPUT_EXPERT_SCALES_INDEX = 2;

constexpr static int ATTR_ENUM_MAGIC = 0;
constexpr static int ATTR_ENUM_BATCH_SIZE = 1;
constexpr static int ATTR_ENUM_HIDDEN_SIZE = 2;
constexpr static int ATTR_ENUM_TOPK = 3;
constexpr static int ATTR_ENUM_MOE_RANK_NUM = 4;
constexpr static int ATTR_ENUM_ATTN_RANK_NUM = 5;
constexpr static int ATTR_ENUM_ROUTE_EXPERT_NUM_PER_MOE = 6;
constexpr static int ATTR_ENUM_ATTN_RANK_ID = 7;
constexpr static int ATTR_ENUM_WORLD_SIZE = 8;
constexpr static int ATTR_ENUM_HCCL_GROUP_NAME = 9;

constexpr static int TILING_KEY_BF16 = 100;
constexpr static int TILING_KEY_FP16 = 101;
constexpr static int TILING_KEY_BF16_PREFIX = 102;
constexpr static int TILING_KEY_FP16_PREFIX = 103;

constexpr static int LIMIT_BATCH_SIZE_MIN = 1;
constexpr static int LIMIT_BATCH_SIZE_MAX = 1024 * 256;
constexpr static int LIMIT_TOPK_MIN = 1;
constexpr static int LIMIT_HIDDEN_SIZE_MIN = 1;
constexpr static int LIMIT_ATTENTION_RANK_SIZE_MIN = 1;
constexpr static int LIMIT_EXPERT_RANK_SIZE_MIN = 1;
constexpr static int LIMIT_ROUTE_EXPERT_NUM_PER_MOE_MIN = 1;

constexpr static int BATCH_INFO_VAL_NUM = 5;
constexpr static int UB_ALIGN = 32;
constexpr static int MAX_AIV_NUM = 48;

constexpr static uint32_t OP_TYPE_ALL_TO_ALL = 8U;

static uint64_t MathCeil(uint64_t n, uint64_t align)
{
    return (n + align - 1) / align * align;
}

namespace optiling {
static void SetHcommCfg(AfdAsyncCombineRecvTilingData *tiling, const std::string &groupName, bool isAscend950)
{
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType1, algConfigAllToAllStr);
#ifdef AFD_TILING_HAS_COMM_ENGINE
    // Match the A5 A2E/E2A MTE resource path: this field is HcclAccelerator::AIV.
    if (isAscend950) {
        mc2CcTilingConfig.SetCommEngine(afd_async::host::MTE_HCCL_ACCELERATOR);
    }
#else
    (void)isAscend950;
#endif
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}
// Patch reason: empty source partitions add launch/synchronization work on A5.
// Patch functionality: match dispatch send's small-batch core count without
// reducing baseline UB acceptance or changing compact-v2 window offsets.
// Signature: matches the original TilingFunc; no added parameters.
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    AfdAsyncCombineRecvTilingData *tilingData = context->GetTilingData<AfdAsyncCombineRecvTilingData>();

    auto attrs = context->GetAttrs();
    int64_t magic = *(attrs->GetInt(ATTR_ENUM_MAGIC));
    int64_t batchSize = *(attrs->GetInt(ATTR_ENUM_BATCH_SIZE));
    int64_t hiddenSize = *(attrs->GetInt(ATTR_ENUM_HIDDEN_SIZE));
    int64_t topk = *(attrs->GetInt(ATTR_ENUM_TOPK));
    int64_t moeRankNum = *(attrs->GetInt(ATTR_ENUM_MOE_RANK_NUM));
    int64_t attnRankNum = *(attrs->GetInt(ATTR_ENUM_ATTN_RANK_NUM));
    int64_t routeExpertNumPerMoe = *(attrs->GetInt(ATTR_ENUM_ROUTE_EXPERT_NUM_PER_MOE));
    int64_t attnRankId = *(attrs->GetInt(ATTR_ENUM_ATTN_RANK_ID));
    int64_t worldSize = *(attrs->GetInt(ATTR_ENUM_WORLD_SIZE));
    auto groupNamePtr = attrs->GetAttrPointer<char>(ATTR_ENUM_HCCL_GROUP_NAME);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    bool isAscend950 = false;
#ifdef AFD_TILING_HAS_COMM_ENGINE
    isAscend950 = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
#endif
    // ### PATCH START: Keep baseline resource cap separate from source launch
    const uint64_t platformAivNum = afd_async::host::GetActiveAivCount(ascendcPlatform.GetCoreNumAiv());
    uint64_t aivNum = platformAivNum;
    // ### PATCH END: Keep baseline resource cap separate from source launch
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OPS_ERR_IF(aivNum == 0,
        OPS_LOG_E(nodeName, "Platform has no active AIV cores."), return ge::GRAPH_FAILED);
    if (isAscend950) {
        OPS_ERR_IF(worldSize < 1 || worldSize > afd_async::host::A5_MAX_WORLD_SIZE
            || moeRankNum < 1 || attnRankNum < 1 || moeRankNum > worldSize || attnRankNum > worldSize,
            OPS_LOG_E(nodeName, "Ascend 950 compact-v2 requires positive A/F ranks and worldSize <= %ld.",
                afd_async::host::A5_MAX_WORLD_SIZE), return ge::GRAPH_FAILED);
        OPS_ERR_IF(hiddenSize < LIMIT_HIDDEN_SIZE_MIN || hiddenSize > afd_async::host::MAX_KERNEL_DIMENSION
            || hiddenSize % static_cast<int64_t>(afd_async::host::UB_ALIGNMENT) != 0 || routeExpertNumPerMoe < 1
            || routeExpertNumPerMoe > afd_async::host::MAX_KERNEL_DIMENSION / moeRankNum
            || topk < 1 || topk > afd_async::host::MAX_KERNEL_DIMENSION,
            OPS_LOG_E(nodeName, "Ascend 950 compact-v2 requires aligned hiddenSize and representable expert dimensions."),
            return ge::GRAPH_FAILED);
    }

    int64_t expertNum = routeExpertNumPerMoe * moeRankNum;
    uint64_t sharedMemSize = GetMaxWindowSize();

    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *expertIdsShape = context->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const gert::StorageShape *expertScalesShape = context->GetInputShape(INPUT_EXPERT_SCALES_INDEX);

    OPS_ERR_IF(batchSize < LIMIT_BATCH_SIZE_MIN || batchSize > LIMIT_BATCH_SIZE_MAX,
        OPS_LOG_E(nodeName, "batchSize is invalid, must be in range [%d, %d], but got batchSize=%ld.",
            LIMIT_BATCH_SIZE_MIN, LIMIT_BATCH_SIZE_MAX, batchSize), return ge::GRAPH_FAILED);
    OPS_ERR_IF(hiddenSize < LIMIT_HIDDEN_SIZE_MIN,
        OPS_LOG_E(nodeName, "hiddenSize is invalid, must >= %d, but got hiddenSize=%ld.",
            LIMIT_HIDDEN_SIZE_MIN, hiddenSize), return ge::GRAPH_FAILED);
    OPS_ERR_IF(topk < LIMIT_TOPK_MIN,
        OPS_LOG_E(nodeName, "topk is invalid, must >= %d, but got topk=%ld.",
            LIMIT_TOPK_MIN, topk), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeRankNum < LIMIT_EXPERT_RANK_SIZE_MIN,
        OPS_LOG_E(nodeName, "moeRankNum is invalid, must >= %d, but got moeRankNum=%ld.",
            LIMIT_EXPERT_RANK_SIZE_MIN, moeRankNum), return ge::GRAPH_FAILED);
    OPS_ERR_IF(attnRankNum < LIMIT_ATTENTION_RANK_SIZE_MIN,
        OPS_LOG_E(nodeName, "attnRankNum is invalid, must >= %d, but got attnRankNum=%ld.",
            LIMIT_ATTENTION_RANK_SIZE_MIN, attnRankNum), return ge::GRAPH_FAILED);
    OPS_ERR_IF(routeExpertNumPerMoe < LIMIT_ROUTE_EXPERT_NUM_PER_MOE_MIN,
        OPS_LOG_E(nodeName, "routeExpertNumPerMoe is invalid, must >= %d, but got routeExpertNumPerMoe=%ld.",
            LIMIT_ROUTE_EXPERT_NUM_PER_MOE_MIN, routeExpertNumPerMoe), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertNum < topk,
        OPS_LOG_E(nodeName, "expertNum is invalid, must >= topk=%ld, but got expertNum=%ld.",
            topk, expertNum), return ge::GRAPH_FAILED);
    OPS_ERR_IF(worldSize != (moeRankNum + attnRankNum),
        OPS_LOG_E(nodeName, "worldSize is invalid, must be moeRankNum+attnRankNum, but got worldSize=%ld.",
            worldSize), return ge::GRAPH_FAILED);
    OPS_ERR_IF(attnRankId < 0 || attnRankId > (worldSize - 1),
        OPS_LOG_E(nodeName, "attnRankId is invalid, only support [0, %ld), but got attnRankId=%ld.",
            worldSize, attnRankId), return ge::GRAPH_FAILED);

    OPS_ERR_IF(xShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OPS_LOG_E(nodeName, "xShape dim is invalid, must be %d, but got dimNum=%u.",
            ONE_DIM, xShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDim(0) != 1,
        OPS_LOG_E(nodeName, "xShape dim0 is invalid, must be 1, but got dim0=%u.",
            xShape->GetStorageShape().GetDim(0)), return ge::GRAPH_FAILED);

    OPS_ERR_IF(expertIdsShape == nullptr, OPS_LOG_E(nodeName, "expertIdsShape is null."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertIdsShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName, "expertIdsShape dim is invalid, must be %d, but got dimNum=%u.",
            TWO_DIMS, expertIdsShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertIdsShape->GetStorageShape().GetDim(0) != batchSize,
        OPS_LOG_E(nodeName, "expertIdsShape dim0 is invalid, must be batchSize=%ld, but got dim0=%u.",
            batchSize, expertIdsShape->GetStorageShape().GetDim(0)), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertIdsShape->GetStorageShape().GetDim(1) != topk,
        OPS_LOG_E(nodeName, "expertIdsShape dim1 is invalid, must be topk=%ld, but got dim1=%u.",
            topk, expertIdsShape->GetStorageShape().GetDim(1)), return ge::GRAPH_FAILED);

    OPS_ERR_IF(expertScalesShape == nullptr, OPS_LOG_E(nodeName, "expertScalesShape is null."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertScalesShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName, "expertScalesShape dim is invalid, must be %d, but got dimNum=%u.",
            TWO_DIMS, expertScalesShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertScalesShape->GetStorageShape().GetDim(0) != batchSize,
        OPS_LOG_E(nodeName, "expertScalesShape dim0 is invalid, must be batchSize=%ld, but got dim0=%u.",
            batchSize, expertScalesShape->GetStorageShape().GetDim(0)), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertScalesShape->GetStorageShape().GetDim(1) != topk,
        OPS_LOG_E(nodeName, "expertScalesShape dim1 is invalid, must be topk=%ld, but got dim1=%u.",
            topk, expertScalesShape->GetStorageShape().GetDim(1)), return ge::GRAPH_FAILED);

    OPS_ERR_IF(isAscend950 && batchSize > afd_async::host::MAX_SOURCE_TOKEN_COUNT,
        OPS_LOG_E(nodeName, "Ascend 950 compact-v2 source batch must fit uint16 counters (maximum %ld).",
            afd_async::host::MAX_SOURCE_TOKEN_COUNT), return ge::GRAPH_FAILED);

    // ### PATCH START: Match DS/CR source partitions for A5 small batches
    if (isAscend950) {
        aivNum = afd_async::host::GetSourceActiveAivCount(platformAivNum, static_cast<uint64_t>(batchSize));
    }
    // ### PATCH END: Match DS/CR source partitions for A5 small batches

    if (isAscend950) {
        // ### PATCH START: Keep the original platform-cap UB acceptance
        uint64_t requiredUbSize = afd_async::host::CombineRecvUbBytes(moeRankNum, expertNum, hiddenSize, platformAivNum);
        // ### PATCH END: Keep the original platform-cap UB acceptance
        OPS_ERR_IF(ubSize < requiredUbSize,
            OPS_LOG_E(nodeName, "Ascend 950 compact-v2 requires %lu UB bytes but platform provides %lu.",
                requiredUbSize, ubSize), return ge::GRAPH_FAILED);
    }

    // shared memory needed by dispatch perf optimization (max ~54 KB)
    uint64_t dispatchNeedSize = MathCeil(
        sizeof(uint32_t) * (moeRankNum + routeExpertNumPerMoe * moeRankNum) * MAX_AIV_NUM, UB_ALIGN);
    // combine-send flag region (max ~64 B)
    uint64_t combineSendFlagSize = MathCeil(sizeof(uint32_t) * moeRankNum, UB_ALIGN);
    // tokens returned by combine-send (max ~1008 KB)
    uint64_t combineTokenSize = MathCeil(sizeof(int16_t) * hiddenSize * batchSize * topk, UB_ALIGN);
    uint64_t sharedMemAttnNeedSize = dispatchNeedSize + combineSendFlagSize + combineTokenSize;

    OPS_ERR_IF(sharedMemSize < sharedMemAttnNeedSize,
        OPS_LOG_E(nodeName, "sharedMemSize is %lu but need %lu.", sharedMemSize, sharedMemAttnNeedSize),
        return ge::GRAPH_FAILED);

    std::string groupName(groupNamePtr);
    SetHcommCfg(tilingData, groupName, isAscend950);

    tilingData->afdAsyncCombineRecvInfo.magic = magic;
    tilingData->afdAsyncCombineRecvInfo.batchSize = batchSize;
    tilingData->afdAsyncCombineRecvInfo.hiddenSize = hiddenSize;
    tilingData->afdAsyncCombineRecvInfo.topk = topk;
    tilingData->afdAsyncCombineRecvInfo.moeRankNum = moeRankNum;
    tilingData->afdAsyncCombineRecvInfo.attnRankNum = attnRankNum;
    tilingData->afdAsyncCombineRecvInfo.routeExpertNumPerMoe = routeExpertNumPerMoe;
    tilingData->afdAsyncCombineRecvInfo.attnRankId = attnRankId;
    tilingData->afdAsyncCombineRecvInfo.worldSize = worldSize;

    // Launch the same active AIV count used by the compact statistics and token partitions.
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);

    tilingData->afdAsyncCombineRecvInfo.aivNum = aivNum;
    tilingData->afdAsyncCombineRecvInfo.totalUbSize = ubSize;
    tilingData->afdAsyncCombineRecvInfo.totalWorkspaceSize = sharedMemSize;

    // Keep the original kernel for at most one token per core; larger batches reuse dispatch prefixes.
    bool useDispatchPrefix = batchSize > static_cast<int64_t>(aivNum);
    auto xDtype = context->GetInputDesc(0)->GetDataType();
    if (xDtype == ge::DT_BF16) {
        context->SetTilingKey(useDispatchPrefix ? TILING_KEY_BF16_PREFIX : TILING_KEY_BF16);
    } else if (xDtype == ge::DT_FLOAT16) {
        context->SetTilingKey(useDispatchPrefix ? TILING_KEY_FP16_PREFIX : TILING_KEY_FP16);
    }

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    int64_t batchSize = *(attrs->GetInt(ATTR_ENUM_BATCH_SIZE));
    int64_t hiddenSize = *(attrs->GetInt(ATTR_ENUM_HIDDEN_SIZE));

    gert::Shape *xOutShape = context->GetOutputShape(0);
    xOutShape->SetDimNum(2);
    xOutShape->SetDim(0, batchSize);
    xOutShape->SetDim(1, hiddenSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto expandXDType = context->GetInputDataType(0);
    context->SetOutputDataType(0, expandXDType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AfdAsyncCombineRecv : public OpDef {
public:
    explicit AfdAsyncCombineRecv(const char* name) : OpDef(name)
    {
        this->Input("expandX")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expertIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expertScales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("commArgs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("magic").Int();
        this->Attr("batchSize").Int();
        this->Attr("hiddenSize").Int();
        this->Attr("topk").Int();
        this->Attr("moeRankNum").Int();
        this->Attr("attnRankNum").Int();
        this->Attr("routeExpertNumPerMoe").Int();
        this->Attr("attnRankId").Int();
        this->Attr("worldSize").Int();
        this->Attr("hccl_group_name").String();

        this->AICore().AddConfig("ascend950");
        this->MC2().HcclGroup({"hccl_group_name"});

        // support graph
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
    }
};

OP_ADD(AfdAsyncCombineRecv);

} // namespace ops
