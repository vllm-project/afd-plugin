/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncCombineSend tiling function implementation file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncCombineSend tiling function file
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/afd_async_combine_send_tiling.h"

#include "ops_log.h"
#include "ops_error.h"

using namespace ge;
using namespace Cam;

constexpr static int ONE_DIM = 1;
constexpr static int TWO_DIMS = 2;

constexpr static int INPUT_X_INDEX = 0;
constexpr static int INPUT_BATCH_INFO_INDEX = 2;

constexpr static int ATTR_ENUM_MAGIC = 0;
constexpr static int ATTR_ENUM_MAX_SEQ_LEN = 1;
constexpr static int ATTR_ENUM_HIDDEN_SIZE = 2;
constexpr static int ATTR_ENUM_TOPK = 3;
constexpr static int ATTR_ENUM_MOE_RANK_NUM = 4;
constexpr static int ATTR_ENUM_ATTN_RANK_NUM = 5;
constexpr static int ATTR_ENUM_ROUTE_EXPERT_NUM_PER_MOE = 6;
constexpr static int ATTR_ENUM_MOE_RANK_ID = 7;
constexpr static int ATTR_ENUM_WORLD_SIZE = 8;
constexpr static int ATTR_ENUM_TP_SIZE = 9;
constexpr static int ATTR_ENUM_HCCL_GROUP_NAME = 10;

constexpr static int TILING_KEY_BF16 = 100;
constexpr static int TILING_KEY_FP16 = 101;

constexpr static int INFO_NUM = 5; // number of valid batch-info fields; also start/end expert per chunk

constexpr static int LIMIT_MAX_SEQ_LEN_MIN = 1;
constexpr static int LIMIT_MAX_SEQ_LEN_MAX = 1024 * 256;
constexpr static int LIMIT_TOPK_MIN = 1;
constexpr static int LIMIT_HIDDEN_SIZE_MIN = 1;
constexpr static int LIMIT_ATTENTION_RANK_SIZE_MIN = 1;
constexpr static int LIMIT_EXPERT_RANK_SIZE_MIN = 1;
constexpr static int LIMIT_ROUTE_EXPERT_NUM_PER_MOE_MIN = 1;
constexpr static int LIMIT_TP_SIZE_MIN = 1;

constexpr static int BATCH_INFO_VAL_NUM = 5;
constexpr static int UB_ALIGN = 32;
constexpr static int MAX_AIV_NUM = 48;

constexpr static uint32_t OP_TYPE_ALL_TO_ALL = 8U;

static uint64_t MathCeil(uint64_t n, uint64_t align)
{
    return (n + align - 1) / align * align;
}

namespace optiling {
static void SetHcommCfg(AfdAsyncCombineSendTilingData *tiling, const std::string &groupName)
{
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    AfdAsyncCombineSendTilingData *tilingData = context->GetTilingData<AfdAsyncCombineSendTilingData>();

    auto attrs = context->GetAttrs();
    int64_t magic = *(attrs->GetInt(ATTR_ENUM_MAGIC));
    int64_t maxSeqLen = *(attrs->GetInt(ATTR_ENUM_MAX_SEQ_LEN));
    int64_t hiddenSize = *(attrs->GetInt(ATTR_ENUM_HIDDEN_SIZE));
    int64_t topk = *(attrs->GetInt(ATTR_ENUM_TOPK));
    int64_t moeRankNum = *(attrs->GetInt(ATTR_ENUM_MOE_RANK_NUM));
    int64_t attnRankNum = *(attrs->GetInt(ATTR_ENUM_ATTN_RANK_NUM));
    int64_t routeExpertNumPerMoe = *(attrs->GetInt(ATTR_ENUM_ROUTE_EXPERT_NUM_PER_MOE));
    int64_t moeRankId = *(attrs->GetInt(ATTR_ENUM_MOE_RANK_ID));
    int64_t worldSize = *(attrs->GetInt(ATTR_ENUM_WORLD_SIZE));
    int64_t tpSize = *(attrs->GetInt(ATTR_ENUM_TP_SIZE));
    auto groupNamePtr = attrs->GetAttrPointer<char>(ATTR_ENUM_HCCL_GROUP_NAME);

    int64_t expertNum = routeExpertNumPerMoe * moeRankNum;
    uint64_t sharedMemSize = GetMaxWindowSize();

    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *batchInfoShape = context->GetInputShape(INPUT_BATCH_INFO_INDEX);

    OPS_ERR_IF(maxSeqLen < LIMIT_MAX_SEQ_LEN_MIN || maxSeqLen > LIMIT_MAX_SEQ_LEN_MAX,
        OPS_LOG_E(nodeName, "maxSeqLen is invalid, only support [%d, %d], but got maxSeqLen=%ld.",
            LIMIT_MAX_SEQ_LEN_MIN, LIMIT_MAX_SEQ_LEN_MAX, maxSeqLen),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(hiddenSize < LIMIT_HIDDEN_SIZE_MIN,
        OPS_LOG_E(nodeName, "hiddenSize is invalid, must >= %d, but got hiddenSize=%ld.",
            LIMIT_HIDDEN_SIZE_MIN, hiddenSize), return ge::GRAPH_FAILED);
    OPS_ERR_IF(topk < LIMIT_TOPK_MIN,
        OPS_LOG_E(nodeName, "topk is invalid, must >= %d, but got topk=%ld.",
            LIMIT_TOPK_MIN, topk), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeRankNum < LIMIT_EXPERT_RANK_SIZE_MIN,
        OPS_LOG_E(nodeName, "moeRankNum is invalid, must >= %d, but got moeRankNum=%ld.",
            LIMIT_EXPERT_RANK_SIZE_MIN, moeRankNum), return ge::GRAPH_FAILED);
    OPS_ERR_IF(routeExpertNumPerMoe < LIMIT_ROUTE_EXPERT_NUM_PER_MOE_MIN,
        OPS_LOG_E(nodeName, "routeExpertNumPerMoe is invalid, must >= %d, but got routeExpertNumPerMoe=%ld.",
            LIMIT_ROUTE_EXPERT_NUM_PER_MOE_MIN, routeExpertNumPerMoe), return ge::GRAPH_FAILED);
    OPS_ERR_IF(attnRankNum < LIMIT_ATTENTION_RANK_SIZE_MIN,
        OPS_LOG_E(nodeName, "attnRankNum is invalid, must >= %d, but got attnRankNum=%ld.",
            LIMIT_ATTENTION_RANK_SIZE_MIN, attnRankNum), return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertNum < topk,
        OPS_LOG_E(nodeName, "expertNum is invalid, routeExpertNumPerMoe=%ld moeRankNum=%ld.",
            routeExpertNumPerMoe, moeRankNum), return ge::GRAPH_FAILED);
    OPS_ERR_IF(worldSize != (moeRankNum + attnRankNum),
        OPS_LOG_E(nodeName, "worldSize is invalid, must be moeRankNum+attnRankNum, but got worldSize=%ld.",
            worldSize), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeRankId < 0 || moeRankId > (worldSize - 1),
        OPS_LOG_E(nodeName, "moeRankId is invalid, only support [0, %ld), but got moeRankId=%ld.",
            worldSize, moeRankId), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tpSize < LIMIT_TP_SIZE_MIN || (attnRankNum % tpSize) != 0,
        OPS_LOG_E(nodeName, "tpSize is invalid, must >= %d and divide attnRankNum, but got tpSize=%ld.",
            LIMIT_TP_SIZE_MIN, tpSize), return ge::GRAPH_FAILED);

    OPS_ERR_IF(xShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OPS_LOG_E(nodeName, "xShape dim is invalid, must be %d, but got dimNum=%u.",
            TWO_DIMS, xShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDim(1) != hiddenSize,
        OPS_LOG_E(nodeName, "xShape dim1 is invalid, must be hiddenSize=%ld, but got dim1=%u.",
            hiddenSize, xShape->GetStorageShape().GetDim(1)), return ge::GRAPH_FAILED);

    int64_t batchInfoNum = INFO_NUM + tpSize + routeExpertNumPerMoe * tpSize;
    OPS_ERR_IF(batchInfoShape == nullptr,
        OPS_LOG_E(nodeName, "batchInfoShape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(batchInfoShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OPS_LOG_E(nodeName, "batchInfoShape dim is invalid, must be %d, but got dimNum=%u.",
            ONE_DIM, batchInfoShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(batchInfoShape->GetStorageShape().GetDim(0) != batchInfoNum,
        OPS_LOG_E(nodeName, "batchInfoShape dim0 is invalid, must be batchInfoNum=%ld, but got dim0=%u.",
            batchInfoNum, batchInfoShape->GetStorageShape().GetDim(0)), return ge::GRAPH_FAILED);

    std::string groupName(groupNamePtr);
    SetHcommCfg(tilingData, groupName);

    tilingData->moeDistributeCombineInfo.magic = magic;
    tilingData->moeDistributeCombineInfo.maxSeqLen = maxSeqLen;
    tilingData->moeDistributeCombineInfo.hiddenSize = hiddenSize;
    tilingData->moeDistributeCombineInfo.topk = topk;
    tilingData->moeDistributeCombineInfo.moeRankNum = moeRankNum;
    tilingData->moeDistributeCombineInfo.attnRankNum = attnRankNum;
    tilingData->moeDistributeCombineInfo.routeExpertNumPerMoe = routeExpertNumPerMoe;
    tilingData->moeDistributeCombineInfo.moeRankId = moeRankId;
    tilingData->moeDistributeCombineInfo.worldSize = worldSize;
    tilingData->moeDistributeCombineInfo.tpSize = tpSize;

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);

    tilingData->moeDistributeCombineInfo.aivNum = aivNum;
    tilingData->moeDistributeCombineInfo.totalUbSize = ubSize;
    tilingData->moeDistributeCombineInfo.totalWorkspaceSize = sharedMemSize;

    auto xDtype = context->GetInputDesc(0)->GetDataType();
    if (xDtype == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
    } else if (xDtype == ge::DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_FP16);
    }

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    gert::Shape* expandXOutShape = context->GetOutputShape(0);

    expandXOutShape->SetDimNum(1);
    expandXOutShape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AfdAsyncCombineSend : public OpDef {
public:
    explicit AfdAsyncCombineSend(const char* name) : OpDef(name)
    {
        this->Input("expandX")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("commArgs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("batchInfo")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("magic").Int();
        this->Attr("maxSeqLen").Int();
        this->Attr("hiddenSize").Int();
        this->Attr("topk").Int();
        this->Attr("moeRankNum").Int();
        this->Attr("attnRankNum").Int();
        this->Attr("routeExpertNumPerMoe").Int();
        this->Attr("moeRankId").Int();
        this->Attr("worldSize").Int();
        this->Attr("tpSize").Int();
        this->Attr("hccl_group_name").String();

        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup({"hccl_group_name"});

        // support graph
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
    }
};

OP_ADD(AfdAsyncCombineSend);

} // namespace ops
