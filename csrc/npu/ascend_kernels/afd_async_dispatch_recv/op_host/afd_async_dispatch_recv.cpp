/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: AfdAsyncDispatchRecv tiling function implementation file
 * Author: Yan Ming
 * Create: 2025-12-18
 * Note:
 * History:
 * 2025-12-18 create AfdAsyncDispatchRecv tiling function file
 */
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/afd_async_dispatch_recv_tiling.h"

#include "ops_log.h"
#include "ops_error.h"

using namespace ge;
using namespace Cam;

constexpr static int ONE_DIM = 1;

constexpr static int INPUT_X_INDEX = 0;

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
constexpr static int ATTR_ENUM_DYNAMIC_QUANT = 10;
constexpr static int ATTR_ENUM_HCCL_GROUP_NAME = 11;

constexpr static int OUTPUT_EXPAND_X = 0;
constexpr static int OUTPUT_DYNAMIC_SCALES = 1;
constexpr static int OUTPUT_BATCH_INFO = 2;
constexpr static int OUTPUT_EP_RECV_COUNT_ROUTED = 3;

constexpr static int TILING_KEY_BF16 = 100;
constexpr static int TILING_KEY_FP16 = 101;

constexpr static int INFO_NUM = 5; // number of valid batch-info fields; also start/end expert per chunk

// maxSeqLen [1, 8192]
constexpr static int LIMIT_MAX_SEQ_LEN_MIN = 1;
constexpr static int LIMIT_MAX_SEQ_LEN_MAX = 1024 * 256;
constexpr static int LIMIT_HIDDEN_SIZE_MIN = 1;
constexpr static int LIMIT_TOPK_MIN = 1;
constexpr static int LIMIT_ATTENTION_RANK_SIZE_MIN = 1;
constexpr static int LIMIT_TP_SIZE_MIN = 1;
constexpr static int LIMIT_EXPERT_RANK_SIZE_MIN = 1;

constexpr static int BATCH_INFO_VAL_NUM = 5;
constexpr static int UB_ALIGN = 32;
constexpr static int MAX_AIV_NUM = 48;

constexpr static float EP_BALANCE_FACTOR = 1.2;

constexpr static uint32_t OP_TYPE_ALL_TO_ALL = 8U;

static uint32_t MathCeil(uint32_t n, uint32_t align)
{
    return (n + align - 1) / align * align;
}

namespace optiling {
static void SetHcommCfg(AfdAsyncDispatchRecvTilingData *tiling, const std::string &groupName)
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
    AfdAsyncDispatchRecvTilingData *tilingData =
        context->GetTilingData<AfdAsyncDispatchRecvTilingData>();

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
    int64_t dynamicQuant = *(attrs->GetInt(ATTR_ENUM_DYNAMIC_QUANT));
    auto groupNamePtr = attrs->GetAttrPointer<char>(ATTR_ENUM_HCCL_GROUP_NAME);

    int64_t expertNum = routeExpertNumPerMoe * moeRankNum;
    uint64_t sharedMemSize = GetMaxWindowSize();

    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);

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

    OPS_ERR_IF(dynamicQuant != 0 && dynamicQuant != 1,
        OPS_LOG_E(nodeName, "dynamicQuant is invalid, only support 0 or 1, but got dynamicQuant=%ld.",
            dynamicQuant), return ge::GRAPH_FAILED);

    OPS_ERR_IF(xShape == nullptr, OPS_LOG_E(nodeName, "xShape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OPS_LOG_E(nodeName, "xShape dim is invalid, must be %d, but got dimNum=%u.",
            ONE_DIM, xShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape->GetStorageShape().GetDim(0) != 1,
        OPS_LOG_E(nodeName, "xShape dim0 is invalid, must be 1, but got dim0=%u.",
            xShape->GetStorageShape().GetDim(0)), return ge::GRAPH_FAILED);

    int64_t limitMaxSeqLenPerRank = maxSeqLen / tpSize;

    // 64 B, info sent by dispatch send
    uint64_t dispatchInfoSize = MathCeil(sizeof(int64_t) * BATCH_INFO_VAL_NUM, UB_ALIGN);
    // max ~96 B, token count received per expert / chunk flag for whether expert already processed
    uint64_t expertTokenCntSize = MathCeil(sizeof(int32_t) * routeExpertNumPerMoe, UB_ALIGN);
    // max ~112 KB, real tokens received
    uint64_t dispatchTokenSize = 0;
    if (dynamicQuant == 1) {
        dispatchTokenSize = MathCeil((sizeof(int8_t) * hiddenSize + UB_ALIGN) * limitMaxSeqLenPerRank, UB_ALIGN);
    } else {
        dispatchTokenSize = MathCeil(sizeof(int16_t) * hiddenSize * limitMaxSeqLenPerRank, UB_ALIGN);
    }
    // max ~144 KB, offsets of received tokens
    uint64_t tokenAddrSize = MathCeil(sizeof(uint16_t) * limitMaxSeqLenPerRank * topk, UB_ALIGN);
    // min shared memory needed on the moe side
    uint64_t sharedMemMoeNeedSize =
        (dispatchInfoSize + expertTokenCntSize * 2 + dispatchTokenSize + tokenAddrSize) * attnRankNum;

    OPS_ERR_IF(sharedMemSize < sharedMemMoeNeedSize,
        OPS_LOG_E(nodeName, "sharedMemSize is %lu but need %lu.", sharedMemSize, sharedMemMoeNeedSize),
        return ge::GRAPH_FAILED);

    float maxSeqLenFactor = GetBatchSizeFactor();
    OPS_ERR_IF(!(maxSeqLenFactor > 0.0f && maxSeqLenFactor <= 1.0f),
        OPS_LOG_E(nodeName, "maxSeqLenFactor is invalid, only support (0, 1], but got maxSeqLenFactor=%f.",
            maxSeqLenFactor), return ge::GRAPH_FAILED);
    uint64_t maxTokenNum = (int64_t)(LIMIT_MAX_SEQ_LEN_MAX * maxSeqLenFactor);

    std::string groupName(groupNamePtr);
    SetHcommCfg(tilingData, groupName);

    tilingData->moeDistributeDispatchInfo.magic = magic;
    tilingData->moeDistributeDispatchInfo.maxSeqLen = maxSeqLen;
    tilingData->moeDistributeDispatchInfo.hiddenSize = hiddenSize;
    tilingData->moeDistributeDispatchInfo.topk = topk;
    tilingData->moeDistributeDispatchInfo.moeRankNum = moeRankNum;
    tilingData->moeDistributeDispatchInfo.attnRankNum = attnRankNum;
    tilingData->moeDistributeDispatchInfo.routeExpertNumPerMoe = routeExpertNumPerMoe;
    tilingData->moeDistributeDispatchInfo.moeRankId = moeRankId;
    tilingData->moeDistributeDispatchInfo.worldSize = worldSize;
    tilingData->moeDistributeDispatchInfo.tpSize = tpSize;
    tilingData->moeDistributeDispatchInfo.maxTokenNum = maxTokenNum;
    tilingData->moeDistributeDispatchInfo.dynamicQuant = dynamicQuant;

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);

    tilingData->moeDistributeDispatchInfo.aivNum = aivNum;
    tilingData->moeDistributeDispatchInfo.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchInfo.totalWorkspaceSize = sharedMemSize;

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
    auto attrs = context->GetAttrs();
    int64_t maxSeqLen = *(attrs->GetInt(ATTR_ENUM_MAX_SEQ_LEN));
    int64_t hiddenSize = *(attrs->GetInt(ATTR_ENUM_HIDDEN_SIZE));
    int64_t topk = *(attrs->GetInt(ATTR_ENUM_TOPK));
    int64_t moeRankNum = *(attrs->GetInt(ATTR_ENUM_MOE_RANK_NUM));
    int64_t routeExpertNumPerMoe = *(attrs->GetInt(ATTR_ENUM_ROUTE_EXPERT_NUM_PER_MOE));
    int64_t tpSize = *(attrs->GetInt(ATTR_ENUM_TP_SIZE));
    int64_t dynamicQuant = *(attrs->GetInt(ATTR_ENUM_DYNAMIC_QUANT));

    // max token count a routing expert can receive
    float maxSeqLenFactor = optiling::GetBatchSizeFactor();
    OPS_ERR_IF(!(maxSeqLenFactor > 0.0f && maxSeqLenFactor <= 1.0f),
        OPS_LOG_E(context->GetNodeName(), "maxSeqLenFactor is invalid, only support (0, 1], but got maxSeqLenFactor=%f.",
            maxSeqLenFactor), return ge::GRAPH_FAILED);
    int64_t maxTokenNum = (int64_t)(LIMIT_MAX_SEQ_LEN_MAX * maxSeqLenFactor);

    gert::Shape *expandXOutShape = context->GetOutputShape(OUTPUT_EXPAND_X);
    expandXOutShape->SetDimNum(2);
    expandXOutShape->SetDim(0, maxTokenNum);
    expandXOutShape->SetDim(1, hiddenSize);

    // scales for tokens received by routing experts
    gert::Shape *dynamicScalesOutShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES);
    dynamicScalesOutShape->SetDimNum(1);
    if (dynamicQuant != 0) {
        dynamicScalesOutShape->SetDim(0, maxTokenNum);
    } else {
        dynamicScalesOutShape->SetDim(0, 1);
    }

    int64_t batchInfoNum = INFO_NUM + tpSize + routeExpertNumPerMoe * tpSize;
    gert::Shape *batchInfoOutShape = context->GetOutputShape(OUTPUT_BATCH_INFO);
    batchInfoOutShape->SetDimNum(1);
    batchInfoOutShape->SetDim(0, batchInfoNum);

    gert::Shape *epRecvCountRoutedOutShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNT_ROUTED);
    epRecvCountRoutedOutShape->SetDimNum(1);
    epRecvCountRoutedOutShape->SetDim(0, routeExpertNumPerMoe);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    int64_t dynamicQuant = *(attrs->GetInt(ATTR_ENUM_DYNAMIC_QUANT));

    auto xDtype = context->GetInputDesc(0)->GetDataType();

    if (dynamicQuant != 0) {
        context->SetOutputDataType(OUTPUT_EXPAND_X, ge::DT_INT8);
    } else {
        context->SetOutputDataType(OUTPUT_EXPAND_X, xDtype);
    }

    context->SetOutputDataType(OUTPUT_DYNAMIC_SCALES, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_BATCH_INFO, ge::DT_INT64);
    context->SetOutputDataType(OUTPUT_EP_RECV_COUNT_ROUTED, ge::DT_INT64);

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AfdAsyncDispatchRecv : public OpDef {
public:
    explicit AfdAsyncDispatchRecv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("commArgs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("expandX")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dynamicScales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("batchInfo")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("epRecvCountRouted")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
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
        this->Attr("dynamicQuant").Int();
        this->Attr("hccl_group_name").String();

        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup({"hccl_group_name"});

        // support graph
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
    }
};

OP_ADD(AfdAsyncDispatchRecv);

} // namespace ops
