/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <string>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/a2e_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"

using namespace Moe;
using namespace ge;
namespace {
constexpr int TILING_KEY_FP16 = 20;
constexpr int TILING_KEY_BF16 = 21;
constexpr int TILING_KEY_FP32 = 22;
constexpr int TILING_KEY_ELSE = 23;

constexpr int ATTR_ENUM_BATCH_SIZE = 0;
constexpr int ATTR_ENUM_HIDDEN_SIZE = 1;
constexpr int ATTR_ENUM_TOPK = 2;
constexpr int ATTR_ENUM_EP_RANK_SIZE = 3;
constexpr int ATTR_ENUM_ATTN_RANK_SIZE = 4;
constexpr int ATTR_ENUM_RANK = 5;
constexpr int ATTR_ENUM_GROUP_EP = 6;
constexpr int ATTR_AIV_NUM = 7;
constexpr int ATTR_COMPUTE_GATE = 8;

constexpr int INPUT_EXPANDX_IDX = 0;
constexpr int INPUT_EXPERT_SCALES_IDX = 2;
constexpr int INPUT_EXPERT_IDS_IDX = 1;

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
}

namespace optiling {
    static ge::graphStatus A2eTilingFuncImpl(gert::TilingContext* context)
    {
        A2ETilingData *tiling = context->GetTilingData<A2ETilingData>();

        auto xDtype = context->GetInputDesc(0)->GetDataType();
        if (xDtype == ge::DT_FLOAT16) {
            context->SetTilingKey(TILING_KEY_FP16);
        } else if (xDtype == ge::DT_BF16) {
            context->SetTilingKey(TILING_KEY_BF16);
        } else if (xDtype == ge::DT_FLOAT) {
            context->SetTilingKey(TILING_KEY_FP32);
        } else {
            context->SetTilingKey(TILING_KEY_ELSE);
        }

        auto attrPointers = context->GetAttrs();
        int batchSize = *(attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE));
        int hiddenSize = *(attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE));
        int topk = *(attrPointers->GetInt(ATTR_ENUM_TOPK));
        int expertRankSize = *(attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE));
        int attentionRankSize = *(attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE));
        int rank = *(attrPointers->GetInt(ATTR_ENUM_RANK));
        int aivAlgNum = *(attrPointers->GetInt(ATTR_AIV_NUM));
        int computeGate = *(attrPointers->GetInt(ATTR_COMPUTE_GATE));

        if (rank < 0 || rank >= expertRankSize + attentionRankSize) {
            printf("[ERROR] CAM A2E PARAMETER INVALID: rank must >= 0 and < expertRankSize + attentionRankSize, "
                    "but rank = %d, expertRankSize = %d, attentionRankSize = %d\n", rank, expertRankSize, attentionRankSize);
            return ge::GRAPH_FAILED;
        }

        tiling->batchSize = batchSize;
        tiling->hiddenSize = hiddenSize;
        tiling->topk = topk;
        tiling->expertRankSize = expertRankSize;
        tiling->attentionRankSize = attentionRankSize;
        tiling->rank = rank;
        tiling->computeGate = computeGate;

        context->SetBlockDim(aivAlgNum);

        auto groupEpPtr = attrPointers->GetAttrPointer<char>(static_cast<int>(ATTR_ENUM_GROUP_EP));
        std::string groupEp = std::string(groupEpPtr);
        uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
        std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
#ifdef AFD_TILING_HAS_COMM_ENGINE
        // On A5 (Ascend 950) the MC2 tiling's commEngine field is a
        // HcclAccelerator, not a CommEngine. The AIV value (=3, see CANN
        // hccl_params/MAKE_ENUM HcclAccelerator: DEFAULT,HOSTCPU_TS,AICPU_TS,
        // AIV) is required for the MTE path. SetCommEngine(2)=AICPU_TS is
        // rejected by HCCL GetTilingAccelerator with HCCL_E_NOT_SUPPORT.
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
            mc2CcTilingConfig.SetCommEngine(3);
        }
#endif
        mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus A2eTilingFunc(gert::TilingContext *context)
    {
        ge::graphStatus ret = A2eTilingFuncImpl(context);
        return ret;
    }

    struct A2eInfo {};
    ge::graphStatus TilingParseForA2e(gert::TilingParseContext *context)
    {
        (void)context;
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(A2e)
        .Tiling(A2eTilingFunc)
        .TilingParse<A2eInfo>(TilingParseForA2e);
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        auto attrPointers = context->GetAttrs();
        int batchSize = *(attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE));
        int hiddenSize = *(attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE));
        int topk = *(attrPointers->GetInt(ATTR_ENUM_TOPK));
        int expertRankSize = *(attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE));
        int attentionRankSize = *(attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE));
        int rank = *(attrPointers->GetInt(ATTR_ENUM_RANK));
        batchSize = batchSize * (attentionRankSize + expertRankSize - 1) / expertRankSize;

        gert::Shape* expandXShape = context->GetOutputShape(0);
        expandXShape->SetDimNum(2);
        if (rank < expertRankSize) {
            expandXShape->SetDim(0, batchSize);
            expandXShape->SetDim(1, hiddenSize);
        } else {
            expandXShape->SetDim(0, 1);
            expandXShape->SetDim(1, 1);
        }

        gert::Shape* simulateExpertIdsShape = context->GetOutputShape(1);
        simulateExpertIdsShape->SetDimNum(2);
        if (rank < attentionRankSize) {
            simulateExpertIdsShape->SetDim(0, batchSize);
            simulateExpertIdsShape->SetDim(1, topk);
        } else {
            simulateExpertIdsShape->SetDim(0, 1);
            simulateExpertIdsShape->SetDim(1, 1);
        }

        gert::Shape* simulateExpertScalesShape = context->GetOutputShape(2);
        simulateExpertScalesShape->SetDimNum(2);
        if (rank < attentionRankSize) {
            simulateExpertScalesShape->SetDim(0, batchSize);
            simulateExpertScalesShape->SetDim(1, topk);
        } else {
            simulateExpertScalesShape->SetDim(0, 1);
            simulateExpertScalesShape->SetDim(1, 1);
        }

        gert::Shape* attenBatchSizeShape = context->GetOutputShape(3);
        attenBatchSizeShape->SetDimNum(1);
        attenBatchSizeShape->SetDim(0, (attentionRankSize + expertRankSize - 1) / expertRankSize);

        gert::Shape* xActiveMaskOutShape = context->GetOutputShape(4);
        xActiveMaskOutShape->SetDimNum(1);
        if (rank < attentionRankSize) {
            xActiveMaskOutShape->SetDim(0, batchSize);
        } else {
            xActiveMaskOutShape->SetDim(0, 1);
        }

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto expertIdsType = context->GetInputDataType(INPUT_EXPERT_IDS_IDX);
        const auto expertScalesType = context->GetInputDataType(INPUT_EXPERT_SCALES_IDX);
        const auto expandXDType = context->GetInputDataType(INPUT_EXPANDX_IDX);

        int outputIdx = 0;
        context->SetOutputDataType(outputIdx++, expandXDType);
        context->SetOutputDataType(outputIdx++, expertIdsType);
        context->SetOutputDataType(outputIdx++, expertScalesType);
        context->SetOutputDataType(outputIdx++, ge::DT_INT32);
        context->SetOutputDataType(outputIdx++, ge::DT_BOOL);
        return ge::GRAPH_SUCCESS;
    }
    IMPL_OP(A2e).InferShape(InferShape).InferDataType(InferDataType);
}

namespace ops {
class A2e : public OpDef {
public:
    explicit A2e(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_ids")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("expand_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("simulate_expert_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("simulate_expert_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("atten_batch_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("x_active_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("batch_size").Int();
        this->Attr("hidden_size").Int();
        this->Attr("topk").Int();
        this->Attr("expert_rank_size").Int();
        this->Attr("attention_rank_size").Int();
        this->Attr("rank").Int();
        this->Attr("group_ep").String();
        this->Attr("aiv_num").Int();
        this->Attr("compute_gate").Int();

        this->MC2().HcclGroup({"group_ep"});
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(A2e);
}  // namespace ops
