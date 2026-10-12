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
#include <limits>
#include "../op_kernel/a2e_packet.h"

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
constexpr int TILING_KEY_MXFP8 = 23;
constexpr int TILING_KEY_ELSE = 24;

constexpr int ATTR_ENUM_BATCH_SIZE = 0;
constexpr int ATTR_ENUM_HIDDEN_SIZE = 1;
constexpr int ATTR_ENUM_TOPK = 2;
constexpr int ATTR_ENUM_EP_RANK_SIZE = 3;
constexpr int ATTR_ENUM_ATTN_RANK_SIZE = 4;
constexpr int ATTR_ENUM_RANK = 5;
constexpr int ATTR_ENUM_GROUP_EP = 6;
constexpr int ATTR_AIV_NUM = 7;
constexpr int ATTR_COMPUTE_GATE = 8;
constexpr int ATTR_QUANT_MODE = 9;

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
        const bool isFp8 = xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT8_E5M2;
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
        const int64_t batchSize64 = *attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE);
        const int64_t hiddenSize64 = *attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE);
        const int64_t topk64 = *attrPointers->GetInt(ATTR_ENUM_TOPK);
        const int64_t expertRankSize64 = *attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE);
        const int64_t attentionRankSize64 = *attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE);
        const int64_t rank64 = *attrPointers->GetInt(ATTR_ENUM_RANK);
        const int64_t aivAlgNum64 = *attrPointers->GetInt(ATTR_AIV_NUM);
        const int64_t computeGate64 = *attrPointers->GetInt(ATTR_COMPUTE_GATE);
        auto quantModePtr = attrPointers->GetInt(ATTR_QUANT_MODE);
        const int64_t quantMode = quantModePtr == nullptr ? 0 : *quantModePtr;
        const int64_t maxInt = std::numeric_limits<int32_t>::max();
        if (batchSize64 < 0 || batchSize64 > maxInt || hiddenSize64 <= 0 || hiddenSize64 > maxInt ||
            topk64 <= 0 || topk64 > maxInt || expertRankSize64 <= 0 || expertRankSize64 > maxInt ||
            attentionRankSize64 <= 0 || attentionRankSize64 > maxInt || rank64 < 0 || rank64 > maxInt ||
            aivAlgNum64 <= 0 || aivAlgNum64 > maxInt || computeGate64 < 0 || computeGate64 > maxInt) {
            return ge::GRAPH_FAILED;
        }
        const int batchSize = static_cast<int>(batchSize64);
        const int hiddenSize = static_cast<int>(hiddenSize64);
        const int topk = static_cast<int>(topk64);
        const int expertRankSize = static_cast<int>(expertRankSize64);
        const int attentionRankSize = static_cast<int>(attentionRankSize64);
        const int rank = static_cast<int>(rank64);
        const int aivAlgNum = static_cast<int>(aivAlgNum64);
        const int computeGate = static_cast<int>(computeGate64);
        if (quantMode != 0 && quantMode != A2E_QUANT_MODE_MXFP8) {
            return ge::GRAPH_FAILED;
        }
        if (quantMode == A2E_QUANT_MODE_MXFP8) {
            auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            const int64_t batch = *attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE);
            const int64_t hidden = *attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE);
            const int64_t k = *attrPointers->GetInt(ATTR_ENUM_TOPK);
            const int64_t experts = *attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE);
            const int64_t attention = *attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE);
            const int64_t cores = *attrPointers->GetInt(ATTR_AIV_NUM);
            const int64_t gate = *attrPointers->GetInt(ATTR_COMPUTE_GATE);
            const int64_t maxInt = std::numeric_limits<int32_t>::max();
            if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND950 ||
                batch <= 0 || batch > maxInt || hidden <= 0 || hidden > maxInt - 31 ||
                k <= 0 || k > maxInt || experts <= 0 || experts > A2E_A5_MAX_RANKS / 2 ||
                attention != experts || rank64 >= experts + attention ||
                cores <= 0 || cores > platform.GetCoreNumAiv() ||
                (gate != 0 && gate != 1) || (gate == 1 && cores < 3) ||
                batch * hidden > maxInt - cores || batch * k > maxInt / 8 ||
                (gate == 1 && A2EAlignPacket(batch * k * sizeof(int32_t)) > 192 * 1024 - 1024)) {
                printf("[ERROR] A2E MXFP8 invalid dimensions, ranks, cores, or platform\n");
                return ge::GRAPH_FAILED;
            }
            auto layout = GetA2EPacketLayout(batch, hidden, k, 1, quantMode);
            const auto scaleDesc = context->GetOptionalInputDesc(3);
            const auto scaleShape = context->GetOptionalInputShape(3);
            const auto xShape = context->GetInputShape(0);
            const bool sender = rank >= experts;
            const bool validReceiverRows = xShape->GetStorageShape().GetDim(0) == 0 ||
                xShape->GetStorageShape().GetDim(0) == batch;
            const bool validScale = scaleDesc != nullptr && scaleShape != nullptr &&
                scaleDesc->GetDataType() == ge::DT_FLOAT8_E8M0 &&
                scaleShape->GetStorageShape().GetDimNum() == 2 &&
                scaleShape->GetStorageShape().GetDim(1) == (hidden + 31) / 32 &&
                scaleShape->GetStorageShape().GetDim(0) == xShape->GetStorageShape().GetDim(0);
            const auto scaleOutDesc = context->GetOutputDesc(5);
            if (!isFp8 || (sender && !validScale) ||
                (!sender && (scaleDesc != nullptr || scaleShape != nullptr) && !validScale) ||
                scaleOutDesc == nullptr || scaleOutDesc->GetDataType() != ge::DT_FLOAT8_E8M0 ||
                layout.endOffset > maxInt || xShape->GetStorageShape().GetDimNum() != 2 ||
                xShape->GetStorageShape().GetDim(1) != hidden ||
                (sender ? xShape->GetStorageShape().GetDim(0) != batch : !validReceiverRows)) {
                printf("[ERROR] A2E MXFP8 requires E4M3 or E5M2 x and matching E8M0 x_scale\n");
                return ge::GRAPH_FAILED;
            }
            if (sender && gate == 1) {
                const auto idsDesc = context->GetOptionalInputDesc(INPUT_EXPERT_IDS_IDX);
                const auto idsShape = context->GetOptionalInputShape(INPUT_EXPERT_IDS_IDX);
                const auto routingScalesDesc = context->GetOptionalInputDesc(INPUT_EXPERT_SCALES_IDX);
                const auto routingScalesShape = context->GetOptionalInputShape(INPUT_EXPERT_SCALES_IDX);
                if (idsDesc == nullptr || idsShape == nullptr ||
                    routingScalesDesc == nullptr || routingScalesShape == nullptr ||
                    idsDesc->GetDataType() != ge::DT_INT32 ||
                    routingScalesDesc->GetDataType() != ge::DT_FLOAT ||
                    idsShape->GetStorageShape().GetDimNum() != 2 ||
                    routingScalesShape->GetStorageShape().GetDimNum() != 2 ||
                    idsShape->GetStorageShape().GetDim(0) != batch ||
                    idsShape->GetStorageShape().GetDim(1) != k ||
                    routingScalesShape->GetStorageShape().GetDim(0) != batch ||
                    routingScalesShape->GetStorageShape().GetDim(1) != k) {
                    printf("[ERROR] A2E MXFP8 compute_gate=1 senders require int32 expert_ids and "
                        "float32 scales with shape [batch_size, topk]\n");
                    return ge::GRAPH_FAILED;
                }
            }
            context->SetTilingKey(TILING_KEY_MXFP8);
        } else if (isFp8) {
            return ge::GRAPH_FAILED;
        }

        if (rank64 >= expertRankSize64 + attentionRankSize64) {
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
        tiling->quantMode = quantMode;

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
        int64_t batchSize = *(attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE));
        int64_t hiddenSize = *(attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE));
        int64_t topk = *(attrPointers->GetInt(ATTR_ENUM_TOPK));
        int64_t expertRankSize = *(attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE));
        int64_t attentionRankSize = *(attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE));
        int64_t rank = *(attrPointers->GetInt(ATTR_ENUM_RANK));
        auto quantModePtr = attrPointers->GetInt(ATTR_QUANT_MODE);
        int64_t quantMode = quantModePtr == nullptr ? 0 : *quantModePtr;
        const int64_t maxInt = std::numeric_limits<int32_t>::max();
        if (expertRankSize <= 0 || expertRankSize > maxInt || attentionRankSize <= 0 ||
            attentionRankSize > maxInt || batchSize < 0 || batchSize > maxInt ||
            hiddenSize <= 0 || hiddenSize > maxInt - 31 || topk <= 0 || topk > maxInt ||
            rank < 0 || rank >= expertRankSize + attentionRankSize ||
            (quantMode != 0 && quantMode != A2E_QUANT_MODE_MXFP8) ||
            (quantMode == A2E_QUANT_MODE_MXFP8 &&
             (expertRankSize != attentionRankSize || expertRankSize > A2E_A5_MAX_RANKS / 2))) {
            return GRAPH_FAILED;
        }
        if (quantMode == 0) {
            batchSize = batchSize * (attentionRankSize + expertRankSize - 1) / expertRankSize;
        }

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

        gert::Shape* xScaleOutShape = context->GetOutputShape(5);
        if (xScaleOutShape != nullptr) {
            xScaleOutShape->SetDimNum(2);
            xScaleOutShape->SetDim(0, rank < attentionRankSize ? batchSize : 1);
            xScaleOutShape->SetDim(1, rank < attentionRankSize ? (hiddenSize + 31) / 32 : 1);
        } else if (quantMode == A2E_QUANT_MODE_MXFP8) {
            return GRAPH_FAILED;
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
        if (context->GetComputeNodeOutputNum() > static_cast<size_t>(outputIdx)) {
            context->SetOutputDataType(outputIdx, ge::DT_FLOAT8_E8M0);
        }
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
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_ids")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("x_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("expand_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("simulate_expert_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("simulate_expert_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("atten_batch_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("x_active_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("x_scale_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("batch_size").Int();
        this->Attr("hidden_size").Int();
        this->Attr("topk").Int();
        this->Attr("expert_rank_size").Int();
        this->Attr("attention_rank_size").Int();
        this->Attr("rank").Int();
        this->Attr("group_ep").String();
        this->Attr("aiv_num").Int();
        this->Attr("compute_gate").Int();
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);

        this->MC2().HcclGroup({"group_ep"});
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(A2e);
}  // namespace ops
