/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/e2a_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"

using namespace Moe;
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

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
}

namespace optiling {
    static ge::graphStatus E2aTilingFuncImpl(gert::TilingContext* context)
    {
        E2ATilingData *tiling = context->GetTilingData<E2ATilingData>();

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

        if (rank < 0 || rank >= expertRankSize + attentionRankSize) {
            printf("[ERROR] CAM E2A PARAMETER INVALID: rank must >= 0 and < expertRankSize + attentionRankSize, "
                    "but rank = %d, expertRankSize = %d, attentionRankSize = %d\n", rank, expertRankSize, attentionRankSize);
            return ge::GRAPH_FAILED;
        }

        tiling->batchSize = batchSize;
        tiling->hiddenSize = hiddenSize;
        tiling->topk = topk;
        tiling->expertRankSize = expertRankSize;
        tiling->attentionRankSize = attentionRankSize;
        tiling->rank = rank;

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

    static ge::graphStatus E2aTilingFunc(gert::TilingContext *context)
    {
        ge::graphStatus ret = E2aTilingFuncImpl(context);
        return ret;
    }

    struct E2aInfo {};
    ge::graphStatus TilingParseForE2a(gert::TilingParseContext *context)
    {
        (void)context;
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(E2a)
        .Tiling(E2aTilingFunc)
        .TilingParse<E2aInfo>(TilingParseForE2a);
}
