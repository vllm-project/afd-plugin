// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/afd_mega_moe_comm_tiling.h"
#include "../op_kernel/afd_pull_layout.h"
#include "ops_log.h"
namespace optiling {
static ge::graphStatus MegaCommTiling(gert::TilingContext *ctx) {
    const auto attrs = ctx->GetAttrs();
    const auto config = attrs->GetListInt(0);
    if (!config || config->GetSize() != 11) return ge::GRAPH_FAILED;
    const auto c = config->GetData();
    if ((c[0] != 0 && c[0] != 1 && c[0] != 4) || c[8] <= 0 || c[1] % c[8] || c[9] < 0 ||
        c[7] < 0 || c[7] >= c[4] + c[5] || c[10] < 0 || c[10] >= 65536 ||
        ((c[0] == 1 || c[0] == 4) && (c[7] >= c[5] || c[9] > c[1] / c[8]))) return ge::GRAPH_FAILED;
    Cam::Pull::Shape shape{uint64_t(c[1] / c[8]), uint64_t(c[2]), uint64_t(c[3]),
                           uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[8])};
    Cam::Pull::Layout layout{};
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const auto aiv = platform.GetCoreNumAiv();
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (!aiv || aiv > Cam::Pull::MAX_AIVS || !Cam::Pull::MakeLayout(shape, layout) ||
        layout.bytes > GetMaxWindowSize() || Cam::Pull::TransportUbBytes(shape) > ub ||
        (c[0] == 1 && Cam::Pull::PublisherUbBytes(shape, aiv) + Cam::Pull::TransportUbBytes(shape) > ub) ||
        (c[0] == 4 && Cam::Pull::CombineUbBytes(shape, aiv) > ub))
        return ge::GRAPH_FAILED;
    auto data = ctx->GetTilingData<Cam::AfdMegaMoeCommTilingData>();
    std::memset(data, 0, sizeof(*data));
    std::copy(c, c + 11, data->config);
    data->aivNum = aiv; data->ubSize = ub; data->windowSize = GetMaxWindowSize();
    AscendC::Mc2CcTilingConfig comm(attrs->GetAttrPointer<char>(1), 8U,
                                  "AlltoAll=level0:fullmesh;level1:pairwise");
    comm.GetTiling(data->mc2InitTiling); comm.GetTiling(data->mc2CcTiling1);
    ctx->SetBlockDim(aiv); ctx->SetTilingKey(1); ctx->SetScheduleMode(1);
    ctx->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class AfdMegaMoeComm : public OpDef {
public:
    explicit AfdMegaMoeComm(const char *name) : OpDef(name) {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("expertIds").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        Input("expertScales").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        Input("state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Output("combined").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Output("completion").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        Attr("config").ListInt(); Attr("hccl_group_name").String();
        SetInferShape([](gert::InferShapeContext *ctx) {
            const auto c = ctx->GetAttrs()->GetListInt(0)->GetData();
            const int64_t sizes[] = {c[0] == 4 ? std::max<int64_t>(1, c[9]) * c[2] : 1, 1};
            for (uint32_t j = 0; j < 2; ++j) {
                auto shape = ctx->GetOutputShape(j); shape->SetDimNum(1); shape->SetDim(0, sizes[j]);
            }
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext *ctx) {
            const ge::DataType types[] = {ge::DT_BF16,ge::DT_INT64};
            for (uint32_t j = 0; j < 2; ++j) ctx->SetOutputDataType(j, types[j]);
            return ge::GRAPH_SUCCESS;
        });
        AICore().AddConfig("ascend910_93"); AICore().SetTiling(optiling::MegaCommTiling);
        MC2().HcclGroup({"hccl_group_name"});
    }
};
OP_ADD(AfdMegaMoeComm);
}
