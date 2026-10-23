// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <string>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/afd_ffn_faas_tiling.h"
#include "ops_log.h"

namespace optiling {
using namespace AfdFaas;
static bool ProjectionTiling(platform_ascendc::PlatformAscendC &platform,
                             uint32_t fullN, uint32_t fullK,
                             AscendC::tiling::TCubeTiling &result)
{
    uint64_t l1 = 0, l0c = 0, ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    matmul_tiling::MatmulApiTiling tile(platform);
    using matmul_tiling::TPosition;
    using matmul_tiling::CubeFormat;
    using matmul_tiling::DataType;
    tile.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false);
    tile.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, true);
    tile.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16);
    tile.SetBias(false);
    const uint32_t n = std::min(TILE_N, fullN);
    const uint32_t k = std::min(128U, fullK);
    tile.SetOrgShape(TILE_M, fullN, fullK);
    tile.SetShape(TILE_M, n, fullK);
    tile.SetFixSplit(TILE_M, n, k);
    tile.SetBufferSpace(l1, l0c, ub);
    return tile.GetTiling(result) == 0;
}

static ge::graphStatus Tiling(gert::TilingContext *context)
{
    auto *data = context->GetTilingData<AfdFfnFaasTilingData>();
    const auto *attributes = context->GetAttrs();
    const auto *config = attributes->GetListInt(1);
    if (config == nullptr || config->GetSize() != CONFIG_FIELDS) {
        return ge::GRAPH_FAILED;
    }
    const auto *c = static_cast<const int64_t *>(config->GetData());
    for (uint32_t i = 0; i < CONFIG_FIELDS; ++i) {
        data->info.config[i] = c[i];
    }
    const auto mode = *attributes->GetInt(0);
    if (mode < INITIALIZE || mode > FINISH || c[HIDDEN] <= 0 || c[HIDDEN] > 4096 || c[HIDDEN] % 16 ||
        c[INTERMEDIATE] <= 0 || c[INTERMEDIATE] > 4096 || c[INTERMEDIATE] % 16 ||
        c[ATTENTION_RANKS] <= 0 || c[FFN_RANKS] <= 0 || c[ATTENTION_TP] <= 0 ||
        c[ATTENTION_RANKS] % c[ATTENTION_TP] || c[SLOTS] <= 0 || c[SLOTS] > 8 ||
        c[EXPERTS] <= 0 || c[EXPERTS] % c[FFN_RANKS] || c[TOPK] <= 0 || c[TOPK] > c[EXPERTS] ||
        c[TOPK] > c[HIDDEN] || c[CAPACITY] <= 0 || c[CAPACITY] > 4096 ||
        c[WORLD_RANK] < 0 || c[WORLD_RANK] >= c[ATTENTION_RANKS] + c[FFN_RANKS] ||
        c[MODEL_LAYERS] <= 0 || c[LOADED_LAYERS] <= 0 || c[LOADED_LAYERS] > c[MODEL_LAYERS] ||
        c[POLL_BUDGET] <= 0 || c[POLL_BUDGET] > 100000000) {
        return ge::GRAPH_FAILED;
    }
    if ((mode == PUBLISH || mode == COMBINE || mode == EXCHANGE || mode == SCOPED_EXCHANGE || mode == END_SOURCE || mode == PEEK_RESULTS) && c[WORLD_RANK] >= c[ATTENTION_RANKS]) {
        return ge::GRAPH_FAILED;
    }
    if ((mode == EXECUTE || mode == SERVE || mode == PREPARE || mode == FINISH) && c[WORLD_RANK] < c[ATTENTION_RANKS]) {
        return ge::GRAPH_FAILED;
    }
    data->info.mode = mode;
    data->info.inputRows = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    data->info.probabilityBf16 = context->GetInputDesc(2)->GetDataType() == ge::DT_BF16;
    const Layout layout(data->info);
    data->info.windowBytes = GetMaxWindowSize();
    if (std::max(layout.inputBytes, layout.resultBytes) > data->info.windowBytes) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t width = std::max(c[HIDDEN], c[INTERMEDIATE]);
    const uint64_t requiredUb = 2 * width * sizeof(float) +
        3 * c[INTERMEDIATE] * sizeof(float) + c[HIDDEN] * sizeof(float) +
        LINE_BYTES + Layout::Align(c[TOPK] * sizeof(float));
    if (requiredUb > ubBytes) {
        return ge::GRAPH_FAILED;
    }
    data->info.blocks = platform.GetCoreNumAiv();
    const uint32_t cubes = platform.GetCoreNumAic();
    if (data->info.blocks != 2 * cubes) {
        return ge::GRAPH_FAILED;
    }
    if (mode == PUBLISH || mode == COMBINE || mode == EXCHANGE || mode == SCOPED_EXCHANGE || mode == PEEK_RESULTS) {
        // A parallelizes by token (or by route for a diagnostic peek). Small
        // decode payloads do not need a full-device barrier or every AIV while
        // waiting for F; keep the other cores available for shared experts.
        const uint32_t tasks = data->info.inputRows * (mode == PEEK_RESULTS ? c[TOPK] : 1);
        data->info.blocks = std::min(data->info.blocks, std::max(1U, tasks));
    } else if (mode == END_SOURCE || mode == ABORT_SESSION) {
        data->info.blocks = 1;
    }
    if (mode == EXECUTE || mode == SERVE) {
        // The CANN mixed-kernel wrapper clears its Matmul system workspace
        // before our entry, including idle work. Cube-only MatmulImpl avoids
        // KFC execution, but does not remove that wrapper requirement.
        context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
        const ScratchLayout scratch(data->info);
        if (context->GetInputShape(8)->GetStorageShape().GetShapeSize() < scratch.bytes ||
            !ProjectionTiling(platform, 2 * c[INTERMEDIATE], c[HIDDEN], data->gateUpTiling) ||
            !ProjectionTiling(platform, c[HIDDEN], c[INTERMEDIATE], data->downTiling)) {
            return ge::GRAPH_FAILED;
        }
    }
    if ((mode == PREPARE || mode == FINISH) &&
        context->GetInputShape(8)->GetStorageShape().GetShapeSize() < ScratchLayout(data->info).bytes) {
        return ge::GRAPH_FAILED;
    }
    const bool compute = mode == EXECUTE || mode == SERVE;
    context->SetBlockDim(compute ? cubes :
        platform.CalcTschBlockDim(data->info.blocks, 0, data->info.blocks));
    context->SetTilingKey(compute ? 2 : 1);
    const char *group = attributes->GetAttrPointer<char>(2);
    AscendC::Mc2CcTilingConfig communication(group, 8U, "AlltoAll=level0:fullmesh;level1:pairwise");
    communication.GetTiling(data->mc2InitTiling);
    communication.GetTiling(data->mc2CcTiling);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    auto *output = context->GetOutputShape(0);
    if (*context->GetAttrs()->GetInt(0) == AfdFaas::PEEK_RESULTS) {
        const auto *config = context->GetAttrs()->GetListInt(1);
        const auto *c = static_cast<const int64_t *>(config->GetData());
        output->SetDimNum(3);
        output->SetDim(0, context->GetInputShape(0)->GetDim(0));
        output->SetDim(1, c[AfdFaas::TOPK]);
        output->SetDim(2, c[AfdFaas::HIDDEN]);
    } else {
        *output = *context->GetInputShape(0);
    }
    auto *status = context->GetOutputShape(1);
    status->SetDimNum(1);
    status->SetDim(0, AfdFaas::STATUS_FIELDS);
    return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DT_BF16);
    context->SetOutputDataType(1, DT_INT64);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AfdFfnFaas : public OpDef {
public:
    explicit AfdFfnFaas(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("expertIds").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("probabilities").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("w13").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("w2").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("layers").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("request").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("state").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("scratch").ParamType(REQUIRED).DataType({ge::DT_UINT8, ge::DT_UINT8}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("status").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("mode").Int();
        this->Attr("config").ListInt();
        this->Attr("group").String();
        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup({"group"});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        this->AICore().SetTiling(optiling::Tiling);
    }
};
OP_ADD(AfdFfnFaas);
}  // namespace ops
