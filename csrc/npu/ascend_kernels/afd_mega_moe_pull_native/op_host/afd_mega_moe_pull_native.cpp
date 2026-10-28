// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <algorithm>
#include <cmath>
#include <cstring>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/hccl/hccl_tiling.h"
#include "../op_kernel/afd_mega_moe_pull_native_tiling.h"
#include "../op_kernel/afd_pull_layout.h"
#include "ops_log.h"

namespace optiling {
static bool CubeTiling(platform_ascendc::PlatformAscendC &platform, int64_t k, uint8_t *dest) {
    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT4);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    mm.SetBias(false);
    mm.SetShape(128, 256, k);
    mm.SetOrgShape(128, 256, k);
    mm.SetFixSplit(128, 256, 256);
    mm.SetBufferSpace(-1, -1, -1);
    TCubeTiling data;
    if (mm.GetTiling(data) == -1 || data.GetDataSize() > 1024) return false;
    // MC2's K1024 L1 tile and K256 L0 tile; reuse the main MSD double-buffer contract.
    data.set_stepKa(4); data.set_stepKb(4);
    data.set_depthA1(8); data.set_depthB1(8);
    data.set_dbL0B(2); data.set_stepM(1); data.set_stepN(1);
    data.SaveToBuffer(dest, 1024);
    return true;
}
static ge::graphStatus NativeMegaTiling(gert::TilingContext *ctx) {
    const auto attrs = ctx->GetAttrs();
    const auto config = attrs->GetListInt(0);
    if (config == nullptr || config->GetSize() != 14) return ge::GRAPH_FAILED;
    const auto c = config->GetData();
    for (uint32_t j = 0; j < 14; ++j)
        if (c[j] <= 0 || c[j] > 262144) return ge::GRAPH_FAILED;
    if (c[8] != c[4] + c[5] || c[7] < c[5] || c[7] >= c[8] ||
        c[9] == 0 || c[0] % c[9] || c[5] % c[9] ||
        c[2] % 256 || c[2] > 8192 || c[6] > 128 ||
        c[10] > Cam::NATIVE_MAX_ROWS || c[11] != 1 || c[12] != 1 ||
        !std::isfinite(*attrs->GetFloat(2))) return ge::GRAPH_FAILED;
    Cam::Pull::Shape shape{uint64_t(c[0] / c[9]), uint64_t(c[1]), uint64_t(c[3]),
        uint64_t(c[4]), uint64_t(c[5]), uint64_t(c[6]), uint64_t(c[9])};
    Cam::Pull::Layout layout{};
    if (!Cam::Pull::MakeLayout(shape, layout) || layout.bytes > GetMaxWindowSize())
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const auto aic = platform.GetCoreNumAic(), aiv = platform.GetCoreNumAiv();
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (aic == 0 || aiv != 2 * aic || aiv > Cam::NATIVE_MAX_AIVS ||
        Cam::Pull::TransportUbBytes(shape) > ub ||
        29ULL * c[2] + 1024 > ub || 7ULL * c[1] + 256 > ub) return ge::GRAPH_FAILED;
    const Cam::NativeMegaWorkspace w(c[10], c[1], c[2], c[6], c[9]);
    if (ctx->GetInputShape(3)->GetStorageShape().GetDim(0) != int64_t(w.bytes))
        return ge::GRAPH_FAILED;
    auto data = ctx->GetTilingData<Cam::AfdMegaMoePullNativeTilingData>();
    std::memset(data, 0, sizeof(*data));
    std::copy(c, c + 14, data->config);
    data->aicNum = aic;
    data->aivNum = aiv;
    data->commAivNum = std::max(1U, aiv / 4);
    if (!CubeTiling(platform, std::max(c[1], c[2]), data->cube)) return ge::GRAPH_FAILED;
    data->layerMapSize = ctx->GetInputShape(2)->GetStorageShape().GetDim(0);
    if (data->layerMapSize == 0 || data->layerMapSize > 65536) return ge::GRAPH_FAILED;
    data->routedScale = *attrs->GetFloat(2);
    AscendC::Mc2CcTilingConfig comm(attrs->GetAttrPointer<char>(1), 8U,
                                   "AlltoAll=level0:fullmesh;level1:pairwise");
    comm.GetTiling(data->mc2InitTiling);
    comm.GetTiling(data->mc2CcTiling1);
    ctx->SetBlockDim(aic);
    ctx->SetTilingKey(1);
    ctx->SetScheduleMode(1);
    ctx->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AfdMegaMoePullNative : public OpDef {
public:
    explicit AfdMegaMoePullNative(const char *name) : OpDef(name) {
        Input("anchor").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("commArgs").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        Input("layerMap").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        Input("scratch").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        for (const char *w : {"w13", "w2"})
            Input(w).ParamType(DYNAMIC).DataType({ge::DT_INT32}).Format({ge::FORMAT_FRACTAL_NZ});
        for (const char *s : {"s13", "s2"})
            Input(s).ParamType(DYNAMIC).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        for (const char *b : {"b13", "b2"})
            Input(b).ParamType(DYNAMIC).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        Output("completion").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        Attr("config").ListInt();
        Attr("hccl_group_name").String();
        Attr("routed_scale").Float();
        SetInferShape([](gert::InferShapeContext *ctx) {
            auto shape = ctx->GetOutputShape(0);
            shape->SetDimNum(1); shape->SetDim(0, 1);
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext *ctx) {
            ctx->SetOutputDataType(0, ge::DT_INT64); return ge::GRAPH_SUCCESS;
        });
        AICore().AddConfig("ascend910_93");
        AICore().SetTiling(optiling::NativeMegaTiling);
        MC2().HcclGroup({"hccl_group_name"});
    }
};
OP_ADD(AfdMegaMoePullNative);
}
