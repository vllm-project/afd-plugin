/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * License: CANN Open Software License Agreement Version 2.0
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: Register operator inputs, outputs, attributes, and supported devices.
 * Note:
 * History: 2026-11-17 adapt the split Mega operator for AFD integration.
 */

#include "register/op_def_registry.h"

namespace ops {
class AfdAsyncMegaFfnArch35 : public OpDef {
public:
    explicit AfdAsyncMegaFfnArch35(const char *name) : OpDef(name)
    {
        this->Input("context").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // Each column pairs both weights: E4M3/ND, E4M3/NZ, E5M2/ND, E2M1/NZ_C0_32.
        // Keep dtype and format correlated, matching the host tiling admission checks.
        this->Input("weight1").ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32});
        this->Input("weight2").ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ_C0_32});
        this->Input("weight_scales1").ParamType(DYNAMIC).DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("weight_scales2").ParamType(DYNAMIC).DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("expert_token_nums").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Attr("configuration").AttrType(REQUIRED).ListInt();
        this->Attr("activation_parameters").AttrType(REQUIRED).ListFloat();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque").ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "afd_async_mega_ffn_arch35");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(AfdAsyncMegaFfnArch35);
}
