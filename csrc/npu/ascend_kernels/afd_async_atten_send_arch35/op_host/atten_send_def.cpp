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
class AfdAsyncAttenSendArch35 : public OpDef {
public:
    explicit AfdAsyncAttenSendArch35(const char *name) : OpDef(name)
    {
        this->Input("context").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("x").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_weights").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("snapshot_data").ParamType(REQUIRED).DataTypeList({ge::DT_UINT8}).FormatList({ge::FORMAT_ND});
        this->Output("snapshot_scales").ParamType(REQUIRED).DataTypeList({ge::DT_UINT8}).FormatList({ge::FORMAT_ND});
        this->Output("completion").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Attr("configuration").AttrType(REQUIRED).ListInt();
        this->Attr("dp_ranks").AttrType(REQUIRED).ListInt();
        this->Attr("activation_parameters").AttrType(REQUIRED).ListFloat();
        this->Attr("layer_index").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque").ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "afd_async_atten_send_arch35");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(AfdAsyncAttenSendArch35);
}
