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
 * Description: Infer output shapes and data types for graph construction.
 * Note:
 * History: 2026-11-17 adapt the split MegaMoE operator for AFD integration.
 */

#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    auto *values = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
    auto *config = reinterpret_cast<const int64_t *>(values->GetData());
    int64_t tokens = config[12] > 0 ? config[13] : 0;
    *context->GetOutputShape(0) = gert::Shape({tokens, config[4]});
    *context->GetOutputShape(1) = gert::Shape({tokens, (config[4] + 63) / 64 * 2});
    *context->GetOutputShape(2) = gert::Shape({1});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT8);
    context->SetOutputDataType(1, ge::DT_UINT8);
    context->SetOutputDataType(2, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(CamMegamoeDispatchSend).InferShape(InferShape).InferDataType(InferType);
}
