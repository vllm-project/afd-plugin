// Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
// Single TU: upstream conversion helper contains non-inline definitions.
// Isolate it from CAM's existing pytorch_npu_helper.hpp in a separate native DSO.
#include <torch/extension.h>
#include "aclnn_common.h"
#include "megamoe_sequence_limit.h"
#include "comm_context.cpp"
#include "megamoe_dispatch_send.cpp"
#include "megamoe_combine_recv.cpp"
#include "megamoe_fused_moe.cpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.attr("SPLIT_SEQUENCE_MAX") = CamMegamoe::SPLIT_SEQUENCE_MAX;
    op_api::RegisterCommContext(m);
    cam_megamoe_dispatch_send::RegisterMegamoeDispatchSend(m);
    cam_megamoe_combine_recv::RegisterMegamoeCombineRecv(m);
    cam_megamoe_fused_moe::RegisterMegamoeFusedMoe(m);
}
