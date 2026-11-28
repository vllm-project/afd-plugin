// Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
// Single TU: upstream conversion helper contains non-inline definitions.
// Isolate it from CAM's existing pytorch_npu_helper.hpp in a separate native DSO.
#include <torch/extension.h>
#include "aclnn_common.h"
#include "mega_sequence_limit.h"
#include "comm_context.cpp"
#include "atten_send.cpp"
#include "atten_recv.cpp"
#include "mega_ffn.cpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.attr("SPLIT_SEQUENCE_MAX") = AfdAsync::SPLIT_SEQUENCE_MAX;
    op_api::RegisterCommContext(m);
    afd_async_atten_send::RegisterAttenSend(m);
    afd_async_atten_recv::RegisterAttenRecv(m);
    afd_async_mega_ffn::RegisterMegaFfn(m);
}
