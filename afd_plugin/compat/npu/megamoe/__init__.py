# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
import torch_npu  # noqa: F401

# Register the NPU backend before loading the native bindings.
# ### PATCH START: load AFD native bindings before schema registration
from afd_plugin.compat.npu.ops import ensure_megamoe_ops_available

ensure_megamoe_ops_available()
# ### PATCH END: load AFD native bindings before schema registration

from .common.megamoe_split import (  # noqa: E402
    SPLIT_SEQUENCE_MAX,
    DispatchTicket,
    SplitBuffer,
    create_megamoe_split_buffer,
    split_sequence,
)
from .common.megamoe_split_layout import SplitLayout  # noqa: E402
from .megamoe_combine_recv import megamoe_combine_recv  # noqa: E402
from .megamoe_dispatch_send import megamoe_dispatch_send  # noqa: E402
from .megamoe_fused_moe import megamoe_fused_moe  # noqa: E402

__all__ = [
    "SPLIT_SEQUENCE_MAX",
    "DispatchTicket",
    "SplitBuffer",
    "SplitLayout",
    "create_megamoe_split_buffer",
    "megamoe_combine_recv",
    "megamoe_dispatch_send",
    "megamoe_fused_moe",
    "split_sequence",
]
