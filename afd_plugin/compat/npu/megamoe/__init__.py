# Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under CANN OSL v2.
import torch_npu  # Register the NPU backend before loading the native bindings.

# ### PATCH START: load AFD native bindings before schema registration
from afd_plugin.compat.npu.ops import ensure_megamoe_ops_available

ensure_megamoe_ops_available()
# ### PATCH END: load AFD native bindings before schema registration

from .common.megamoe_split import create_megamoe_split_buffer, SplitBuffer, DispatchTicket
from .common.megamoe_split_layout import SplitLayout
from .megamoe_dispatch_send import megamoe_dispatch_send
from .megamoe_fused_moe import megamoe_fused_moe
from .megamoe_combine_recv import megamoe_combine_recv
