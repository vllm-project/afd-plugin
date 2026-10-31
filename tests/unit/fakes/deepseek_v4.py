# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native CPU FlashMLA reservation seam; no CUDA profiling or kernel proof."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm import forward_context
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.v1.worker import workspace


class FailingFlashAttention(DeepseekV4FlashMLAAttention):
    def forward_mqa(self, q, kv, positions, output):
        raise self.failure


def profile_attention(compress_ratio, error=None):
    cls = FailingFlashAttention if error is not None else DeepseekV4FlashMLAAttention
    attention = object.__new__(cls)
    nn.Module.__init__(attention)
    attention.compress_ratio = compress_ratio
    attention.max_model_len = 4096
    attention.max_num_batched_tokens = 1024
    attention.window_size = 128
    attention.max_image_tokens = 0
    attention.head_dim = 512
    attention.padded_heads = 64
    attention.attn_sink = nn.Parameter(torch.zeros(64), requires_grad=False)
    attention.topk_indices_buffer = (
        None if compress_ratio <= 1 else torch.empty((1, 512), dtype=torch.int32)
    )
    if error is not None:
        attention.failure = error
    return attention


class WorkspaceCase:
    def __init__(self, monkeypatch, *, error=None):
        monkeypatch.setattr(
            forward_context.current_platform,
            "set_additional_forward_context",
            lambda *args, **kwargs: {},
        )
        self.ubatch = 0
        self.manager = workspace.WorkspaceManager(torch.device("cpu"), 2, num_lanes=2)
        monkeypatch.setattr(workspace, "_manager", self.manager)
        monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: self.ubatch)
        monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v4")
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                use_sequence_parallel_moe=False,
                is_moe_model=True,
                use_ubatching=True,
            ),
            compilation_config=SimpleNamespace(
                fast_moe_cold_start=False,
                static_forward_context={},
            ),
        )
        self.attention = [profile_attention(r) for r in (1, 4, 128)]
        if error is not None:
            self.attention = [profile_attention(4, error)]
        self.model = nn.ModuleList(self.attention)

    def request(self, ubatch, lane):
        self.ubatch = ubatch
        with workspace.use_workspace_lane(lane), set_forward_context(None, self.config):
            for attention in self.attention:
                q = torch.empty(
                    (1, attention.padded_heads, attention.head_dim),
                    dtype=torch.bfloat16,
                )
                attention.forward_mqa(
                    q,
                    q[:, 0, :],
                    torch.zeros(1, dtype=torch.int64),
                    torch.empty_like(q),
                )
            return self.manager.get_simultaneous(((1,), torch.uint8))[0]
