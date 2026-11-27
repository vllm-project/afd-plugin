# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared NPU DSV4 model shell; connectors choose their own transport tests."""

from types import SimpleNamespace

import torch

from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter
from afd_plugin.model_executor.models.npu import deepseek_v4_attention_gate


def make_attention_gate_proxy(
    monkeypatch,
    *,
    use_sequence_parallel: bool,
    in_profile_run: bool,
    mega: bool = False,
):
    proxy = object.__new__(adapter.AFDDeepseekV4AttentionGateRemoteMoE)
    torch.nn.Module.__init__(proxy)
    proxy.use_sequence_parallel_moe = use_sequence_parallel
    proxy.mega = mega
    proxy.shared_experts = None
    proxy.layer_idx = 3
    connector = SimpleNamespace()
    monkeypatch.setattr(
        adapter,
        "get_afd_metadata_from_forward_context",
        lambda: SimpleNamespace(connector=connector, stage_idx=0),
    )
    monkeypatch.setattr(adapter, "maybe_apply_dbo_yield", lambda states, **kw: states)
    topk_weights = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    topk_ids = torch.arange(32, dtype=torch.int32).reshape(16, 2)
    monkeypatch.setattr(
        deepseek_v4_attention_gate,
        "compute_attention_gate_topk",
        lambda _proxy, _hidden_states, **_kwargs: (topk_weights, topk_ids),
    )
    monkeypatch.setattr(
        adapter,
        "get_forward_context",
        lambda: SimpleNamespace(
            in_profile_run=in_profile_run,
        ),
    )
    return proxy, topk_weights, topk_ids
