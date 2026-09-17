# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Where the DSV4 router contract finds its gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402


def test_the_routing_spec_reads_the_gate_through_ffn():
    # A decoder layer has no gate of its own: on Attention it hangs off the
    # remote-FFN proxy, on FFN off the native MoE. Reading `.gate` on the layer
    # was an AttributeError waiting for its first caller.
    gate = SimpleNamespace(out_dtype=torch.float32, weight=SimpleNamespace(dtype=None))
    layer = SimpleNamespace(ffn=SimpleNamespace(gate=gate))
    model = object.__new__(adapter.AFDDeepseekV4Model)
    model.layers = {3: layer}
    model.config = SimpleNamespace(n_routed_experts=64)

    spec = model.get_experts_routing_spec(3)

    assert spec.router_logits_width == 64
    assert spec.router_logits_dtype is torch.float32


def test_the_routing_spec_falls_back_to_the_gate_weight_dtype():
    gate = SimpleNamespace(out_dtype=None, weight=SimpleNamespace(dtype=torch.bfloat16))
    model = object.__new__(adapter.AFDDeepseekV4Model)
    model.layers = {0: SimpleNamespace(ffn=SimpleNamespace(gate=gate))}
    model.config = SimpleNamespace(n_routed_experts=8)

    assert model.get_experts_routing_spec(0).router_logits_dtype is torch.bfloat16
