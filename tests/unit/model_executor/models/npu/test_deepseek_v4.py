# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native V4 model contracts for both routed-expert transports."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend.models.deepseek_v4.model")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402
from tests.unit.fakes.weights import OneShotWeights  # noqa: E402


@pytest.mark.parametrize("connector", ["CAMAsyncAFDConnector", "FaaSAsyncAFDConnector"])
def test_attention_native_loader_receives_shared_weights_for_both_transports(
    connector, monkeypatch
):
    names = [
        "layers.0.attn.fused_wqa_wkv.weight",
        "layers.0.ffn.gate.tid2eid",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.ffn.shared_experts.w1.weight_scale",
        "layers.0.ffn.experts.0.w1.weight",
    ]
    weights = OneShotWeights(names)
    monkeypatch.setattr(
        adapter.native.AscendDeepseekV4ForCausalLM,
        "load_weights",
        lambda self, filtered: {name for name, _ in filtered},
    )
    model = object.__new__(adapter.AFDDeepseekV4ForCausalLM)
    object.__setattr__(model, "afd_role", "attention")
    object.__setattr__(
        model,
        "afd_config",
        AFDConfig(connector=connector, compute_gate_on_attention=True),
    )

    loaded = model.load_weights(weights)

    assert loaded == set(names[:-1])
    assert weights.iterations == 1
