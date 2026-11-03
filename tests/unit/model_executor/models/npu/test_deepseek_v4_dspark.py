# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role ownership at the strict native DSpark checkpoint-loader seam."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

import torch  # noqa: E402
from vllm.config.compilation import CompilationMode  # noqa: E402

from afd_plugin.model_executor.models.npu import deepseek_v4_dspark  # noqa: E402
from afd_plugin.model_executor.models.npu.deepseek_v4_dspark import (  # noqa: E402
    _iter_role_weights,
)
from tests.unit.fakes.vllm_config import npu_dspark_config  # noqa: E402


@pytest.fixture
def eager_role_model(monkeypatch):
    class Inner(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.layers = torch.nn.ModuleDict()

        def forward(self, input_ids, positions):
            return input_ids + positions

    config = npu_dspark_config()
    config.quant_config = None
    config.compilation_config.mode = CompilationMode.NONE
    monkeypatch.setattr(deepseek_v4_dspark, "AFDDeepseekV4DSparkModel", Inner)
    monkeypatch.setattr(deepseek_v4_dspark, "ParallelLMHead", torch.nn.Identity)
    monkeypatch.setattr(deepseek_v4_dspark, "LogitsProcessor", torch.nn.Identity)
    return deepseek_v4_dspark.AFDDSparkDeepseekV4ForCausalLM(vllm_config=config)


def test_role_constructor_preserves_native_eager_call(eager_role_model):
    ids, positions = torch.arange(3), torch.arange(3)
    assert torch.equal(
        eager_role_model(input_ids=ids, positions=positions), ids + positions
    )


@pytest.mark.parametrize(
    "name,roles",
    [
        ("mtp.0.ffn.experts.17.w1.weight", {"ffn"}),
        ("mtp.2.ffn.experts.255.w2.weight_scale", {"ffn"}),
        ("mtp.1.mlp.experts.0.w3.scale_bias", {"ffn"}),
        ("mtp.0.ffn.gate.weight", {"attention", "ffn"}),
        ("mtp.0.ffn.gate.e_score_correction_bias", {"attention", "ffn"}),
        ("mtp.0.ffn.shared_experts.w1.weight", {"attention"}),
        ("mtp.0.attn.wq_a.weight", {"attention"}),
        ("mtp.0.main_proj.weight", {"attention"}),
        ("mtp.2.markov_head.embed.weight", {"attention"}),
        ("mtp.2.confidence_head.weight", {"attention"}),
        ("embed.weight", {"attention"}),
        ("head.weight", {"attention"}),
    ],
)
def test_role_receives_only_owned_checkpoint_weights(name, roles):
    weight = torch.ones(1)
    for role in ("attention", "ffn"):
        actual = list(_iter_role_weights(iter([(name, weight)]), role=role))
        assert [key for key, _ in actual] == ([name] if role in roles else [])
        if actual:
            assert actual[0][1] is weight
