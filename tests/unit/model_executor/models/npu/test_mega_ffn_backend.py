# SPDX-License-Identifier: Apache-2.0
"""Adapter lifetime/IO checks with a CPU runner, without loading native blockrt."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from afd_plugin.model_executor.models.npu.mega_ffn_backend import (  # noqa: E402
    MegaFFNBackend,
)


def test_shapes_share_two_graphs_and_padding_does_not_leak_old_inputs(monkeypatch):
    built = []
    calls = []
    conversions = []

    class Runner:
        def __init__(self, args, device_id, weights):
            self.args = args

        def build_graph(self, tokens):
            self.tokens = tokens
            built.append(tokens)

        def run_ffn(self, hidden, routing, ids):
            assert hidden.shape == (self.tokens, 64)
            assert routing.shape == ids.shape == (self.tokens, 2)
            calls.append((hidden.clone(), routing.clone(), ids.clone()))
            return hidden * routing.sum(-1, keepdim=True)

    modules = {}
    for name in ("ffn_args", "ffn_runtime", "weight_loader"):
        module = ModuleType("blockrt.models.mega_ffn." + name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, module.__name__, module)
    modules["ffn_args"].MegaFfnArgs = lambda **kw: SimpleNamespace(**kw)
    modules["ffn_runtime"].MegaFfnRunner = Runner

    def convert(*args):
        conversions.append(args)
        return {}

    modules["weight_loader"].convert_routed_expert_weights = convert
    config = SimpleNamespace(
        hidden_size=64,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    experts = SimpleNamespace(w13_weight=object(), w2_weight=object())
    adapter = MegaFFNBackend(experts, config, 0, max_tokens=8)
    for m in (8, 1, 3, 7, 2, 1):
        x = torch.arange(m * 64).reshape(m, 64).bfloat16()
        routing = torch.tensor([0.25, 1.25]).repeat(m, 1)
        ids = torch.tensor([1, 3], dtype=torch.int32).repeat(m, 1)
        output = adapter.forward(x, topk_ids=ids, topk_weights=routing)
        torch.testing.assert_close(output, x * 1.5, check_dtype=False)
        h, r, i = calls[-1]
        torch.testing.assert_close(h[:m], x)
        torch.testing.assert_close(r[:m], routing)
        torch.testing.assert_close(i[:m].int(), ids)
        assert not h[m:].count_nonzero() and not r[m:].count_nonzero()
        assert ((i >= 0) & (i < 4)).all()
    assert built == [8, 1]
    assert len(conversions) == 1
    with pytest.raises(ValueError, match="capacity"):
        adapter.forward(
            torch.ones(9, 64, dtype=torch.bfloat16),
            topk_ids=torch.zeros(9, 2, dtype=torch.int32),
            topk_weights=torch.ones(9, 2),
        )
    assert built == [8, 1]
