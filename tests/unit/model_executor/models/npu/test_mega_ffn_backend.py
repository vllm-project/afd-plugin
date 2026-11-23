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


@pytest.mark.parametrize("mode", ["normal", "persistent"])
def test_shapes_reuse_graphs_and_padding_does_not_leak_old_inputs(monkeypatch, mode):
    # long-test: one adapter lifetime verifies reuse, changing inputs and shutdown.
    built = []
    calls = []
    conversions = []
    lifecycle = []

    class Runner:
        def __init__(self, args, device_id, weights):
            self.args = args

        def build_graph(self, tokens, persistent=False):
            self.tokens = tokens
            built.append(tokens)
            self.persistent = persistent
            self.running = False

        def start_persistent(self):
            assert self.persistent and not self.running
            assert lifecycle == []  # Native runtime supports only one active graph.
            self.running = True
            lifecycle.append("start")

        def stop_persistent(self):
            assert self.running
            self.running = False
            lifecycle.append("stop")

        def submit_ffn(self, hidden, routing, ids):
            assert self.running
            return self.calculate(hidden, routing, ids)

        def run_ffn(self, hidden, routing, ids):
            assert not self.persistent
            return self.calculate(hidden, routing, ids)

        def calculate(self, hidden, routing, ids):
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
    adapter = MegaFFNBackend(experts, config, 0, max_tokens=8, execution_mode=mode)
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
    assert built == ([8] if mode == "persistent" else [8, 1])
    assert len(conversions) == 1
    with pytest.raises(ValueError, match="capacity"):
        adapter.forward(
            torch.ones(9, 64, dtype=torch.bfloat16),
            topk_ids=torch.zeros(9, 2, dtype=torch.int32),
            topk_weights=torch.ones(9, 2),
        )
    adapter.close()
    adapter.close()
    assert lifecycle == (["start", "stop"] if mode == "persistent" else [])
    with pytest.raises(RuntimeError, match="closed"):
        adapter.forward(x, topk_ids=ids, topk_weights=routing)


def test_pooled_layers_select_distinct_weights_share_lifetime_and_check_local_ids(monkeypatch):
    # long-test: model lifetime includes alternating layers, retained outputs and stop.
    from afd_plugin.model_executor.models.npu.mega_ffn_backend import (
        prepare_persistent_expert_pool,
    )

    lifecycle = []

    class Runner:
        def __init__(self, args, device_id, weights):
            self.weights = weights
            assert args.n_routed_experts == 8

        def build_graph(self, tokens, persistent=False):
            assert persistent

        def start_persistent(self):
            lifecycle.append("start")

        def stop_persistent(self):
            lifecycle.append("stop")

        def submit_ffn(self, hidden, routing, ids):
            factors = self.weights["w13_weight"][ids.long()]
            return hidden * (factors * routing).sum(-1, keepdim=True)

    for name in ("ffn_args", "ffn_runtime", "weight_loader"):
        module = ModuleType("blockrt.models.mega_ffn." + name)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        if name == "ffn_args":
            module.MegaFfnArgs = lambda **kw: SimpleNamespace(**kw)
        elif name == "ffn_runtime":
            module.MegaFfnRunner = Runner
        else:
            module.convert_routed_expert_weights = lambda args, w13, w2: {
                "w13_weight": w13.clone(), "w2_weight": w2.clone(),
                "w13_weight_scale": torch.ones_like(w13),
                "w2_weight_scale": torch.ones_like(w2),
            }
    config = SimpleNamespace(hidden_size=64, moe_intermediate_size=64,
                             n_routed_experts=4, num_experts_per_tok=2)
    layers = [
        MegaFFNBackend(SimpleNamespace(w13_weight=torch.arange(1, 5.) * scale,
                                      w2_weight=torch.ones(4)), config, index,
                       max_tokens=8, execution_mode="persistent")
        for index, scale in [(2, 1), (5, 10)]
    ]
    prepare_persistent_expert_pool(layers)
    prepare_persistent_expert_pool(layers)  # Model post-load hooks may repeat.
    previous = []
    for index, m, expected in [(0, 8, 3.5), (1, 1, 35.), (0, 3, 3.5)]:
        x = torch.ones(m, 64, dtype=torch.bfloat16)
        ids = torch.tensor([1, 3], dtype=torch.int32).repeat(m, 1)
        out = layers[index].forward(x, topk_ids=ids,
                                    topk_weights=torch.tensor([0.25, 0.75]).repeat(m, 1))
        torch.testing.assert_close(out, torch.full_like(out, expected), check_dtype=False)
        previous.append((out, out.clone()))
    for out, saved in previous:
        torch.testing.assert_close(out, saved)
    for bad in (-1, 4):
        with pytest.raises(ValueError, match="local expert"):
            layers[0].forward(x, topk_ids=torch.full_like(ids, bad),
                              topk_weights=torch.ones_like(ids).float())
    assert lifecycle == ["start"]
    for layer in layers:
        layer.close()
        layer.close()
    assert lifecycle == ["start", "stop"]
    with pytest.raises(RuntimeError, match="closed"):
        layers[1].forward(x, topk_ids=ids, topk_weights=torch.ones_like(ids).float())
