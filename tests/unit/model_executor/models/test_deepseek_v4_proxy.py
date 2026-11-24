# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from torch import nn  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.distributed import parallel_state  # noqa: E402

from afd_plugin.model_executor.ffn import remote  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.vllm_config import attention_moe_vllm_config  # noqa: E402

HIDDEN_SIZE = 8
HASH_LAYER, GATED_LAYER = 0, 1


class _FakeAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


def _attention_layer(monkeypatch, layer_idx: int):
    """An Attention-role decoder layer around vLLM's real ``DeepseekV4MoE``."""
    hf_config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        vocab_size=32,
        hidden_act="silu",
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        swiglu_limit=10.0,
        norm_topk_prob=True,
        num_hash_layers=1,
        rms_norm_eps=1e-6,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
    )
    config = attention_moe_vllm_config(hf_config=hf_config)
    native = adapter.native
    monkeypatch.setattr(remote, "current_platform", SimpleNamespace(device_type="cuda"))
    # The native block builds its gate and shared experts before AFD drops them.
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        native, "get_ep_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(native, "_select_dsv4_attn_cls", lambda _config: _FakeAttention)
    monkeypatch.setattr(adapter, "afd_v4_attention_cls", lambda attn_cls: attn_cls)
    with set_current_vllm_config(config):
        return adapter.AFDDeepseekV4DecoderLayer(
            config, prefix=f"model.layers.{layer_idx}"
        )


@pytest.mark.parametrize("layer_idx", [HASH_LAYER, GATED_LAYER])
def test_v4_attention_moe_block_is_native_with_remote_experts(monkeypatch, layer_idx):
    """Why: Attention runs vLLM's own MoE block and holds none of its weights."""
    native_factory = adapter.native.FusedMoEFactory

    layer = _attention_layer(monkeypatch, layer_idx)

    assert type(layer.ffn) is adapter.native.DeepseekV4MoE
    assert isinstance(layer.ffn.experts, remote.AFDRemoteMoERunner)
    assert layer.ffn.experts.layer_idx == layer_idx
    assert list(layer.ffn.parameters()) == []
    assert list(layer.ffn.buffers()) == []
    assert adapter.native.FusedMoEFactory is native_factory


def test_v4_attention_restores_factory_after_failed_construction(monkeypatch):
    native_factory = adapter.native.FusedMoEFactory

    def fail(*args, **kwargs):
        raise RuntimeError("no experts")

    monkeypatch.setattr(adapter, "build_expert", fail)
    with pytest.raises(RuntimeError, match="no experts"):
        _attention_layer(monkeypatch, HASH_LAYER)

    assert adapter.native.FusedMoEFactory is native_factory


@pytest.mark.parametrize("layer_idx", [HASH_LAYER, GATED_LAYER])
def test_v4_attention_moe_block_sends_token_ids(monkeypatch, layer_idx):
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    layer = _attention_layer(monkeypatch, layer_idx)
    hidden_states = torch.full((2, HIDDEN_SIZE), 8.0)
    input_ids = torch.tensor([11, 13], dtype=torch.int32)

    # vLLM's decoder layer calls its MoE as ffn(x, input_ids, mega_gate_metadata).
    output = layer.ffn(hidden_states, input_ids, None)

    assert adapter.AFDDeepseekV4ForCausalLM.afd_requires_input_ids
    assert [event[0] for event in events] == ["send", "yield", "recv"]
    assert events[0][2].metadata.layer_idx == layer_idx
    assert set(events[0][3]) == {"input_ids"}
    assert events[0][3]["input_ids"] is input_ids
    assert torch.equal(output, hidden_states * 0.25)


def test_v4_ffn_compute_rejects_attention_role_before_input_ids(monkeypatch):
    layer = _attention_layer(monkeypatch, HASH_LAYER)

    with pytest.raises(RuntimeError, match="FFN compute is FFN-role only"):
        layer.compute_ffn_output(torch.ones((2, HIDDEN_SIZE)), input_ids=None)


def test_v4_ffn_compute_requires_input_ids(monkeypatch):
    class FakeMoE(nn.Module):
        pass

    monkeypatch.setattr(adapter.native, "DeepseekV4MoE", FakeMoE)
    layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.attn = adapter.native.PPMissingLayer()
    layer.ffn = FakeMoE()

    with pytest.raises(RuntimeError, match="FFN compute requires input_ids"):
        layer.compute_ffn_output(torch.ones((2, 4)), input_ids=None)
