# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = pytest.importorskip("torch")
    nn = torch.nn

pytest.importorskip("vllm")

from vllm.config import CompilationMode, set_current_vllm_config  # noqa: E402

from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.ffn import remote  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2 as adapter  # noqa: E402
from tests.unit.fakes.connector import install_proxy_context  # noqa: E402
from tests.unit.fakes.model_layers import GateLinear
from tests.unit.fakes.model_layers import Stage as _FakeStage  # noqa: E402
from tests.unit.fakes.model_layers import stage_type as _stage_type
from tests.unit.fakes.vllm_config import attention_moe_vllm_config  # noqa: E402


@pytest.fixture
def construction_env(monkeypatch):
    calls: dict[str, list[str]] = {
        "attention": [],
        "dense": [],
        "gate": [],
        "moe": [],
        "norm": [],
        "stage": [],
    }

    def bind(stage_type):
        # Native shared-expert layer discovery uses isinstance().
        class _BoundStage(stage_type):  # type: ignore[valid-type,misc]
            def __init__(self, *args, **kwargs):
                super().__init__(calls, *args, **kwargs)

        return _BoundStage

    attention_type = _stage_type("attention")
    dense_type = _stage_type("dense")
    moe_type = _stage_type("moe")
    gate_type = _stage_type("gate")
    norm_type = _stage_type("norm")

    monkeypatch.setattr(adapter.native, "DeepseekAttention", bind(attention_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2Attention", bind(attention_type))
    monkeypatch.setattr(
        adapter.native,
        "DeepseekV2MLAAttention",
        bind(attention_type),
    )
    monkeypatch.setattr(adapter.native, "DeepseekV2MLP", bind(dense_type))
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", bind(moe_type))
    monkeypatch.setattr(adapter.native, "GateLinear", bind(gate_type))
    monkeypatch.setattr(adapter.native, "RMSNorm", bind(norm_type))
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    )
    return calls


def _vllm_config(*, layer_count: int = 2):
    config = SimpleNamespace(
        first_k_dense_replace=1,
        hidden_act="silu",
        hidden_size=8,
        intermediate_size=16,
        model_type="deepseek",
        moe_intermediate_size=8,
        moe_layer_freq=1,
        n_group=1,
        n_routed_experts=4,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts_per_tok=2,
        num_hidden_layers=layer_count,
        q_lora_rank=None,
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        rms_norm_eps=1e-6,
        routed_scaling_factor=1.0,
        topk_method="noaux_tc",
        v_head_dim=0,
        vocab_size=32,
    )
    return SimpleNamespace(
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(hf_config=config, use_mla=False),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=4),
        speculative_config=None,
    )


def _make_layer(
    monkeypatch,
    *,
    role: str,
    layer_idx: int,
    attention_gate: bool = False,
    vllm_config=None,
    connector: str = "P2pNcclAFDConnector",
):
    afd_config = AFDConfig(
        role=role,
        connector=connector,
        compute_gate_on_attention=attention_gate,
    )
    # The model shell and the native-factory runner read the same placement.
    for module in (adapter, remote):
        monkeypatch.setattr(
            module,
            "parse_afd_config",
            lambda *_args, **_kwargs: afd_config,
        )
    monkeypatch.setattr(remote, "current_platform", adapter.native.current_platform)
    if vllm_config is None:
        vllm_config = _vllm_config()
    return adapter.AFDDeepseekV2DecoderLayer(
        vllm_config,
        f"model.layers.{layer_idx}",
    )


def _single_ep_rank():
    return SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0
    )


@pytest.fixture
def cuda_construction_env(monkeypatch, construction_env):
    """A real vLLM config and CUDA placement for native-factory MoE runners."""
    config = attention_moe_vllm_config(
        hf_config=_vllm_config().model_config.hf_config,
        use_mla=False,
    )
    monkeypatch.setattr(
        adapter.native, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(adapter.native, "get_ep_group", _single_ep_rank)
    with set_current_vllm_config(config):
        yield config


def _parameter_names(module: nn.Module) -> set[str]:
    return {name for name, _ in module.named_parameters()}


def test_standard_attention_constructs_no_ffn_parameters(
    monkeypatch,
    construction_env,
):
    dense = _make_layer(
        monkeypatch, role="attention", layer_idx=0, connector="CAMP2pAFDConnector"
    )
    moe = _make_layer(
        monkeypatch, role="attention", layer_idx=1, connector="CAMP2pAFDConnector"
    )

    assert isinstance(dense.mlp, adapter.RemoteFFNProxy)
    assert isinstance(moe.mlp, adapter.RemoteFFNProxy)
    assert not any(name.startswith("mlp.") for name in _parameter_names(dense))
    assert not any(name.startswith("mlp.") for name in _parameter_names(moe))


def test_ffn_layer_inventory_excludes_dense_and_pp_placeholders(
    monkeypatch, cuda_construction_env
):
    model = object.__new__(adapter.AFDDeepseekV2Model)
    nn.Module.__init__(model)
    dense = _make_layer(monkeypatch, role="ffn", layer_idx=0)
    moe = _make_layer(monkeypatch, role="ffn", layer_idx=2)
    model.layers = nn.ModuleList([dense, nn.Identity(), moe])
    assert model.get_experts_layer_indices() == (2,)


def test_cuda_attention_gate_uses_native_gate_contract(
    monkeypatch,
    cuda_construction_env,
):
    """With the gate on Attention, the MoE builds the gate-applying native
    factory runner, and its only parameters are the native gate's, at their
    checkpoint names.

    Why: the gate's weights load from ``mlp.gate.*``; the experts must
    allocate no local weights.
    """
    config = cuda_construction_env
    config.model_config.hf_config.moe_router_dtype = "float32"
    monkeypatch.setattr(adapter.native, "GateLinear", GateLinear)

    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=True,
        vllm_config=config,
    )

    assert isinstance(moe.mlp, adapter.AFDDeepseekV2RemoteExpertsMoE)
    assert moe.mlp.gate.weight.dtype == torch.float32
    assert {name for name in _parameter_names(moe) if name.startswith("mlp.")} == {
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
    }
    assert list(moe.mlp.experts.buffers()) == []


@pytest.mark.parametrize(
    ("connector", "attention_gate", "message"),
    [
        ("P2pNcclAFDConnector", False, "Attention-local EPLB"),
        ("P2pNcclAFDConnector", True, "Attention-local EPLB"),
        (DEEPEP_M2N_CONNECTOR, True, "Attention-local EPLB"),
    ],
    ids=["p2p-ffn-gate", "p2p-attention-gate", "m2n"],
)
def test_cuda_remote_experts_reject_eplb_on_attention(
    monkeypatch,
    cuda_construction_env,
    connector,
    attention_gate,
    message,
):
    """Attention-side remote experts reject EPLB, with either transport.

    Why: the Attention role owns no expert weights to balance.
    """
    config = cuda_construction_env
    config.parallel_config.enable_eplb = True

    with pytest.raises(RuntimeError, match=message):
        _make_layer(
            monkeypatch,
            role="attention",
            layer_idx=1,
            attention_gate=attention_gate,
            vllm_config=config,
            connector=connector,
        )

    assert config.compilation_config.static_forward_context == {}


def test_cuda_ffn_gate_shell_owns_no_mlp_parameters(
    monkeypatch,
    cuda_construction_env,
):
    """With the gate on FFN, the Attention MoE builds no gate and no weights.

    Why: the FFN role owns the gate and all experts; the native factory
    runner only sends hidden states.
    """
    monkeypatch.setattr(
        adapter.native,
        "GateLinear",
        lambda *_args, **_kwargs: pytest.fail(
            "FFN-side gate must not construct an Attention gate",
        ),
    )

    moe = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=False,
        vllm_config=cuda_construction_env,
    )

    assert isinstance(moe.mlp, adapter.AFDDeepseekV2RemoteExpertsMoE)
    assert moe.mlp.gate is None
    assert not any(name.startswith("mlp.") for name in _parameter_names(moe))
    assert list(moe.mlp.buffers()) == []


def test_deepep_m2n_attention_owns_gate_and_shared_experts(
    monkeypatch,
    cuda_construction_env,
):
    """Under DeepEP M2N, Attention builds the gate and the native shared MLP at
    their checkpoint paths and no routed experts.

    Why: the DeepEP M2N FFN runs routed experts only, so the shared experts'
    output must come from Attention.
    """
    monkeypatch.setattr(adapter.native, "GateLinear", GateLinear)

    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=True,
        vllm_config=cuda_construction_env,
        connector=DEEPEP_M2N_CONNECTOR,
    )

    assert {name for name in _parameter_names(layer) if name.startswith("mlp.")} == {
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.shared_experts.weight",
    }
    assert list(layer.mlp.buffers()) == []


@pytest.mark.parametrize("tp_size", [1, 2])
def test_deepep_m2n_attention_tp_ranks_send_their_own_token_slice(
    monkeypatch,
    cuda_construction_env,
    tp_size,
):
    """With Attention TP > 1 the DeepEP M2N MoE is sequence-parallel, and its
    shared MLP is built sequence-parallel (replicated weights, no all-reduce);
    with TP 1 both are plain native.

    Why: TP peers hold the same tokens, so each must send a disjoint slice;
    a TP-sharded shared MLP would all-reduce partial sums of different tokens.
    """
    shared_kwargs = []

    class _RecordingMLP(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            shared_kwargs.append(kwargs)

    monkeypatch.setattr(adapter.native, "DeepseekV2MLP", _RecordingMLP)
    monkeypatch.setattr(adapter.native, "GateLinear", GateLinear)
    cuda_construction_env.parallel_config.tensor_parallel_size = tp_size

    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=1,
        attention_gate=True,
        vllm_config=cuda_construction_env,
        connector=DEEPEP_M2N_CONNECTOR,
    )

    assert layer.mlp.is_sequence_parallel is (tp_size > 1)
    (kwargs,) = shared_kwargs
    assert kwargs["prefix"] == "model.layers.1.mlp.shared_experts"
    assert kwargs["is_sequence_parallel"] is (tp_size > 1)


def test_ffn_constructs_no_real_attention(
    monkeypatch,
    construction_env,
):
    dense = _make_layer(monkeypatch, role="ffn", layer_idx=0)
    moe = _make_layer(monkeypatch, role="ffn", layer_idx=1)

    assert isinstance(dense.self_attn, adapter.native.PPMissingLayer)
    assert isinstance(moe.self_attn, adapter.native.PPMissingLayer)
    assert not any(name.startswith("self_attn.") for name in _parameter_names(dense))
    assert not any(name.startswith("self_attn.") for name in _parameter_names(moe))


def _remote_experts_shell(monkeypatch, *, compute_gate_on_attention: bool):
    """A CUDA P2P Attention MoE shell with its native factory runner."""
    monkeypatch.setattr(remote, "current_platform", SimpleNamespace(device_type="cuda"))
    monkeypatch.setattr(adapter.native, "get_ep_group", _single_ep_rank)
    monkeypatch.setattr(adapter.native, "GateLinear", GateLinear)
    config = attention_moe_vllm_config(
        compute_gate_on_attention=compute_gate_on_attention
    )
    with set_current_vllm_config(config):
        return adapter.AFDDeepseekV2RemoteExpertsMoE(
            config=_vllm_config().model_config.hf_config,
            vllm_config=config,
            prefix="model.layers.1.mlp",
            compute_gate_on_attention=compute_gate_on_attention,
            sends_routes=False,
        )


def test_attention_shell_computes_gate_before_transport(monkeypatch):
    """CUDA compute_gate_on_attention=True ships real router logits.

    The native MoE forward passes router_logits=hidden_states and lets the
    experts runner's gate route. The Attention runner holds the shell's gate,
    so it computes the logits before transport; shipping hidden states would
    break the P2P width contract (hidden_size vs n_routed_experts) on the FFN.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events, stage_idx=0)
    shell = _remote_experts_shell(monkeypatch, compute_gate_on_attention=True)
    shell.gate.weight.data.fill_(1)
    hidden_states = torch.ones((2, 8), dtype=torch.bfloat16)

    output = shell(hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    assert torch.equal(
        events[0][3]["router_logits"], torch.full((2, 4), 8.0, dtype=torch.bfloat16)
    )
    assert torch.equal(output, hidden_states * 0.25)


def test_attention_shell_without_gate_keeps_native_delegation(monkeypatch):
    """With the gate on the FFN side, the native forward ships hidden states
    only.

    Why: the FFN role runs its own native gate on the received rows.
    """
    events: list[tuple] = []
    install_proxy_context(monkeypatch, events)
    shell = _remote_experts_shell(monkeypatch, compute_gate_on_attention=False)
    hidden_states = torch.ones((2, 8))

    output = shell(hidden_states)

    assert [event[0] for event in events] == ["send", "yield", "recv"]
    assert torch.equal(events[0][1], hidden_states)
    assert events[0][3] == {}
    assert torch.equal(output, hidden_states * 0.25)


def test_npu_ffn_refreshes_native_fused_moe_factory(
    monkeypatch,
    construction_env,
):
    stale_factory = object()
    ascend_factory = object()
    factories_seen_by_native_moe = []

    class _FakeMoE(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            factories_seen_by_native_moe.append(adapter.native.FusedMoEFactory)

    monkeypatch.setattr(adapter.native, "FusedMoEFactory", stale_factory)
    monkeypatch.setattr(adapter.fused_moe, "FusedMoEFactory", ascend_factory)
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", _FakeMoE)

    _make_layer(monkeypatch, role="ffn", layer_idx=1)

    assert factories_seen_by_native_moe == [ascend_factory]


@pytest.mark.parametrize(
    ("aiter_enabled", "apply_routed_scale_to_output"),
    [(False, True)],
)
def test_ffn_moe_preserves_native_routed_scale_placement(
    monkeypatch,
    construction_env,
    aiter_enabled,
    apply_routed_scale_to_output,
):
    moe_calls = []

    class _FakeMoE(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            moe_calls.append(kwargs)
            self.experts = SimpleNamespace(gate=object())

    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", _FakeMoE)
    monkeypatch.setattr(
        adapter.native.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: aiter_enabled,
    )

    moe = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=1,
        attention_gate=True,
    )

    assert moe_calls[0]["apply_routed_scale_to_output"] is apply_routed_scale_to_output
    assert moe.mlp.experts.gate is None


def test_ffn_side_gate_keeps_native_internal_router(
    monkeypatch,
    construction_env,
):
    native_gate = object()

    class _FakeMoE(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.experts = SimpleNamespace(gate=native_gate)

    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", _FakeMoE)
    monkeypatch.setattr(
        adapter.native.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: False,
    )

    moe = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=1,
        attention_gate=False,
    )

    assert moe.mlp.experts.gate is native_gate


def test_attention_gate_ffn_dense_uses_non_executing_placeholder(
    monkeypatch,
    cuda_construction_env,
):
    dense = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=0,
        attention_gate=True,
    )
    assert isinstance(dense.mlp, adapter.native.PPMissingLayer)


def _patch_model_constructor_dependencies(
    monkeypatch,
    construction_env,
):
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        adapter.native,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: _FakeStage(construction_env, *args, **kwargs),
    )

    def make_layers(count, factory, *, prefix):
        layers = nn.ModuleList(
            factory(f"{prefix}.{layer_idx}") for layer_idx in range(count)
        )
        return 0, count, layers

    monkeypatch.setattr(adapter.native, "make_layers", make_layers)
    monkeypatch.setattr(
        adapter.native,
        "make_empty_intermediate_tensors_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        adapter.native.DeepseekV2Model,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("native constructor was called"),
    )


def test_model_constructor_uses_role_aware_layers(
    monkeypatch,
    construction_env,
):
    vllm_config = _vllm_config()
    afd_config = AFDConfig(role="attention")
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    _patch_model_constructor_dependencies(monkeypatch, construction_env)

    model = adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")

    assert isinstance(model, adapter.native.DeepseekV2Model)
    assert all(
        isinstance(layer, adapter.AFDDeepseekV2DecoderLayer) for layer in model.layers
    )
    assert model.hidden_size == vllm_config.model_config.hf_config.hidden_size
    assert model.use_mha
    assert model.num_redundant_experts == 0


@pytest.mark.parametrize("fused", [False, True])
def test_npu_role_local_model_exposes_shared_expert_layers_to_native_loader(
    monkeypatch,
    construction_env,
    fused,
):
    def load_weights(model, weights):
        assert model.fused_shared_expert_layers == (
            frozenset({1}) if fused else frozenset()
        )
        return {name for name, _ in weights}

    monkeypatch.setattr(adapter.native.DeepseekV2Model, "load_weights", load_weights)
    monkeypatch.setattr(
        adapter, "parse_afd_config", lambda *_args, **_kwargs: AFDConfig(role="ffn")
    )
    monkeypatch.setattr(
        adapter.native.DeepseekV2MoE,
        "is_fused_shared_expert_enabled",
        fused,
        raising=False,
    )
    _patch_model_constructor_dependencies(monkeypatch, construction_env)
    model = adapter.AFDDeepseekV2Model(vllm_config=_vllm_config(), prefix="model")
    weight_name = "layers.1.mlp.shared_experts.down_proj.weight"

    assert model.load_weights([(weight_name, torch.empty(1))]) == {weight_name}


def test_afd_alias_without_activation_fails_before_construction(monkeypatch):
    vllm_config = SimpleNamespace(
        additional_config={},
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")
    with pytest.raises(ValueError, match="requires additional_config"):
        adapter.AFDDeepseekV2DecoderLayer(vllm_config, "model.layers.0")


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_model_constructor_rejects_sequence_parallel_moe_before_allocation(
    monkeypatch,
    construction_env,
    role,
):
    vllm_config = _vllm_config()
    vllm_config.parallel_config.use_sequence_parallel_moe = True
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role=role, connector="CAMP2pAFDConnector"),
    )

    with pytest.raises(RuntimeError, match="sequence-parallel MoE"):
        adapter.AFDDeepseekV2Model(vllm_config=vllm_config, prefix="model")


def test_m2n_ffn_omits_attention_owned_shared_with_real_hf_config(
    monkeypatch, cuda_construction_env
):
    from transformers import DeepseekV2Config

    original = DeepseekV2Config(
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
    )
    config = _vllm_config()
    config.model_config.hf_config = original
    native_configs = []

    class RoutedMoE(nn.Module):
        def __init__(self, *, config, **kwargs):
            super().__init__()
            native_configs.append(config)
            self.experts = SimpleNamespace(gate=None)

    monkeypatch.setattr(adapter.native, "DeepseekV2MoE", RoutedMoE)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(
            role="ffn",
            connector=DEEPEP_M2N_CONNECTOR,
            compute_gate_on_attention=True,
        ),
    )
    adapter.AFDDeepseekV2DecoderLayer(config, "model.layers.3")
    assert native_configs[0] is not original
    assert native_configs[0].n_shared_experts is None
    assert original.n_shared_experts == 1
