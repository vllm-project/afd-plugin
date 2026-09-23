# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contract tests for the Kimi K3 AFD adapter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm.models.kimi_k3")
import torch.nn as nn  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig  # noqa: E402

from afd_plugin.model_executor.models import kimi_k3 as adapter  # noqa: E402
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDAttentionFusedMoE,
)


def _kimi_k3_text_config(
    *,
    num_hidden_layers: int = 6,
    attn_res_block_size: int | None = 3,
) -> KimiLinearConfig:
    """Small latent-MoE config shaped like Kimi-K3's text config."""
    return KimiLinearConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=2,
        first_k_dense_replace=1,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        mla_use_output_gate=True,
        attn_res_block_size=attn_res_block_size,
        latent_moe_use_norm=True,
        routed_expert_hidden_size=32,
        linear_attn_config={
            "kda_layers": [0, 2, 4],
            "full_attn_layers": [1, 3, 5],
            "use_full_rank_gate": True,
            "num_heads": 2,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
        },
    )


def _afd_vllm_config(
    config: KimiLinearConfig,
    *,
    compute_gate_on_attention: bool = True,
    role: str = "attention",
    language_model_only: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(),
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            enable_expert_parallel=False,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        speculative_config=None,
        lora_config=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=config,
                vision_config=SimpleNamespace(num_attention_heads=2),
                media_placeholder_token_id=1,
            ),
            hf_text_config=config,
            multimodal_config=SimpleNamespace(
                language_model_only=language_model_only,
            ),
        ),
        _afd_role=role,
        _afd_compute_gate_on_attention=compute_gate_on_attention,
    )


@pytest.fixture(autouse=True)
def kimi_current_vllm_config():
    """Kimi K3 modules build vLLM CustomOps (RMSNorm) at construction time."""
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture(autouse=True)
def kimi_afd_config(monkeypatch: pytest.MonkeyPatch):
    def fake_parse(vllm_config, validate=False):  # noqa: ARG001
        return SimpleNamespace(
            role=getattr(vllm_config, "_afd_role", "attention"),
            compute_gate_on_attention=getattr(
                vllm_config, "_afd_compute_gate_on_attention", True
            ),
        )

    monkeypatch.setattr(adapter, "parse_optional_afd_config", fake_parse)


class _FakeStage(nn.Module):
    kind = "stage"

    def __init__(self, *args, prefix="", **kwargs):
        super().__init__()
        self.prefix = prefix
        self.weight = nn.Parameter(torch.ones(2, 2))

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class _FakeGate(_FakeStage):
    kind = "gate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dtype = torch.float32


class _FakeAttention(_FakeStage):
    kind = "attention"


class _FakeFusedMoE(_FakeStage):
    kind = "experts_factory"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.factory_kwargs = kwargs


@pytest.fixture
def construction_env(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, list[str]] = {
        kind: [] for kind in ("stage", "gate", "attention", "experts_factory")
    }

    def bind(fake_cls):
        def construct(*args, **kwargs):
            instance = fake_cls(*args, **kwargs)
            calls[fake_cls.kind].append(kwargs.get("prefix", ""))
            return instance

        return construct

    for name, fake_cls in (
        ("KimiK3DeltaAttention", _FakeAttention),
        ("KimiLinearGatedDeltaNetAttention", _FakeAttention),
        ("MultiHeadLatentAttention", _FakeAttention),
        ("KimiMLP", _FakeStage),
        ("GateLinear", _FakeGate),
        ("ReplicatedLinear", _FakeStage),
        ("FusedMoEFactory", _FakeFusedMoE),
    ):
        monkeypatch.setattr(adapter.native, name, bind(fake_cls))
    monkeypatch.setattr(adapter.native, "aux_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )
    return calls


def test_kimi_adapter_keeps_native_signatures_and_forward_methods():
    assert inspect.signature(adapter.AFDKimiK3DecoderLayer.__init__) == (
        inspect.signature(adapter.native.KimiDecoderLayer.__init__)
    )
    assert inspect.signature(adapter.AFDKimiLinearModel.__init__) == (
        inspect.signature(adapter.native.KimiLinearModel.__init__)
    )
    assert inspect.signature(adapter.AFDKimiLinearForCausalLM.__init__) == (
        inspect.signature(adapter.native.KimiLinearForCausalLM.__init__)
    )
    assert inspect.signature(
        adapter.AFDKimiK3ForConditionalGeneration.__init__
    ) == inspect.signature(adapter.native.KimiK3ForConditionalGeneration.__init__)
    assert inspect.signature(
        adapter.AFDKimiK3AttentionMoE.forward
    ) == inspect.signature(adapter.native.KimiMoE.forward)
    assert (
        adapter.AFDKimiK3DecoderLayer.forward is adapter.native.KimiDecoderLayer.forward
    )
    assert adapter.AFDKimiLinearModel.forward is adapter.native.KimiLinearModel.forward
    assert (
        adapter.AFDKimiLinearForCausalLM.forward
        is adapter.native.KimiLinearForCausalLM.forward
    )
    assert (
        adapter.AFDKimiK3ForConditionalGeneration.forward
        is adapter.native.KimiK3ForConditionalGeneration.forward
    )


def test_attention_moe_forward_applies_native_latent_tail(monkeypatch):
    hidden_size, latent_size, num_experts = 8, 4, 6
    torch.manual_seed(0)
    hidden_states = torch.randn(3, hidden_size)

    class FakeLinear:
        def __init__(self, out_features, in_features):
            self.weight = nn.Parameter(torch.randn(out_features, in_features))

        def __call__(self, x):
            return x @ self.weight.t(), None

    class FakeShared(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))

        def forward(self, x):
            return x @ self.weight.t()

    attention_moe = object.__new__(adapter.AFDKimiK3AttentionMoE)
    nn.Module.__init__(attention_moe)
    attention_moe.gate = FakeLinear(num_experts, hidden_size)
    attention_moe.routed_expert_down_proj = FakeLinear(latent_size, hidden_size)
    attention_moe.shared_experts = FakeShared()
    attention_moe.routed_expert_norm = adapter.native.RMSNorm(latent_size, eps=1e-6)
    attention_moe.routed_expert_up_proj = FakeLinear(hidden_size, latent_size)
    native_transform = adapter.native.KimiRoutedOutputTransform(
        attention_moe.routed_expert_norm,
        SimpleNamespace(weight=attention_moe.routed_expert_up_proj.weight),
    )
    attention_moe.routed_output_transform = native_transform
    attention_moe.use_mega_moe = False
    attention_moe._down_proj_stream = None
    attention_moe._down_proj_events = (object(), object())
    monkeypatch.setattr(
        adapter.native,
        "maybe_execute_in_parallel",
        lambda first, second, *_args, **_kwargs: (first(), second()),
    )

    class FakeRemoteExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []

        def forward(self, hidden_states, router_logits):
            self.received.append((hidden_states, router_logits))
            return hidden_states * 2

    attention_moe.experts = FakeRemoteExperts()

    output = attention_moe(hidden_states)

    routed_latent, router_logits = attention_moe.experts.received[0]
    assert routed_latent.shape == (3, latent_size)
    assert router_logits.shape == (3, num_experts)
    # Native latent-MoE tail: up(norm(remote latent)) + shared output.
    reference = native_transform(
        routed_latent * 2,
        residual=attention_moe.shared_experts(hidden_states),
    )
    assert torch.allclose(output, reference)


def test_ffn_compute_experts_output_uses_external_routing():
    calls = []

    class FakeExperts(nn.Module):
        def forward(self, *, hidden_states, router_logits):
            calls.append((hidden_states, router_logits))
            return hidden_states

    ffn_moe = object.__new__(adapter.AFDKimiK3FFNMoE)
    nn.Module.__init__(ffn_moe)
    ffn_moe.experts = FakeExperts()

    layer = object.__new__(adapter.AFDKimiK3DecoderLayer)
    nn.Module.__init__(layer)
    layer.is_moe_layer = True
    layer.block_sparse_moe = ffn_moe
    hidden_states = torch.zeros(2, 4)
    router_logits = torch.zeros(2, 6)

    output = layer.compute_experts_output(hidden_states, router_logits)

    assert calls == [(hidden_states, router_logits)]
    assert output is hidden_states


def test_ffn_compute_experts_output_rejects_attention_role():
    layer = object.__new__(adapter.AFDKimiK3DecoderLayer)
    nn.Module.__init__(layer)
    layer.is_moe_layer = True
    layer.block_sparse_moe = object.__new__(adapter.AFDKimiK3AttentionMoE)

    with pytest.raises(RuntimeError, match="native Kimi experts"):
        layer.compute_experts_output(torch.zeros(1, 4), torch.zeros(1, 6))


def test_attention_layer_constructs_attention_owned_stages(construction_env):
    config = _kimi_k3_text_config()
    vllm_config = _afd_vllm_config(config, role="attention")

    layer = adapter.AFDKimiK3DecoderLayer(config, vllm_config, prefix="layers.1")

    assert isinstance(layer.mlp, adapter.AFDKimiK3AttentionMoE)
    assert isinstance(layer.block_sparse_moe.experts, AFDAttentionFusedMoE)
    assert layer.block_sparse_moe.experts.is_internal_router is False
    assert list(layer.block_sparse_moe.experts.parameters()) == []
    # Attention never constructs the routed experts.
    assert construction_env["experts_factory"] == []
    assert construction_env["gate"] == ["layers.1.block_sparse_moe.gate"]
    assert construction_env["stage"] == [
        "layers.1.block_sparse_moe.shared_experts",
        "layers.1.block_sparse_moe.routed_expert_down_proj",
        "layers.1.block_sparse_moe.routed_expert_up_proj",
        "layers.1.self_attention_res_proj",
        "layers.1.mlp_res_proj",
    ]
    assert construction_env["attention"] == ["layers.1.self_attn"]
    parameter_names = {name for name, _ in layer.named_parameters()}
    assert "block_sparse_moe.gate.e_score_correction_bias" in parameter_names
    assert "block_sparse_moe.routed_expert_norm.weight" in parameter_names
    assert not any(".experts." in name for name in parameter_names)


def test_ffn_layer_constructs_native_latent_experts_only(construction_env):
    config = _kimi_k3_text_config()
    vllm_config = _afd_vllm_config(config, role="ffn")

    layer = adapter.AFDKimiK3DecoderLayer(config, vllm_config, prefix="layers.1")

    assert isinstance(layer.mlp, adapter.AFDKimiK3FFNMoE)
    assert construction_env["experts_factory"] == ["layers.1.block_sparse_moe.experts"]
    factory_kwargs = layer.block_sparse_moe.experts.factory_kwargs
    assert factory_kwargs["hidden_size"] == config.routed_expert_hidden_size
    assert factory_kwargs["shared_experts"] is None
    assert factory_kwargs["routed_output_transform"] is None
    assert factory_kwargs["routed_input_transform"] is None
    assert factory_kwargs["e_score_correction_bias"] is (
        layer.block_sparse_moe.gate.e_score_correction_bias
    )
    assert layer.block_sparse_moe.shared_experts is None
    assert layer.block_sparse_moe.routed_output_transform is None
    assert construction_env["attention"] == []
    # Gate weights load on FFN for the routing correction bias only.
    assert construction_env["gate"] == ["layers.1.block_sparse_moe.gate"]
    assert isinstance(layer.input_layernorm, adapter.native.PPMissingLayer)


def test_dense_layer_runs_locally_per_role(construction_env):
    config = _kimi_k3_text_config()
    attention_layer = adapter.AFDKimiK3DecoderLayer(
        config, _afd_vllm_config(config, role="attention"), prefix="layers.0"
    )
    ffn_layer = adapter.AFDKimiK3DecoderLayer(
        config, _afd_vllm_config(config, role="ffn"), prefix="layers.0"
    )

    assert not attention_layer.is_moe_layer
    assert attention_layer.mlp.prefix == "layers.0.mlp"
    assert isinstance(ffn_layer.mlp, adapter.native.PPMissingLayer)
    assert construction_env["experts_factory"] == []


def test_layer_requires_attention_side_gate(construction_env):
    config = _kimi_k3_text_config()
    vllm_config = _afd_vllm_config(config, compute_gate_on_attention=False)

    with pytest.raises(ValueError, match="compute_gate_on_attention=true only"):
        adapter.AFDKimiK3DecoderLayer(config, vllm_config, prefix="layers.1")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("speculative_config", "speculative decoding"),
        ("lora_config", "LoRA"),
    ],
)
def test_afd_config_rejects_unsupported_features(field, message):
    vllm_config = _afd_vllm_config(_kimi_k3_text_config())
    setattr(vllm_config, field, SimpleNamespace())

    with pytest.raises(ValueError, match=message):
        adapter._require_kimi_k3_afd_config(vllm_config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "parallel_config",
            SimpleNamespace(
                enable_eplb=True,
                enable_expert_parallel=False,
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
                data_parallel_size=1,
            ),
            "EPLB",
        ),
        (
            "parallel_config",
            SimpleNamespace(
                enable_eplb=False,
                enable_expert_parallel=False,
                pipeline_parallel_size=2,
                tensor_parallel_size=1,
                data_parallel_size=1,
            ),
            "pipeline_parallel_size=1 only",
        ),
        (
            "parallel_config",
            SimpleNamespace(
                enable_eplb=False,
                enable_expert_parallel=True,
                pipeline_parallel_size=1,
                tensor_parallel_size=2,
                data_parallel_size=2,
            ),
            "sequence-parallel topologies",
        ),
        (
            "kernel_config",
            SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
            "deep_gemm_mega_moe",
        ),
    ],
)
def test_afd_config_rejects_unsupported_topologies(field, value, message):
    vllm_config = _afd_vllm_config(_kimi_k3_text_config())
    setattr(vllm_config, field, value)

    with pytest.raises(ValueError, match=message):
        adapter._require_kimi_k3_afd_config(vllm_config)


def test_afd_config_requires_latent_moe():
    config = _kimi_k3_text_config()
    config.routed_expert_hidden_size = None
    vllm_config = _afd_vllm_config(config)

    with pytest.raises(ValueError, match="latent MoE"):
        adapter._require_kimi_k3_afd_config(vllm_config)


def test_afd_config_requires_cuda(monkeypatch):
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: False),
    )
    vllm_config = _afd_vllm_config(_kimi_k3_text_config())

    with pytest.raises(ValueError, match="CUDA platform only"):
        adapter._require_kimi_k3_afd_config(vllm_config)


def test_conditional_model_rejects_multimodal_before_visual_construction(
    construction_env,
    monkeypatch,
):
    vllm_config = _afd_vllm_config(_kimi_k3_text_config(), language_model_only=False)
    monkeypatch.setattr(
        adapter.native,
        "MoonViT3dPretrainedModel",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )
    monkeypatch.setattr(
        adapter.native,
        "KimiK25MultiModalProjector",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )

    with pytest.raises(ValueError, match="pass --language-model-only"):
        adapter.AFDKimiK3ForConditionalGeneration(vllm_config)


def test_conditional_model_requires_attention_side_gate():
    vllm_config = _afd_vllm_config(
        _kimi_k3_text_config(),
        compute_gate_on_attention=False,
    )

    with pytest.raises(ValueError, match="compute_gate_on_attention=true only"):
        adapter.AFDKimiK3ForConditionalGeneration(vllm_config)


def test_experts_layer_indices_exclude_dense_layers():
    config = _kimi_k3_text_config()
    model = object.__new__(adapter.AFDKimiLinearModel)
    model.config = config
    model.start_layer = 0
    model.end_layer = config.num_hidden_layers

    assert model.get_experts_layer_indices() == (1, 2, 3, 4, 5)


def test_routing_spec_reports_native_gate_contract():
    class FakeGate:
        out_dtype = torch.float32
        weight = SimpleNamespace(shape=[6])

    class FakeMoE:
        gate = FakeGate()

    class FakeLayer:
        block_sparse_moe = FakeMoE()

    model = object.__new__(adapter.AFDKimiLinearModel)
    model.layers = {1: FakeLayer()}

    spec = model.get_experts_routing_spec(1)

    assert spec.router_logits_width == 6
    assert spec.router_logits_dtype == torch.float32
