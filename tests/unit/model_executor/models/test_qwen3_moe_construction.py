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
from torch import nn  # noqa: E402
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper  # noqa: E402
from vllm.config import CompilationMode  # noqa: E402

from afd_plugin.config import DEEPEP_M2N_CONNECTOR, AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import qwen3_moe as adapter  # noqa: E402
from tests.unit.fakes.model_layers import stage_type as _stage_type  # noqa: E402


class _FakeMissingLayer(nn.Module):
    pass


@pytest.fixture
def construction_env(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, list[str]] = {
        "attention": [],
        "dense": [],
        "gate": [],
        "moe": [],
        "norm": [],
    }

    def bind(stage_type):
        return lambda *args, **kwargs: stage_type(calls, *args, **kwargs)

    monkeypatch.setattr(
        adapter.native,
        "Qwen3MoeAttention",
        bind(_stage_type("attention")),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3MoeMLP",
        bind(_stage_type("dense")),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3MoeSparseMoeBlock",
        bind(_stage_type("moe")),
    )
    monkeypatch.setattr(adapter.native, "RMSNorm", bind(_stage_type("norm")))
    monkeypatch.setattr(
        adapter.native,
        "GateLinear",
        bind(_stage_type("gate")),
    )
    monkeypatch.setattr(adapter.native, "PPMissingLayer", _FakeMissingLayer)
    return calls


def _model_config(
    *,
    layer_count: int = 48,
    mlp_only_layers: list[int] | None = None,
    decoder_sparse_step: int = 1,
):
    config = SimpleNamespace(
        attention_bias=False,
        decoder_sparse_step=decoder_sparse_step,
        head_dim=8,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=4096,
        mlp_only_layers=mlp_only_layers or [],
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=layer_count,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        tie_word_embeddings=False,
        vocab_size=128,
    )
    return SimpleNamespace(
        cache_config=None,
        lora_config=None,
        model_config=SimpleNamespace(hf_text_config=config),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        speculative_config=None,
    )


def _platform(*, is_cuda: bool, device_type: str = "cuda") -> SimpleNamespace:
    return SimpleNamespace(device_type=device_type, is_cuda=lambda: is_cuda)


def _make_layer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str,
    layer_idx: int,
    vllm_config=None,
    afd_config: AFDConfig | None = None,
):
    if afd_config is None:
        afd_config = AFDConfig(role=role)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    if vllm_config is None:
        vllm_config = _model_config()
    return adapter.AFDQwen3MoeDecoderLayer(
        vllm_config,
        f"model.layers.{layer_idx}",
    )


def test_attention_role_constructs_only_attention_and_norms(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    layer = _make_layer(monkeypatch, role="attention", layer_idx=0)

    assert construction_env["attention"] == ["model.layers.0.self_attn"]
    assert construction_env["norm"] == ["", ""]
    assert construction_env["dense"] == []
    assert construction_env["moe"] == []
    assert construction_env["gate"] == []
    assert type(layer.mlp) is adapter.RemoteFFNProxy
    assert layer.mlp.layer_idx == 0
    assert set(dict(layer.named_parameters())) == {
        "self_attn.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }


def test_attention_role_routes_with_native_gate_under_deepep_m2n(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    """With DeepEP M2N, Attention sets the proxy's gate and top-k settings.

    Why: DeepEP M2N dispatches by top-k routes computed on Attention; the gate
    keeps the native ``mlp.gate`` path so the checkpoint loads unchanged.
    """
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=5,
        afd_config=AFDConfig(
            role="attention",
            connector=DEEPEP_M2N_CONNECTOR,
            compute_gate_on_attention=True,
        ),
    )

    assert construction_env["gate"] == ["model.layers.5.mlp.gate"]
    assert construction_env["moe"] == []
    assert type(layer.mlp) is adapter.RemoteFFNProxy
    assert layer.mlp.gate is not None
    assert layer.mlp.layer_idx == 5
    assert layer.mlp.top_k == 2
    assert layer.mlp.renormalize is True
    assert layer.mlp.tp_size == 1
    assert set(dict(layer.named_parameters())) == {
        "self_attn.weight",
        "mlp.gate.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }


def test_attention_role_gives_the_proxy_the_attention_tp_size(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    """Under DeepEP M2N the proxy gets the Attention TP size.

    Why: with Attention TP > 1 each TP rank must dispatch only its slice of
    the tokens; without the TP size the proxy would dispatch every token from
    every TP rank.
    """
    vllm_config = _model_config()
    vllm_config.parallel_config.tensor_parallel_size = 2
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=5,
        vllm_config=vllm_config,
        afd_config=AFDConfig(
            role="attention",
            connector=DEEPEP_M2N_CONNECTOR,
            compute_gate_on_attention=True,
        ),
    )

    assert type(layer.mlp) is adapter.RemoteFFNProxy
    assert layer.mlp.tp_size == 2


def test_ffn_role_constructs_only_native_moe(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    layer = _make_layer(monkeypatch, role="ffn", layer_idx=0)

    assert construction_env["attention"] == []
    assert construction_env["norm"] == []
    assert construction_env["dense"] == []
    assert construction_env["moe"] == ["model.layers.0.mlp"]
    assert isinstance(layer.self_attn, _FakeMissingLayer)
    assert isinstance(layer.input_layernorm, _FakeMissingLayer)
    assert set(dict(layer.named_parameters())) == {"mlp.weight"}


def test_ffn_role_preserves_native_dense_layer_schedule(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    vllm_config = _model_config(mlp_only_layers=[0])
    layer = _make_layer(
        monkeypatch,
        role="ffn",
        layer_idx=0,
        vllm_config=vllm_config,
    )

    assert not layer.is_moe_layer
    assert construction_env["dense"] == ["model.layers.0.mlp"]
    assert construction_env["moe"] == []


def test_native_decoder_forward_is_inherited_unchanged() -> None:
    assert (
        adapter.AFDQwen3MoeDecoderLayer.forward
        is adapter.native.Qwen3MoeDecoderLayer.forward
    )


def test_model_injects_role_aware_decoder_through_native_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, type[nn.Module]] = {}

    def fake_native_init(
        self,
        *,
        vllm_config,
        prefix="",
        decoder_layer_type=adapter.native.Qwen3MoeDecoderLayer,
    ) -> None:
        nn.Module.__init__(self)
        captured["decoder_layer_type"] = decoder_layer_type
        self.config = vllm_config.model_config.hf_text_config
        self.layers = nn.ModuleList()

    vllm_config = _model_config()
    vllm_config.compilation_config = SimpleNamespace(mode=CompilationMode.NONE)
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=True),
    )
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role="attention"),
    )
    monkeypatch.setattr(
        adapter.native.Qwen3MoeModel,
        "__init__",
        fake_native_init,
    )

    model = adapter.AFDQwen3MoeModel(vllm_config=vllm_config)

    assert captured["decoder_layer_type"] is adapter.AFDQwen3MoeDecoderLayer
    assert model.afd_role == "attention"


def test_model_inherits_native_compile_support_once() -> None:
    assert TorchCompileWithNoGuardsWrapper in adapter.native.Qwen3MoeModel.__bases__
    assert adapter.AFDQwen3MoeModel.__bases__ == (adapter.native.Qwen3MoeModel,)


def test_attention_causal_lm_reports_no_local_moe_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = object.__new__(adapter.AFDQwen3MoeDecoderLayer)
    nn.Module.__init__(layer)
    layer.mlp = adapter.RemoteFFNProxy(layer_idx=0)

    class FakeAFDModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList((layer,))
            self.embed_tokens = nn.Embedding(128, 16)
            self.make_empty_intermediate_tensors = lambda *args, **kwargs: None

        @staticmethod
        def get_experts_layer_indices() -> tuple[int, ...]:
            return (0,)

    class FakeLMHead(nn.Module):
        def __init__(self, vocab_size, hidden_size, **kwargs) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    vllm_config = _model_config(layer_count=1)
    native_mapping = dict(adapter.native.Qwen3MoeForCausalLM.packed_modules_mapping)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role="attention"),
    )
    monkeypatch.setattr(
        adapter,
        "AFDQwen3MoeModel",
        lambda **_kwargs: FakeAFDModel(),
    )
    monkeypatch.setattr(adapter.native, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(adapter.native, "LogitsProcessor", lambda *_args: object())

    model = adapter.AFDQwen3MoeForCausalLM(vllm_config=vllm_config)

    assert model.moe_layers == []
    assert model.num_moe_layers == 0
    assert model.num_local_physical_experts == 0
    assert model.num_logical_experts == 8
    assert model.get_experts_layer_indices() == (0,)
    assert adapter.native.Qwen3MoeForCausalLM.packed_modules_mapping == native_mapping


@pytest.mark.parametrize(
    ("is_cuda", "afd_config", "parallel_overrides", "message"),
    [
        (False, AFDConfig(), {}, "CUDA only"),
        (
            True,
            AFDConfig(compute_gate_on_attention=True),
            {},
            "compute_gate_on_attention=false",
        ),
        (
            True,
            AFDConfig(),
            {"use_sequence_parallel_moe": True},
            "sequence-parallel MoE",
        ),
        (True, AFDConfig(), {"enable_eplb": True}, "EPLB"),
        (
            True,
            AFDConfig(),
            {"pipeline_parallel_size": 2},
            "pipeline parallelism",
        ),
    ],
)
def test_unsupported_modes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    afd_config: AFDConfig,
    parallel_overrides: dict[str, int | bool],
    message: str,
) -> None:
    vllm_config = _model_config()
    for name, value in parallel_overrides.items():
        setattr(vllm_config.parallel_config, name, value)
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=is_cuda),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, afd_config)


def test_gate_on_attention_is_accepted_with_deepep_m2n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepEP M2N routes on Attention, so Qwen3 accepts the gate there.

    Why: its FFN runs only the routed experts and never the native gate.
    """
    monkeypatch.setattr(adapter, "current_platform", _platform(is_cuda=True))

    adapter._validate_supported_config(
        _model_config(),
        AFDConfig(connector="DeepEPM2NAFDConnector", compute_gate_on_attention=True),
    )


def test_attention_role_selects_routing_proxy_by_connector_only(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    """Without DeepEP M2N, Attention builds the plain proxy even if the
    gate-on-Attention flag is set.

    Why: only DeepEP M2N dispatches by Attention-side routes; for any other
    connector the flag is rejected by model validation, and the layer itself
    must not pick the routing proxy from the flag alone.
    """
    layer = _make_layer(
        monkeypatch,
        role="attention",
        layer_idx=0,
        afd_config=AFDConfig(role="attention", compute_gate_on_attention=True),
    )

    assert type(layer.mlp) is adapter.RemoteFFNProxy
    assert layer.mlp.gate is None
    assert construction_env["gate"] == []


def test_gate_on_attention_rejects_dense_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate-on-Attention requires every layer to be MoE.

    Why: the routed FFN runner serves MoE layers only, so a dense layer's MLP
    would run on neither role.
    """
    monkeypatch.setattr(adapter, "current_platform", _platform(is_cuda=True))

    with pytest.raises(RuntimeError, match="every layer to be MoE"):
        adapter._validate_supported_config(
            _model_config(layer_count=4, mlp_only_layers=[3]),
            AFDConfig(
                connector=DEEPEP_M2N_CONNECTOR,
                compute_gate_on_attention=True,
            ),
        )


def test_deepep_m2n_rejects_shared_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepEP M2N fails closed on a Qwen3 checkpoint with shared experts.

    Why: neither the Attention proxy nor the routed FFN runner computes a
    shared expert, so its output would be dropped without an error.
    """
    monkeypatch.setattr(adapter, "current_platform", _platform(is_cuda=True))
    vllm_config = _model_config()
    vllm_config.model_config.hf_text_config.shared_expert_intermediate_size = 64

    with pytest.raises(RuntimeError, match="does not support shared experts"):
        adapter._validate_supported_config(
            vllm_config,
            AFDConfig(
                connector=DEEPEP_M2N_CONNECTOR,
                compute_gate_on_attention=True,
            ),
        )


def test_rocm_platform_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=False, device_type="cuda"),
    )

    with pytest.raises(RuntimeError, match="CUDA only"):
        adapter._validate_supported_config(_model_config(), AFDConfig())


@pytest.mark.parametrize(
    ("config_name", "message"),
    [
        ("speculative_config", "speculative decoding"),
        ("lora_config", "LoRA"),
    ],
)
def test_unsupported_model_features_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    message: str,
) -> None:
    vllm_config = _model_config()
    setattr(vllm_config, config_name, SimpleNamespace())
    monkeypatch.setattr(
        adapter,
        "current_platform",
        _platform(is_cuda=True),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, AFDConfig())


def test_ffn_compute_delegates_to_native_mlp(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    layer = _make_layer(monkeypatch, role="ffn", layer_idx=0)
    expected = torch.full((2, 4), 7.0)
    layer.mlp.forward = lambda hidden_states: expected

    output = layer.compute_ffn_output(torch.zeros(2, 4))

    assert output is expected


def test_attention_role_rejects_local_ffn_compute(
    monkeypatch: pytest.MonkeyPatch,
    construction_env,
) -> None:
    layer = _make_layer(monkeypatch, role="attention", layer_idx=0)

    with pytest.raises(RuntimeError, match="requires the AFD FFN role"):
        layer.compute_ffn_output(torch.zeros(2, 4))
