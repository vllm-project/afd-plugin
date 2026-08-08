# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
nn = torch.nn

from vllm.config import CompilationMode  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import qwen3_moe as adapter  # noqa: E402


class _FakeStage(nn.Module):
    kind = "stage"

    def __init__(self, calls: dict[str, list[str]], *args, prefix="", **kwargs):
        super().__init__()
        calls[self.kind].append(prefix)
        self.weight = nn.Parameter(torch.empty(1))


def _stage_type(kind: str):
    return type(f"Fake{kind.title()}", (_FakeStage,), {"kind": kind})


class _FakeMissingLayer(nn.Module):
    pass


@pytest.fixture
def construction_env(monkeypatch: pytest.MonkeyPatch):
    calls = {
        "attention": [],
        "dense": [],
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
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        speculative_config=None,
    )


def _make_layer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str,
    layer_idx: int,
    vllm_config=None,
):
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role=role),
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
    assert isinstance(layer.mlp, adapter.RemoteFFNProxy)
    assert layer.mlp.layer_idx == 0
    assert set(dict(layer.named_parameters())) == {
        "self_attn.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }


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
        SimpleNamespace(device_type="cuda"),
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


def test_adapter_signatures_match_native_contract() -> None:
    assert inspect.signature(adapter.AFDQwen3MoeDecoderLayer.__init__) == (
        inspect.signature(adapter.native.Qwen3MoeDecoderLayer.__init__)
    )
    assert inspect.signature(adapter.AFDQwen3MoeModel.__init__) == inspect.signature(
        adapter.native.Qwen3MoeModel.__init__
    )
    assert inspect.signature(adapter.AFDQwen3MoeForCausalLM.__init__) == (
        inspect.signature(adapter.native.Qwen3MoeForCausalLM.__init__)
    )
    assert inspect.signature(adapter.AFDQwen3MoeForCausalLM.load_weights) == (
        inspect.signature(adapter.native.Qwen3MoeForCausalLM.load_weights)
    )


@pytest.mark.parametrize(
    ("device_type", "afd_config", "parallel_overrides", "message"),
    [
        ("cpu", AFDConfig(), {}, "CUDA only"),
        (
            "cuda",
            AFDConfig(compute_gate_on_attention=True),
            {},
            "compute_gate_on_attention=false",
        ),
        (
            "cuda",
            AFDConfig(),
            {"use_sequence_parallel_moe": True},
            "sequence-parallel MoE",
        ),
        ("cuda", AFDConfig(), {"enable_eplb": True}, "EPLB"),
        (
            "cuda",
            AFDConfig(),
            {"pipeline_parallel_size": 2},
            "pipeline parallelism",
        ),
    ],
)
def test_unsupported_modes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
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
        SimpleNamespace(device_type=device_type),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, afd_config)


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
        SimpleNamespace(device_type="cuda"),
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
