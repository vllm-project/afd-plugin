from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
nn = torch.nn

from afd_plugin.model_executor.models import deepseek_v4 as adapter  # noqa: E402


class _FakeStage(nn.Module):
    def __init__(self, *args, prefix: str = "", **kwargs) -> None:
        super().__init__()
        self.prefix = prefix


class _FakeAttention(_FakeStage):
    pass


class _FakeMoE(_FakeStage):
    pass


class _FakeMissingLayer(_FakeStage):
    pass


def _vllm_config(*, layer_count: int = 2):
    config = SimpleNamespace(
        hc_eps=1e-6,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hidden_size=8,
        index_topk=4,
        num_hidden_layers=layer_count,
        rms_norm_eps=1e-6,
        vocab_size=32,
    )
    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="cutlass"),
        model_config=SimpleNamespace(hf_config=config, dtype=torch.float16),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )


@pytest.fixture
def construction_env(monkeypatch):
    monkeypatch.setattr(
        adapter.native,
        "current_platform",
        SimpleNamespace(device_type="cuda"),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        adapter.native,
        "_select_dsv4_attn_cls",
        lambda _config: _FakeAttention,
    )
    monkeypatch.setattr(adapter.native, "DeepseekV4MoE", _FakeMoE)
    monkeypatch.setattr(adapter.native, "PPMissingLayer", _FakeMissingLayer)
    monkeypatch.setattr(adapter.native, "RMSNorm", _FakeStage)
    monkeypatch.setattr(adapter.native, "VocabParallelEmbedding", _FakeStage)
    monkeypatch.setattr(adapter.torch.cuda, "Stream", lambda: object())

    def fake_make_layers(layer_count, layer_factory, *, prefix):
        layers = nn.ModuleList(
            layer_factory(f"{prefix}.{layer_idx}") for layer_idx in range(layer_count)
        )
        return 0, layer_count, layers

    monkeypatch.setattr(adapter.native, "make_layers", fake_make_layers)


def _make_model(monkeypatch, *, role: str):
    afd_config = SimpleNamespace(
        compute_gate_on_attention=False,
        connector="P2pNcclAFDConnector",
        role=role,
    )
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )
    return adapter.AFDDeepseekV4Model(vllm_config=_vllm_config())


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_v4_model_constructs_only_role_owned_decoder_stages(
    monkeypatch,
    construction_env,
    role,
):
    model = _make_model(monkeypatch, role=role)

    for layer in model.layers:
        if role == "attention":
            assert isinstance(layer.attn, _FakeAttention)
            assert isinstance(layer.ffn, adapter.RemoteDeepseekV4FFN)
            assert hasattr(layer, "hc_attn_fn")
        else:
            assert isinstance(layer.attn, _FakeMissingLayer)
            assert isinstance(layer.ffn, _FakeMoE)
            assert not hasattr(layer, "hc_attn_fn")
            assert not hasattr(layer, "attn_norm")


def test_v4_ffn_model_has_no_head_mhc_parameters_or_mtp_buffer(
    monkeypatch,
    construction_env,
):
    model = _make_model(monkeypatch, role="ffn")
    parameter_names = {name for name, _ in model.named_parameters()}

    assert model.hc_head_fn is None
    assert model.hc_head_base is None
    assert model.hc_head_scale is None
    assert not {"hc_head_fn", "hc_head_base", "hc_head_scale"} & parameter_names
    assert model._mtp_hidden_buffer is None


def test_v4_attention_model_owns_head_mhc_parameters_and_mtp_buffer(
    monkeypatch,
    construction_env,
):
    model = _make_model(monkeypatch, role="attention")
    parameter_names = {name for name, _ in model.named_parameters()}

    assert {"hc_head_fn", "hc_head_base", "hc_head_scale"} <= parameter_names
    assert model._mtp_hidden_buffer.shape == (8, 16)


def test_v4_decoder_constructor_rejects_unknown_role(
    monkeypatch,
    construction_env,
):
    afd_config = SimpleNamespace(role="invalid")
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: afd_config,
    )

    with pytest.raises(ValueError, match="unsupported AFD role 'invalid'"):
        adapter.AFDDeepseekV4DecoderLayer(
            _vllm_config(),
            prefix="model.layers.0",
        )
