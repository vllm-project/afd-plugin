# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402
from vllm.config.multimodal import MultiModalConfig  # noqa: E402
from vllm.model_executor.models import qwen3_5 as native  # noqa: E402
from vllm.model_executor.models.utils import StageMissingLayer  # noqa: E402

from afd_plugin.model_executor.models import qwen3_5 as adapter  # noqa: E402


def test_qwen_remote_experts_proxy_preserves_completed_ffn_output_under_tp(
    monkeypatch,
):
    class FakeRemoteExperts(nn.Module):
        gate = object()

        def forward(self, *, hidden_states, router_logits):
            assert router_logits is hidden_states
            return hidden_states + 1

    remote_moe = object.__new__(adapter.AFDQwen3_5RemoteExpertsMoE)
    nn.Module.__init__(remote_moe)
    remote_moe.experts = FakeRemoteExperts()
    remote_moe.tp_size = 2
    remote_moe.is_sequence_parallel = False
    remote_moe.shared_expert = None
    remote_moe.replicate_shared_expert = False
    monkeypatch.setattr(
        adapter.next_native,
        "tensor_model_parallel_all_gather",
        lambda *_args, **_kwargs: pytest.fail("unexpected TP collective"),
    )
    hidden_states = torch.zeros(2, 4)

    output = remote_moe(hidden_states)

    assert torch.equal(output, hidden_states + 1)


def test_ffn_compute_ffn_output_calls_native_internal_router():
    calls = []

    class FakeInternalMoe(native.Qwen3NextSparseMoeBlock):
        def forward(self, hidden_states):
            calls.append(hidden_states)
            return hidden_states + 1

    moe = object.__new__(FakeInternalMoe)
    nn.Module.__init__(moe)
    moe.experts = type("Experts", (), {"gate": object()})()
    layer = object.__new__(adapter.AFDQwen3_5DecoderLayer)
    nn.Module.__init__(layer)
    layer.afd_role = "ffn"
    layer.mlp = moe
    hidden_states = torch.zeros(2, 4)

    output = layer.compute_ffn_output(hidden_states)

    assert torch.equal(output, hidden_states + 1)


def test_ffn_compute_ffn_output_rejects_external_router():
    moe = object.__new__(native.Qwen3NextSparseMoeBlock)
    nn.Module.__init__(moe)
    moe.experts = type("Experts", (), {"gate": None})()
    layer = object.__new__(adapter.AFDQwen3_5DecoderLayer)
    nn.Module.__init__(layer)
    layer.afd_role = "ffn"
    layer.mlp = moe

    with pytest.raises(RuntimeError, match="must use its local router"):
        layer.compute_ffn_output(torch.zeros(2, 4))


def test_qwen_causal_model_retains_afd_aligned_cache_scope(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "parse_optional_afd_config",
        lambda *_args, **_kwargs: SimpleNamespace(role="attention"),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        scheduler_config=None,
    )

    with pytest.raises(NotImplementedError, match="AFD Qwen3.5/3.6 requires"):
        adapter.AFDQwen3_5MoeForCausalLM(vllm_config=vllm_config)


def test_qwen_text_only_uses_upstream_missing_vision_tower_stage(
    monkeypatch,
):  # long-test: native multimodal constructor consumes the text-only AFD wrapper
    class FakeCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.make_empty_intermediate_tensors = object()

    class FakeVisionTower(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    multimodal_config = MultiModalConfig(
        language_model_only=True,
        mm_device_do_normalize=False,
    )
    # Native _mark_tower_model resolves the MM config through the
    # ModelConfig method rather than the raw attribute.
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(out_hidden_size=8),
        ),
        multimodal_config=multimodal_config,
        get_multimodal_config=lambda: multimodal_config,
    )
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=model_config,
        quant_config=None,
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(
        adapter,
        "parse_optional_afd_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            role="attention",
            compute_gate_on_attention=False,
        ),
    )
    monkeypatch.setattr(adapter.native, "Qwen3_VisionTransformer", FakeVisionTower)
    monkeypatch.setattr(
        adapter.native,
        "cached_tokenizer_from_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        adapter,
        "AFDQwen3_5MoeForCausalLM",
        lambda **_kwargs: FakeCausalLM(),
    )
    monkeypatch.setattr(
        adapter.AFDQwen3_5MoeForConditionalGeneration,
        "set_moe_parameters",
        lambda _self: None,
    )

    model = adapter.AFDQwen3_5MoeForConditionalGeneration(
        vllm_config=vllm_config,
    )

    assert model.multimodal_config.language_model_only is True
    assert multimodal_config.get_limit_per_prompt("image") == 0
    assert multimodal_config.get_limit_per_prompt("video") == 0
    assert isinstance(model.visual, StageMissingLayer)
    assert list(model.visual.named_parameters()) == []
    assert [
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("visual.")
    ] == []


@pytest.fixture
def conditional_config(monkeypatch):
    afd = SimpleNamespace(role="attention", compute_gate_on_attention=False)
    config = SimpleNamespace(
        lora_config=None,
        speculative_config=None,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
    )
    monkeypatch.setattr(adapter, "parse_optional_afd_config", lambda *a, **kw: afd)
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *a, **kw: pytest.fail("visual path was constructed"),
    )
    return SimpleNamespace(afd=afd, config=config)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("afd.compute_gate_on_attention", True, "compute_gate_on_attention=False"),
        ("config.speculative_config", SimpleNamespace(), "speculative decoding"),
        ("config.lora_config", SimpleNamespace(), "LoRA"),
        (
            "config.model_config.multimodal_config.language_model_only",
            False,
            "pass --language-model-only",
        ),
        ("config.parallel_config.use_sequence_parallel_moe", True, "SP MoE"),
        ("config.parallel_config.enable_eplb", True, "EPLB"),
        (
            "config.parallel_config.pipeline_parallel_size",
            2,
            "pipeline_parallel_size=1",
        ),
    ],
)
def test_qwen_rejects_unsupported_configuration_before_construction(
    conditional_config, path, value, message
):
    *parents, field = path.split(".")
    target = conditional_config
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, field, value)

    with pytest.raises(ValueError, match=message):
        adapter.AFDQwen3_5MoeForConditionalGeneration(
            vllm_config=conditional_config.config
        )
