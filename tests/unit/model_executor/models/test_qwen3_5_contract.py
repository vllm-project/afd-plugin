from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
nn = torch.nn

from vllm.config.multimodal import MultiModalConfig  # noqa: E402
from vllm.model_executor.models import qwen3_5 as native  # noqa: E402
from vllm.model_executor.models.utils import StageMissingLayer  # noqa: E402

from afd_plugin.model_executor.models import qwen3_5 as adapter  # noqa: E402
from afd_plugin.model_executor.models.deepseek_v2 import (  # noqa: E402
    AFDAttentionFusedMoE,
)


def test_qwen_adapter_keeps_native_signatures_and_forward_methods():
    assert inspect.signature(adapter.AFDQwen3_5DecoderLayer.__init__) == (
        inspect.signature(native.Qwen3_5DecoderLayer.__init__)
    )
    assert inspect.signature(adapter.AFDQwen3_5Model.__init__) == inspect.signature(
        native.Qwen3_5Model.__init__
    )
    assert inspect.signature(
        adapter.AFDQwen3_5MoeForConditionalGeneration.__init__
    ) == inspect.signature(native.Qwen3_5MoeForConditionalGeneration.__init__)
    assert adapter.AFDQwen3_5DecoderLayer.forward is native.Qwen3_5DecoderLayer.forward
    assert adapter.AFDQwen3_5Model.forward is native.Qwen3_5Model.forward
    assert (
        adapter.AFDQwen3_5MoeForConditionalGeneration.forward
        is native.Qwen3_5MoeForConditionalGeneration.forward
    )


def test_attention_moe_uses_native_forward_and_parameter_free_proxy():
    assert (
        adapter.AFDQwen3_5RemoteExpertsMoE.forward
        is native.Qwen3NextSparseMoeBlock.forward
    )
    proxy = AFDAttentionFusedMoE(
        layer_idx=7,
        is_internal_router=True,
    )
    assert list(proxy.parameters()) == []


def test_qwen_remote_experts_proxy_preserves_completed_ffn_output_under_tp(
    monkeypatch,
):
    class FakeRemoteExperts(nn.Module):
        is_internal_router = True

        def forward(self, *, hidden_states, router_logits):
            assert router_logits is hidden_states
            return hidden_states + 1

    remote_moe = object.__new__(adapter.AFDQwen3_5RemoteExpertsMoE)
    nn.Module.__init__(remote_moe)
    remote_moe.experts = FakeRemoteExperts()
    remote_moe.tp_size = 2
    remote_moe.is_sequence_parallel = False
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

    assert calls == [hidden_states]
    assert torch.equal(output, hidden_states + 1)


def test_qwen_conditional_model_rejects_multimodal_before_visual_construction(
    monkeypatch,
):
    model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(language_model_only=False),
    )
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=model_config,
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
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )

    with pytest.raises(ValueError, match="pass --language-model-only"):
        adapter.AFDQwen3_5MoeForConditionalGeneration(vllm_config=vllm_config)


def test_qwen_text_only_validation_accepts_language_model_only():
    adapter._validate_qwen_text_only(
        SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
    )


def test_qwen_text_only_validation_exposes_missing_multimodal_config():
    with pytest.raises(AttributeError, match="multimodal_config"):
        adapter._validate_qwen_text_only(SimpleNamespace())


def test_qwen_text_only_validation_exposes_missing_language_model_only():
    with pytest.raises(AttributeError, match="language_model_only"):
        adapter._validate_qwen_text_only(
            SimpleNamespace(multimodal_config=SimpleNamespace()),
        )


def test_qwen_conditional_model_rejects_attention_side_gate_before_construction(
    monkeypatch,
):
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(
        adapter,
        "parse_optional_afd_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            role="attention",
            compute_gate_on_attention=True,
        ),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )

    with pytest.raises(
        ValueError,
        match="Qwen3.5/3.6 CUDA supports compute_gate_on_attention=False only",
    ):
        adapter.AFDQwen3_5MoeForConditionalGeneration(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("config_name", "message"),
    [
        ("lora_config", "LoRA"),
        ("speculative_config", "speculative decoding"),
    ],
)
def test_qwen_conditional_model_rejects_unsupported_feature_before_construction(
    monkeypatch,
    config_name,
    message,
):
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        speculative_config=None,
    )
    setattr(vllm_config, config_name, SimpleNamespace())
    monkeypatch.setattr(
        adapter,
        "parse_optional_afd_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            role="attention",
            compute_gate_on_attention=False,
        ),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )

    with pytest.raises(ValueError, match=message):
        adapter.AFDQwen3_5MoeForConditionalGeneration(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("config_name", "message"),
    [
        ("use_sequence_parallel_moe", "SP MoE"),
        ("enable_eplb", "EPLB"),
    ],
)
def test_qwen_rejects_unsupported_parallel_feature_before_construction(
    monkeypatch,
    config_name,
    message,
):
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        pipeline_parallel_size=1,
        use_sequence_parallel_moe=False,
    )
    setattr(parallel_config, config_name, True)
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        parallel_config=parallel_config,
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
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )

    with pytest.raises(ValueError, match=message):
        adapter.AFDQwen3_5MoeForConditionalGeneration(vllm_config=vllm_config)


def test_qwen_conditional_model_rejects_pipeline_parallelism_before_construction(
    monkeypatch,
):
    vllm_config = SimpleNamespace(
        lora_config=None,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=2,
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
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: pytest.fail("visual path was constructed"),
    )

    with pytest.raises(
        ValueError,
        match="Qwen3.5/3.6 CUDA supports pipeline_parallel_size=1 only",
    ):
        adapter.AFDQwen3_5MoeForConditionalGeneration(vllm_config=vllm_config)


def test_qwen_conditional_model_initializes_for_text_only(monkeypatch):
    class FakeCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.make_empty_intermediate_tensors = object()

    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(vision_config=SimpleNamespace()),
        multimodal_config=SimpleNamespace(
            language_model_only=True,
            mm_encoder_tp_mode="weights",
        ),
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
    monkeypatch.setattr(
        adapter.AFDQwen3_5MoeForConditionalGeneration,
        "_mark_tower_model",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        adapter.AFDQwen3_5MoeForConditionalGeneration,
        "_mark_language_model",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        adapter.native,
        "Qwen3_VisionTransformer",
        lambda *_args, **_kwargs: nn.Identity(),
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


def test_qwen_text_only_uses_upstream_missing_vision_tower_stage(monkeypatch):
    class FakeCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.make_empty_intermediate_tensors = object()

    class FakeVisionTower(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    multimodal_config = MultiModalConfig(language_model_only=True)
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(vision_config=SimpleNamespace()),
        multimodal_config=multimodal_config,
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

    assert multimodal_config.get_limit_per_prompt("image") == 0
    assert multimodal_config.get_limit_per_prompt("video") == 0
    assert isinstance(model.visual, StageMissingLayer)
    assert list(model.visual.named_parameters()) == []
    assert [
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("visual.")
    ] == []
    assert all(
        parameter.is_meta for parameter in model.visual.__dict__["module"].parameters()
    )
