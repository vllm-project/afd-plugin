# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD architecture selection preserves native configs and DSpark contracts."""

from types import SimpleNamespace

import pytest

from tests.unit.fakes.async_ffn import load_host_module
from tests.unit.fakes.vllm_config import npu_dspark_config


@pytest.fixture
def configure_dspark(model_utils):
    return model_utils.configure_afd_dspark


@pytest.fixture
def dspark_config():
    return npu_dspark_config()


def test_dspark_registration_preserves_native_text_config_alias(
    configure_dspark, dspark_config
):
    original = dspark_config.speculative_config.draft_model_config
    configure_dspark(dspark_config)
    draft = dspark_config.speculative_config.draft_model_config
    assert draft.hf_config.architectures == ["AFDDSparkDraftModel"]
    assert draft.hf_text_config is draft.hf_config
    assert original.hf_config.architectures == ["DSparkDraftModel"]


def test_dspark_rejects_connector_without_draft_ffn_loader(
    configure_dspark, dspark_config
):
    dspark_config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    dspark_config.additional_config["afd"]["async"] = False
    dspark_config.additional_config["afd"]["compute_gate_on_attention"] = False
    with pytest.raises(ValueError, match="requires CAMAsyncAFDConnector"):
        configure_dspark(dspark_config)


@pytest.mark.parametrize(
    "field", ["hidden_size", "num_experts_per_tok", "n_routed_experts"]
)
def test_dspark_rejects_incompatible_remote_transport(
    configure_dspark, dspark_config, field
):
    draft = dspark_config.speculative_config.draft_model_config.hf_text_config
    setattr(draft, field, getattr(draft, field) + 1)
    with pytest.raises(ValueError, match=f"transport {field} mismatch"):
        configure_dspark(dspark_config)


@pytest.fixture
def model_utils(monkeypatch):
    return load_host_module(monkeypatch, "afd_plugin.model_executor.models.model_utils")


@pytest.mark.parametrize(
    "architecture,device_type",
    [
        ("DeepseekV2ForCausalLM", "cuda"),
        ("DeepseekV2ForCausalLM", "npu"),
        ("DeepseekV4ForCausalLM", "cuda"),
        ("Qwen3MoeForCausalLM", "cuda"),
        ("Qwen3MoeForCausalLM", "npu"),
        ("Qwen3_5MoeForConditionalGeneration", "cuda"),
    ],
)
def test_model_config_rewrites_architecture_in_private_copy(
    model_utils, architecture, device_type
):
    hf_config = SimpleNamespace(architectures=[architecture])
    original = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    mapped = model_utils.get_afd_model_config(original, device_type=device_type)

    assert mapped is not original
    assert mapped.hf_config is not hf_config
    assert mapped.hf_text_config is mapped.hf_config
    assert mapped.hf_config.architectures == [f"AFD{architecture}"]
    assert hf_config.architectures == [architecture]


@pytest.mark.parametrize(
    "routed_experts_connector,expected_architecture",
    [
        (False, "AFDQwen3MoeForCausalLM"),
        (True, "AFDQwen3MoeRoutedExperts"),
    ],
    ids=["full-model", "routed-experts-shell"],
)
def test_qwen3_routed_connector_selects_the_ffn_only_shell(
    model_utils, routed_experts_connector, expected_architecture
):
    hf_config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
    original = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)

    mapped = model_utils.get_afd_model_config(
        original,
        device_type="cuda",
        routed_experts_connector=routed_experts_connector,
    )

    assert mapped.hf_config.architectures == [expected_architecture]
    assert hf_config.architectures == ["Qwen3MoeForCausalLM"]


def test_model_config_preserves_distinct_nested_text_config(model_utils):
    text = SimpleNamespace(model_type="qwen3_5_moe_text")
    original = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]),
        hf_text_config=text,
    )
    mapped = model_utils.get_afd_model_config(original, device_type="cuda")

    assert mapped.hf_text_config is not text
    assert mapped.hf_text_config is not mapped.hf_config
    assert mapped.hf_config.architectures == ["AFDQwen3_5MoeForConditionalGeneration"]
    assert original.hf_config.architectures == ["Qwen3_5MoeForConditionalGeneration"]


def test_model_config_preserves_unregistered_architecture(model_utils):
    original = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["UnknownForCausalLM"])
    )
    assert model_utils.get_afd_model_config(original, device_type="cuda") is original


def test_model_config_rejects_qwen_on_npu_without_mutation(model_utils):
    hf_config = SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
    with pytest.raises(ValueError, match="Qwen3.5/3.6 supports CUDA execution only"):
        model_utils.get_afd_model_config(
            SimpleNamespace(hf_config=hf_config), device_type="npu"
        )
    assert hf_config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
