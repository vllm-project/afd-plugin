from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import afd_plugin
from afd_plugin.compat import is_vllm_version_supported


def test_package_import_is_cpu_safe():
    assert afd_plugin.__version__
    assert afd_plugin.AFDConfig().connector == "P2pNcclAFDConnector"


def test_register_afd_is_idempotent():
    afd_plugin.register_afd()
    afd_plugin.register_afd()


def test_deepseek_afd_model_registration_paths_are_lazy_strings():
    registrations = afd_plugin._DEEPSEEK_MODEL_REGISTRATIONS

    assert registrations["DeepseekV2ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV2ForCausalLM"
    )
    assert registrations["DeepseekV3ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV3ForCausalLM"
    )
    assert registrations["DeepseekV32ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV3ForCausalLM"
    )


def test_qwen_afd_model_registration_path_is_lazy_string():
    registrations = afd_plugin._QWEN_MODEL_REGISTRATIONS

    assert registrations["Qwen3_5MoeForConditionalGeneration"] == (
        "afd_plugin.model_executor.models.qwen3_5:AFDQwen3_5MoeForConditionalGeneration"
    )


def test_merged_model_registrations_include_deepseek_and_qwen():
    registrations = afd_plugin._MODEL_REGISTRATIONS

    assert registrations["DeepseekV2ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v2:AFDDeepseekV2ForCausalLM"
    )
    assert registrations["Qwen3_5MoeForConditionalGeneration"] == (
        "afd_plugin.model_executor.models.qwen3_5:AFDQwen3_5MoeForConditionalGeneration"
    )


def test_register_afd_does_not_replace_native_deepseek_model():
    pytest.importorskip("vllm")
    from vllm.model_executor.models import ModelRegistry

    afd_plugin.register_afd()

    native_registration = ModelRegistry.models["DeepseekV2ForCausalLM"]
    assert native_registration.module_name == "vllm.model_executor.models.deepseek_v2"
    assert native_registration.class_name == "DeepseekV2ForCausalLM"
    assert "AFDDeepseekV2ForCausalLM" in ModelRegistry.models


def test_register_afd_does_not_replace_native_qwen_model():
    pytest.importorskip("vllm")
    from vllm.model_executor.models import ModelRegistry

    afd_plugin.register_afd()

    native_registration = ModelRegistry.models["Qwen3_5MoeForConditionalGeneration"]
    assert native_registration.module_name == "vllm.model_executor.models.qwen3_5"
    assert native_registration.class_name == "Qwen3_5MoeForConditionalGeneration"
    assert "AFDQwen3_5MoeForConditionalGeneration" in ModelRegistry.models


def test_afd_model_config_uses_private_architecture_copy():
    pytest.importorskip("vllm")
    from afd_plugin.model_executor.models.model_utils import get_afd_model_config

    hf_config = SimpleNamespace(architectures=["DeepseekV2ForCausalLM"])
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
    )

    afd_model_config = get_afd_model_config(model_config)

    assert afd_model_config is not model_config
    assert afd_model_config.hf_config is not model_config.hf_config
    assert afd_model_config.hf_text_config is afd_model_config.hf_config
    assert afd_model_config.hf_config.architectures == ["AFDDeepseekV2ForCausalLM"]
    assert model_config.hf_config.architectures == ["DeepseekV2ForCausalLM"]


def test_afd_model_config_preserves_nested_text_config():
    pytest.importorskip("vllm")
    from afd_plugin.model_executor.models.model_utils import get_afd_model_config

    hf_text_config = SimpleNamespace()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV2ForCausalLM"]),
        hf_text_config=hf_text_config,
    )

    afd_model_config = get_afd_model_config(model_config)

    # deepcopy privatizes the whole graph; a genuinely distinct nested
    # hf_text_config stays distinct from hf_config.
    assert afd_model_config.hf_config is not model_config.hf_config
    assert afd_model_config.hf_text_config is not hf_text_config
    assert afd_model_config.hf_text_config is not afd_model_config.hf_config


def test_afd_model_config_maps_qwen_architecture_without_aliasing():
    pytest.importorskip("vllm")
    from afd_plugin.model_executor.models.model_utils import get_afd_model_config

    hf_text_config = SimpleNamespace(model_type="qwen3_5_moe_text")
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
        ),
        hf_text_config=hf_text_config,
    )

    afd_model_config = get_afd_model_config(model_config)

    assert afd_model_config.hf_config.architectures == [
        "AFDQwen3_5MoeForConditionalGeneration"
    ]
    assert model_config.hf_config.architectures == [
        "Qwen3_5MoeForConditionalGeneration"
    ]
    assert afd_model_config.hf_text_config is not hf_text_config


def test_entry_point_is_registered():
    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matches = [ep for ep in entry_points if ep.name == "afd"]
    assert matches
    assert matches[0].value == "afd_plugin:register_afd"


def test_connectors_export_attn_output_without_recv_alias():
    root = Path(__file__).resolve().parents[3]
    metadata_source = (root / "afd_plugin/connectors/metadata.py").read_text()
    namespace_source = (root / "afd_plugin/connectors/__init__.py").read_text()

    assert "class AFDA2FTransferPayload:" in metadata_source
    assert '"AFDA2FTransferPayload"' in metadata_source
    assert "AFDRecvOutput" not in metadata_source
    assert "AFDA2FTransferPayload," in namespace_source
    assert '"AFDA2FTransferPayload"' in namespace_source
    assert "AFDRecvOutput" not in namespace_source


def test_vllm_version_support_is_exact_target():
    assert is_vllm_version_supported("0.26.0")
    assert not is_vllm_version_supported("0.25.0")
    assert not is_vllm_version_supported("0.26.1")
