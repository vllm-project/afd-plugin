from __future__ import annotations

import builtins
import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import afd_plugin
from afd_plugin.compat import is_vllm_version_supported


def test_package_import_is_cpu_safe():
    assert afd_plugin.__version__
    assert afd_plugin.AFDConfig().connector == "P2pNcclAFDConnector"


def test_register_afd_is_idempotent():
    afd_plugin.register_afd()
    afd_plugin.register_afd()


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_registration_does_not_import_native_npu_models(monkeypatch, device_type):
    registry_module = ModuleType("vllm.model_executor.models")
    registrations = []
    registry_module.ModelRegistry = SimpleNamespace(
        register_model=lambda *args: registrations.append(args)
    )
    monkeypatch.setitem(sys.modules, registry_module.__name__, registry_module)
    platform_module = ModuleType("vllm.platforms")
    platform_module.current_platform = SimpleNamespace(device_type=device_type)
    monkeypatch.setitem(sys.modules, platform_module.__name__, platform_module)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace() if name == "vllm" else original_find_spec(name),
    )
    original_import = builtins.__import__
    early_imports = []

    def record_native_imports(name, *args, **kwargs):
        if name == "vllm.platforms" or name.startswith("vllm_ascend"):
            early_imports.append(name)
            raise AssertionError("native NPU import before worker resolution")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_native_imports)
    monkeypatch.setattr(afd_plugin, "_registered", False)
    afd_plugin.register_afd()
    afd_plugin.register_afd()

    assert len(registrations) == len(afd_plugin._DEEPSEEK_MODEL_REGISTRATIONS)
    assert early_imports == []


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
    assert registrations["DeepseekV4ForCausalLM"] == (
        "afd_plugin.model_executor.models.deepseek_v4:AFDDeepseekV4ForCausalLM"
    )
    assert registrations["DeepSeekV4MTPModel"] == (
        "afd_plugin.model_executor.models.deepseek_v4:AFDDeepSeekV4MTP"
    )


def test_register_afd_does_not_replace_native_deepseek_model():
    pytest.importorskip("vllm")
    from vllm.model_executor.models import ModelRegistry

    afd_plugin.register_afd()

    native_registration = ModelRegistry.models["DeepseekV2ForCausalLM"]
    assert native_registration.module_name == "vllm.model_executor.models.deepseek_v2"
    assert native_registration.class_name == "DeepseekV2ForCausalLM"
    assert "AFDDeepseekV2ForCausalLM" in ModelRegistry.models


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


def test_deepseek_v4_afd_model_config_uses_private_architecture_copy():
    pytest.importorskip("vllm")
    from afd_plugin.model_executor.models.model_utils import get_afd_model_config

    hf_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)

    afd_model_config = get_afd_model_config(model_config)

    assert afd_model_config.hf_config.architectures == ["AFDDeepseekV4ForCausalLM"]
    assert model_config.hf_config.architectures == ["DeepseekV4ForCausalLM"]


def test_deepseek_v4_mtp_draft_config_uses_private_afd_architecture_copy():
    pytest.importorskip("vllm")
    from afd_plugin.model_executor.models.model_utils import (
        install_afd_speculative_model_config,
    )

    hf_config = SimpleNamespace(architectures=["DeepSeekV4MTPModel"])
    draft_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    speculative_config = SimpleNamespace(draft_model_config=draft_config)
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    install_afd_speculative_model_config(vllm_config)

    installed = speculative_config.draft_model_config
    assert installed is not draft_config
    assert installed.hf_config.architectures == ["AFDDeepSeekV4MTPModel"]
    assert draft_config.hf_config.architectures == ["DeepSeekV4MTPModel"]


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
    assert is_vllm_version_supported("0.23.0")
    assert is_vllm_version_supported("0.26.0")
    assert not is_vllm_version_supported("0.23.1")
    assert not is_vllm_version_supported("0.25.0")
    assert not is_vllm_version_supported("0.26.1")
