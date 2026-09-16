# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def config_patch(monkeypatch):
    native = ModuleType("vllm_ascend.ascend_config")
    package = ModuleType("vllm_ascend")
    package.__path__ = []
    package.ascend_config = native
    events = []

    class StrictConfig:
        __dataclass_fields__ = {"rl_config": None, "valid": None}

        @staticmethod
        def _resolve_dump_config_path(additional):
            return None

        def __init__(
            self,
            *,
            scheduler_config,
            sparse_kv_offload_config,
            kvpp_config,
            dump_config_path,
            rl_config=None,
            valid=True,
        ):
            if rl_config is not None and not isinstance(rl_config, dict):
                raise ValueError("invalid rl_config")
            events.append("construct")
            self.valid = valid
            self.initialized = False
            self.rl_config = SimpleNamespace(apply=lambda config: events.append("rl"))
            self.finegrained_tp_config = SimpleNamespace(
                _validate_preconditions=lambda config: events.append("finegrained")
            )
            self.xlite_graph_config = SimpleNamespace(
                _validate_preconditions=lambda config: events.append("xlite")
            )

        def derive_and_validate(self, config):
            events.append(("derive", config))
            if not self.valid:
                raise ValueError("invalid business config")
            self.initialized = True

    def validate_bool(value, name):
        if not isinstance(value, bool):
            raise ValueError(name)
        return value

    native.AscendConfig = StrictConfig
    native.SchedulerConfig = SimpleNamespace(
        from_additional_config=lambda config: config
    )
    native.SparseKVOffloadConfig = SimpleNamespace(
        from_additional_config=lambda config, sparse: sparse
    )
    native.KVPPConfig = SimpleNamespace(from_vllm_config=lambda config: None)
    native._is_ascend_config_initialized = lambda config: config.initialized
    native.validate_additional_config_bool = validate_bool
    native.logger = SimpleNamespace(warning=lambda *args: None)
    native._ASCEND_CONFIG = None
    native._INIT_VLLM_CONFIG = None
    native.init_ascend_config = lambda config: None
    utils = ModuleType("vllm_ascend.utils")
    utils.clear_enable_sp = lambda: events.append(("clear", native._ASCEND_CONFIG))
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/compat/patches/npu/ascend_config.py"
    )
    spec = importlib.util.spec_from_file_location("afd_test_ascend_config_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, native, events


def test_plugin_namespace_preserves_original_config_and_validation_order(config_patch):
    patch, native, events = config_patch
    additional = {
        "afd": {"role": "attention"},
        "enable_force_load_balance": True,
        "force_load_balance_topn_per_rank": 2,
    }
    before = additional.copy()
    config = SimpleNamespace(additional_config=additional)
    result = patch.init_ascend_config(config)
    assert config.additional_config is additional
    assert additional == before
    assert events == [
        "construct",
        ("derive", config),
        "rl",
        "finegrained",
        "xlite",
        ("clear", result),
    ]
    assert native._ASCEND_CONFIG is result
    assert native._INIT_VLLM_CONFIG is config
    assert str(inspect.signature(patch.init_ascend_config)) == "(vllm_config)"


@pytest.mark.parametrize("additional", [None, {}, {"valid": True}])
def test_non_afd_configuration_and_identity_cache(config_patch, additional):
    patch, native, events = config_patch
    config = SimpleNamespace(additional_config=additional)
    first = patch.init_ascend_config(config)
    count = len(events)
    assert patch.init_ascend_config(config) is first
    assert len(events) == count
    second = patch.init_ascend_config(SimpleNamespace(additional_config=additional))
    assert second is not first
    assert native._ASCEND_CONFIG is second


@pytest.mark.parametrize(
    "additional", [{"refresh": True}, {"rl_config": {"enabled": True}}]
)
def test_refresh_bypasses_cache(config_patch, additional):
    patch, native, events = config_patch
    config = SimpleNamespace(additional_config=additional)
    assert patch.init_ascend_config(config) is not patch.init_ascend_config(config)


@pytest.mark.parametrize(
    "invalid", [{"unrelated_typo": True}, {"valid": False}, {"rl_config": 1}]
)
def test_failure_preserves_native_cache(config_patch, invalid):
    patch, native, events = config_patch
    config = SimpleNamespace(additional_config={})
    previous = patch.init_ascend_config(config)
    config.additional_config.update(invalid)
    # Invalid RL input must bypass an identity-cache hit by itself.
    if "rl_config" not in invalid:
        config.additional_config["refresh"] = True
    with pytest.raises((TypeError, ValueError)):
        patch.init_ascend_config(config)
    assert native._ASCEND_CONFIG is previous
    assert native._INIT_VLLM_CONFIG is config
    assert (
        sum(isinstance(event, tuple) and event[0] == "clear" for event in events) == 1
    )


def test_existing_and_future_aliases_use_replacement(config_patch, monkeypatch):
    patch, native, events = config_patch
    aliases = []
    for name in patch._ASCEND_CONFIG_ALIAS_MODULES:
        caller = ModuleType(name)
        caller.init_ascend_config = native.init_ascend_config
        monkeypatch.setitem(sys.modules, name, caller)
        aliases.append(caller)
    absent = patch._ASCEND_CONFIG_ALIAS_MODULES[-1]
    monkeypatch.delitem(sys.modules, absent)
    patch.apply_afd_ascend_config_patch()
    patch.apply_afd_ascend_config_patch()
    assert all(
        caller.init_ascend_config is patch.init_ascend_config for caller in aliases[:-1]
    )
    assert absent not in sys.modules
    from vllm_ascend.ascend_config import init_ascend_config

    assert init_ascend_config is patch.init_ascend_config
    result = aliases[0].init_ascend_config(
        SimpleNamespace(additional_config={"afd": {}})
    )
    assert native._ASCEND_CONFIG is result


def test_native_omni_extension_filtering_is_preserved(config_patch, monkeypatch):
    patch, native, events = config_patch
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace())
    additional = {"omni_extension": True, "afd": {"role": "ffn"}}
    config = SimpleNamespace(additional_config=additional)
    result = patch.init_ascend_config(config)
    assert native._ASCEND_CONFIG is result
    assert config.additional_config is additional
    assert additional["omni_extension"] is True


@pytest.mark.parametrize(
    "additional", [{"refresh": "yes"}, {"rl_config": {"enabled": "yes"}}]
)
def test_invalid_refresh_flags_preserve_cache(config_patch, additional):
    patch, native, events = config_patch
    previous_config = SimpleNamespace(additional_config={})
    previous = patch.init_ascend_config(previous_config)
    with pytest.raises(ValueError, match="additional_config"):
        patch.init_ascend_config(SimpleNamespace(additional_config=additional))
    assert native._ASCEND_CONFIG is previous
    assert native._INIT_VLLM_CONFIG is previous_config
