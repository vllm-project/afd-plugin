# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU contract probes; run directly with Python when pytest is unavailable.

The native config/graph boundaries are simulated. Target-runtime validation
remains necessary; these tests exercise AFD factory and lifecycle code itself.
"""

from __future__ import annotations
import __future__

import ast
import copy
import logging
import runpy
import sys
import types
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
FACTORY_PATH = REPO_ROOT / "afd_plugin/compat/patches/npu/ascend_config.py"


def load_functions(path, namespace, names):
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(
        compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )


@contextmanager
def factory_namespace() -> Iterator[
    tuple[dict[str, Any], SimpleNamespace, types.ModuleType]
]:
    # Child lifecycle probes run in isolation without importing the package.
    from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_optional_afd_config

    runtime = runpy.run_path(str(REPO_ROOT / "afd_plugin/compat/npu/runtime_config.py"))
    native = SimpleNamespace(_ASCEND_CONFIG=None, _INIT_VLLM_CONFIG=None)

    class NativeConfig:
        __dataclass_fields__ = dict.fromkeys(
            {
                "native_option",
                "enable_force_eplb",
                "enable_shared_expert_dp",
                "scheduler_config",
                "sparse_kv_offload_config",
                "kvpp_config",
                "dump_config_path",
                "runtime_config_path",
                "runtime_config_hot_reload",
                "runtime_config",
            }
        )

        def __init__(self, **kwargs):
            unknown = set(kwargs) - self.__dataclass_fields__.keys()
            if unknown:
                raise ValueError(f"unknown keys: {sorted(unknown)}")
            self.native_option = kwargs.get("native_option", 0)
            self.enable_force_eplb = kwargs.get("enable_force_eplb", False)
            self.enable_shared_expert_dp = kwargs.get("enable_shared_expert_dp", False)
            self.runtime_config_path = kwargs["runtime_config_path"]
            self.runtime_config_hot_reload = kwargs["runtime_config_hot_reload"]
            self.runtime_config = kwargs["runtime_config"]
            self.initialized = False
            self.rl_config = SimpleNamespace(apply=lambda config: None)
            self.finegrained_tp_config = SimpleNamespace(
                _validate_preconditions=lambda config: None,
            )
            self.xlite_graph_config = self.finegrained_tp_config

        @staticmethod
        def _resolve_dump_config_path(additional):
            return None

        def derive_and_validate(self, config):
            if self.native_option == "invalid":
                raise ValueError("native option failed validation")
            self.initialized = self.native_option != "partial"

        def _check_mix_placement(self):
            pass

    def validate_bool(value, key):
        if not isinstance(value, bool):
            raise ValueError(key)
        return value

    def get_ascend_config():
        config = native._ASCEND_CONFIG
        if config is None or not config.initialized:
            raise RuntimeError("Ascend config is not initialized")
        return config

    native.get_ascend_config = get_ascend_config
    utils = types.ModuleType("vllm_ascend.utils")

    @lru_cache(maxsize=1)
    def enable_sp():
        return bool(native.get_ascend_config().native_option)

    utils.__dict__["enable_sp"] = enable_sp
    utils.__dict__["clear_enable_sp"] = enable_sp.cache_clear
    runtime_module = types.ModuleType("vllm_ascend.observability.runtime_config")
    runtime_module.__dict__["ADDITIONAL_CONFIG_STRIP_KEYS"] = frozenset(
        {
            "runtime_config",
            "runtime_config_path",
            "runtime_config_hot_reload",
            "runtime_report_dir",
            "runtime_dump_dir",
        }
    )
    runtime_boot = SimpleNamespace(
        path=None, hot_reload=False, runtime_config=SimpleNamespace()
    )
    runtime_module.__dict__["build_runtime_config_from_additional"] = (
        lambda additional: runtime_boot
    )
    logger = logging.getLogger("afd_test_ascend_config")
    ns: dict[str, Any] = {
        "AFD_ASYNC_CONNECTOR": AFD_ASYNC_CONNECTOR,
        "parse_optional_afd_config": parse_optional_afd_config,
        "DEEPSEEK_V2_MODEL_TYPES": runtime["DEEPSEEK_V2_MODEL_TYPES"],
        "npu_model_uses_sequence_parallel_moe": runtime[
            "npu_model_uses_sequence_parallel_moe"
        ],
        "ascend_config": native,
        "AscendConfig": NativeConfig,
        "SchedulerConfig": SimpleNamespace(
            from_additional_config=lambda config: config
        ),
        "SparseKVOffloadConfig": SimpleNamespace(
            from_additional_config=lambda config, value: value,
        ),
        "KVPPConfig": SimpleNamespace(from_vllm_config=lambda config: None),
        "_is_ascend_config_initialized": lambda config: (
            config is not None and config.initialized
        ),
        "validate_additional_config_bool": validate_bool,
        "logger": SimpleNamespace(
            warning_once=logger.warning,
            warning=logger.warning,
        ),
        "importlib": SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: None)),
    }
    load_functions(FACTORY_PATH, ns, {"init_ascend_config"})
    with patch.dict(
        sys.modules,
        {utils.__name__: utils, runtime_module.__name__: runtime_module},
    ):
        yield ns, native, utils


class AscendConfigLifecycleTests(unittest.TestCase):
    def test_namespace_identity_cache_refresh_and_unknown_validation(self):
        additional = {
            "afd": {"role": "attention"},
            "enable_force_eplb": True,
            "gdn_prefill_backend": "flashinfer",
            "kda_prefill_backend": "triton",
            "native_option": 1,
        }
        config = SimpleNamespace(additional_config=additional)
        before = copy.deepcopy(additional)
        with factory_namespace() as (ns, native, utils):
            with self.assertLogs("afd_test_ascend_config", level="WARNING"):
                result = ns["init_ascend_config"](config)
            self.assertIs(ns["init_ascend_config"](config), result)
            self.assertIs(native.get_ascend_config(), result)
            self.assertIs(config.additional_config, additional)
            self.assertEqual(additional, before)
            self.assertTrue(result.enable_force_eplb)
            self.assertTrue(utils.enable_sp())
            for legacy_key in ("typo",):
                config.additional_config = {
                    "afd": {},
                    legacy_key: True,
                    "refresh": True,
                }
                with self.assertRaisesRegex(ValueError, legacy_key):
                    ns["init_ascend_config"](config)
                self.assertIs(native.get_ascend_config(), result)
                self.assertTrue(utils.enable_sp())
            config.additional_config = {"afd": {}, "native_option": 0, "refresh": True}
            refreshed = ns["init_ascend_config"](config)
            self.assertIsNot(refreshed, result)
            self.assertIs(native.get_ascend_config(), refreshed)
            self.assertFalse(utils.enable_sp())

    def test_different_vllm_config_identity_replaces_cached_runtime(self):
        config = SimpleNamespace(additional_config={})
        with factory_namespace() as (ns, native, _):
            first = ns["init_ascend_config"](config)
            copied_config = copy.deepcopy(config)
            self.assertEqual(config, copied_config)
            second = ns["init_ascend_config"](copied_config)
            self.assertIsNot(second, first)
            self.assertIs(native.get_ascend_config(), second)

    def test_native_validation_failure_keeps_published_config(self):
        config = SimpleNamespace(additional_config={"native_option": 1})
        with factory_namespace() as (ns, native, utils):
            first = ns["init_ascend_config"](config)
            self.assertTrue(utils.enable_sp())
            config.additional_config = {"native_option": "invalid", "refresh": True}
            with self.assertRaisesRegex(ValueError, "native option failed validation"):
                ns["init_ascend_config"](config)
            self.assertIs(native.get_ascend_config(), first)
            self.assertTrue(utils.enable_sp())

    def test_partial_native_config_is_returned_without_replacing_singleton(self):
        config = SimpleNamespace(additional_config={"native_option": 1})
        with factory_namespace() as (ns, native, utils):
            first = ns["init_ascend_config"](config)
            self.assertTrue(utils.enable_sp())
            config.additional_config = {"native_option": "partial", "refresh": True}
            with self.assertLogs("afd_test_ascend_config", level="WARNING"):
                partial = ns["init_ascend_config"](config)
            self.assertIsNot(partial, first)
            self.assertIs(native.get_ascend_config(), first)
            self.assertTrue(utils.enable_sp())

    def test_async_shared_replication_only_overrides_effective_model_sp(
        self,
    ):  # long-test: native cached SP decision must use effective model SP
        afd = {
            "connector": "CAMAsyncAFDConnector",
            "role": "attention",
            "compute_gate_on_attention": True,
        }
        cases = [
            (afd, "deepseek_v2", {}, True, 1, True),
            (afd, "deepseek_v3", {"enable_shared_expert_dp": False}, True, 1, True),
            ({**afd, "role": "ffn"}, "deepseek_v2", {}, True, 1, False),
            (
                {**afd, "connector": "CAMP2pAFDConnector"},
                "deepseek_v2",
                {},
                True,
                1,
                False,
            ),
            (
                {**afd, "compute_gate_on_attention": False},
                "deepseek_v2",
                {},
                True,
                1,
                False,
            ),
            (afd, "deepseek_v4", {}, True, 1, False),
            (None, "deepseek_v2", {}, True, 1, False),
            (afd, "deepseek_v2", {}, False, 1, False),
            (afd, "deepseek_v2", {}, True, 2, False),
            (afd, "deepseek_v2", {"enable_shared_expert_dp": True}, False, 1, True),
        ]
        for afd_mapping, model_type, options, use_sp, pp_size, expected in cases:
            with self.subTest(afd=afd_mapping, model_type=model_type, options=options):
                additional: dict[str, Any] = dict(options)
                if afd_mapping is not None:
                    additional["afd"] = afd_mapping
                config = SimpleNamespace(
                    additional_config=additional,
                    model_config=SimpleNamespace(
                        hf_text_config=SimpleNamespace(model_type=model_type),
                    ),
                    parallel_config=SimpleNamespace(
                        tensor_parallel_size=4,
                        pipeline_parallel_size=pp_size,
                        use_sequence_parallel_moe=use_sp,
                        all2all_backend="allgather_reducescatter",
                    ),
                )
                before = copy.deepcopy(vars(config))
                with factory_namespace() as (ns, _, _):
                    result = ns["init_ascend_config"](config)
                self.assertIs(result.enable_shared_expert_dp, expected)
                self.assertEqual(vars(config), before)

    def test_factory_binds_already_loaded_native_scheduler_callers(self):
        tree = ast.parse(FACTORY_PATH.read_text())
        aliases = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and cast(ast.Name, node.targets[0]).id == "_ASCEND_CONFIG_ALIAS_MODULES"
        )

        def replacement(config):
            return config

        native = SimpleNamespace()
        ns: dict[str, Any] = {
            "ascend_config": native,
            "sys": sys,
            "_ASCEND_CONFIG_ALIAS_MODULES": aliases,
            "init_ascend_config": replacement,
        }
        load_functions(FACTORY_PATH, ns, {"apply_afd_ascend_config_patch"})
        loaded = {
            name: types.ModuleType(name)
            for name in (
                "vllm_ascend.patch.platform.patch_engine_core",
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler",
            )
        }
        with patch.dict(sys.modules, loaded):
            ns["apply_afd_ascend_config_patch"]()
            config = object()
            for module in loaded.values():
                self.assertIs(module.init_ascend_config(config), config)
            self.assertIs(native.init_ascend_config(config), config)
