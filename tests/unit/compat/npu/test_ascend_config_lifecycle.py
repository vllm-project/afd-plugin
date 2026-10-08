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
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import types
import unittest
from collections.abc import Callable
from enum import Enum
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
FACTORY_PATH = REPO_ROOT / "afd_plugin/compat/patches/npu/ascend_config.py"
VALIDATION_PATH = REPO_ROOT / "afd_plugin/compat/patches/config_validation.py"
PLATFORM_PATH = REPO_ROOT / "afd_plugin/compat/patches/npu/ascend_platform.py"


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


def factory_namespace():
    events = []
    native = SimpleNamespace(_ASCEND_CONFIG=None, _INIT_VLLM_CONFIG=None)

    class NativeConfig:
        __dataclass_fields__ = {"native_option": None}

        def __init__(self, **kwargs):
            unknown = set(kwargs) - {
                "native_option",
                "enable_force_eplb",
                "scheduler_config",
                "sparse_kv_offload_config",
                "kvpp_config",
                "dump_config_path",
            }
            if unknown:
                raise ValueError(f"unknown keys: {sorted(unknown)}")
            events.append(("construct", kwargs))
            self.rl_config = SimpleNamespace(apply=lambda config: None)
            self.finegrained_tp_config = SimpleNamespace(
                _validate_preconditions=lambda config: None,
            )
            self.xlite_graph_config = self.finegrained_tp_config

        @staticmethod
        def _resolve_dump_config_path(additional):
            return None

        def derive_and_validate(self, config):
            events.append(("derive", config))

    def validate_bool(value, key):
        if not isinstance(value, bool):
            raise ValueError(key)
        return value

    ns: dict[str, Any] = {
        "ascend_config": native,
        "AscendConfig": NativeConfig,
        "SchedulerConfig": SimpleNamespace(
            from_additional_config=lambda config: config
        ),
        "SparseKVOffloadConfig": SimpleNamespace(
            from_additional_config=lambda config, value: value,
        ),
        "KVPPConfig": SimpleNamespace(from_vllm_config=lambda config: None),
        "_is_ascend_config_initialized": lambda config: config is not None,
        "validate_additional_config_bool": validate_bool,
        "logger": SimpleNamespace(
            warning_once=lambda *args: events.append(("warning", args)),
            warning=lambda *args: None,
        ),
        "importlib": SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: None)),
    }
    load_functions(FACTORY_PATH, ns, {"init_ascend_config"})
    return ns, native, events


def lifecycle_probe(role, dbo, *, serialized=None, fail=False):
    events = []
    namespace_calls = []
    cached_configs = set()
    constructed = []
    modules = {}
    npu_module = types.ModuleType("afd_plugin.compat.npu")
    npu_module.__dict__["apply_afd_ascend_config_patch_if_needed"] = lambda: (
        namespace_calls.append(1)
    )
    # Native platform already finalizes the backend in this probe. The separate
    # runtime_config root cause owns worker finalization and is tested separately.
    npu_module.__dict__["fix_all2all_backend_for_afd"] = lambda config: None
    npu_module.__dict__["apply_afd_async_dp_engine_patch_if_needed"] = lambda config: (
        None
    )
    npu_module.__dict__["apply_afd_ascend_engine_core_config_patch_if_needed"] = (
        lambda config: None
    )
    modules[npu_module.__name__] = npu_module

    class NPUPlatform:
        device_type = "npu"

        @classmethod
        def check_and_update_config(cls, config):
            parallel = config.parallel_config
            parallel.enable_dbo = False
            parallel.ubatch_size = 0
            if id(config) not in cached_configs:
                cached_configs.add(id(config))
                if config.additional_config["afd"]["role"] == "ffn":
                    parallel.all2all_backend = "flashinfer_all2allv"
            sp = parallel.all2all_backend != "flashinfer_all2allv"
            events.append((parallel.all2all_backend, sp))
            if sp:
                config.capture_sizes = [
                    size for size in config.capture_sizes if size % 4 == 0
                ]
            if fail:
                raise RuntimeError("platform failure")

    platform_module = types.ModuleType("vllm_ascend.platform")
    platform_module.__dict__["NPUPlatform"] = NPUPlatform
    modules[platform_module.__name__] = platform_module
    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.__dict__["current_platform"] = NPUPlatform
    modules[platforms_module.__name__] = platforms_module
    platform_ns: dict[str, Any] = {
        "dataclass": dataclasses.dataclass,
        "parse_optional_afd_config": lambda config, **kwargs: (
            config.additional_config.get("afd")
        ),
        "_ASCEND_PLATFORM_PATCH_ATTR": "_afd_plugin_ascend_platform_patch_state",
    }
    load_functions(
        PLATFORM_PATH,
        platform_ns,
        {
            "AFDAll2AllValidation",
            "_AFDDBOConfigSnapshot",
            "apply_afd_ascend_dbo_config_patch",
            "_snapshot_afd_dbo_config",
            "_restore_afd_dbo_config",
            "_has_valid_afd_config",
        },
    )
    afd_platform_module = types.ModuleType(
        "afd_plugin.compat.patches.npu.ascend_platform"
    )
    afd_platform_module.__dict__["AFDAll2AllValidation"] = platform_ns[
        "AFDAll2AllValidation"
    ]
    modules[afd_platform_module.__name__] = afd_platform_module

    class ParallelConfig(SimpleNamespace):
        @property
        def use_ubatching(self):
            return self.enable_dbo or self.ubatch_size > 1

    class Config(SimpleNamespace):
        __post_init__: Callable[..., None]

        pass

    def native_post_init(config):
        NPUPlatform.check_and_update_config(config)
        if config.parallel_config.use_ubatching:
            assert config.parallel_config.all2all_backend in {
                "deepep_low_latency",
                "deepep_high_throughput",
                "nixl_ep",
            }

    def native_create(args, usage_context, headless):
        config = Config(
            additional_config=args.additional_config,
            parallel_config=ParallelConfig(
                enable_dbo=args.enable_dbo,
                ubatch_size=args.ubatch_size,
                all2all_backend=args.all2all_backend,
                worker_cls=args.worker_cls,
            ),
            capture_sizes=[1, 2, 4, 8],
        )
        constructed.append(config)
        config.__post_init__()
        return config

    ns: dict[str, Any] = {
        "_original_create_engine_config": native_create,
        "_original_vllm_config_post_init": native_post_init,
        "_is_target_vllm_compatible": lambda: True,
        "_select_afd_worker_for_auto": lambda config: None,
        "parse_optional_afd_config": lambda config: (
            config.get("afd")
            if isinstance(config, dict)
            else config.additional_config.get("afd")
        ),
        "_AFD_TEMP_BACKEND": "deepep_low_latency",
    }
    load_functions(
        VALIDATION_PATH,
        ns,
        {
            "create_engine_config",
            "__post_init__",
            "_apply_afd_npu_config_patches",
            "_should_relax_engine_args_backend",
            "_should_relax_vllm_config_backend",
            "_uses_auto_worker_value",
        },
    )
    Config.__post_init__ = ns["__post_init__"]
    with patch.dict(sys.modules, modules):
        platform_ns["apply_afd_ascend_dbo_config_patch"]()
        args = SimpleNamespace(
            additional_config={"afd": {"role": role}},
            enable_dbo=dbo,
            ubatch_size=2 if dbo else 0,
            all2all_backend="allgather_reducescatter",
            worker_cls="explicit",
        )
        if serialized is None:
            try:
                config = ns["create_engine_config"](args)
            except RuntimeError:
                config = constructed[0]
                assert "_afd_all2all_validation" not in vars(config)
                assert config.parallel_config.enable_dbo == dbo
                assert config.parallel_config.all2all_backend == args.all2all_backend
                raise
        else:
            config = Config(
                additional_config=serialized["additional_config"],
                parallel_config=ParallelConfig(**serialized["parallel_config"]),
                capture_sizes=serialized["capture_sizes"],
            )
        config.__post_init__()
        config.__post_init__()  # cached native singleton path
        assert "_afd_all2all_validation" not in vars(config)
        assert namespace_calls
        payload = {
            "additional_config": config.additional_config,
            "parallel_config": vars(config.parallel_config),
            "capture_sizes": config.capture_sizes,
        }
        return {
            "payload": payload,
            "hash": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "events": events,
        }


class AscendConfigLifecycleTests(unittest.TestCase):
    def test_platform_failure_restores_backend_and_scoped_state(self):
        for role in ("attention", "ffn"):
            with (
                self.subTest(role=role),
                self.assertRaisesRegex(RuntimeError, "platform failure"),
            ):
                lifecycle_probe(role, True, fail=True)

    @unittest.skipUnless(
        os.environ.get("AFD_TEST_VLLM_SOURCE"), "needs pinned vLLM source"
    )
    def test_target_splitting_ops_backend_equivalence(self):
        source = Path(os.environ["AFD_TEST_VLLM_SOURCE"]) / "vllm/config/compilation.py"
        tree = ast.parse(source.read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "set_splitting_ops_for_v1"
        )

        class CompilationMode(Enum):
            NONE = 0
            VLLM_COMPILE = 1

        class CUDAGraphMode(Enum):
            NONE = 0
            FULL = 1
            FULL_DECODE_ONLY = 2
            PIECEWISE = 3
            FULL_AND_PIECEWISE = 4

            def has_piecewise_cudagraphs(self):
                return self in (self.PIECEWISE, self.FULL_AND_PIECEWISE)

        ns: dict[str, Any] = {
            "CompilationMode": CompilationMode,
            "CUDAGraphMode": CUDAGraphMode,
            "logger": SimpleNamespace(
                warning_once=lambda *args: None, info=lambda *args: None
            ),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns
        )
        for mode, graph, sp, splitting, backend in product(
            CompilationMode,
            CUDAGraphMode,
            (False, True),
            (None, [], ["attention", "vllm::mla_forward"]),
            ("allgather_reducescatter", "flashinfer_all2allv"),
        ):
            with self.subTest(
                mode=mode, graph=graph, sp=sp, splitting=splitting, backend=backend
            ):
                actual = SimpleNamespace(
                    mode=mode,
                    cudagraph_mode=graph,
                    splitting_ops=copy.deepcopy(splitting),
                    pass_config=SimpleNamespace(
                        enable_sp=sp,
                        fuse_gemm_comms=False,
                        fuse_attn_quant=False,
                        fuse_rope_kvcache=False,
                        fuse_qk_norm_rope_kvcache=False,
                    ),
                    use_inductor_graph_partition=False,
                    _attention_ops=["attention"],
                    cudagraph_capture_sizes=[4, 8] if sp else [1, 2, 4, 8],
                )
                temporary = copy.deepcopy(actual)
                # Initial and cached post-platform calls must have the same
                # side effects for native AFD backends and temporary DeepEP.
                for _ in range(2):
                    ns["set_splitting_ops_for_v1"](actual, backend, 2)
                    ns["set_splitting_ops_for_v1"](temporary, "deepep_low_latency", 2)
                    self.assertEqual(vars(actual), vars(temporary))

    def test_namespace_identity_cache_refresh_and_unknown_validation(self):
        ns, native, events = factory_namespace()
        utils = types.ModuleType("vllm_ascend.utils")
        clears = []
        utils.__dict__["clear_enable_sp"] = lambda: clears.append(1)
        additional = {
            "afd": {"role": "attention"},
            "enable_force_eplb": True,
            "gdn_prefill_backend": "flashinfer",
            "kda_prefill_backend": "triton",
            "native_option": 1,
        }
        config = SimpleNamespace(additional_config=additional)
        with patch.dict(sys.modules, {utils.__name__: utils}):
            result = ns["init_ascend_config"](config)
            self.assertIs(ns["init_ascend_config"](config), result)
            self.assertIs(native._INIT_VLLM_CONFIG, config)
            self.assertIs(config.additional_config, additional)
            self.assertIs(events[2][1], config)
            self.assertTrue(events[1][1]["enable_force_eplb"])
            self.assertEqual(len(clears), 1)
            self.assertTrue(any(event[0] == "warning" for event in events))
            for legacy_key in (
                "enable_force_load_balance",
                "force_load_balance_topn_per_rank",
            ):
                config.additional_config = {
                    "afd": {},
                    legacy_key: True,
                    "refresh": True,
                }
                with self.assertRaisesRegex(ValueError, legacy_key):
                    ns["init_ascend_config"](config)
                self.assertIs(native._ASCEND_CONFIG, result)
                self.assertEqual(len(clears), 1)
            config.additional_config = {"afd": {}, "typo": 1, "refresh": True}
            with self.assertRaisesRegex(ValueError, "typo"):
                ns["init_ascend_config"](config)
            self.assertIs(native._ASCEND_CONFIG, result)
            self.assertEqual(len(clears), 1)
            config.additional_config = {"afd": {}, "refresh": True}
            self.assertIsNot(ns["init_ascend_config"](config), result)
            self.assertEqual(len(clears), 2)

    def test_factory_aliases_include_early_scheduler(self):
        tree = ast.parse(FACTORY_PATH.read_text())
        aliases = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and cast(ast.Name, node.targets[0]).id == "_ASCEND_CONFIG_ALIAS_MODULES"
        )
        self.assertIn("vllm_ascend.patch.platform.patch_engine_core", aliases)

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
        loaded = {name: types.ModuleType(name) for name in aliases}
        with patch.dict(sys.modules, loaded):
            ns["apply_afd_ascend_config_patch"]()
            for module in loaded.values():
                self.assertIs(module.init_ascend_config, replacement)
            self.assertIs(native.init_ascend_config, replacement)

    def test_parent_and_fresh_child_hashes_and_cached_graph_sizes(self):
        for role in ("attention", "ffn"):
            for dbo in (False, True):
                with self.subTest(role=role, dbo=dbo):
                    parent = lifecycle_probe(role, dbo)
                    child = json.loads(
                        subprocess.check_output(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--child-probe",
                            ],
                            input=json.dumps(
                                {
                                    "role": role,
                                    "dbo": dbo,
                                    "serialized": parent["payload"],
                                }
                            ),
                            text=True,
                        )
                    )
                    self.assertEqual(parent["hash"], child["hash"])
                    expected = [4, 8] if role == "attention" else [1, 2, 4, 8]
                    self.assertEqual(child["payload"]["capture_sizes"], expected)
                    self.assertTrue(
                        all(sp == (role == "attention") for _, sp in child["events"])
                    )


if __name__ == "__main__":
    if "--child-probe" in sys.argv:
        print(json.dumps(lifecycle_probe(**json.load(sys.stdin))))
    else:
        unittest.main()
