# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Regression for native EEP's zero-new-remote [-0:] slice."""

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from afd_plugin.elastic.config import is_elastic_attention


def test_only_new_local_actor_is_waited_and_started(monkeypatch, config):
    events = []

    class Actor:
        def __init__(self, name):
            self.wait_for_init = SimpleNamespace(
                remote=lambda: events.append((name, "wait"))
            )
            self.run = SimpleNamespace(remote=lambda: events.append((name, "run")))

    new_actor = Actor("new_local")
    factory = SimpleNamespace(
        options=lambda **kwargs: SimpleNamespace(remote=lambda **kwargs: new_actor)
    )
    stubs: dict[str, dict[str, Any]] = {
        "ray": {"remote": lambda cls: factory, "get": lambda refs: None},
        "ray.runtime_env": {"RuntimeEnv": lambda **kwargs: kwargs},
        "ray.util.scheduling_strategies": {
            "PlacementGroupSchedulingStrategy": lambda **kwargs: kwargs
        },
        "vllm.v1.engine.core": {
            "DPMoEEngineCoreActor": Actor,
            "EngineCoreActor": Actor,
        },
        "afd_plugin.elastic.ray_actor": {"AFDEngineCoreActor": Actor},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[3] / "afd_plugin/compat/patches/elastic_actors.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "scale_up_elastic_ep"
    ]
    namespace = {
        "VllmConfig": SimpleNamespace,
        "is_elastic_attention": is_elastic_attention,
        "upstream": SimpleNamespace(_apply_dp_identity_suffix=lambda cfg, rank: None),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    config.model_config.is_moe = True
    config.parallel_config.world_size = 1
    config.parallel_config.data_parallel_size_local = 1
    config.parallel_config.data_parallel_master_ip = "10.0.0.1"
    pg = SimpleNamespace(bundle_specs=[{"GPU": 1, "node:10.0.0.1": 0.001}, {"CPU": 1}])
    manager = SimpleNamespace(
        local_engine_actors=[Actor("old_local")],
        remote_engine_actors=[Actor("old_remote")],
        add_dp_placement_groups=lambda cfg, size: ([pg], [1]),
        env_vars_dict={},
        executor_class=None,
        log_stats=False,
        addresses=None,
        created_placement_groups=[],
        placement_group_is_local=[],
        run_refs=[],
        actor_run_ref_dict={},
    )
    namespace["scale_up_elastic_ep"](manager, config, 3, 0)
    assert events == [("new_local", "wait"), ("new_local", "run")]
    assert len(manager.actor_run_ref_dict) == 1
