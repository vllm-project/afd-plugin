# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Ascend device operations are doubles; worker lifecycle/order is production code.

Mirrors test_gpu_lifecycle.py for the NPU action module: afd_plugin/elastic/npu.py
is loaded after afd_plugin/elastic/gpu.py is loaded by path and registered under
its real name, so the executor inheritance and hardware-neutral actions are the
production implementations.
"""

import importlib.util
import sys
import types
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).parents[3]


@pytest.fixture
def npu(monkeypatch):
    events: list[object] = []
    workspace = [False]
    pool_handles = iter(f"pool-{i}" for i in range(100))

    def init_workspace(device, num_ubatches):
        workspace[0] = num_ubatches
        events.append("workspace_init")

    def unlock():
        assert workspace[0], "workspace reset during runner shutdown"
        events.append("unlock")

    class Base:
        def __init__(self, worker):
            self.worker = worker

    class AscendUBatchWrapper:
        def __init__(self):
            self.cudagraphs = {}

        def clear_graphs(self):
            self.cudagraphs.clear()
            events.append("clear_dbo_graphs")

    def create_groups(**kwargs):
        assert not kwargs["enable_eplb"]
        events.append(("create_groups", kwargs["new_dp_size"]))

    torch_stub = types.ModuleType("torch")
    torch_stub.__dict__.update(
        {
            "inference_mode": lambda: lambda fn: fn,
            "cuda": SimpleNamespace(
                synchronize=lambda *args: None,
                empty_cache=lambda: None,
            ),
            "npu": SimpleNamespace(
                synchronize=lambda *args: events.append("sync"),
                empty_cache=lambda: events.append("empty_cache"),
                graph_pool_handle=lambda: next(pool_handles),
            ),
            "compiler": SimpleNamespace(reset=lambda: events.append("compile_reset")),
        }
    )

    acl_graph_stub = types.ModuleType("vllm_ascend.compilation.acl_graph")
    acl_graph_stub.__dict__.update(
        {
            "_acl_graph_wrappers": weakref.WeakSet(),
            "_graph_params": object(),
            "_draft_graph_params": object(),
        }
    )
    vllm_ascend_compilation = types.ModuleType("vllm_ascend.compilation")
    vllm_ascend_compilation.acl_graph = acl_graph_stub  # type: ignore[attr-defined]

    class Platform:
        _global_graph_pool = next(pool_handles)

    platform_stub = types.ModuleType("vllm.platforms")
    platform_stub.current_platform = Platform()  # type: ignore[attr-defined]

    stubs: dict[str, Any] = {
        "torch": torch_stub,
        "torch_npu": types.ModuleType("torch_npu"),
        "vllm.platforms": platform_stub,
        "vllm.compilation.cuda_graph": {
            "CUDAGraphWrapper": SimpleNamespace(
                clear_all_graphs=lambda: events.append("clear_graphs")
            )
        },
        "vllm.compilation.wrapper": {"reset_compile_wrapper": lambda model: None},
        "vllm.config": {"set_current_vllm_config": lambda config: nullcontext()},
        "vllm.distributed": {"get_world_group": lambda: None},
        "vllm.distributed.elastic_ep.elastic_execute": {
            "ElasticEPScalingExecutor": Base
        },
        "vllm.distributed.elastic_ep.standby_state": {
            "create_standby_groups": create_groups,
            "pop_standby_groups": lambda: {},
        },
        "vllm.distributed.parallel_state": {
            "_replace_active_groups": lambda **kwargs: events.append("switch_groups")
        },
        "vllm.v1.engine": {
            "ReconfigureDistributedRequest": SimpleNamespace,
            "ReconfigureRankType": SimpleNamespace(KEEP_CURRENT_RANK=-1),
        },
        "vllm.v1.worker.workspace": {
            "init_workspace_manager": init_workspace,
            "unlock_workspace": unlock,
            "lock_workspace": lambda: events.append("lock"),
        },
        "vllm.v1.worker.gpu_ubatch_wrapper": {"UBatchWrapper": AscendUBatchWrapper},
        "vllm_ascend.compilation": vllm_ascend_compilation,
        "afd_plugin.connectors": {
            "AFDConnectorFactory": None,
            "AFDControlPayload": lambda *args, **kwargs: SimpleNamespace(**kwargs),
        },
        "afd_plugin.v1.worker.npu.npu_ubatch_wrapper": {
            "AscendUBatchWrapper": AscendUBatchWrapper
        },
        "afd_plugin.v1.worker.npu.attention_model_runner_v2": {},
        "afd_plugin.compat.npu": {
            "npu_afd_num_ubatches": lambda cfg: (
                2 if cfg.parallel_config.use_ubatching else 1
            )
        },
    }
    for name, attrs in stubs.items():
        if isinstance(attrs, dict):
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
        else:
            module = attrs
        monkeypatch.setitem(sys.modules, name, module)

    # Load the real GPU module under its importable name so npu.py's
    # "from afd_plugin.elastic.gpu import ..." resolves to production code.
    gpu_spec = importlib.util.spec_from_file_location(
        "afd_plugin.elastic.gpu", ROOT / "afd_plugin/elastic/gpu.py"
    )
    assert gpu_spec is not None and gpu_spec.loader is not None
    gpu_module = importlib.util.module_from_spec(gpu_spec)
    sys.modules["afd_plugin.elastic.gpu"] = gpu_module
    gpu_spec.loader.exec_module(gpu_module)

    spec = importlib.util.spec_from_file_location(
        "_afd_npu_cpu_test", ROOT / "afd_plugin/elastic/npu.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module,
        events=events,
        workspace=workspace,
        acl_graph=acl_graph_stub,
        platform=Platform,
        wrappers=acl_graph_stub._acl_graph_wrappers,
        ubatch_wrapper=AscendUBatchWrapper,
    )


def test_release_drops_acl_graphs_before_closing_connector(npu, config):
    # V2: DBO captures live in ACLGraphWrapper registry instances; there is
    # no separate MRV1 ubatch-wrapper graph state to retire.
    events = npu.events
    retired_params = npu.acl_graph._graph_params
    retired_pool = npu.platform._global_graph_pool

    class Graph:
        pass

    graph = Graph()
    old_graph = weakref.ref(graph)

    class ACLGraphWrapperDouble:
        def __init__(self, entries):
            self.concrete_aclgraph_entries = entries
            self.graph_pool = retired_pool

    wrapper = ACLGraphWrapperDouble({8: graph})
    npu.wrappers.add(wrapper)
    model = SimpleNamespace()
    del graph

    def close():
        assert old_graph() is None, "graph still holds the old communicator"
        events.append("close_connector")

    runner = SimpleNamespace(
        model=model,
        get_model=lambda: model,
        connector=SimpleNamespace(close=close),
        _afd_pending_metadata="old",
    )
    worker = SimpleNamespace(
        model_runner=runner,
        device="npu:0",
        vllm_config=config,
        afd_expected_role="attention",
        afd_runtime_deferred=False,
    )
    npu.module.release_link(worker)
    assert runner.model is model  # F-only resize keeps A's wrapper.
    assert worker.afd_runtime_deferred and runner._afd_pending_metadata is None
    assert npu.acl_graph._graph_params is not retired_params
    assert npu.acl_graph._draft_graph_params is not retired_params
    # New captures allocate in a fresh pool, not the retired one.
    assert npu.platform._global_graph_pool is not retired_pool
    assert wrapper.concrete_aclgraph_entries == {}
    assert wrapper.graph_pool is not retired_pool
    assert events.index("close_connector") < events.index("empty_cache")


def test_release_ffn_resets_graph_cache_and_bumps_pool(npu):
    events = npu.events
    graph = SimpleNamespace(reset=lambda: events.append("graph_reset"))
    old_pool = "ffn-old-pool"

    def close():
        events.append("close_connector")

    runner = SimpleNamespace(
        _acl_graphs={8: {"graph": graph, "output": None}},
        graph_pool=old_pool,
        use_aclgraph=True,
        connector=SimpleNamespace(close=close),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        device="npu:0",
        vllm_config=None,
        afd_expected_role="ffn",
    )
    npu.module.release_link(worker)
    assert runner._acl_graphs == {}
    assert runner.graph_pool is not old_pool
    assert events.index("graph_reset") < events.index("close_connector")
    assert events[-1] == "empty_cache"


def test_send_stop_payload_carries_stop_and_synchronizes(npu):
    events = npu.events
    sent = []

    def send(payload):
        sent.append(payload)

    runner = SimpleNamespace(
        connector=SimpleNamespace(
            control_plane=SimpleNamespace(send_dp_metadata_list=send)
        )
    )
    worker = SimpleNamespace(model_runner=runner, device="npu:0")
    npu.module.send_stop(worker)
    assert sent[0].stop is True
    assert sent[0].is_graph_capturing is False and sent[0].is_warmup is False
    assert events[-1] == "sync"


def test_finish_rebuilds_when_no_caches_resident_and_enables_real_warmup(npu):
    npu.workspace[0] = True
    events = npu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="npu:0"
    )

    def profile():
        assert not worker.afd_runtime_deferred
        events.append("profile")

    # V2: no separate profiling-KV free / block_table clear steps;
    # initialize_from_config reassigns KV buffers and block tables.
    worker.model_runner = SimpleNamespace(profile_run=profile)
    worker.afd_kv_cache_config = {"blocks": 16}
    worker.initialize_from_config = lambda cfg: events.append("allocate_kv")
    worker.compile_or_warm_up_model = lambda: events.append("warmup")
    npu.module.finish_attention(worker)
    assert (
        events.index("allocate_kv") < events.index("profile") < events.index("warmup")
    )
    assert events[-2:] == ["lock", "sync"]


def test_reload_recreates_workspace_before_model_and_full_kv(npu, config):
    events = npu.events
    config.parallel_config.enable_dbo = True
    config.parallel_config.use_ubatching = True

    def old_shutdown():
        events.append("shutdown")
        npu.workspace[0] = False

    class Runner:
        def __init__(self, cfg, device):
            assert npu.workspace[0] == 2, "DBO workspace width must be restored"
            events.append("runner")

    runner_module = sys.modules["afd_plugin.v1.worker.npu.attention_model_runner_v2"]
    runner_module.AFDNPUAttentionModelRunnerV2 = Runner  # type: ignore[attr-defined]
    saved_kv = {"groups": ["attention"], "blocks": 16}

    def initialize_kv(kv):
        assert kv == saved_kv and kv is not saved_kv
        events.append("full_kv_init")

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(shutdown=old_shutdown),
        device="npu:0",
        parallel_config=config.parallel_config,
        vllm_config=config,
        afd_kv_cache_config=saved_kv,
        load_model=lambda **kwargs: events.append(
            ("checkpoint", kwargs["load_dummy_weights"])
        ),
        initialize_from_config=initialize_kv,
    )
    executor = npu.module.AFDElasticNPUExecutor(worker)
    executor.reconfig_request = SimpleNamespace(
        new_data_parallel_size=4,
        new_data_parallel_rank=-1,
        new_data_parallel_rank_local=-1,
        new_data_parallel_master_ip="10.0.0.1",
        new_data_parallel_master_port=30000,
        new_data_parallel_master_port_list=[30001],
        coord_store_port=30002,
    )
    executor.switch_and_prepare()
    assert (
        events.index("shutdown")
        < events.index("workspace_init")
        < events.index("runner")
    )
    assert (
        events.index("switch_groups")
        < events.index(("checkpoint", False))
        < events.index("full_kv_init")
    )
    assert worker.parallel_config.data_parallel_size == 4


def test_new_a_loads_checkpoint_without_receiving_experts(npu):
    # NPUWorker.load_model takes no arguments; the call must not forward the
    # GPU executor's load_dummy_weights keyword.
    worker = SimpleNamespace(
        load_model=lambda *args, **kwargs: npu.events.append((args, kwargs))
    )
    executor = npu.module.AFDElasticNPUExecutor(worker)
    executor.load_model()
    executor.broadcast_expert_mapping()
    executor.transfer_weights(2, 4)
    executor.receive_weights()
    executor.prepare_new_worker()
    executor.perform_eplb_reshuffle()
    executor.perform_scale_down_eplb_reshuffle(2)
    executor.rewarm_workspace()
    assert npu.events == [((), {})]


def test_finish_skips_kv_realloc_when_snapshot_already_applied(npu):
    # P1 review allocation-lifetime: initial startup and F-only resize reach
    # afd_finish with the KV already allocated from this exact snapshot
    # (boot EngineCore init or switch_and_prepare). Re-entering
    # initialize_from_config allocates the replacement before stock rebinds,
    # so two full KV budgets are resident at once -- OOM for a
    # --kv-cache-memory-bytes value that fits once.
    npu.workspace[0] = True
    events = npu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="npu:0"
    )

    def profile():
        assert not worker.afd_runtime_deferred
        events.append("profile")

    runner = SimpleNamespace(profile_run=profile, kv_caches=[object()])
    worker.model_runner = runner
    worker.afd_kv_initialized_runner = runner
    worker.afd_kv_cache_config = {"blocks": 16}
    worker.initialize_from_config = lambda cfg: events.append("allocate_kv")
    worker.compile_or_warm_up_model = lambda: events.append("warmup")
    npu.module.finish_attention(worker)
    assert "allocate_kv" not in events
    assert events.index("profile") < events.index("warmup")
    assert events[-2:] == ["lock", "sync"]


def test_finish_falls_back_to_rebuild_when_applied_config_differs(npu):
    # The reuse guard may only skip when the runner verifiably applied this
    # exact layout; anything else keeps the legacy rebuild path.
    npu.workspace[0] = True
    events = npu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="npu:0"
    )
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: None, kv_caches=[object()]
    )
    worker.afd_kv_initialized_runner = SimpleNamespace()  # stale stamp
    worker.afd_kv_cache_config = {"blocks": 16}
    worker.initialize_from_config = lambda cfg: events.append("allocate_kv")
    worker.compile_or_warm_up_model = lambda: events.append("warmup")
    npu.module.finish_attention(worker)
    assert "allocate_kv" in events
