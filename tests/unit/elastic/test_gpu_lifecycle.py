# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Device operations are doubles; worker lifecycle/order is production code."""

import ast
import copy
import importlib.util
import sys
import types
import weakref
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from afd_plugin.config import parse_afd_config

ROOT = Path(__file__).parents[3]


@pytest.fixture
def gpu(monkeypatch):
    events: list[object] = []
    workspace = [False]

    def init_workspace(device, num_ubatches):
        workspace[0] = num_ubatches
        events.append("workspace_init")

    def unlock():
        assert workspace[0], "workspace reset during runner shutdown"
        events.append("unlock")

    class Base:
        def __init__(self, worker):
            self.worker = worker

    class UBatchWrapper:
        def __init__(self):
            self.cudagraphs = {}

        def clear_graphs(self):
            self.cudagraphs.clear()
            events.append("clear_dbo_graphs")

    def create_groups(**kwargs):
        assert not kwargs["enable_eplb"]
        events.append(("create_groups", kwargs["new_dp_size"]))

    stubs: dict[str, dict[str, Any]] = {
        "torch": {
            "inference_mode": lambda: lambda fn: fn,
            "cuda": SimpleNamespace(
                synchronize=lambda *args: events.append("sync"),
                empty_cache=lambda: events.append("empty_cache"),
            ),
            "compiler": SimpleNamespace(reset=lambda: events.append("compile_reset")),
        },
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
        "vllm.v1.worker.gpu_ubatch_wrapper": {"UBatchWrapper": UBatchWrapper},
        "afd_plugin.connectors": {
            "AFDConnectorFactory": None,
            "AFDControlPayload": lambda *args, **kwargs: SimpleNamespace(**kwargs),
        },
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_afd_gpu_cpu_test", ROOT / "afd_plugin/elastic/gpu.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, events=events, workspace=workspace)


@pytest.mark.parametrize("enable_dbo", [False, True])
def test_release_drops_dbo_graphs_before_closing_connector(gpu, config, enable_dbo):
    class Graph:
        pass

    model = gpu.module.UBatchWrapper() if enable_dbo else SimpleNamespace()
    graph = Graph()
    old_graph = weakref.ref(graph)
    if enable_dbo:
        model.cudagraphs[8] = graph
    del graph

    def close():
        assert old_graph() is None, "graph still holds the old communicator"
        gpu.events.append("close_connector")

    runner = SimpleNamespace(
        model=model,
        get_model=lambda: model,
        connector=SimpleNamespace(close=close),
        _afd_pending_metadata="old",
    )
    worker = SimpleNamespace(
        model_runner=runner,
        device="cuda:0",
        vllm_config=config,
        afd_expected_role="attention",
        afd_runtime_deferred=False,
    )
    gpu.module.release_link(worker)
    assert runner.model is model  # F-only resize keeps A's wrapper.
    assert worker.afd_runtime_deferred and runner._afd_pending_metadata is None
    if enable_dbo:
        assert gpu.events.index("clear_dbo_graphs") < gpu.events.index(
            "close_connector"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_topology_rpc_restores_config_after_msgpack(gpu, config, role):
    msgspec = pytest.importorskip("msgspec")
    previous = parse_afd_config(config)
    target = replace(
        previous,
        role=role,
        host="10.0.0.2",
        port=30000,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )
    # EngineCore collective_rpc has an untyped nested args tuple. Msgspec
    # decodes dataclasses/maps there as dicts, without restoring worker types.
    (decoded,) = msgspec.msgpack.decode(msgspec.msgpack.encode((asdict(target),)))
    wrapper = SimpleNamespace(afd_config=previous)
    runner = SimpleNamespace(
        afd_config=previous,
        get_model=lambda: SimpleNamespace(modules=lambda: [wrapper]),
    )
    worker = SimpleNamespace(
        vllm_config=config,
        model_runner=runner,
        afd_expected_role=role,
    )
    gpu.module.update_topology(worker, decoded)
    assert runner.afd_config == target
    assert wrapper.afd_config == target
    assert parse_afd_config(config) == target


@pytest.mark.parametrize("enable_dbo", [False, True])
def test_reload_recreates_workspace_before_model_and_full_kv(
    gpu, monkeypatch, config, enable_dbo
):
    events = gpu.events
    config.parallel_config.enable_dbo = enable_dbo
    config.parallel_config.use_ubatching = enable_dbo

    def old_shutdown():
        events.append("shutdown")
        gpu.workspace[0] = False

    class Runner:
        def __init__(self, cfg, device):
            assert gpu.workspace[0] == (2 if enable_dbo else 1)
            events.append("runner")

    module = types.ModuleType("afd_plugin.v1.worker.attention_model_runner_v2")
    module.AFDAttentionModelRunnerV2 = Runner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    saved_kv = {"groups": ["attention"], "blocks": 16}

    def initialize_kv(kv):
        assert kv == saved_kv and kv is not saved_kv
        events.append("full_kv_init")

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(shutdown=old_shutdown),
        device="cuda:0",
        parallel_config=config.parallel_config,
        vllm_config=config,
        afd_kv_cache_config=saved_kv,
        load_model=lambda **kwargs: events.append(
            ("checkpoint", kwargs["load_dummy_weights"])
        ),
        initialize_from_config=initialize_kv,
    )
    executor = gpu.module.AFDElasticGPUExecutor(worker)
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


def test_new_a_loads_checkpoint_without_receiving_experts(gpu):
    worker = SimpleNamespace(load_model=lambda **kwargs: gpu.events.append(kwargs))
    executor = gpu.module.AFDElasticGPUExecutor(worker)
    executor.load_model()
    executor.broadcast_expert_mapping()
    executor.transfer_weights(2, 4)
    executor.receive_weights()
    executor.prepare_new_worker()
    executor.perform_eplb_reshuffle()
    executor.perform_scale_down_eplb_reshuffle(2)
    executor.rewarm_workspace()
    assert gpu.events == [{"load_dummy_weights": False}]


def test_finish_rebuilds_when_no_caches_resident_and_enables_real_warmup(gpu):
    gpu.workspace[0] = True
    events = gpu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="cuda:0"
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
    gpu.module.finish_attention(worker)
    assert (
        events.index("allocate_kv") < events.index("profile") < events.index("warmup")
    )
    assert events[-2:] == ["lock", "sync"]


def _worker_methods(class_name, methods):
    """Compile actual method bodies without importing the native GPU worker."""
    path = ROOT / "afd_plugin/v1/worker/ffn_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    tree.body = [
        copy.deepcopy(node)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                set_device=lambda device: None, synchronize=lambda *args: None
            )
        )
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_stop_exits_before_touching_batch_or_model():
    methods = _worker_methods("AFDFFNWorker", {"_run_ffn_server_loop"})
    # STOP has no metadata/graph fields. Any attempt to run a batch will fail.
    runner = SimpleNamespace(
        connector=SimpleNamespace(
            control_plane=SimpleNamespace(
                recv_dp_metadata_list=lambda: SimpleNamespace(stop=True)
            )
        )
    )
    worker = SimpleNamespace(
        _ffn_shutdown_event=SimpleNamespace(is_set=lambda: False),
        model_runner=runner,
        device=SimpleNamespace(type="cuda"),
    )
    methods["_run_ffn_server_loop"](worker)


def test_join_timeout_keeps_live_thread_handle():
    methods = _worker_methods("AFDFFNWorker", {"join_ffn_server_loop"})
    thread = SimpleNamespace(join=lambda **kwargs: None, is_alive=lambda: True)
    worker = SimpleNamespace(_ffn_thread=thread)
    with pytest.raises(TimeoutError, match="STOP"):
        methods["join_ffn_server_loop"](worker)
    assert worker._ffn_thread is thread


@pytest.mark.vllm_runtime
def test_switch_and_prepare_constructs_the_real_v2_runner():
    # The seam pin needs the real runtime (torch + vLLM). Guard both
    # production imports so the dependency-light CPU selection (-m "not gpu
    # and not vllm_runtime") and hosts without vLLM skip cleanly instead of
    # turning ModuleNotFoundError into a red suite; the mocked lifecycle
    # regressions above stay CPU-safe.
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    # Regression: this seam used to import the never-shipped V1 name
    # (afd_plugin.v1.worker.attention_model_runner); the lifecycle fixture
    # injecting a fake under exactly that name masked the ModuleNotFound.
    # Pin the seam to the module and class the attention worker really uses.
    import importlib
    import inspect

    from afd_plugin.elastic.gpu import AFDElasticGPUExecutor

    runner_module = importlib.import_module(
        "afd_plugin.v1.worker.attention_model_runner_v2"
    )
    source = inspect.getsource(AFDElasticGPUExecutor.switch_and_prepare)
    assert "attention_model_runner_v2" in source
    assert "AFDAttentionModelRunnerV2" in source
    assert list(inspect.signature(runner_module.AFDAttentionModelRunnerV2).parameters)[
        :2
    ] == ["vllm_config", "device"]


def test_finish_skips_kv_realloc_when_snapshot_already_applied(gpu):
    # P1 review allocation-lifetime: initial startup and F-only resize reach
    # afd_finish with the KV already allocated from this exact snapshot
    # (boot EngineCore init or switch_and_prepare). Re-entering
    # initialize_from_config allocates the replacement before stock rebinds,
    # so two full KV budgets are resident at once -- OOM for a
    # --kv-cache-memory-bytes value that fits once.
    gpu.workspace[0] = True
    events = gpu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="cuda:0"
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
    gpu.module.finish_attention(worker)
    assert "allocate_kv" not in events
    assert events.index("profile") < events.index("warmup")
    assert events[-2:] == ["lock", "sync"]


def test_finish_falls_back_to_rebuild_when_applied_config_differs(gpu):
    # The reuse guard may only skip when the runner verifiably applied this
    # exact layout; anything else keeps the legacy rebuild path.
    gpu.workspace[0] = True
    events = gpu.events
    worker = SimpleNamespace(
        afd_runtime_deferred=True, vllm_config=None, device="cuda:0"
    )
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: None, kv_caches=[object()]
    )
    worker.afd_kv_initialized_runner = SimpleNamespace()  # stale stamp
    worker.afd_kv_cache_config = {"blocks": 16}
    worker.initialize_from_config = lambda cfg: events.append("allocate_kv")
    worker.compile_or_warm_up_model = lambda: events.append("warmup")
    gpu.module.finish_attention(worker)
    assert "allocate_kv" in events
