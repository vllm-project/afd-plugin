# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from threading import Event, Thread
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def probes(monkeypatch, tmp_path, capsys):
    graph_modes = SimpleNamespace(EAGER="eager", REPLAY="replay")
    ffn_module = ModuleType("afd_plugin.v1.worker.ffn_model_runner")
    ffn_worker_module = ModuleType("afd_plugin.v1.worker.ffn_worker")
    start_marker = tmp_path / "evaluation-start"
    monkeypatch.setenv("AFD_E2E_MRV2_EVIDENCE_START", str(start_marker))
    result = object()

    class GraphManager:
        def run_fullgraph(self, desc):
            return result

    class NativeRunner:
        def __init__(self):
            self.graph = True
            self.fail = False
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=2)
            )
            self.afd_config = SimpleNamespace(async_dp=False)
            self._afd_pending_metadata = SimpleNamespace(
                num_stages=2, tokens_unpadded_lens=[2, 2], tokens_lens=[2, 2]
            )

        def execute_model(self, scheduler_output, *args):
            if self.graph:
                GraphManager().run_fullgraph("descriptor")
            if self.fail:
                raise RuntimeError("execution failed")
            return result

    class FFNRunner:
        def __init__(self):
            self.graph = True
            self.fail = False
            self.on_execute = lambda: None

        def execute_model(self, *args, **kwargs):
            self.on_execute()
            ffn_module.__dict__["graph_run_mode"](
                is_warmup=kwargs["is_warmup"],
                is_graph_capturing=kwargs["is_graph_capturing"],
                is_graph_replaying=kwargs["is_graph_replaying"],
                graph_enabled=True,
                graph_exists=self.graph,
            )
            if self.fail:
                raise RuntimeError("execution failed")

    class FFNWorker:
        def __init__(self):
            self.health_checks = 0
            self.on_health = lambda: None
            self.fail = False
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_rank=1)
            )

        def raise_ffn_loop_error_if_any(self) -> None:
            self.health_checks += 1
            self.on_health()
            if self.fail:
                raise RuntimeError("FFN loop failed")

    modules = {
        name: ModuleType(name)
        for name in (
            "vllm",
            "vllm.distributed",
            "vllm.platforms",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu",
            "vllm.v1.worker.gpu.cudagraph_utils",
            "vllm.v1.worker.gpu.model_runner",
            "afd_plugin",
            "afd_plugin.v1",
            "afd_plugin.v1.worker",
            "afd_plugin.v1.worker.cuda_graph",
        )
    }
    modules["vllm.platforms"].__dict__["current_platform"] = SimpleNamespace(
        device_type="cuda"
    )
    modules["vllm.distributed"].__dict__["get_tensor_model_parallel_rank"] = lambda: 0
    modules["vllm.v1.worker.gpu.cudagraph_utils"].__dict__["ModelCudaGraphManager"] = (
        GraphManager
    )
    modules["vllm.v1.worker.gpu.model_runner"].__dict__["GPUModelRunner"] = NativeRunner
    modules["afd_plugin.v1.worker"].__dict__["ffn_model_runner"] = ffn_module
    modules["afd_plugin.v1.worker"].__dict__["ffn_worker"] = ffn_worker_module
    ffn_worker_module.__dict__["AFDFFNWorker"] = FFNWorker
    modules["afd_plugin.v1.worker.cuda_graph"].__dict__["AFDGraphRunMode"] = graph_modes
    ffn_module.__dict__["GPUFFNModelRunner"] = FFNRunner
    modules["afd_plugin.v1.worker.cuda_graph"].__dict__["make_ffn_graph_key"] = (
        lambda metadata: tuple(sorted(metadata.items()))
    )
    ffn_module.__dict__["graph_run_mode"] = lambda **kwargs: (
        graph_modes.REPLAY
        if kwargs["is_graph_replaying"]
        and kwargs["graph_exists"]
        and not (kwargs["is_warmup"] or kwargs["is_graph_capturing"])
        else graph_modes.EAGER
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "tests/e2e/mrv2_evidence.py"
    spec = importlib.util.spec_from_file_location("execution_probe_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    worker = module.Worker()
    worker.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=1)
    )
    native = NativeRunner()
    ffn = FFNRunner()

    def execute(role, tokens=2, **flags):
        if role == "attention":
            native._afd_pending_metadata.tokens_lens = [tokens, tokens]
            return native.execute_model("scheduler-output", **flags)
        return ffn.execute_model(
            dp_metadata_list={0: (tokens, tokens), 1: (tokens, tokens)},
            is_graph_replaying=True,
            **flags,
        )

    def summary():
        worker.afd_log_execution_evidence()
        return capsys.readouterr().out

    return SimpleNamespace(
        worker=worker,
        native=native,
        ffn=ffn,
        ffn_worker=FFNWorker(),
        module=module,
        start_marker=start_marker,
        complete_marker=start_marker.with_name("evaluation-complete"),
        result=result,
        execute=execute,
        summary=summary,
    )


ZERO_COUNTS = "graph_replays=0 two_stage_steps=0 two_stage_graph_replays=0"
STARTUP_FLAG = {"attention": "dummy_run", "ffn": "is_warmup"}


def test_ffn_markers_define_window_and_health_rpc_reports_once(probes, capsys):
    # A startup call can still be in flight when the harness creates the marker.
    probes.ffn.on_execute = probes.start_marker.touch
    probes.execute("ffn")
    probes.ffn.on_execute = lambda: None
    probes.execute("ffn")
    probes.execute("ffn")
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert capsys.readouterr().out == ""
    probes.complete_marker.touch()
    # The idle FFN loop never executes again; the health RPC alone reports.
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    summary = capsys.readouterr().out
    assert "graph_replays=2 two_stage_steps=2 two_stage_graph_replays=2" in summary
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert probes.ffn_worker.health_checks == 3
    assert capsys.readouterr().out == ""


def test_ffn_health_error_propagates_without_later_success_evidence(probes, capsys):
    probes.start_marker.touch()
    probes.execute("ffn")
    probes.complete_marker.touch()
    probes.ffn_worker.fail = True
    with pytest.raises(RuntimeError, match="FFN loop failed"):
        probes.ffn_worker.raise_ffn_loop_error_if_any()
    probes.ffn_worker.fail = False
    probes.ffn_worker.raise_ffn_loop_error_if_any()
    assert capsys.readouterr().out == ""


def test_ffn_health_snapshot_waits_for_atomic_counter_update(
    probes, monkeypatch, capsys
):
    updating = Event()
    release_update = Event()
    health_entered = Event()
    native_record = probes.module._ExecutionCounts.record

    def record(counts, replayed, layout):
        native_record(counts, replayed, layout)
        updating.set()
        assert release_update.wait(timeout=2)

    monkeypatch.setattr(probes.module._ExecutionCounts, "record", record)
    probes.ffn_worker.on_health = health_entered.set
    probes.start_marker.touch()
    executing = Thread(target=probes.execute, args=("ffn",))
    reporting = Thread(target=probes.ffn_worker.raise_ffn_loop_error_if_any)
    executing.start()
    try:
        assert updating.wait(timeout=1)
        probes.complete_marker.touch()
        reporting.start()
        assert health_entered.wait(timeout=1)
        assert capsys.readouterr().out == ""
    finally:
        release_update.set()
        executing.join(timeout=1)
        if reporting.ident is not None:
            reporting.join(timeout=1)
    assert not executing.is_alive() and not reporting.is_alive()
    summary = capsys.readouterr().out
    assert "graph_replays=1 two_stage_steps=1 two_stage_graph_replays=1" in summary


def test_probe_summary_is_one_stdout_write_despite_logging_and_other_workers(
    probes, monkeypatch, caplog
):
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARNING")
    caplog.set_level("WARNING")
    other_worker_summary = (
        "AFD E2E execution: dp_rank=0 tp_rank=0 graph_replays=0 "
        "two_stage_steps=0 two_stage_graph_replays=0 layouts={}\n"
    )

    class InterleavedOutput(io.StringIO):
        def write(self, value: str) -> int:
            count = super().write(value)
            # A second worker can write between any two shared-pipe writes.
            if value.startswith("AFD E2E execution: "):
                super().write(other_worker_summary)
            return count

    output = InterleavedOutput()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", output)
        probes.worker.afd_log_execution_evidence()
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    for rank, line in zip((1, 0), lines, strict=True):
        assert line.startswith(f"AFD E2E execution: dp_rank={rank} ")
        json.loads(line.split(" layouts=", 1)[1])


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_probe_counts_without_per_step_logging_and_resets_window(probes, capsys, role):
    # Startup calls before reset cannot satisfy the evaluation gate.
    result = probes.execute(role)
    assert result is (probes.result if role == "attention" else None)
    probes.worker.afd_reset_execution_evidence()
    for _ in range(20):
        probes.execute(role)
    assert capsys.readouterr().out == ""
    summary = probes.summary()
    assert summary.count("AFD E2E execution: ") == 1
    assert (
        "dp_rank=1 tp_rank=0 graph_replays=20 "
        "two_stage_steps=20 two_stage_graph_replays=20" in summary
    )
    assert "class=" not in summary
    probes.execute(role)
    probes.worker.afd_reset_execution_evidence()
    assert ZERO_COUNTS in probes.summary()


@pytest.mark.parametrize(
    ("role", "flags"),
    [
        ("attention", {"dummy_run": True}),
        ("attention", {"is_profile": True}),
        ("ffn", {"is_warmup": True}),
        ("ffn", {"is_graph_capturing": True}),
    ],
)
def test_probe_excludes_startup_execution(probes, role, flags):
    probes.worker.afd_reset_execution_evidence()
    probes.execute(role, **flags)
    assert ZERO_COUNTS in probes.summary()


@pytest.mark.parametrize("real_tokens", [[0, 0], [4, 0]])
def test_attention_probe_requires_two_nonempty_microbatches(probes, real_tokens):
    probes.worker.afd_reset_execution_evidence()
    probes.native._afd_pending_metadata.tokens_unpadded_lens = real_tokens
    probes.execute("attention")
    assert "two_stage_steps=0 two_stage_graph_replays=0" in probes.summary()


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_probe_distinguishes_eager_fallback_and_excludes_failed_calls(probes, role):
    target = probes.native if role == "attention" else probes.ffn
    target.graph = False
    probes.worker.afd_reset_execution_evidence()
    probes.execute(role)
    summary = probes.summary()
    assert "graph_replays=0 two_stage_steps=1 two_stage_graph_replays=0" in summary
    probes.worker.afd_reset_execution_evidence()
    target.graph = target.fail = True
    with pytest.raises(RuntimeError, match="execution failed"):
        probes.execute(role)
    assert ZERO_COUNTS in probes.summary()


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("interruption", ["eager", "layout", "startup"])
def test_probe_requires_consecutive_same_layout_replays(probes, role, interruption):
    target = probes.native if role == "attention" else probes.ffn
    probes.worker.afd_reset_execution_evidence()
    probes.execute(role)
    if interruption == "layout":
        probes.execute(role, tokens=3)
    else:
        target.graph = interruption != "eager"
        probes.execute(role, **{STARTUP_FLAG[role]: interruption == "startup"})
        target.graph = True
        probes.execute(role)
    assert '"consecutive_graph":[]' in probes.summary()

    probes.worker.afd_reset_execution_evidence()
    probes.execute(role)
    probes.execute(role)
    assert '"consecutive_graph":[[[0,[2,2]],[1,[2,2]]]]' in probes.summary()
