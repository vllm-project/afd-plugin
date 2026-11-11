# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host checks for Pull Mega execution evidence."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("header", "native", "expected"),
    [([1, 0, 2], True, 1), ([0, 0, 2], True, 0), ([1, 0, 2], False, 0)],
)
def test_pull_mega_evidence_records_only_completed_native_work(
    monkeypatch, tmp_path, capsys, header, native, expected
):
    marker = tmp_path / "evaluation-start"
    marker.write_text("window")
    monkeypatch.setenv("AFD_E2E_MRV2_EVIDENCE_START", str(marker))

    class Connector:
        def recv_ffn_output(self, *args, **kwargs):
            return SimpleNamespace(shape=(1,))

    class PullSession:
        pass

    class Executor:
        def __init__(self, op, values):
            self.op = op
            self.batch_info = SimpleNamespace(
                cpu=lambda: SimpleNamespace(tolist=lambda: values)
            )

        def __call__(self):
            return "result"

    class Worker:
        def __init__(self):
            self.model_runner = SimpleNamespace(
                connector=SimpleNamespace(
                    extra_info=SimpleNamespace(ffn_mega_moe=True),
                    _session=PullSession(),
                )
            )
            self.vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_rank=0)
            )

        def raise_ffn_loop_error_if_any(self):
            return None

    native_op = object()
    npu = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
    )
    ops = SimpleNamespace(
        afd_ascend=SimpleNamespace(afd_mega_moe_pull_native=native_op)
    )

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)

    module("torch", npu=npu, ops=ops)
    module("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0)
    module("afd_plugin.connectors.npu.async_cam", CAMAsyncAFDConnector=Connector)
    module(
        "afd_plugin.connectors.npu.async_cam_runtime.async_cam_mega",
        AsyncCAMMegaExecutor=Executor,
    )
    module(
        "afd_plugin.connectors.npu.async_cam_runtime.async_cam_session",
        PullCAMSession=PullSession,
    )
    module("afd_plugin.v1.worker.npu.ffn_worker", AFDNPUFFNWorker=Worker)
    path = Path(__file__).parent / "models/deepseek_v4_flash/pull_mega_evidence.py"
    spec = importlib.util.spec_from_file_location("_pull_mega_evidence_test", path)
    assert spec is not None and spec.loader is not None
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)

    executor = Executor(native_op if native else object(), header)
    assert executor() == "result"
    (tmp_path / "evaluation-complete").write_text("window")
    worker = Worker()
    worker.raise_ffn_loop_error_if_any()
    assert f"pull_mega=1 native_completed={expected}" in capsys.readouterr().out
