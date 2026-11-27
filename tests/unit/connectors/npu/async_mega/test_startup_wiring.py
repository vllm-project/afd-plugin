# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host startup selection for CAMAsyncMegaAFDConnector."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.unit.fakes.async_ffn import load_async_modules, load_ffn_owners


def test_worker_selects_mega_runner_for_mega_connector(monkeypatch):
    _, _, worker = load_ffn_owners(monkeypatch, load_async_modules(monkeypatch))
    config, device, calls = object(), SimpleNamespace(type="npu", index=3), MagicMock()
    parsed = SimpleNamespace(connector="CAMAsyncMegaAFDConnector")
    monkeypatch.setattr(worker, "parse_afd_config", lambda *a, **k: parsed)
    monkeypatch.setattr(worker, "get_world_group", lambda: SimpleNamespace(rank=7))
    monkeypatch.setattr(worker.AFDConnectorFactory, "create_connector", calls.connector)
    monkeypatch.setattr(worker, "MegaNPUFFNModelRunner", calls.mega)
    assert worker.create_npu_ffn_model_runner(config, device) is calls.mega.return_value
