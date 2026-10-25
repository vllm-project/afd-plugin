# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host ownership checks; native receive/Graph behavior is device-qualified."""

from __future__ import annotations

import threading

import pytest

from afd_plugin.ffn_faas import runner as runner_module
from afd_plugin.ffn_faas.runner import FFNModelRunner
from afd_plugin.ffn_faas.service import FFNService
from afd_plugin.ffn_faas.settings import FFNConfig
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription
from tests.unit.fakes.ffn_faas import HostBackend


@pytest.fixture
def make_runner(checkpoint_dir, monkeypatch):
    def build(backend):
        monkeypatch.setattr(runner_module, "create_backend", lambda *_: backend)
        service = FFNService(rank=0, model_path=str(checkpoint_dir))
        settings = FFNConfig(
            checkpoint_dir,
            attention_ranks=1,
            attention_tp=1,
            ffn_ranks=1,
            shutdown_timeout=0.01,
        )
        return FFNModelRunner(
            service, V4CheckpointDescription.read(checkpoint_dir), settings
        ), service

    return build


@pytest.mark.parametrize("stop_during_prepare", [False, True])
def test_successful_host_exit_releases_after_receiver_exit(
    make_runner, stop_during_prepare
):
    backend = HostBackend(stop_during_prepare=stop_during_prepare)
    runner, service = make_runner(backend)
    runner.run()
    assert service.snapshot()["phase"] == "stopped"
    assert service.snapshot()["resources_released"] is True
    assert backend.released


@pytest.mark.parametrize("fail", ["prepare", "activate", "execute"])
def test_failed_native_boundary_retains_owners(make_runner, fail):
    backend = HostBackend(fail=fail)
    runner, service = make_runner(backend)
    with pytest.raises(RuntimeError):
        runner.run()
    assert service.snapshot()["phase"] == "failed"
    assert service.snapshot()["resources_released"] is False
    assert not backend.released


def test_pending_host_receiver_is_not_closed(make_runner):
    backend = HostBackend(block=True)
    runner, service = make_runner(backend)
    errors = []
    done = threading.Event()

    def run():
        try:
            runner.run()
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert backend.entered.wait(timeout=5)
        service.request_stop()
        assert done.wait(timeout=5)
        assert errors and "CAM receive is pending" in errors[0]
        assert service.snapshot()["resources_released"] is False
        assert not backend.released
    finally:
        backend.release.set()
        thread.join()
        assert backend.finished.wait(timeout=5)
