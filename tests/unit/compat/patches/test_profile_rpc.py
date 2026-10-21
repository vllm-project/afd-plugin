# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat.patches import profile_rpc


@pytest.mark.parametrize(
    "method",
    [
        "cam_profile_prepare",
        "cam_profile_commit",
        "cam_profile_stop",
        "cam_profile_status",
    ],
)
def test_cam_profile_rpc_preserves_every_dp_worker(method):
    calls = []
    engines = [b"\x00\x00", b"\x01\x00"]
    command = {"session_id": "test-session", "command_seq": 1}

    async def call_utility(utility, *args, engine):
        calls.append((utility, args, engine))
        return [f"{engine.hex()}-worker0", f"{engine.hex()}-worker1"]

    client = SimpleNamespace(core_engines=engines, _call_utility_async=call_utility)
    result = asyncio.run(
        profile_rpc.collective_rpc_async(client, method, 5.0, (command,), None)
    )

    assert result == ["0000-worker0", "0000-worker1", "0100-worker0", "0100-worker1"]
    assert calls == [
        ("collective_rpc", (method, 5.0, (command,), None), engine)
        for engine in engines
    ]


@pytest.mark.parametrize("method", ["profile", "cam_profile_unknown", lambda: None])
def test_other_collective_rpcs_preserve_native_forwarding(method):
    calls = []
    expected = [object()]
    kwargs = {"key": "value"}

    async def call_utility(*args):
        calls.append(args)
        return expected

    client = SimpleNamespace(call_utility_async=call_utility)
    result = asyncio.run(
        profile_rpc.collective_rpc_async(client, method, 3.0, (1,), kwargs)
    )

    assert result is expected
    assert calls == [("collective_rpc", method, 3.0, (1,), kwargs)]


def test_cam_profile_rpc_does_not_hide_a_failed_dp_rank():
    calls = []

    async def call_utility(*args, engine):
        calls.append(engine)
        if engine == b"\x01\x00":
            raise RuntimeError("worker unavailable")
        return [{"worker_id": "dp0", "state": "READY"}]

    client = SimpleNamespace(
        core_engines=[b"\x00\x00", b"\x01\x00"], _call_utility_async=call_utility
    )
    with pytest.raises(RuntimeError, match="worker unavailable"):
        asyncio.run(profile_rpc.collective_rpc_async(client, "cam_profile_status"))

    assert calls == client.core_engines


def test_installer_only_changes_internal_dp_collective_rpc(monkeypatch):
    async def native_collective(self, *args, **kwargs):
        return ["native"]

    async def native_utility(self, *args):
        return "utility"

    class AsyncMPClient:
        collective_rpc_async = native_collective

    class DPAsyncMPClient(AsyncMPClient):
        pass

    class DPLBAsyncMPClient(DPAsyncMPClient):
        call_utility_async = native_utility

    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    module = ModuleType("vllm.v1.engine.core_client")
    module.DPLBAsyncMPClient = DPLBAsyncMPClient
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(profile_rpc, "is_target_vllm_compatible", lambda: True)

    assert profile_rpc.apply_cam_profile_rpc_patch()
    assert profile_rpc.apply_cam_profile_rpc_patch()
    assert DPLBAsyncMPClient.collective_rpc_async is profile_rpc.collective_rpc_async
    assert DPLBAsyncMPClient.call_utility_async is native_utility
    assert DPAsyncMPClient.collective_rpc_async is native_collective
    assert AsyncMPClient.collective_rpc_async is native_collective


def test_installer_rejects_unsupported_vllm(monkeypatch):
    monkeypatch.setattr(profile_rpc, "is_target_vllm_compatible", lambda: False)

    assert profile_rpc.apply_cam_profile_rpc_patch() is False
