# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A connector created before ``Worker.init_device`` gets vLLM's device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.platforms import interface as platform_interface  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.v1.worker import connector_device  # noqa: E402


class _CudaLikePlatform(platform_interface.Platform):
    device_control_env_var = "CUDA_VISIBLE_DEVICES"


@pytest.fixture(autouse=True)
def _cuda_like_platform(monkeypatch):
    monkeypatch.setattr(connector_device, "current_platform", _CudaLikePlatform)
    # vLLM keeps the assigned ids process-wide; start every case unset.
    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def _config(
    *,
    data_parallel_rank_local=0,
    distributed_executor_backend="mp",
    assigned_physical_gpu_ids=None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            distributed_executor_backend=distributed_executor_backend,
            data_parallel_backend="mp",
            nnodes_within_dp=1,
            data_parallel_rank_local=data_parallel_rank_local,
            data_parallel_index=data_parallel_rank_local,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            assigned_physical_gpu_ids=assigned_physical_gpu_ids,
        ),
    )


@pytest.mark.parametrize(
    ("executor_backend", "expected_index"),
    [("mp", 1), ("ray", 0), ("external_launcher", 0)],
)
def test_adds_local_dp_offset_like_vllm(executor_backend, expected_index):
    """The local DP rank offsets the device, except where vLLM skips it.

    Why: vLLM adds the offset only inside init_device, after the connector
    bootstrap; without it every DP rank's connector used GPU 0.
    """
    config = _config(
        data_parallel_rank_local=1,
        distributed_executor_backend=executor_backend,
    )

    assert connector_device.connector_device_index(config, 0) == expected_index


def test_maps_assigned_device_ids_without_visible_devices():
    """``--device-ids 4,5`` without CUDA_VISIBLE_DEVICES places rank 0 on cuda:4.

    Why: vLLM maps its logical index through the assigned physical ids, so a
    connector using the logical index would run on another rank's GPU.
    """
    config = _config(assigned_physical_gpu_ids=[4, 5])

    assert connector_device.connector_device_index(config, 0) == 4
    assert connector_device.connector_device_index(config, 1) == 5


def test_maps_assigned_device_ids_through_visible_devices(monkeypatch):
    """With CUDA_VISIBLE_DEVICES the assigned ids map to visible ordinals."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    config = _config(assigned_physical_gpu_ids=[5, 4])

    assert connector_device.connector_device_index(config, 0) == 1
    assert connector_device.connector_device_index(config, 1) == 0


def test_without_assigned_ids_the_index_is_the_local_rank(monkeypatch):
    """The default mp launch keeps the logical index as the device index."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert connector_device.connector_device_index(_config(), 1) == 1


def test_check_accepts_the_worker_device():
    connector_device.check_connector_device(
        SimpleNamespace(local_rank=1),
        torch.device("cuda", 1),
    )


def test_check_rejects_another_device():
    """A connector on another GPU than the worker's fails with both devices.

    Why: a vLLM upgrade that changes device selection must fail at startup
    instead of moving traffic onto another rank's GPU.
    """
    with pytest.raises(RuntimeError, match="uses cuda:0 but vLLM placed"):
        connector_device.check_connector_device(
            SimpleNamespace(local_rank=0),
            torch.device("cuda", 4),
        )


def _role_rank_config(*, tp_size: int):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tp_size,
        ),
    )


def test_role_rank_check_accepts_the_distributed_role_rank(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank", lambda: 1
    )
    connector = SimpleNamespace(
        role_rank=1,
        afd_config=AFDConfig(role="attention", num_attention_ranks=2),
    )

    connector_device.check_connector_role_rank(connector, _role_rank_config(tp_size=2))


def test_role_rank_check_rejects_a_different_role_rank(monkeypatch):
    """A connector whose early role rank differs from vLLM's fails startup.

    Why: the TP part of the role rank is derived from the worker rank before
    vLLM's TP group exists; a vLLM change to its rank order must fail here
    instead of routing another rank's tokens.
    """
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank", lambda: 0
    )
    connector = SimpleNamespace(
        role_rank=1,
        afd_config=AFDConfig(role="attention", num_attention_ranks=2),
    )

    with pytest.raises(RuntimeError, match="has role rank 1 but vLLM"):
        connector_device.check_connector_role_rank(
            connector, _role_rank_config(tp_size=2)
        )
