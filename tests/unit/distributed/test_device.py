# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Worker device resolution matches ``Worker.init_device`` before it runs."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from afd_plugin.distributed import (  # noqa: E402
    register_assigned_physical_gpu_ids,
    resolve_worker_device_index,
)
from tests.unit.fakes.platform import use_cuda_like_platform  # noqa: E402
from tests.unit.fakes.vllm_config import gpu_worker_vllm_config  # noqa: E402


@pytest.fixture(autouse=True)
def _cuda_like_platform(monkeypatch):
    use_cuda_like_platform(monkeypatch)


def _resolve(config, local_rank):
    register_assigned_physical_gpu_ids(config)
    return resolve_worker_device_index(config, local_rank)


@pytest.mark.parametrize(
    ("executor_backend", "expected_index"),
    [("mp", 1), ("ray", 0), ("external_launcher", 0)],
)
def test_adds_local_dp_offset_like_vllm(executor_backend, expected_index):
    """The local DP rank offsets the device, except where vLLM skips it.

    Why: vLLM adds the offset only inside init_device, after the connector
    joined the AFD world; without it every DP rank's connector used GPU 0.
    """
    config = gpu_worker_vllm_config(
        data_parallel_rank_local=1,
        data_parallel_index=1,
        distributed_executor_backend=executor_backend,
    )

    assert _resolve(config, 0) == expected_index


def test_maps_assigned_device_ids_without_visible_devices():
    """``--device-ids 4,5`` without CUDA_VISIBLE_DEVICES places rank 0 on cuda:4.

    Why: vLLM maps its logical index through the assigned physical ids, so a
    connector using the logical index would run on another rank's GPU.
    """
    config = gpu_worker_vllm_config(assigned_physical_gpu_ids=[4, 5])

    assert _resolve(config, 0) == 4
    assert _resolve(config, 1) == 5


def test_maps_assigned_device_ids_through_visible_devices(monkeypatch):
    """With CUDA_VISIBLE_DEVICES the assigned ids map to visible ordinals."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    config = gpu_worker_vllm_config(assigned_physical_gpu_ids=[5, 4])

    assert _resolve(config, 0) == 1
    assert _resolve(config, 1) == 0


def test_without_assigned_ids_the_index_is_the_local_rank(monkeypatch):
    """The default mp launch keeps the logical index as the device index."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert _resolve(gpu_worker_vllm_config(), 1) == 1


def test_resolving_before_registering_assigned_ids_fails():
    """Resolving with ``--device-ids`` set but not registered raises.

    Why: vLLM would map the index without those ids and place the connector
    on another rank's GPU; the side effect must be explicit and come first.
    """
    config = gpu_worker_vllm_config(assigned_physical_gpu_ids=[4, 5])

    with pytest.raises(RuntimeError, match="register_assigned_physical_gpu_ids"):
        resolve_worker_device_index(config, 0)
