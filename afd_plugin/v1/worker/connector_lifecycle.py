# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector lifecycle for GPU workers whose connector joins before vLLM.

Every connector follows one lifecycle: create, then ``init_afd_connector``
once the runner has loaded the model. A routed-experts connector
(``AFDRoutedExpertsConnector``) also has a step that must run before vLLM's
distributed init: it joins the AFD world and applies its NCCL environment,
which NCCL reads once per process, when the first communicator is created.
vLLM creates its communicators in ``Worker.init_device`` before it builds the
runner, so the worker owns that step:

1. ``create_worker_connector`` creates the connector and runs
   ``prepare_before_distributed_init``.
2. The worker hands that same instance to the runner it constructs.
3. ``check_worker_connector`` compares what was derived before vLLM's init
   (device, TP rank) with what vLLM then chose.
4. The runner calls ``init_afd_connector`` after loading the model.

Other connectors are created by their runner, as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDConnectorFactory,
    AFDRoutedExpertsConnector,
)
from afd_plugin.distributed import (
    register_assigned_physical_gpu_ids,
    resolve_role_rank,
    resolve_worker_device_index,
    tp_rank_before_distributed_init,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig


def create_worker_connector(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    *,
    rank: int,
    local_rank: int,
) -> AFDRoutedExpertsConnector | None:
    """Create a connector that must join the AFD world before vLLM's init.

    Call before vLLM's distributed init. Returns ``None`` for connectors that
    are not ``AFDRoutedExpertsConnector``; their runner creates them.
    Otherwise the connector gets the device vLLM will select and a role rank
    whose TP part comes from the worker ``rank`` (vLLM's TP group does not
    exist yet), and has run ``prepare_before_distributed_init``. The caller
    owns it until it hands it to the runner.

    Side effect: registers vLLM's assigned physical GPU ids
    (``register_assigned_physical_gpu_ids``).
    """
    connector_cls = AFDConnectorFactory.get_connector_class(afd_config.connector)
    if not issubclass(connector_cls, AFDRoutedExpertsConnector):
        return None
    register_assigned_physical_gpu_ids(vllm_config)
    connector = AFDConnectorFactory.create_connector(
        rank,
        resolve_worker_device_index(vllm_config, local_rank),
        vllm_config,
        afd_config,
        tp_rank=tp_rank_before_distributed_init(vllm_config, rank),
    )
    assert isinstance(connector, AFDRoutedExpertsConnector)
    try:
        connector.prepare_before_distributed_init()
    except BaseException:
        connector.close()
        raise
    return connector


def check_worker_connector(
    connector: AFDConnectorBase,
    vllm_config: VllmConfig,
    worker_device: torch.device,
) -> None:
    """Fail when a connector created before vLLM's init disagrees with vLLM.

    Call after vLLM's distributed init. The connector must use the device vLLM
    gave the worker, and its role rank must match vLLM's distributed ranks, so
    a vLLM upgrade that changes device selection or rank order fails at
    startup instead of moving traffic onto another rank's GPU or tokens.
    """
    if connector.local_rank != worker_device.index:
        raise RuntimeError(
            f"{type(connector).__name__} uses cuda:{connector.local_rank} but "
            f"vLLM placed this worker on {worker_device}; the device selection "
            "in afd_plugin/distributed/device.py no longer matches vLLM's "
            "Worker.init_device",
        )
    role_rank = resolve_role_rank(vllm_config, connector.afd_config)
    if connector.role_rank != role_rank:
        raise RuntimeError(
            f"{type(connector).__name__} has role rank {connector.role_rank} but "
            f"vLLM's distributed ranks give {role_rank}; the TP rank derived "
            "before distributed init (tp_rank_before_distributed_init) no longer "
            "matches vLLM's rank layout",
        )


__all__ = ["check_worker_connector", "create_worker_connector"]
