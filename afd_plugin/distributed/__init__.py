# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD distributed helper namespace."""

from afd_plugin.distributed.device import (
    register_assigned_physical_gpu_ids,
    resolve_worker_device_index,
)
from afd_plugin.distributed.topology import (
    AFDRankMapping,
    build_rank_mapping,
    resolve_role_rank,
    subgroup_attention_block,
    topology_from_config,
    tp_rank_before_distributed_init,
    validate_p2p_topology,
)


def __getattr__(name: str):
    if name in {
        "DefaultProcessGroupSwitcher",
        "create_hccl_process_group_options",
        "init_afd_process_group",
    }:
        from afd_plugin.distributed import afd_process_group

        value = getattr(afd_process_group, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AFDRankMapping",
    "DefaultProcessGroupSwitcher",
    "build_rank_mapping",
    "create_hccl_process_group_options",
    "init_afd_process_group",
    "register_assigned_physical_gpu_ids",
    "resolve_role_rank",
    "resolve_worker_device_index",
    "subgroup_attention_block",
    "topology_from_config",
    "tp_rank_before_distributed_init",
    "validate_p2p_topology",
]
