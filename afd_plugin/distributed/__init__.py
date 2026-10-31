# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD distributed helper namespace."""

from afd_plugin.distributed.topology import (
    AFDRankMapping,
    AFDWindowExpertLayout,
    AFDWindowRankMapping,
    build_rank_mapping,
    build_window_expert_layout,
    build_window_rank_mapping,
    resolve_role_rank,
    topology_from_config,
    validate_p2p_topology,
)


def __getattr__(name: str):
    if name in {"DefaultProcessGroupSwitcher", "init_afd_process_group"}:
        from afd_plugin.distributed import afd_process_group

        value = getattr(afd_process_group, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AFDRankMapping",
    "AFDWindowExpertLayout",
    "AFDWindowRankMapping",
    "DefaultProcessGroupSwitcher",
    "build_rank_mapping",
    "build_window_expert_layout",
    "build_window_rank_mapping",
    "init_afd_process_group",
    "resolve_role_rank",
    "topology_from_config",
    "validate_p2p_topology",
]
