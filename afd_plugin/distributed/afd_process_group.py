# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD distributed helpers for P2P topology and process-group setup."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
from torch.distributed import Backend
from torch.distributed.distributed_c10d import (
    PrefixStore,
    ProcessGroup,
    Store,
    _new_process_group_helper,
    _update_default_pg,
    _world,
)
from torch.distributed.rendezvous import rendezvous
from vllm.distributed import parallel_state
from vllm.utils.torch_utils import is_torch_equal_or_newer


class ProcessGroupRendezvousContext:
    """Borrow the Store and process group created by one CAM rendezvous."""

    def __init__(self) -> None:
        self._store: Store | None = None
        self._process_group: ProcessGroup | None = None
        self._closed = False

    def retain_store(self, store: Store) -> None:
        if self._closed or self._store is not None:
            raise RuntimeError("CAM rendezvous context cannot accept another Store")
        self._store = store

    def bind(self, process_group: ProcessGroup) -> None:
        if self._closed or self._store is None or self._process_group is not None:
            raise RuntimeError("CAM rendezvous context is not ready to bind")
        self._process_group = process_group

    def borrow(self) -> tuple[Store, ProcessGroup]:
        if self._closed or self._store is None or self._process_group is None:
            raise RuntimeError("CAM rendezvous context is not bound")
        return self._store, self._process_group

    def invalidate(self) -> None:
        self._closed = True
        self._store = None
        self._process_group = None


class DefaultProcessGroupSwitcher:
    """Temporarily switch PyTorch's default process group."""

    def __init__(
        self,
        default_group: ProcessGroup,
        new_default_group: ProcessGroup,
    ) -> None:
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self) -> None:
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type: object, exc_value: object, tb: object) -> None:
        _update_default_pg(self.default_group)


def create_hccl_process_group_options(
    hccl_buffer_size_mb: int | None,
) -> Any | None:
    """Create fresh HCCL options for one plugin-owned process group.

    Returning ``None`` preserves torch-npu's environment-variable and built-in
    fallback. A fresh options object keeps a configured MB value local to one
    connector-owned HCCL process group.
    """
    if hccl_buffer_size_mb is None:
        return None

    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size_mb}
    return options


def init_afd_process_group(
    *,
    backend: str,
    init_method: str,
    world_size: int,
    rank: int,
    group_name: str,
    timeout: timedelta,
    pg_options: Any | None = None,
    on_rendezvous: Callable[[Store], None] | None = None,
) -> ProcessGroup:
    """Create a plugin-owned process group without patching vLLM source.

    The helper keeps process-group setup isolated in the plugin. It relies on
    PyTorch/vLLM private APIs and fails fast if the target runtime stack is
    unavailable.
    """

    rendezvous_iterator = rendezvous(
        init_method,
        rank,
        world_size,
        timeout=timeout,
    )
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)
    if on_rendezvous is not None:
        on_rendezvous(store)
    prefixed_store = PrefixStore(group_name, store)
    backend_value = Backend(backend) if backend else Backend("undefined")
    pg_options_param_name = (
        "backend_options" if is_torch_equal_or_newer("2.6.0") else "pg_options"
    )

    process_group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend_value,
        prefixed_store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    group_ranks = {i: i for i in range(world_size)}
    _world.pg_group_ranks[process_group] = group_ranks

    try:
        world = parallel_state.get_world_group()
        world.pg_group_ranks[process_group] = group_ranks
    except Exception:
        if torch.distributed.is_initialized():
            default_group = torch.distributed.distributed_c10d._get_default_group()
            pg_group_ranks = getattr(default_group, "pg_group_ranks", None)
            if pg_group_ranks is not None:
                pg_group_ranks[process_group] = group_ranks

    return process_group


__all__ = [
    "DefaultProcessGroupSwitcher",
    "ProcessGroupRendezvousContext",
    "create_hccl_process_group_options",
    "init_afd_process_group",
]
