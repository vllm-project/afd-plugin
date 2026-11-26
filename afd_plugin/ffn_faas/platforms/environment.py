# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared invariants for standalone FFN device and compute environments."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    import torch
    from vllm.distributed.parallel_state import GroupCoordinator


class FFNEnvironmentSettings(Protocol):
    """The standalone-environment subset shared by GPU and NPU settings."""

    @property
    def ffn_ranks(self) -> int: ...

    @property
    def rank(self) -> int: ...


FFNEnvironmentSettingsT = TypeVar(
    "FFNEnvironmentSettingsT", bound=FFNEnvironmentSettings
)


class FFNEnvironmentBase(ABC, Generic[FFNEnvironmentSettingsT]):
    """Own common F-only environment state without owning native setup."""

    def __init__(
        self, settings: FFNEnvironmentSettingsT, *, platform_name: str
    ) -> None:
        self.settings = settings
        self._platform_name = platform_name
        self._device: torch.device | None = None
        self._ep_group: GroupCoordinator | None = None
        self._groups_owned = False

    @property
    def device(self) -> torch.device:
        """This rank's device, available after ``initialize_device``."""
        if self._device is None:
            raise RuntimeError(
                f"{self._platform_name} FFN device has not been initialized"
            )
        return self._device

    @property
    def ep_group(self) -> GroupCoordinator:
        """The F-only expert-parallel group, after ``initialize_compute``."""
        if self._ep_group is None:
            raise RuntimeError(
                f"{self._platform_name} FFN expert group has not been initialized"
            )
        return self._ep_group

    def require_uninitialized(self) -> None:
        """Reject a second device initialization."""
        if self._groups_owned or self._device is not None:
            raise RuntimeError(
                f"{self._platform_name} FFN environment is already initialized"
            )

    def require_launcher_environment(self) -> None:
        """Require the launcher world and rank to still match the settings."""
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"{self._platform_name} FFN requires an external F-only launcher"
            ) from error
        if world_size != self.settings.ffn_ranks or rank != self.settings.rank:
            raise ValueError(
                f"{self._platform_name} FFN environment differs from the "
                "F-only launcher"
            )

    def require_no_borrowed_native_groups(self) -> None:
        """Refuse process groups this standalone F process does not own."""
        import torch
        from vllm.distributed.parallel_state import model_parallel_is_initialized

        if torch.distributed.is_initialized() or model_parallel_is_initialized():
            raise RuntimeError(
                f"{self._platform_name} FFN cannot borrow existing native groups"
            )

    def require_compute_initializable(self) -> None:
        """Require device selection and reject repeated compute initialization."""
        if self._device is None or self._groups_owned:
            raise RuntimeError(
                f"{self._platform_name} FFN compute initialization requires its device"
            )

    def mark_groups_owned(self) -> None:
        """Record ownership before the first collective can partially fail."""
        self._groups_owned = True

    def validate_ep_group(self, *, error_message: str) -> None:
        """Require the expert group to span exactly the F-only world."""
        ep_group = self.ep_group
        if (
            ep_group.world_size != self.settings.ffn_ranks
            or ep_group.rank_in_group != self.settings.rank
        ):
            raise ValueError(error_message)

    def close(self) -> None:
        """Destroy owned groups, then clear environment ownership state."""
        if not self._groups_owned:
            return
        self._destroy_groups()
        self._groups_owned = False
        self._ep_group = None
        self._device = None

    @abstractmethod
    def _destroy_groups(self) -> None:
        """Destroy this platform's model-parallel and process groups."""
