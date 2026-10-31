# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only boundaries between the service loop and a selected FFN platform."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

    from afd_plugin.ffn_faas.service import FFNService


class FFNBackend(Protocol):
    """One rank's resources; failures retain the owner until process isolation."""

    def prepare(self, service: FFNService) -> None: ...

    def start_receiver(self) -> None: ...

    def execute_step(self) -> None: ...

    def receiver_exited(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FFNLaunch:
    """Platform-independent information needed before starting the host loop."""

    rank: int
    local_rank: int
    model_path: str
    backend_name: str
    shutdown_timeout: float
    backend_factory: Callable[[], FFNBackend]
    initial_status: Mapping[str, str | int | bool] = field(default_factory=dict)


class FFNPlatformAdapter(Protocol):
    """Register platform options and validate a CPU-only launch description."""

    def add_arguments(self, parser: ArgumentParser) -> None: ...

    def configure(self, args: Namespace) -> FFNLaunch: ...
