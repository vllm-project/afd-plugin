# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared launch-time CLI and environment checks for standalone FFN backends.

Importing this module stays CPU-only; ``create_ffn_engine_parser`` imports the
selected vLLM CLI layer only when the platform adapter builds its parser.
"""

from __future__ import annotations

import math
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig, parse_afd_config

EXTERNAL_LAUNCHER = "external_launcher"

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class FFNLauncherEnvironment:
    """One rank's externally launched F-only world coordinates."""

    rank: int
    local_rank: int
    world_size: int


def read_ffn_launcher_environment() -> FFNLauncherEnvironment:
    """Read the ``torchrun`` coordinates of this F-only rank."""
    rank, local_rank, world_size = (
        int(os.environ.get(key, default))
        for key, default in (
            ("RANK", "0"),
            ("LOCAL_RANK", "0"),
            ("WORLD_SIZE", "1"),
        )
    )
    return FFNLauncherEnvironment(
        rank=rank, local_rank=local_rank, world_size=world_size
    )


def validate_ffn_rank_settings(
    *, ffn_ranks: int, rank: int, local_rank: int, shutdown_timeout: float
) -> None:
    """Validate the launcher and host-shutdown facts every backend keeps.

    Engine and connector definitions are validated by vLLM and AFD config.
    """
    if ffn_ranks <= 0 or not 0 <= rank < ffn_ranks:
        raise ValueError("invalid F-only rank or world size")
    if local_rank < 0:
        raise ValueError("local rank must be nonnegative")
    if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
        raise ValueError("shutdown timeout must be finite and positive")


def create_ffn_engine_parser() -> ArgumentParser:
    """Build vLLM's native engine CLI for an externally launched F-only rank."""
    # Selected-platform imports: the Service/host entry stays CPU-only.
    from vllm.engine.arg_utils import EngineArgs
    from vllm.plugins import load_general_plugins
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    load_general_plugins()
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    # This process is already launched by torchrun, not a vLLM executor.
    parser.set_defaults(distributed_executor_backend=EXTERNAL_LAUNCHER)
    return parser


def configure_ffn_engine_args(
    args: Namespace, *, connector: str
) -> tuple[AFDConfig, FFNLauncherEnvironment]:
    """Validate the native engine arguments against the F-only launcher.

    Applies vLLM serve's positional-model precedence, parses the FFN-role AFD
    config, requires ``connector``, and requires the launcher ``WORLD_SIZE``,
    ``--data-parallel-size`` and ``afd.num_ffn_ranks`` to agree. An omitted
    ``--data-parallel-rank`` is taken from ``RANK``. Mutates ``args`` so
    ``EngineArgs.from_cli_args`` sees the resolved model and DP rank.
    """
    if args.model_tag is not None:
        # Match vLLM serve's positional-model precedence.
        args.model = args.model_tag
    afd = parse_afd_config(args.additional_config, expected_role="ffn")
    if afd.connector != connector:
        raise ValueError(f"this standalone FFN backend requires {connector}")
    launcher = read_ffn_launcher_environment()
    if (
        launcher.world_size != afd.num_ffn_ranks
        or launcher.world_size != args.data_parallel_size
    ):
        raise ValueError(
            "F-only WORLD_SIZE, --data-parallel-size and "
            "additional_config.afd.num_ffn_ranks must agree"
        )
    if args.data_parallel_rank is not None and args.data_parallel_rank != launcher.rank:
        raise ValueError("--data-parallel-rank must match the F-only launcher RANK")
    args.data_parallel_rank = launcher.rank
    if args.distributed_executor_backend != EXTERNAL_LAUNCHER:
        raise ValueError("standalone FFN requires the external_launcher backend")
    return afd, launcher


def reassert_external_launcher(config: VllmConfig) -> None:
    """Keep an already launched F-only world from borrowing an executor."""
    config.parallel_config.distributed_executor_backend = EXTERNAL_LAUNCHER
