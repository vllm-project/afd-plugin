# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU async NVSHMEM CLI adaptation; importing this module remains CPU-only.

Imports neither torch, vLLM nor NVSHMEM at module import: ``create_parser``
loads vLLM's CLI layer when selected, and every check in ``configure`` runs
before a device or process group exists, so a bad launch exits with a usage
error and allocates nothing. Native imports happen in the backend factory.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from afd_plugin.config import AFD_ASYNC_GPU_CONNECTOR
from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.gpu.adapter import uses_cuda_graphs
from afd_plugin.ffn_faas.platforms.gpu.settings import (
    ASYNC_FFN_DRAIN_TIMEOUT_S,
    GPUAsyncFFNConfig,
)
from afd_plugin.ffn_faas.platforms.launch import (
    configure_ffn_engine_args,
    create_ffn_engine_parser,
)

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend

LINEAR_EXPERT_PLACEMENT = "linear"


def create_async_backend(settings: GPUAsyncFFNConfig, args: Namespace) -> FFNBackend:
    """Import the async GPU backend only when the host loop constructs it."""
    from afd_plugin.ffn_faas.platforms.gpu.async_backend import AsyncGPUFFNBackend

    return AsyncGPUFFNBackend(settings, args)


class GPUAsyncAdapter:
    """Keep async GPU, topology and drain defaults out of the host loop."""

    def create_parser(self) -> ArgumentParser:
        """Build vLLM's engine CLI with the standalone async GPU defaults."""
        parser = create_ffn_engine_parser()
        parser.set_defaults(
            skip_tokenizer_init=True,
            enable_ep_weight_filter=True,
            receiver_shutdown_timeout=ASYNC_FFN_DRAIN_TIMEOUT_S,
        )
        return parser

    def configure(self, args: Namespace) -> FFNLaunch:
        """Validate launch flags on CPU and describe the async GPU backend."""
        afd, launcher = configure_ffn_engine_args(
            args, connector=AFD_ASYNC_GPU_CONNECTOR
        )
        if args.tensor_parallel_size != 1 or not args.enable_expert_parallel:
            raise ValueError(
                "GPU standalone FFN requires --tensor-parallel-size 1 and "
                "--enable-expert-parallel"
            )
        if args.expert_placement_strategy != LINEAR_EXPERT_PLACEMENT:
            raise ValueError(
                f"{AFD_ASYNC_GPU_CONNECTOR} requires linear FFN expert placement"
            )
        if args.speculative_config is not None:
            raise ValueError(
                f"{AFD_ASYNC_GPU_CONNECTOR} does not support speculative decoding"
            )
        model_path = Path(args.model).resolve(strict=True)
        args.model = str(model_path)
        settings = GPUAsyncFFNConfig(
            model=model_path,
            ffn_ranks=launcher.world_size,
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            graph=uses_cuda_graphs(args),
            dbo=bool(args.enable_dbo),
            connector_extra_config=MappingProxyType(dict(afd.connector_extra_config)),
            shutdown_timeout=args.receiver_shutdown_timeout,
        )
        return FFNLaunch(
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            model_path=str(model_path),
            backend_name=AFD_ASYNC_GPU_CONNECTOR,
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_async_backend, settings, args),
            initial_status={
                "graph_mode": settings.graph_mode,
                "dbo": settings.dbo,
                "num_stages": settings.num_stages,
                "async_paired": False,
            },
        )
