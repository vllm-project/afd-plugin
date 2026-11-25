# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend/V4 CLI adaptation; importing this module remains CPU-only."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.config import AFD_ASYNC_CONNECTOR
from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.launch import (
    configure_ffn_engine_args,
    create_ffn_engine_parser,
)
from afd_plugin.ffn_faas.platforms.npu import get_model
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend
    from afd_plugin.ffn_faas.platforms.npu.model import CAMRoutedModel


def create_backend(
    model: CAMRoutedModel, settings: FFNConfig, args: Namespace
) -> FFNBackend:
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    from afd_plugin.ffn_faas.platforms.npu.backend import CAMFFNBackend

    return CAMFFNBackend(model, settings, args)


class AscendCAMAdapter:
    """Keep model, quantization, topology and device options out of the host loop."""

    def create_parser(self) -> ArgumentParser:
        return create_ffn_engine_parser()

    def configure(self, args: Namespace) -> FFNLaunch:
        _, launcher = configure_ffn_engine_args(args, connector=AFD_ASYNC_CONNECTOR)
        settings = FFNConfig(
            model=Path(args.model),
            ffn_ranks=launcher.world_size,
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            shutdown_timeout=args.receiver_shutdown_timeout,
        )
        model = get_model(settings.model, settings.ffn_ranks, settings.rank)
        return FFNLaunch(
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            model_path=str(model.root),
            backend_name="CAMAsyncAFDConnector",
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, model, settings, args),
            initial_status={"cam_paired": False},
        )
