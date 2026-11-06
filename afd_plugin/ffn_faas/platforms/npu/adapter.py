# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend/V4 CLI adaptation; importing this module remains CPU-only."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config
from afd_plugin.ffn_faas.interfaces import FFNLaunch
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
        # Selected-platform imports: the Service/host entry stays CPU-only.
        from vllm.engine.arg_utils import EngineArgs
        from vllm.plugins import load_general_plugins
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        load_general_plugins()
        parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
        # This process is already launched by torchrun, not a vLLM executor.
        parser.set_defaults(distributed_executor_backend="external_launcher")
        return parser

    def configure(self, args: Namespace) -> FFNLaunch:
        if args.model_tag is not None:
            # Match vLLM serve's positional-model precedence.
            args.model = args.model_tag
        afd = parse_afd_config(args.additional_config, expected_role="ffn")
        if afd.connector != AFD_ASYNC_CONNECTOR:
            raise ValueError("standalone Ascend FFN requires CAMAsyncAFDConnector")
        rank, local_rank, world = (
            int(os.environ.get(key, default))
            for key, default in (
                ("RANK", "0"),
                ("LOCAL_RANK", "0"),
                ("WORLD_SIZE", "1"),
            )
        )
        if world != afd.num_ffn_ranks or world != args.data_parallel_size:
            raise ValueError(
                "F-only WORLD_SIZE, --data-parallel-size and "
                "additional_config.afd.num_ffn_ranks must agree"
            )
        if args.data_parallel_rank is not None and args.data_parallel_rank != rank:
            raise ValueError("--data-parallel-rank must match the F-only launcher RANK")
        args.data_parallel_rank = rank
        if args.distributed_executor_backend != "external_launcher":
            raise ValueError("standalone FFN requires the external_launcher backend")
        settings = FFNConfig(
            model=Path(args.model),
            ffn_ranks=world,
            rank=rank,
            local_rank=local_rank,
            shutdown_timeout=args.receiver_shutdown_timeout,
        )
        model = get_model(settings.model, settings.ffn_ranks, settings.rank)
        return FFNLaunch(
            rank=rank,
            local_rank=local_rank,
            model_path=str(model.root),
            backend_name="CAMAsyncAFDConnector",
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, model, settings, args),
            initial_status={"cam_paired": False},
        )
