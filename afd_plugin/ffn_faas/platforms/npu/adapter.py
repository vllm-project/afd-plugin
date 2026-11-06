# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only Ascend/V4 launch adaptation; native imports happen on preparation."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.launch import (
    read_ffn_launcher_environment,
    require_v2_model_runner,
)
from afd_plugin.ffn_faas.platforms.npu import get_model
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend
    from afd_plugin.ffn_faas.platforms.npu.model import CAMRoutedModel


def create_backend(model: CAMRoutedModel, settings: FFNConfig) -> FFNBackend:
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    from afd_plugin.ffn_faas.platforms.npu.backend import CAMFFNBackend

    return CAMFFNBackend(model, settings)


class AscendCAMAdapter:
    """Keep model, quantization, topology and device options out of the host loop."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=29600)
        parser.add_argument("--attention-ranks", type=int, required=True)
        parser.add_argument("--attention-tp", type=int, required=True)
        parser.add_argument("--ffn-ranks", type=int, required=True)
        parser.add_argument("--devices", help="allocated logical NPU IDs on this node")
        parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
        parser.add_argument("--max-num-seqs", type=int, default=256)
        parser.add_argument("--full-graph", action="store_true")
        parser.add_argument("--hccl-buffer-size", type=int)

    def configure(self, args: Namespace) -> FFNLaunch:
        launcher = read_ffn_launcher_environment(args.ffn_ranks)
        require_v2_model_runner()
        local_world = launcher.local_world_size
        devices = (
            tuple(map(int, args.devices.split(",")))
            if args.devices
            else tuple(range(local_world))
        )
        if (
            not 0 <= launcher.local_rank < len(devices)
            or len(set(devices)) != len(devices)
            or min(devices) < 0
        ):
            raise ValueError("--devices must supply distinct local NPU IDs")
        if os.environ.get("AFD_ASYNC_CAM_LAYERED_GMM") != "1":
            raise ValueError("standalone V4 requires AFD_ASYNC_CAM_LAYERED_GMM=1")
        settings = FFNConfig(
            model=args.model,
            attention_ranks=args.attention_ranks,
            attention_tp=args.attention_tp,
            ffn_ranks=args.ffn_ranks,
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            device=devices[launcher.local_rank],
            host=args.host,
            port=args.port,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            full_graph=args.full_graph,
            hccl_buffer_size=args.hccl_buffer_size,
            shutdown_timeout=args.shutdown_timeout,
        )
        model = get_model(args.model, settings.ffn_ranks, settings.rank)
        return FFNLaunch(
            rank=launcher.rank,
            local_rank=launcher.local_rank,
            model_path=str(model.root),
            backend_name="CAMAsyncAFDConnector",
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, model, settings),
            initial_status={"cam_paired": False},
        )
