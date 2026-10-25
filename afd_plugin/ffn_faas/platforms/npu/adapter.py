# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only Ascend/V4 launch adaptation; native imports happen on preparation."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig
from afd_plugin.ffn_faas.platforms.npu.v4_model import V4CheckpointDescription

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend


def create_backend(
    checkpoint: V4CheckpointDescription, settings: FFNConfig
) -> FFNBackend:
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    from afd_plugin.ffn_faas.platforms.npu.backend import CAMFFNBackend

    return CAMFFNBackend(checkpoint, settings)


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
        parser.add_argument("--ffn-two-stream", action="store_true")
        parser.add_argument("--ffn-compute-aic-num", type=int)
        parser.add_argument("--ffn-compute-aiv-num", type=int)
        parser.add_argument("--ffn-combine-aiv-num", type=int)
        parser.add_argument("--hccl-buffer-size", type=int)

    def configure(self, args: Namespace) -> FFNLaunch:
        rank, local_rank, world = (
            int(os.environ.get(key, default))
            for key, default in (
                ("RANK", "0"),
                ("LOCAL_RANK", "0"),
                ("WORLD_SIZE", "1"),
            )
        )
        if world != args.ffn_ranks:
            raise ValueError("F-only launcher WORLD_SIZE must equal --ffn-ranks")
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
        devices = (
            tuple(map(int, args.devices.split(",")))
            if args.devices
            else tuple(range(local_world))
        )
        if (
            not 0 <= local_rank < len(devices)
            or len(set(devices)) != len(devices)
            or min(devices) < 0
        ):
            raise ValueError("--devices must supply distinct local NPU IDs")
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
            raise ValueError("set VLLM_USE_V2_MODEL_RUNNER=1 on both A and F")
        if os.environ.get("AFD_ASYNC_CAM_LAYERED_GMM") != "1":
            raise ValueError("standalone V4 requires AFD_ASYNC_CAM_LAYERED_GMM=1")
        settings = FFNConfig(
            model=args.model,
            attention_ranks=args.attention_ranks,
            attention_tp=args.attention_tp,
            ffn_ranks=args.ffn_ranks,
            rank=rank,
            local_rank=local_rank,
            device=devices[local_rank],
            host=args.host,
            port=args.port,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            full_graph=args.full_graph,
            ffn_two_stream=args.ffn_two_stream,
            ffn_compute_aic_num=args.ffn_compute_aic_num,
            ffn_compute_aiv_num=args.ffn_compute_aiv_num,
            ffn_combine_aiv_num=args.ffn_combine_aiv_num,
            hccl_buffer_size=args.hccl_buffer_size,
            shutdown_timeout=args.shutdown_timeout,
        )
        checkpoint = V4CheckpointDescription.read(args.model)
        return FFNLaunch(
            rank=rank,
            local_rank=local_rank,
            model_path=str(checkpoint.root),
            backend_name="CAMAsyncAFDConnector",
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, checkpoint, settings),
            initial_status={"cam_paired": False},
        )
