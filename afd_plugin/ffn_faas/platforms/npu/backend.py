# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Routed-only parameter ownership composed with the mature Async CAM runtime.

Imported after the Ascend OPP bootstrap, inside the externally launched F process.
No Worker, upstream ModelRunner, EngineCore, scheduler or KV cache is constructed.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

from afd_plugin.compat.npu import fix_all2all_backend_for_afd
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.ffn_faas.platforms.npu.environment import AscendFFNEnvironment
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig
from afd_plugin.ffn_faas.platforms.npu.v4_model import V4CheckpointDescription
from afd_plugin.ffn_faas.platforms.npu.v4_runtime import V4FFNModel
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner


class CAMFFNBackend:
    """Keep weights and communication owners until receiver exit is proven."""

    def __init__(
        self, checkpoint: V4CheckpointDescription, settings: FFNConfig
    ) -> None:
        self.settings = settings
        self.environment = AscendFFNEnvironment(settings)
        self.model = V4FFNModel(checkpoint, settings.ffn_ranks, settings.rank)
        self.runtime: AsyncNPUFFNModelRunner | None = None
        self.connector: CAMAsyncAFDConnector | None = None

    def prepare(self, service: FFNService) -> None:
        self.environment.initialize_device()
        args = EngineArgs(
            model=str(self.settings.model),
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            enforce_eager=not self.settings.full_graph,
            compilation_config={"cudagraph_mode": "FULL"}
            if self.settings.full_graph
            else {},
            tensor_parallel_size=1,
            data_parallel_size=self.settings.ffn_ranks,
            data_parallel_rank=self.settings.rank,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            # F owns a dispatch envelope, not a request-context/KV capacity.
            max_model_len=self.settings.max_num_batched_tokens,
            max_num_batched_tokens=self.settings.max_num_batched_tokens,
            max_num_seqs=self.settings.max_num_seqs,
            additional_config=self.settings.additional_config(),
        )
        config = args.create_engine_config()
        # Platform configuration can choose an executor for ordinary Workers;
        # this loader belongs to the already launched F-only torchrun world.
        config.parallel_config.distributed_executor_backend = "external_launcher"
        fix_all2all_backend_for_afd(config)
        self.environment.initialize_compute(config)
        with set_current_vllm_config(config):
            self.model.load(
                self.environment.device, config, progress=service.loading_progress
            )
            service.phase(
                ServicePhase.CONNECTING,
                model_loaded=True,
                parameter_bytes=self.model.parameter_bytes,
            )
            self.connector = CAMAsyncAFDConnector(
                rank=self.settings.rank,
                local_rank=self.settings.device,
                vllm_config=config,
                afd_config=parse_afd_config(config, expected_role="ffn"),
                role_rank=self.settings.rank,
            )
            self.runtime = AsyncNPUFFNModelRunner(
                config, self.environment.device, self.connector
            )
            self.runtime.attach_model(
                self.model, parameter_bytes=self.model.parameter_bytes
            )
            # Reuse paired signatures, warmup, capture, Graph pools and two-slot
            # ready/done/FIFO ownership from the existing CAM implementation.
            self.runtime.initialize_afd_connector()
            self.environment.bind_host_threads(config)
            service.phase(
                ServicePhase.CONNECTING,
                cam_paired=True,
                compute_ep_world_size=self.environment.ep_group.world_size,
            )

    def start_receiver(self) -> None:
        """Select the device once on the receiving thread, as the Worker does."""
        self.environment.activate_thread()

    def execute_step(self) -> None:
        assert self.runtime is not None
        self.runtime.execute_connector_driven_step()
        if not self.settings.full_graph and not self.settings.ffn_two_stream:
            # Preserve the mature eager loop's completion fence.
            torch.npu.synchronize()

    def receiver_exited(self) -> None:
        assert self.runtime is not None
        self.runtime.mark_receiver_exited()

    def close(self) -> None:
        assert self.runtime is not None
        # The runtime rejects pending graphs before closing CAM. Only then may
        # its borrowed parameter views and the original owner be released.
        self.runtime.shutdown()
        self.environment.close()
        self.model.close()
