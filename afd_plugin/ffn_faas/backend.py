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
from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.ffn_faas.settings import FFNConfig
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription
from afd_plugin.ffn_faas.v4_runtime import V4FFNModel
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner


class CAMFFNBackend:
    """Keep weights and communication owners until receiver exit is proven."""

    def __init__(
        self, checkpoint: V4CheckpointDescription, settings: FFNConfig
    ) -> None:
        self.settings = settings
        self.device = torch.device(f"npu:{settings.device}")
        self.model = V4FFNModel(checkpoint, settings.ffn_ranks, settings.rank)
        self.runtime: AsyncNPUFFNModelRunner | None = None
        self.connector: CAMAsyncAFDConnector | None = None

    def prepare(self, service: FFNService) -> None:
        # Install the package-local OPP path before native packing initializes
        # GE, and fail before loading parameters if this build is unavailable.
        ensure_cam_async_ops_available()
        torch.npu.set_device(self.device)
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
        with set_current_vllm_config(config):
            self.model.load(
                self.settings.device, config, progress=service.loading_progress
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
            self.runtime = AsyncNPUFFNModelRunner(config, self.device, self.connector)
            self.runtime.attach_model(
                self.model, parameter_bytes=self.model.parameter_bytes
            )
            # Reuse paired signatures, warmup, capture, Graph pools and two-slot
            # ready/done/FIFO ownership from the existing CAM implementation.
            self.runtime.initialize_afd_connector()

    def start_receiver(self) -> None:
        """Select the device once on the receiving thread, as the Worker does."""
        torch.npu.set_device(self.device)

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
        self.model.close()
