# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Routed-only parameter ownership composed with the mature Async CAM runtime.

Imported after the Ascend OPP bootstrap, inside the externally launched F process.
No Worker, upstream ModelRunner, EngineCore, scheduler or KV cache is constructed.
"""

from __future__ import annotations

from argparse import Namespace

import torch
import torch_npu  # noqa: F401
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

from afd_plugin.compat.npu import (
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
)
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.ffn_faas.platforms.npu.environment import AscendFFNEnvironment
from afd_plugin.ffn_faas.platforms.npu.model import CAMRoutedModel
from afd_plugin.ffn_faas.platforms.npu.settings import FFNConfig
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.validation import validate_npu_model_runner_v2_config


class CAMFFNBackend:
    """Keep weights and communication owners until receiver exit is proven."""

    def __init__(
        self, model: CAMRoutedModel, settings: FFNConfig, args: Namespace
    ) -> None:
        self.settings = settings
        self.args = args
        self.environment = AscendFFNEnvironment(settings)
        self.model = model
        self.runtime: AsyncNPUFFNModelRunner | None = None
        self.connector: CAMAsyncAFDConnector | None = None

    def prepare(self, service: FFNService) -> None:
        args = EngineArgs.from_cli_args(self.args)
        config = args.create_engine_config()
        # Keep the externally launched F-only ownership after platform defaults.
        config.parallel_config.distributed_executor_backend = "external_launcher"
        fix_all2all_backend_for_afd(config)
        validate_npu_model_runner_v2_config(
            config, expected_role="ffn", device_type="npu"
        )
        fail_if_unsupported_npu_afd_features(config)
        if config.speculative_config is not None:
            raise ValueError("standalone V4 loads main-model routed layers only")
        self.environment.initialize_device(config)
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
                local_rank=self.environment.device.index,
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
            # Reuse paired signatures, warmup, capture and Graph pools from
            # the existing CAM implementation.
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
        if self.runtime.effective_mode == "eager":
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
