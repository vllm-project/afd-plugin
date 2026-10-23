# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention-side connector for the independent native FFN service."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.metadata import AFDA2FTransferPayload, AFDTransferContext
from afd_plugin.ffn_faas.client import AttentionSession
from afd_plugin.ffn_faas.npu import (
    NativeDeviceError,
    NPUExecutor,
    SessionPeerFailureError,
    UnsafeSessionCleanupError,
)
from afd_plugin.ffn_faas.settings import FaaSOptions
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class FaaSExtraInfo(FaaSOptions, ConnectorExtraInfo):
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> FaaSExtraInfo:
        return cls(**FaaSOptions.from_mapping(raw).to_mapping())


class FaaSAsyncAFDConnector(AFDConnectorBase):
    """Use the existing model's router/shared MLP with a different transport."""

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> FaaSExtraInfo:
        return FaaSExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        if not afd_config.is_attention_server:
            raise ValueError("FFN FaaS ranks use python -m afd_plugin.ffn_faas serve")
        self.attn_size, self.ffn_size = (
            afd_config.num_attention_ranks,
            afd_config.num_ffn_ranks,
        )
        self._session: AttentionSession | None = None
        self._executor: NPUExecutor | None = None
        self._pending_context: AFDTransferContext | None = None
        if vllm_config.model_config.hf_config.model_type != "deepseek_v4":
            raise ValueError("FFN FaaS currently requires DeepSeek-V4 W4A8")

    @property
    def is_initialized(self) -> bool:
        return self._session is not None

    def init_afd_connector(self) -> None:
        if self.is_initialized:
            raise RuntimeError("FFN FaaS connector is already initialized")
        extra = self.extra_info
        assert isinstance(extra, FaaSExtraInfo)
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        if extra.attn_ranks_per_dp != tp:
            raise ValueError("attn_ranks_per_dp must equal Attention TP")
        checkpoint = V4CheckpointDescription.read(self.vllm_config.model_config.model)
        spec = checkpoint.session_spec(
            attention_ranks=self.attn_size,
            attention_tp=tp,
            ffn_ranks=self.ffn_size,
            capacity=extra.capacity,
            slots=extra.slots,
        )
        self._executor = NPUExecutor(
            spec,
            rank=self.role_rank,
            device=self.local_rank,
            host=self.afd_config.host,
            port=self.afd_config.port,
            timeout=extra.timeout_s,
            startup_timeout=extra.startup_timeout_s,
            window_mb=extra.hccl_buffer_size,
            auto_initialize=False,
        )
        try:
            self._executor.initialize()
            session = AttentionSession(self._executor, tp_rows_aligned=True)
            session.warmup()
            self._session = session
        except BaseException as error:
            self._executor.publish_failure(error)
            try:
                self._executor.close(abort=True)
            except UnsafeSessionCleanupError as cleanup_error:
                self._exit_unsafe_worker(cleanup_error)
            raise

    def _exit_unsafe_worker(self, error: UnsafeSessionCleanupError) -> None:
        # This connector lives in a task-owned vLLM model worker. Preserve self
        # and its windows through exit rather than entering upstream destroy-all
        # group/model cleanup with possibly active remote accesses.
        assert self._executor is not None
        self._executor.publish_failure(error)
        print(
            f"FFN FaaS worker exiting without unsafe destructors: {error}",
            file=sys.stderr,
            flush=True,
        )
        sys.stdout.flush()
        os._exit(1)

    def _abort_transport_failure(self, error: BaseException) -> None:
        # Run before the exception reaches vLLM's shutdown, whose first action
        # is a whole-device synchronize. An incomplete native event must reach
        # this bounded close/exit boundary before that upstream synchronization.
        assert self._executor is not None
        self._executor.publish_failure(error)
        try:
            self._executor.close(abort=True)
        except UnsafeSessionCleanupError as cleanup_error:
            self._exit_unsafe_worker(cleanup_error)
        self._session = None
        self._executor = None
        self._pending_context = None

    @contextmanager
    def model_execution_scope(self) -> Iterator[None]:
        if self._session is None:
            raise RuntimeError("FFN FaaS connector has not completed warmup")
        try:
            with self._session.model_execution_scope():
                yield
        except BaseException as error:
            if self._executor is not None:
                self._abort_transport_failure(error)
            # A downstream Attention/KV consumer may have run before the status
            # check. Discard this worker even if communication quiesced safely;
            # raising to a caller that could retry the model is insufficient.
            print(
                f"FFN FaaS model scope failed; discarding worker state: {error}",
                file=sys.stderr,
                flush=True,
            )
            sys.stdout.flush()
            os._exit(1)

    def select_experts(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # The V4 proxy owns Hash/sqrtsoftplus and the single probability scale.
        raise RuntimeError("V4 FaaS requires the native Hash/sqrtsoftplus router")

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        if self._session is None:
            raise RuntimeError("FFN FaaS connector has not completed warmup")
        if self._pending_context is not None:
            raise RuntimeError("FFN FaaS v1 supports one outstanding model call")
        # V4 routing already includes its scale; keep probabilities in FP32.
        probabilities = kwargs["topk_weights"].float()
        try:
            self._session.send(
                hidden_states,
                kwargs["topk_ids"],
                probabilities,
                layer=context.metadata.layer_idx,
            )
        except (SessionPeerFailureError, NativeDeviceError, TimeoutError) as error:
            self._abort_transport_failure(error)
            raise
        self._pending_context = context

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._session is None or self._pending_context is None:
            raise RuntimeError("no FFN FaaS call is pending")
        if kwargs["context"] is not self._pending_context:
            raise ValueError("FFN result belongs to a different transfer context")
        try:
            output = self._session.receive()
        except (SessionPeerFailureError, NativeDeviceError, TimeoutError) as error:
            self._abort_transport_failure(error)
            raise
        self._pending_context = None
        return output

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        raise RuntimeError("FFN uses the standalone service's executor")

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        raise RuntimeError("FFN result publication belongs to the standalone executor")

    def close(self) -> None:
        try:
            if self._session is not None:
                self._session.close(abort=self._pending_context is not None)
                self._session = None
            elif self._executor is not None:
                self._executor.close(abort=True)
        except UnsafeSessionCleanupError as error:
            self._exit_unsafe_worker(error)
        self._executor = None
