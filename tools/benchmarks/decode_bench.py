# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Decode benchmark KV connector for AFD decode-side pressure tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

KVCacheLayer: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


@dataclass
class AFDDecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Requests whose allocated KV blocks should be filled with dummy values."""

    # request_id -> (block_ids_per_group, num_tokens_to_fill)
    reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]]


class AFDDecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """KV connector used to benchmark AFD decode instances with long ISL.

    The connector emulates remote prefill by marking all prompt tokens except
    the final decode token as externally provided, then fills the allocated KV
    cache pages with deterministic or random non-zero values on the worker.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        self.connector_scheduler: AFDDecodeBenchConnectorScheduler | None = None
        self.connector_worker: AFDDecodeBenchConnectorWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = AFDDecodeBenchConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = AFDDecodeBenchConnectorWorker(
                vllm_config,
                kv_cache_config,
            )

    def register_kv_caches(self, kv_caches: dict[str, KVCacheLayer]) -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        # Keep the original AFD decode-bench behavior: only emulate external KV
        # availability on the scheduler side, without writing dummy values into
        # the worker KV cache.
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(
            request,
            blocks,
            num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class AFDDecodeBenchConnectorScheduler:
    """Scheduler-side implementation for AFDDecodeBenchConnector."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._filled_requests: set[str] = set()
        self._pending_fills: dict[str, tuple[tuple[list[int], ...], int]] = {}

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        req_id = request.request_id
        allow_refill_after_preempt = (
            request.status == RequestStatus.PREEMPTED
            and request.num_preemptions > 0
            and num_computed_tokens == 0
        )

        if req_id in self._filled_requests and not allow_refill_after_preempt:
            return 0, False

        num_uncomputed_tokens = request.num_tokens - num_computed_tokens
        num_tokens_to_fill = max(0, num_uncomputed_tokens - 1)
        if num_tokens_to_fill == 0:
            return 0, False
        return num_tokens_to_fill, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        num_blocks_to_fill = cdiv(num_external_tokens, self.block_size)
        block_ids_per_group = tuple(
            group_blocks[:num_blocks_to_fill] for group_blocks in block_groups
        )

        req_id = request.request_id
        self._pending_fills[req_id] = (block_ids_per_group, num_external_tokens)
        self._filled_requests.add(req_id)

        logger.debug(
            "AFDDecodeBenchConnector: allocated %d blocks across %d KV cache "
            "groups for request %s",
            num_blocks_to_fill,
            len(block_groups),
            req_id,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = AFDDecodeBenchConnectorMetadata(
            reqs_to_fill=self._pending_fills.copy(),
        )
        self._pending_fills.clear()
        return meta

    def request_finished(self, request: Request) -> None:
        self._filled_requests.discard(request.request_id)


class AFDDecodeBenchConnectorWorker:
    """Worker-side implementation for AFDDecodeBenchConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.block_size = vllm_config.cache_config.block_size

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.fill_mean = kv_transfer_config.get_from_extra_config("fill_mean", 0.015)
        self.fill_std = kv_transfer_config.get_from_extra_config("fill_std", 0.0)

        self.kv_caches: dict[str, KVCacheLayer] | None = None
        self.group_to_layers: dict[int, list[str]] | None = None

    def register_kv_caches(self, kv_caches: dict[str, KVCacheLayer]) -> None:
        self.kv_caches = kv_caches
        self.group_to_layers = self._build_group_to_layers(kv_caches)

        logger.debug(
            "AFDDecodeBenchConnector: registered %d KV cache layers across %d groups",
            len(kv_caches),
            len(self.group_to_layers),
        )

    def start_fill_kv(self, metadata: AFDDecodeBenchConnectorMetadata) -> None:
        if not metadata.reqs_to_fill:
            return

        assert self.kv_caches is not None, "KV caches must be registered first"
        assert self.group_to_layers is not None, "KV group mapping is not initialized"

        for req_id, (block_ids_per_group, num_tokens) in metadata.reqs_to_fill.items():
            for group_idx, block_ids in enumerate(block_ids_per_group):
                self._fill_blocks(group_idx, block_ids)

            logger.debug(
                "AFDDecodeBenchConnector: filled %d blocks (%d tokens) across "
                "%d groups for request %s",
                len(block_ids_per_group[0]) if block_ids_per_group else 0,
                num_tokens,
                len(block_ids_per_group),
                req_id,
            )

    def _build_group_to_layers(
        self,
        kv_caches: dict[str, KVCacheLayer],
    ) -> dict[int, list[str]]:
        if self.kv_cache_config is None:
            return {0: list(kv_caches.keys())}

        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        if not kv_cache_groups:
            return {0: list(kv_caches.keys())}

        return {
            group_idx: [
                layer_name
                for layer_name in group.layer_names
                if layer_name in kv_caches
            ]
            for group_idx, group in enumerate(kv_cache_groups)
        }

    def _fill_blocks(self, group_idx: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        assert self.kv_caches is not None
        assert self.group_to_layers is not None

        for layer_name in self.group_to_layers[group_idx]:
            kv_cache = self.kv_caches[layer_name]
            caches_to_fill = kv_cache if isinstance(kv_cache, tuple) else (kv_cache,)

            for cache_tensor in caches_to_fill:
                block_ids_tensor = torch.tensor(
                    block_ids,
                    dtype=torch.long,
                    device=cache_tensor.device,
                )
                valid_block_ids = block_ids_tensor[
                    block_ids_tensor < cache_tensor.shape[0]
                ]
                if len(valid_block_ids) == 0:
                    continue

                fill_shape = (len(valid_block_ids),) + cache_tensor.shape[1:]
                if self.fill_std > 0:
                    fill_values = torch.normal(
                        mean=self.fill_mean,
                        std=self.fill_std,
                        size=fill_shape,
                        dtype=cache_tensor.dtype,
                        device=cache_tensor.device,
                    )
                else:
                    fill_values = torch.full(
                        fill_shape,
                        self.fill_mean,
                        dtype=cache_tensor.dtype,
                        device=cache_tensor.device,
                    )
                cache_tensor[valid_block_ids] = fill_values

        logger.debug(
            "AFDDecodeBenchConnector: filled %d blocks in group %d with %s "
            "values (mean=%.3f, std=%.3f)",
            len(block_ids),
            group_idx,
            "random" if self.fill_std > 0 else "constant",
            self.fill_mean,
            self.fill_std,
        )


__all__ = [
    "AFDDecodeBenchConnector",
    "AFDDecodeBenchConnectorMetadata",
    "AFDDecodeBenchConnectorScheduler",
    "AFDDecodeBenchConnectorWorker",
]
