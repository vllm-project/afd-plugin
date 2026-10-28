# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Decode benchmark KV connector for AFD decode-side pressure tests.

Two modes, chosen by ``kv_connector_extra_config``:

- Dummy KV (``num_prompts`` unset or 0): every prompt token but the last is
  reported as externally computed and never written, so requests skip prefill.
- Prompt pool (``num_prompts`` > 0): each Attention DP rank prefills a pool of
  ``num_prompts`` seeded random prompts for real, from its first
  ``num_prompts`` requests (the warmup), and pins their KV blocks in the prefix
  cache. Every later request is remapped, whatever its content, onto a pool
  prompt round-robin, hits the prefix cache for all but the last prompt token,
  and goes straight to decode over real KV.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# vLLM configures handlers only for the "vllm" logger tree; a logger outside it
# falls back to the root WARNING level and drops the prompt-pool INFO lines.
logger = init_logger(f"vllm.{__name__}")

KVCacheLayer = torch.Tensor | tuple[torch.Tensor, ...]

# A warmup request only has to prefill its pool prompt; one output token also
# fixes how many blocks each pinned pool entry holds.
WARMUP_OUTPUT_TOKENS = 1
# Finished measured requests between two prompt-pool hit/miss log lines.
POOL_STATS_LOG_INTERVAL = 1000
# Warmup requests that end this way have fully prefilled their pool prompt.
PINNABLE_WARMUP_STATUSES = (
    RequestStatus.FINISHED_STOPPED,
    RequestStatus.FINISHED_LENGTH_CAPPED,
)


@dataclass(frozen=True)
class PromptPoolConfig:
    """Prompt-pool keys of ``kv_connector_extra_config``.

    Attributes:
        num_prompts: Pool entries per Attention DP rank; 0 keeps the dummy-KV
            mode.
        prompt_len: Requested pool prompt length. It is rounded down to one
            token past a block boundary, where a prefix-cache hit leaves exactly
            the last token to prefill.
        seed: Seed of the pool tokens, shared by every rank so all ranks hold
            the same pool.
        max_output_len: Longest output a measured request decodes; sizes the KV
            capacity check. ``None`` uses ``max_model_len - prompt_len``.
    """

    num_prompts: int = 0
    prompt_len: int = 1025
    seed: int = 0
    max_output_len: int | None = None


@dataclass
class AFDDecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Requests whose allocated KV blocks should be filled with dummy values."""

    # request_id -> (block_ids_per_group, num_tokens_to_fill)
    reqs_to_fill: dict[str, tuple[tuple[list[int], ...], int]]


class AFDDecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """KV connector used to benchmark AFD decode instances with long ISL.

    In the dummy-KV mode the connector emulates remote prefill by marking all
    prompt tokens except the final decode token as externally provided. In the
    prompt-pool mode only a prefix-cache miss is emulated that way; see the
    module docstring.
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
            self.connector_scheduler = AFDDecodeBenchConnectorScheduler(
                vllm_config,
                kv_cache_config,
            )
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

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        super().bind_kv_cache_manager(kv_cache_manager)
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_kv_cache_manager(kv_cache_manager)

    def on_new_request(self, request: Request) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

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
        delay_free_blocks = self.connector_scheduler.request_finished(
            request,
            len(block_ids),
        )
        return delay_free_blocks, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        delay_free_blocks = self.connector_scheduler.request_finished(
            request,
            sum(len(group_block_ids) for group_block_ids in block_ids),
        )
        return delay_free_blocks, None


class AFDDecodeBenchConnectorScheduler:
    """Scheduler-side implementation for AFDDecodeBenchConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self._filled_requests: set[str] = set()
        self._pending_fills: dict[str, tuple[tuple[list[int], ...], int]] = {}

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        pool_config = PromptPoolConfig(
            num_prompts=kv_transfer_config.get_from_extra_config(
                "num_prompts", PromptPoolConfig.num_prompts
            ),
            prompt_len=kv_transfer_config.get_from_extra_config(
                "prompt_len", PromptPoolConfig.prompt_len
            ),
            seed=kv_transfer_config.get_from_extra_config(
                "seed", PromptPoolConfig.seed
            ),
            max_output_len=kv_transfer_config.get_from_extra_config(
                "max_output_len", PromptPoolConfig.max_output_len
            ),
        )
        self.prompt_pool = (
            PromptPool(pool_config, vllm_config, kv_cache_config)
            if pool_config.num_prompts > 0
            else None
        )

    def bind_kv_cache_manager(self, kv_cache_manager: KVCacheManager) -> None:
        if self.prompt_pool is not None:
            self.prompt_pool.check_capacity(
                kv_cache_manager.block_pool.get_num_free_blocks()
            )

    def on_new_request(self, request: Request) -> None:
        if self.prompt_pool is not None:
            self.prompt_pool.assign(request)

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.prompt_pool is not None:
            num_tokens_to_fill = self.prompt_pool.num_tokens_to_fill(
                request,
                num_computed_tokens,
            )
            return num_tokens_to_fill, False

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

    def request_finished(self, request: Request, num_blocks: int) -> bool:
        """Return True to keep the request's blocks allocated (pinned)."""
        self._filled_requests.discard(request.request_id)
        if self.prompt_pool is None:
            return False
        return self.prompt_pool.request_finished(request, num_blocks)


class PromptPool:
    """Per-rank pool of real-KV prompts that measured requests decode over.

    Pinning: a warmup request that prefilled its pool prompt reports
    ``delay_free_blocks`` from ``request_finished`` and is never returned from
    ``get_finished``, so the scheduler keeps its blocks allocated for the
    server's lifetime. Pinned blocks never enter the free queue, so LRU
    eviction cannot drop a pool entry, and they stay in the prefix cache, so a
    measured request that maps to the entry shares them like any prefix hit.
    """

    def __init__(
        self,
        config: PromptPoolConfig,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
    ) -> None:
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "AFDDecodeBenchConnector prompt pool requires prefix caching; "
                "launch with --enable-prefix-caching"
            )
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError(
                "AFDDecodeBenchConnector prompt pool supports a single KV cache "
                f"group, got {len(kv_cache_config.kv_cache_groups)}"
            )

        self.block_size = vllm_config.cache_config.block_size
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        self.num_prompts = config.num_prompts
        # A prefix-cache hit covers full blocks only and stops one token short
        # of the prompt, so a prompt one token past a block boundary hits every
        # token but the last.
        self.prompt_len = (
            config.prompt_len - 1
        ) // self.block_size * self.block_size + 1
        if self.prompt_len == 1:
            raise ValueError(
                f"prompt_len ({config.prompt_len}) must exceed the KV block size "
                f"({self.block_size})"
            )
        self.max_output_len = (
            config.max_output_len
            if config.max_output_len is not None
            else max_model_len - self.prompt_len
        )
        if self.max_output_len < 1 or (
            self.prompt_len + self.max_output_len > max_model_len
        ):
            raise ValueError(
                f"prompt_len ({self.prompt_len}) + max_output_len "
                f"({self.max_output_len}) must fit max_model_len ({max_model_len})"
            )

        # A CPU generator gives every rank the same pool for the same seed.
        generator = torch.Generator().manual_seed(config.seed)
        self.prompt_tokens = torch.randint(
            vllm_config.model_config.get_vocab_size(),
            (self.num_prompts, self.prompt_len),
            generator=generator,
            dtype=torch.int32,
        )
        # Hashing a long prompt per request is avoidable work on the scheduler:
        # an entry's hashes are computed once and copied after that.
        self._entry_block_hashes: dict[int, list[BlockHash]] = {}
        self._unwarmed_entries = deque(range(self.num_prompts))
        self._warming_entries: dict[str, int] = {}
        self._num_pinned_entries = 0
        self._num_pinned_blocks = 0
        self._num_assigned_measured = 0
        # request_id -> prompt tokens the prefix cache missed (0 on a hit).
        self._measured_misses: dict[str, int] = {}
        self._num_finished_measured = 0
        self._num_missed_requests = 0
        self._num_missed_tokens = 0

    def check_capacity(self, num_free_blocks: int) -> None:
        """Fail fast when the pool and the running requests cannot both fit."""
        blocks_per_entry = cdiv(
            self.prompt_len + WARMUP_OUTPUT_TOKENS,
            self.block_size,
        )
        # A measured request shares its entry's full blocks and allocates only
        # the last prompt token and its output.
        blocks_per_seq = cdiv(1 + self.max_output_len, self.block_size)
        num_reserved_blocks = self.max_num_seqs * blocks_per_seq
        num_pool_blocks = self.num_prompts * blocks_per_entry
        if num_pool_blocks + num_reserved_blocks > num_free_blocks:
            max_num_prompts = max(
                0,
                (num_free_blocks - num_reserved_blocks) // blocks_per_entry,
            )
            raise ValueError(
                f"AFDDecodeBenchConnector prompt pool needs {num_pool_blocks} KV "
                f"blocks for {self.num_prompts} prompts of {self.prompt_len} "
                f"tokens plus {num_reserved_blocks} for {self.max_num_seqs} "
                f"sequences of up to {self.max_output_len} output tokens, but "
                f"only {num_free_blocks} are free; use num_prompts <= "
                f"{max_num_prompts} or shorten prompt_len/max_output_len"
            )
        if self.num_prompts < self.max_num_seqs:
            logger.warning(
                "AFDDecodeBenchConnector: %d pool prompts are fewer than "
                "max_num_seqs (%d); concurrent requests will share KV blocks",
                self.num_prompts,
                self.max_num_seqs,
            )
        logger.info(
            "AFDDecodeBenchConnector prompt pool: %d prompts x %d tokens use "
            "%d of %d free KV blocks; %d reserved for running requests",
            self.num_prompts,
            self.prompt_len,
            num_pool_blocks,
            num_free_blocks,
            num_reserved_blocks,
        )

    def assign(self, request: Request) -> None:
        """Rewrite a new request's prompt to a pool entry's, before scheduling.

        The first ``num_prompts`` requests warm one unwarmed entry each; every
        later request maps round-robin onto the pool, whatever it asked for.
        """
        if self._unwarmed_entries:
            entry = self._unwarmed_entries.popleft()
            self._warming_entries[request.request_id] = entry
            request.max_tokens = WARMUP_OUTPUT_TOKENS
        else:
            entry = self._num_assigned_measured % self.num_prompts
            self._num_assigned_measured += 1

        prompt_token_ids = self.prompt_tokens[entry].tolist()
        request.prompt_token_ids = prompt_token_ids
        request.num_prompt_tokens = len(prompt_token_ids)
        # all_token_ids is a read-only view of _all_token_ids: replace in place.
        request._all_token_ids[:] = prompt_token_ids
        request.block_hashes.clear()
        entry_block_hashes = self._entry_block_hashes.get(entry)
        if entry_block_hashes is None:
            request.update_block_hashes()
            self._entry_block_hashes[entry] = list(request.block_hashes)
        else:
            request.block_hashes.extend(entry_block_hashes)

    def num_tokens_to_fill(self, request: Request, num_computed_tokens: int) -> int:
        """Prompt tokens to fake as external: 0 on a hit or during warmup."""
        if request.request_id in self._warming_entries:
            return 0
        # A pinned entry hits every prompt token but the last. Anything the
        # cache misses is faked so the step stays decode-only, and counted.
        num_missed_tokens = max(0, request.num_tokens - 1 - num_computed_tokens)
        self._measured_misses[request.request_id] = num_missed_tokens
        return num_missed_tokens

    def request_finished(self, request: Request, num_blocks: int) -> bool:
        """Pin a finished warmup request's blocks; tally measured hits."""
        entry = self._warming_entries.pop(request.request_id, None)
        if entry is not None:
            if request.status not in PINNABLE_WARMUP_STATUSES:
                # Aborted before its prompt was cached: warm it again later.
                self._unwarmed_entries.appendleft(entry)
                return False
            self._num_pinned_entries += 1
            self._num_pinned_blocks += num_blocks
            if self._num_pinned_entries == self.num_prompts:
                logger.info(
                    "AFDDecodeBenchConnector prompt pool warm: %d prompts "
                    "pinned in %d KV blocks",
                    self._num_pinned_entries,
                    self._num_pinned_blocks,
                )
            return True

        num_missed_tokens = self._measured_misses.pop(request.request_id, None)
        if num_missed_tokens is None:
            return False
        self._num_finished_measured += 1
        if num_missed_tokens > 0:
            self._num_missed_requests += 1
            self._num_missed_tokens += num_missed_tokens
        if self._num_finished_measured % POOL_STATS_LOG_INTERVAL == 0:
            logger.info(
                "AFDDecodeBenchConnector prompt pool: %d measured requests, "
                "%d prefix-cache misses (%d prompt tokens faked)",
                self._num_finished_measured,
                self._num_missed_requests,
                self._num_missed_tokens,
            )
        return False


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
    "PromptPool",
    "PromptPoolConfig",
]
