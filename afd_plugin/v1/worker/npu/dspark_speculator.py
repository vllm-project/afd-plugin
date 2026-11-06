# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native DSpark proposal with an Attention runner's AFD context."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    prepare_dflash_inputs_factory,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

from afd_plugin.v1.worker.npu.dspark_graph import (
    use_afd_draft_capture,
    use_afd_draft_replay,
)

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


class AFDDSparkSpeculator(AscendDSparkSpeculator):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        afd_context: Callable[[bool, bool], AbstractContextManager[None]],
        afd_runner: AFDNPUAttentionModelRunnerV2,
    ) -> None:
        super().__init__(vllm_config, device)
        self.afd_context = afd_context
        self.afd_runner = afd_runner

    @contextmanager
    def _physical_input_preparation(self, *, enabled: bool = True) -> Iterator[None]:
        original_prepare = dflash_speculator.prepare_dflash_inputs
        if enabled:
            dflash_speculator.prepare_dflash_inputs = self._afd_prepare_inputs
        try:
            yield
        finally:
            dflash_speculator.prepare_dflash_inputs = original_prepare

    # Upstream: vLLM 5df63a72 DFlashSpeculator.capture, inherited by Ascend
    # bd42eea0 DSpark. Signature matches. Delegate native context/KV and Markov
    # capture, adding draft-local AFD metadata and the physical block size.
    # Remove when native capture exposes metadata and physical-size hooks.
    def capture(self) -> None:
        # ### PATCH START: scope native draft capture.
        assert self.query_cudagraph_manager is not None
        manager = cast(DFlashAclGraphManager, self.query_cudagraph_manager)
        with (
            self.afd_context(True, False),
            self._physical_input_preparation(),
            use_afd_draft_capture(self.afd_runner, manager),
        ):
            super().capture()
        # ### PATCH END: scope native draft capture.

    # Upstream: Ascend bd42eea0, AscendDSparkSpeculator.set_attn.
    # Its input factory captures the global cache_config.block_size, which can
    # differ from hybrid draft groups. Keep native initialization and replace
    # only this proposal's input factory with the allocated physical size.
    # Signature matches upstream. Remove after the native factory uses each
    # group's physical block size instead of the global scheduler setting.
    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        # ### PATCH START: retain the actual physical draft block size
        sizes = {block_tables.block_sizes[gid] for gid in self.draft_kv_cache_group_ids}
        if len(sizes) != 1:
            raise ValueError("AFD DSpark requires one physical draft KV block size")
        self._afd_prepare_inputs = prepare_dflash_inputs_factory(
            sizes.pop(), block_tables
        )
        # ### PATCH END: retain the actual physical draft block size

    # Upstream: Ascend bd42eea0, AscendDSparkSpeculator.propose.
    # Patch reason: native sample_tokens proposes after execute_model returns;
    # the target's AFD provider has already left scope. Draft query token counts
    # also differ from the target counts.
    # Patch functionality: install fresh AFD metadata and bypass cross-DP sync
    # for this proposal. All native KV, sampling and confidence work is delegated.
    # Signature matches upstream. Remove when native proposal accepts a plugin
    # context-manager hook.
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        # ### PATCH START: scope only the native draft proposal
        manager = self.query_cudagraph_manager
        # Native memory profiling runs before KV/graph-manager initialization.
        replay_scope = (
            use_afd_draft_replay(
                self.afd_runner,
                cast(DFlashAclGraphManager, manager),
                input_batch.num_reqs * self.num_query_per_req,
            )
            if manager is not None
            else nullcontext()
        )
        with (
            self.afd_context(dummy_run, is_profile),
            self._physical_input_preparation(
                enabled=not (dummy_run and skip_attn_for_dummy_run)
            ),
            replay_scope,
        ):
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                None,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )
        # ### PATCH END: scope only the native draft proposal
