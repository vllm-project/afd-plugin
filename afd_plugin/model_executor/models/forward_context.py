# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for plugin-owned model wrappers to read AFD forward metadata."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Final

import vllm.forward_context as forward_context_module
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backend import AttentionMetadata

from afd_plugin.async_moe import AsyncMoeStage
from afd_plugin.connectors import AFDForwardContextMetadata

ASYNC_MOE_UBATCH_METADATA_KEY: Final[str] = "afd_async_moe_ubatch_metadata"


@dataclass(frozen=True)
class AsyncMoeUbatchMetadata:
    """Execution plan for one two-stage Async CAM model forward.

    Each stage's token slice uses the parent real-token coordinate space, while
    ``stage.input_tokens`` includes its minimum physical padding. Under sequence
    parallelism each physical stage is divided evenly across TP ranks by the
    model-side layout helper. ``parent_input_tokens`` records the original
    padded layout restored after both stages.
    """

    attn_metadata: list[dict[str, AttentionMetadata]]
    stages: tuple[AsyncMoeStage, ...]
    parent_input_tokens: int
    use_sequence_parallel: bool

    def __post_init__(self) -> None:
        num_stages = len(self.stages)
        if not num_stages or len(self.attn_metadata) != num_stages:
            raise ValueError(
                "Async CAM execution-plan fields must describe the same "
                f"non-empty stage count: attention={len(self.attn_metadata)}, "
                f"slices={num_stages}",
            )

        expected_token_start = 0
        for stage_slice in self.stages:
            token_slice = stage_slice.token_slice
            token_start = int(token_slice.start)
            token_stop = int(token_slice.stop)
            actual_token_extent = token_stop - token_start
            input_tokens = int(stage_slice.input_tokens)
            if token_start != expected_token_start or actual_token_extent <= 0:
                raise ValueError(
                    "Async CAM stage token slices must be contiguous, ordered, "
                    f"and non-empty: token_slice={token_slice}, "
                    f"expected_start={expected_token_start}",
                )
            if not 0 < actual_token_extent <= input_tokens:
                raise ValueError(
                    "Async CAM stage actual-token count must fit its physical "
                    f"extent: actual={actual_token_extent}, input={input_tokens}",
                )
            expected_token_start = token_stop
        if self.parent_input_tokens < expected_token_start:
            raise ValueError(
                "Async CAM parent input extent must cover every real token: "
                f"parent_input_tokens={self.parent_input_tokens}, "
                f"actual_tokens={expected_token_start}",
            )


def get_afd_metadata_from_forward_context(
    forward_context: ForwardContext | None = None,
) -> AFDForwardContextMetadata | None:
    """Return AFD metadata from vLLM ``ForwardContext.additional_kwargs``.

    Model wrappers use this helper so AFD metadata stays outside vLLM's
    ``ForwardContext`` schema.
    """

    if forward_context is None:
        forward_context = get_forward_context()

    additional_kwargs = forward_context.additional_kwargs or {}
    # Keep the type refinement static: torch.compile traces this helper and
    # cannot wrap the runtime ``types.UnionType`` created by typing.cast.
    metadata: AFDForwardContextMetadata | None = additional_kwargs.get("afd_metadata")
    return metadata


def get_async_moe_ubatch_metadata_from_forward_context(
    forward_context: ForwardContext | None = None,
) -> AsyncMoeUbatchMetadata | None:
    """Return async MoE ubatch sidecar metadata from the current context."""

    if forward_context is None:
        forward_context = get_forward_context()

    additional_kwargs = forward_context.additional_kwargs or {}
    metadata: AsyncMoeUbatchMetadata | None = additional_kwargs.get(
        ASYNC_MOE_UBATCH_METADATA_KEY,
    )
    return metadata


@contextmanager
def use_afd_metadata_provider(provider: Any) -> Iterator[None]:
    """Install AFD metadata as vLLM creates a forward context.

    Native vLLM dummy runs call the model directly, bypassing
    ``AFDAttentionModelRunner._model_forward()``. Out-of-tree plugins cannot
    extend the ``set_forward_context()`` signature, so during dummy runs we
    temporarily wrap ``create_forward_context()`` and mutate
    ``additional_kwargs`` immediately after vLLM creates the context. Model code
    can then do a simple metadata read, which keeps ``torch.compile`` away from
    provider lookups.
    """

    original_create = forward_context_module.create_forward_context
    install = provider._install_afd_metadata_on_forward_context

    @wraps(original_create)
    def create_forward_context_with_afd(*args: Any, **kwargs: Any) -> ForwardContext:
        forward_context = original_create(*args, **kwargs)
        install(forward_context)
        return forward_context

    forward_context_module.create_forward_context = create_forward_context_with_afd
    try:
        yield
    finally:
        forward_context_module.create_forward_context = original_create


__all__ = [
    "ASYNC_MOE_UBATCH_METADATA_KEY",
    "AsyncMoeUbatchMetadata",
    "get_afd_metadata_from_forward_context",
    "get_async_moe_ubatch_metadata_from_forward_context",
    "use_afd_metadata_provider",
]
