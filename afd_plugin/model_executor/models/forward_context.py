# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Helpers for plugin-owned model wrappers to read AFD forward metadata."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import vllm.forward_context as forward_context_module
from vllm.forward_context import ForwardContext, get_forward_context

from afd_plugin.connectors import AFDForwardContextMetadata


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
    "get_afd_metadata_from_forward_context",
    "use_afd_metadata_provider",
]
