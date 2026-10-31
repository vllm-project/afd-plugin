# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ascend forward-context helpers for connector-driven AFD steps."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm.forward_context import ForwardContext

    from afd_plugin.connectors import AFDForwardContextMetadata


@contextmanager
def ascend_forward_context(
    *,
    vllm_config: VllmConfig,
    afd_metadata: AFDForwardContextMetadata,
    model_instance: torch.nn.Module | None = None,
    input_ids: torch.Tensor | None = None,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    aclgraph_runtime_mode: CUDAGraphMode | None = None,
    is_draft_model: bool = False,
) -> Iterator[ForwardContext]:
    """Create the minimal forward context needed by connector-driven FFN steps."""

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    if aclgraph_runtime_mode is None:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    with set_ascend_forward_context(
        None,
        vllm_config,
        batch_descriptor=None,
        aclgraph_runtime_mode=aclgraph_runtime_mode,
        model_instance=model_instance,
        num_tokens=int(num_tokens),
        num_tokens_across_dp=num_tokens_across_dp,
        input_ids=input_ids,
        is_draft_model=is_draft_model,
    ):
        forward_context = get_forward_context()
        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs["afd_metadata"] = afd_metadata
        yield forward_context


__all__ = ["ascend_forward_context"]
