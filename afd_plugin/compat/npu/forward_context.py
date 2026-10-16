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
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    in_profile_run: bool = False,
    aclgraph_runtime_mode: CUDAGraphMode | None = None,
    skip_mc2_mask: bool = False,
    input_ids: torch.Tensor | None = None,
) -> Iterator[ForwardContext]:
    """Create the minimal forward context needed by connector-driven FFN steps.

    ``input_ids`` carries the token ids of exactly the tokens this FFN rank
    computes on. Models whose router is keyed by token identity read it from the
    forward context, so a connector that transports ids must install them here
    before running the FFN compute.
    """

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    if aclgraph_runtime_mode is None:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    from vllm.forward_context import set_forward_context
    from vllm_ascend.ascend_forward_context import override_mrv2_in_profile_run

    # Ascend MRV2 reads its state from additional_kwargs. FFN supplies exactly
    # this native context layout, including the thread-local profile marker.
    with (
        override_mrv2_in_profile_run(in_profile_run),
        set_forward_context(
            None,
            vllm_config,
            num_tokens=int(num_tokens),
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=None,
        ),
    ):
        forward_context = get_forward_context()
        forward_context.additional_kwargs["afd_metadata"] = afd_metadata
        forward_context.additional_kwargs["model_instance"] = model_instance
        if skip_mc2_mask:
            forward_context.additional_kwargs["mc2_mask"] = None
        if input_ids is not None:
            forward_context.input_ids = input_ids
        yield forward_context


__all__ = ["ascend_forward_context"]
