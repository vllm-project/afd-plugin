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
) -> Iterator[ForwardContext]:
    """Create the minimal forward context needed by connector-driven FFN steps."""

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    if aclgraph_runtime_mode is None:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    if vllm_config.use_v2_model_runner:
        from vllm.forward_context import set_forward_context
        from vllm_ascend.ascend_forward_context import override_mrv2_in_profile_run

        # MRv2 stores Ascend-specific state in ForwardContext.additional_kwargs.
        # Using the legacy context manager here writes attributes directly on
        # ForwardContext, while vllm-ascend's _EXTRA_CTX proxy reads only the
        # additional mapping in MRv2. Scope the upstream MRv2 profile marker so
        # its platform hook installs balanced-MoE and communication state in the
        # same layout used by the native runner.
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
            yield forward_context
        return

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    with set_ascend_forward_context(
        None,
        vllm_config,
        batch_descriptor=None,
        aclgraph_runtime_mode=aclgraph_runtime_mode,
        model_instance=model_instance,
        num_tokens=int(num_tokens),
        num_tokens_across_dp=num_tokens_across_dp,
        in_profile_run=in_profile_run,
    ):
        forward_context = get_forward_context()
        if skip_mc2_mask:
            forward_context.mc2_mask = None
        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        forward_context.additional_kwargs["afd_metadata"] = afd_metadata
        yield forward_context


__all__ = ["ascend_forward_context"]
