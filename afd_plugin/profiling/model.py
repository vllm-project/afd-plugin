# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connect model execution and Attention batch metadata to stream tracing.

Runner/model hooks cache tracing.py tracks for each forward and DBO child.
CPU counts annotate dispatch without device reads; connector hooks mark phase
changes and the runner marks WAIT only at step boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.v1.worker.ubatch_utils import UBatchSlices

    from afd_plugin.connectors import AFDConnectorBase, AFDForwardContextMetadata
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.profiling.tracing import Execution


def transition(execution: Execution | None, phase: str, annotation: str = "") -> None:
    if execution is not None:
        if annotation:
            execution.tracer.transition(execution, phase, annotation)
        else:
            execution.tracer.transition(execution, phase)


def set_attention_profile_batch(
    metadata: AFDForwardContextMetadata,
    num_requests: int,
    num_tokens: int,
    ubatch_slices: UBatchSlices | None,
) -> None:
    """Copy only CPU scalar counts, excluding native request/token padding."""
    if ubatch_slices and len(ubatch_slices) > 1:
        metadata.profile_batch = [
            (
                index,
                max(
                    0, min(ub.request_slice.stop, num_requests) - ub.request_slice.start
                )
                if ub.token_slice.start < num_tokens
                else 0,
                max(0, min(ub.token_slice.stop, num_tokens) - ub.token_slice.start),
            )
            for index, ub in enumerate(ubatch_slices)
        ]
    else:
        metadata.profile_batch = [(0, num_requests if num_tokens else 0, num_tokens)]


def attention_profile_annotation(
    metadata: AFDForwardContextMetadata, layer_idx: int
) -> str:
    if not metadata.profile_batch:
        return ""
    index = metadata.stage_idx if len(metadata.profile_batch) > 1 else 0
    microbatch, requests, tokens = metadata.profile_batch[index]
    return f"{microbatch}|{layer_idx}|{requests}|{tokens}"


def begin_profile_call(metadata: AFDForwardContextMetadata) -> Execution | None:
    execution = metadata.profile_pipeline
    if execution is not None and execution.tracer.retain_execution(execution):
        return execution
    return None


def begin_profile_step(connector: AFDConnectorBase) -> Execution | None:
    """Keep inter-layer compute contiguous; idle only at the model-step boundary."""
    session = cast("CAMAsyncAFDConnector", connector).profile_session
    if session is None or session.tracer is None:
        return None
    execution = session.tracer.execution()
    if execution is not None:
        execution.tracer.transition(execution, "COMPUTE")
    return execution


def install_pipeline_hook(
    module: Module,
    metadata_provider: Callable[[], AFDForwardContextMetadata | None],
) -> None:
    """Initialize once per forward, including each native DBO child stream."""

    def begin(model_module, args) -> None:
        metadata = metadata_provider()
        if metadata is None:
            return
        metadata.profile_pipeline = begin_profile_step(metadata.connector)

    # vLLM's compiled model __call__ bypasses model hooks, even in eager mode.
    module.layers[module.start_layer].register_forward_pre_hook(begin)
