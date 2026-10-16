# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-specific NPU tensor layouts around native MRV2 execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from afd_plugin.compat.npu.runtime_config import npu_model_uses_sharded_pp_tensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.ubatch_utils import UBatchSlices


def pp_transport_rows(
    vllm_config: VllmConfig,
    num_tokens: int,
    ubatch_slices: UBatchSlices | None = None,
) -> int:
    """Count physical rows, including independent padding in each stage.

    DSV4 gathers before a PP boundary. DSV2 disables MoE SP with PP, so its
    PP boundary is global too. Only an effective sharded layout divides by TP.
    """
    sizes = (
        [stage.num_tokens for stage in ubatch_slices]
        if ubatch_slices is not None
        else [num_tokens]
    )
    if npu_model_uses_sharded_pp_tensors(vllm_config):
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return sum((size + tp_size - 1) // tp_size for size in sizes)
    return sum(sizes)


def pp_transport_slice(vllm_config: VllmConfig, token_slice: slice) -> slice:
    if not npu_model_uses_sharded_pp_tensors(vllm_config):
        return token_slice
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    start = (token_slice.start + tp_size - 1) // tp_size
    rows = (token_slice.stop - token_slice.start + tp_size - 1) // tp_size
    return slice(start, start + rows)


def prepare_pp_transport_buffers(
    buffers: dict[str, torch.Tensor],
    incoming: dict[str, torch.Tensor],
    capacity: int,
) -> dict[str, torch.Tensor]:
    """Preserve every transported field before native MRV2 copies its inputs.

    Normal buffers already have scheduler capacity and keep their addresses.
    Allocate a missing auxiliary field with its own trailing shape and dtype;
    preserve existing contents when a smaller buffer needs to grow.
    """
    result = dict(buffers)
    for key, source in (buffers | incoming).items():
        rows = max(capacity, source.shape[0])
        previous = buffers.get(key)
        if previous is None or previous.shape[0] < rows:
            target = source.new_empty((rows, *source.shape[1:]))
            if previous is not None:
                target[: previous.shape[0]].copy_(previous)
            result[key] = target
    return result


def cat_ubatch_outputs(
    outputs: list[torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Keep Ascend's nested auxiliary-output contract when native threads merge."""
    first = outputs[0]
    if isinstance(first, tuple):
        pairs = [output for output in outputs if isinstance(output, tuple)]
        assert len(pairs) == len(outputs)
        assert all(len(pair[1]) == len(first[1]) for pair in pairs)
        return (
            torch.cat([pair[0] for pair in pairs], dim=0),
            [
                torch.cat([pair[1][index] for pair in pairs], dim=0)
                for index in range(len(first[1]))
            ],
        )
    tensors = [output for output in outputs if isinstance(output, torch.Tensor)]
    assert len(tensors) == len(outputs)
    return torch.cat(tensors, dim=0)
