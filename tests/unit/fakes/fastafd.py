# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FastAFD connector and custom-op doubles for model tests on CPU tensors."""

from types import SimpleNamespace

import torch

from afd_plugin.connectors.gpu.fastafd import FastAFDAFDConnector
from afd_plugin.model_executor.models import fastafd_moe


def recording_connector(
    events: list[object],
    payloads: list[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
) -> FastAFDAFDConnector:
    """A connector that records ``(op, lane)`` and the routing payload it gets.

    ``combine`` returns twice the hidden states it was given.
    """
    connector = object.__new__(FastAFDAFDConnector)

    def record(op: str, ids: torch.Tensor, weights: torch.Tensor, lane: int) -> None:
        events.append((op, lane))
        if payloads is not None:
            payloads.append((op, ids, weights))

    def dispatch(hidden_states, ids, weights, lane):
        record("dispatch", ids, weights, lane)

    def combine(hidden_states, ids, weights, lane):
        record("combine", ids, weights, lane)
        return hidden_states * 2

    connector.dispatch = dispatch  # type: ignore[method-assign]
    connector.combine = combine  # type: ignore[method-assign]
    return connector


def custom_ops_torch() -> SimpleNamespace:
    """Stand-in for a model module's ``torch`` that runs the FastAFD op bodies.

    The registered ops use the platform dispatch key, which is CUDA on GPU hosts,
    so CPU tensors have to call the implementations directly.
    """
    return SimpleNamespace(
        ops=SimpleNamespace(
            vllm=SimpleNamespace(
                afd_fastafd_dispatch=fastafd_moe._fastafd_dispatch,
                afd_fastafd_combine=fastafd_moe._fastafd_combine,
            ),
        ),
    )
