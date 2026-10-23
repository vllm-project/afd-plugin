# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FastAFD dispatch and combine custom ops shared by attention-routed MoE models.

A model whose Attention role routes tokens itself calls
``torch.ops.vllm.afd_fastafd_dispatch`` to send each layer's routed tokens to
the FFN ranks and ``torch.ops.vllm.afd_fastafd_combine`` to collect and reduce
the expert outputs. Both ops resolve the connector and the microbatch lane from
the forward context at run time.
"""

import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from afd_plugin.config import FASTAFD_CONNECTOR
from afd_plugin.connectors.gpu.fastafd import FastAFDAFDConnector
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)


def _fastafd_connector_and_lane() -> tuple[FastAFDAFDConnector, int]:
    afd_metadata = get_afd_metadata_from_forward_context()
    if afd_metadata is None:
        raise RuntimeError("Attention-routed MoE needs AFD forward metadata")
    connector = afd_metadata.connector
    if not isinstance(connector, FastAFDAFDConnector):
        raise RuntimeError(
            f"Attention-routed MoE requires {FASTAFD_CONNECTOR}, got "
            f"{type(connector).__name__}",
        )
    lane = int(getattr(get_forward_context(), "ubatch_idx", afd_metadata.stage_idx))
    return connector, lane


def _fastafd_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Send the routed tokens; return a copy of the input for local work."""
    connector, lane = _fastafd_connector_and_lane()
    connector.dispatch(hidden_states, topk_ids, topk_weights, lane)
    return hidden_states.clone()


def _fastafd_dispatch_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _fastafd_combine(
    local_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Wait for the routed-expert outputs; ``local_output`` only orders it."""
    connector, lane = _fastafd_connector_and_lane()
    return connector.combine(hidden_states, topk_ids, topk_weights, lane)


def _fastafd_combine_fake(
    local_output: torch.Tensor,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


for _op_name, _op_func, _fake_impl in (
    ("afd_fastafd_dispatch", _fastafd_dispatch, _fastafd_dispatch_fake),
    ("afd_fastafd_combine", _fastafd_combine, _fastafd_combine_fake),
):
    if not hasattr(torch.ops.vllm, _op_name):
        direct_register_custom_op(
            op_name=_op_name,
            op_func=_op_func,
            mutates_args=[],
            fake_impl=_fake_impl,
        )
