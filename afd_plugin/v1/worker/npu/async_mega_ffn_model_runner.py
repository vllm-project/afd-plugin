# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Mega FFN runner: the fused connector operator owns each FFN step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from afd_plugin.connectors.npu.async_mega import CAMAsyncMegaAFDConnector
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.async_mega.mega_layers import LayerWeights


class MegaFFNModel(Protocol):
    def get_mega_expert_weights(self, rank: int) -> dict[int, LayerWeights]: ...


class MegaNPUFFNModelRunner(AFDNPUFFNModelRunner):
    """Run Mega FFN steps instead of pulling per-layer work from a control plane.

    Mega receives Attention output, computes the routed experts and sends the
    combine inside one operator, so there is no DP metadata payload to receive
    and no per-layer model forward to drive. Every ubatch shares one window and
    one submission lane; each step submits the fused operator once, and the
    driver does not fence steps on device completion.

    The base runner is inherited for FFN model ownership only: config parsing,
    connector construction, native model loading (MC2 capacity/mask, memory
    accounting) and shutdown ordering. Its CAMP2P step and ACL graph paths stay
    unused because Mega overrides the connector initialization and the step.
    """

    def initialize_afd_connector(self) -> None:
        if self.connector.is_initialized:
            return
        connector = cast(CAMAsyncMegaAFDConnector, self.connector)
        # Layer weights exist only after the model is loaded; the connector
        # packs them once and selects the layer on device per dispatched tile.
        model = cast(MegaFFNModel, self.get_model())
        connector.prepare_layers(model.get_mega_expert_weights(connector.role_rank))
        connector.init_afd_connector()

    def execute_connector_driven_step(self) -> None:
        """Run one fused dispatch/compute/combine step for the next ready ubatch."""
        cast(CAMAsyncMegaAFDConnector, self.connector).execute_next_experts()


__all__ = ["MegaNPUFFNModelRunner"]
