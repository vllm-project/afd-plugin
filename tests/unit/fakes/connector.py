# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A recording ``AFDRoutedExpertsConnector`` for worker and runner tests.

Constructed without arguments it skips ``AFDConnectorBase.__init__`` (no vLLM
config needed); the factory constructs it with the base arguments.
``events`` records the lifecycle and data-path calls in order.
"""

from __future__ import annotations

from typing import Any

from afd_plugin.connectors import (
    AFDA2FTransferPayload,
    AFDRoutedExpertsConnector,
    AFDTransferContext,
    AFDTransferMetadata,
)


class FakeRoutedConnector(AFDRoutedExpertsConnector):
    # Raised by prepare_before_distributed_init when set.
    prepare_error: BaseException | None = None

    def __init__(self, *base_args: Any) -> None:
        if base_args:
            super().__init__(*base_args)
        self.events: list[tuple] = []
        self.control_updates: list[Any] = []
        self.init_kwargs: dict[str, Any] | None = None
        self.initialized = False
        self.control_plane = self  # type: ignore[assignment]

    @classmethod
    def parse_extra_config(cls, raw):
        return None

    @property
    def is_initialized(self) -> bool:
        return self.initialized

    def close(self) -> None:
        self.events.append(("close",))

    def prepare_before_distributed_init(self) -> None:
        self.events.append(("prepare",))
        if self.prepare_error is not None:
            raise self.prepare_error

    def init_afd_connector(self, *, moe_layer_indices=None, local_experts=None):
        self.init_kwargs = {
            "moe_layer_indices": moe_layer_indices,
            "local_experts": local_experts,
        }
        self.initialized = True

    # Control plane: the fake serves as its own.
    def update_state_from_dp_metadata(self, payload) -> None:
        self.control_updates.append(payload)

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send_attn",))

    def recv_ffn_output(self, ref_tensor, ubatch_idx=0, **kwargs):
        return ref_tensor

    def recv_attn_output(self, ubatch_idx=0, **kwargs):
        assert not kwargs, "the routed path passes no P2P receive options"
        self.events.append(("recv", ubatch_idx))
        return AFDA2FTransferPayload(
            hidden_states=f"rows{ubatch_idx}",
            context=AFDTransferContext(
                metadata=AFDTransferMetadata.create_ffn_metadata(
                    layer_idx=0,
                    stage_idx=ubatch_idx,
                    seq_lens=[1],
                ),
            ),
        )

    def run_experts(self, payload, weights):
        self.events.append(("run", payload.hidden_states, weights))
        return f"out({payload.hidden_states})"

    def send_ffn_output(self, ffn_output, context, **kwargs) -> None:
        self.events.append(("send", ffn_output, context.metadata.stage_idx))

    def synchronize_lanes(self) -> None:
        self.events.append(("sync",))
