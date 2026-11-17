# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reusable execution configuration; models select cases and acceptance settings."""

from dataclasses import dataclass

from tests.e2e.case import (
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    DboConfig,
    Deployment,
    Role,
)


@dataclass(frozen=True)
class ExecutionPreset:
    """A shared execution mode for both roles of a synchronous deployment."""

    execution: str

    def deployment(
        self,
        *,
        backend: str,
        attention: tuple[int, int],
        ffn: tuple[int, int] | None,
        connector: str | None,
        capture_size: int,
        dbo: DboConfig | None = None,
        compute_gate_on_attention: bool = False,
    ) -> Deployment:
        """Layouts are (DP, TP); models may replace explicit budget fields."""
        graph = self.execution == EXECUTION_FULL_DECODE_GRAPH
        return Deployment(
            backend=backend,
            attention=Role(*attention, execution=self.execution),
            ffn=Role(*ffn, execution=self.execution) if ffn is not None else None,
            connector=connector,
            dbo=dbo,
            compute_gate_on_attention=compute_gate_on_attention,
            cudagraph_capture_size=capture_size if graph else None,
            max_num_seqs=capture_size if graph else None,
            max_num_batched_tokens=capture_size if ffn is None else None,
        )


EAGER = ExecutionPreset(EXECUTION_EAGER)
DECODE_GRAPH = ExecutionPreset(EXECUTION_FULL_DECODE_GRAPH)
