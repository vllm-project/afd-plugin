# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Routing inputs the Attention side computed and the FFN side must reuse.

The FFN side runs the MoE with routing that arrived over the A2E transport
instead of computing it locally. That changes what ``prepare`` has to do: the
activation and the routing must be padded and gathered by exactly the same rule,
or a token row ends up weighted by another row's routing.

Two shapes of work exist, and this module owns the part they share:

- ALLGATHER pads to ``max_tokens_across_dp`` and then gathers across the DP
  group, so every rank sees every token.
- MC2 pads to ``padded_num_tokens`` and does not gather; the dispatcher moves
  tokens to their expert ranks.

``padded``, ``gathered`` and ``split_rows`` are row-aligned transforms. They keep
the caller's row order and are not required to share storage with the input, so
callers compare contents, not buffers.

``gathered`` takes the collective as an argument so the DP group, a fake, and a
unit test all use the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import torch

# Expert indices reach the MoE kernels as integers, and gather/combine treat them as
# integers. Listing the accepted dtypes explicitly also rejects bool and complex,
# which ``is_floating_point()`` alone would let through.
SUPPORTED_INDEX_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.int32,
    torch.int64,
)


@dataclass(frozen=True, eq=False)
class AFDExternalRoutingInputs:
    """Routing computed on the Attention side, in the FFN rank's local row order.

    Attributes:
        topk_ids: Expert ids per token, ``[tokens, top_k]``.
        topk_weights: Gate weights per token, ``[tokens, top_k]``. Any model-level
            routed scaling is already folded into these, mirroring the native
            router, so the FFN side must not apply it again.
    """

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    def __post_init__(self) -> None:
        # Shapes and dtypes are checked here rather than at the padding site: the
        # expert ids and the weights are consumed by kernels that do not validate
        # them, so a wrong dtype is only noticed as wrong results.
        _require_2d(self.topk_ids, "topk_ids")
        _require_2d(self.topk_weights, "topk_weights")
        if self.topk_ids.shape != self.topk_weights.shape:
            raise ValueError(
                "topk_ids and topk_weights must have the same shape, got "
                f"{tuple(self.topk_ids.shape)} and {tuple(self.topk_weights.shape)}",
            )
        if self.topk_ids.dtype not in SUPPORTED_INDEX_DTYPES:
            raise ValueError(
                "topk_ids must hold expert indices in one of "
                f"{[str(dtype) for dtype in SUPPORTED_INDEX_DTYPES]}, got "
                f"{self.topk_ids.dtype}",
            )
        if not self.topk_weights.is_floating_point():
            raise ValueError(
                "topk_weights must hold gate weights, got dtype "
                f"{self.topk_weights.dtype}",
            )

    @property
    def token_count(self) -> int:
        """Rows before any padding."""

        return int(self.topk_ids.shape[0])

    def padded(self, pad_size: int) -> AFDExternalRoutingInputs:
        """Append ``pad_size`` inert rows so later collectives stay aligned.

        Padding rows carry weight 0, so whichever expert they name contributes
        nothing. That keeps the padding value irrelevant to the result and lets
        the ids pad stay at 0.
        """

        if pad_size < 0:
            raise ValueError(f"pad_size must be non-negative, got {pad_size}")
        if pad_size == 0:
            return self

        return replace(
            self,
            topk_ids=_pad_rows(self.topk_ids, pad_size, 0),
            topk_weights=_pad_rows(self.topk_weights, pad_size, 0),
        )

    def gathered(self, all_gather) -> AFDExternalRoutingInputs:
        """Concatenate the routing of every DP rank, in rank order.

        ``all_gather`` must be the same collective that gathers the activation,
        invoked the same way, so the row order matches token for token.

        The gathered row count is ``dp_size * max(local_rows)``, not the sum of
        the local row counts: every rank pads to the DP-wide maximum before the
        gather, so each rank's padding rows survive into the result.
        """

        return replace(
            self,
            topk_ids=all_gather(self.topk_ids),
            topk_weights=all_gather(self.topk_weights),
        )

    def split_rows(self, split) -> AFDExternalRoutingInputs:
        """Keep one rank's slice, exactly as the activation is sliced.

        MC2 shards the token dimension across TP ranks instead of gathering it, so
        the routing has to take the same slice. ``split`` must be the split the
        caller applies to the activation.
        """

        return replace(
            self,
            topk_ids=split(self.topk_ids),
            topk_weights=split(self.topk_weights),
        )


def _require_2d(tensor: torch.Tensor, name: str) -> None:
    """Reject anything that is not a token-by-something matrix."""

    if tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2-D (tokens by something), got shape "
            f"{tuple(tensor.shape)}",
        )


def _pad_rows(tensor: torch.Tensor, pad_size: int, value: int) -> torch.Tensor:
    """Append ``pad_size`` rows filled with ``value``."""

    pad = torch.full(
        (pad_size, *tensor.shape[1:]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=0)


__all__ = [
    "AFDExternalRoutingInputs",
]
