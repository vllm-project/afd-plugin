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

``padded`` and ``gathered`` are the shared pieces. ``gathered`` takes the
collective as an argument so the DP group, a fake, and a unit test all use the
same code path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import torch

# E8M0 encodes 2**(byte - 127), so the byte 127 is exactly 1.0. Padding rows use
# it because 1.0 is the neutral scale for an activation row of zeros: the padded
# activation is zero regardless, and a neutral factor keeps the padding contract
# uniform with the rest of the transport instead of introducing a special case.
#
# This is a byte value, not a float value: E8M0 only represents powers of two, so
# filling a tensor with the *float* 127 would round to 2**7 and store byte 134.
SCALE_PAD_BYTE: Final[int] = 127

# MX block size the A2E MXFP8 contract is defined against: one E8M0 exponent per 32
# activation values, so a scale row has ceil(hidden_size / 32) columns.
MX_BLOCK_SIZE: Final[int] = 32

# Expert indices reach the MoE kernels as 32-bit, and gather/combine treat them as
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
        scale: Optional E8M0 MX scale, ``[tokens, ceil(hidden/32)]``. Present only
            when the transport carries a quantized activation.
    """

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # Shape and dtype are checked here rather than at the padding site: a
        # wrong dtype is not rejected by ``view(uint8)``, it silently changes what
        # the bytes mean. A float32 scale would be read as eight exponents per
        # row instead of four bytes per value.
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
        if self.scale is None:
            return
        _require_2d(self.scale, "scale")
        if self.scale.dtype != torch.float8_e8m0fnu:
            raise ValueError(
                "activation scale must be E8M0 (torch.float8_e8m0fnu), got "
                f"{self.scale.dtype}",
            )
        if self.scale.shape[0] != self.topk_ids.shape[0]:
            raise ValueError(
                "activation scale must have one row per token, got "
                f"{self.scale.shape[0]} scale rows for {self.topk_ids.shape[0]} tokens",
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
            scale=None if self.scale is None else _pad_scale_rows(self.scale, pad_size),
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
            scale=None if self.scale is None else _gather_scale(self.scale, all_gather),
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
            scale=None if self.scale is None else _split_scale(self.scale, split),
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


def _pad_scale_rows(scale: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Append exponent-only scale rows carrying 1.0.

    Padding happens in byte space because E8M0 stores an exponent, not a
    magnitude: a float-space fill cannot express "the byte 127".
    """

    padded = _pad_rows(scale.view(torch.uint8), pad_size, SCALE_PAD_BYTE)
    return padded.view(torch.float8_e8m0fnu)


def _gather_scale(scale: torch.Tensor, all_gather) -> torch.Tensor:
    """Gather an E8M0 scale through its byte view.

    HCCL collectives are not required to accept the float8 E8M0 dtype, while the
    byte view moves exactly the same values.
    """

    return all_gather(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)


def _split_scale(scale: torch.Tensor, split) -> torch.Tensor:
    """Slice an E8M0 scale through its byte view, for the same reason as gather."""

    return split(scale.view(torch.uint8)).view(torch.float8_e8m0fnu)


def require_matching_activation_scale(
    activation: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Reject a scale that does not describe this activation.

    Row count, block-column count and device all have to line up, because the MoE uses
    the scale to skip its own quantization: a scale of the wrong width is read as a
    different block layout, and one on another device is an operator error deep inside
    the collective. Checking here turns either into a message at the consumer.

    ``AFDExternalRoutingInputs`` validates the scale's dtype and that it has one row per
    token; it cannot check the column count, because it never sees the activation.
    """

    if activation.dim() != 2 or scale.dim() != 2:
        raise RuntimeError(
            "external routing expects a two-dimensional activation and scale; got "
            f"activation {tuple(activation.shape)} and scale {tuple(scale.shape)}",
        )
    expected_columns = -(-int(activation.shape[1]) // MX_BLOCK_SIZE)
    if int(scale.shape[1]) != expected_columns:
        raise RuntimeError(
            "external routing scale does not match the activation width: "
            f"{int(activation.shape[1])} values need {expected_columns} blocks of "
            f"{MX_BLOCK_SIZE}, got {int(scale.shape[1])}",
        )
    if int(scale.shape[0]) != int(activation.shape[0]):
        raise RuntimeError(
            "external routing needs one scale row per activation row; got "
            f"{int(scale.shape[0])} rows for {int(activation.shape[0])}",
        )
    if scale.device != activation.device:
        raise RuntimeError(
            "external routing needs the scale on the activation's device; got "
            f"{scale.device} and {activation.device}",
        )


__all__ = [
    "AFDExternalRoutingInputs",
    "MX_BLOCK_SIZE",
    "SCALE_PAD_BYTE",
    "require_matching_activation_scale",
]
