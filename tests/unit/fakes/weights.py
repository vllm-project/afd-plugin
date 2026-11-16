# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import torch

FP8_WEIGHT_BLOCK_SIZE = 128
FP8_SCALE_EPS = 1e-10


def quantize_fp8_block128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build test weights in vLLM's FP8 per-128x128-block checkpoint layout."""
    num_experts, n, k = w.shape
    if n % FP8_WEIGHT_BLOCK_SIZE or k % FP8_WEIGHT_BLOCK_SIZE:
        raise ValueError(f"weight shape {tuple(w.shape)} is not 128-block aligned")
    blocks = w.float().view(
        num_experts,
        n // FP8_WEIGHT_BLOCK_SIZE,
        FP8_WEIGHT_BLOCK_SIZE,
        k // FP8_WEIGHT_BLOCK_SIZE,
        FP8_WEIGHT_BLOCK_SIZE,
    )
    limits = torch.finfo(torch.float8_e4m3fn)
    scale = blocks.abs().amax(dim=(2, 4)).clamp(min=FP8_SCALE_EPS) / limits.max
    w_q = (
        (blocks / scale[:, :, None, :, None])
        .clamp(limits.min, limits.max)
        .to(torch.float8_e4m3fn)
    )
    return w_q.view(num_experts, n, k), scale


class OneShotWeights:
    """Checkpoint iterator that fails if a loader consumes it twice."""

    def __init__(self, names: list[str]) -> None:
        self.items = [(name, torch.tensor([idx])) for idx, name in enumerate(names)]
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("checkpoint iterator was consumed more than once")
        return iter(self.items)
