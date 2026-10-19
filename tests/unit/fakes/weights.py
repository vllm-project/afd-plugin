# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import torch


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
