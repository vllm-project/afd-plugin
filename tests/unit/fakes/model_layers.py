# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import torch
from torch import nn


class Stage(nn.Module):
    """Layer stage double that records the prefix of each constructed `kind`."""

    kind = "stage"

    def __init__(self, calls: dict[str, list[str]], *args, prefix="", **kwargs):
        super().__init__()
        calls[self.kind].append(prefix)
        self.weight = nn.Parameter(torch.empty(1))


def stage_type(kind: str):
    return type(f"Fake{kind.title()}", (Stage,), {"kind": kind})
