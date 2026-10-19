# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import torch


class DPMetadata:
    def __init__(self, values):
        # The connector reads token counts with .flatten().tolist(), so this
        # must be a tensor like the real DP metadata, not a plain list.
        self.num_tokens_across_dp_cpu = torch.tensor(values, dtype=torch.int32)


class DPMetadataValues:
    """DP metadata double holding token counts as given, for runners that only
    forward them."""

    def __init__(self, values):
        self.num_tokens_across_dp_cpu = values


def dp_tokens(dp_metadata) -> list[int]:
    values = dp_metadata.num_tokens_across_dp_cpu
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)
