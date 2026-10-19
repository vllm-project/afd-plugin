# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Allow the native Ascend DeepSeek-V4 model to load a config-cropped checkpoint."""

from collections.abc import Iterable, Iterator

import torch

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM


ORIGINAL_LOAD_WEIGHTS = AscendDeepseekV4ForCausalLM.load_weights


# Patch reason: a config-cropped checkpoint can still point to complete weight
# shards, whose layers beyond num_hidden_layers have no native model parameters.
# Patch functionality: skip only those extra layers before the native loader.
# Native load_weights is large and owns quantized expert mappings, so keep it
# intact through delegation. Signature matches the native method.
def load_weights(
    self: AscendDeepseekV4ForCausalLM,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    # ### PATCH START: filter layers beyond the configured model depth.
    def configured_layer_weights() -> Iterator[tuple[str, torch.Tensor]]:
        for name, weight in weights:
            parts = name.removeprefix("model.").split(".")
            if (
                len(parts) >= 2
                and parts[0] == "layers"
                and parts[1].isdigit()
                and int(parts[1]) >= self.config.num_hidden_layers
            ):
                continue
            yield name, weight

    filtered_weights = configured_layer_weights()
    # ### PATCH END: filter layers beyond the configured model depth.
    return ORIGINAL_LOAD_WEIGHTS(self, filtered_weights)


AscendDeepseekV4ForCausalLM.load_weights = load_weights
