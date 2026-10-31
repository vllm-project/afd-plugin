# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Load config-cropped DeepSeek-V4 checkpoints through the native Ascend loader.

Upstream: vllm_ascend/models/deepseek_v4.py at 11ee45653.
Install after the NPU worker is resolved, before model loading. Remove when
the native loader skips checkpoint layers beyond the configured target depth.
"""

import inspect
from collections.abc import Iterable, Iterator

import torch

_PATCH_ATTR = "_afd_plugin_cropped_deepseek_v4_weights_patch"
_LOAD_WEIGHTS_PARAMETERS = ("self", "weights")


def apply_afd_cropped_deepseek_v4_weights_patch() -> bool:
    """Install once, only after the Ascend worker has completed its imports."""
    from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM

    model_cls = AscendDeepseekV4ForCausalLM
    if _PATCH_ATTR in model_cls.__dict__:
        return True
    original_load_weights = model_cls.load_weights
    parameters = inspect.signature(original_load_weights).parameters
    if tuple(parameters) != _LOAD_WEIGHTS_PARAMETERS or any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        or parameter.default is not inspect.Parameter.empty
        for parameter in parameters.values()
    ):
        return False

    # Patch reason: a cropped config can point to full checkpoint shards whose
    # extra decoder layers have no parameters in the native Prefill model.
    # Patch functionality: filter only decoder layers beyond the target depth.
    # Expansion exception: the native loader owns the large quantized expert
    # mapping pipeline; delegate it unchanged after this single-pass filter.
    # Signature: matches the pinned native load_weights; no added parameters.
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        # ### PATCH START: filter cropped decoder layers
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
        # ### PATCH END: filter cropped decoder layers
        return original_load_weights(self, filtered_weights)

    model_cls.load_weights = load_weights
    setattr(model_cls, _PATCH_ATTR, original_load_weights)
    return True
