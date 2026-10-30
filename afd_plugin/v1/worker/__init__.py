# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime classes loadable by explicit vLLM class paths."""

from importlib import import_module

_RUNTIME_EXPORTS = {
    "AFDAttentionModelRunnerV2": "afd_plugin.v1.worker.attention_model_runner_v2",
    "AFDAttentionWorker": "afd_plugin.v1.worker.attention_worker",
    "AFDFFNWorker": "afd_plugin.v1.worker.ffn_worker",
    "GPUFFNModelRunner": "afd_plugin.v1.worker.ffn_model_runner",
    "GPUFFNRoutedExpertsModelRunner": (
        "afd_plugin.v1.worker.ffn_routed_experts_model_runner"
    ),
}


def __getattr__(name: str):
    module_name = _RUNTIME_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "AFDAttentionModelRunnerV2",
    "AFDAttentionWorker",
    "AFDFFNWorker",
    "GPUFFNModelRunner",
    "GPUFFNRoutedExpertsModelRunner",
]
