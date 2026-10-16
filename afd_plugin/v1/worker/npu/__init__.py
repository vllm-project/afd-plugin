# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU runtime classes loadable by explicit vLLM class paths."""

from importlib import import_module

_RUNTIME_EXPORTS = {
    "AFDNPUAttentionModelRunner": "afd_plugin.v1.worker.npu.attention_model_runner",
    "AFDNPUAttentionModelRunnerV2": (
        "afd_plugin.v1.worker.npu.attention_model_runner_v2"
    ),
    "AFDNPUAttentionWorker": "afd_plugin.v1.worker.npu.attention_worker",
    "AFDNPUFFNModelRunner": "afd_plugin.v1.worker.npu.ffn_model_runner",
    "AsyncNPUFFNModelRunner": "afd_plugin.v1.worker.npu.async_ffn_model_runner",
    "AFDNPUFFNWorker": "afd_plugin.v1.worker.npu.ffn_worker",
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
    "AFDNPUAttentionModelRunner",
    "AFDNPUAttentionModelRunnerV2",
    "AFDNPUAttentionWorker",
    "AFDNPUFFNModelRunner",
    "AFDNPUFFNWorker",
    "AsyncNPUFFNModelRunner",
]
