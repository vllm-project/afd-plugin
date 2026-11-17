# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native Ascend metadata seams without device allocation."""

from types import SimpleNamespace

from tests.unit.fakes.vllm_config import npu_vllm_config


def metadata_runner():
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )

    runner = object.__new__(AFDNPUAttentionModelRunnerV2)
    runner.vllm_config = npu_vllm_config()
    runner.vllm_config.kv_transfer_config = None
    runner.speculative_config = SimpleNamespace(enforce_eager=False)
    runner.connector = SimpleNamespace(control_plane=None)
    runner._afd_pending_metadata = None
    runner._afd_suppress_metadata_send = False
    runner._afd_transaction_counter = 0
    runner._is_warmup = runner._afd_is_profile = False
    runner._afd_is_graph_capturing = runner._afd_is_graph_replaying = False
    return runner


def forward_context(num_tokens):
    return SimpleNamespace(
        batch_descriptor=SimpleNamespace(num_tokens=num_tokens),
        attn_metadata={},
        additional_kwargs={},
        ubatch_slices=None,
    )


class DraftGraphManager:
    def __init__(self, runner, descs=(), *, fail=False):
        from vllm.config import CUDAGraphMode

        self.runner = runner
        self._capture_descs = {CUDAGraphMode.FULL: list(descs)}
        self._max_full_descs_to_capture = None
        self.fail = fail
        self.emitted = []

    def run_fullgraph(self, desc):
        context = forward_context(desc.num_tokens)
        self.runner.install_afd_metadata_on_forward_context(context)
        metadata = context.additional_kwargs["afd_metadata"]
        self.emitted.append(metadata)
        if self.fail:
            raise RuntimeError("native replay failed")
        return metadata
