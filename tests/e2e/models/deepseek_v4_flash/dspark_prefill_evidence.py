# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in host-only evidence that P runs the target without a drafter."""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer import get_kv_transfer_group

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.dspark_prefill import AFDDSparkPrefillModelRunner


class DSparkPrefillEvidence:
    model_runner: "AFDDSparkPrefillModelRunner"

    def afd_dspark_prefill_snapshot(self) -> dict:
        runner = self.model_runner
        names = tuple(name for name, _ in runner.model.named_parameters())
        return {
            "runner": type(runner).__name__,
            "connector": type(get_kv_transfer_group()).__name__,
            "speculator_is_none": runner.speculator is None,
            "speculative_config_is_none": runner.speculative_config is None,
            "owns_mtp_parameters": any("mtp." in name for name in names),
            "captures_auxiliary_states": runner.use_aux_hidden_state_outputs,
            "target_parameter_tensors": len(names),
        }
