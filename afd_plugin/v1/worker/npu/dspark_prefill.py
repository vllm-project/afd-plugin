# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Target-only Prefill capture for the DSV4 DSpark PD connector."""

from __future__ import annotations

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.utils import register_ascend_fla_customop
from vllm_ascend.worker.dspark_utils import get_dspark_aux_layer_ids
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.distributed.kv_transfer.dspark_mooncake import (
    AFDDSparkMooncakeConnector,
)


class AFDDSparkPrefillModelRunner(NPUModelRunner):
    """Keep native P inference and copy auxiliary states before sampling."""

    # Upstream: Ascend bd42eea0 NPUModelRunner.load_model.
    # AFD configures native target capture after loading, avoiding construction
    # of any speculative model on P. Remove once native PD supports target-only
    # DSV4 context export. All native weight loading remains delegated.
    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        super().load_model(load_dummy_weights, *args, **kwargs)
        # ### PATCH START: enable native auxiliary outputs without a P drafter
        layers = get_dspark_aux_layer_ids(self.vllm_config)
        if not layers:
            raise ValueError("DSpark Prefill requires agreed auxiliary capture layers")
        self.model.set_aux_hidden_state_layers(layers)
        self.use_aux_hidden_state_outputs = True
        # ### PATCH END: enable native auxiliary outputs without a P drafter

    # Upstream: Ascend bd42eea0 NPUModelRunner.execute_model; identical signature.
    # Native execution is too large to copy. This composition seam only copies
    # context rows before sample_tokens can finish P and release its workspaces.
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        result = super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run,
            skip_attn_for_dummy_run,
            is_profile,
            context_len,
            valid_dummy_state_slots,
            randomize_inputs,
        )
        # ### PATCH START: retain actual P prompt rows, including chunked prefill
        if (
            not dummy_run
            and not is_profile
            and scheduler_output.total_num_scheduled_tokens
        ):
            connector = get_kv_transfer_group()
            if not isinstance(connector, AFDDSparkMooncakeConnector):
                raise TypeError("DSpark Prefill requires AFDDSparkMooncakeConnector")
            if self.execute_model_state is None:
                raise RuntimeError("Native Prefill produced no execution state")
            connector.capture_prefill(self.execute_model_state)
        # ### PATCH END: retain actual P prompt rows, including chunked prefill
        return result


class AFDDSparkPrefillWorker(NPUWorker):
    """Use a native target model and the auxiliary-capture runner on P."""

    # Upstream: Ascend bd42eea0 NPUWorker.init_device. Native has no MRV2 runner
    # factory. Preserve its text-only steps while selecting the capture runner.
    # Remove when the worker exposes an overridable runner factory.
    def init_device(self) -> None:
        # ### PATCH START: validate the target-only eager Prefill configuration
        config = self.vllm_config
        parallel = config.parallel_config
        if (
            not self.use_v2_model_runner
            or config.is_mm_encoder_only
            or config.speculative_config is not None
            or not config.model_config.enforce_eager
            or config.cache_config.enable_prefix_caching
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_ubatching
            or parallel.enable_dbo
            or config.kv_transfer_config is None
            or not config.kv_transfer_config.is_kv_producer
        ):
            raise ValueError(
                "DSpark Prefill requires target-only MRV2 eager, PP/CP=1, "
                "a KV producer, and prefix caching/DBO disabled"
            )
        # ### PATCH END: validate the target-only eager Prefill configuration
        self.device = self._init_device()
        register_ascend_fla_customop()
        init_workspace_manager(self.device, 1)
        # ### PATCH START: select the target-only capture runner
        self.model_runner = AFDDSparkPrefillModelRunner(config, self.device)
        # ### PATCH END: select the target-only capture runner
        if self.rank == 0:
            report_usage_stats(config)
