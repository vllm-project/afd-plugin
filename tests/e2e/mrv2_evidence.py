# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Test-only worker entry point for B/E/G execution evidence.

Re-export the native Worker for the selected E2E processes and log completed
Attention/FFN calls for B/E/G assertions.
"""

import json
import logging
from functools import wraps
from typing import cast

from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.v1.worker import ffn_model_runner
from afd_plugin.v1.worker.cuda_graph import AFDGraphRunMode

__all__ = ["Worker"]

EVIDENCE_INTERVAL = 128


def _install_probes():
    attention_count = ffn_count = 0
    attention_mode = "eager"
    ffn_mode: AFDGraphRunMode | None = None
    logger = logging.getLogger("vllm.e2e")
    native_execute = GPUModelRunner.execute_model
    native_replay = ModelCudaGraphManager.run_fullgraph
    ffn_execute = ffn_model_runner.GPUFFNModelRunner.execute_model
    select_mode = ffn_model_runner.graph_run_mode

    @wraps(native_replay)
    def replay(self, desc):
        nonlocal attention_mode
        result = native_replay(self, desc)
        attention_mode = "FULL"
        return result

    @wraps(native_execute)
    def execute(self, *args, **kwargs):
        nonlocal attention_count, attention_mode
        attention_mode = "eager"
        result = native_execute(self, *args, **kwargs)
        attention_count += 1
        metadata = self._afd_pending_metadata
        if metadata is not None and attention_count % EVIDENCE_INTERVAL in (1, 2):
            phase = (
                "profile"
                if kwargs.get("is_profile", False)
                else "dummy"
                if kwargs.get("dummy_run", False)
                else "live"
            )
            logger.info(
                "AFD execution: runner=MRV2 phase=%s mode=%s stages=%d "
                "tokens=%s real_tokens=%s count=%d",
                phase,
                attention_mode,
                metadata.num_stages,
                metadata.tokens_lens,
                metadata.tokens_unpadded_lens,
                attention_count,
            )
        return result

    @wraps(select_mode)
    def mode(**kwargs):
        nonlocal ffn_mode
        ffn_mode = select_mode(**kwargs)
        return ffn_mode

    @wraps(ffn_execute)
    def execute_ffn(self, *args, **kwargs):
        nonlocal ffn_count
        result = ffn_execute(self, *args, **kwargs)
        ffn_count += 1
        if ffn_count % EVIDENCE_INTERVAL in (1, 2):
            metadata = kwargs["dp_metadata_list"]
            logger.info(
                "AFD execution: runner=FFN mode=%s stages=%d layout=%s count=%d",
                cast(AFDGraphRunMode, ffn_mode).value,
                len(metadata),
                json.dumps(
                    ffn_model_runner.make_ffn_graph_key(metadata), separators=(",", ":")
                ),
                ffn_count,
            )
        return result

    GPUModelRunner.execute_model = execute
    ModelCudaGraphManager.run_fullgraph = replay
    ffn_model_runner.graph_run_mode = mode
    ffn_model_runner.GPUFFNModelRunner.execute_model = execute_ffn


_install_probes()
