# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import pytest


@pytest.fixture
def config():
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "P2pNcclAFDConnector",
                "role": "attention",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 1,
            }
        },
        parallel_config=SimpleNamespace(
            enable_elastic_ep=True,
            enable_eplb=False,
            enable_expert_parallel=True,
            worker_cls="afd_plugin.v1.worker.AFDAttentionWorker",
            tensor_parallel_size=1,
            data_parallel_size=2,
            data_parallel_size_local=2,
            data_parallel_backend="ray",
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            distributed_executor_backend="uni",
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            use_ubatching=False,
            elastic_ep_max_dp_size=8,
            eplb_config=SimpleNamespace(num_redundant_experts=0, use_async=False),
            _api_process_count=1,
        ),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1048576),
        model_config=SimpleNamespace(
            enable_sleep_mode=False, enforce_eager=True, get_num_experts=lambda: 256
        ),
        use_v2_model_runner=True,
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
    )
