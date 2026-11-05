# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

import copy
from types import SimpleNamespace

import pytest

from afd_plugin.config import parse_afd_config
from afd_plugin.elastic.config import (
    ElasticTopology,
    is_elastic_attention_worker,
    set_afd_config,
    validate_elastic_config,
)


def test_default_and_independent_role_sizes(config):
    topology = validate_elastic_config(config)
    assert topology == ElasticTopology(2, 1)
    assert topology.resize("attention", 4) == ElasticTopology(4, 1)
    assert topology.resize("ffn", 2) == ElasticTopology(2, 2)
    # Non-divisible fanout (3A x 2F) is rejected: the CAMP2P transport wires
    # fixed N:1 windows (ratio = attn // ffn) and _num_tokens_for_ffn_rank
    # expects uniform row counts per source; a dangling source hangs the a2e
    # kernels (verified on device: AI-core vector timeout).
    with pytest.raises(ValueError, match="divisible"):
        ElasticTopology(3, 1).resize("ffn", 2)


@pytest.mark.parametrize("mode", ["NONE", "VLLM_COMPILE"])
def test_full_decode_graph_config(config, mode):
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        mode=SimpleNamespace(name=mode),
        cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
    )
    assert validate_elastic_config(config) == ElasticTopology(2, 1)


def test_eager_dbo_config(config):
    config.parallel_config.enable_dbo = True
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    assert validate_elastic_config(config) == ElasticTopology(2, 1)


@pytest.mark.parametrize("mode", ["NONE", "VLLM_COMPILE"])
def test_dbo_graph_config(config, mode):
    config.parallel_config.enable_dbo = True
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        mode=SimpleNamespace(name=mode),
        cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
    )
    assert validate_elastic_config(config) == ElasticTopology(2, 1)


def test_stock_compile_graph_rejected_before_role_initialization(config):
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        mode=SimpleNamespace(name="STOCK_TORCH_COMPILE"),
        cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
    )
    with pytest.raises(ValueError, match="STOCK_TORCH_COMPILE"):
        validate_elastic_config(config)


def test_unsupported_graph_mode_rejected_before_role_initialization(config):
    config.model_config.enforce_eager = False
    config.compilation_config = SimpleNamespace(
        mode=SimpleNamespace(name="VLLM_COMPILE"),
        cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"),
    )
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        validate_elastic_config(config)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "2"])
def test_bad_size_is_rejected(size):
    with pytest.raises(ValueError):
        ElasticTopology(4, 1).resize("ffn", size)


def test_unsupported_targets_fail_before_mutation():
    topology = ElasticTopology(2, 1)
    for role, size in [("attention", 1), ("ffn", 3), ("both", 2)]:
        with pytest.raises(ValueError):
            topology.resize(role, size)
    assert topology == ElasticTopology(2, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_eplb", True),
        ("enable_expert_parallel", False),
        ("data_parallel_backend", "mp"),
        ("distributed_executor_backend", "ray"),
        ("data_parallel_external_lb", True),
        ("data_parallel_hybrid_lb", True),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("use_ubatching", True),
        ("_api_process_count", 2),
        ("worker_cls", "auto"),
    ],
)
def test_runtime_boundaries(config, field, value):
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError):
        validate_elastic_config(config)


def test_npu_connector_now_qualified(config):
    # The stateless HCCL EEP group qualification landed in vllm-ascend
    # (feat/eep-device-group); elastic execution is enabled for CAMP2P.
    config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    config.parallel_config.worker_cls = "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
    assert validate_elastic_config(config) == ElasticTopology(2, 1)


def test_unqualified_connector_is_explicit(config):
    config.additional_config["afd"]["connector"] = "CAMAsyncAFDConnector"
    config.parallel_config.worker_cls = "afd_plugin.v1.worker.npu.AFDNPUAttentionWorker"
    with pytest.raises(ValueError, match="CAMP2pAFDConnector"):
        validate_elastic_config(config)


def test_no_global_eplb_exemption(config):
    parallel = config.parallel_config
    assert is_elastic_attention_worker(parallel)
    parallel.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
    assert not is_elastic_attention_worker(parallel)
    parallel.worker_cls = "afd_plugin.v1.worker:AFDAttentionWorker"
    assert is_elastic_attention_worker(parallel)
    parallel.enable_elastic_ep = False
    assert not is_elastic_attention_worker(parallel)


def test_config_update_preserves_connector_options_and_input(config):
    config.additional_config["afd"]["connector_extra_config"] = {"test": 12}
    old = copy.deepcopy(config.additional_config)
    afd = ElasticTopology(4, 2).afd_config(
        parse_afd_config(config), port=12345, role="attention"
    )
    set_afd_config(config, afd)
    assert config.additional_config["afd"]["connector_extra_config"] == {"test": 12}
    assert config.additional_config["afd"]["num_ffn_ranks"] == 2
    assert old["afd"]["num_ffn_ranks"] == 1
