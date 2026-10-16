# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.validation import (
    ATTENTION_WORKER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
    validate_npu_model_runner_v2_config,
)


def _vllm_like_config(*, afd, worker_cls):
    return SimpleNamespace(
        additional_config={"afd": afd},
        parallel_config=SimpleNamespace(worker_cls=worker_cls),
    )


def test_attention_stack_validation_accepts_matching_worker():
    vllm_config = _vllm_like_config(
        afd={"role": "attention"},
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="attention",
    )

    assert config.role == "attention"


def test_ffn_stack_validation_accepts_matching_worker():
    vllm_config = _vllm_like_config(
        afd={"role": "ffn"},
        worker_cls=FFN_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="ffn",
    )

    assert config.role == "ffn"


def test_stack_validation_rejects_missing_afd_config():
    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(worker_cls=ATTENTION_WORKER_FQCN),
    )

    with pytest.raises(ValueError, match="requires additional_config"):
        assert_compatible_afd_stack(vllm_config, caller="test")


def test_stack_validation_rejects_wrong_worker():
    vllm_config = _vllm_like_config(
        afd={"role": "ffn"},
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    with pytest.raises(ValueError, match="invalid worker class") as exc_info:
        assert_compatible_afd_stack(vllm_config, caller="test")
    assert "remove --worker-cls" in str(exc_info.value)


def test_stack_validation_rejects_auto_worker():
    vllm_config = _vllm_like_config(
        afd={"role": "attention"},
        worker_cls="auto",
    )

    with pytest.raises(ValueError, match="remained 'auto'") as exc_info:
        assert_compatible_afd_stack(vllm_config, caller="test")
    assert "ensure the AFD general plugin is loaded" in str(exc_info.value)


def test_stack_validation_accepts_npu_worker_override():
    vllm_config = _vllm_like_config(
        afd={
            "role": "attention",
            "connector": "CAMP2pAFDConnector",
        },
        worker_cls=NPU_ATTENTION_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="attention",
        expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
    )

    assert config.connector == "CAMP2pAFDConnector"


def test_async_connector_requires_npu_attention_worker():
    vllm_config = _vllm_like_config(
        afd={
            "role": "attention",
            "connector": "CAMAsyncAFDConnector",
        },
        worker_cls=ATTENTION_WORKER_FQCN,
    )

    with pytest.raises(ValueError, match="requires Ascend NPU worker"):
        assert_compatible_afd_stack(
            vllm_config,
            caller="test",
            expected_role="attention",
        )

    vllm_config.parallel_config.worker_cls = NPU_ATTENTION_WORKER_FQCN
    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="attention",
    )
    assert config.connector == "CAMAsyncAFDConnector"


def test_async_connector_requires_npu_ffn_worker():
    vllm_config = _vllm_like_config(
        afd={
            "role": "ffn",
            "connector": "CAMAsyncAFDConnector",
        },
        worker_cls=NPU_FFN_WORKER_FQCN,
    )

    config = assert_compatible_afd_stack(
        vllm_config,
        caller="test",
        expected_role="ffn",
    )

    assert config.connector == "CAMAsyncAFDConnector"


@pytest.fixture
def mrv2_config(monkeypatch):
    # Model registration is tested with the real vLLM runtime elsewhere.
    # Isolate that import so deployment policy tests remain CPU/dependency-safe.
    model_utils = ModuleType("afd_plugin.model_executor.models.model_utils")
    monkeypatch.setattr(
        model_utils, "has_afd_model_registration", lambda config: True, raising=False
    )
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)

    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={
            "afd": {
                "role": "attention",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 2,
            }
        },
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
        ),
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(
            cudagraph_mode="FULL_DECODE_ONLY",
            pass_config=SimpleNamespace(enable_sp=False),
        ),
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("dp_size, tp_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("compute_gate_on_attention", [False, True])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_gpu_v2_validator_accepts_two_ubatches(
    mrv2_config, role, dp_size, tp_size, compute_gate_on_attention, enforce_eager
):
    mrv2_config.additional_config["afd"]["role"] = role
    mrv2_config.additional_config["afd"]["compute_gate_on_attention"] = (
        compute_gate_on_attention
    )
    mrv2_config.model_config.enforce_eager = enforce_eager
    mrv2_config.parallel_config.data_parallel_size = dp_size
    mrv2_config.parallel_config.tensor_parallel_size = tp_size
    mrv2_config.additional_config["afd"].update(
        num_attention_ranks=dp_size * tp_size,
        num_ffn_ranks=dp_size * tp_size,
    )
    validate_gpu_model_runner_v2_config(
        mrv2_config, expected_role=role, device_type="cuda"
    )


@pytest.mark.parametrize("role, dp_size", [("attention", 2), ("ffn", 1)])
def test_gpu_v2_validator_accepts_asymmetric_attention_to_ffn_topology(
    mrv2_config, role, dp_size
):
    mrv2_config.additional_config["afd"].update(
        role=role, num_attention_ranks=2, num_ffn_ranks=1
    )
    mrv2_config.parallel_config.data_parallel_size = dp_size

    validate_gpu_model_runner_v2_config(
        mrv2_config, expected_role=role, device_type="cuda"
    )


@pytest.mark.parametrize(
    "num_ubatches",
    [1, 3],
)
def test_gpu_v2_validator_rejects_unsupported_ubatching(mrv2_config, num_ubatches):
    mrv2_config.parallel_config.num_ubatches = num_ubatches
    with pytest.raises(RuntimeError, match="exactly two microbatches"):
        validate_gpu_model_runner_v2_config(
            mrv2_config, expected_role="attention", device_type="cuda"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_gpu_validator_rejects_v1_before_importing_model_runtime(
    monkeypatch, mrv2_config, role
):
    mrv2_config.use_v2_model_runner = False
    mrv2_config.additional_config["afd"]["role"] = role
    # The V1 deployment must fail before importing or constructing a runner.
    # Block the runtime module so an import before rejection fails visibly even
    # when vLLM is installed in the test environment.
    monkeypatch.setitem(
        sys.modules, "afd_plugin.model_executor.models.model_utils", None
    )

    with pytest.raises(RuntimeError, match="requires ModelRunnerV2"):
        validate_gpu_model_runner_v2_config(
            mrv2_config, expected_role=role, device_type="cuda"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_npu_validator_rejects_v1_before_model_runtime(monkeypatch, mrv2_config, role):
    mrv2_config.use_v2_model_runner = False
    mrv2_config.additional_config["afd"].update(
        role=role, connector="CAMP2pAFDConnector"
    )
    monkeypatch.setitem(
        sys.modules, "afd_plugin.model_executor.models.model_utils", None
    )
    with pytest.raises(RuntimeError, match="AFD NPU requires ModelRunnerV2"):
        validate_npu_model_runner_v2_config(
            mrv2_config, expected_role=role, device_type="npu"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_npu_sync_dbo_policy_allows_eager_and_full_graph(
    mrv2_config, role, enforce_eager
):
    mrv2_config.additional_config["afd"].update(
        role=role, connector="CAMP2pAFDConnector"
    )
    mrv2_config.model_config.enforce_eager = enforce_eager
    mrv2_config.speculative_config = None
    validate_npu_model_runner_v2_config(
        mrv2_config, expected_role=role, device_type="npu"
    )
