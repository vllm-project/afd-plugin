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

    def make(
        *,
        role="attention",
        num_attention_ranks=1,
        num_ffn_ranks=1,
        data_parallel_size=1,
        enforce_eager=True,
        cudagraph_mode="FULL_DECODE_ONLY",
    ):
        return SimpleNamespace(
            additional_config={
                "afd": {
                    "role": role,
                    "num_attention_ranks": num_attention_ranks,
                    "num_ffn_ranks": num_ffn_ranks,
                }
            },
            parallel_config=SimpleNamespace(
                data_parallel_size=data_parallel_size,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                enable_expert_parallel=True,
                enable_elastic_ep=False,
                enable_eplb=False,
                use_sequence_parallel_moe=False,
                enable_dbo=False,
                use_ubatching=False,
                num_ubatches=1,
            ),
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
            compilation_config=SimpleNamespace(
                cudagraph_mode=cudagraph_mode,
                pass_config=SimpleNamespace(enable_sp=False),
            ),
        )

    return make


@pytest.mark.parametrize("role, dp_size", [("attention", 2), ("ffn", 2), ("ffn", 1)])
@pytest.mark.parametrize("enable_dbo", [True, False])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_gpu_v2_validator_accepts_two_ubatches_for_paired_roles(
    mrv2_config, role, dp_size, enable_dbo, enforce_eager
):
    config = mrv2_config(
        role=role,
        num_attention_ranks=2,
        num_ffn_ranks=dp_size if role == "ffn" else 2,
        data_parallel_size=dp_size,
        enforce_eager=enforce_eager,
        cudagraph_mode="FULL_DECODE_ONLY",
    )
    config.parallel_config.enable_dbo = enable_dbo
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    validate_gpu_model_runner_v2_config(config, expected_role=role, device_type="cuda")


def test_gpu_v2_validator_rejects_dbo_without_native_attention_ubatch_runner(
    mrv2_config,
):
    config = mrv2_config()
    config.parallel_config.enable_dbo = True
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    with pytest.raises(RuntimeError, match="Attention DP > 1"):
        validate_gpu_model_runner_v2_config(
            config, expected_role="attention", device_type="cuda"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_gpu_v2_validator_rejects_more_than_two_ubatches(mrv2_config, role):
    config = mrv2_config(
        role=role, num_attention_ranks=2, num_ffn_ranks=2, data_parallel_size=2
    )
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 3
    with pytest.raises(RuntimeError, match="exactly two microbatches"):
        validate_gpu_model_runner_v2_config(
            config, expected_role=role, device_type="cuda"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_npu_v2_validator_still_rejects_two_ubatches(mrv2_config, role):
    config = mrv2_config(
        role=role, num_attention_ranks=2, num_ffn_ranks=2, data_parallel_size=2
    )
    config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    config.parallel_config.enable_dbo = True
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    with pytest.raises(RuntimeError, match="DBO or ubatching"):
        validate_npu_model_runner_v2_config(
            config, expected_role=role, device_type="npu"
        )
