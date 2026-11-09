# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD preserves model-owned SP and normalizes only the non-SP backend."""

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import (
    configure_async_cam_dbo,
    fix_all2all_backend_for_afd,
)


@pytest.mark.parametrize(
    "native_sp,backend,expected",
    [
        (False, "allgather_reducescatter", "flashinfer_all2allv"),
        (False, "flashinfer_all2allv", "flashinfer_all2allv"),
        (True, "allgather_reducescatter", "allgather_reducescatter"),
        (True, "deepep_low_latency", "deepep_low_latency"),
    ],
)
def test_backend_fix_preserves_native_moe_sp_without_compiler_flag(
    native_sp, backend, expected
):
    parallel = SimpleNamespace(
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        use_sequence_parallel_moe=native_sp,
        all2all_backend=backend,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        # PassConfig in the target runtime has no enable_sp flag.
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace()),
    )
    fix_all2all_backend_for_afd(config)

    assert parallel.all2all_backend == expected
    assert parallel.use_sequence_parallel_moe is native_sp
    assert parallel.enable_expert_parallel is True
    assert parallel.tensor_parallel_size == 4


@pytest.mark.parametrize("native_flag", [False, True])
def test_native_ubatching_normalizes_to_dbo(native_flag):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            use_ubatching=native_flag,
            enable_dbo=False,
            num_ubatches=2,
            dbo_decode_token_threshold=2,
            dbo_prefill_token_threshold=8,
        ),
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "attention",
                "async_dp": True,
            }
        },
    )
    configure_async_cam_dbo(config)
    assert config.parallel_config.enable_dbo is native_flag
    configure_async_cam_dbo(config)
    assert config.parallel_config.enable_dbo is native_flag


@pytest.mark.parametrize(
    "role,connector",
    [("ffn", "CAMAsyncAFDConnector"), ("attention", "CAMP2pAFDConnector")],
)
def test_async_normalization_does_not_enable_ffn_or_synchronous_dbo(role, connector):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True, enable_dbo=False),
        additional_config={
            "afd": {
                "connector": connector,
                "role": role,
            }
        },
    )
    configure_async_cam_dbo(config)
    assert not config.parallel_config.enable_dbo


@pytest.mark.parametrize(
    "threshold", ["dbo_decode_token_threshold", "dbo_prefill_token_threshold"]
)
def test_native_ubatching_revalidates_thresholds(threshold):
    parallel = SimpleNamespace(
        use_ubatching=True,
        enable_dbo=False,
        num_ubatches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=8,
    )
    setattr(parallel, threshold, 1)
    config = SimpleNamespace(
        parallel_config=parallel,
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "attention",
                "async_dp": True,
            }
        },
    )
    with pytest.raises(ValueError, match="at least the number of microbatches"):
        configure_async_cam_dbo(config)
