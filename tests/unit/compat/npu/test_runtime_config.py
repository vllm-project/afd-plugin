# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD backend selection after native Ascend resolves model-owned MoE SP."""

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import fix_all2all_backend_for_afd


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("connector", ["CAMP2pAFDConnector", "CAMAsyncAFDConnector"])
@pytest.mark.parametrize(
    "tp_size, enable_ep, native_sp, backend, expected",
    [
        (1, False, False, "allgather_reducescatter", "flashinfer_all2allv"),
        (4, False, False, "allgather_reducescatter", "flashinfer_all2allv"),
        (1, True, False, "allgather_reducescatter", "flashinfer_all2allv"),
        (4, True, True, "allgather_reducescatter", "allgather_reducescatter"),
        (4, True, True, "deepep_low_latency", "deepep_low_latency"),
        (4, True, False, "flashinfer_all2allv", "flashinfer_all2allv"),
    ],
)
def test_backend_fix_preserves_native_moe_sp_without_compiler_flag(
    role, connector, tp_size, enable_ep, native_sp, backend, expected
):
    # Native config/FlashComm has already resolved this public SP property.
    # PassConfig in vLLM 0.32 has no enable_sp field.
    parallel = SimpleNamespace(
        tensor_parallel_size=tp_size,
        enable_expert_parallel=enable_ep,
        use_sequence_parallel_moe=native_sp,
        all2all_backend=backend,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(pass_config=SimpleNamespace()),
        additional_config={"afd": {"role": role, "connector": connector}},
    )

    fix_all2all_backend_for_afd(config)

    assert parallel.all2all_backend == expected
    assert parallel.use_sequence_parallel_moe is native_sp
    assert parallel.enable_expert_parallel is enable_ep
    assert parallel.tensor_parallel_size == tp_size
