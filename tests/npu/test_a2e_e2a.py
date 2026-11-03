# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Two-rank A2E/E2A roundtrip. Run with AFD_RUN_ASCEND_OP_RUNTIME=1 and
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/npu/test_a2e_e2a.py
"""

import os

import pytest

from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded

pytestmark = pytest.mark.npu


def test_a2e_e2a_runtime_roundtrip() -> None:
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("set AFD_RUN_ASCEND_OP_RUNTIME=1 for the two-rank NPU test")

    import torch

    ensure_afd_ascend_ops_loaded()

    try:
        import torch.distributed as dist
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise AssertionError("torch_npu and torch.distributed are required") from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise AssertionError(
            "runtime A2E/E2A smoke requires torchrun with exactly 2 ranks",
        )

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    rank = int(os.environ["RANK"])
    torch.npu.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    assert dist.get_world_size() == 2
    group = dist.distributed_c10d._get_default_group()
    group_ep = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    batch_size = 4
    hidden_size = 16
    topk = 1
    expert_rank_size = 1
    attention_rank_size = 1
    aiv_num = int(os.environ.get("AFD_A2E_E2A_AIV_NUM", "1"))

    expected = (
        torch.arange(batch_size * hidden_size, dtype=torch.float32, device="npu")
        .reshape(batch_size, hidden_size)
        .to(torch.bfloat16)
    )
    x = expected if rank == 1 else torch.zeros_like(expected)

    dist.barrier()
    a2e_out = torch.ops.afd_ascend.a2e(
        x,
        None,
        None,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        rank,
        group_ep,
        aiv_num,
        0,
    )
    torch.npu.synchronize()
    dist.barrier()

    expand_x, _, _, atten_batch_size, _ = a2e_out
    if rank == 0:
        torch.testing.assert_close(
            expand_x.cpu().float(),
            expected.cpu().float(),
            rtol=0,
            atol=0,
        )

    e2a_input = expand_x if rank == 0 else torch.empty_like(expected)
    dist.barrier()
    x_roundtrip = torch.ops.afd_ascend.e2a(
        e2a_input,
        atten_batch_size,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        rank,
        group_ep,
        aiv_num,
    )
    torch.npu.synchronize()
    dist.barrier()

    if rank == 1:
        torch.testing.assert_close(
            x_roundtrip.cpu().float(),
            expected.cpu().float(),
            rtol=0,
            atol=0,
        )

    dist.barrier()
