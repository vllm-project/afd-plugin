# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 / Ascend 950 1A1F hardware sample for A2E and E2A.

This is a standalone mp.spawn script, not a default pytest case. The filename
avoids the ``test_*.py`` pattern so collection does not import torch_npu.

Run:

    AFD_TEST_DEVICES=2,3 python tests/unit/compat/npu/a2e_e2a_a5.py

Environment:
    AFD_TEST_DEVICES  physical NPU ids, default ``2,3``
    AFD_TEST_E        expert/FFN rank count, default 1
    AFD_TEST_A        attention rank count, default same as E
    AFD_TEST_WORLD    HCCL world size, default E + A
    AFD_TEST_PORT     MASTER_PORT, default 29601
"""

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

# afd registers a2e/e2a in torch.ops.afd_ascend via this extension. Each
# mp.spawn child is a fresh interpreter, so import it in every process.
# ensure_cam_p2p_ops_available() also sets AFD_CUST_OPAPI_LIB_PATH and the
# vendor paths so the aclnnA2e/aclnnE2a symbols are resolvable at runtime.
from afd_plugin.compat.npu.ops import ensure_cam_p2p_ops_available

ensure_cam_p2p_ops_available()
import afd_plugin._C_ascend  # noqa: E402, F401


class A2eE2aModule(torch.nn.Module):
    def forward(
        self,
        x,
        expert_ids,
        scales,
        batch_size,
        hidden_size,
        topk,
        expert_rank_size,
        attention_rank_size,
        rank,
        group_ep,
        aiv_num,
    ):
        is_attention_side = rank >= expert_rank_size

        if is_attention_side:
            # Attention side: send data to MOE side via A2E.
            a2e_output = torch.ops.afd_ascend.a2e(
                x=x,
                expert_ids=expert_ids,
                scales=scales,
                batch_size=batch_size,
                hidden_size=hidden_size,
                topk=topk,
                expert_rank_size=expert_rank_size,
                attention_rank_size=attention_rank_size,
                rank=rank,
                group_ep=group_ep,
                aiv_num=aiv_num,
                compute_gate=1,
            )
            (
                expand_x,
                simulate_expert_ids,
                simulate_expert_scales,
                atten_batch_size,
                x_active_mask_out,
            ) = a2e_output

            # Attention side: receive data back from MOE side via E2A.
            e2a_output = torch.ops.afd_ascend.e2a(
                expand_x=x,
                atten_batch_size=atten_batch_size,
                batch_size=batch_size,
                hidden_size=hidden_size,
                topk=topk,
                expert_rank_size=expert_rank_size,
                attention_rank_size=attention_rank_size,
                rank=rank,
                group_ep=group_ep,
                aiv_num=aiv_num,
            )

            return (
                e2a_output,
                expand_x,
                simulate_expert_ids,
                simulate_expert_scales,
                atten_batch_size,
                x_active_mask_out,
            )

        dummy_x = torch.empty(0, hidden_size, dtype=x.dtype, device=x.device)
        dummy_expert_ids = torch.empty(0, topk, dtype=torch.int32, device=x.device)
        dummy_scales = torch.empty(0, topk, dtype=torch.float, device=x.device)

        a2e_output = torch.ops.afd_ascend.a2e(
            x=dummy_x,
            expert_ids=dummy_expert_ids,
            scales=dummy_scales,
            batch_size=batch_size,
            hidden_size=hidden_size,
            topk=topk,
            expert_rank_size=expert_rank_size,
            attention_rank_size=attention_rank_size,
            rank=rank,
            group_ep=group_ep,
            aiv_num=aiv_num,
            compute_gate=1,
        )
        (
            expand_x,
            simulate_expert_ids,
            simulate_expert_scales,
            atten_batch_size,
            x_active_mask_out,
        ) = a2e_output

        e2a_output = torch.ops.afd_ascend.e2a(
            expand_x=expand_x,
            atten_batch_size=atten_batch_size,
            batch_size=batch_size,
            hidden_size=hidden_size,
            topk=topk,
            expert_rank_size=expert_rank_size,
            attention_rank_size=attention_rank_size,
            rank=rank,
            group_ep=group_ep,
            aiv_num=aiv_num,
        )

        return (
            e2a_output,
            expand_x,
            simulate_expert_ids,
            simulate_expert_scales,
            atten_batch_size,
            x_active_mask_out,
        )


def gen_x(rank, batch_size, hidden_size):
    """Generate input tensor data."""
    return [
        rank * batch_size + i + 1 for i in range(batch_size) for _ in range(hidden_size)
    ]


def gen_expert_ids(rank, batch_size, topk, expert_rank_size):
    """Generate expert indices data."""
    arr = [0] * (batch_size * topk)
    for i in range(batch_size):
        for j in range(topk):
            arr[i * topk + j] = (rank + i + j) % expert_rank_size
    return arr


def gen_scales(batch_size, topk):
    """Generate scaling factors data."""
    return [1.0 / topk] * (batch_size * topk)


def run_once(local_rank_id, ep_world_size):
    """Single run test function (1A1F by default; scale via env)."""
    devices = [
        int(d) for d in os.environ.get("AFD_TEST_DEVICES", "2,3").split(",") if d != ""
    ]
    assert len(devices) >= ep_world_size, (
        f"Need at least {ep_world_size} devices via AFD_TEST_DEVICES, got {devices}"
    )
    physical_dev = devices[local_rank_id]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("AFD_TEST_PORT", "29601")
    rank = local_rank_id
    world_size = ep_world_size

    torch.npu.set_device(physical_dev)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    # Test parameters. Default is 1A1F (expert_rank_size=1, attention_rank_size=1).
    batch_size = 16
    hidden_size = 512
    topk = 2
    expert_rank_size = int(os.environ.get("AFD_TEST_E", "1"))
    attention_rank_size = int(os.environ.get("AFD_TEST_A", expert_rank_size))
    aiv_num = 4
    data_type = torch.bfloat16

    is_attention_side = rank >= expert_rank_size

    # Communication domain: a single group of both ranks (1 EP + 1 Attn).
    ep_ranks_list = list(range(world_size))
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    torch.npu.synchronize()

    if is_attention_side:
        x_data = np.array(gen_x(rank, batch_size, hidden_size)).reshape(
            batch_size, hidden_size
        )
        x_tensor = torch.tensor(x_data, dtype=data_type, device="npu")

        expert_ids_data = np.array(
            gen_expert_ids(rank, batch_size, topk, expert_rank_size)
        ).reshape(batch_size, topk)
        expert_ids_tensor = torch.tensor(expert_ids_data, dtype=torch.int32, device="npu")

        scales_data = np.array(gen_scales(batch_size, topk)).reshape(batch_size, topk)
        scales_tensor = torch.tensor(scales_data, dtype=torch.float, device="npu")
    else:
        x_tensor = torch.empty(0, hidden_size, dtype=data_type, device="npu")
        expert_ids_tensor = torch.empty(0, topk, dtype=torch.int32, device="npu")
        scales_tensor = torch.empty(0, topk, dtype=torch.float, device="npu")

    mod = A2eE2aModule().npu()

    (
        e2a_output,
        expand_x,
        simulate_expert_ids,
        simulate_expert_scales,
        atten_batch_size,
        x_active_mask_out,
    ) = mod(
        x=x_tensor,
        expert_ids=expert_ids_tensor,
        scales=scales_tensor,
        batch_size=batch_size,
        hidden_size=hidden_size,
        topk=topk,
        expert_rank_size=expert_rank_size,
        attention_rank_size=attention_rank_size,
        rank=rank,
        group_ep=ep_hcomm_info,
        aiv_num=aiv_num,
    )

    torch.npu.synchronize()

    if is_attention_side:
        print(
            f"Attention Side Rank {rank} (dev{physical_dev}): "
            "A2E-E2A sample run completed!"
        )
        print(f"  Input shape: {x_tensor.shape}")
        print(f"  E2A output shape: {e2a_output.shape}")
        assert e2a_output.shape == x_tensor.shape, (
            f"E2A output shape mismatch: {e2a_output.shape} vs {x_tensor.shape}"
        )
        assert torch.allclose(e2a_output, x_tensor, atol=1e-3), (
            "E2A output does not match input x"
        )
        print("  Input and output are consistent!")
    else:
        print(
            f"MOE Side Rank {rank} (dev{physical_dev}): A2E-E2A sample run completed!"
        )
        print(f"  A2E expand_x shape: {expand_x.shape}")
        print(f"  A2E simulate_expert_ids shape: {simulate_expert_ids.shape}")
        print(f"  A2E simulate_expert_scales shape: {simulate_expert_scales.shape}")
        print(f"  A2E atten_batch_size shape: {atten_batch_size.shape}")
        print(f"  A2E x_active_mask_out shape: {x_active_mask_out.shape}")

    dist.destroy_process_group()


if __name__ == "__main__":
    e = int(os.environ.get("AFD_TEST_E", "1"))
    a = int(os.environ.get("AFD_TEST_A", e))
    ep_world_size = int(os.environ.get("AFD_TEST_WORLD", e + a))  # default 1A1F
    devices = os.environ.get("AFD_TEST_DEVICES", "2,3")
    if len([d for d in devices.split(",") if d]) < ep_world_size:
        print(
            f"Need >= {ep_world_size} devices for world_size={ep_world_size}; "
            "set AFD_TEST_DEVICES."
        )
        raise SystemExit(1)
    print("A2E-E2A A5 sample started!")
    print(f"Running with {ep_world_size} ranks (E={e}, A={a}), devices={devices}")
    mp.spawn(run_once, args=(ep_world_size,), nprocs=ep_world_size, join=True)
    print("A2E-E2A A5 sample completed successfully!")
