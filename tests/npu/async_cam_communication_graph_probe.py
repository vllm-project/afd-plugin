# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Two-rank CAM DR -> simple compute -> CS graph capture probe.

Run with torchrun --standalone --nproc-per-node=2. Rank 0 is Attention and
rank 1 is FFN. ``cold`` captures immediately; ``warmup`` completes one eager
CAM transaction on both ranks before capture. Both modes send a matched task
for graph replay only after capture completes.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - register torch.npu

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.distributed.afd_process_group import init_afd_process_group

HIDDEN_SIZE = 512
MAX_SEQUENCE_LENGTH = 262144
MAX_SEQ_LEN = 1
TOP_K = 1
ATTN_RANKS = 1
FFN_RANKS = 1
EXPERTS_PER_RANK = 1
DYNAMIC_QUANT = 1
COMM_ID = 0
RENDEZVOUS_TIMEOUT_SECONDS = 120


def log(rank: int, phase: str) -> None:
    print(f"rank={rank} {phase}", flush=True)


def attention_transaction(
    *, rank: int, group_name: str, comm_args: torch.Tensor, anchor: torch.Tensor
) -> None:
    ops = torch.ops.afd_ascend
    hidden = torch.full(
        (MAX_SEQ_LEN, HIDDEN_SIZE), 0.25, device="npu", dtype=torch.bfloat16
    )
    expert_ids = torch.zeros((MAX_SEQ_LEN, TOP_K), device="npu", dtype=torch.int32)
    expert_weights = torch.ones((MAX_SEQ_LEN, TOP_K), device="npu", dtype=torch.float32)
    log(rank, "DS start")
    ops.afd_async_dispatch_send(
        hidden,
        expert_ids,
        comm_args,
        COMM_ID,
        MAX_SEQ_LEN,
        MAX_SEQ_LEN,
        HIDDEN_SIZE,
        TOP_K,
        FFN_RANKS,
        ATTN_RANKS,
        EXPERTS_PER_RANK,
        rank,
        ATTN_RANKS + FFN_RANKS,
        0,
        ATTN_RANKS,
        DYNAMIC_QUANT,
        group_name,
    )
    log(rank, "DS returned; CR start")
    output = ops.afd_async_combine_recv(
        anchor,
        expert_ids,
        expert_weights,
        comm_args,
        COMM_ID,
        MAX_SEQ_LEN,
        HIDDEN_SIZE,
        TOP_K,
        FFN_RANKS,
        ATTN_RANKS,
        EXPERTS_PER_RANK,
        rank,
        ATTN_RANKS + FFN_RANKS,
        group_name,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(output, hidden, atol=0.01, rtol=0.01)
    log(rank, "CR complete")


def ffn_transaction(
    *, rank: int, group_name: str, comm_args: torch.Tensor, anchor: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    ops = torch.ops.afd_ascend
    log(rank, "DR start")
    expanded, scales, batch_info, counts = ops.afd_async_dispatch_recv(
        anchor,
        comm_args,
        COMM_ID,
        MAX_SEQ_LEN,
        HIDDEN_SIZE,
        TOP_K,
        FFN_RANKS,
        ATTN_RANKS,
        EXPERTS_PER_RANK,
        rank,
        ATTN_RANKS + FFN_RANKS,
        ATTN_RANKS,
        DYNAMIC_QUANT,
        group_name,
    )
    log(rank, "DR returned; compute start")
    output = (expanded.float() * scales.unsqueeze(1)).to(torch.bfloat16)
    log(rank, "compute returned; CS start")
    ops.afd_async_combine_send(
        output,
        comm_args,
        batch_info,
        COMM_ID,
        MAX_SEQ_LEN,
        HIDDEN_SIZE,
        TOP_K,
        FFN_RANKS,
        ATTN_RANKS,
        EXPERTS_PER_RANK,
        rank,
        ATTN_RANKS + FFN_RANKS,
        ATTN_RANKS,
        group_name,
    )
    log(rank, "CS returned")
    # Keep the graph's intermediate output and the receive metadata alive.
    return output, expanded, scales, batch_info, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cold", "warmup"), required=True)
    parser.add_argument("--cam-port", type=int, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != ATTN_RANKS + FFN_RANKS:
        raise ValueError("Probe requires exactly two ranks")
    os.environ["BATCH_SIZE_FACTOR"] = str(1 / MAX_SEQUENCE_LENGTH)
    torch.npu.set_device(local_rank)
    ensure_cam_async_ops_available()
    dist.init_process_group(
        "gloo", timeout=timedelta(seconds=RENDEZVOUS_TIMEOUT_SECONDS)
    )
    # torchrun sets this for its own agent store. The CAM tcp:// rendezvous
    # needs its own rank-0 server on --cam-port.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    cam_group = init_afd_process_group(
        backend="hccl",
        init_method=f"tcp://127.0.0.1:{args.cam_port}",
        world_size=ATTN_RANKS + FFN_RANKS,
        rank=rank,
        group_name="afd_async_cam_communication_graph_probe",
        timeout=timedelta(seconds=RENDEZVOUS_TIMEOUT_SECONDS),
    )
    group_name = str(
        cam_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    )
    comm_args = torch.empty((1,), dtype=torch.float16, device="npu")
    anchor = torch.empty((1,), dtype=torch.bfloat16, device="npu")
    log(rank, "CAM HCCL group initialized")

    with torch.inference_mode():
        if args.mode == "warmup":
            log(rank, "eager warmup start")
            if rank == 0:
                attention_transaction(
                    rank=rank, group_name=group_name, comm_args=comm_args, anchor=anchor
                )
            else:
                ffn_transaction(
                    rank=rank, group_name=group_name, comm_args=comm_args, anchor=anchor
                )
                torch.npu.synchronize()
            log(rank, "eager warmup complete")
        dist.barrier()

        graph = None
        references = None
        if rank == ATTN_RANKS:
            graph = torch.npu.NPUGraph()
            log(rank, "graph capture start")
            with torch.npu.graph(graph):
                log(rank, "graph context entered")
                references = ffn_transaction(
                    rank=rank, group_name=group_name, comm_args=comm_args, anchor=anchor
                )
            log(rank, "graph capture complete")
        dist.barrier()

        log(rank, "matched replay transaction start")
        if rank == 0:
            attention_transaction(
                rank=rank, group_name=group_name, comm_args=comm_args, anchor=anchor
            )
        else:
            assert graph is not None and references is not None
            graph.replay()
            torch.npu.synchronize()
            log(rank, "graph replay complete")
        dist.barrier()

    dist.destroy_process_group(cam_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
