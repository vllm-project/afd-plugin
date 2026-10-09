# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Two-rank CAM process-group graph bootstrap probe without CAM work items.

Launch with ``torchrun --nproc-per-node=2`` and pass a free ``--cam-port``.
The default torchrun rendezvous port must be different from ``--cam-port``.
This isolates HCCL startup work from dispatch-recv capture behavior.
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - register torch.npu before the probe uses it
from torch.distributed.distributed_c10d import Store

from afd_plugin.distributed.afd_process_group import init_afd_process_group

PROBE_TIMEOUT_SECONDS = 120
NONCE_BYTES = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-port", type=int, required=True)
    parser.add_argument("--skip-nonce", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError("This probe requires exactly two ranks")

    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="gloo", timeout=timedelta(seconds=PROBE_TIMEOUT_SECONDS)
    )
    # This second rendezvous owns its TCPStore; torchrun's agent serves only
    # the default group port, so rank zero must create the CAM store here.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    rendezvous_store: Store | None = None

    def retain_store(store: Store) -> None:
        nonlocal rendezvous_store
        rendezvous_store = store

    cam_group = init_afd_process_group(
        backend="hccl",
        init_method=f"tcp://127.0.0.1:{args.cam_port}",
        world_size=world_size,
        rank=rank,
        group_name="afd_async_cam_graph_bootstrap_probe",
        timeout=timedelta(seconds=PROBE_TIMEOUT_SECONDS),
        on_rendezvous=retain_store,
    )
    if rendezvous_store is None:
        raise RuntimeError("CAM rendezvous store was not retained")
    print(f"rank={rank} HCCL group initialized", flush=True)

    if not args.skip_nonce:
        nonce = torch.zeros(NONCE_BYTES, dtype=torch.uint8, device=f"npu:{local_rank}")
        if rank == 0:
            nonce.copy_(
                torch.tensor(
                    list(os.urandom(NONCE_BYTES)),
                    dtype=torch.uint8,
                    device=nonce.device,
                )
            )
        dist.broadcast(nonce, src=0, group=cam_group)
        print(f"rank={rank} nonce broadcast returned", flush=True)
        torch.npu.synchronize()
        print(f"rank={rank} nonce NPU synchronize returned", flush=True)

    graph = torch.npu.NPUGraph()
    value = torch.ones((1,), device=f"npu:{local_rank}")
    print(f"rank={rank} graph capture start", flush=True)
    with torch.npu.graph(graph):
        print(f"rank={rank} graph context entered", flush=True)
        result = value + 1
    print(f"rank={rank} graph capture complete", flush=True)
    graph.replay()
    torch.npu.synchronize()
    if result.item() != 2:
        raise AssertionError(f"Unexpected graph result on rank {rank}")
    print(f"rank={rank} graph replay complete", flush=True)
    dist.destroy_process_group(cam_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
