# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real two-node transport oracle; launch one Slurm task per allocated GPU.

Reuse the single-node affine oracle, including zero-route destinations, BF16
cancellation, independent producers and two-stage CUDA Graph counter overflow.
Slurm's --kill-on-bad-exit=1 bounds failure cleanup across hosts.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_process_group_store

from tests.gpu.test_async_gpu import (
    PROCESS_TIMEOUT_S,
    _connector,
    _free_port,
    _run_attention,
    _run_ffn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=6281)
    parser.add_argument("--attention-ranks", type=int, default=2)
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--replays", type=int, default=64)
    parser.add_argument("--interleave-roles", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("eager", "graph", "precision", "cross_rank_precision", "independent"),
        default="graph",
    )
    args = parser.parse_args()
    faulthandler.dump_traceback_later(PROCESS_TIMEOUT_S, exit=True)
    rank = int(os.environ["SLURM_PROCID"])
    if args.interleave_roles:
        assert args.attention_ranks == args.ffn_ranks
        rank = rank // 2 if rank % 2 == 0 else args.attention_ranks + rank // 2
    assert int(os.environ["SLURM_NTASKS"]) == args.attention_ranks + args.ffn_ranks
    device_index = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(device_index)
    # The default group intentionally covers one worker, as in separate vLLM
    # role processes. Only the connector's AFD group spans the remote workers.
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        rank=0,
        world_size=1,
    )
    attention = rank < args.attention_ranks
    role_rank = rank if attention else rank - args.attention_ranks
    connector = _connector(
        "attention" if attention else "ffn",
        device_index,
        role_rank,
        args.port,
        args.attention_ranks,
        args.ffn_ranks,
        args.host,
    )
    connector.init_afd_connector()
    assert connector.pg is not None
    hosts = [None] * (args.attention_ranks + args.ffn_ranks)
    dist.all_gather_object(hosts, socket.gethostname(), group=connector.pg)
    assert len(set(hosts)) > 1, "this oracle requires GPUs on distinct hosts"
    window = connector.window
    assert window is not None
    assert window.remote_peers, "no remote peers: this run cannot qualify RMA"
    store = _get_process_group_store(connector.pg)
    if attention:
        if args.mode == "independent" and role_rank > 0:
            store.wait(["oracle/a0-finished"], timedelta(seconds=120))
        mode = "eager" if args.mode == "independent" and role_rank == 0 else args.mode
        _run_attention(
            connector, torch.device("cuda", device_index), mode, args.replays
        )
        if role_rank == 0:
            store.set("oracle/a0-finished", "1")
    else:
        _run_ffn(connector, role_rank, args.ffn_ranks, args.mode)
    connector.close()
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    print(
        json.dumps(
            {
                "rank": rank,
                "host": socket.gethostname(),
                "mode": args.mode,
                "replays": args.replays,
                "status": "passed",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
