# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real CAM echo/receiver-stop probe, without model or expert weights.

Run with torchrun --nproc-per-node=3 -m tests.e2e.operators.async_cam_stop
--attention-ranks=1 --ffn-ranks=2. Use four processes for attention-ranks=2.
This qualifies transport and orderly receive exit, not DSpark model numerics.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from afd_plugin.async_cam_startup import FFNGraphPlan
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam import (
    RECV_STOPPED,
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)

HIDDEN_SIZE = 64
TOKENS = 4
ROUNDS = 3
EXPERTS_PER_FFN = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--dynamic-quant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--idle-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.idle_seconds < 0:
        parser.error("idle-seconds must be nonnegative")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = args.attention_ranks + args.ffn_ranks
    assert world == int(os.environ["WORLD_SIZE"])
    is_ffn = rank >= args.attention_ranks
    os.environ["BATCH_SIZE_FACTOR"] = str(64 / 262144)
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.idle_seconds + 120))
    print(json.dumps({"rank": rank, "stage": "default-group-ready"}), flush=True)
    port = [None]
    if rank == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port[0] = sock.getsockname()[1]
    dist.broadcast_object_list(port, src=0)
    config = SimpleNamespace(
        additional_config={
            "afd": {
                "role": "ffn" if is_ffn else "attention",
                "connector": "CAMAsyncAFDConnector",
                "host": "127.0.0.1",
                "port": port[0],
                "async": True,
                "compute_gate_on_attention": True,
                "num_attention_ranks": args.attention_ranks,
                "num_ffn_ranks": args.ffn_ranks,
                "connector_extra_config": {
                    "attn_ranks_per_dp": 1,
                    "dynamicQuant": args.dynamic_quant,
                    "hccl_buffer_size": 64,
                },
            }
        },
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                hidden_size=HIDDEN_SIZE,
                num_experts_per_tok=1,
                n_routed_experts=EXPERTS_PER_FFN * args.ffn_ranks,
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
    )
    connector = CAMAsyncAFDConnector(
        rank,
        local_rank,
        config,
        parse_afd_config(config),
        rank - args.attention_ranks if is_ffn else rank,
    )
    if is_ffn:
        # Explicitly opt the protocol probe into the same negotiated path used
        # by the DSpark legacy runner, without constructing a draft model.
        connector.graph_startup.plan = FFNGraphPlan(
            cancellable_eager=True, layer_ids=(43, 44, 45)
        )
    # torchrun owns the default rendezvous Store. CAM has a separate tcp://
    # rendezvous, for which rank 0 must create a server at the chosen new port.
    previous_agent_store = os.environ.get("TORCHELASTIC_USE_AGENT_STORE")
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    try:
        connector.init_afd_connector()
    finally:
        if previous_agent_store is None:
            os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        else:
            os.environ["TORCHELASTIC_USE_AGENT_STORE"] = previous_agent_store
    print(json.dumps({"rank": rank, "stage": "cam-ready"}), flush=True)
    report = {
        "rank": rank,
        "dynamic_quant": args.dynamic_quant,
        "echo_returns": 0,
        "queued_stops": 0,
        "idle_returns": 0,
        "warmup_returns": 0,
    }
    errors = []

    def echo() -> None:
        try:
            torch.npu.set_device(local_rank)
            while True:
                item = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=64)
                if item is None:
                    if connector.receiver_stopped:
                        break
                    report["idle_returns"] += 1
                    if report["idle_returns"] == 1:
                        print(
                            json.dumps({"rank": rank, "stage": "idle-returned"}),
                            flush=True,
                        )
                    continue
                # CAM dispatch restores rows by expert, and combine restores
                # their original positions. Echo therefore tests routing too.
                output = item.hidden_states
                if args.dynamic_quant:
                    state = item.context.states
                    assert isinstance(state, AFDAsyncTransferState)
                    assert state.dynamic_scales is not None
                    output = (
                        output.float() * state.dynamic_scales[: item.num_tokens, None]
                    ).to(torch.bfloat16)
                connector.send_ffn_work_item_output(item, output)
                if report["warmup_returns"] < args.attention_ranks:
                    report["warmup_returns"] += 1
                    if report["warmup_returns"] == args.attention_ranks:
                        torch.npu.synchronize()
                        connector.graph_startup.phase("eager-warmed")
                else:
                    report["echo_returns"] += 1
            queued = [connector.recv_attn_output() for _ in range(2)]
            for payload in queued:
                state = payload.context.states
                assert isinstance(state, AFDAsyncTransferState)
                assert state.recv_status is not None
                assert state.recv_status.item() == RECV_STOPPED
                report["queued_stops"] += 1
            connector.finish_receiver_shutdown()
        except Exception as exc:
            errors.append(exc)
            connector.graph_startup.fail(exc)

    if is_ffn:
        receiver = threading.Thread(
            target=echo, name="receiver-stop-probe", daemon=True
        )
        receiver.start()
    else:
        receiver = None
        time.sleep(args.idle_seconds)
        for step in range(ROUNDS):
            hidden = (
                torch.arange(TOKENS * HIDDEN_SIZE, device=device, dtype=torch.float32)
                .remainder(16)
                .div(16)
                .reshape(TOKENS, HIDDEN_SIZE)
                .to(torch.bfloat16)
            )
            hidden += rank + step
            experts = torch.full(
                (TOKENS, 1),
                rank % (EXPERTS_PER_FFN * args.ffn_ranks),
                device=device,
                dtype=torch.int32,
            )
            weights = torch.ones(TOKENS, 1, device=device, dtype=torch.float32)
            context = AFDTransferContext(
                AFDTransferMetadata.create_attention_metadata(
                    layer_idx=43 + step, stage_idx=0, seq_len=TOKENS
                )
            )
            connector.send_attn_output(
                hidden, context, topk_ids=experts, topk_weights=weights
            )
            actual = connector.recv_ffn_output(hidden, context=context)
            torch.testing.assert_close(
                actual,
                hidden,
                rtol=0.01 if args.dynamic_quant else 0,
                atol=0.04 if args.dynamic_quant else 0,
            )
            report["echo_returns"] += 1
        time.sleep(min(args.idle_seconds, 1.0))
        connector.drain_attention()
    if receiver is not None:
        receiver.join(timeout=args.idle_seconds + 60)
        if receiver.is_alive():
            raise RuntimeError("Receiver did not exit; resources retained")
        if errors:
            raise errors[0]
        assert report["queued_stops"] == 2
        assert report["echo_returns"] == args.attention_ranks * ROUNDS
        if args.idle_seconds:
            assert report["idle_returns"] > 0
    connector.close()
    dist.barrier()
    print(json.dumps({**report, "normal_shutdown": True}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
