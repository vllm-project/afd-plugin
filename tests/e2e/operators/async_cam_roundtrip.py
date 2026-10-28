# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Independent source-operator numerical check, launched with torchrun.

Example (six NPUs: Attention TP4 + FFN EP2)::

    torchrun --standalone --nproc-per-node=6 \
        -m tests.e2e.operators.async_cam_roundtrip \
        --tp 4 --dtype bfloat16 --dynamic-quant 1

Run TP1/2/4, float16/bfloat16, dynamic-quant 0/1 separately. Every run covers
expert zero/last, sparse routes, an empty FFN rank, multiple receive chunks,
single-token decode, and repeated window reuse. Tolerances are fixed below.

The 16-NPU DSV4 topology can be checked with ``--attention-dp 2
--ffn-ranks 8 --experts-per-rank 32 --tp 4``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

HIDDEN_SIZE = 512
FFN_RANKS = 2
EXPERTS_PER_RANK = 4
TOP_K = 2
BATCH_SIZE = 8
REPETITIONS = 2
CASES = ("edge", "empty-rank-multichunk", "sparse", "decode")
MAX_CAPACITY = 262144
# BF16 arithmetic rounds each expert output before the FP32 weighted sum.
TOLERANCES = {"float16": (0.008, 0.004), "bfloat16": (0.06, 0.02)}


def make_input(
    rank: int,
    case: str,
    dtype: torch.dtype,
    *,
    ffn_ranks: int,
    experts_per_rank: int,
    hidden_size: int,
    top_k: int,
):
    batch = 1 if case == "decode" else BATCH_SIZE
    values = torch.arange(batch * hidden_size, dtype=torch.float32).reshape(
        batch, hidden_size
    )
    x = (torch.sin(values * 0.013 + rank) * 0.5).to(dtype)
    ids = torch.empty((batch, top_k), dtype=torch.int32)
    expert_offsets = torch.arange(top_k, dtype=torch.int32)
    total_experts = ffn_ranks * experts_per_rank
    if case == "empty-rank-multichunk":
        ids[:] = expert_offsets.remainder(experts_per_rank)
        ids[:, -1] = experts_per_rank - 1
    elif case == "sparse":
        # Reserve the fixed final expert: duplicate TopK routes can make its
        # count exceed the one-expert receive capacity.
        ids[:] = (
            torch.arange(batch, dtype=torch.int32)[:, None]
            + rank
            + expert_offsets[None, :] * experts_per_rank
        ).remainder(total_experts - 1)
        ids[:, -1] = total_experts - 1
    else:
        ids[:] = expert_offsets.remainder(total_experts)
        ids[:, -1] = total_experts - 1
    weights = torch.arange(1, top_k + 1, dtype=torch.float32)
    weights = (weights / weights.sum()).repeat(batch, 1)
    return x, ids, weights


def reference_output(x, ids, weights, dynamic_quant):
    """CPU oracle, independent of dispatch counts and received activations."""
    values = x.float()
    if dynamic_quant:
        scale = values.abs().amax(dim=-1, keepdim=True) / 127
        values = (values / scale).round().clamp(-127, 127) * scale
    expert_outputs = (values[:, None, :] * (ids.float() + 1)[:, :, None]).to(x.dtype)
    return (expert_outputs.float() * weights[:, :, None]).sum(dim=1).to(x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--dtype", choices=tuple(TOLERANCES), required=True)
    parser.add_argument("--dynamic-quant", type=int, choices=(0, 1), required=True)
    parser.add_argument("--attention-dp", type=int, default=1)
    parser.add_argument("--ffn-ranks", type=int, default=FFN_RANKS)
    parser.add_argument("--experts-per-rank", type=int, default=EXPERTS_PER_RANK)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--receive-capacity", type=int)
    parser.add_argument("--zero-shards", action="store_true")
    parser.add_argument(
        "--fault", choices=("return-epoch", "return-counts", "return-rows")
    )
    args = parser.parse_args()
    if args.pull and (args.dtype != "bfloat16" or args.dynamic_quant != 1):
        parser.error("pull requires bfloat16 and dynamic-quant=1")
    if args.zero_shards and not args.pull:
        parser.error("zero shards require --pull")
    if args.fault and not args.pull:
        parser.error("fault injection requires --pull")
    assert args.attention_dp > 0 and args.ffn_ranks > 0 and args.experts_per_rank > 0
    assert args.hidden_size > 0 and 1 < args.top_k <= args.experts_per_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    attention_ranks = args.tp * args.attention_dp
    world_size = attention_ranks + args.ffn_ranks
    assert int(os.environ["WORLD_SIZE"]) == world_size
    torch.npu.set_device(local_rank)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # One expert fits; two fully populated experts require separate chunks.
    max_seq_len = args.tp * BATCH_SIZE
    capacity = args.receive_capacity or max_seq_len
    assert 0 < capacity <= MAX_CAPACITY
    os.environ["BATCH_SIZE_FACTOR"] = str(capacity / MAX_CAPACITY)
    ensure_cam_async_ops_available()
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    group_name = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(
        rank
    )
    comm = torch.empty(1, dtype=torch.float16, device="npu")
    anchor = torch.empty(1, dtype=dtype, device="npu")
    ops = torch.ops.afd_ascend
    cases = CASES + (("zero-shard", "all-zero") if args.zero_shards else ())
    if args.fault:
        cases = ("edge",)
    comm_id = 0
    if args.pull:
        comm_id = 0x41464401
        comm = ops.afd_mega_pull_init(
            anchor,
            [
                BATCH_SIZE,
                args.hidden_size,
                args.top_k,
                args.ffn_ranks,
                attention_ranks,
                args.experts_per_rank,
                rank,
                args.tp,
            ],
            list(range(len(cases) * REPETITIONS)) if rank >= attention_ranks else [],
            1,
            group_name,
        )
        torch.npu.synchronize()
        dist.barrier()
    reports = []
    try:
        with torch.inference_mode():
            for iteration, case in enumerate(
                cases * (1 if args.fault else REPETITIONS)
            ):
                dist.barrier()
                if rank < attention_ranks:
                    x_cpu, ids_cpu, weights_cpu = make_input(
                        rank,
                        case,
                        dtype,
                        ffn_ranks=args.ffn_ranks,
                        experts_per_rank=args.experts_per_rank,
                        hidden_size=args.hidden_size,
                        top_k=args.top_k,
                    )
                    if case == "all-zero" or (
                        case == "zero-shard" and rank % args.tp == 0
                    ):
                        x_cpu, ids_cpu, weights_cpu = (
                            x_cpu[:0],
                            ids_cpu[:0],
                            weights_cpu[:0],
                        )
                    x, ids, weights = x_cpu.npu(), ids_cpu.npu(), weights_cpu.npu()
                    (
                        ops.afd_mega_dispatch_send
                        if args.pull
                        else ops.afd_async_dispatch_send
                    )(
                        x,
                        ids,
                        comm,
                        comm_id,
                        max_seq_len,
                        x.shape[0],
                        args.hidden_size,
                        args.top_k,
                        args.ffn_ranks,
                        attention_ranks,
                        args.experts_per_rank,
                        rank,
                        world_size,
                        iteration,
                        args.tp,
                        args.dynamic_quant,
                        group_name,
                    )
                    output = (
                        ops.afd_mega_combine_recv
                        if args.pull
                        else ops.afd_async_combine_recv
                    )(
                        anchor,
                        ids,
                        weights,
                        comm,
                        comm_id,
                        x.shape[0],
                        args.hidden_size,
                        args.top_k,
                        args.ffn_ranks,
                        attention_ranks,
                        args.experts_per_rank,
                        rank,
                        world_size,
                        group_name,
                        *((max_seq_len, args.tp) if args.pull else ()),
                    ).cpu()
                    expected = reference_output(
                        x_cpu, ids_cpu, weights_cpu, args.dynamic_quant
                    )
                    atol, rtol = TOLERANCES[args.dtype]
                    if args.fault:
                        assert comm.view(torch.int64)[24].item() == -9
                    else:
                        torch.testing.assert_close(
                            output, expected, atol=atol, rtol=rtol
                        )
                    reports.append(
                        {
                            "case": case,
                            "iteration": iteration,
                            "fault": args.fault,
                            "max_abs_error": None
                            if args.fault
                            else (
                                (output.float() - expected.float()).abs().max().item()
                                if output.numel()
                                else 0.0
                            ),
                        }
                    )
                else:
                    chunks = 0
                    completed_dp_groups = 0
                    while completed_dp_groups < args.attention_dp:
                        expanded, scales, batch_info, counts = (
                            ops.afd_mega_dispatch_recv
                            if args.pull
                            else ops.afd_async_dispatch_recv
                        )(
                            anchor,
                            comm,
                            comm_id,
                            max_seq_len,
                            args.hidden_size,
                            args.top_k,
                            args.ffn_ranks,
                            attention_ranks,
                            args.experts_per_rank,
                            rank,
                            world_size,
                            args.tp,
                            args.dynamic_quant,
                            group_name,
                        )
                        count_list = counts.cpu().tolist()
                        header = batch_info[:5].cpu().tolist()
                        assert header[2] == iteration, (
                            rank,
                            iteration,
                            header,
                            count_list,
                        )
                        num_tokens = sum(count_list)
                        values = expanded[:num_tokens].float()
                        if args.dynamic_quant:
                            values = values * scales[:num_tokens, None]
                        offset = 0
                        for expert, count in enumerate(count_list):
                            multiplier = (
                                (rank - attention_ranks) * args.experts_per_rank
                                + expert
                                + 1
                            )
                            values[offset : offset + count] *= multiplier
                            offset += count
                        result = values.to(dtype).contiguous()
                        if not num_tokens:
                            result = torch.zeros(
                                (1, args.hidden_size), dtype=dtype, device="npu"
                            )
                        inject = args.fault and rank == attention_ranks
                        if inject:
                            if args.fault == "return-epoch":
                                batch_info[-2].add_(1)
                            elif args.fault == "return-counts":
                                batch_info[5 + args.tp].add_(1)
                            else:
                                result = result[:1]
                        (
                            ops.afd_mega_combine_send
                            if args.pull
                            else ops.afd_async_combine_send
                        )(
                            result,
                            comm,
                            batch_info,
                            comm_id,
                            max_seq_len,
                            args.hidden_size,
                            args.top_k,
                            args.ffn_ranks,
                            attention_ranks,
                            args.experts_per_rank,
                            rank,
                            world_size,
                            args.tp,
                            group_name,
                        )
                        chunks += 1
                        if inject:
                            assert comm.view(torch.int64)[24].item() == -9
                            break
                        complete = (
                            header[4] >= header[0]
                            if args.pull
                            else header[4] == args.experts_per_rank - 1
                        )
                        if complete:
                            completed_dp_groups += 1
                    if case == "empty-rank-multichunk" and rank == attention_ranks:
                        assert chunks > 1
                    reports.append(
                        {"case": case, "iteration": iteration, "chunks": chunks}
                    )
                torch.npu.synchronize()
                dist.barrier()
        print(json.dumps({"rank": rank, **vars(args), "passed": reports}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
