# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""End-to-end pass over the async GPU connector's public API, two processes.

Rank 0 runs the Attention side (``send_attn_output`` / ``recv_ffn_output``),
rank 1 runs the FFN side (``recv_ffn_work_item`` / ``send_ffn_work_item_output``)
with the real grouped-GEMM helper. Everything between the gate and the combined
result is exercised: routing, one-sided dispatch, local expert compute, the
write-back, and the weighted reduction.

It runs twice over: first eagerly, then with the Attention half captured into a
CUDA graph and replayed. The replays are what pin the flag protocol down. A
replay runs no Python, so a dispatch sequence number kept on the host and a
stream wait told to expect one would both freeze at whatever was live during
capture -- the wait would then find a flag already holding the value it wants,
fall straight through, and combine the previous replay's data. That failure is
silent, so each replay is fed different tokens and checked against its own
reference, which is what turns it into an assertion.

Run with two GPUs::

    python tests/e2e/async_gpu_connector_e2e.py

Deliberately *not* launched with torchrun. ``init_afd_process_group`` builds its
own TCPStore on the AFD port, and under torchelastic every rank is forced to
``is_master=False`` (``torch/distributed/rendezvous.py:188``), so no rank hosts
the store and the group never forms. Production launches the two roles as
separate ``vllm serve`` processes, which this mirrors.
"""

import multiprocessing as mp
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from afd_plugin.model_executor.models.gpu.deepseek_v2_attention_gate import (
    compute_attention_gate_moe_ffn,
)

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.gpu.async_gpu import GpuAsyncAFDConnector
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata

NUM_TOKENS = 48
HIDDEN = 128
INTERMEDIATE = 256
TOPK = 4
NUM_EXPERTS = 8
NUM_LAYERS = 3
PORT = 29655
WORLD_PORT = 29656
SCALING = 1.7
# One eager pass on a side stream before capture, the usual prerequisite: the
# caching allocator and the window's cached views have to be warm, and a graph
# cannot be captured off a cold stream.
GRAPH_WARMUPS = 1
GRAPH_REPLAYS = 4
# Capture itself records kernels without running them, so it asks nothing of
# the FFN rank; only the eager layers, the warmup and the replays do.
FFN_WORK_ITEMS = NUM_LAYERS + GRAPH_WARMUPS + GRAPH_REPLAYS


def build_connector(role: str, local_rank: int) -> GpuAsyncAFDConnector:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=HIDDEN,
                num_experts_per_tok=TOPK,
                n_routed_experts=NUM_EXPERTS,
                # This test's FFN loop runs routed experts only.
                n_shared_experts=0,
            ),
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=NUM_TOKENS),
        additional_config={
            "afd": {
                "role": role,
                "connector": "GpuAsyncAFDConnector",
                "async": True,
                "compute_gate_on_attention": True,
                "num_attention_ranks": 1,
                "num_ffn_ranks": 1,
                "port": PORT,
                "connector_extra_config": {"ring_depth": 1},
            },
        },
    )
    afd_config = AFDConfig(
        role=role,
        connector="GpuAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        num_attention_ranks=1,
        num_ffn_ranks=1,
        host="127.0.0.1",
        port=PORT,
    )
    connector = GpuAsyncAFDConnector(
        rank=local_rank,
        local_rank=local_rank,
        vllm_config=vllm_config,
        afd_config=afd_config,
        role_rank=0,
    )
    connector.init_afd_connector()
    return connector


def make_weights(device):
    generator = torch.Generator(device="cpu").manual_seed(11)
    w13 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, generator=generator)
        / HIDDEN**0.5
    ).to(device, torch.bfloat16)
    w2 = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator=generator)
        / INTERMEDIATE**0.5
    ).to(device, torch.bfloat16)
    return w13, w2


def make_layer_inputs(layer_idx, device):
    gen = torch.Generator(device="cpu").manual_seed(100 + layer_idx)
    x = torch.randn(NUM_TOKENS, HIDDEN, generator=gen).to(device, torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=gen)[:TOPK] for _ in range(NUM_TOKENS)],
    ).to(device, torch.int32)
    topk_weights = torch.rand(NUM_TOKENS, TOPK, generator=gen).to(device)
    return x, topk_ids, topk_weights


def reference_moe(x, w13, w2, topk_ids, topk_weights):
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            hidden = x[token].to(torch.float32) @ w13[expert].to(torch.float32).T
            gate, up = hidden.chunk(2, dim=-1)
            y = (F.silu(gate) * up) @ w2[expert].to(torch.float32).T
            out[token] += float(topk_weights[token, slot]) * y * SCALING
    return out


def init_world(rank: int) -> None:
    """Mimic a single `vllm serve`: a private default group of size 1.

    The connector bootstraps NVSHMEM on the AFD group itself, so the default
    group deliberately does *not* span both roles -- that is the topology the
    real deployment has.
    """
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{WORLD_PORT + rank}",
        world_size=1,
        rank=0,
    )


def run_graph_phase(connector, device, w13, w2) -> None:
    """Capture one dispatch, then replay it against tokens it never saw.

    A graph replays out of the buffers it was captured with, so the tokens and
    the routing are copied into fixed tensors rather than handed in as new
    ones. Everything else is the same call pair the eager loop above makes.
    """
    x = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
    topk_ids = torch.zeros(NUM_TOKENS, TOPK, dtype=torch.int32, device=device)
    topk_weights = torch.zeros(NUM_TOKENS, TOPK, dtype=torch.float32, device=device)

    def dispatch():
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=0,
                stage_idx=0,
                seq_len=NUM_TOKENS,
            ),
        )
        connector.send_attn_output(
            x,
            context,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        return connector.recv_ffn_output(ref_tensor=x, ubatch_idx=0)

    def load(iteration: int):
        """Fill the captured buffers with one iteration's own tokens."""
        tokens, ids, weights = make_layer_inputs(NUM_LAYERS + iteration, device)
        x.copy_(tokens)
        topk_ids.copy_(ids)
        topk_weights.copy_(weights)
        return reference_moe(tokens, w13, w2, ids, weights)

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    for iteration in range(GRAPH_WARMUPS):
        expected = load(iteration)
        with torch.cuda.stream(warmup_stream):
            got = dispatch()
        warmup_stream.synchronize()
        torch.testing.assert_close(
            got.to(torch.float32),
            expected,
            rtol=8e-2,
            atol=8e-2,
        )
        print(f"[A] graph warmup {iteration}: eager combine matches", flush=True)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combined = dispatch()
    print("[A] captured the attention dispatch", flush=True)

    for replay in range(GRAPH_REPLAYS):
        expected = load(GRAPH_WARMUPS + replay)
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            combined.to(torch.float32),
            expected,
            rtol=8e-2,
            atol=8e-2,
        )
        print(f"[A] replay {replay}: combine matches this replay's tokens", flush=True)


def run_attention(rank: int) -> None:
    init_world(0)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    connector = build_connector("attention", rank)
    w13, w2 = make_weights(device)

    for layer_idx in range(NUM_LAYERS):
        x, topk_ids, topk_weights = make_layer_inputs(layer_idx, device)
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=layer_idx,
                stage_idx=0,
                seq_len=NUM_TOKENS,
            ),
        )
        connector.send_attn_output(
            x,
            context,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        got = connector.recv_ffn_output(ref_tensor=x, ubatch_idx=0)
        expected = reference_moe(x, w13, w2, topk_ids, topk_weights)
        torch.testing.assert_close(
            got.to(torch.float32),
            expected,
            rtol=8e-2,
            atol=8e-2,
        )
        print(
            f"[A] layer {layer_idx}: combined output matches reference MoE", flush=True
        )

    run_graph_phase(connector, device, w13, w2)

    print("PASS: async GPU connector end-to-end, eager and CUDA graph", flush=True)
    connector.close()


def run_ffn(rank: int) -> None:
    init_world(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    connector = build_connector("ffn", rank)
    w13, w2 = make_weights(device)
    layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=SimpleNamespace(
                routed_experts=SimpleNamespace(w13_weight=w13, w2_weight=w2),
                _shared_experts=None,
                routed_scaling_factor=SCALING,
            ),
        ),
    )

    for _ in range(FFN_WORK_ITEMS):
        while True:
            try:
                work_item = connector.recv_ffn_work_item(
                    stage_idx=0,
                    max_num_tokens=NUM_TOKENS,
                )
                break
            except TimeoutError:
                continue
        states = work_item.context.states
        payload = compute_attention_gate_moe_ffn(
            layer,
            hidden_states=work_item.hidden_states,
            group_list=states.group_list,
            expand_x_shared=None,
        )
        connector.send_ffn_work_item_output(work_item, payload)
        print(
            f"[F] layer {work_item.layer_idx}: served "
            f"{work_item.num_tokens} routed tokens",
            flush=True,
        )

    connector.close()


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("this test needs two visible GPUs")
    mp.set_start_method("spawn", force=True)
    procs = [
        mp.Process(target=run_ffn, args=(1,)),
        mp.Process(target=run_attention, args=(0,)),
    ]
    for proc in procs:
        proc.start()
    failed = False
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode != 0:
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
