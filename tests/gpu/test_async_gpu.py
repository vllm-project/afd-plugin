# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real NVLink transport checks; native expert-kernel/E2E qualification is separate."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import traceback
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

NUM_TOKENS = 7
HIDDEN_SIZE = 16
NUM_EXPERTS = 4
TOP_K = 2
NUM_STAGES = 2
GRAPH_REPLAYS = 3
PROCESS_TIMEOUT_S = 120
SHARED_SCALE = 0.25
CANCELLATION_FACTOR = 256
WEIGHT_PERTURBATION = 0.0001


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _connector(
    role, device_index, role_rank, port, attention_ranks, ffn_ranks, host="127.0.0.1"
):
    from afd_plugin.config import AFDConfig
    from afd_plugin.connectors.gpu.async_gpu import GpuAsyncAFDConnector

    afd = AFDConfig(
        connector="GpuAsyncAFDConnector",
        async_dp=True,
        role=role,
        port=port,
        host=host,
        compute_gate_on_attention=True,
        num_attention_ranks=attention_ranks,
        num_ffn_ranks=ffn_ranks,
    )
    config = SimpleNamespace(
        additional_config={
            "afd": {"connector_extra_config": {}},
        },
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                hidden_size=HIDDEN_SIZE,
                num_experts_per_tok=TOP_K,
                n_routed_experts=NUM_EXPERTS,
                n_shared_experts=1,
                num_hidden_layers=1,
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=NUM_STAGES,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=NUM_TOKENS),
    )
    config.model_config.hf_text_config = config.model_config.hf_config
    return GpuAsyncAFDConnector(device_index, device_index, config, afd, role_rank)


def _inputs(stage: int, iteration: int, device, mode="eager"):
    values = torch.arange(NUM_TOKENS * HIDDEN_SIZE, device=device)
    hidden = (values.reshape(NUM_TOKENS, HIDDEN_SIZE) % 17 + iteration + stage) / 16
    hidden = hidden.to(torch.bfloat16)
    ids = torch.tensor([[stage * TOP_K, stage * TOP_K + 1]], device=device)
    if mode == "cross_rank_precision":
        ids = torch.tensor([[0, 2]], device=device)
    ids = ids.expand(NUM_TOKENS, -1).contiguous().to(torch.int32)
    weights = torch.full((NUM_TOKENS, TOP_K), 0.5, device=device)
    if mode == "precision":
        # These differ in FP32 but round to the same BF16 weight. Opposite
        # expert contributions make a lost routing weight observable.
        weights[:, 0] = 0.5 + WEIGHT_PERTURBATION
        weights[:, 1] = 0.5 - WEIGHT_PERTURBATION
    return hidden, ids, weights


def _roundtrip(connector, inputs, streams):
    from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata

    for stage, (hidden, ids, weights) in enumerate(inputs):
        with torch.cuda.stream(streams[stage]):
            connector.send_attn_output(
                hidden,
                AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=0,
                        stage_idx=stage,
                        seq_len=hidden.shape[0],
                    )
                ),
                topk_ids=ids,
                topk_weights=weights,
            )
    outputs = []
    for stage, (hidden, _, _) in enumerate(inputs):
        with torch.cuda.stream(streams[stage]):
            outputs.append(
                connector.recv_ffn_output(ref_tensor=hidden, ubatch_idx=stage)
            )
    return outputs


def _assert_outputs(outputs, inputs, mode):
    for output, (hidden, ids, weights) in zip(outputs, inputs, strict=True):
        factors = (
            torch.where(ids % 2 == 0, CANCELLATION_FACTOR, -CANCELLATION_FACTOR)
            if mode == "precision"
            else ids.float() + 1
        )
        if mode == "cross_rank_precision":
            factors = torch.where(ids == 0, CANCELLATION_FACTOR, -CANCELLATION_FACTOR)
        factor = (factors * weights).sum(1, keepdim=True) + SHARED_SCALE
        expected = hidden.float() * factor
        # BF16 wire replies and the final output each round, including across peers.
        torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


def _run_attention(connector, device, mode, replays=GRAPH_REPLAYS):
    if mode == "graph":
        # Cross the last dispatch marker during replay, without billions of
        # layer calls. Results and the producer shutdown handshake are the oracle.
        connector._seq_device.fill_(torch.iinfo(torch.int32).max - 2)
    streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STAGES)]
    inputs = [_inputs(stage, 0, device, mode) for stage in range(NUM_STAGES)]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    outputs = _roundtrip(connector, inputs, streams)
    for stream in streams:
        stream.synchronize()
    _assert_outputs(outputs, inputs, mode)
    if mode in ("eager", "precision", "cross_rank_precision"):
        return

    # A single graph records two distinct stages. Each replay changes the input
    # values and the routing weights, so stale replies cannot match the oracle.
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_stream = torch.cuda.current_stream(device)
        outputs = _roundtrip(connector, inputs, [capture_stream] * NUM_STAGES)
    for iteration in range(1, replays + 1):
        for stage, (hidden, ids, weights) in enumerate(inputs):
            changed, changed_ids, changed_weights = _inputs(stage, iteration, device)
            hidden.copy_(changed)
            ids.copy_(changed_ids)
            weights.copy_(changed_weights * iteration)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_outputs(outputs, inputs, mode)


def _run_ffn(connector, role_rank, ffn_ranks, mode):
    from afd_plugin.connectors.gpu.async_gpu import ConnectorShutdown
    from afd_plugin.connectors.metadata import AFDF2ATransferPayload

    session = connector.ffn_session()
    session.before_first_step()
    while True:
        try:
            work = connector.recv_ffn_work_item(stage_idx=0, max_num_tokens=NUM_TOKENS)
        except ConnectorShutdown:
            break
        except TimeoutError:
            continue
        if work is None:
            continue
        state = work.context.states
        experts_per_rank = NUM_EXPERTS // ffn_ranks
        expert_ids = torch.repeat_interleave(
            torch.arange(
                role_rank * experts_per_rank,
                (role_rank + 1) * experts_per_rank,
                device=work.hidden_states.device,
            ),
            state.group_list.long(),
            output_size=work.hidden_states.shape[0],
        )
        factors = (
            torch.where(expert_ids % 2 == 0, CANCELLATION_FACTOR, -CANCELLATION_FACTOR)
            if mode == "precision"
            else expert_ids + 1
        )
        if mode == "cross_rank_precision":
            factors = torch.where(
                expert_ids == 0, CANCELLATION_FACTOR, -CANCELLATION_FACTOR
            )
        routed = work.hidden_states * factors[:, None].to(work.hidden_states.dtype)
        shared = state.expand_x_shared * SHARED_SCALE
        connector.send_ffn_work_item_output(
            work,
            AFDF2ATransferPayload(routed_output=routed, shared_output=shared),
        )
        session.after_step()
    session.finish_receiver()


def _rank(
    rank,
    port,
    default_stores,
    mode,
    result_queue,
    attention_ranks,
    ffn_ranks,
    first_done,
):
    import torch.distributed as dist

    connector = None
    try:
        os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{default_stores[rank]}",
            rank=0,
            world_size=1,
        )
        attention = rank < attention_ranks
        role_rank = rank if attention else rank - attention_ranks
        connector = _connector(
            "attention" if attention else "ffn",
            rank,
            role_rank,
            port,
            attention_ranks,
            ffn_ranks,
        )
        connector.init_afd_connector()
        if attention:
            if mode == "independent" and role_rank == 1:
                assert first_done.wait(PROCESS_TIMEOUT_S), (
                    "A0 could not progress while A1 was idle"
                )
            attention_mode = (
                "eager" if mode == "independent" and role_rank == 0 else mode
            )
            _run_attention(connector, torch.device("cuda", rank), attention_mode)
            if role_rank == 0:
                first_done.set()
        else:
            _run_ffn(connector, role_rank, ffn_ranks, mode)
            assert first_done.is_set(), (
                "FFN closed before its Attention producer finished"
            )
        connector.close()
        connector = None
        result_queue.put((rank, None))
    except BaseException:
        result_queue.put((rank, traceback.format_exc()))
    finally:
        # Collective transport teardown belongs on the successful path. If a
        # peer failed, the parent terminates the remaining process after timeout.
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode,attention_ranks,ffn_ranks",
    [
        ("eager", 1, 1),
        ("graph", 1, 1),
        ("precision", 1, 1),
        ("cross_rank_precision", 1, 2),
        ("independent", 2, 2),
    ],
)
def test_async_gpu_two_stage_transport_and_shared_outputs(  # long-test: CUDA lifecycle
    mode, attention_ranks, ffn_ranks, tmp_path
):
    world_size = attention_ranks + ffn_ranks
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} allocated NVLink GPUs")
    port = _free_port()
    default_stores = tuple(
        str(tmp_path / f"rank-{rank}.store") for rank in range(world_size)
    )
    context = mp.get_context("spawn")
    results = context.Queue()
    first_done = context.Event()
    processes = [
        context.Process(
            target=_rank,
            args=(
                rank,
                port,
                default_stores,
                mode,
                results,
                attention_ranks,
                ffn_ranks,
                first_done,
            ),
        )
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        messages = []
        for _ in processes:
            rank, error = results.get(timeout=PROCESS_TIMEOUT_S)
            assert error is None, f"rank {rank}: {error}"
            messages.append((rank, error))
        for process in processes:
            process.join(PROCESS_TIMEOUT_S)
        assert all(process.exitcode == 0 for process in processes), messages
        assert len(messages) == len(processes), messages
        assert all(error is None for _, error in messages), messages
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
