# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synthetic-weight integration of the real Async runner and CAM connector.

Run on three isolated 910C devices with torchrun --nproc-per-node=3.
This covers native transport, layered compute, repeated replay, and finite
completion. Production DR cancellation is not part of this fixture. It does not
claim checkpoint-loader or model E2E coverage.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import socket
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

HIDDEN = 256
EXPERTS = 4
LAYERS = (2, 5)
LIMIT = 10.0


class SyntheticModel(torch.nn.Module):
    """Diagonal experts shared by eager/FULL runs; no checkpoint is loaded."""

    def __init__(
        self,
        device,
        per_channel,
        hidden=HIDDEN,
        intermediate=HIDDEN,
        weight_seed=None,
        routed_scale=1.0,
    ):
        super().__init__()
        self.layers = []
        for slot, layer_id in enumerate(LAYERS):
            seed = None if weight_seed is None else weight_seed + slot * 2
            w13, s13, b13 = self.make_weight(
                device, per_channel, slot + 1, True, hidden, intermediate, seed
            )
            w2, s2, b2 = self.make_weight(
                device,
                per_channel,
                1,
                False,
                hidden,
                intermediate,
                None if seed is None else seed + 1,
            )
            self.layers.append(
                W4A8LayerWeights(
                    layer_id,
                    w13,
                    w2,
                    s13,
                    s2,
                    b13,
                    b2,
                    per_channel,
                    LIMIT,
                    routed_scale,
                )
            )

    @staticmethod
    def make_weight(
        device,
        per_channel,
        factor,
        gate_up,
        hidden=HIDDEN,
        intermediate=HIDDEN,
        seed=None,
    ):
        k = hidden if gate_up else intermediate
        width = 2 * intermediate if gate_up else hidden
        if seed is None:
            values = torch.zeros(EXPERTS, k, width, dtype=torch.int8)
            diagonal = torch.arange(min(k, intermediate if gate_up else width))
            values[:, diagonal, diagonal] = 1
            if gate_up:
                values[:, diagonal, diagonal + intermediate] = 1
        else:
            generator = torch.Generator().manual_seed(seed)
            values = torch.randint(
                -3, 4, (EXPERTS, k, width), dtype=torch.int8, generator=generator
            )
        groups = 1 if per_channel else 2
        scale = torch.full((EXPERTS, groups, width), float(factor))
        if seed is not None:
            scale *= 0.01 + 0.02 * torch.rand(scale.shape, generator=generator)
        if not per_channel:
            scale[:, 1] *= 2
        dequant = values.float() * scale.repeat_interleave(k // groups, 1)
        bias = (8 * dequant.sum(1)).to(device)
        packed = values.reshape(-1, 2)
        packed = ((packed[:, 1] << 4) | (packed[:, 0] & 15)).reshape(
            EXPERTS, k, width // 2
        )
        weight = torch_npu.npu_format_cast(packed.to(device), 29).view(torch.int32)
        encoded = scale.view(torch.int32).to(torch.int64)
        if per_channel and gate_up:
            encoded = encoded.squeeze(1)
        return weight.contiguous(), encoded.to(device), bias

    def get_experts_layer_indices(self):
        return LAYERS

    def get_async_cam_w4a8_layers(self):
        return self.layers, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group", action="store_true")
    parser.add_argument("--mega", action="store_true")
    parser.add_argument("--idle-probe", action="store_true")
    parser.add_argument(
        "--hidden-size", type=int, choices=(256, 512, 1024, 2048), default=HIDDEN
    )
    parser.add_argument(
        "--intermediate-size", type=int, choices=(256, 512, 1024, 2048), default=HIDDEN
    )
    parser.add_argument("--weight-seed", type=int)
    parser.add_argument("--routed-scale", type=float, default=1.0)
    parser.add_argument("--full-graph", action="store_true")
    parser.add_argument("--attn-ranks", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--attention-tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--boundary", action="store_true")
    parser.add_argument("--receive-capacity", type=int, default=16)
    parser.add_argument("--boundary-tokens", type=int, default=16)
    parser.add_argument("--idle-shutdown", action="store_true")
    parser.add_argument("--save-outputs")
    parser.add_argument("--reference-outputs")
    args = parser.parse_args()
    if not 1 <= args.receive_capacity <= 512 or not 7 <= args.boundary_tokens <= 512:
        parser.error("capacity must be in [1,512], boundary tokens in [7,512]")
    if args.attn_ranks % args.attention_tp:
        parser.error("Attention rank count must be divisible by TP")
    if (args.idle_shutdown or args.idle_probe) and not args.mega:
        parser.error("idle shutdown/probe requires --mega")
    if args.mega and args.per_group:
        parser.error("native Mega requires per-channel weights")
    if args.full_graph and not args.reference_outputs:
        parser.error("FULL replay requires saved matching eager outputs")
    faulthandler.dump_traceback_later(60)
    os.environ["AFD_ASYNC_CAM_LAYERED_GMM"] = "1"
    os.environ["BATCH_SIZE_FACTOR"] = str(args.receive_capacity / 262144)
    os.environ["HCCL_BUFFSIZE"] = "64"
    ensure_cam_async_ops_available()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = args.attn_ranks + 2
    assert int(os.environ["WORLD_SIZE"]) == world_size
    is_ffn = rank >= args.attn_ranks
    torch.npu.set_device(local_rank)
    torch.npu.config.allow_internal_format = True
    device = torch.device("npu", local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    # torchrun's agent owns only the default rendezvous endpoint. The actual
    # connector creates a distinct Store, as in separate vLLM A/F workers.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    port = [None]
    if rank == 0:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port[0] = probe.getsockname()[1]
    dist.broadcast_object_list(port, src=0)
    config = SimpleNamespace(
        additional_config={
            "afd": {
                "connector": "CAMAsyncAFDConnector",
                "role": "ffn" if is_ffn else "attention",
                "async": True,
                "compute_gate_on_attention": True,
                "num_attention_ranks": args.attn_ranks,
                "num_ffn_ranks": 2,
                "port": port[0],
                "connector_extra_config": {
                    "dynamicQuant": 1,
                    "ffn_mega_moe": args.mega,
                    "attn_ranks_per_dp": args.attention_tp,
                    "hccl_buffer_size": 64,
                },
            }
        },
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=not (args.full_graph and is_ffn),
            hf_config=SimpleNamespace(
                hidden_size=args.hidden_size,
                num_experts_per_tok=2,
                n_routed_experts=8,
                num_hidden_layers=max(LAYERS) + 1,
                architectures=["DeepseekV4ForCausalLM"],
                model_type="deepseek_v4",
            ),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=args.boundary_tokens * args.attention_tp
        ),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            use_ubatching=False,
            num_ubatches=2,
            data_parallel_size=2 if is_ffn else args.attn_ranks // args.attention_tp,
            tensor_parallel_size=1 if is_ffn else args.attention_tp,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(num_gpu_blocks=None),
        compilation_config=SimpleNamespace(
            static_forward_context={}, cudagraph_mode=SimpleNamespace(name="FULL")
        ),
        use_v2_model_runner=True,
    )
    connector = CAMAsyncAFDConnector(
        rank,
        local_rank,
        config,
        parse_afd_config(config),
        rank - args.attn_ranks if is_ffn else rank,
    )
    report = {
        "rank": rank,
        "per_group": args.per_group,
        "mega": args.mega,
        "hidden": args.hidden_size,
        "intermediate": args.intermediate_size,
        "weight_seed": args.weight_seed,
        "routed_scale": args.routed_scale,
        "attention_tp": args.attention_tp,
        "work": [],
    }
    print(json.dumps({"rank": rank, "stage": "connector_created"}), flush=True)
    if is_ffn:
        runner = AsyncNPUFFNModelRunner(config, device, connector)
        runner.model = SyntheticModel(
            device,
            not args.per_group,
            args.hidden_size,
            args.intermediate_size,
            None if args.weight_seed is None else args.weight_seed + rank * 10000,
            args.routed_scale,
        )
        print(json.dumps({"rank": rank, "stage": "weights_ready"}), flush=True)
        runner.initialize_afd_connector()
        assert runner._layered_executor is not None

        def work_state():
            if args.mega:
                mega = runner._mega_executor
                return SimpleNamespace(
                    token_nums_rankid_layeridx=mega.batch_info,
                    group_list=mega.group_list,
                )
            return runner._graph_payload.context.states

        if args.full_graph:
            assert runner.effective_mode == "full"
            assert runner._graph_capture_count == 1

            def capture_addresses():
                session = runner
                if args.mega:
                    mega = session._mega_executor
                    return tuple(
                        t.data_ptr()
                        for t in (
                            mega.scratch,
                            mega.layer_map,
                            mega.anchor,
                            session._graph_output,
                        )
                    )
                state = work_state()
                return tuple(
                    tensor.data_ptr()
                    for tensor in (
                        session._graph_payload.hidden_states,
                        state.dynamic_scales,
                        state.group_list,
                        state.token_nums_rankid_layeridx,
                        state.send_completion,
                        session._graph_output,
                        *session._graph_tensors,
                    )
                )

            captured_addresses = capture_addresses()
            gc.collect()
            torch.npu.empty_cache()

            def forbid_python_transaction(*args, **kwargs):
                raise AssertionError(
                    "FULL service must replay the captured transaction"
                )

            connector.recv_attn_output = forbid_python_transaction
            connector.send_ffn_output = forbid_python_transaction
            # Test-only per-replay observations, fenced once at the batch end.
            # Production batching performs no host metadata copy.
            session = runner
            state = work_state()
            headers = torch.empty(
                (len(LAYERS), state.token_nums_rankid_layeridx.numel()),
                dtype=torch.int64,
                pin_memory=True,
            )
            counts = torch.empty(
                (len(LAYERS), state.group_list.numel()),
                dtype=state.group_list.dtype,
                pin_memory=True,
            )
            graph = session._graph
            batch_index = 0

            def observed_replay():
                nonlocal batch_index
                graph.replay()
                headers[batch_index].copy_(
                    state.token_nums_rankid_layeridx, non_blocking=True
                )
                counts[batch_index].copy_(state.group_list, non_blocking=True)
                batch_index += 1

            session._graph = SimpleNamespace(replay=observed_replay, reset=graph.reset)
    else:
        connector.init_afd_connector()
    if not args.full_graph:
        # Cold MC2 needs all A/F peers to participate. This finite eager fixture
        # warms the original operators concurrently before ordered DP cases.
        with torch.inference_mode():
            if is_ffn:
                for _ in range(args.attn_ranks // args.attention_tp):
                    runner._ffn_forward_connector_driven(single=True)
            else:
                hidden = torch.ones(
                    (1, args.hidden_size), dtype=torch.bfloat16, device=device
                )
                context = AFDTransferContext(AFDTransferMetadata(LAYERS[0], 0, [1]))
                connector.send_attn_output(
                    hidden,
                    context,
                    topk_ids=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
                    topk_weights=torch.full((1, 2), 0.5, device=device),
                )
                connector.recv_ffn_output(hidden, context=context)
            torch.npu.synchronize()
        dist.barrier()
    print(json.dumps({"rank": rank, "stage": "transport_ready"}), flush=True)
    if args.idle_probe:
        assert args.mega
        if is_ffn:
            pull_state = connector.comm_args.view(torch.int64)
            before = int(pull_state[96].item())
            pull_state[64].fill_(8)
            if args.full_graph:
                runner._graph_stream.wait_stream(torch.npu.current_stream(device))
                connector.graph_startup.service_ready()
            with torch.inference_mode():
                for _ in range(3):
                    if args.full_graph:
                        batch_index = 0
                        runner._submit_graph(1)
                        runner._complete_graph()
                    else:
                        runner._ffn_forward_connector_driven(single=True)
                torch.npu.synchronize()
            assert int(pull_state[96].item()) == before
            assert int(pull_state[32].item()) == -1
            connector.check_pull_status()
            pull_state[64].zero_()
            torch.npu.synchronize()
            report["idle_probe"] = "PASS"
        dist.barrier()
    errors = []
    accuracy_failures: list[str] = []
    data_replays = 0
    if not is_ffn:
        runner = thread = None

    cases = [(2, False, 7), (5, True, 7), (2, False, 7)]
    if args.boundary:
        cases.extend(
            ((5, False, args.boundary_tokens), (2, True, args.boundary_tokens))
        )
    expected: list[tuple[int, int, int | None]] = []
    for layer_id, sparse, tokens in cases:
        ids = torch.arange(tokens * 2).remainder(2 if sparse else 8)
        local_ffn = rank - args.attn_ranks
        rows = int(
            ((ids >= local_ffn * EXPERTS) & (ids < (local_ffn + 1) * EXPERTS)).sum()
        )
        rows *= args.attention_tp
        for source in range(0, args.attn_ranks, args.attention_tp):
            if args.mega:
                # The exposed metadata is the final slice of one fused transaction.
                last_rows = (rows - 1) % args.receive_capacity + 1 if rows else 0
                expected.append((source, layer_id, last_rows))
                continue
            for start in range(0, max(1, rows), args.receive_capacity):
                expected.append(
                    (source, layer_id, min(args.receive_capacity, rows - start))
                )

    def observe(context):
        # Test-only host inspection after the matching DR/FFN/CS fence.
        state = context.states
        header = state.token_nums_rankid_layeridx.cpu().tolist()
        report["work"].append(
            {
                "source": header[1],
                "layer": header[2],
                "rows": int(state.group_list.sum().item()),
            }
        )

    if is_ffn and not args.full_graph and args.mega:
        original_mega = runner._mega_executor.op

        def observe_mega(*op_args):
            result = original_mega(*op_args)
            torch.npu.current_stream().synchronize()
            observe(SimpleNamespace(states=work_state()))
            return result

        runner._mega_executor.op = observe_mega
    elif is_ffn and not args.full_graph:
        original_send = connector.send_ffn_output

        def observe_eager(output, context, **kwargs):
            result = original_send(output, context, **kwargs)
            torch.npu.current_stream().synchronize()
            observe(context)
            return result

        connector.send_ffn_output = observe_eager

    def finish_receiver():
        if args.idle_shutdown:
            if args.full_graph:
                runner._graph = graph
            elif args.mega:
                runner._mega_executor.op = original_mega
            else:
                connector.send_ffn_output = original_send
            event = threading.Event()
            # A local shutdown must still serve until all A ranks stop publishing.
            event.set()
            worker = SimpleNamespace(
                _ffn_shutdown_event=event, device=device, model_runner=runner
            )
            AFDNPUFFNWorker._run_ffn_server_loop(worker)
            report["idle_shutdown"] = "PASS"
        runner._graph_receiver_exited = True

    def serve():
        nonlocal batch_index, data_replays
        try:
            torch.npu.set_device(local_rank)
            if not args.full_graph:
                # A finite fixture drives known old eager transactions. Eager
                # service termination is outside this graph change's contract.
                with torch.inference_mode():
                    for _ in expected:
                        runner._ffn_forward_connector_driven(single=True)
                finish_receiver()
                return
            connector.graph_startup.service_ready()
            replay_before = runner._graph_replay_count
            for start in range(0, len(expected), len(LAYERS)):
                batch_index = 0
                # Test-only finite work count, including the final short batch.
                # The data check uses a fixed work count before the idle-stop probe.
                count = min(len(LAYERS), len(expected) - start)
                runner._submit_graph(count)
                runner._complete_graph()
                session = runner
                assert capture_addresses() == captured_addresses
                for header, group in zip(
                    headers[:count].tolist(), counts[:count], strict=True
                ):
                    if header[2] >= 0:
                        report["work"].append(
                            {
                                "source": header[1],
                                "layer": header[2],
                                "rows": int(group.sum()),
                            }
                        )
                if args.boundary and start + count < len(expected):
                    # Poison between completed replays on the graph stream.
                    # DR must overwrite valid rows and ignore unused capacity.
                    with (
                        torch.inference_mode(),
                        torch.npu.stream(session._graph_stream),
                    ):
                        if args.mega:
                            session._mega_executor.scratch.fill_(113)
                        else:
                            session._graph_payload.hidden_states.fill_(113)
                            work_state().dynamic_scales.fill_(3)
                        session._graph_output.fill_(123)
            data_replays = runner._graph_replay_count - replay_before
            finish_receiver()
        except Exception as exc:
            errors.append(exc)
            print(json.dumps({"rank": rank, "service_error": repr(exc)}), flush=True)
            # Report fixture failure to peers; never substitute eager execution.
            connector.graph_startup.fail(exc)

    if is_ffn:
        thread = threading.Thread(target=serve, name="afd-runtime-test", daemon=True)
        thread.start()
    dist.barrier()
    outputs: list[torch.Tensor] = []
    references = None
    if not is_ffn and args.reference_outputs:
        references = torch.load(
            f"{args.reference_outputs}.rank{rank}.pt", weights_only=True
        )
    for layer_id, sparse, tokens in cases:
        for source in range(0, args.attn_ranks, args.attention_tp):
            if source <= rank < source + args.attention_tp:
                values = torch.tensor([-12, -9, -0.5, 0.5, 9, 12, 1])
                values = values.repeat((tokens + values.numel() - 1) // values.numel())[
                    :tokens
                ]
                inputs = (
                    (values * (rank + 1))
                    .bfloat16()[:, None]
                    .expand(-1, args.hidden_size)
                    .contiguous()
                )
                if args.weight_seed is not None:
                    feature = torch.sin(torch.arange(args.hidden_size).float() * 0.17)
                    inputs = (inputs.float() * feature[None, :] + 0.125).bfloat16()
                ids = (
                    torch.arange(tokens * 2)
                    .reshape(tokens, 2)
                    .remainder(2 if sparse else 8)
                )
                hidden = inputs.to(device)
                context = AFDTransferContext(AFDTransferMetadata(layer_id, 0, [tokens]))
                connector.send_attn_output(
                    hidden,
                    context,
                    topk_ids=ids.to(device=device, dtype=torch.int32),
                    topk_weights=torch.full((tokens, 2), 0.5, device=device),
                )
                output = connector.recv_ffn_output(hidden, context=context)
                done = torch.npu.Event()
                done.record()
                deadline = time.monotonic() + 30
                while not done.query():
                    connector.graph_startup.check_failure()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "runtime test CR did not finish in 30 seconds"
                        )
                    time.sleep(0.01)
                actual = output.cpu()
                if references is not None:
                    try:
                        torch.testing.assert_close(
                            actual, references[len(outputs)], rtol=0, atol=0
                        )
                    except AssertionError as exc:
                        if args.save_outputs and not accuracy_failures:
                            torch.save(
                                {
                                    "actual": actual,
                                    "reference": references[len(outputs)],
                                    "layer": layer_id,
                                    "sparse": sparse,
                                    "tokens": tokens,
                                },
                                f"{args.save_outputs}.mismatch.rank{rank}.pt",
                            )
                        # Finish all peer transactions before reporting numerical
                        # failure, so teardown cannot invalidate a pending pull.
                        accuracy_failures.append(str(exc))
                outputs.append(actual)
                report["work"].append(
                    {"layer": layer_id, "sparse": sparse, "tokens": tokens}
                )
            # Harness-only scheduling makes source changes reproducible. The
            # production connector has no per-work cross-DP barrier.
            dist.barrier()
    if not is_ffn:
        if args.save_outputs:
            torch.save(outputs, f"{args.save_outputs}.rank{rank}.pt")
        if references is not None:
            assert len(references) == len(outputs)
            report["eager_exact_match"] = not accuracy_failures
            report["accuracy_failures"] = len(accuracy_failures)
        assert not connector._pending_attention_payloads
        if args.idle_shutdown:
            connector.close()
    else:
        assert thread is not None
        thread.join(timeout=30)
        assert not thread.is_alive(), "finite work did not complete within 30 seconds"
        if errors:
            raise errors[0]
        assert len(report["work"]) == len(expected)
        for observed, (source, layer, expected_rows) in zip(
            report["work"], expected, strict=True
        ):
            assert (observed["source"], observed["layer"]) == (source, layer)
            if expected_rows is not None:
                assert observed["rows"] == expected_rows, "zero work or chunk mismatch"
        if args.full_graph:
            report["graph"] = {
                "capture_count": runner._graph_capture_count,
                "replay_count": runner._graph_replay_count,
                "data_replay_count": data_replays,
                "python_transaction_disabled": True,
                "stable_exposed_addresses": len(captured_addresses),
            }
            assert data_replays == len(expected)
    # Retain the TCPStore owner until finite eager peers have consumed all work.
    if not args.idle_shutdown:
        dist.barrier()
    if is_ffn:
        runner.shutdown()
        runner.shutdown()
        assert runner.model is None and runner._graph is None
    else:
        connector.close()
    assert not connector.is_initialized and connector.cam_pg is None
    report["cleanup_complete"] = True
    reports: list[dict] = [{} for _ in range(world_size)]
    dist.all_gather_object(reports, report)
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    failed = any(item.get("accuracy_failures", 0) for item in reports)
    if rank == 0:
        print(
            json.dumps({"result": "FAIL" if failed else "PASS", "reports": reports}),
            flush=True,
        )
    if failed:
        raise AssertionError(
            accuracy_failures[0]
            if accuracy_failures
            else "peer accuracy comparison failed"
        )


if __name__ == "__main__":
    main()
