# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Finite real-device protocol/MLP probe; launch with torchrun on A3/910C.

No vLLM runtime, fake device operations or ordinary expert fallback is used.
CPU FP64 reference projections preserve BF16 boundaries and FP32 activation /
combination independently of the native implementation. This is functional
instrumentation, not a throughput run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as functional
import torch_npu  # noqa: F401

from afd_plugin.ffn_faas.client import AttentionSession
from afd_plugin.ffn_faas.model import CheckpointDescription, FFNModel
from afd_plugin.ffn_faas.npu import (
    NativeCompletion,
    NativeDeviceError,
    NativeWeightView,
    NPUExecutor,
)
from afd_plugin.ffn_faas.protocol import (
    MAX_GENERATION,
    MAX_WORK_PER_SERVE,
    Mode,
    Result,
    SessionSpec,
)

# Existing BF16 arithmetic standard, fixed before native execution:
# tests/e2e/operators/async_cam_roundtrip.py::TOLERANCES["bfloat16"].
ATOL = 0.06
RTOL = 0.02
MAX_NORMALIZED_RMSE = 0.02
WEIGHT_SEED = 12345
INPUT_SEED = 7810
TIMEOUT_SECONDS = 30.0
DELAY_SECONDS = 0.03


@dataclass
class Work:
    sequence: int
    layer: int
    x: torch.Tensor
    ids: torch.Tensor
    probabilities: torch.Tensor
    device_x: torch.Tensor
    device_ids: torch.Tensor
    device_probabilities: torch.Tensor


@dataclass
class OracleOutput:
    experts: torch.Tensor
    combined: torch.Tensor


def make_weights(spec: SessionSpec) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(WEIGHT_SEED)
    w13 = (
        torch.randn(
            len(spec.layer_indices),
            spec.num_experts,
            2 * spec.intermediate_size,
            spec.hidden_size,
            generator=generator,
        )
        * 0.25
    )
    w2 = (
        torch.randn(
            len(spec.layer_indices),
            spec.num_experts,
            spec.hidden_size,
            spec.intermediate_size,
            generator=generator,
        )
        * 0.25
    )
    for layer in range(len(spec.layer_indices)):
        w13[layer] *= layer + 1
    return w13.to(torch.bfloat16), w2.to(torch.bfloat16)


def make_work(
    spec: SessionSpec,
    rank: int,
    sequence: int,
    probability_dtype: torch.dtype = torch.float32,
) -> Work:
    source = rank // spec.attention_tp
    generator = torch.Generator().manual_seed(INPUT_SEED + source * 1000 + sequence)
    count = (sequence * 3 + rank) % (spec.capacity + 1)
    if sequence % 3 == 0 and rank % spec.attention_tp == 1:
        count = 0
    x = torch.randn((count, spec.hidden_size), generator=generator).to(torch.bfloat16)
    ids = torch.randint(
        spec.num_experts, (count, spec.top_k), generator=generator, dtype=torch.int32
    )
    if sequence % 4 == 0:
        ids.fill_(0)  # all other experts, including an entire F, receive no routes
    elif sequence % 4 == 1:
        ids.fill_(spec.num_experts - 1)
    probabilities = torch.softmax(
        torch.randn((count, spec.top_k), generator=generator), dim=-1
    )
    if sequence % 5 == 0:
        probabilities.zero_()
        probabilities[:, (sequence // 5) % spec.top_k] = 1
    if sequence % 4 == 3:
        ids[:, -1] = -1
        probabilities[:, -1] = 0
    probabilities = probabilities.to(probability_dtype)
    layer = spec.layer_indices[(sequence + source) % len(spec.layer_indices)]
    return Work(
        sequence, layer, x, ids, probabilities, x.npu(), ids.npu(), probabilities.npu()
    )


def reference(
    work: Work, spec: SessionSpec, w13: torch.Tensor, w2: torch.Tensor
) -> OracleOutput:
    compact = spec.layer_indices.index(work.layer)
    result = torch.zeros_like(work.x, dtype=torch.float32)
    experts = torch.zeros(
        (len(work.x), spec.top_k, spec.hidden_size), dtype=torch.bfloat16
    )
    for token in range(work.x.shape[0]):
        for route in range(spec.top_k):
            expert = int(work.ids[token, route])
            if expert < 0:
                continue
            # FP32 accumulation error can cross a BF16 midpoint and amplify
            # through SwiGLU. Resolve the reference projection before rounding.
            projected = functional.linear(
                work.x[token].double(), w13[compact, expert].double()
            )
            projected = projected.to(torch.bfloat16).float()
            gate, up = projected.chunk(2)
            activated = (functional.silu(gate) * up).to(torch.bfloat16)
            expert_output = functional.linear(
                activated.double(), w2[compact, expert].double()
            ).to(torch.bfloat16)
            experts[token, route] = expert_output
            result[token] += expert_output.float() * work.probabilities[token, route]
    return OracleOutput(experts, result.to(torch.bfloat16))


class CheckpointOracle:
    """Independent CPU expert reference; reads only selected routed tensors."""

    def __init__(self, checkpoint: CheckpointDescription) -> None:
        from safetensors import safe_open

        self.checkpoint = checkpoint
        self.stack = ExitStack()
        self.readers = {
            name: self.stack.enter_context(
                safe_open(str(checkpoint.root / name), framework="pt", device="cpu")
            )
            for name in sorted(set(checkpoint.weight_map.values()))
        }

    def reference(self, work: Work) -> OracleOutput:
        result = torch.zeros_like(work.x, dtype=torch.float32)
        experts = torch.zeros((*work.ids.shape, work.x.shape[1]), dtype=torch.bfloat16)
        for token in range(len(work.x)):
            for route in range(work.ids.shape[1]):
                expert = int(work.ids[token, route])
                if expert < 0:
                    continue
                prefix = f"model.layers.{work.layer}.mlp.experts.{expert}"
                tensors = []
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{prefix}.{projection}.weight"
                    tensors.append(
                        self.readers[self.checkpoint.weight_map[name]].get_tensor(name)
                    )
                gate = (
                    functional.linear(work.x[token].double(), tensors[0].double())
                    .to(torch.bfloat16)
                    .float()
                )
                up = (
                    functional.linear(work.x[token].double(), tensors[1].double())
                    .to(torch.bfloat16)
                    .float()
                )
                activated = (functional.silu(gate) * up).to(torch.bfloat16).float()
                output = functional.linear(activated.double(), tensors[2].double()).to(
                    torch.bfloat16
                )
                experts[token, route] = output
                result[token] += output.float() * work.probabilities[token, route]
        return OracleOutput(experts, result.to(torch.bfloat16))

    def close(self) -> None:
        self.stack.close()


def publish(executor: NPUExecutor, work: Work) -> Result:
    # Inference may set a default device; metadata must still be pinned CPU.
    with torch.device(executor.device):
        return executor.retire(
            executor.submit(
                Mode.PUBLISH,
                sequence=work.sequence,
                layer=work.layer,
                x=work.device_x,
                expert_ids=work.device_ids,
                probabilities=work.device_probabilities,
            )
        )


def combine(
    executor: NPUExecutor, work: Work, mode: Mode = Mode.COMBINE
) -> torch.Tensor:
    deadline = time.monotonic() + executor.timeout
    while time.monotonic() < deadline:
        executor.check_peers()
        completion = executor.submit(
            mode,
            sequence=work.sequence,
            layer=work.layer,
            x=work.device_x,
            expert_ids=work.device_ids,
            probabilities=work.device_probabilities,
        )
        code = executor.retire(completion)
        if code == Result.OK:
            return completion.output[: work.x.shape[0]].cpu()
        if code != Result.TIMED_OUT:
            raise AssertionError(f"unexpected combine result {code}")
    raise TimeoutError(
        f"combine sequence {work.sequence} exceeded its session deadline"
    )


def attention(
    executor: NPUExecutor,
    rounds: int,
    cpu_reference: Callable[[Work], OracleOutput],
    report: dict,
    probability_dtype: torch.dtype = torch.float32,
) -> None:
    spec = executor.spec
    pending: deque[Work] = deque()
    report["backpressure_checks"] = 0
    report["comparisons"] = []
    report["rejected_invalid_admissions"] = 0

    def rejected(mode: Mode, work: Work) -> None:
        completion = executor.submit(
            mode,
            sequence=work.sequence,
            layer=work.layer,
            x=work.device_x,
            expert_ids=work.device_ids,
            probabilities=work.device_probabilities,
        )
        try:
            executor.retire(completion)
        except RuntimeError as error:
            assert "INVALID" in str(error), str(error)
        else:
            raise AssertionError("native accepted an invalid admission")
        report["rejected_invalid_admissions"] += 1

    def consume() -> None:
        work = pending.popleft()
        if executor.rank // spec.attention_tp == 1:
            time.sleep(DELAY_SECONDS)
        expert_actual = combine(executor, work, Mode.PEEK_RESULTS)
        oracle = cpu_reference(work)
        expected = oracle.combined
        try:
            torch.testing.assert_close(
                expert_actual, oracle.experts, atol=ATOL, rtol=RTOL
            )
        except AssertionError:
            evidence = Path(report["evidence_dir"])
            evidence.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "x": work.x,
                    "ids": work.ids,
                    "probabilities": work.probabilities,
                    "layer": work.layer,
                    "expert_actual": expert_actual,
                    "expert_reference": oracle.experts,
                    "weight_seed": WEIGHT_SEED,
                },
                evidence / f"sequence-{work.sequence}-failed-expert.pt",
            )
            raise
        # Complete the independent expert check while the slot is still held;
        # only the following COMBINE may return credit and advance consumed.
        expert_difference = (expert_actual.float() - oracle.experts.float()).abs()
        expert_nrmse = float(
            torch.linalg.vector_norm(expert_difference)
            / torch.linalg.vector_norm(oracle.experts.float()).clamp_min(1e-12)
        )
        assert expert_nrmse <= MAX_NORMALIZED_RMSE
        actual = combine(executor, work)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        difference = (actual.float() - expected.float()).abs()
        normalized_rmse = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        )
        assert normalized_rmse <= MAX_NORMALIZED_RMSE
        evidence = Path(report["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "x": work.x,
                "ids": work.ids,
                "probabilities": work.probabilities,
                "layer": work.layer,
                "actual": actual,
                "reference": expected,
                "expert_actual": expert_actual,
                "expert_reference": oracle.experts,
                "weight_seed": WEIGHT_SEED,
            },
            evidence / f"sequence-{work.sequence}.pt",
        )
        report["comparisons"].append(
            {
                "sequence": work.sequence,
                "layer": work.layer,
                "tokens": len(actual),
                "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
                "normalized_rmse": normalized_rmse,
                "expert_normalized_rmse": expert_nrmse,
            }
        )

    # Invalid IDs at the end must be rejected before any ready publication,
    # including a short DMA tail after the first 16 int32 route entries.
    invalid = make_work(spec, executor.rank, 1)
    x = torch.zeros((spec.capacity, spec.hidden_size), dtype=torch.bfloat16)
    ids = torch.zeros((spec.capacity, spec.top_k), dtype=torch.int32)
    ids[-1, -1] = spec.num_experts
    probabilities = torch.full(ids.shape, 1 / spec.top_k, dtype=torch.float32)
    rejected(
        Mode.PUBLISH,
        replace(
            invalid,
            x=x,
            ids=ids,
            probabilities=probabilities,
            device_x=x.npu(),
            device_ids=ids.npu(),
            device_probabilities=probabilities.npu(),
        ),
    )
    report["rejected_invalid_id_route"] = ids.numel() - 1

    for sequence in range(1, rounds + 1):
        work = make_work(spec, executor.rank, sequence, probability_dtype)
        if sequence == 2:
            # A skipped generation must not overwrite generation one's slot.
            rejected(Mode.PUBLISH, replace(work, sequence=3))
        if len(pending) == spec.slots:
            assert publish(executor, work) == Result.BACKPRESSURE
            blocked = executor.submit(
                Mode.EXCHANGE,
                sequence=work.sequence,
                layer=work.layer,
                x=work.device_x,
                expert_ids=work.device_ids,
                probabilities=work.device_probabilities,
            )
            try:
                executor.submit(Mode.PUBLISH)
            except RuntimeError as error:
                assert "active submission" in str(error)
            else:
                raise AssertionError("executor admitted a second in-flight command")
            assert executor.retire(blocked) == Result.BACKPRESSURE
            report["backpressure_checks"] += 1
            consume()
        if sequence == 1:
            # F waits for two prepublications. EXCHANGE must time out only in
            # combine, retaining its published generation for later consumption.
            exchange = executor.submit(
                Mode.EXCHANGE,
                sequence=work.sequence,
                layer=work.layer,
                x=work.device_x,
                expert_ids=work.device_ids,
                probabilities=work.device_probabilities,
            )
            assert executor.retire(exchange) == Result.TIMED_OUT
            report["exchange_timeout_after_publication"] = True
        else:
            assert publish(executor, work) == Result.OK
        pending.append(work)
        if sequence in (1, 3):
            other = (
                spec.layer_indices[0]
                if work.layer != spec.layer_indices[0]
                else spec.layer_indices[-1]
            )
            poisoned = replace(
                work, layer=other, device_x=torch.full_like(work.device_x, 171)
            )
            rejected(Mode.PUBLISH, poisoned)
        if sequence == 1:
            rejected(Mode.END_SOURCE, work)
        if sequence == 2:
            rejected(Mode.COMBINE, work)
        if sequence == spec.slots:
            executor.store.set(f"prepublished/{executor.rank}", "1")
    while pending:
        consume()
    assert (
        executor.retire(executor.submit(Mode.END_SOURCE, sequence=rounds)) == Result.OK
    )
    rejected_after_eos(executor, rounds, report)


def scoped_attention(
    executor: NPUExecutor,
    rounds: int,
    cpu_reference: Callable[[Work], OracleOutput],
    report: dict,
    probability_dtype: torch.dtype = torch.float32,
) -> None:
    session = AttentionSession(executor, tp_rows_aligned=True)
    # An empty model step is a valid scope and publishes no work.
    with session.model_execution_scope():
        pass
    work_items = []
    with session.model_execution_scope():
        for sequence in range(1, rounds + 1):
            work = make_work(executor.spec, executor.rank, sequence, probability_dtype)
            session.send(
                work.device_x,
                work.device_ids,
                work.device_probabilities,
                layer=work.layer,
            )
            output = session.receive()
            # Queue a real consumer on Attention's stream. Completion/request
            # recycling must not recycle its producer storage prematurely.
            actual = output.clone()
            work_items.append((work, actual))
    report["comparisons"] = []
    evidence = Path(report["evidence_dir"])
    evidence.mkdir(parents=True, exist_ok=True)
    for work, actual in work_items:
        expected = cpu_reference(work).combined
        actual = actual.cpu()
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        difference = actual.float() - expected.float()
        nrmse = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        )
        assert nrmse <= MAX_NORMALIZED_RMSE
        torch.save(
            {
                "actual": actual,
                "reference": expected,
                "layer": work.layer,
                "x": work.x,
                "ids": work.ids,
                "probabilities": work.probabilities,
            },
            evidence / f"sequence-{work.sequence}.pt",
        )
        report["comparisons"].append(
            {"sequence": work.sequence, "tokens": len(actual), "nrmse": nrmse}
        )
    # Scope tail has retired before ordinary source-end is admitted.
    assert (
        executor.retire(executor.submit(Mode.END_SOURCE, sequence=rounds)) == Result.OK
    )
    report["model_scope"] = {"empty": True, "calls": rounds, "consumer_clone": True}


def rejected_after_eos(executor: NPUExecutor, last: int, report: dict) -> None:
    work = make_work(
        executor.spec, executor.rank, 1 if last == MAX_GENERATION else last + 1
    )
    try:
        publish(executor, work)
    except RuntimeError as error:
        assert "already drained" in str(error)
        report["executor_rejected_after_eos"] = True
    else:
        raise AssertionError("public executor accepted work after EOS")
    # Bypass the public guard to test the actual native terminal state. F may
    # already have returned DRAINED, but cannot free windows until A quiesces.
    view = executor.weights
    assert view is not None
    with torch.npu.stream(executor.stream):
        request = torch.tensor(
            [work.sequence, work.layer, len(work.x)],
            dtype=torch.int64,
            device=executor.device,
        )
        output, status = torch.ops.afd_ascend.ffn_faas(
            work.device_x
            if len(work.x)
            else torch.zeros(
                (1, executor.spec.hidden_size),
                dtype=torch.bfloat16,
                device=executor.device,
            ),
            work.device_ids
            if len(work.x)
            else torch.zeros(
                (1, executor.spec.top_k), dtype=torch.int32, device=executor.device
            ),
            work.device_probabilities
            if len(work.x)
            else torch.zeros(
                (1, executor.spec.top_k), dtype=torch.float32, device=executor.device
            ),
            view.w13,
            view.w2,
            view.layer_map,
            request,
            executor.state,
            executor.scratch,
            int(Mode.PUBLISH),
            executor.spec.native_config(executor.rank, 16),
            executor.group_name,
        )
    completion = NativeCompletion(
        Mode.PUBLISH,
        output,
        status,
        request,
        (request, work.device_x, work.device_ids, work.device_probabilities),
        executor.stream,
    )
    assert completion.wait(executor.timeout) == Result.INVALID
    report["native_rejected_after_eos"] = True
    report["terminal_boundary"] = last


def ffn(
    executor: NPUExecutor, rounds: int, report: dict, *, serve: bool = False
) -> None:
    report["executed"] = []
    report["idle_launches"] = 0
    report["serve_batches"] = []
    completed = 0
    deadline = time.monotonic() + max(executor.timeout, rounds * 2)
    while time.monotonic() < deadline:
        executor.check_peers()
        completion = executor.submit(Mode.SERVE if serve else Mode.EXECUTE)
        code = executor.retire(completion)
        if serve:
            count = completion.completed_work
            assert count is not None
            assert 0 <= count <= MAX_WORK_PER_SERVE
            completed += count
            report["serve_batches"].append({"code": code.name, "completed": count})
        if code == Result.IDLE:
            assert not serve or count == 0
            report["idle_launches"] += 1
            continue
        if code == Result.DRAINED:
            actual = completed if serve else len(report["executed"])
            assert actual == rounds * executor.spec.source_count
            report["completed_work"] = actual
            first_source = (
                executor.rank - executor.spec.attention_ranks
            ) % executor.spec.source_count
            if rounds and not serve:
                assert report["executed"][0]["source"] == first_source
            return
        assert code == Result.OK
        if serve:
            assert count > 0
            continue
        fields = completion.status.cpu().tolist()
        report["executed"].append(
            {
                "source": fields[1],
                "sequence": fields[2],
                "layer": fields[3],
                "tokens": fields[4],
            }
        )
    raise TimeoutError("FFN did not drain all source boundaries")


def check_served_receipts(
    executor: NPUExecutor, completed: NativeCompletion, report: dict
) -> None:
    """Fault-inject completed headers without altering live protocol state."""
    rejected = []
    for code, count in (
        (Result.OK, 0),
        (Result.IDLE, 1),
        (Result.OK, MAX_WORK_PER_SERVE + 1),
        (Result.TIMED_OUT, 0),
        (12345, 0),
    ):
        with torch.npu.stream(executor.stream):
            status = torch.tensor(
                [code, 0, 0, 0, 0, count, 0, 0],
                dtype=torch.int64,
                device=executor.device,
            )
        receipt = replace(completed, status=status, completed_work=None)
        try:
            receipt.wait(executor.timeout)
        except NativeDeviceError:
            assert receipt.completed_work is None
            rejected.append([int(code), count])
        else:
            raise AssertionError(f"invalid receipt accepted: {code}, {count}")
    report["rejected_completed_headers"] = rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-ranks", type=int, default=2)
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--attention-tp", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29619)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=32)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--probability-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--empty-eos", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model-scope", action="store_true")
    parser.add_argument(
        "--max-eos",
        action="store_true",
        help="seed private counters to test the final ABI boundary",
    )
    args = parser.parse_args()
    if args.empty_eos and args.max_eos:
        parser.error("select only one EOS boundary probe")
    if args.rounds < 2:
        parser.error("at least two rounds are needed to exercise every slot")
    rank = int(os.environ["RANK"])
    device = int(os.environ["LOCAL_RANK"])
    spec = SessionSpec(
        "synthetic-bf16-seed12345",
        args.hidden,
        args.intermediate,
        args.experts,
        args.top_k,
        args.attention_ranks,
        args.ffn_ranks,
        args.attention_tp,
        5,
        (1, 4),
        capacity=args.capacity,
    )
    checkpoint = None
    if args.model is not None:
        checkpoint = CheckpointDescription.read(args.model)
        spec = checkpoint.session_spec(
            attention_ranks=args.attention_ranks,
            ffn_ranks=args.ffn_ranks,
            attention_tp=args.attention_tp,
            capacity=args.capacity,
            layer_indices=checkpoint.layer_indices[:2],
        )
    assert int(os.environ["WORLD_SIZE"]) == spec.world_size
    torch.npu.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "spec": json.loads(spec.to_json()),
        "atol": ATOL,
        "rtol": RTOL,
        "max_normalized_rmse": MAX_NORMALIZED_RMSE,
        "result": "running",
        "evidence_dir": str(args.output_dir / f"rank-{rank}-tensors"),
        "torch_version": str(torch.__version__),
        "probability_dtype": args.probability_dtype,
        "reference_projection_dtype": "float64",
    }
    executor = None
    owned_model = None
    oracle = None
    started = time.monotonic()
    try:
        weights = None if checkpoint is not None else make_weights(spec)
        view = None
        if rank >= spec.attention_ranks:
            if checkpoint is not None:
                owned_model = FFNModel(checkpoint, spec, rank - spec.attention_ranks)
                load_started = time.monotonic()
                owned_model.load(device)
                report["model_load_seconds"] = time.monotonic() - load_started
                report["parameter_bytes"] = owned_model.parameter_bytes
                view = owned_model.native_view
            else:
                assert weights is not None
                first = (rank - spec.attention_ranks) * spec.local_experts
                last = first + spec.local_experts
                layer_map = torch.full((spec.model_layers,), -1, dtype=torch.int32)
                for compact, layer in enumerate(spec.layer_indices):
                    layer_map[layer] = compact
                view = NativeWeightView(
                    weights[0][:, first:last].contiguous().npu(),
                    weights[1][:, first:last].contiguous().npu(),
                    layer_map.npu(),
                )
        executor = NPUExecutor(
            spec,
            rank=rank,
            device=device,
            host=args.host,
            port=args.port,
            weights=view,
            timeout=args.timeout,
        )
        # Force unequal native launch counts before business work: F0 executes
        # alone while all other ranks remain at a host Store boundary. A hidden
        # MC2 per-launch collective pairing would fail this bounded assertion.
        if rank == spec.attention_ranks:
            for _ in range(3):
                mode = Mode.SERVE if args.serve else Mode.EXECUTE
                completion = executor.submit(mode)
                assert executor.retire(completion) == Result.IDLE
                assert not args.serve or completion.completed_work == 0
            if args.serve:
                check_served_receipts(executor, completion, report)
            executor.store.set("independent-idle-proven", "1")
        else:
            executor.store.wait(["independent-idle-proven"])
        if executor.is_ffn:
            # Test-only initialization creates different F arbitration order.
            executor.state[0] = (rank - spec.attention_ranks) % spec.source_count
            if args.max_eos:
                executor.state[1:] = MAX_GENERATION
            # This test deliberately mutates F's otherwise executor-owned
            # state on the current stream after initialization.
            executor.stream.wait_stream(torch.npu.current_stream())
            if not (args.empty_eos or args.max_eos or args.model_scope):
                executor.store.wait(
                    [f"prepublished/{a}" for a in range(spec.attention_ranks)]
                )
            ffn(
                executor,
                0 if args.empty_eos or args.max_eos else args.rounds,
                report,
                serve=args.serve,
            )
        else:
            if args.empty_eos or args.max_eos:
                last = MAX_GENERATION if args.max_eos else 0
                if args.max_eos:
                    executor.state[:2] = MAX_GENERATION
                assert (
                    executor.retire(executor.submit(Mode.END_SOURCE, sequence=last))
                    == Result.OK
                )
                rejected_after_eos(executor, last, report)
            elif checkpoint is not None:
                oracle = CheckpointOracle(checkpoint)
                cpu_reference = oracle.reference
            else:
                assert weights is not None
                cpu_reference = partial(
                    reference, spec=spec, w13=weights[0], w2=weights[1]
                )
            if not (args.empty_eos or args.max_eos):
                probability_dtype = (
                    torch.bfloat16
                    if args.probability_dtype == "bfloat16"
                    else torch.float32
                )
                if args.model_scope:
                    scoped_attention(
                        executor, args.rounds, cpu_reference, report, probability_dtype
                    )
                else:
                    attention(
                        executor, args.rounds, cpu_reference, report, probability_dtype
                    )
        executor.close()
        if owned_model is not None:
            owned_model.close()
        report["result"] = "passed"
        report["clean_close"] = True
        report["torch_peak_allocated_bytes"] = torch.npu.max_memory_allocated(device)
    except BaseException as error:
        report["result"] = "failed"
        report["error"] = traceback.format_exc()
        if executor is not None:
            executor.publish_failure(error)
        raise
    finally:
        if oracle is not None:
            oracle.close()
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / f"rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
