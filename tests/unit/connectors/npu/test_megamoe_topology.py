# SPDX-License-Identifier: Apache-2.0
"""Exercise configuration validation without loading CANN or creating groups."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validate():
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/async_megamoe.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_megamoe_config"
        or isinstance(node, ast.Assign)
    ]
    from datetime import timedelta

    ns = dict(
        timedelta=timedelta,
        torch=SimpleNamespace(bfloat16="bf16"),
        get_ascend_device_type=lambda: "A5",
        AscendDeviceType=SimpleNamespace(A5="A5"),
    )
    exec(compile(tree, str(path), "exec"), ns)
    return ns["validate_megamoe_config"]


@pytest.mark.parametrize(
    "a,f,role,dp,valid",
    [
        (2, 2, "ffn", 2, True),
        (6, 2, "attention", 3, False),
        (6, 4, "ffn", 2, False),
        (1, 1, "attention", 1, True),
        (3, 2, "attention", 3, True),
        (3, 2, "ffn", 2, True),
        (1, 4, "ffn", 4, True),
        (3, 2, "attention", 2, False),
        (2, 3, "ffn", 3, False),
        (0, 1, "ffn", 1, False),
        (1, 0, "attention", 1, False),
        (63, 1, "ffn", 1, True),
        (64, 1, "ffn", 1, False),
        (768, 1, "ffn", 1, False),
    ],
)
@pytest.mark.parametrize("tp", [1, 2])
def test_role_sizes_and_tiling_limits(validate, a, f, role, dp, valid, tp):
    obj = SimpleNamespace
    hf = obj(
        model_type="deepseek_v4",
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        num_experts_per_tok=6,
        expert_dtype="fp4",
    )
    parallel = obj(
        data_parallel_size=dp,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        enable_eplb=False,
        enable_elastic_ep=False,
        use_sequence_parallel_moe=False,
        eplb_config=obj(num_redundant_experts=0),
        enable_dbo=False,
        use_ubatching=False,
    )
    config = obj(
        model_config=obj(hf_config=hf, dtype="bf16", enforce_eager=True),
        scheduler_config=obj(max_num_batched_tokens=1024),
        additional_config={},
        parallel_config=parallel,
        use_v2_model_runner=True,
        speculative_config=None,
    )
    afd = obj(
        role=role,
        num_attention_ranks=a,
        num_ffn_ranks=f,
        async_dp=True,
        compute_gate_on_attention=True,
    )
    if tp == 2:
        valid = (
            a > 0
            and f > 0
            and 256 % f == 0
            and a + f <= 768
            and (a + f) * (256 // f) <= 16384
            and dp * tp == (a if role == "attention" else f)
        )
    if valid:
        validate(config, afd)
    else:
        with pytest.raises(ValueError):
            validate(config, afd)


@pytest.fixture
def connector_class():
    # Exercise the connector's wire calls without importing the NPU runtime.
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/async_megamoe.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name.startswith("CAMAsync")
    )
    cls.bases = []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"send_attn_output", "recv_ffn_output", "dp_members", "execute_next_experts"}
    ]
    tree.body = [cls]
    ns = {
        "torch": SimpleNamespace(npu=SimpleNamespace()),
        "time": SimpleNamespace(sleep=lambda _: None),
        "POLL_SECONDS": 0,
    }
    exec(compile(tree, str(path), "exec"), ns)
    return ns[cls.name], ns["torch"].npu


def test_tp_members_share_generation_and_publish_once(connector_class):
    connector, _ = connector_class
    notifications = [[], []]
    tensor = SimpleNamespace()
    tensor.contiguous = tensor.int = tensor.float = lambda: tensor

    def dispatch(hidden, ids, weights, buffer, members, sequence):
        return buffer, members, sequence

    def combine(ticket, buffer):
        assert ticket[0] == buffer
        return ticket

    for i in range(6):
        rank = connector()
        rank.role_rank, rank.attn_tp = i, 2
        rank.ops = SimpleNamespace(
            split_sequence=lambda n: (n - 1) % 32 + 1,
            megamoe_dispatch_send=dispatch,
            megamoe_combine_recv=combine,
        )
        rank.stages = [
            SimpleNamespace(
                buffer=stage,
                sequence=seq,
                ticket=None,
                work=SimpleNamespace(publish=lambda *args, log=log: log.append(args)),
            )
            for stage, (seq, log) in enumerate(
                zip((32, 12), notifications, strict=True)
            )
        ]
        for stage in (0, 1):
            ctx = SimpleNamespace(
                metadata=SimpleNamespace(stage_idx=stage, layer_idx=7)
            )
            rank.send_attn_output(tensor, ctx, topk_ids=tensor, topk_weights=tensor)
            with pytest.raises(RuntimeError, match="outstanding"):
                rank.send_attn_output(tensor, ctx, topk_ids=tensor, topk_weights=tensor)
        members = rank.dp_members(i // 2)
        assert rank.recv_ffn_output(tensor, 1) == (1, members, 13)
        assert rank.recv_ffn_output(tensor, 0) == (0, members, 1)
        with pytest.raises(RuntimeError, match="matching ubatch"):
            rank.recv_ffn_output(tensor, 1)
    assert notifications == [[(dp, 7, seq) for dp in range(3)] for seq in (1, 13)]


def test_ffn_submits_free_stage_without_waiting_for_busy_peer(connector_class):
    connector, npu = connector_class

    class Event:
        done = False

        def record(self, stream):
            self.done = False
            self.stream = stream

        def query(self):
            return self.done

    npu.stream = lambda stream: nullcontext()
    npu.Event = Event
    npu.synchronize = lambda: pytest.fail("unexpected device-wide wait")
    rank = connector()
    rank.role_rank, rank.attn_tp, rank.next_stage = 0, 2, 0
    rank.weights = {7: ("layer7",), 8: ("layer8",)}
    calls = []
    rank.ops = SimpleNamespace(megamoe_fused_moe=lambda *args: calls.append(args))
    rank.stages = []
    for stage in range(2):
        tasks = iter([(stage, 7 + stage, 1), (stage, 7 + stage, 2)])
        rank.stages.append(
            SimpleNamespace(
                buffer=stage,
                stream=stage,
                completion=None,
                work=SimpleNamespace(poll=lambda _, q=tasks: next(q)),
            )
        )
    for _ in range(3):
        rank.execute_next_experts()
    assert calls == [("layer7", 0, [0, 1], 1), ("layer8", 1, [2, 3], 1)]
    assert [lane.completion.stream for lane in rank.stages] == [0, 1]
    rank.stages[1].completion.done = True
    rank.execute_next_experts()
    assert calls[-1] == ("layer8", 1, [2, 3], 2)
