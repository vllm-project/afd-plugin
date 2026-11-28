# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise configuration validation without loading CANN or creating groups."""

from __future__ import annotations

import ast
import sys
import threading
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.validation import validate_npu_model_runner_v2_config
from tests.unit.fakes.vllm_config import npu_async_cam_target_graph_config


@pytest.fixture
def validate(monkeypatch):
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/async_mega/connector.py"
    )
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_mega_config"
        or isinstance(node, ast.Assign)
    ]
    from datetime import timedelta

    utils = ModuleType("vllm_ascend.utils")
    monkeypatch.setattr(utils, "get_ascend_device_type", lambda: "A5", raising=False)
    monkeypatch.setattr(
        utils, "AscendDeviceType", SimpleNamespace(A5="A5"), raising=False
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    ns = dict(
        timedelta=timedelta,
        torch=SimpleNamespace(bfloat16="bf16"),
    )
    exec(compile(tree, str(path), "exec"), ns)
    return ns["validate_mega_config"]


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
    model = obj(num_experts=256)
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
        model_config=obj(dtype="bf16", enforce_eager=True),
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
        validate(config, afd, model)
    else:
        with pytest.raises(ValueError):
            validate(config, afd, model)


@pytest.fixture
def connector_class():
    # Exercise the connector's wire calls without importing the NPU runtime.
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/async_mega/connector.py"
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
        in {
            "send_attn_output",
            "recv_ffn_output",
            "_receive",
            "dp_members",
            "execute_next_experts",
            "close",
            "_wait_device_idle",
            "_wait_peer_drain",
        }
    ]
    tree.body = [cls]
    clock = iter(range(1000))
    npu = SimpleNamespace()
    ns = {
        "timedelta": timedelta,
        "CLOSE_LOCK_TIMEOUT_SECONDS": 5,
        "CLOSE_DEVICE_TIMEOUT_SECONDS": 5,
        "CLOSE_PEER_TIMEOUT_SECONDS": 5,
        "dist": SimpleNamespace(destroy_process_group=lambda group: group.destroy()),
        "torch": SimpleNamespace(npu=npu),
        "time": SimpleNamespace(sleep=lambda _: None, monotonic=lambda: next(clock)),
        "POLL_SECONDS": 0,
        "dbo_enabled": lambda: False,
        "dbo_maybe_run_recv_hook": lambda: None,
        "dbo_register_recv_hook": lambda _hook: None,
    }
    exec(compile(tree, str(path), "exec"), ns)
    return ns[cls.name], npu


def test_both_ubatches_share_one_window_and_one_sequence_ring(connector_class):
    connector, _ = connector_class
    tensor = SimpleNamespace()
    tensor.contiguous = tensor.int = tensor.float = lambda: tensor

    def dispatch(hidden, ids, weights, buffer, members, sequence, *, layer_index):
        assert layer_index == 1
        return buffer, members, sequence

    def combine(ticket, buffer):
        assert ticket[0] == buffer
        return ticket

    for i in range(6):
        rank = connector()
        rank.role_rank, rank.attn_tp = i, 2
        rank.layer_slots = {2: 0, 7: 1}
        rank.num_ubatches = 2
        rank.ubatch_tickets = [None, None]
        rank.ubatch_outputs = [None, None]
        rank._recv_locks = [threading.Lock(), threading.Lock()]
        rank._send_lock = threading.Lock()
        rank.ops = SimpleNamespace(
            atten_send=dispatch,
            atten_recv=combine,
            split_sequence=lambda value: (value - 1) % 32 + 1,
        )
        rank.lane = SimpleNamespace(buffer=0, sequence=32)
        for stage in (0, 1):
            ctx = SimpleNamespace(
                metadata=SimpleNamespace(stage_idx=stage, layer_idx=7)
            )
            rank.send_attn_output(tensor, ctx, topk_ids=tensor, topk_weights=tensor)
            with pytest.raises(RuntimeError, match="outstanding"):
                rank.send_attn_output(tensor, ctx, topk_ids=tensor, topk_weights=tensor)
        start = i // 2 * 2
        members = [start, start + 1]
        assert rank.recv_ffn_output(tensor, 1) == (0, members, 2)
        assert rank.recv_ffn_output(tensor, 1) == (0, members, 2)
        assert rank.recv_ffn_output(tensor, 0) == (0, members, 1)
        with pytest.raises(RuntimeError, match="Unexpected Mega Attention ubatch"):
            rank.recv_ffn_output(tensor, 2)


def test_ffn_submits_the_single_lane_only_when_free(connector_class):
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
    rank._step_lock = threading.Lock()
    rank._closed = False
    rank.layered_weights = (("layer7", "layer8"),) * 4
    calls = []
    rank.ops = SimpleNamespace(mega_ffn=lambda *args: calls.append(args))
    rank.lane = SimpleNamespace(buffer="window", stream="s0", completion=None)

    for _ in range(3):
        rank.execute_next_experts()
    assert calls == [(*rank.layered_weights, "window")]
    assert rank.lane.completion.stream == "s0"

    rank.lane.completion.done = True
    rank.execute_next_experts()
    assert [args[-1] for args in calls] == ["window", "window"]


class _ShutdownStore:
    """Deterministic transport: missing peers fail the explicitly bounded wait."""

    def __init__(self, peers=(), disconnected=False):
        self.messages = {f"drained/{rank}": "1" for rank in peers}
        self.disconnected = disconnected
        self.waited = []

    def set_timeout(self, timeout):
        assert 0 < timeout.total_seconds() <= 5

    def set(self, key, value):
        if self.disconnected:
            raise OSError("Store server disconnected")
        self.messages[key] = value

    def wait(self, keys, timeout):
        assert 0 < timeout.total_seconds() <= 5
        self.waited.append(keys)
        if any(key not in self.messages for key in keys):
            raise TimeoutError("peer absent")


@pytest.fixture
def closing_connector(connector_class):
    cls, npu = connector_class
    npu.Event = lambda: SimpleNamespace(record=lambda: None, query=lambda: True)

    def make(store, rank=0, device_done=True):
        connector = cls()
        released = []
        connector._step_lock = threading.Lock()
        connector._closed = False
        connector._shutdown_store = store
        connector.world_rank = rank
        connector.attn_size = connector.ffn_size = 1
        connector.layer_ids = [0]
        connector.layer_slots = {0: 0}
        connector.layered_weights = (["weights"],)
        connector.lane = SimpleNamespace(
            completion=SimpleNamespace(query=lambda: device_done),
            buffer=SimpleNamespace(destroy=lambda: released.append("window")),
            group=SimpleNamespace(destroy=lambda: released.append("group")),
        )
        return connector, released

    return make


@pytest.mark.parametrize("disconnected", [False, True])
def test_close_retains_resources_when_peer_cannot_acknowledge(
    closing_connector, disconnected
):
    store = _ShutdownStore(disconnected=disconnected)
    connector, released = closing_connector(store)
    with pytest.raises(
        RuntimeError, match="peers did not confirm shutdown; retain resources"
    ):
        connector.close()
    assert released == []
    assert connector.layered_weights == (["weights"],)
    if not disconnected:
        assert store.messages == {"drained/0": "1"}
        assert store.waited == [["drained/0", "drained/1"]]


def test_failed_local_drain_leaves_peer_wait_bounded(closing_connector):
    store = _ShutdownStore()
    failed, failed_released = closing_connector(store, rank=1, device_done=False)
    drained, drained_released = closing_connector(store)
    with pytest.raises(RuntimeError, match="device work did not stop"):
        failed.close()
    assert store.messages == {}
    with pytest.raises(RuntimeError, match="peers did not confirm shutdown"):
        drained.close()
    assert failed_released == drained_released == []
    assert store.messages == {"drained/0": "1"}


def test_close_releases_only_after_all_ranks_acknowledge(closing_connector):
    store = _ShutdownStore(peers=(1,))
    connector, released = closing_connector(store)
    connector.close()
    connector.close()  # Already closed: no second destruction or handshake.
    assert store.messages == {"drained/0": "1", "drained/1": "1"}
    assert store.waited == [["drained/0", "drained/1"]]
    assert released == ["window", "group"]


@pytest.mark.parametrize("dp_size,tp_size", [(1, 1), (1, 2), (2, 2)])
def test_mega_eager_dbo_admits_attention_topology(monkeypatch, dp_size, tp_size):
    model_utils = ModuleType("afd_plugin.model_executor.models.model_utils")
    monkeypatch.setattr(
        model_utils, "has_afd_model_registration", lambda config: True, raising=False
    )
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)
    config = npu_async_cam_target_graph_config(
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        num_attention_ranks=dp_size * tp_size,
    )
    config.additional_config["afd"]["connector"] = "CAMAsyncMegaAFDConnector"
    config.model_config.enforce_eager = True
    config.parallel_config.enable_dbo = True
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    validate_npu_model_runner_v2_config(
        config, expected_role="attention", device_type="npu"
    )
