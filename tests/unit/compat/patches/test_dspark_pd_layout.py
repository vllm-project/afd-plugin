# SPDX-License-Identifier: Apache-2.0
"""Copy real CPU cache views through heterogeneous PD layouts without an NPU."""

from __future__ import annotations

import ctypes
import logging
import pickle
import sys
import threading
from collections import defaultdict
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

from afd_plugin.compat.patches.npu import dspark_pd_layout as patch


@pytest.mark.parametrize(
    "connector", [None, "P2pHcclAFDConnector", "WindowAFDConnector"]
)
@pytest.mark.parametrize("consumer", [False, True])
@pytest.mark.parametrize("dspark", [False, True])
def test_layout_scope_covers_native_and_afd(monkeypatch, connector, consumer, dspark):
    utils = ModuleType("vllm_ascend.utils")
    utils.is_dspark_config = lambda config: config.dspark
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    config = SimpleNamespace(
        additional_config=(
            {"afd": {"role": "attention", "connector": connector}}
            if connector is not None
            else {}
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakeHybridConnector",
            is_kv_consumer=consumer,
            is_kv_producer=not consumer,
        ),
        speculative_config=SimpleNamespace() if dspark else None,
        dspark=dspark,
        model_config=SimpleNamespace(dspark=dspark),
    )
    assert patch._uses_layer_layout(config) is dspark


@pytest.fixture
def layouts(monkeypatch):
    interface = ModuleType("vllm.v1.kv_cache_interface")
    interface.UniformTypeKVCacheSpecs = type("UniformTypeKVCacheSpecs", (), {})
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=4)
        )
    )

    def build(groups, padding, reverse=False):
        caches = {}
        cache_groups = []
        tensors = []
        names = [name for group in groups for name in group]
        # Different allocation/registration order on P and D is intentional.
        for name in reversed(names) if reverse else names:
            raw = torch.full((16 * padding,), 255, dtype=torch.uint8)
            view = torch.as_strided(raw, (16, 8), (padding, 1))
            caches[name] = [view]
            tensors.append(SimpleNamespace(shared_by=[name], size=raw.numel()))
        for group in groups:
            cache_groups.append(
                SimpleNamespace(
                    layer_names=group,
                    kv_cache_spec=SimpleNamespace(block_size=4, compress_ratio=1),
                )
            )
        cache_config = SimpleNamespace(
            num_blocks=16, kv_cache_groups=cache_groups, kv_cache_tensors=tensors
        )
        return (
            patch._target_layer_layouts(config, cache_config, caches),
            caches,
            cache_config,
        )

    names = [
        f"model.layers.{i}.self_attn.{kind}"
        for i, kind in (
            (0, "c4"),
            (1, "c128"),
            (0, "swa"),
            (1, "swa"),
            (2, "state_c4"),
            (3, "state_c128"),
        )
    ]
    remote, remote_cache, remote_config = build([[n] for n in names], 16)
    local, local_cache, local_config = build(
        [
            [names[0]],
            [names[1]],
            [names[2], names[3], "model.layers.4.self_attn.swa"],
            [names[4]],
            [names[5]],
        ],
        32,
        reverse=True,
    )
    # Exactly the failing attachment: P has six groups, D has five groups.
    remote_blocks = [[1], [], [2], [3], [0, 5, 6, 7, 8], [9, 10, 11]]
    local_blocks = [[1], [], [2], [3, 4, 5, 6], [7, 8, 9]]
    return SimpleNamespace(
        local=local,
        remote=remote,
        local_cache=local_cache,
        remote_cache=remote_cache,
        local_config=local_config,
        remote_config=remote_config,
        config=config,
        local_blocks=local_blocks,
        remote_blocks=remote_blocks,
    )


def test_six_to_five_groups_copy_correct_layers_and_keep_draft_local(layouts):
    v = layouts
    for layer in v.remote:
        for block in range(16):
            v.remote_cache[layer.name][0][block].fill_(layer.group * 16 + block)
    local_addrs, remote_addrs, lengths = patch._plan_layer_transfers(
        v.local, v.remote, v.local_blocks, v.remote_blocks
    )
    for dst, src, length in zip(local_addrs, remote_addrs, lengths, strict=True):
        ctypes.memmove(dst, src, length)
    remote_by_name = {layer.name: layer for layer in v.remote}
    for dst in v.local:
        src = remote_by_name[dst.name]
        dst_ids = v.local_blocks[dst.group]
        src_ids = v.remote_blocks[src.group][-len(dst_ids) :] if dst_ids else []
        for src_id, dst_id in zip(src_ids, dst_ids, strict=True):
            assert torch.equal(
                v.local_cache[dst.name][0][dst_id], v.remote_cache[src.name][0][src_id]
            )
    assert torch.all(v.local_cache["model.layers.4.self_attn.swa"][0] == 255)
    # Copies occupy eight bytes per page; trailing destination padding stays intact.
    for layer in v.local:
        raw = v.local_cache[layer.name][0].untyped_storage()
        for block in v.local_blocks[layer.group]:
            assert raw[block * layer.block_stride + 8] == 255


def test_overlapping_indexer_views_are_copied_once(layouts):
    layer = layouts.local[0]
    components = (
        patch.CacheComponent(0, 4, (4,), (1,), "torch.int8"),
        patch.CacheComponent(4, 4, (2,), (2,), "torch.float16"),
        patch.CacheComponent(0, 8, (8,), (1,), "torch.int8"),
    )
    local = replace(layer, components=components)
    remote = replace(layouts.remote[0], components=components)
    _, _, lengths = patch._plan_layer_transfers([local], [remote], [[1]], [[2]])
    assert lengths == [8]


@pytest.mark.parametrize(
    "fault",
    [
        "missing-layer",
        "dtype",
        "empty-remote",
        "too-short",
        "null-block",
        "bad-group",
        "bad-span",
    ],
)
def test_invalid_layout_or_block_list_fails_before_copy(layouts, fault):
    v = layouts
    remote = list(v.remote)
    remote_blocks = [list(ids) for ids in v.remote_blocks]
    if fault == "missing-layer":
        remote.pop()
    elif fault == "dtype":
        remote[0] = replace(
            remote[0],
            components=(replace(remote[0].components[0], dtype="torch.int8"),),
        )
    elif fault == "empty-remote":
        remote_blocks[4] = []
    elif fault == "too-short":
        remote_blocks[4] = [5, 6]
    elif fault == "null-block":
        remote_blocks[0] = [0]
    elif fault == "bad-group":
        remote[0] = replace(remote[0], group=99)
    else:
        remote[0] = replace(remote[0], block_stride=4)
    with pytest.raises(ValueError, match="DSpark PD:"):
        patch._plan_layer_transfers(v.local, remote, v.local_blocks, remote_blocks)
    assert all(torch.all(t[0] == 255) for t in v.local_cache.values())


@pytest.fixture
def protocol(monkeypatch):
    msgspec = pytest.importorskip("msgspec")
    module_path = "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector"
    # Mock only module ownership/threads. Use real msgpack encoding and the
    # patch's copied registration body; no worker or NPU is launched.
    for i in range(1, len(module_path.split(".")) + 1):
        name = ".".join(module_path.split(".")[:i])
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if i > 1:
            setattr(sys.modules[name.rsplit(".", 1)[0]], name.rsplit(".", 1)[1], module)
    upstream = sys.modules[module_path]

    class Metadata(msgspec.Struct, omit_defaults=True, dict=True):
        engine_id: str
        te_rpc_port: int
        block_size: int
        kv_caches_base_addr: list[int]
        num_blocks: int
        block_lens: list[int]
        ssm_sizes: tuple[int, int]
        local_ip: str = ""
        runtime_id: str = ""

    class Sender:
        def __init__(self, config, rank, tp, eid, host, port, metadata, ready, caches):
            self.metadata, self.ready = metadata, ready

        def start(self):
            self.wire = msgspec.msgpack.encode(self.metadata)
            self.ready.set()

    class Receiver:
        def __init__(self, *args):
            self.ready = args[15]
            self.engine = args[3]
            self.local_engine_id = args[4]
            self.decoder = msgspec.msgpack.Decoder(upstream.MooncakeAgentMetadata)
            self.encoder = msgspec.msgpack.Encoder()
            self.remote_metadata_lock = threading.Lock()
            self.remote_te_host = defaultdict(dict)
            self.remote_te_port = defaultdict(dict)
            self.kv_caches_base_addr = defaultdict(dict)

        def start(self):
            self.ready.set()

        def _get_remote_metadata(
            self, expected_engine_id, remote_host, remote_handshake_port
        ) -> None:
            raise AssertionError("original metadata getter")

        def _transfer_kv_cache_all_groups(self, req_meta):
            self.original_called = True

        def _get_remote_socket(self, host, port):
            return object()

        def _return_remote_socket(self, sock, host, port):
            pass

        def _validate_remote_runtime(self, metadata):
            assert metadata.runtime_id == "test"

        def _resolve_remote_transfer_endpoint(self, metadata, host):
            return host, metadata.te_rpc_port

    class Worker:
        def register_kv_caches(self, kv_caches):
            raise AssertionError("original registration")

        def start_load_kv(self, metadata):
            raise AssertionError("original start_load_kv")

    upstream.MooncakeAgentMetadata = Metadata
    upstream.msgspec = msgspec
    upstream.MooncakeConnectorWorker = Worker
    upstream.KVCacheSendingThread = Sender
    upstream.KVCacheRecvingThread = Receiver
    upstream.get_ip = lambda: "127.0.0.1"
    upstream.global_te = SimpleNamespace(register_buffer=lambda ptrs, sizes: None)
    upstream.logger = logging.getLogger("test-layer-layout")
    upstream.SizedDict = dict
    upstream.zmq = SimpleNamespace(Socket=type("Socket", (), {}))
    upstream.GET_META_MSG = b"get_meta_msg"
    upstream.ensure_zmq_send = lambda *args: None
    upstream.ensure_zmq_recv = lambda *args: upstream.wire
    monkeypatch.setattr(patch, "version", lambda package: "0.23.0")
    monkeypatch.setattr(patch, "_uses_layer_layout", lambda config: config.enabled)
    assert patch.apply_afd_dspark_pd_layout_patch()
    assert patch.apply_afd_dspark_pd_layout_patch()
    return upstream


def test_handshake_and_transfer_use_layer_mapping_before_threads_start(
    layouts, protocol
):
    v, u = layouts, protocol
    workers = []
    for role, config, caches in (
        ("kv_producer", v.remote_config, v.remote_cache),
        ("kv_consumer", v.local_config, v.local_cache),
    ):
        worker = u.MooncakeConnectorWorker()
        v.config.enabled = True
        v.config.model_config.is_deepseek_mla = True
        worker.__dict__.update(
            vllm_config=v.config,
            kv_cache_config=config,
            use_hybrid=True,
            use_mamba=False,
            use_compress=True,
            engine_id=role,
            te_rpc_port=8000,
            block_size=4,
            _mamba_ssm_size=(0, 0),
            mooncake_runtime_id="test",
            kv_role=role,
            tp_rank=0,
            tp_size=1,
            _prefill_tp_size=1,
            _prefill_pp_size=1,
            side_channel_host="127.0.0.1",
            side_channel_port=30000,
            handshake_port=30000,
            hma_group_size=len(config.kv_cache_groups),
            _prefill_pp_layer_partition=None,
            engine=SimpleNamespace(batch_transfer_sync_read=lambda *args: 0),
        )
        worker.register_kv_caches(caches)
        workers.append(worker)
    u.wire = workers[0].kv_send_thread.wire
    # Check byte-for-byte msgpack roundtrip of typed dataclasses.
    decoded = workers[1].kv_recv_thread.decoder.decode(workers[0].kv_send_thread.wire)
    assert decoded.afd_layer_layout == v.remote
    assert pickle.loads(pickle.dumps(decoded)) == decoded
    recv = workers[1].kv_recv_thread
    recv._transfer_kv_cache_all_groups(
        dict(
            remote_request_id="eos",
            remote_engine_id="kv_producer",
            remote_host="127.0.0.1",
            remote_handshake_port=30000,
            local_block_ids=v.local_blocks,
            remote_block_ids=v.remote_blocks,
        )
    )
    assert recv._afd_remote_layouts[("kv_producer", 30000)] == v.remote
    # A legacy sender is rejected explicitly; no positional fallback is allowed.
    import msgspec

    legacy = u.MooncakeAgentMetadata(
        "legacy", 8000, 4, [], 16, [], (0, 0), runtime_id="test"
    )
    protocol.wire = msgspec.msgpack.encode(legacy)
    with pytest.raises(RuntimeError, match="Prefill lacks layer layout metadata"):
        recv._get_remote_metadata("legacy", "127.0.0.1", 30000)
    # A non-DSpark worker keeps the original schema, decoder and transfer path.
    v.config.enabled = False
    workers[1].register_kv_caches(v.local_cache)
    assert type(workers[1].xfer_handshake_metadata) is u.MooncakeAgentMetadata
    normal_receiver = workers[1].kv_recv_thread
    normal_receiver._transfer_kv_cache_all_groups({})
    assert normal_receiver.original_called is True


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_fast_receive_completion_is_visible_to_scheduler(protocol, batch_size):
    """Force completion before enqueue returns; never rely on timing/sleeps."""
    tracker = SimpleNamespace(pending=set(), finished=set(), lost=[])
    batch_ids = {f"request-{i}" for i in range(batch_size)}

    def add_request(**kwargs):
        request_id = kwargs["request_id"]
        # A real transfer thread can reach completion while the model worker
        # is still enqueueing the rest of the same metadata batch.
        if request_id in tracker.pending:
            tracker.pending.remove(request_id)
            tracker.finished.add(request_id)
        else:
            tracker.lost.append(request_id)

    tracker.add_req_to_process = tracker.pending.add
    receiver = SimpleNamespace(task_tracker=tracker, add_request=add_request)
    worker = protocol.MooncakeConnectorWorker()
    worker.__dict__.update(
        _afd_layer_layout=["target"],
        kv_send_thread=None,
        kv_recv_thread=receiver,
        _prefill_tp_size=1,
        _prefill_pp_size=1,
        use_mamba=False,
        _get_tp_num_need_pulls=lambda tp: 1,
        _get_remote_rank=lambda request_id, tp: [0],
        _get_remote_host_info_by_port=lambda *args: ("prefill", "remote-engine"),
    )
    metadata = SimpleNamespace(
        reqs_in_batch=batch_ids,
        requests={
            request_id: SimpleNamespace(
                remote_request_id=f"remote-{request_id}",
                remote_engine_id="remote-engine",
                local_block_ids=[[1]],
                remote_block_ids=[[2]],
                remote_ptp_size=1,
                remote_port=30000,
                remote_host="prefill",
                remote_multi_nodes_meta_mapping={},
            )
            for request_id in batch_ids
        },
        requests_to_send={},
    )
    worker.start_load_kv(metadata)
    assert tracker.finished == batch_ids
    assert tracker.lost == []
    assert tracker.pending == set()  # no re-registration after completion
    # Demonstrate the same forced ordering reproduces the old upstream bug.
    worker._afd_layer_layout = None
    tracker.finished.clear()
    worker.start_load_kv(metadata)
    assert set(tracker.lost) == batch_ids
    assert tracker.pending == batch_ids
    assert tracker.finished == set()


def test_prefill_registers_before_immediate_no_transfer_completion(protocol):
    """Prefill tracking covers both delayed sending and ranks without a send."""
    pending, finished, delayed = set(), set(), {}
    tracker = SimpleNamespace(add_req_to_process=pending.add)

    def no_transfer(request_id):
        assert request_id in pending
        pending.remove(request_id)
        finished.add(request_id)

    def delay(request_id, start):
        assert request_id in pending
        delayed[request_id] = start

    sender = SimpleNamespace(
        task_tracker=tracker,
        add_delayed_request=delay,
        add_not_transfer_request=no_transfer,
    )
    worker = protocol.MooncakeConnectorWorker()
    worker.__dict__.update(
        _afd_layer_layout=["target"],
        kv_send_thread=sender,
        kv_recv_thread=None,
        tp_rank=0,
        _prefill_get_remote_rank=lambda rid: [0] if rid == "send" else [1],
    )
    worker.start_load_kv(
        SimpleNamespace(
            reqs_in_batch={"send", "skip"},
            requests={},
            requests_to_send={"send": 123.0, "skip": 124.0},
        )
    )
    assert pending == {"send"}
    assert finished == {"skip"}
    assert delayed == {"send": 123.0}
