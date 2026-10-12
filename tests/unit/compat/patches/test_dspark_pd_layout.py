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
