# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Layer-addressed Mooncake transfers for DeepSeek V4 DSpark PD.

Pinned upstream: vllm-ascend 11ee45653, mooncake_hybrid_connector.py.
DSpark changes both hybrid group membership and physical tensor ordering.
Group indices and registration-array positions are therefore local to each
worker. Exchange target-layer layouts before copying; never zip those arrays.
Remove this patch when upstream negotiates heterogeneous hybrid layouts.
"""

from __future__ import annotations

import inspect
import math
import re
import threading
import time
from dataclasses import dataclass, fields
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from afd_plugin.compat.patches.npu.dspark_pd import _uses_dspark_pd

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_PATCH_ATTR = "_afd_plugin_dspark_pd_layer_layout_patch"
_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class CacheComponent:
    offset: int
    size: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LayerLayout:
    name: str
    group: int
    spec_type: str
    block_size: int
    compress_ratio: int
    sliding_window: int | None
    base_addr: int
    block_stride: int
    num_blocks: int
    components: tuple[CacheComponent, ...]


@dataclass
class LayerAgentMetadata:
    """Same map wire format as Mooncake metadata, with an optional extension.

    A module-level dataclass is also pickleable for worker-to-scheduler RPC.
    Defining a local msgspec subclass in the installer breaks that RPC.
    """

    engine_id: str
    te_rpc_port: int
    block_size: int
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_lens: list[int]
    ssm_sizes: tuple[int, int]
    local_ip: str = ""
    runtime_id: str = ""
    afd_layout_version: int = 0
    afd_layer_layout: list[LayerLayout] | None = None


def _uses_layer_layout(config: VllmConfig) -> bool:
    kv = config.kv_transfer_config
    if kv is None or kv.kv_connector != "MooncakeHybridConnector":
        return False
    if kv.is_kv_consumer:
        return _uses_dspark_pd(config)
    # Prefill has no speculative config, but uses the same DSpark checkpoint.
    from vllm_ascend.utils import is_dspark_config

    return kv.is_kv_producer and is_dspark_config(config.model_config)


def _target_layer_layouts(
    config: VllmConfig,
    cache_config: KVCacheConfig,
    caches: dict[str, torch.Tensor],
) -> list[LayerLayout]:
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    layouts = []
    num_target_layers = config.model_config.hf_text_config.num_hidden_layers
    for group_id, group in enumerate(cache_config.kv_cache_groups):
        for name in group.layer_names:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if match is None:
                raise ValueError(f"DSpark PD: unrecognized cache layer name {name}")
            if int(match[1]) >= num_target_layers:
                # DSpark draft caches have no Prefill counterpart.
                continue
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            tensors = caches[name]
            if not isinstance(tensors, (tuple, list)):
                tensors = [tensors]
            base = min(t.data_ptr() for t in tensors)
            strides = {t.stride(0) * t.element_size() for t in tensors}
            if len(strides) != 1:
                raise ValueError(f"DSpark PD: inconsistent block strides for {name}")
            block_stride = strides.pop()
            components = []
            for tensor in tensors:
                if tensor.shape[0] != cache_config.num_blocks:
                    raise ValueError(f"DSpark PD: inconsistent block count for {name}")
                shape = tuple(tensor.shape[1:])
                inner_strides = tuple(
                    stride * tensor.element_size() for stride in tensor.stride()[1:]
                )
                size = tensor.element_size() + sum(
                    (dim - 1) * stride
                    for dim, stride in zip(shape, inner_strides, strict=True)
                )
                offset = tensor.data_ptr() - base
                if size <= 0 or offset < 0 or offset + size > block_stride:
                    raise ValueError(f"DSpark PD: component exceeds block for {name}")
                components.append(
                    CacheComponent(
                        offset, size, shape, inner_strides, str(tensor.dtype)
                    )
                )
            layouts.append(
                LayerLayout(
                    name,
                    group_id,
                    type(spec).__name__,
                    spec.block_size,
                    # These optional fields distinguish MLA and compressor/SWA
                    # specs; their absence is part of the upstream spec contract.
                    getattr(spec, "compress_ratio", 1),
                    getattr(spec, "sliding_window", None),
                    base,
                    block_stride,
                    cache_config.num_blocks,
                    tuple(components),
                )
            )
    if not layouts:
        raise ValueError("DSpark PD: no target cache layers to transfer")
    return layouts


def _plan_layer_transfers(
    local: list[LayerLayout],
    remote: list[LayerLayout],
    local_blocks: list[list[int]],
    remote_blocks: list[list[int]],
) -> tuple[list[int], list[int], list[int]]:
    """Validate the entire request before producing a Mooncake read batch."""
    remote_by_name = {layer.name: layer for layer in remote}
    if len(remote_by_name) != len(remote) or len(
        {layer.name for layer in local}
    ) != len(local):
        raise ValueError("DSpark PD: duplicate layer names in handshake")
    if {layer.name for layer in local} != remote_by_name.keys():
        raise ValueError("DSpark PD: target layer names differ between peers")
    transfers = []
    for dst in local:
        src = remote_by_name[dst.name]
        if (
            dst.spec_type,
            dst.block_size,
            dst.compress_ratio,
            dst.sliding_window,
            dst.components,
        ) != (
            src.spec_type,
            src.block_size,
            src.compress_ratio,
            src.sliding_window,
            src.components,
        ):
            raise ValueError(f"DSpark PD: incompatible cache layout for {dst.name}")
        if not (
            0 <= dst.group < len(local_blocks) and 0 <= src.group < len(remote_blocks)
        ):
            raise ValueError(f"DSpark PD: missing block group for {dst.name}")
        dst_ids, src_ids = local_blocks[dst.group], remote_blocks[src.group]
        if not dst_ids:
            continue
        if len(src_ids) < len(dst_ids):
            raise ValueError(
                f"DSpark PD: insufficient remote blocks for {dst.name}: "
                f"local group {dst.group}={len(dst_ids)}, "
                f"remote group {src.group}={len(src_ids)}"
            )
        # Preserve upstream SWA/state suffix selection after matching the layer.
        src_ids = src_ids[-len(dst_ids) :]
        if any(block <= 0 or block >= dst.num_blocks for block in dst_ids) or any(
            block <= 0 or block >= src.num_blocks for block in src_ids
        ):
            raise ValueError(
                f"DSpark PD: null/out-of-range transfer block for {dst.name}"
            )
        # The A5 int8 indexer exposes overlapping K, scale and full-page views.
        # Merge their byte intervals so every occupied region is copied once.
        intervals = []
        for c in sorted(dst.components, key=lambda c: c.offset):
            end = c.offset + c.size
            if (
                c.offset < 0
                or c.size <= 0
                or end > min(dst.block_stride, src.block_stride)
            ):
                raise ValueError(f"DSpark PD: invalid component span for {dst.name}")
            if intervals and c.offset <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            else:
                intervals.append((c.offset, end))
        for src_id, dst_id in zip(src_ids, dst_ids, strict=True):
            for begin, end in intervals:
                # Mooncake read takes local destination first, remote source second.
                transfers.append(
                    (
                        dst.base_addr + dst_id * dst.block_stride + begin,
                        src.base_addr + src_id * src.block_stride + begin,
                        end - begin,
                    )
                )
    unique = list(dict.fromkeys(transfers))
    return ([t[0] for t in unique], [t[1] for t in unique], [t[2] for t in unique])


def apply_afd_dspark_pd_layout_patch() -> bool:
    """Install after Ascend worker resolution, before KV cache registration."""
    if not version("vllm").startswith("0.23.0"):
        return False
    # Lazy worker import avoids partially initialized Ascend ops/platforms.
    from vllm_ascend.distributed.kv_transfer.kv_p2p import (
        mooncake_hybrid_connector as upstream,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        GET_META_MSG,
        KVCacheRecvingThread,
        KVCacheSendingThread,
        MooncakeAgentMetadata,
        ensure_zmq_recv,
        ensure_zmq_send,
        get_ip,
        global_te,
        logger,
        zmq,
    )

    worker = upstream.MooncakeConnectorWorker
    receiver = upstream.KVCacheRecvingThread
    if hasattr(worker, _PATCH_ATTR):
        return True
    if (
        tuple(inspect.signature(worker.register_kv_caches).parameters)
        != ("self", "kv_caches")
        or tuple(inspect.signature(receiver._get_remote_metadata).parameters)
        != ("self", "expected_engine_id", "remote_host", "remote_handshake_port")
        or tuple(inspect.signature(receiver._transfer_kv_cache_all_groups).parameters)
        != ("self", "req_meta")
        or upstream.MooncakeAgentMetadata.__struct_fields__
        != tuple(
            field.name
            for field in fields(LayerAgentMetadata)
            if not field.name.startswith("afd_")
        )
    ):
        return False

    original_transfer = receiver._transfer_kv_cache_all_groups

    # Patch reason: upstream zips group IDs and raw addresses, both of which
    # change when DSpark cache layers are added only on the consumer.
    # Patch functionality: negotiate and validate per-target-layer addresses.
    # Expansion exception: retain the 86-line upstream implementation for other
    # connectors/layouts; the scoped path replaces the whole transfer algorithm.
    # Signature: matches the pinned upstream method, with no return annotation.
    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        # ### PATCH START: layer-addressed DSpark KV reads
        if self._afd_layer_layout is None:
            return original_transfer(self, req_meta)
        if not any(req_meta["local_block_ids"]):
            return
        peer = (req_meta["remote_engine_id"], req_meta["remote_handshake_port"])
        with self.remote_metadata_lock:
            cached = (
                peer in self._afd_remote_layouts
                and peer[0] in self.remote_te_host
                and peer[1] in self.remote_te_host[peer[0]]
                and peer[0] in self.remote_te_port
                and peer[1] in self.remote_te_port[peer[0]]
            )
        if not cached:
            self._get_remote_metadata(peer[0], req_meta["remote_host"], peer[1])
        with self.remote_metadata_lock:
            remote_layout = self._afd_remote_layouts[peer]
            host = self.remote_te_host[peer[0]][peer[1]]
            port = self.remote_te_port[peer[0]][peer[1]]
        local_addrs, remote_addrs, lengths = _plan_layer_transfers(
            self._afd_layer_layout,
            remote_layout,
            req_meta["local_block_ids"],
            req_meta["remote_block_ids"],
        )
        session_id = f"{host}:{port}"
        start = time.perf_counter()
        if lengths:
            ret = self.engine.batch_transfer_sync_read(
                session_id, local_addrs, remote_addrs, lengths
            )
            if ret < 0:
                raise RuntimeError(
                    f"Mooncake transfer failed for session {session_id}, ret: {ret}"
                )
        logger.info(
            "AFD DSpark PD layer KV transfer: request=%s local_groups=%d "
            "remote_groups=%d regions=%d bytes=%d elapsed_ms=%.2f",
            req_meta["remote_request_id"],
            len(req_meta["local_block_ids"]),
            len(req_meta["remote_block_ids"]),
            len(lengths),
            sum(lengths),
            (time.perf_counter() - start) * 1000,
        )
        # ### PATCH END: layer-addressed DSpark KV reads

    # The pinned register_kv_caches and _get_remote_metadata bodies below are
    # copied verbatim except for the marked protocol-extension blocks.

    # Patch reason: the handshake needs layer/group/stride information that the
    # original flat registration arrays omit.
    # Patch functionality: publish target layouts and initialize receiver state.
    # Signature: matches register_kv_caches; body copied from 11ee45653.
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(
            self.vllm_config.model_config.hf_text_config, "index_topk"
        )

        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        self.kv_caches_base_addr = []
        self.block_len_per_addr: list[int] = []
        self.block_stride_per_addr: list[int] = []
        self.addr_group_idx: list[int] = []
        ptrs = []
        lengths = []
        if not self.use_hybrid:
            for layer_name, kv_cache_tuple in kv_caches.items():  # noqa: B007 - copied upstream
                if isinstance(kv_cache_tuple, (list, tuple)) is False:
                    kv_cache_tuple = [kv_cache_tuple]
                for single_kv_cache in kv_cache_tuple:
                    tensor_num_blocks = single_kv_cache.shape[0]
                    block_size_scale = tensor_num_blocks // self.num_blocks
                    block_shape = single_kv_cache.shape[1:]
                    self.block_len_per_addr.append(
                        single_kv_cache.element_size()
                        * math.prod(block_shape)
                        * block_size_scale
                    )
                    self.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                    ptrs.append(single_kv_cache.data_ptr())
                    lengths.append(
                        single_kv_cache.element_size()
                        * math.prod(single_kv_cache.shape)
                    )
        elif self.use_mamba:
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                share_tensor_addr = []
                for layer_name in kv_cache_tensor.shared_by:
                    kv_cache_tuple = kv_caches[layer_name]
                    if isinstance(kv_cache_tuple, (list, tuple)) is False:
                        kv_cache_tuple = [kv_cache_tuple]
                    for single_kv_cache in kv_cache_tuple:
                        if single_kv_cache.data_ptr() in self.kv_caches_base_addr:
                            continue
                        tensor_num_blocks = single_kv_cache.shape[0]
                        block_size_scale = tensor_num_blocks // self.num_blocks
                        block_shape = single_kv_cache.shape[1:]
                        self.block_len_per_addr.append(
                            single_kv_cache.element_size()
                            * math.prod(block_shape)
                            * block_size_scale
                        )
                        self.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                        share_tensor_addr.append(single_kv_cache.data_ptr())
                if share_tensor_addr:
                    ptrs.append(min(share_tensor_addr))
                    lengths.append(kv_cache_tensor.size)
            self.block_stride_per_addr.extend(self.block_len_per_addr)
        elif self.use_compress:
            layer_group_idx = dict[str, int]()
            for i, group in enumerate(self.kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    layer_group_idx[layer_name] = i
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                if not kv_cache_tensor.shared_by:
                    continue
                share_tensor_addr = []
                share_tensor_stride = []
                cur_tensor_group_idx = []
                for layer_name in kv_cache_tensor.shared_by:
                    cur_tensor_group_idx.append(layer_group_idx[layer_name])
                    kv_cache_tuple = kv_caches[layer_name]
                    if not isinstance(kv_cache_tuple, (tuple, list)):
                        kv_cache_tuple = kv_cache_tuple
                    for single_tensor in kv_cache_tuple:
                        tensor_addr = single_tensor.data_ptr()
                        if (
                            tensor_addr in share_tensor_addr
                            or tensor_addr in self.kv_caches_base_addr
                        ):
                            continue
                        share_tensor_addr.append(tensor_addr)
                        share_tensor_stride.append(
                            single_tensor.stride(0) * single_tensor.element_size()
                        )
                cur_tensor_group_idx = sorted(list(set(cur_tensor_group_idx)))
                self.kv_caches_base_addr.append(min(share_tensor_addr))
                self.addr_group_idx.append(cur_tensor_group_idx)  # type: ignore[arg-type]
                self.block_stride_per_addr.append(share_tensor_stride[0])
                self.block_len_per_addr.append(share_tensor_stride[0])
                ptrs.append(min(share_tensor_addr))
                lengths.append(kv_cache_tensor.size)
        else:
            raise TypeError(
                "Mooncake connector does not support this type kv_cache now."
            )

        # ### PATCH START: describe target layers before publishing handshake
        layer_layout = None
        if _uses_layer_layout(self.vllm_config):
            layer_layout = _target_layer_layouts(
                self.vllm_config, self.kv_cache_config, kv_caches
            )
            logger.info(
                "AFD DSpark PD layer layout active: role=%s groups=%d "
                "target_layers=%d mapping=%s",
                self.kv_role,
                self.hma_group_size,
                len(layer_layout),
                [(layer.name, layer.group) for layer in layer_layout],
            )
        # ### PATCH END: describe target layers before publishing handshake
        global_te.register_buffer(ptrs, lengths)
        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            block_size=self.block_size,
            kv_caches_base_addr=self.kv_caches_base_addr,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_addr,
            ssm_sizes=self._mamba_ssm_size,
            local_ip=get_ip(),
            runtime_id=self.mooncake_runtime_id,
        )
        # ### PATCH START: extend scoped handshake with target layer layouts
        if layer_layout is not None:
            metadata = LayerAgentMetadata(
                **upstream.msgspec.to_builtins(metadata),
                afd_layout_version=_LAYOUT_VERSION,
                afd_layer_layout=layer_layout,
            )
        # ### PATCH END: extend scoped handshake with target layer layouts
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self._prefill_tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
            )
            self.kv_send_thread.start()
        else:
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                self.kv_caches_base_addr,
                self.block_len_per_addr,
                self.block_stride_per_addr,
                self.addr_group_idx,
                self._mamba_ssm_size,
                self.use_hybrid,
                self.use_mamba,
                self.hma_group_size,
                ready_event,
                self.vllm_config,
                self.kv_cache_config,
                self.kv_caches,
                self._prefill_pp_layer_partition,
                self.mooncake_runtime_id,
            )
            # ### PATCH START: initialize transfer state before thread startup
            self.kv_recv_thread._afd_layer_layout = layer_layout
            self.kv_recv_thread._afd_remote_layouts = upstream.SizedDict()
            if layer_layout is not None:
                self.kv_recv_thread.decoder = upstream.msgspec.msgpack.Decoder(
                    LayerAgentMetadata
                )
            # ### PATCH END: initialize transfer state before thread startup
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = (
            self.kv_send_thread
            if self.kv_role == "kv_producer"
            else self.kv_recv_thread
        )
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)

    # Patch reason: a DSpark consumer must not read from a legacy positional peer.
    # Patch functionality: validate and cache the optional versioned layout.
    # Signature: matches _get_remote_metadata; body copied from 11ee45653.
    def _get_remote_metadata(
        self,
        expected_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Get the metadata from the remote host."""
        if not remote_host or not remote_host.strip():
            raise RuntimeError("Mooncake remote handshake host is empty.")
        if remote_handshake_port <= 0 or remote_handshake_port > 65535:
            raise RuntimeError(
                f"Mooncake remote handshake port is invalid: {remote_handshake_port}."
            )

        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(
                sock,
                self.encoder.encode((GET_META_MSG, "")),
                f"{remote_host}:{remote_handshake_port}",
            )
            metadata_bytes = ensure_zmq_recv(
                sock, f"{remote_host}:{remote_handshake_port}"
            )
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            )
            if engine_id != expected_engine_id:
                raise RuntimeError(
                    f"Mooncake remote metadata engine ID mismatch: expected {expected_engine_id}, received {engine_id}."
                )
            self._validate_remote_runtime(agent_meta)
            # ### PATCH START: require compatible target layouts for DSpark PD
            if self._afd_layer_layout is not None and (
                agent_meta.afd_layout_version != _LAYOUT_VERSION
                or agent_meta.afd_layer_layout is None
            ):
                raise RuntimeError(
                    "AFD DSpark PD: Prefill lacks layer layout metadata; "
                    "update the AFD plugin and restart both Prefill and Attention."
                )
            # ### PATCH END: require compatible target layouts for DSpark PD
            transfer_host, transfer_port = self._resolve_remote_transfer_endpoint(
                agent_meta, remote_host
            )
            with self.remote_metadata_lock:
                # ### PATCH START: cache negotiated layer layout under peer identity
                if self._afd_layer_layout is not None:
                    self._afd_remote_layouts[(engine_id, remote_handshake_port)] = (
                        agent_meta.afd_layer_layout
                    )
                # ### PATCH END: cache negotiated layer layout under peer identity
                self.kv_caches_base_addr[engine_id][remote_handshake_port] = (
                    agent_meta.kv_caches_base_addr
                )
                self.remote_te_host[engine_id][remote_handshake_port] = transfer_host
                self.remote_te_port[engine_id][remote_handshake_port] = transfer_port
        except Exception:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
            raise
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug(
                    "Returned socket to pool for %s:%d",
                    remote_host,
                    remote_handshake_port,
                )

    # ### PATCH START: install the negotiated-layout protocol and transfer methods
    worker.register_kv_caches = register_kv_caches
    receiver._get_remote_metadata = _get_remote_metadata
    receiver._transfer_kv_cache_all_groups = _transfer_kv_cache_all_groups
    setattr(worker, _PATCH_ATTR, True)
    # ### PATCH END: install the negotiated-layout protocol and transfer methods
    return True
