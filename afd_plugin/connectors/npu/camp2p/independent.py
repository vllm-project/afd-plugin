# SPDX-License-Identifier: Apache-2.0
"""Host-scheduled independent Attention clients over existing A2E/E2A kernels.

Each client has a separate two-rank HCCL domain. Its PrefixStore carries only
small control messages; hidden states and routing tensors still use A2E/E2A.
All clients join at startup. This is scheduling independence, not hot joining
or recovery after a peer/device failure.
"""

from __future__ import annotations

from datetime import timedelta
from threading import Event
from typing import cast

import torch
from torch.distributed import Store
from vllm.logger import init_logger

from afd_plugin.connectors.base import AFDControlPlaneClosedError
from afd_plugin.connectors.metadata import (
    AFDControlPayload,
    decode_control_payload,
    encode_control_payload,
)
from afd_plugin.connectors.npu.camp2p.connector import (
    CAMP2pAFDConnector,
    CAMP2pAFDControlPlane,
    CAMP2PExtraInfo,
)
from afd_plugin.distributed import (
    create_hccl_process_group_options,
    init_afd_process_group,
)

logger = init_logger(__name__)
IDLE_POLL_INTERVAL_SECONDS = 0.001


class IndependentMetadataMailbox:
    """Per-client FIFO control messages with fair selection among ready clients.

    A single producer and a single consumer own each mailbox. A's device stream
    orders each A2E after the preceding E2A. Numbered messages also preserve
    metadata when the host schedules ahead of device completion. F consumes one
    whole model step before calling recv again, and deletes consumed keys.
    """

    def __init__(self, stores: dict[int, Store]) -> None:
        self.stores = stores
        self._peers = tuple(sorted(stores))
        self._send_sequence = dict.fromkeys(stores, 0)
        self._recv_sequence = dict.fromkeys(stores, 0)
        self._next_peer = 0
        self._closed = Event()

    @staticmethod
    def validate_payload(payload: AFDControlPayload) -> None:
        if payload.is_graph_capturing or payload.is_graph_replaying:
            raise ValueError("independent Attention requires eager execution")
        if set(payload.dp_metadata_list) != {0}:
            raise ValueError("independent Attention requires a single stage 0")
        counts = payload.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist()
        if len(counts) != 1 or counts[0] <= 0:
            raise ValueError("independent Attention requires one positive token count")

    def send(self, peer: int, payload: AFDControlPayload) -> None:
        self.validate_payload(payload)
        if self._closed.is_set():
            raise AFDControlPlaneClosedError("independent metadata mailbox is closed")
        sequence = self._send_sequence[peer]
        self.stores[peer].set(f"request/{sequence}", encode_control_payload(payload))
        self._send_sequence[peer] += 1

    def recv(self) -> tuple[int, AFDControlPayload]:
        while not self._closed.is_set():
            for offset in range(len(self._peers)):
                index = (self._next_peer + offset) % len(self._peers)
                peer = self._peers[index]
                store = self.stores[peer]
                key = f"request/{self._recv_sequence[peer]}"
                if not store.check([key]):
                    continue
                payload = decode_control_payload(store.get(key))
                self.validate_payload(payload)
                store.delete_key(key)
                self._recv_sequence[peer] += 1
                self._next_peer = (index + 1) % len(self._peers)
                return peer, payload
            self._closed.wait(IDLE_POLL_INTERVAL_SECONDS)
        raise AFDControlPlaneClosedError("independent metadata mailbox is closed")

    def close(self) -> None:
        self._closed.set()


class IndependentCAMP2pControlPlane(CAMP2pAFDControlPlane):
    """Select one pair for the existing single-threaded FFN runner."""

    def __init__(self, connector: CAMP2pAFDConnector) -> None:
        super().__init__(connector)
        self.mailbox: IndependentMetadataMailbox | None = None
        self.channel_names: dict[int, str] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        connector = self.connector
        config = connector.vllm_config
        afd = connector.afd_config
        extra = cast(CAMP2PExtraInfo, connector.extra_info)
        if afd.num_ffn_ranks != 1:
            raise ValueError("independent Attention requires one F rank")
        if afd.port + afd.num_attention_ranks > 65535:
            raise ValueError("independent Attention channel ports exceed 65535")
        if afd.is_attention_server:
            if (
                extra.attention_id is None
                or not 0 <= extra.attention_id < afd.num_attention_ranks
            ):
                raise ValueError(
                    "attention_id must identify one configured Attention client"
                )
        elif extra.attention_id is not None:
            raise ValueError("FFN must not set attention_id")
        parallel = config.parallel_config
        for field, size in (
            ("data_parallel_size", parallel.data_parallel_size),
            ("tensor_parallel_size", parallel.tensor_parallel_size),
            ("pipeline_parallel_size", parallel.pipeline_parallel_size),
            ("prefill_context_parallel_size", parallel.prefill_context_parallel_size),
            ("decode_context_parallel_size", parallel.decode_context_parallel_size),
        ):
            if size != 1:
                raise ValueError(f"independent Attention requires {field}=1")
        if not config.model_config.enforce_eager:
            raise ValueError("independent Attention requires eager execution")
        if config.parallel_config.use_ubatching or config.parallel_config.enable_dbo:
            raise ValueError("independent Attention does not support DBO or ubatching")
        if config.speculative_config is not None:
            raise ValueError("independent Attention does not support speculation")

    def initialize_channels(self) -> None:
        connector = self.connector
        afd = connector.afd_config
        extra = cast(CAMP2PExtraInfo, connector.extra_info)
        peers = (
            range(afd.num_attention_ranks)
            if afd.is_ffn_server
            else (cast(int, extra.attention_id),)
        )
        stores: dict[int, Store] = {}
        timeout = timedelta(seconds=afd.afd_process_group_timeout_s)
        for peer in peers:

            def save_store(store: Store, peer: int = peer) -> None:
                stores[peer] = store

            group = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{afd.host}:{afd.port + peer}",
                world_size=2,
                rank=connector.world_rank,
                group_name=f"afd_independent_{peer}",
                timeout=timeout,
                pg_options=create_hccl_process_group_options(
                    connector.hccl_buffer_size_mb
                ),
                store_callback=save_store,
            )
            connector.afd_pg_list.append(group)
            self.channel_names[peer] = str(
                group._get_backend(torch.device("npu")).get_hccl_comm_name(
                    connector.world_rank
                )
            )
        if afd.is_ffn_server:
            connector.ffn_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{afd.host}:{afd.port + afd.num_attention_ranks}",
                world_size=1,
                rank=0,
                group_name="afd_independent_moe",
                timeout=timeout,
                pg_options=create_hccl_process_group_options(
                    connector.hccl_buffer_size_mb
                ),
            )
            connector.hccl_comm_name1 = str(
                connector.ffn_pg._get_backend(torch.device("npu")).get_hccl_comm_name(0)
            )
        self.mailbox = IndependentMetadataMailbox(stores)
        self._select_channel(next(iter(stores)))
        logger.info(
            "Independent CAMP2p ready: role=%s clients=%s DP=1 pair_world=2",
            afd.role,
            tuple(stores),
        )

    def _select_channel(self, peer: int) -> None:
        connector = self.connector
        connector.hccl_comm_name = self.channel_names[peer]
        connector.hccl_comm_name2 = connector.hccl_comm_name
        connector.hccl_comm_name3 = ""
        connector.hccl_comm_name_list = [connector.hccl_comm_name]

    def send_dp_metadata_list(self, payload: AFDControlPayload) -> None:
        assert self.mailbox is not None
        extra = cast(CAMP2PExtraInfo, self.connector.extra_info)
        assert extra.attention_id is not None
        self.mailbox.send(extra.attention_id, payload)

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        assert self.mailbox is not None
        peer, payload = self.mailbox.recv()
        self._select_channel(peer)
        logger.info(
            "Independent CAMP2p step: attention_id=%d tokens=%s profile=%s",
            peer,
            payload.dp_metadata_list[0].num_tokens_across_dp_cpu.tolist(),
            payload.is_profile,
        )
        return payload

    def close(self) -> None:
        if self.mailbox is not None:
            self.mailbox.close()
