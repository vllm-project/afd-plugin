# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD metadata objects shared by runtime classes and model wrappers."""

from __future__ import annotations

import copy
import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch
from vllm.forward_context import DPMetadata

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

    from afd_plugin.connectors.base import AFDConnectorBase


@dataclass(slots=True)
class AFDDPMetadata:
    """Serializable DPMetadata-compatible payload for AFD control traffic."""

    num_tokens_across_dp_cpu: torch.Tensor
    max_tokens_across_dp_cpu: torch.Tensor | None = None
    local_sizes: list[int] | None = None

    def __post_init__(self) -> None:
        self.num_tokens_across_dp_cpu = torch.as_tensor(
            self.num_tokens_across_dp_cpu,
            dtype=torch.int32,
            device="cpu",
        )
        if self.max_tokens_across_dp_cpu is None:
            self.max_tokens_across_dp_cpu = self.num_tokens_across_dp_cpu.max()
        else:
            self.max_tokens_across_dp_cpu = torch.as_tensor(
                self.max_tokens_across_dp_cpu,
                dtype=torch.int32,
                device="cpu",
            )

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int) -> Generator[list[int]]:
        self.local_sizes = (
            (
                (self.num_tokens_across_dp_cpu + sequence_parallel_size - 1)
                // sequence_parallel_size
            )
            .repeat_interleave(sequence_parallel_size)
            .tolist()
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        return self.local_sizes

    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens = self.num_tokens_across_dp_cpu
        num_tokens_across_sp_cpu = (num_tokens - 1 + sp_size) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(
            sp_size,
        )
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)

    @contextmanager
    def chunked_sizes(
        self,
        sequence_parallel_size: int,
        max_chunk_size_per_rank: int,
        chunk_idx: int,
    ) -> Generator[list[int]]:
        sp_tokens = (
            (
                (self.num_tokens_across_dp_cpu + sequence_parallel_size - 1)
                // sequence_parallel_size
            )
            .repeat_interleave(sequence_parallel_size)
            .tolist()
        )
        self.local_sizes = [
            max(
                1,
                min(
                    max_chunk_size_per_rank,
                    size - max_chunk_size_per_rank * chunk_idx,
                ),
            )
            for size in sp_tokens
        ]
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None


AFDSingleDPMetadata = AFDDPMetadata


@dataclass(slots=True)
class AFDControlPayload:
    """Structured DP metadata control-plane payload.

    This object is the envelope used by connector control-plane methods:
    ``update_state_from_dp_metadata()``, ``send_dp_metadata_list()``, and
    ``recv_dp_metadata_list()``.

    Attributes:
        dp_metadata_list: Mapping from stage or ubatch index to DP token
            metadata. Values are normalized to ``AFDDPMetadata`` in
            ``__post_init__`` so connector implementations can rely on a stable
            plugin-owned representation instead of vLLM-internal DP metadata
            objects.
        is_graph_capturing: Whether the Attention side is currently executing
            a graph-capture path. Connectors use this to decide how to prepare
            local buffers or backend state before data-path operations.
        is_warmup: Whether the payload belongs to a warmup step. This is
            separate from graph capture because warmup may prepare state without
            representing a real serving step.
    """

    dp_metadata_list: dict[int, AFDDPMetadata]
    is_graph_capturing: bool
    is_warmup: bool

    def __post_init__(self) -> None:
        self.dp_metadata_list = {
            # stage_idx: _ensure_afd_dp_metadata(dp_metadata)
            stage_idx: AFDDPMetadata(
                num_tokens_across_dp_cpu=dp_metadata.num_tokens_across_dp_cpu,
            )
            if isinstance(dp_metadata, DPMetadata)
            else dp_metadata
            for stage_idx, dp_metadata in self.dp_metadata_list.items()
        }


@dataclass(slots=True)
class AFDTransferState:
    """Base class for backend-specific connector transfer state.

    Backends subclass ``AFDTransferState`` to carry per-transfer state between
    the receive and send phases of one AFD exchange. The concrete instance is
    held by ``AFDTransferContext.states``. Transfers without additional state
    leave the slot as ``None``.
    """


class AFDExpertRoutingSpec(NamedTuple):
    """Model-owned contract for receiving optional experts routing tensors."""

    router_logits_width: int
    router_logits_dtype: torch.dtype


@dataclass(slots=True)
class AFDTransferMetadata:
    """Communication metadata for one AFD Attention/FFN exchange.

    ``AFDTransferMetadata`` describes the logical tensor transfer for a single
    layer/stage pair. It is intentionally backend-neutral; backend-specific
    state lives on ``AFDTransferState`` instead.

    Attributes:
        layer_idx: Model layer index associated with this transfer.
        stage_idx: Stage or ubatch index associated with this transfer.
        seq_lens: Per-peer or per-split token lengths. The sum of this list is
            the expected leading dimension for tensors validated against this
            metadata. For one-to-one transfers this is usually a single-item
            list; for fan-in/fan-out paths it can describe split sizes.
    """

    layer_idx: int
    stage_idx: int
    seq_lens: list[int]

    def __post_init__(self) -> None:
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("all sequence lengths must be positive")

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @classmethod
    def create_attention_metadata(
        cls,
        *,
        layer_idx: int,
        stage_idx: int,
        seq_len: int,
    ) -> AFDTransferMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=[seq_len],
        )

    @classmethod
    def create_ffn_metadata(
        cls,
        *,
        layer_idx: int,
        stage_idx: int,
        seq_lens: list[int],
    ) -> AFDTransferMetadata:
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=list(seq_lens),
        )

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        return len(tensor_shape) > 0 and tensor_shape[0] == self.total_tokens


@dataclass(slots=True)
class AFDTransferContext:
    """Transfer metadata plus transfer state for one AFD exchange.

    ``AFDTransferContext`` is the object connectors pass through their data
    path. It binds the backend-neutral ``AFDTransferMetadata`` describing the
    layer/stage/token layout to an optional ``AFDTransferState`` subclass
    carrying the backend's per-transfer payloads.

    ``states`` is a pluggable slot for backend-specific connectors. Ordinary
    transfers leave it ``None``. Consumers read it only on paths that populate
    it. Keeping the default ``None`` — rather than a
    ``field(default_factory=AFDTransferState)`` — also keeps the generated
    ``__init__`` traceable by ``torch.compile``/Dynamo, which fails on the
    factory call and is exercised where attention forwards build the context
    under compilation.

    Attributes:
        metadata: Metadata describing the layer/stage and token layout.
        states: Optional ``AFDTransferState`` subclass. ``None`` when the
            transfer has no additional state.
    """

    metadata: AFDTransferMetadata
    states: AFDTransferState | None = None


@dataclass(slots=True)
class AFDA2FTransferPayload:
    """Unified Attention-to-FFN receive payload.

    ``recv_attn_output()`` returns this object on the FFN side. It carries the
    received hidden states alongside the ``AFDTransferContext`` describing the
    transfer. FFN runners pass the context through the FFN compute and back
    into ``send_ffn_output()``.

    Attributes:
        hidden_states: Hidden-state tensor received from the Attention side.
        context: Transfer context describing the received transfer, including
            transfer metadata and backend-produced transfer state.
        router_logits: Optional Attention-side routing tensor.
    """

    hidden_states: torch.Tensor
    context: AFDTransferContext
    router_logits: torch.Tensor | None = None


@dataclass(slots=True)
class AFDF2ATransferPayload:
    """Unified FFN -> Attention payload for separated routed/shared outputs."""

    routed_output: torch.Tensor
    shared_output: torch.Tensor | None = None


@dataclass(slots=True)
class AFDForwardContextMetadata:
    """Forward-context metadata visible to plugin-owned model wrappers."""

    tokens_start_loc: list[int]
    requests_start_loc: list[int]
    stage_idx: int
    connector: AFDConnectorBase
    tokens_lens: list[int]
    num_stages: int
    transaction_id: str | None = None
    tokens_unpadded_lens: list[int] = field(default_factory=list)

    def clone(self) -> AFDForwardContextMetadata:
        cloned = copy.copy(self)
        cloned.tokens_start_loc = list(self.tokens_start_loc)
        cloned.requests_start_loc = list(self.requests_start_loc)
        cloned.tokens_lens = list(self.tokens_lens)
        cloned.tokens_unpadded_lens = list(self.tokens_unpadded_lens)
        return cloned


def _to_int(value: object) -> int:
    item = getattr(value, "item", None)
    return int(item() if callable(item) else value)


def encode_control_payload(payload: AFDControlPayload) -> bytes:
    """Serialize an ``AFDControlPayload`` to a compact JSON byte string.

    Only ``num_tokens_across_dp_cpu`` / ``max_tokens_across_dp_cpu`` per stage
    and the graph-capturing / warmup flags are carried; these are the fields the
    FFN-side connectors read back after decode. Serializing to a plugin-owned
    minimal schema keeps the wire format decoupled from vLLM-internal DP
    metadata objects.
    """
    metadata_payload: dict[str, dict[str, int | list[int]]] = {}
    for stage_idx, dp_metadata in payload.dp_metadata_list.items():
        metadata_payload[str(int(stage_idx))] = {
            "num_tokens_across_dp_cpu": dp_metadata.num_tokens_across_dp_cpu.tolist(),
            "max_tokens_across_dp_cpu": _to_int(dp_metadata.max_tokens_across_dp_cpu),
        }

    wire_payload = {
        "dp_metadata_list": metadata_payload,
        "is_graph_capturing": bool(payload.is_graph_capturing),
        "is_warmup": bool(payload.is_warmup),
    }
    return json.dumps(wire_payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8",
    )


def decode_control_payload(payload_bytes: bytes) -> AFDControlPayload:
    """Rebuild an ``AFDControlPayload`` from ``encode_control_payload``."""
    payload = json.loads(payload_bytes.decode("utf-8"))
    dp_metadata_list = {
        int(stage_idx): AFDDPMetadata(
            num_tokens_across_dp_cpu=torch.tensor(
                [int(value) for value in metadata["num_tokens_across_dp_cpu"]],
                dtype=torch.int32,
                device="cpu",
            ),
            max_tokens_across_dp_cpu=torch.tensor(
                int(metadata["max_tokens_across_dp_cpu"]),
                dtype=torch.int32,
                device="cpu",
            ),
        )
        for stage_idx, metadata in payload["dp_metadata_list"].items()
    }
    return AFDControlPayload(
        dp_metadata_list=dp_metadata_list,
        is_graph_capturing=bool(payload.get("is_graph_capturing", False)),
        is_warmup=bool(payload.get("is_warmup", False)),
    )


def send_control_payload(
    payload: AFDControlPayload,
    *,
    dst: int | list[int] | tuple[int, ...],
    group: ProcessGroup,
    device: torch.device,
) -> None:
    """Encode and send a DP-metadata payload to ``dst`` over ``group``.

    The payload is sent as two messages: a ``long`` size tensor followed by the
    ``uint8`` object tensor, both staged on ``device`` so the caller controls
    whether the backend transport sees CUDA/NPU or CPU tensors.
    """
    object_bytes = encode_control_payload(payload)
    object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)
    object_tensor = object_tensor_cpu.to(device)
    size_tensor = torch.tensor(
        [object_tensor.numel()],
        dtype=torch.long,
        device=device,
    )
    dsts = [dst] if isinstance(dst, int) else dst
    for d in dsts:
        torch.distributed.send(size_tensor, dst=d, group=group)
        torch.distributed.send(object_tensor, dst=d, group=group)


def recv_control_payload(
    *,
    src: int,
    group: ProcessGroup,
    device: torch.device,
) -> AFDControlPayload:
    """Receive and decode a DP-metadata payload from ``src`` over ``group``."""
    size_tensor = torch.empty(1, dtype=torch.long, device=device)
    rank_size = torch.distributed.recv(size_tensor, src=src, group=group)
    object_tensor = torch.empty(
        int(size_tensor.item()),
        dtype=torch.uint8,
        device=device,
    )
    rank_object = torch.distributed.recv(object_tensor, src=src, group=group)
    if rank_object != rank_size:
        raise RuntimeError("received AFD DP metadata fragments from different ranks")
    return decode_control_payload(object_tensor.cpu().numpy().tobytes())


__all__ = [
    "AFDExpertRoutingSpec",
    "AFDTransferState",
    "AFDTransferContext",
    "AFDTransferMetadata",
    "AFDDPMetadata",
    "AFDControlPayload",
    "AFDF2ATransferPayload",
    "AFDForwardContextMetadata",
    "AFDA2FTransferPayload",
    "AFDSingleDPMetadata",
    "decode_control_payload",
    "encode_control_payload",
    "recv_control_payload",
    "send_control_payload",
]
