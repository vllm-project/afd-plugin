# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DSV4 DSpark auxiliary-state extension of the native Mooncake PD carrier.

The carrier keeps native target allocation and transfer-engine registration.
Its existing sender socket also serves target page descriptors and a bounded
auxiliary tail. D owns projection and draft SWA writes, including connector-only
rounds. This initial path requires eager execution, PP/CP1, D TP1 and no prefix
cache. It does not transfer draft KV or execute a drafter on P.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import msgspec
import torch
import zmq
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_sliding_window,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_hybrid_connector as native,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    KV_CACHE_BUFFER_ALIGNMENT,
    as_kv_cache_tensors,
)
from vllm_ascend.worker.dspark_utils import get_dspark_aux_layer_ids

from afd_plugin.distributed.kv_transfer.dspark_context import (
    DSparkContextDescriptor,
    DSparkKVPage,
    build_draft_context_slots,
    build_target_page_reads,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
        AscendDSparkSpeculator,
    )

GET_DSPARK_STATE = b"afd_dspark_state_v1"
ACK_DSPARK_STATE = b"afd_dspark_ack_v1"
DONE_DSPARK_TRANSFER = b"afd_dspark_done_v1"
STATE_POLL_INTERVAL_S = 0.01
CONTROL_POLL_INTERVAL_MS = 100
THREAD_STOP_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class _Allocation:
    generation: str
    source_generation: str
    prompt_tokens: int
    local_blocks: tuple[tuple[int, ...], ...]
    remote_blocks: tuple[tuple[int, ...], ...]


class _Metadata(KVConnectorMetadata):
    def __init__(
        self,
        transfer: native.MooncakeConnectorMetadata,
        source_generations: dict[str, str],
        allocations: dict[str, _Allocation],
    ) -> None:
        self.transfer = transfer
        self.source_generations = source_generations
        self.allocations = allocations


@dataclass
class _Source:
    generation: str
    slot: int
    end: int = 0
    descriptor: DSparkContextDescriptor | None = None
    event: torch.npu.Event | None = None
    leased: str | None = None


@dataclass
class _Destination:
    allocation: _Allocation
    slot: int | None = None
    descriptor: DSparkContextDescriptor | None = None
    received: bool = False
    target_done: bool = False
    cancelled: bool = False
    failure: str | None = None
    event: torch.npu.Event | None = None


class _AuxPool:
    """Fixed registered storage with explicit transfer ownership."""

    # Mooncake's compiled SDK has no Python type declarations. This owner is
    # cleared after region retirement; native methods are used directly.
    engine: Any

    def __init__(self, worker: _Worker, device: torch.device) -> None:
        shape = (worker.max_requests, worker.window, worker.feature_width)
        size = 2 * worker.max_requests * worker.window * worker.feature_width
        self.raw = torch.empty(
            size + KV_CACHE_BUFFER_ALIGNMENT, dtype=torch.uint8, device=device
        )
        offset = (-self.raw.data_ptr()) % KV_CACHE_BUFFER_ALIGNMENT
        self.tensor = self.raw[offset : offset + size].view(torch.bfloat16).view(shape)
        self.engine = worker.engine
        self.condition = threading.Condition()
        self.available = list(reversed(range(worker.max_requests)))
        self.closed = False
        self.registered = False
        self.timeout = float(native.get_transfer_timeout_value())
        result = self.engine.register_memory(self.tensor.data_ptr(), size)
        if result != 0:
            raise RuntimeError("DSpark auxiliary memory registration failed")
        self.registered = True

    def acquire(self) -> int:
        with self.condition:
            if not self.condition.wait_for(
                lambda: bool(self.available) or self.closed, timeout=self.timeout
            ):
                raise RuntimeError("DSpark auxiliary staging capacity timed out")
            if self.closed:
                raise RuntimeError("DSpark auxiliary staging is shutting down")
            return self.available.pop()

    def release(self, slot: int) -> None:
        with self.condition:
            if slot in self.available:
                raise RuntimeError("DSpark auxiliary slot was released twice")
            self.available.append(slot)
            self.condition.notify_all()

    def stop_admission(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def close(self) -> None:
        if self.registered:
            if self.engine.unregister_memory(self.tensor.data_ptr()) != 0:
                raise RuntimeError("DSpark auxiliary memory unregistration failed")
            self.registered = False


class _Scheduler(native.MooncakeConnectorScheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source_generations: dict[str, str] = {}
        self._pending_sources: dict[str, str] = {}
        self._pending_allocations: dict[str, _Allocation] = {}

    # Native source: Ascend bd42eea0 mooncake_hybrid_connector.py.
    # Keep native request admission; capture identity and complete tables before
    # native transfer compacts null entries. Remove when extra tensors and
    # draft initialization are exposed by the upstream connector.
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        params = request.kv_transfer_params or {}
        # ### PATCH START: auxiliary allocation association
        if params.get("do_remote_decode"):
            generation = self._source_generations.setdefault(
                request.request_id, uuid.uuid4().hex
            )
            self._pending_sources[request.request_id] = generation
        if params.get("do_remote_prefill") and num_external_tokens > 0:
            source_generation = params.get("afd_dspark_generation")
            remote_blocks = params.get("afd_dspark_block_ids")
            prompt_tokens = params.get("afd_dspark_prompt_tokens")
            if (
                not isinstance(source_generation, str)
                or not source_generation
                or not isinstance(remote_blocks, (tuple, list))
                or type(prompt_tokens) is not int
                or prompt_tokens != num_external_tokens
            ):
                raise ValueError(
                    "DSpark PD requires matching P auxiliary handoff metadata"
                )
            self._pending_allocations[request.request_id] = _Allocation(
                generation=uuid.uuid4().hex,
                source_generation=source_generation,
                prompt_tokens=prompt_tokens,
                local_blocks=tuple(tuple(ids) for ids in blocks.get_block_ids()),
                remote_blocks=tuple(tuple(ids) for ids in remote_blocks),
            )
        # ### PATCH END: auxiliary allocation association
        super().update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        metadata = _Metadata(
            cast(
                native.MooncakeConnectorMetadata,
                super().build_connector_meta(scheduler_output),
            ),
            self._pending_sources,
            self._pending_allocations,
        )
        self._pending_sources = {}
        self._pending_allocations = {}
        return metadata

    # Retain native delayed target-block release. Complete block tables preserve
    # absolute indices independently of P/D grouping and physical page strides.
    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        delay, params = super().request_finished_all_groups(request, block_ids)
        # ### PATCH START: auxiliary handoff identity and source tables
        generation = self._source_generations.pop(request.request_id, None)
        if params is not None:
            if generation is None:
                raise RuntimeError("P request finished without a DSpark generation")
            params.update(
                afd_dspark_generation=generation,
                afd_dspark_prompt_tokens=request.num_prompt_tokens,
                afd_dspark_block_ids=block_ids,
            )
        # ### PATCH END: auxiliary handoff identity and source tables
        return delay, params


def _target_pages(
    config: KVCacheConfig,
    caches: dict[str, Any],
    draft_names: set[str],
) -> tuple[DSparkKVPage, ...]:
    """Describe target payloads without changing the actual shared allocation."""
    groups = {
        name: (group_id, group)
        for group_id, group in enumerate(config.kv_cache_groups)
        for name in group.layer_names
    }
    pages = []
    for descriptor in config.kv_cache_tensors:
        for layer_index, name in enumerate(
            native.get_kv_cache_tensor_layers(descriptor)
        ):
            if name in draft_names:
                continue
            group_id, group = groups[name]
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if native.is_circular_kv_cache_spec(spec):
                raise ValueError(
                    "DSpark PD target circular KV requires a separate page contract"
                )
            views = as_kv_cache_tensors(caches[name])
            storage = views[0].untyped_storage()
            if any(
                view.untyped_storage().data_ptr() != storage.data_ptr()
                for view in views
            ):
                raise ValueError("DSpark PD requires one backing per target KV layer")
            base = (
                (storage.data_ptr() + KV_CACHE_BUFFER_ALIGNMENT - 1)
                // KV_CACHE_BUFFER_ALIGNMENT
                * KV_CACHE_BUFFER_ALIGNMENT
            )
            page_address = (
                base + descriptor.offset + layer_index * descriptor.layer_stride
            )
            if base + descriptor.size > storage.data_ptr() + storage.nbytes() or any(
                not page_address
                <= view.data_ptr()
                < page_address + spec.page_size_bytes
                for view in views
            ):
                raise ValueError(
                    "DSpark PD target view does not match configured KV placement"
                )
            pages.append(
                DSparkKVPage(
                    layer_name=name,
                    group_id=group_id,
                    base_address=page_address,
                    block_stride=descriptor.block_stride,
                    page_size=spec.page_size_bytes,
                    block_size=spec.block_size,
                    num_blocks=config.num_blocks,
                    format_tag=f"{type(spec).__name__}:{spec.dtype}:{spec.num_kv_heads}:{spec.head_size}",
                )
            )
    return tuple(pages)


class _SendingThread(native.KVCacheSendingThread):
    def __init__(self, *args: Any, owner: _Worker, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.owner = owner
        self.stop_event = threading.Event()

    # Copy the pinned native socket dispatch because no custom-message callback
    # exists. Keep GET_META/DONE semantics; add STATE/ACK on that same socket.
    # Remove this override when upstream provides an auxiliary-message hook.
    def run_busy_loop(self, sock: zmq.Socket) -> None:
        torch.npu.set_device(self.owner.pool.tensor.device)
        encoder = msgspec.msgpack.Encoder()
        encoded_metadata = encoder.encode(self.metadata)
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while not self.stop_event.is_set():
            # ### PATCH START: bounded receiver exit for the existing socket
            if not sock.poll(CONTROL_POLL_INTERVAL_MS):
                continue
            # ### PATCH END: bounded receiver exit for the existing socket
            frames = sock.recv_multipart()
            payload = [frame for frame in frames[1:] if frame]
            if len(payload) != 1:
                continue
            identity = frames[0]
            try:
                message = decoder.decode(payload[0])
                if message[0] == native.GET_META_MSG:
                    response = encoded_metadata
                # ### PATCH START: source auxiliary lease and acknowledgement
                elif message[0] == GET_DSPARK_STATE:
                    response = encoder.encode(
                        self.owner.source_state(message[1], message[2], message[3])
                    )
                elif message[0] == ACK_DSPARK_STATE:
                    self.owner.acknowledge_source(message[1], message[2], message[3])
                    response = b"ACK"
                elif message[0] == DONE_DSPARK_TRANSFER:
                    if self.owner.complete_source(message[1], message[2], message[3]):
                        self.task_tracker.update_done_task_count(message[1])
                    response = b"ACK"
                # ### PATCH END: source auxiliary lease and acknowledgement
                elif message[0] == native.DONE_RECVING_MSG:
                    request_id, remote_port_send_num = message[1], message[2]
                    if request_id in self.owner._source_generations:
                        raise ValueError(
                            "DSpark source completion requires allocation generations"
                        )
                    if remote_port_send_num:
                        self.port_send_num[request_id] = (
                            self.port_send_num.get(request_id, 0) + 1
                        )
                        handshake_port = (
                            self.side_channel_port
                            + self.pp_rank * self.tp_size
                            + self.tp_rank
                        )
                        if (
                            self.port_send_num[request_id]
                            >= remote_port_send_num[handshake_port]["num"]
                        ):
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    response = b"ACK"
                else:
                    raise ValueError("Unknown Mooncake control message")
            except Exception as error:
                response = encoder.encode({"status": "error", "error": str(error)})
            sock.send_multipart((identity, b"", response))


class _ReceivingThread(native.KVCacheRecvingThread):
    def __init__(self, *args: Any, owner: _Worker, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.owner = owner

    def run(self) -> None:
        self.ready_event.set()
        while True:
            request = self.request_queue.get()
            if request is None:
                self.request_queue.task_done()
                return
            self._submit_request(request)

    # Native _handle_request swallows failures yet reports receive completion.
    # Preserve its finalizers and surface AFD receive failure to the scheduler.
    # Pinned source: Ascend bd42eea0 mooncake_hybrid_connector.py:676.
    # Remove when the native carrier reports explicit failed receive outcomes.
    def _handle_request(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        try:
            # ### PATCH START: target pages and auxiliary states share one read
            self.owner.receive_request(self, req_meta)
            # ### PATCH END: target pages and auxiliary states share one read
        except Exception as error:
            self.owner.fail_request(request_id, str(error))
            native.logger.exception("DSpark PD receive failed for %s", request_id)
        finally:
            # ### PATCH START: reject late completion of reused request IDs
            try:
                self.owner.notify_source_finished(self, req_meta)
            except Exception as error:
                self.owner.fail_request(request_id, str(error))
                native.logger.exception(
                    "DSpark source completion failed for %s", request_id
                )
            # ### PATCH END: reject late completion of reused request IDs
            if self._mark_request_task_done(request_id, req_meta["all_task_done"]):
                self.task_tracker.update_done_task_count(request_id)
                with self.proc_not_transfer_request_lock:
                    self.proc_not_transfer_request.pop(remote_request_id, None)
            self.request_queue.task_done()


@contextmanager
def _thread_factories(owner: _Worker) -> Iterator[None]:
    """Scope the native startup-only constructor seam; never patch online work.

    Ascend bd42eea0 register_kv_caches() imports constructors by module binding
    and exposes no factories. Delegating avoids copying its registration code.
    Worker cache initialization is serialized. Remove when factories exist.
    """
    sender, receiver = native.KVCacheSendingThread, native.KVCacheRecvingThread

    def create_sender(*args: Any, **kwargs: Any) -> _SendingThread:
        return _SendingThread(*args, owner=owner, **kwargs)

    def create_receiver(*args: Any, **kwargs: Any) -> _ReceivingThread:
        return _ReceivingThread(*args, owner=owner, **kwargs)

    # ### PATCH START: plugin-local PD thread construction
    native.KVCacheSendingThread = create_sender
    native.KVCacheRecvingThread = create_receiver
    try:
        yield
    finally:
        native.KVCacheSendingThread = sender
        native.KVCacheRecvingThread = receiver
    # ### PATCH END: plugin-local PD thread construction


class _Worker(native.MooncakeConnectorWorker):
    # The compiled SDK is untyped and its owner is cleared after shutdown.
    engine: Any
    kv_recv_thread: native.KVCacheRecvingThread | None
    kv_send_thread: native.KVCacheSendingThread | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = self.vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.aux_ids = get_dspark_aux_layer_ids(self.vllm_config)
        self.hidden_size = self.vllm_config.model_config.get_hidden_size()
        self.feature_width = len(self.aux_ids) * self.hidden_size
        self.window = extra["dspark_context_window"]
        self.max_requests = extra.get(
            "dspark_context_max_requests",
            self.vllm_config.scheduler_config.max_num_seqs,
        )
        if not self.aux_ids or type(self.window) is not int or self.window <= 0:
            raise ValueError(
                "DSpark PD requires auxiliary capture IDs and a positive context window"
            )
        if type(self.max_requests) is not int or self.max_requests <= 0:
            raise ValueError("DSpark PD staging capacity must be positive")
        if (
            self.kv_role == "kv_producer"
            and self.max_requests < self.vllm_config.scheduler_config.max_num_seqs
        ):
            raise ValueError(
                "DSpark P staging capacity must cover max_num_seqs so a full "
                "prefill batch can finish before waiting for D acknowledgements"
            )
        self.draft: AscendDSparkSpeculator | None = None
        self.pages: tuple[DSparkKVPage, ...] = ()
        self._pool: _AuxPool | None = None
        self._lock = threading.RLock()
        self._sources: dict[str, _Source] = {}
        self._source_generations: dict[str, str] = {}
        self._retiring_sources: set[str] = set()
        self._done_sources: set[str] = set()
        self._pending_send_completion: set[str] = set()
        self._acknowledged: OrderedDict[tuple[str, str, str], None] = OrderedDict()
        self._completed_sources: OrderedDict[tuple[str, str, str], None] = OrderedDict()
        self._destinations: dict[str, _Destination] = {}
        self._load_errors: set[int] = set()
        self._failed_recving: set[str] = set()
        self._draining = False
        self._shutdown_complete = False
        # DSV4 target caches are replicated over TP; one selected P rank owns
        # this request's native transfer, including when P TP4 serves D TP1.
        self.tp_num_need_pulls = 1

    def _get_tp_num_need_pulls(self, prefill_tp_size: int) -> int:
        return 1

    @property
    def pool(self) -> _AuxPool:
        if self._pool is None:
            raise RuntimeError("DSpark auxiliary storage is not registered")
        return self._pool

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        draft_names = (
            set(self.draft.model.get_draft_kv_cache_layer_names())
            if self.draft is not None
            else set()
        )
        if self.kv_role == "kv_consumer" and not draft_names:
            raise RuntimeError(
                "Bind the loaded DSpark model before PD cache registration"
            )
        required_windows = []
        for group in self.kv_cache_config.kv_cache_groups:
            for name in group.layer_names:
                if name not in draft_names:
                    continue
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                required = get_kv_cache_spec_sliding_window(spec)
                if required is None:
                    raise ValueError("DSpark PD requires a finite draft SWA window")
                required_windows.append(required)
        if required_windows and self.window < max(required_windows):
            raise ValueError(
                "DSpark auxiliary tail is shorter than the loaded draft SWA window"
            )
        layout = self.vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_compact:
            raise ValueError("DSpark PD requires block-compact target cache pages")
        self.pages = _target_pages(self.kv_cache_config, kv_caches, draft_names)
        device = as_kv_cache_tensors(next(iter(kv_caches.values())))[0].device
        self._pool = _AuxPool(self, device)
        with _thread_factories(self):
            super().register_kv_caches(kv_caches)

    @torch.inference_mode()
    def capture_prefill(self, state: ExecuteModelState) -> None:
        if self.kv_role != "kv_producer":
            return
        batch = state.input_batch
        if state.aux_hidden_states is None or len(state.aux_hidden_states) != len(
            self.aux_ids
        ):
            raise RuntimeError(
                "P target did not return the configured DSpark auxiliary states"
            )
        features = torch.cat(state.aux_hidden_states, dim=-1)
        if (
            features.dtype != torch.bfloat16
            or features.ndim != 2
            or features.shape[1] != self.feature_width
        ):
            raise ValueError(
                "P auxiliary states disagree with the DSpark BF16 capture schema"
            )
        for request_index, request_id in enumerate(batch.req_ids[: batch.num_reqs]):
            generation = self._source_generations.get(request_id)
            if generation is None or self.tp_rank not in self._prefill_get_remote_rank(
                request_id
            ):
                continue
            prompt_tokens = int(batch.prefill_len_np[request_index])
            offset = int(batch.num_computed_tokens_np[request_index])
            count = min(
                int(batch.num_scheduled_tokens[request_index]), prompt_tokens - offset
            )
            if count <= 0:
                continue
            with self._lock:
                source = self._sources.get(request_id)
            if source is None:
                source = _Source(generation, self.pool.acquire())
                with self._lock:
                    self._sources[request_id] = source
            if (
                offset == 0
                and source.end > 0
                and source.generation == generation
                and source.descriptor is None
                and source.leased is None
            ):
                # Native recompute may restart an unfinished chunked prefill.
                # No address was published; model-stream ordering protects the
                # old copy while the replacement tail reuses this same slot.
                source.end = 0
            if source.generation != generation or offset != source.end:
                raise RuntimeError(
                    "P DSpark chunk order or allocation generation changed"
                )
            row = int(batch.query_start_loc_np[request_index])
            end = offset + count
            start = max(0, end - self.window)
            old_start = max(0, source.end - self.window)
            retained = max(0, source.end - start)
            previous = self.pool.tensor[
                source.slot, start - old_start : start - old_start + retained
            ].clone()
            incoming = features[row + max(0, start - offset) : row + count]
            tail = torch.cat((previous, incoming), dim=0)
            self.pool.tensor[source.slot, : end - start].copy_(tail)
            event = torch.npu.Event()
            event.record()
            with self._lock:
                source.end = end
                source.event = event
                if end == prompt_tokens:
                    source.descriptor = DSparkContextDescriptor(
                        request_id,
                        generation,
                        prompt_tokens,
                        start,
                        end,
                        self.aux_ids,
                        self.hidden_size,
                    )

    def source_state(
        self, request_id: str, generation: str, destination_generation: str
    ) -> dict[str, Any]:
        with self._lock:
            if self._draining:
                raise RuntimeError("P auxiliary carrier is draining")
            source = self._sources.get(request_id)
            if source is None or source.generation != generation:
                raise ValueError("Stale DSpark source allocation")
            if (
                source.descriptor is None
                or source.event is None
                or not source.event.query()
            ):
                return {"status": "pending"}
            if (
                not isinstance(destination_generation, str)
                or not destination_generation
            ):
                raise ValueError("DSpark reads require a D allocation generation")
            if source.leased is not None and source.leased != destination_generation:
                raise ValueError("P auxiliary state is leased to another D allocation")
            source.leased = destination_generation
            return {
                "status": "ready",
                "descriptor": source.descriptor.to_dict(),
                "address": self.pool.tensor[source.slot].data_ptr(),
                "te_rpc_port": self.te_rpc_port,
                "pages": [asdict(page) for page in self.pages],
            }

    def acknowledge_source(
        self, request_id: str, generation: str, destination_generation: str
    ) -> None:
        key = (request_id, generation, destination_generation)
        with self._lock:
            if key in self._acknowledged:
                return
            source = self._sources.get(request_id)
            if (
                source is None
                or source.generation != generation
                or source.leased != destination_generation
            ):
                raise ValueError("Stale DSpark source acknowledgement")
            self.pool.release(source.slot)
            del self._sources[request_id]
            self._retiring_sources.discard(request_id)
            self._done_sources.discard(request_id)
            self._acknowledged[key] = None
            while len(self._acknowledged) > 4 * self.max_requests:
                self._acknowledged.popitem(last=False)

    def complete_source(
        self, request_id: str, generation: str, destination_generation: str
    ) -> bool:
        """Gate native target-block completion by the original allocation."""
        key = (request_id, generation, destination_generation)
        with self._lock:
            if (
                key in self._completed_sources
                or self._source_generations.get(request_id) != generation
            ):
                return False
            source = self._sources.get(request_id)
            if source is not None:
                if (
                    source.leased is not None
                    and source.leased != destination_generation
                ):
                    raise ValueError(
                        "DSpark completion belongs to another D allocation"
                    )
                if source.event is not None and not source.event.query():
                    raise RuntimeError("P auxiliary writes are still pending")
                # A failed receive before STATE can also finish this allocation;
                # no source address has been leased in that case.
                source.leased = destination_generation
                self.acknowledge_source(request_id, generation, destination_generation)
            elif key not in self._acknowledged:
                raise ValueError("DSpark completion has no matching source lease")
            self._completed_sources[key] = None
            while len(self._completed_sources) > 4 * self.max_requests:
                self._completed_sources.popitem(last=False)
            return True

    def notify_source_finished(
        self, thread: _ReceivingThread, request: dict[str, Any]
    ) -> None:
        with self._lock:
            allocation = self._destinations[request["request_id"]].allocation
        host, port = request["remote_host"], request["remote_handshake_port"]
        socket = thread._get_remote_socket(host, port)
        try:
            message = (
                DONE_DSPARK_TRANSFER,
                request["remote_request_id"],
                allocation.source_generation,
                allocation.generation,
            )
            native.ensure_zmq_send(
                socket, thread.encoder.encode(message), f"{host}:{port}"
            )
            if native.ensure_zmq_recv(socket, f"{host}:{port}") != b"ACK":
                raise RuntimeError("P generation-qualified completion failed")
        finally:
            thread._return_remote_socket(socket, host, port)

    def fail_request(self, request_id: str, reason: str) -> None:
        with self._lock:
            self._destinations[request_id].failure = reason

    def receive_request(
        self, thread: _ReceivingThread, request: dict[str, Any]
    ) -> None:
        request_id = request["request_id"]
        with self._lock:
            destination = self._destinations[request_id]
        slot = self.pool.acquire()
        destination.slot = slot
        host, port = request["remote_host"], request["remote_handshake_port"]
        source_id = request["remote_request_id"]
        generation = destination.allocation.source_generation
        socket = thread._get_remote_socket(host, port)
        leased = False
        try:
            deadline = time.monotonic() + self.pool.timeout
            while True:
                native.ensure_zmq_send(
                    socket,
                    thread.encoder.encode(
                        (
                            GET_DSPARK_STATE,
                            source_id,
                            generation,
                            destination.allocation.generation,
                        )
                    ),
                    f"{host}:{port}",
                )
                response = msgspec.msgpack.decode(
                    native.ensure_zmq_recv(socket, f"{host}:{port}")
                )
                if response["status"] != "pending":
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("P auxiliary tail readiness timed out")
                time.sleep(STATE_POLL_INTERVAL_S)
            if response["status"] != "ready":
                raise RuntimeError(f"P auxiliary state request failed: {response}")
            leased = True
            descriptor = DSparkContextDescriptor.from_dict(response["descriptor"])
            if (
                descriptor.request_id != source_id
                or descriptor.generation != generation
                or descriptor.prompt_tokens != destination.allocation.prompt_tokens
                or descriptor.aux_layer_ids != self.aux_ids
                or descriptor.hidden_size != self.hidden_size
                or descriptor.window_end != descriptor.prompt_tokens
                or descriptor.window_start
                != max(0, descriptor.prompt_tokens - self.window)
            ):
                raise ValueError(
                    "P/D DSpark context identity, schema or window mismatch"
                )
            pages = tuple(DSparkKVPage(**page) for page in response["pages"])
            local, remote, lengths = build_target_page_reads(
                self.pages,
                pages,
                destination.allocation.local_blocks,
                destination.allocation.remote_blocks,
                descriptor.prompt_tokens,
            )
            address = response["address"]
            if type(address) is not int or address <= 0:
                raise ValueError("Invalid P auxiliary address")
            local.append(self.pool.tensor[slot].data_ptr())
            remote.append(address)
            lengths.append(descriptor.num_tokens * descriptor.feature_width * 2)
            session = f"{host}:{response['te_rpc_port']}"
            if (
                self.engine.batch_transfer_sync_read(session, local, remote, lengths)
                != 0
            ):
                raise RuntimeError("DSpark target/auxiliary transfer failed")
            with self._lock:
                destination.descriptor = descriptor
                destination.received = True
        finally:
            if leased:
                native.ensure_zmq_send(
                    socket,
                    thread.encoder.encode(
                        (
                            ACK_DSPARK_STATE,
                            source_id,
                            generation,
                            destination.allocation.generation,
                        )
                    ),
                    f"{host}:{port}",
                )
                if native.ensure_zmq_recv(socket, f"{host}:{port}") != b"ACK":
                    raise RuntimeError("P auxiliary acknowledgement failed")
            thread._return_remote_socket(socket, host, port)

    @torch.inference_mode()
    def advance(self, finished_req_ids: set[str], done_recving: set[str]) -> set[str]:
        assert self.draft is not None
        ready = set()
        with self._lock:
            for request_id in finished_req_ids:
                if request_id in self._destinations:
                    self._destinations[request_id].cancelled = True
            for request_id in done_recving:
                self._destinations[request_id].target_done = True
            destinations = list(self._destinations.items())
        for request_id, state in destinations:
            if not state.target_done:
                continue
            if state.failure is None and not state.cancelled and state.event is None:
                if not state.received or state.descriptor is None or state.slot is None:
                    raise RuntimeError(
                        "Native completion preceded DSpark auxiliary receipt"
                    )
                try:
                    groups, layer_groups, sizes = (
                        self.draft.get_draft_context_group_layout()
                    )
                    blocks = {
                        group: state.allocation.local_blocks[group] for group in groups
                    }
                    descriptor = state.descriptor
                    positions = range(descriptor.window_start, descriptor.window_end)
                    slots = build_draft_context_slots(
                        positions, blocks, sizes, layer_groups
                    )
                    device = self.pool.tensor.device
                    with set_current_vllm_config(self.draft.attn_vllm_config):
                        features = self.pool.tensor[state.slot, : descriptor.num_tokens]
                        context = self.draft.model.combine_hidden_states(features)
                        self.draft.model.precompute_and_store_context_kv(
                            context,
                            torch.arange(
                                descriptor.window_start,
                                descriptor.window_end,
                                device=device,
                            ),
                            [
                                torch.tensor(slot, dtype=torch.int32, device=device)
                                for slot in slots
                            ],
                        )
                    state.event = torch.npu.Event()
                    state.event.record()
                except Exception as error:
                    # Enqueued partial writes retain the allocation and staging
                    # until their stream event completes, even on failure.
                    state.failure = str(error)
                    state.event = torch.npu.Event()
                    state.event.record()
            if state.event is not None and not state.event.query():
                continue
            if state.failure is not None:
                self._failed_recving.add(request_id)
                self._load_errors.update(
                    block
                    for blocks in state.allocation.local_blocks
                    for block in blocks
                    if block
                )
            if state.slot is not None:
                self.pool.release(state.slot)
            ready.add(request_id)
            with self._lock:
                del self._destinations[request_id]
        return ready

    def retire_sources(self, finished: set[str], done_sending: set[str]) -> None:
        with self._lock:
            self._retiring_sources.update(finished | done_sending)
            self._done_sources.update(done_sending)
            for request_id in tuple(self._retiring_sources):
                source = self._sources.get(request_id)
                if source is None:
                    self._source_generations.pop(request_id, None)
                    self._retiring_sources.discard(request_id)
                    self._done_sources.discard(request_id)
                    continue
                if source.leased:
                    continue
                if (
                    request_id not in self._done_sources
                    and source.descriptor is not None
                ):
                    continue
                if source.event is not None and not source.event.query():
                    continue
                self.pool.release(source.slot)
                del self._sources[request_id]
                self._source_generations.pop(request_id, None)
                self._retiring_sources.discard(request_id)
                self._done_sources.discard(request_id)

    def finish_sending(self, finished: set[str], sending: set[str]) -> set[str]:
        """A native send timeout cannot free target pages during a leased read."""
        with self._lock:
            self._pending_send_completion.update(sending)
            protected = {
                request_id
                for request_id, source in self._sources.items()
                if source.leased is not None
                or (source.event is not None and not source.event.query())
            }
            releasable = self._pending_send_completion - protected
            self.retire_sources(finished, self._pending_send_completion)
            self._pending_send_completion.difference_update(releasable)
            return releasable

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown_complete:
                return
            self._draining = True
        if self._pool is None:
            return
        self.pool.stop_admission()
        receiver, sender = self.kv_recv_thread, self.kv_send_thread
        if isinstance(receiver, _ReceivingThread):
            receiver.request_queue.put(None)
            receiver.join(THREAD_STOP_TIMEOUT_S)
            if receiver.is_alive():
                raise RuntimeError("DSpark receive thread is pending; retain resources")
            receiver.executor.shutdown(wait=True)
        if isinstance(sender, _SendingThread):
            with self._lock:
                if any(source.leased for source in self._sources.values()):
                    raise RuntimeError(
                        "DSpark source reads remain leased; retain resources"
                    )
            sender.stop_event.set()
            sender.join(THREAD_STOP_TIMEOUT_S)
            if sender.is_alive():
                raise RuntimeError("DSpark sender thread is pending; retain resources")
        torch.npu.current_stream(self.pool.tensor.device).synchronize()
        # Native runner shutdown frees target tensors next. Reuse the native
        # deregistration owner to disconnect peers and retire their registered
        # target pages before releasing that storage. The auxiliary region is
        # registered separately and must be retired separately as well.
        if callable(getattr(self.engine, "disconnect_all_peers", None)):
            native.global_te.unregister_buffer()
        else:
            # Mooncake 0.3.13 exposes unregister_memory but not the disconnect
            # method required by the pinned native helper. All our reads and
            # source leases are drained above; stop admission before retiring
            # the same native-owned region list with its supported API.
            # Remove this branch once the qualified engine exports that helper.
            with native.global_te.register_buffer_lock:
                if native.global_te.is_register_buffer:
                    for pointer, _ in native.global_te.registered_buffers:
                        if self.engine.unregister_memory(pointer) != 0:
                            raise RuntimeError(
                                "DSpark target memory unregistration failed; "
                                "retain resources"
                            )
                    native.global_te.is_register_buffer = False
        pool = self.pool
        pool.close()
        # Native MRV2 retains its KV adapter after worker shutdown. That keeps
        # this worker, its stopped threads and the SDK alive until C++ process
        # finalization. Retire every local SDK owner while the NPU runtime is
        # still live, after all reads and registered regions have been retired.
        # The qualified carrier owns the process's single Mooncake engine.
        if isinstance(receiver, _ReceivingThread):
            receiver.engine = None
        with native.global_te.transfer_engine_lock:
            if native.global_te.transfer_engine is self.engine:
                native.global_te.transfer_engine = None
        self.engine = None
        pool.engine = None
        self.draft = None
        self.kv_recv_thread = self.kv_send_thread = None
        self._pool = None
        self._shutdown_complete = True
        native.logger.info("AFD DSpark PD reads drained and registered memory released")


class AFDDSparkMooncakeConnector(native.MooncakeConnector):
    """Native target PD plus target-only P auxiliary export for DSpark on D."""

    # Pinned source: Ascend bd42eea0 MooncakeConnector.__init__.
    # Native construction has no worker/scheduler factories. Copy that small
    # constructor to select subclasses; native registration and request methods
    # stay inherited. Remove once upstream supports these factories.
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or (
                vllm_config.kv_transfer_config.is_kv_consumer
                and parallel.tensor_parallel_size != 1
            )
            or vllm_config.cache_config.enable_prefix_caching
            or not vllm_config.model_config.enforce_eager
            or vllm_config.model_config.dtype != torch.bfloat16
            or vllm_config.model_config.hf_text_config.model_type != "deepseek_v4"
        ):
            raise ValueError(
                "DSpark PD requires BF16 DSV4 eager, PP/CP1, D TP1 "
                "and prefix caching disabled"
            )
        assert kv_cache_config is not None
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata: _Metadata | None = None
        # ### PATCH START: use AFD PD subclasses on native lifecycle
        self.connector_scheduler = (
            _Scheduler(vllm_config, str(self.engine_id), kv_cache_config)
            if role == KVConnectorRole.SCHEDULER
            else None
        )
        self.connector_worker = (
            _Worker(vllm_config, str(self.engine_id), kv_cache_config)
            if role == KVConnectorRole.WORKER
            else None
        )
        # ### PATCH END: use AFD PD subclasses on native lifecycle

    def bind_draft_model(self, speculator: AscendDSparkSpeculator) -> None:
        assert self.connector_worker is not None
        self.connector_worker.draft = speculator

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(connector_metadata, _Metadata)
        self._connector_metadata = connector_metadata
        assert self.connector_worker is not None
        self.connector_worker._source_generations.update(
            connector_metadata.source_generations
        )

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert self._connector_metadata is not None
        worker = self.connector_worker
        with worker._lock:
            if any(
                request_id in worker._destinations
                for request_id in self._connector_metadata.allocations
            ):
                # Validate the whole batch before creating any state. Even an
                # identical duplicate must not reset a slot/event or requeue
                # its native read. Retain the active allocation on rejection.
                raise RuntimeError("DSpark allocation already has an active receive")
            for request_id, allocation in self._connector_metadata.allocations.items():
                worker._destinations[request_id] = _Destination(allocation)
        worker.start_load_kv(self._connector_metadata.transfer)

    def capture_prefill(self, state: ExecuteModelState) -> None:
        assert self.connector_worker is not None
        self.connector_worker.capture_prefill(state)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        worker = self.connector_worker
        sending, recving = worker.get_finished()
        if worker.kv_role == "kv_producer":
            return worker.finish_sending(finished_req_ids, sending), set()
        return sending, worker.advance(finished_req_ids, recving)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        sending, recving = self.get_finished(finished_req_ids)
        assert self.connector_worker is not None
        failed = self.connector_worker._failed_recving
        self.connector_worker._failed_recving = set()
        return KVConnectorTransferResults(
            finished_sending=sending, finished_recving=recving, failed_recving=failed
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        errors = self.connector_worker._load_errors
        self.connector_worker._load_errors = set()
        return errors

    def shutdown(self) -> None:
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
