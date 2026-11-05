# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Runtime patches for elastic AFD on an UNMODIFIED vLLM-Ascend tree.

Policy: vllm-ascend stays pristine; the NPU platform/communicator/worker
seams elastic AFD needs are applied at runtime from the plugin.

Upstream source: vllm-ascend branch ``feat/eep-npu-stateless-pg``
(18756f51a, diffed against main ceac1a2bd) — originally carried as
vllm-ascend PR#19961, relocated into the plugin under the single-repo
delivery policy. Ported near-verbatim; divergences are marked.

Patch surface (idempotent, loud anchors, see apply_afd_stateless_pg_patches):

1. ``NPUPlatform.stateless_init_device_torch_dist_pg`` — stateless HCCL PG
   direct construction for vllm elastic-EP standby groups (per-group HCCL
   identity + per-group host-port window: EI0019/EI0020 hygiene).
2. ``ensure_hccl_socket_port_range`` — per-physical-device NPU-adapter
   socket range, hooked ahead of ``NPUWorker._init_device`` (the local
   ws=1 torch world there is the process's first HCCL comm).
3. ``vllm.distributed.stateless_coordinator.CudaCommunicator`` symbol swap
   — standby-group construction hardcodes CudaCommunicator; swap in
   NPUCommunicator (module-attribute swap, mirroring ascend's own patch
   chain technique).
4. ``NPUCommunicator`` — accept the stateless constructor signature
   (global_ranks/global_world_size/tcp_store_group) and route
   send/recv/broadcast/batch_isend_irecv straight to ProcessGroupHCCL:
   stateless groups are registered in torch.distributed by name only, so
   the dist.* wrappers raise "Group is not registered", and manually
   batching async ops trips EI0015/EI0005 (per-op lazy comm creation);
   the native batch API builds one comm for the whole batch.
5. ``init_ascend_model_parallel`` — early-return under a
   StatelessGroupCoordinator world (DP lives outside the local c10d
   world; Ascend aux groups cannot be laid out and are unused: elastic
   requires TP=1 with EPLB off). Rebound on BOTH the defining module and
   ``vllm_ascend.worker.worker``'s from-import binding.
6. ``BlockTable.clear`` — CpuGpuBuffer is a {cpu, gpu} container; the
   stock bare ``fill_`` crashes with AttributeError on elastic rebuild.
7. ``NPUModelRunner`` (V1) ``load_model`` — accept ``load_dummy_weights``
   (elastic standby lane instantiates with dummy weights, real weights
   arrive over the standby groups).
"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from datetime import timedelta

    from torch.distributed import ProcessGroup
    from torch.distributed.distributed_c10d import PrefixStore

_PATCH_ATTR = "_afd_plugin_stateless_pg_patch_state"


def ensure_hccl_socket_port_range() -> None:
    """Reserve a per-process NPU-adapter port range for HCCL rootInfo detection.

    The adapter socket used for rootInfo detection defaults to a single
    host-wide port (16666). An elastic-EP worker hosts several HCCL comms
    at once (its local ws=1 torch world plus multiple stateless groups),
    and concurrent comm creation across processes then fails with
    Communication_Error_Bind_IP_Port (EI0020). Give each physical device
    its own range; rootInfo carries the actual port, so ranks of one comm
    may use different ranges (verified on 910, 4 ranks x distinct ranges).
    An explicit HCCL_NPU_SOCKET_PORT_RANGE (operator override) wins.

    Ray-fenced workers see one device via ASCEND_RT_VISIBLE_DEVICES, so the
    physical id comes from the env var. Multi-device processes (harness
    workers sharing one visible list) must map their current logical device
    to the physical id — divergence from the feat-branch placement (which
    ran after torch.npu.set_device): this hook may run before device init,
    so the mapping is skipped when torch.npu is uninitialized; such harnesses
    must call torch.npu.set_device first or set the env var explicitly.
    """
    if "HCCL_NPU_SOCKET_PORT_RANGE" in os.environ:
        return
    visible = [
        t.strip()
        for t in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if t.strip()
    ]
    try:
        if len(visible) == 1:
            device = int(visible[0])
        elif len(visible) > 1:
            if not torch.npu.is_initialized():
                return
            device = int(visible[torch.npu.current_device()])
        else:
            device = torch.npu.current_device()
    except (ValueError, IndexError):
        device = torch.npu.current_device()
    base = 17000 + device * 20
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = f"{base}-{base + 19}"


def _stateless_init_device_torch_dist_pg(
    cls,
    backend: str,
    prefix_store: PrefixStore,
    group_rank: int,
    group_size: int,
    timeout: timedelta,
) -> ProcessGroup:
    """Create a stateless HCCL process group (NPUPlatform classmethod).

    Used by vLLM elastic EP to create standby groups that live outside
    ``torch.distributed``'s global world state, so that groups can be
    created, switched to, and destroyed during elastic scaling. Mirrors
    the CUDA platform implementation (vllm/platforms/cuda.py) and the
    proven direct-construction pattern from the ascend netloader
    (``vllm_ascend/model_loader/netloader/executor/netloader_pg.py``).
    """
    from datetime import timedelta as _timedelta

    from torch.distributed import ProcessGroup, is_hccl_available
    from torch.distributed.distributed_c10d import Backend
    from torch_npu._C._distributed_c10d import ProcessGroupHCCL

    if not is_hccl_available():
        raise RuntimeError("HCCL is not available")
    if backend != "hccl":
        raise ValueError(
            f"Unsupported stateless distributed backend on Ascend NPU: "
            f"{backend}, expected 'hccl'"
        )

    hccl_backend = Backend("hccl")
    pg: ProcessGroup = ProcessGroup(prefix_store, group_rank, group_size)
    pg._set_default_backend(Backend.backend_type_map[hccl_backend])

    pg_options = ProcessGroupHCCL.Options()
    pg_options.is_high_priority_stream = False
    pg_options._timeout = timeout or _timedelta(minutes=30)
    pg_options.global_ranks_in_group = []
    # Each stateless group rendezvouses on its own TCP port, so the
    # underlying store's host:port is unique per group and identical on
    # every rank. Use it as the HCCL comm identity, like the netloader
    # does; an empty group_id lets sequentially created comms collide on
    # HCCL's identity-scoped resources (e.g. the EI0019 port-60000 bind).
    underlying_store = prefix_store.underlying_store
    pg_options.group_id = f"tcp://{underlying_store.host}:{underlying_store.port}/"
    # Give each stateless group its own HCCL host-socket port window:
    # rootInfo-detection listeners are not always released on group
    # destruction, so a later group can hit EI0019 on the default 60000
    # range. The rendezvous port is unique per group and identical on
    # every rank, and HCCL comm init is lazy (first collective), so the
    # env value seen at init time is this group's. An explicitly
    # configured HCCL_IF_BASE_PORT (operator override) wins; otherwise
    # every group gets a unique window (reusing one value across groups
    # re-collides).
    if not hasattr(cls, "_operator_hccl_if_base_port"):
        cls._operator_hccl_if_base_port = os.environ.get("HCCL_IF_BASE_PORT")
    if cls._operator_hccl_if_base_port is not None:
        os.environ["HCCL_IF_BASE_PORT"] = cls._operator_hccl_if_base_port
    else:
        os.environ["HCCL_IF_BASE_PORT"] = str(
            60000 + (underlying_store.port % 50) * 100
        )
    ensure_hccl_socket_port_range()

    backend_class = ProcessGroupHCCL(prefix_store, group_rank, group_size, pg_options)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(
        torch.device("npu"), ProcessGroup.BackendType.CUSTOM, backend_class
    )
    return pg


def _npu_communicator_init(
    self,
    cpu_group,
    device: torch.device | None = None,
    device_group=None,
    unique_name: str = "",
    global_ranks: list[int] | None = None,
    global_world_size: int | None = None,
    use_all2all: bool = False,
    tcp_store_group=None,
) -> None:
    """NPUCommunicator.__init__ accepting the stateless constructor signature.

    ``StatelessGroupCoordinator`` builds standby DP/EP communicators with
    CudaCommunicator's signature (global_ranks/global_world_size/
    tcp_store_group). ``tcp_store_group`` is accepted for signature
    compatibility and not stored: HCCL P2P goes through the device_group
    directly. Legacy behavior (bare device index) is kept when no device
    is passed by the regular GroupCoordinator path; the stateless path
    passes a proper ``torch.device("npu:N")``.
    """
    from vllm_ascend.distributed.device_communicators.npu_communicator import (
        _NpuAll2AllManager,
    )

    base_cls = type(self).__mro__[1]
    base_cls.__init__(
        self,
        cpu_group,
        device,
        device_group,
        unique_name,
        global_ranks,
        global_world_size,
        use_all2all=use_all2all,
    )
    if device is None:
        self.device = torch.npu.current_device()
    self.ca_comm = None
    # vLLM #53576 reads this CUDA-only communicator during graph capture.
    # Keep the shared coordinator protocol available without enabling the
    # FlashInfer PCIe IPC backend on NPU.
    self.fi_pcie_ipc_ar_comm = None
    self.all2all_manager = _NpuAll2AllManager()


def _hccl_backend(self):
    """Return the ProcessGroupHCCL backend of the device group.

    vLLM's stateless groups are registered in torch.distributed by name
    only (not in ``_world.pg_group_ranks``), so
    ``torch.distributed.send/recv/broadcast/isend/irecv`` would raise
    "Group ... is not registered" — and would also misinterpret
    group-local ranks as global ranks. Calling the backend directly
    bypasses both, mirroring how CudaCommunicator routes P2P and
    broadcast through pynccl.
    """
    if self.device_group is None:
        raise ValueError("No HCCL device group found")
    return self.device_group._get_backend(torch.device("npu"))


def _communicator_send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
    """Send a tensor to the destination group-local rank (blocking)."""
    if dst is None:
        dst = (self.rank_in_group + 1) % self.world_size
    # pg bind signature: send(tensors: Sequence[Tensor], dst, tag)
    self._hccl_backend().send([tensor], dst, 0).wait()


def _communicator_recv(
    self, size: torch.Size, dtype: torch.dtype, src: int | None = None
) -> torch.Tensor:
    """Receive a tensor from the source group-local rank."""
    if src is None:
        src = (self.rank_in_group - 1) % self.world_size
    tensor = torch.empty(size, dtype=dtype, device=self.device)
    self._hccl_backend().recv([tensor], src, 0).wait()
    return tensor


def _communicator_broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from source rank to all ranks."""
    if self.world_size == 1:
        return tensor
    # pg bind overload: broadcast(tensor: Tensor, root) -> Work
    self._hccl_backend().broadcast(tensor, src).wait()
    return tensor


def _communicator_batch_isend_irecv(self, p2p_ops: list):
    """Batch point-to-point ops over the (possibly stateless) HCCL group.

    Delegates to ProcessGroupHCCL's native
    ``batch_isend_irecv(op_types, tensors, remote_ranks)``. Manually
    looping async ``send``/``recv`` calls does NOT work: each op lazily
    creates its own HCCL comm via rootInfo exchange, and with more than
    one op in flight (any bidirectional pattern, or a ring exchange) that
    per-op comm creation fails with EI0015 (rootInfo detect timeout) or
    EI0005 (inconsistent arguments) — verified on 910 with netloader-style
    stateless groups, where single unidirectional send/recv pairs work but
    manual batches never do. The native batch API creates one comm for the
    whole batch — the HCCL equivalent of ncclGroupStart/End batching used
    by CudaCommunicator.batch_isend_irecv.

    Ops carry group-local peer ranks (``op.group_peer``), matching the
    native API's rank semantics; stateless groups are not in
    torch.distributed's ``_world`` registry, so
    torch.distributed.batch_isend_irecv would raise "Group ... is not
    registered" anyway.
    """
    backend = self._hccl_backend()
    op_types: list[str] = []
    tensors: list[torch.Tensor] = []
    remote_ranks: list[int] = []
    for op in p2p_ops:
        if op.op is torch.distributed.isend:
            op_types.append("isend")
            # HCCL reads the send buffer as-is; hand it contiguous memory
            # (no-op for already-contiguous tensors).
            tensors.append(op.tensor.contiguous())
        elif op.op is torch.distributed.irecv:
            op_types.append("irecv")
            # Never copy recv buffers: data must land in op.tensor.
            assert op.tensor.is_contiguous(), "Recv buffer must be contiguous"
            tensors.append(op.tensor)
        else:
            raise ValueError(f"Unsupported p2p op: {op.op}")
        remote_ranks.append(op.group_peer)
    backend.batch_isend_irecv(op_types, tensors, remote_ranks).wait()


def _block_table_clear(self) -> None:
    # CpuGpuBuffer is a {cpu, gpu} container; match vllm core's
    # BlockTable.clear (the bare fill_ call crashes with AttributeError
    # on the elastic worker startup / rebuild path).
    self.block_table.gpu.fill_(0)
    self.block_table.cpu.fill_(0)


def _wrap_init_ascend_model_parallel(original):
    def guarded(*args: Any, **kwargs: Any):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                from vllm.distributed.stateless_coordinator import (
                    StatelessGroupCoordinator,
                )
            except ImportError:
                StatelessGroupCoordinator = None
            if StatelessGroupCoordinator is not None:
                from vllm.distributed.parallel_state import get_world_group

                if isinstance(get_world_group(), StatelessGroupCoordinator):
                    # Elastic EP: the DP dimension lives in stateless groups
                    # outside the local c10d world, so the Ascend-only
                    # auxiliary groups (MC2, fine-grained TP, dynamic EPLB)
                    # cannot be laid out from here. The elastic topology
                    # requires TP=1 with EPLB disabled, which leaves them
                    # unused; core initialize_model_parallel has already
                    # built the stateless TP/PP/DP/EP groups.
                    return
        return original(*args, **kwargs)

    return guarded


def _wrap_v1_load_model(original):
    def load_model(self, load_dummy_weights: bool = False, *args: Any, **kwargs: Any):
        if load_dummy_weights:
            # Elastic EP standby path: instantiate the model with dummy
            # weights; the real weights are transferred over the standby
            # groups afterwards (mirrors gpu_model_runner.load_model).
            self.load_config.load_format = "dummy"
        return original(self, *args, **kwargs)

    return load_model


def _wrap_init_device(original):
    def _init_device(self, *args: Any, **kwargs: Any):
        # Must precede every HCCL comm creation in this process (the local
        # ws=1 torch world inside _init_device is the first one): the
        # default adapter socket port is a single host-wide port and
        # collides across the Ray-scheduled elastic-EP workers.
        ensure_hccl_socket_port_range()
        return original(self, *args, **kwargs)

    return _init_device


def _make_thread_local_store_client(original):
    """Per-thread replacement for vllm's process-wide lru_cache'd store client.

    Re-anchor main@544f182d + UniProcExecutor topology: vllm assumes elastic
    EP workers run in their own processes, so ``get_cached_tcp_store_client``
    caches ONE TCPStore per (host, port) per process. Under AFD NPU (ray
    actor per DP rank, uni executor) the engine-side
    ``coord_store.wait(done_keys)`` (300 s) and the worker-side
    ``coord_store.set(...)`` that would release it share that single client
    object; torch's TCPStore serializes operations on it, so the wait blocks
    the set — a self-deadlock observed live (worker frozen at
    ``coord_store.set("world_0")`` while the engine waited on
    ``/eep_async/prepare_reconfiguration/*``). Thread-local clients are cheap
    and independent; per-thread maxsize mirrors the original lru_cache(1).
    """
    import threading

    tls = threading.local()

    def thread_cached_tcp_store_client(host: str, port: int):
        entry = getattr(tls, "entry", None)
        if entry is None or entry[0] != (host, port):
            from torch.distributed import TCPStore

            entry = (
                (host, port),
                TCPStore(host, port, is_master=False, wait_for_workers=False),
            )
            tls.entry = entry
        return entry[1]

    thread_cached_tcp_store_client._afd_original = original
    return thread_cached_tcp_store_client


def apply_afd_stateless_pg_patches() -> bool:
    """Apply every patch in this module. Idempotent; loud on anchor drift.

    Returns True when all patches are in place. Raises RuntimeError when
    the vLLM / vLLM-Ascend seams this port was anchored to have drifted
    (fail fast beats wrongly routed collectives at elastic-scaling time).
    """
    from vllm_ascend.platform import NPUPlatform

    if getattr(NPUPlatform, _PATCH_ATTR, False):
        return True

    import vllm_ascend.distributed.parallel_state as ascend_parallel_state
    import vllm_ascend.worker.worker as ascend_worker_module
    from vllm.distributed.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase,
    )
    from vllm_ascend.distributed.device_communicators.npu_communicator import (
        NPUCommunicator,
    )
    from vllm_ascend.worker.block_table import BlockTable
    from vllm_ascend.worker.v1.model_runner import NPUModelRunner as NPUModelRunnerV1
    from vllm_ascend.worker.worker import NPUWorker

    # --- anchors: fail loudly when the ported seams no longer match ---
    base_params = inspect.signature(DeviceCommunicatorBase.__init__).parameters
    if "global_ranks" not in base_params:
        raise RuntimeError(
            "AFD stateless-PG patch anchor: DeviceCommunicatorBase.__init__ "
            "lacks global_ranks; vllm generation drifted from the "
            "feat/eep-npu-stateless-pg port (see stateless_pg.py header)."
        )
    try:
        import vllm.distributed.stateless_coordinator as stateless_coordinator
    except ImportError as exc:
        raise RuntimeError(
            "AFD stateless-PG patch anchor: vllm.distributed.stateless_coordinator "
            "is missing; elastic EP requires the vllm elastic generation."
        ) from exc
    if "tcp_store_group" in inspect.signature(NPUCommunicator.__init__).parameters:
        # Ascend mainline absorbed the stateless signature; skip comm patches.
        comm_absorbed = True
    else:
        comm_absorbed = False
    if not hasattr(stateless_coordinator, "CudaCommunicator"):
        raise RuntimeError(
            "AFD stateless-PG patch anchor: stateless_coordinator.CudaCommunicator "
            "symbol gone; upstream fixed the hardcoded communicator — drop the "
            "symbol swap from this module."
        )
    if not hasattr(NPUWorker, "_init_device"):
        raise RuntimeError(
            "AFD stateless-PG patch anchor: NPUWorker._init_device missing."
        )
    if not hasattr(ascend_worker_module, "init_ascend_model_parallel"):
        raise RuntimeError(
            "AFD stateless-PG patch anchor: worker.from-import binding of "
            "init_ascend_model_parallel missing."
        )

    # --- 1/2: platform hook + socket-range helper ---
    NPUPlatform.stateless_init_device_torch_dist_pg = classmethod(
        _stateless_init_device_torch_dist_pg
    )

    # --- 3: standby-group communicator symbol swap ---
    stateless_coordinator.CudaCommunicator = NPUCommunicator

    # --- 3b: all2all manager shim for the new elastic prepare contract ---
    # vllm main@544f's elastic prepare calls all2all_manager.stage_ep_size()
    # (All2AllManagerBase API); ascend's _NpuAll2AllManager is a deliberate
    # no-op placeholder (NPU MoE uses mc2/all_gather, no all2all state to
    # stage), so the no-op shim is the semantically correct NPU behavior.
    from vllm_ascend.distributed.device_communicators.npu_communicator import (
        _NpuAll2AllManager,
    )

    if not hasattr(_NpuAll2AllManager, "stage_ep_size"):

        def _noop_stage_ep_size(self) -> None:
            return None

        _NpuAll2AllManager.stage_ep_size = _noop_stage_ep_size

    # --- 4: communicator stateless signature + direct PGHCCL routing ---
    if not comm_absorbed:
        NPUCommunicator.__init__ = _npu_communicator_init
        NPUCommunicator._hccl_backend = _hccl_backend
        NPUCommunicator.send = _communicator_send
        NPUCommunicator.recv = _communicator_recv
        NPUCommunicator.broadcast = _communicator_broadcast
        NPUCommunicator.batch_isend_irecv = _communicator_batch_isend_irecv

    # --- 5: stateless-world guard for Ascend aux-group layout ---
    guarded = _wrap_init_ascend_model_parallel(
        ascend_parallel_state.init_ascend_model_parallel
    )
    ascend_parallel_state.init_ascend_model_parallel = guarded
    ascend_worker_module.init_ascend_model_parallel = guarded

    # --- 6: BlockTable.clear container fix ---
    BlockTable.clear = _block_table_clear

    # --- 7: V1 runner dummy-weight lane ---
    if (
        "load_dummy_weights"
        not in inspect.signature(NPUModelRunnerV1.load_model).parameters
    ):
        NPUModelRunnerV1.load_model = _wrap_v1_load_model(NPUModelRunnerV1.load_model)

    # --- 2b: socket range ahead of the first in-process HCCL comm ---
    NPUWorker._init_device = _wrap_init_device(NPUWorker._init_device)

    # --- 9: harden torch's c10d exception logger against stateless groups ---
    # When a wrapped collective raises, torch's _exception_logger builds a
    # message dict via dist.get_backend(group) — which itself raises for
    # vllm's stateless groups (unregistered in _world) and REPLACES the
    # original error. Degrade gracefully so real failures stay visible
    # (this bit us in _warm_target_groups over standby dp/ep groups).
    import torch.distributed.c10d_logger as c10d_logger_mod

    if not getattr(c10d_logger_mod._get_msg_dict, "_afd_hardened", False):
        _original_msg_dict = c10d_logger_mod._get_msg_dict

        def _safe_msg_dict(func_name, *args: Any, **kwargs: Any):
            try:
                return _original_msg_dict(func_name, *args, **kwargs)
            except Exception:
                return {"func_name": f"{func_name}"}

        _safe_msg_dict._afd_hardened = True
        c10d_logger_mod._get_msg_dict = _safe_msg_dict

    # --- 8: thread-local TCPStore clients (uni-executor self-deadlock) ---
    import importlib
    import sys as _sys

    import vllm.distributed.utils as dist_utils

    original_client = dist_utils.get_cached_tcp_store_client
    if not hasattr(original_client, "_afd_original"):
        replacement = _make_thread_local_store_client(original_client)
        # Defining module (covers the function-level from-imports in
        # standby_state / elastic_execute, which resolve per call)...
        dist_utils.get_cached_tcp_store_client = replacement
        # ...plus every module-level binding that captured the original.
        for mod_name in (
            "vllm.distributed.elastic_ep.elastic_state",
            "vllm.distributed.elastic_ep.elastic_execute",
            "vllm.distributed.elastic_ep.standby_state",
            "vllm.distributed.elastic_ep.external_elastic_ep",
            "vllm.distributed.parallel_state",
            "vllm.config.parallel",
        ):
            mod = _sys.modules.get(mod_name)
            if mod is None:
                try:
                    mod = importlib.import_module(mod_name)
                except ImportError:
                    continue
            if getattr(mod, "get_cached_tcp_store_client", None) is original_client:
                mod.get_cached_tcp_store_client = replacement

    setattr(NPUPlatform, _PATCH_ATTR, True)
    return True


__all__ = [
    "apply_afd_stateless_pg_patches",
    "ensure_hccl_socket_port_range",
]
