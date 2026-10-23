# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone V4 parameter lifetime using the pinned native routed loader.

Metadata construction imports no tensor/runtime libraries. Loading creates
F-local native groups only for parameter construction, sharding and packing;
the completed tensor views survive their teardown. No CausalLM, EngineCore,
ModelRunner, forward, or per-request native collective is constructed here.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import ExitStack
from datetime import timedelta
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.protocol import SessionSpec
from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.v4_weights import V4RoutedWeightOwner
    from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights

PACKED_WEIGHTS_PER_BYTE = 2
PROJECTIONS_PER_EXPERT = 3
SOURCE_PARAMETERS_PER_PROJECTION = 4
ENCODED_SCALE_BYTES = 8
FLOAT_PARAMETER_BYTES = 4
GATE_UP_PROJECTIONS = 2
LOADING_GROUP_TIMEOUT_SECONDS = 180
LOADING_MAX_MODEL_LENGTH = 1024
LOADING_MAX_BATCHED_TOKENS = 32
LOADING_MAX_SEQUENCES = 2


def validate_native_expert_map(
    spec: SessionSpec, ffn_rank: int, expert_map: list[int]
) -> frozenset[int]:
    """Bind native placement to the current service's contiguous wire layout."""
    if len(expert_map) != spec.num_experts:
        raise ValueError("native V4 expert map has the wrong global expert count")
    owned = frozenset(expert for expert, local in enumerate(expert_map) if local >= 0)
    if sorted(expert_map[expert] for expert in owned) != list(
        range(spec.local_experts)
    ):
        raise ValueError("native V4 expert map is not a local bijection")
    first = ffn_rank * spec.local_experts
    end = first + spec.local_experts
    if owned != frozenset(range(first, end)) or expert_map[first:end] != list(
        range(spec.local_experts)
    ):
        raise ValueError("native V4 placement differs from contiguous service routing")
    return owned


class V4FFNModel:
    """Own only routed W4A8 parameters; executors borrow the cached views."""

    def __init__(
        self,
        checkpoint: V4CheckpointDescription,
        spec: SessionSpec,
        ffn_rank: int,
    ) -> None:
        expected = checkpoint.session_spec(
            attention_ranks=spec.attention_ranks,
            attention_tp=spec.attention_tp,
            ffn_ranks=spec.ffn_ranks,
            capacity=spec.capacity,
            slots=spec.slots,
            layer_indices=spec.layer_indices,
        )
        if expected != spec:
            raise ValueError("V4 FFN session does not match the W4A8 checkpoint")
        if not 0 <= ffn_rank < spec.ffn_ranks:
            raise ValueError("V4 FFN rank is outside the F-only loading world")
        self.checkpoint, self.spec, self.ffn_rank = checkpoint, spec, ffn_rank
        self._owner: V4RoutedWeightOwner | None = None
        self._views: tuple[W4A8LayerWeights, ...] | None = None
        self._loading = False
        self._loading_groups_owned = False
        self._loaded = False

        # Native ModelSlim retains offsets in addition to the six kernel views.
        # Before load this is an aligned-geometry estimate; afterward we record
        # the actual distinct parameter storage sizes, including retained views.
        matrices = (
            PROJECTIONS_PER_EXPERT
            * spec.hidden_size
            * spec.intermediate_size
            // PACKED_WEIGHTS_PER_BYTE
        )
        channels = GATE_UP_PROJECTIONS * spec.intermediate_size + spec.hidden_size
        auxiliary = channels * (
            ENCODED_SCALE_BYTES + FLOAT_PARAMETER_BYTES + FLOAT_PARAMETER_BYTES
        )
        self._parameter_bytes = (
            len(spec.layer_indices) * spec.local_experts * (matrices + auxiliary)
        )
        self._sources: dict[int, list[tuple[int, str, str]]] = {
            layer: [] for layer in spec.layer_indices
        }
        for name, shard in checkpoint.routed_weight_map.items():
            fields = name.split(".", 5)
            layer = int(fields[1])
            if layer in self._sources:
                self._sources[layer].append((int(fields[4]), name, shard))

    @property
    def parameter_bytes(self) -> int:
        return self._parameter_bytes

    @property
    def native_view(self) -> tuple[W4A8LayerWeights, ...]:
        if not self._loaded or self._views is None:
            raise RuntimeError("V4 FFN weights have not been loaded")
        return self._views

    def _validate_launcher(self) -> None:
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                "V4 FFN loading requires an external F-only torchrun launcher"
            ) from error
        if world_size != self.spec.ffn_ranks or rank != self.ffn_rank:
            raise ValueError("V4 FFN loading requires the matching F-only world/rank")

    def load(
        self,
        device: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if self._loading or self._owner is not None or self._loaded:
            raise RuntimeError("V4 FFN weights are already loading or loaded")
        self._validate_launcher()
        self._loading = True
        try:
            self._load_native(device, progress)
        except BaseException:
            self._loading = False
            self.close()
            raise
        finally:
            self._loading = False

    def _load_native(
        self, device: int, progress: Callable[[int, int], None] | None
    ) -> None:
        # Prepare the native vendor search path before the first device init
        # or imports of the parameter owner/engine configuration. This public
        # bootstrap only updates OPP paths; extension loading remains below.
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()

        # Runtime imports stay at the isolated worker-loading boundary.
        # Inherited CANN Python/library search paths are left intact.
        import torch
        import torch_npu  # noqa: F401
        from safetensors import safe_open
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.distributed.parallel_state import (
            get_ep_group,
            model_parallel_is_initialized,
        )
        from vllm.engine.arg_utils import EngineArgs
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
        from vllm_ascend.utils import enable_custom_op, register_ascend_customop

        from afd_plugin.ffn_faas.v4_weights import V4RoutedWeightOwner

        if torch.distributed.is_initialized() or model_parallel_is_initialized():
            raise RuntimeError("V4 FFN loading cannot borrow existing native groups")
        torch.npu.set_device(device)
        # Pinned Ascend model_runner_v1.py:277 installs this worker policy at
        # import time. The independent service bypasses that runner; preserving
        # internal format is required by native W4A8 NZ packing and dtype views.
        torch.npu.config.allow_internal_format = True
        # Native Worker._init_device bootstraps OPP paths and loads the Ascend
        # extension here, before any weight processing can initialize GE.
        if not enable_custom_op():
            raise RuntimeError("native V4 W4A8 custom operators are unavailable")
        target = torch.device(f"npu:{device}")
        spec = self.spec
        args = EngineArgs(
            model=str(self.checkpoint.root),
            trust_remote_code=True,
            dtype="bfloat16",
            quantization="ascend",
            enforce_eager=True,
            tensor_parallel_size=1,
            data_parallel_size=spec.ffn_ranks,
            data_parallel_rank=self.ffn_rank,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            max_model_len=LOADING_MAX_MODEL_LENGTH,
            max_num_batched_tokens=LOADING_MAX_BATCHED_TOKENS,
            max_num_seqs=LOADING_MAX_SEQUENCES,
            additional_config={
                "enable_prefill_mc2": False,
                "enable_force_eplb": False,
                "enable_cpu_binding": False,
                "ascend_fusion_config": {"fusion_ops_gmmswigluquant": True},
            },
        )
        config = args.create_engine_config()
        config.parallel_config.distributed_executor_backend = "external_launcher"
        total = len(spec.layer_indices) * spec.local_experts
        completed = 0
        with set_current_vllm_config(config):
            # Native Worker registers Ascend routed/runner implementations
            # before config initialization and any parameter construction.
            register_ascend_customop(config)
            init_ascend_config(config)
            # Retain cleanup ownership before initialization can partially fail.
            self._loading_groups_owned = True
            init_distributed_environment(
                world_size=spec.ffn_ranks,
                rank=self.ffn_rank,
                local_rank=device,
                distributed_init_method="env://",
                backend="hccl",
                timeout=timedelta(seconds=LOADING_GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                backend="hccl",
            )
            init_ascend_model_parallel(config.parallel_config)
            group = get_ep_group()
            if (
                group.world_size != spec.ffn_ranks
                or group.rank_in_group != self.ffn_rank
            ):
                raise ValueError("native V4 expert group differs from the F world")
            with torch.device(target):
                owner = V4RoutedWeightOwner(self.checkpoint, config)
            self._owner = owner
            for layer_index in spec.layer_indices:
                with torch.device(target):
                    layer = owner.create_layer(layer_index)
                routed = layer.mlp.experts.routed_experts
                if routed.local_num_experts != spec.local_experts:
                    raise ValueError(
                        "native V4 local expert count differs from session"
                    )
                mapping = routed.expert_map
                if mapping is None:
                    raise ValueError("native V4 loader requires an explicit expert map")
                entries = mapping.cpu().tolist()
                owned = validate_native_expert_map(spec, self.ffn_rank, entries)
                selected = [
                    (name, shard)
                    for expert, name, shard in self._sources[layer_index]
                    if expert in owned
                ]
                required = (
                    spec.local_experts
                    * PROJECTIONS_PER_EXPERT
                    * SOURCE_PARAMETERS_PER_PROJECTION
                )
                if len(selected) != required:
                    raise ValueError(
                        "V4 layer is missing local native source parameters"
                    )
                with ExitStack() as stack:
                    readers = {
                        shard: stack.enter_context(
                            safe_open(
                                str(self.checkpoint.root / shard),
                                framework="pt",
                                device="cpu",
                            )
                        )
                        for shard in sorted({shard for _, shard in selected})
                    }
                    owner.load_weights(
                        (name, readers[shard].get_tensor(name))
                        for name, shard in selected
                    )
                    owner.finish_layer(layer_index)
                    # Startup-only fence: source mappings and packing temporaries
                    # can retire before the next layer, and progress means packed.
                    torch.npu.synchronize(device)
                completed += len(owned)
                if progress is not None:
                    progress(completed, total)
            views = owner.weight_views()
            if tuple(view.layer_idx for view in views) != spec.layer_indices:
                raise ValueError("native V4 views do not match the session layers")
            self._views = views
            storage_sizes: dict[tuple[str, int], int] = {}
            for parameter in owner.parameters():
                storage = parameter.untyped_storage()
                key = (str(parameter.device), storage.data_ptr())
                storage_sizes[key] = storage.nbytes()
            self._parameter_bytes = sum(storage_sizes.values())

        # Cached views contain only tensors and static layer semantics. No
        # quantizer/loader call occurs after this point, so native F groups need
        # not survive into the independent A/F service's communication lifetime.
        self._close_loading_groups()
        self._loaded = True

    def _close_loading_groups(self) -> None:
        if not self._loading_groups_owned:
            return
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()
        self._loading_groups_owned = False

    def close(self) -> None:
        """Called after executor completion; retain weights on group teardown error."""
        if self._loading:
            raise RuntimeError("cannot close V4 FFN weights during loading")
        self._close_loading_groups()
        self._views = None
        self._owner = None
        self._loaded = False
