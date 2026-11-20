# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 eager Mega backend for DeepSeek-V4-Flash."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from afd_plugin.connectors.base import AFDConnectorBase, ConnectorExtraInfo
from afd_plugin.connectors.npu.mega_layers import (
    agree_layer_selection,
    pack_layer_weights,
)
from afd_plugin.connectors.npu.mega_work import POLL_SECONDS, ExpertWorkQueue
from afd_plugin.distributed.afd_process_group import (
    DefaultProcessGroupSwitcher,
    init_afd_process_group,
)
from afd_plugin.envs import mega_layered_gmm_enabled

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.config import AFDConfig
    from afd_plugin.connectors.metadata import AFDTransferContext
    from afd_plugin.connectors.npu.mega_layers import LayerWeights

FLASH_GEOMETRY = (4096, 2048, 256, 6)
STARTUP_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True)
class MegaExtraInfo(ConnectorExtraInfo):
    attn_ranks_per_dp: int | None = None

    def to_mapping(self):
        if self.attn_ranks_per_dp is None:
            return {}
        return {"attn_ranks_per_dp": self.attn_ranks_per_dp}


@dataclass
class MegaStage:
    group: Any
    buffer: Any
    work: ExpertWorkQueue
    sequence: int = 0
    ticket: Any = None
    stream: Any = None
    completion: Any = None


def validate_mega_config(config: VllmConfig, afd: AFDConfig) -> None:
    hf = config.model_config.hf_config
    if (
        get_ascend_device_type() != AscendDeviceType.A5
        or hf.model_type != "deepseek_v4"
        or (
            hf.hidden_size,
            hf.moe_intermediate_size,
            hf.n_routed_experts,
            hf.num_experts_per_tok,
        )
        != FLASH_GEOMETRY
        or hf.expert_dtype != "fp4"
        or config.model_config.dtype != torch.bfloat16
    ):
        raise ValueError(
            "Mega requires A5 and native FP4 DeepSeek-V4-Flash with BF16 input"
        )
    if config.additional_config.get("enable_fused_mc2", 0):
        raise ValueError(
            "Mega split requires enable_fused_mc2=0 for packed layer weights"
        )
    attn, ffn = afd.num_attention_ranks, afd.num_ffn_ranks
    world = attn + ffn
    experts = hf.n_routed_experts
    # UMDK SplitLayout / arch35 tiling bound the union group and count table.
    if not (
        attn > 0
        and ffn > 0
        and world <= 768
        and ffn <= experts <= 2048
        and experts % ffn == 0
        and experts // ffn <= 1024
        and world * (experts // ffn) <= 16384
    ):
        raise ValueError(
            f"Invalid Mega split geometry: A={attn}, F={ffn}, E={experts}"
        )
    p = config.parallel_config
    if (
        p.data_parallel_size * p.tensor_parallel_size
        != (attn if afd.role == "attention" else ffn)
        or (
            p.pipeline_parallel_size,
            p.prefill_context_parallel_size,
            p.decode_context_parallel_size,
        )
        != (1, 1, 1)
        or (
            afd.role == "ffn"
            and (not p.enable_expert_parallel or p.use_sequence_parallel_moe)
        )
        or p.expert_placement_strategy != "linear"
        or p.enable_eplb
        or p.enable_elastic_ep
        or p.eplb_config.num_redundant_experts
        or (
            afd.role == "attention"
            and config.scheduler_config.max_num_batched_tokens % p.tensor_parallel_size
        )
    ):
        raise ValueError(
            "Mega requires role DP * TP equal to its rank count, PP/CP=1, "
            "TP-aligned Attention token capacity and static FFN EP without SP"
        )
    if (p.enable_dbo or p.use_ubatching) and (
        afd.role != "attention" or p.num_ubatches != 2
    ):
        raise ValueError(
            "Mega DBO requires two Attention ubatches and an unbatched FFN"
        )
    if (
        not config.model_config.enforce_eager
        or not config.use_v2_model_runner
        or config.speculative_config is not None
        or not afd.compute_gate_on_attention
        or not afd.async_dp
    ):
        raise ValueError(
            "Mega requires eager ModelRunnerV2, async DP, Attention routing"
        )


class CAMAsyncMegaAFDConnector(AFDConnectorBase):
    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> ConnectorExtraInfo:
        raw = raw or {}
        if set(raw) - {"attn_ranks_per_dp"}:
            raise ValueError("Unknown Mega connector_extra_config field")
        width = raw.get("attn_ranks_per_dp")
        if "attn_ranks_per_dp" in raw and (type(width) is not int or width < 1):
            raise ValueError("attn_ranks_per_dp must be a positive integer")
        return MegaExtraInfo(width)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.attn_size = afd_config.num_attention_ranks
        self.ffn_size = afd_config.num_ffn_ranks
        self.world_rank = (
            role_rank if afd_config.role == "attention" else self.attn_size + role_rank
        )
        self.stages: list[MegaStage] = []
        self.next_stage = 0
        self.weights: dict[int, LayerWeights] = {}
        self.layered_weights: LayerWeights | None = None
        self.layer_slots: dict[int, int] = {}
        self.layered_gmm_enabled = (
            afd_config.role == "ffn" and mega_layered_gmm_enabled()
        )

    @property
    def is_initialized(self) -> bool:
        return bool(self.stages)

    def prepare_layers(self, layers) -> None:
        from afd_plugin.connectors.npu.mega_weights import expert_weights

        hf = self.vllm_config.model_config.hf_config
        layers = list(layers)
        ids = [layer.layer_idx for layer in layers]
        if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
            raise ValueError("Mega requires unique nonnegative model layer IDs")
        self.weights = {
            layer.layer_idx: expert_weights(
                layer.mlp, self.role_rank, hf.hidden_size, hf.moe_intermediate_size
            )
            for layer in layers
        }
        if self.layered_gmm_enabled:
            self.layered_weights = pack_layer_weights(self.weights)

    def init_afd_connector(self) -> None:
        if self.is_initialized:
            return
        from afd_plugin.connectors.npu import mega_runtime as ops

        self.ops = ops
        stores = []
        group = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
            world_size=self.attn_size + self.ffn_size,
            rank=self.world_rank,
            group_name="afd_mega",
            timeout=STARTUP_TIMEOUT,
            store_callback=stores.append,
        )
        try:
            store = dist.PrefixStore("afd_mega", stores[0])
            # FFN TP can differ from Attention TP. Share actual widths and check
            # all explicit declarations before anyone enters buffer collectives.
            attention = self.afd_config.role == "attention"
            self.layer_slots = agree_layer_selection(
                store,
                self.ffn_size,
                None if attention else self.role_rank,
                self.layered_gmm_enabled,
                list(self.weights),
            )
            width = (
                self.vllm_config.parallel_config.tensor_parallel_size
                if attention
                else 0
            )
            parallel = self.vllm_config.parallel_config
            dbo = attention and (parallel.enable_dbo or parallel.use_ubatching)
            store.set(
                f"tp/{self.world_rank}",
                f"{width} {self.extra_info.attn_ranks_per_dp or 0} {int(dbo)}",
            )
            widths = [
                tuple(map(int, store.get(f"tp/{i}").split()))
                for i in range(self.attn_size + self.ffn_size)
            ]
            self.attn_tp = widths[0][0]
            if (
                self.attn_size % self.attn_tp
                or any(
                    actual != self.attn_tp for actual, _, _ in widths[: self.attn_size]
                )
                or any(declared not in (0, self.attn_tp) for _, declared, _ in widths)
            ):
                raise ValueError("Inconsistent Attention TP / attn_ranks_per_dp")
            dual_stage = widths[0][2]
            if any(dbo != dual_stage for _, _, dbo in widths[: self.attn_size]):
                raise ValueError("All Attention ranks must agree on Mega ubatching")
            # Only Attention schedulers own the token limit. All window offsets
            # must be derived from the same capacity, including on FFN workers.
            if self.afd_config.role == "attention":
                store.set(
                    f"capacity/{self.role_rank}",
                    str(
                        self.vllm_config.scheduler_config.max_num_batched_tokens
                        // self.attn_tp
                    ),
                )
            capacity = max(
                int(store.get(f"capacity/{dp}")) for dp in range(self.attn_size)
            )
            hf = self.vllm_config.model_config.hf_config
            for stage in range(1 + dual_stage):
                if stage:
                    # An independent HCCL domain isolates UMDK's payload AND its
                    # per-FFN AIV progress words. Reuse the rendezvous TCPStore.
                    group = init_afd_process_group(
                        backend="hccl",
                        existing_store=stores[0],
                        world_size=self.attn_size + self.ffn_size,
                        rank=self.world_rank,
                        group_name="afd_mega_stage1",
                        timeout=STARTUP_TIMEOUT,
                    )
                with DefaultProcessGroupSwitcher(dist.group.WORLD, group):
                    buffer = ops.create_mega_split_buffer(
                        group,
                        self.attn_size,
                        self.ffn_size,
                        hf.n_routed_experts,
                        capacity,
                        hf.hidden_size,
                        hf.moe_intermediate_size,
                        hf.num_experts_per_tok,
                        shared_experts=0,
                        combine_quant_mode=0,
                        activation_parameters=(hf.swiglu_limit, 1.702, 1.0),
                    )
                lane = MegaStage(
                    group,
                    buffer,
                    ExpertWorkQueue(
                        dist.PrefixStore(f"stage/{stage}", store),
                        self.attn_size // self.attn_tp,
                        self.ffn_size,
                        ops.split_sequence,
                        read_published=(
                            buffer.create_published_reader(self.attn_tp)
                            if dual_stage and not attention
                            else None
                        ),
                    ),
                )
                self.stages.append(lane)
                if dual_stage and not attention:
                    lane.stream = torch.npu.Stream()
                    lane.stream.wait_stream(torch.npu.current_stream())
        except Exception:
            if all(group is not lane.group for lane in self.stages):
                dist.destroy_process_group(group)
            for lane in reversed(self.stages):
                lane.buffer.destroy()
                dist.destroy_process_group(lane.group)
            self.stages.clear()
            raise

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: torch.Tensor,
    ) -> None:
        stage = context.metadata.stage_idx
        if not 0 <= stage < len(self.stages):
            raise RuntimeError("Unexpected Mega Attention ubatch")
        lane = self.stages[stage]
        if lane.ticket is not None:
            raise RuntimeError(
                "Mega eager supports one outstanding ticket per ubatch"
            )
        dp = self.role_rank // self.attn_tp
        members = self.dp_members(dp)
        layer_slot = self.layer_slots[context.metadata.layer_idx]
        sequence = self.ops.split_sequence(lane.sequence + 1)
        ticket = self.ops.atten_send(
            hidden_states.contiguous(),
            kwargs["topk_ids"].int().contiguous(),
            kwargs["topk_weights"].float().contiguous(),
            lane.buffer,
            members,
            sequence,
            layer_index=layer_slot,
        )

        lane.ticket = ticket
        lane.sequence = sequence
        if self.role_rank == members[0]:
            lane.work.publish(dp, context.metadata.layer_idx, sequence)

    def dp_members(self, dp: int) -> list[int]:
        start = dp * self.attn_tp
        return list(range(start, start + self.attn_tp))

    def recv_ffn_output(
        self, ref_tensor: torch.Tensor, ubatch_idx: int = 0, **kwargs
    ) -> torch.Tensor:
        if (
            not 0 <= ubatch_idx < len(self.stages)
            or self.stages[ubatch_idx].ticket is None
        ):
            raise RuntimeError("Mega combine requires the matching ubatch ticket")
        lane = self.stages[ubatch_idx]
        output = self.ops.atten_recv(lane.ticket, lane.buffer)
        lane.ticket = None
        return output

    def execute_next_experts(self) -> None:
        for offset in range(len(self.stages)):
            idx = (self.next_stage + offset) % len(self.stages)
            lane = self.stages[idx]
            if lane.completion is not None and not lane.completion.query():
                continue
            item = lane.work.poll(self.role_rank)
            if item is None:
                continue
            dp, layer, sequence = item
            if self.layered_gmm_enabled:
                assert self.layered_weights is not None
                weights = self.layered_weights
            else:
                weights = self.weights[layer]
            args = (*weights, lane.buffer, self.dp_members(dp), sequence)
            if lane.stream is None:
                self.ops.mega_ffn(*args)
                torch.npu.synchronize()
            else:
                with torch.npu.stream(lane.stream):
                    self.ops.mega_ffn(*args)
                    if lane.completion is None:
                        lane.completion = torch.npu.Event()
                    lane.completion.record(lane.stream)
            self.next_stage = (idx + 1) % len(self.stages)
            return
        time.sleep(POLL_SECONDS)

    def recv_attn_output(self, ubatch_idx=0, **kwargs):
        raise RuntimeError(
            "Mega FFN runs recv, compute and send in execute_next_experts"
        )

    def send_ffn_output(self, ffn_output, context, **kwargs):
        raise RuntimeError("Mega FFN sends results inside fused_moe")

    def close(self) -> None:
        if self.stages:
            torch.npu.synchronize()
        for lane in reversed(self.stages):
            dist.barrier(group=lane.group)
            lane.buffer.destroy()
            dist.destroy_process_group(lane.group)
        self.stages.clear()
        self.next_stage = 0
        self.weights.clear()
        self.layered_weights = None
        self.layer_slots.clear()
