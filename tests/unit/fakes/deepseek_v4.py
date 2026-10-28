# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU V4 decoder/protocol doubles; no mHC or device-kernel proof."""

from types import MethodType, SimpleNamespace

import torch
from torch import nn
from vllm.forward_context import (
    ForwardContext,
    override_forward_context,
    set_forward_context,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.v1.worker import workspace

from afd_plugin.connectors import AFDForwardContextMetadata


class Connector:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def send_attn_output(self, hidden_states, context, **kwargs) -> None:
        self.events.append(("send", hidden_states, context, kwargs))

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        self.events.append(("recv", ref_tensor, ubatch_idx))
        return ref_tensor * 0.25


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.failure = None

    def forward(self, positions, x, unused):
        if self.failure is not None:
            raise self.failure
        return x * 3


def mhc_step(x, residual, post_mix, res_mix, *args, **kwargs):
    # Functional CPU transforms make both retained mHC boundaries observable
    # through the decoder's return values, without asserting helper calls.
    return residual + 2, post_mix + 3, res_mix + 4, x + 1


class DecoderCase:
    def __init__(self, adapter, *, real_tokens=0, num_stages=2, all_padding=False):
        self.events: list[tuple] = []
        self.connector = Connector(self.events)
        self.metadata = AFDForwardContextMetadata(
            tokens_start_loc=[0],
            requests_start_loc=[0],
            stage_idx=1 if num_stages > 1 else 0,
            connector=self.connector,
            tokens_lens=[4],
            num_stages=num_stages,
            tokens_unpadded_lens=[] if real_tokens is None else [real_tokens],
        )
        # Native UB contexts are NONE even during FULL preparation/capture.
        self.context = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={"afd_metadata": self.metadata},
            is_padding=torch.full((4,), all_padding, dtype=torch.bool),
        )
        self.layer = object.__new__(adapter.AFDDeepseekV4DecoderLayer)
        nn.Module.__init__(self.layer)
        self.layer.attn = Attention()
        self.layer.ffn = adapter.RemoteDeepseekV4FFN(layer_idx=3)
        norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
        self.layer.attn_norm = self.layer.ffn_norm = norm
        self.layer.hc_attn_fn = self.layer.hc_attn_scale = self.layer.hc_attn_base = (
            torch.ones(1)
        )
        self.layer.hc_ffn_fn = self.layer.hc_ffn_scale = self.layer.hc_ffn_base = (
            torch.ones(1)
        )
        self.layer.rms_norm_eps = self.layer.hc_eps = 1e-6
        self.layer.hc_post_alpha = self.layer.hc_sinkhorn_iters = 1
        self.x = torch.full((4, 4), 8.0, dtype=torch.float16)
        self.positions = torch.arange(4)
        self.input_ids = torch.tensor([11, 13, 17, 19], dtype=torch.int32)
        self.residual = torch.full_like(self.x, 10)
        self.post_mix = torch.full_like(self.x, 20)
        self.res_mix = torch.full_like(self.x, 30)

    def record_yield(self, hidden_states, *, role):
        self.events.append(("yield", hidden_states, role))
        return hidden_states

    def forward(self):
        with override_forward_context(self.context):
            return self.layer(
                self.x,
                self.positions,
                self.input_ids,
                residual=self.residual,
                post_mix=self.post_mix,
                res_mix=self.res_mix,
            )


CPU_PROFILE_BUDGET_BYTES = 256 * 1024 * 1024
CPU_PROFILE_TRANSIENT_BYTES = 1024
OVERBOUND_WORKSPACE_BYTES = 1024 * 1024 * 1024


class FailingFlashAttention(DeepseekV4FlashMLAAttention):
    def forward_mqa(self, q, kv, positions, output):
        raise self.failure


def profile_attention(compress_ratio, error=None):
    cls = FailingFlashAttention if error is not None else DeepseekV4FlashMLAAttention
    attention = object.__new__(cls)
    nn.Module.__init__(attention)
    attention.compress_ratio = compress_ratio
    attention.max_model_len = 4096
    attention.max_num_batched_tokens = 1024
    attention.window_size = 128
    attention.max_image_tokens = 0
    attention.head_dim = 512
    attention.padded_heads = 64
    attention.attn_sink = nn.Parameter(torch.zeros(64), requires_grad=False)
    attention.topk_indices_buffer = (
        None if compress_ratio <= 1 else torch.empty((1, 512), dtype=torch.int32)
    )
    if error is not None:
        attention.failure = error
    return attention


class WorkspaceCase:
    """Real native CPU capacity/NoMeta seam, not a CUDA profiler or kernel fake."""

    def __init__(self, monkeypatch, *, num_lanes=1, backend=True, error=None):
        self.num_ubatches = 2
        self.num_lanes = num_lanes
        self.ubatch = 0
        self.manager = workspace.WorkspaceManager(
            torch.device("cpu"), self.num_ubatches, num_lanes=num_lanes
        )
        monkeypatch.setattr(workspace, "_manager", self.manager)
        monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: self.ubatch)
        monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v4")
            ),
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                use_sequence_parallel_moe=False,
                is_moe_model=True,
                use_ubatching=True,
            ),
            compilation_config=SimpleNamespace(
                fast_moe_cold_start=False,
                static_forward_context={},
            ),
        )
        self.attention = [profile_attention(r) for r in (1, 4, 128)] if backend else []
        if error is not None:
            self.attention = [profile_attention(4, error)]
        self.model = nn.ModuleList(self.attention or [nn.Identity()])
        self.require_all_slots = True
        self.profile_views: list[torch.Tensor] = []
        self.remaining_budget: int | None = None
        self.outer_context = ForwardContext(
            no_compile_layers={}, attn_metadata={"prior": None}, slot_mapping={}
        )

    def request(self, ubatch, lane):
        self.ubatch = ubatch
        with workspace.use_workspace_lane(lane), set_forward_context(None, self.config):
            for attention in self.attention:
                q = torch.empty(
                    (1, attention.padded_heads, attention.head_dim),
                    dtype=torch.bfloat16,
                )
                output = torch.ones_like(q)
                attention.forward_mqa(
                    q, q[:, 0, :], torch.zeros(1, dtype=torch.int64), output
                )
                assert torch.equal(output, torch.zeros_like(output))
            return self.manager.get_simultaneous(((1,), torch.uint8))[0]

    def install_prior_slot_override(self):
        # A pre-existing instance selector intentionally differs from normal
        # ubatch/lane lookup; its returned storage detects exact restoration.
        slot = (self.num_ubatches - 1) * self.num_lanes
        self.manager._get_workspace_id = MethodType(lambda _self: slot, self.manager)


def profile_workspace_peak(runner, randomize_inputs=False):
    case = runner.workspace_case
    # The native single-context profile can reserve slot0 itself. Lock here is
    # a CPU test barrier: all other owners must already reside at peak entry.
    if case.attention:
        case.request(0, 0)
    case.manager.lock()
    if case.require_all_slots:
        case.profile_views = [
            case.request(ubatch, lane)
            for ubatch in range(case.num_ubatches)
            for lane in range(case.num_lanes)
        ]
    elif case.attention:
        case.profile_views = [case.request(0, 0)]
    transient = torch.empty(CPU_PROFILE_TRANSIENT_BYTES, dtype=torch.uint8)
    resident = sum(view.untyped_storage().nbytes() for view in case.profile_views)
    # Observable CPU quota proxy only; no claim about CUDA activation peaks.
    case.remaining_budget = CPU_PROFILE_BUDGET_BYTES - resident - transient.numel()
