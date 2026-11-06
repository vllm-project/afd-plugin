# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host recording boundary for CAM native messages; no device simulation."""

import sys
from inspect import Parameter, Signature
from types import SimpleNamespace

from afd_plugin.config import AFDConfig


class _FakeTensor:
    def __init__(self, shape, *, dtype="bf16", device="npu:0"):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = device

    def to(self, *, dtype):
        self.dtype = dtype
        return self

    def contiguous(self):
        return self

    def new_zeros(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)

    def new_empty(self, shape):
        return _FakeTensor(shape, dtype=self.dtype, device=self.device)


class FakePullState(_FakeTensor):
    """Host status words at the native tensor boundary; no device execution."""

    def __init__(self):
        super().__init__((128,), dtype="fp16")
        self.words = {32: -1}

    def view(self, dtype):
        return self

    def __getitem__(self, index):
        return SimpleNamespace(
            item=lambda: self.words.get(index, 0),
            fill_=lambda value: self.words.__setitem__(index, value),
        )


class _FakeCamOps:
    def __init__(self):
        self.calls = []

    def _record(self, name, fields, args, kwargs):
        # Native schema binding accepts either positional or named arguments.
        signature = Signature(
            [
                Parameter(field, Parameter.POSITIONAL_OR_KEYWORD)
                for field in fields.split()
            ]
        )
        values = tuple(signature.bind(*args, **kwargs).arguments.values())
        self.calls.append((name, values))
        return values

    def afd_mega_pull_init(self, *args):
        self.calls.append(("pull_init", args))
        return FakePullState()

    def afd_async_dispatch_send(self, *args, **kwargs):
        values = self._record(
            "dispatch_send",
            "x expert_ids comm_args comm_id max_seq_len batch_size hidden_size "
            "top_k moe_rank_num attn_rank_num route_expert_num_per_moe "
            "attn_rank_id world_size layer_index tp_size dynamic_quant group_name",
            args,
            kwargs,
        )
        return values[0]

    def afd_async_dispatch_recv(self, *args, **kwargs):
        values = self._record(
            "dispatch_recv",
            "x comm_args comm_id max_seq_len hidden_size top_k moe_rank_num "
            "attn_rank_num route_expert_num_per_moe moe_rank_id world_size "
            "tp_size dynamic_quant group_name",
            args,
            kwargs,
        )
        batch_size, hidden_size, expert_per_rank, tp_size = (
            values[3],
            values[4],
            values[8],
            values[11],
        )
        return (
            _FakeTensor((batch_size, hidden_size)),
            _FakeTensor((batch_size,), dtype="fp32"),
            _FakeTensor((5 + tp_size * (1 + expert_per_rank),), dtype="int64"),
            _FakeTensor((expert_per_rank,), dtype="int64"),
        )

    def afd_async_combine_send(self, *args, **kwargs):
        values = self._record(
            "combine_send",
            "expand_x comm_args batch_info comm_id max_seq_len hidden_size "
            "top_k moe_rank_num attn_rank_num route_expert_num_per_moe "
            "moe_rank_id world_size tp_size group_name",
            args,
            kwargs,
        )
        return values[0]

    def afd_async_combine_recv(self, *args, **kwargs):
        values = self._record(
            "combine_recv",
            "expand_x expert_ids expert_scales comm_args comm_id batch_size "
            "hidden_size top_k moe_rank_num attn_rank_num route_expert_num_per_moe "
            "attn_rank_id world_size group_name",
            args,
            kwargs,
        )
        return _FakeTensor((values[5], values[6]))

    def afd_mega_dispatch_send(self, *args, **kwargs):
        result = self.afd_async_dispatch_send(*args, **kwargs)
        self.calls[-1] = ("mega_dispatch_send", self.calls[-1][1])
        return result

    def afd_mega_combine_recv(self, *args, max_seq_len, tp_size, **kwargs):
        result = self.afd_async_combine_recv(*args, **kwargs)
        self.calls[-1] = (
            "mega_combine_recv",
            self.calls[-1][1] + (max_seq_len, tp_size),
        )
        return result


class _FakeTorch:
    def __init__(self):
        self.bfloat16 = "bf16"
        self.float16 = "fp16"
        self.float32 = "fp32"
        self.int32 = "int32"
        self.int64 = "int64"
        self.ops = SimpleNamespace(afd_ascend=_FakeCamOps())
        self.npu = SimpleNamespace(synchronize=lambda: None)

    def device(self, name):
        return name

    def empty(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)

    def zeros(self, shape, *, dtype, device):
        return _FakeTensor(shape, dtype=dtype, device=device)


def cam_vllm_config(*, tp_size=1, pcp_size=1, extra_config=None, dtype="bf16"):
    return SimpleNamespace(
        additional_config={
            "afd": {
                "connector_extra_config": extra_config
                if extra_config is not None
                else {"attn_ranks_per_dp": 4},
            },
        },
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            prefill_context_parallel_size=pcp_size,
            tensor_parallel_size=tp_size,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_config=SimpleNamespace(
                hidden_size=16,
                num_experts_per_tok=2,
                n_routed_experts=8,
            ),
        ),
    )


def _afd_config(*, role: str):
    return AFDConfig(
        connector="CAMAsyncAFDConnector",
        role=role,
        num_attention_ranks=4,
        num_ffn_ranks=2,
    )


def _dp2_afd_config(*, role: str):
    return AFDConfig(
        connector="CAMAsyncAFDConnector",
        role=role,
        num_attention_ranks=8,
        num_ffn_ranks=8,
    )


def _topk_payload(batch_size: int, topk: int = 2):
    return {
        "topk_ids": _FakeTensor((batch_size, topk), dtype="int32"),
        "topk_weights": _FakeTensor((batch_size, topk), dtype="fp32"),
    }


def recording_cam_connector(
    monkeypatch, module, *, role="attention", mega=False, tp_size=4, pcp_size=1
):
    fake_torch = _FakeTorch()
    monkeypatch.setattr(module, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    connector = module.CAMAsyncAFDConnector(
        0,
        0,
        cam_vllm_config(
            pcp_size=pcp_size,
            extra_config={
                "attn_ranks_per_dp": tp_size,
                "ffn_mega_moe": mega,
                "dynamicQuant": 1,
            },
        ),
        _afd_config(role=role),
        0,
    )
    monkeypatch.setattr(connector.graph_startup, "phase", lambda phase: None)
    connector.ffn_session().initialize("npu:0", "recorded")
    connector._initialized = True
    fake_torch.ops.afd_ascend.calls.clear()
    return SimpleNamespace(connector=connector, torch=fake_torch)
