# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the AFD hash-ids alignment patch.

The patch replaces one vLLM-Ascend selector function, so the tests install a fake
``vllm``/``vllm_ascend`` surface, import the patch module against it, and drive the
selector through the module the patch rebinds. What matters is that ids which
already cover the router rows reach the hash operator unchanged, and that ids which
do not keep the upstream alignment.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_PATCH_MODULE = "afd_plugin.compat.patches.npu.hash_ids_alignment"


class _RecordingForwardContext:
    """Forward context exposing the fields the hash selector reads."""

    def __init__(self, *, input_ids: Any, comm_type: str = "mc2") -> None:
        self.input_ids = input_ids
        self.moe_comm_type = comm_type
        self.flash_comm_v1_enabled = True
        self.alignment_calls: list[Any] = []
        self.moe_comm_method = SimpleNamespace(
            pad_and_split_input_ids=self._pad_and_split_input_ids,
            prepare_finalize=SimpleNamespace(
                all_gather_input_id_with_dp_group=self._all_gather,
            ),
        )

    def _pad_and_split_input_ids(self, input_ids: Any) -> Any:
        self.alignment_calls.append(("pad_and_split", int(input_ids.numel())))
        return input_ids

    def _all_gather(self, input_ids: Any) -> Any:
        self.alignment_calls.append(("all_gather", int(input_ids.numel())))
        return input_ids


def _install_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    forward_context: _RecordingForwardContext,
    hash_calls: list[dict[str, Any]],
    with_selector: bool = True,
) -> Any:
    """Install the vLLM and vLLM-Ascend surface the patch module imports.

    The fakes are module objects the patch imports by name, so they are typed
    loosely: the attributes are the whole point of the fake.
    """

    vllm: Any = types.ModuleType("vllm")
    vllm_distributed: Any = types.ModuleType("vllm.distributed")
    vllm_distributed.get_tp_group = lambda: SimpleNamespace(
        world_size=2,
        rank_in_group=0,
    )
    vllm_forward_context: Any = types.ModuleType("vllm.forward_context")
    vllm_forward_context.get_forward_context = lambda: forward_context

    vllm_ascend: Any = types.ModuleType("vllm_ascend")
    ascend_forward_context: Any = types.ModuleType("vllm_ascend.ascend_forward_context")
    ascend_forward_context.MoECommType = SimpleNamespace(ALLGATHER="allgather")
    device_pkg: Any = types.ModuleType("vllm_ascend.device")
    device_op: Any = types.ModuleType("vllm_ascend.device.device_op")
    device_op.DeviceOperator = SimpleNamespace(
        moe_gating_top_k=lambda *args, **kwargs: ("weights", "ids", None),
    )
    distributed_pkg: Any = types.ModuleType("vllm_ascend.distributed")
    distributed_utils: Any = types.ModuleType("vllm_ascend.distributed.utils")
    distributed_utils.split_tensor_along_first_dim = lambda tensor, num_partitions: (
        tensor.chunk(num_partitions)
    )

    ops_pkg: Any = types.ModuleType("vllm_ascend.ops")
    fused_moe_pkg: Any = types.ModuleType("vllm_ascend.ops.fused_moe")
    selector: Any = types.ModuleType("vllm_ascend.ops.fused_moe.experts_selector")

    def fusion_selector(**kwargs: Any) -> tuple[Any, Any]:
        hash_calls.append(kwargs)
        return "weights", "ids"

    def select_experts(**kwargs: Any) -> tuple[Any, Any]:
        # The upstream module calls the fusion selector through its module global,
        # which is what the patch rebinds.
        return selector._select_experts_with_fusion_ops(
            scoring_func="sqrtsoftplus",
            **kwargs,
        )

    if with_selector:
        selector._select_experts_with_fusion_ops = fusion_selector
    selector.select_experts = select_experts

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_distributed)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", vllm_forward_context)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        ascend_forward_context,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.device", device_pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_op", device_op)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed", distributed_pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.utils", distributed_utils)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", fused_moe_pkg)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.fused_moe.experts_selector", selector
    )
    return selector


def _import_patch_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the patch module fresh so its apply-once state starts clean."""

    monkeypatch.delitem(sys.modules, _PATCH_MODULE, raising=False)
    module = importlib.import_module(_PATCH_MODULE)
    monkeypatch.setitem(sys.modules, _PATCH_MODULE, module)
    return module


def _fake_hash_ops(hash_calls: list[dict[str, Any]]) -> SimpleNamespace:
    """Build the ``torch.ops`` surface the selector calls."""

    def moe_gating_top_k_hash(**kwargs: Any) -> tuple[str, str, None]:
        hash_calls.append(kwargs)
        return "weights", "ids", None

    return SimpleNamespace(
        _C_ascend=SimpleNamespace(moe_gating_top_k_hash=moe_gating_top_k_hash),
    )


def test_patch_rebinds_the_fused_selector(monkeypatch):
    forward_context = _RecordingForwardContext(input_ids=torch.arange(4))
    hash_calls: list[dict[str, Any]] = []
    selector = _install_fake_modules(
        monkeypatch,
        forward_context=forward_context,
        hash_calls=hash_calls,
    )
    module = _import_patch_module(monkeypatch)
    monkeypatch.setattr(module.torch, "ops", _fake_hash_ops(hash_calls))

    module.apply_afd_hash_ids_alignment_patch()
    assert (
        selector._select_experts_with_fusion_ops
        is module._select_experts_with_fusion_ops
    )

    selector.select_experts(
        hidden_states=torch.zeros(4, 8),
        router_logits=torch.zeros(4, 8),
        top_k=2,
        use_grouped_topk=True,
        renormalize=False,
        e_score_correction_bias=None,
        topk_group=1,
        num_expert_group=1,
        routed_scaling_factor=1.0,
        tid2eid=torch.zeros(16, 2, dtype=torch.int32),
    )

    assert len(hash_calls) == 1


def test_aligned_ids_skip_the_upstream_realignment(monkeypatch):
    """Ids that already cover the router rows must reach the operator unchanged."""

    ids = torch.tensor([3, -1, 7, 11], dtype=torch.int32)
    forward_context = _RecordingForwardContext(input_ids=ids)
    hash_calls: list[dict[str, Any]] = []
    selector = _install_fake_modules(
        monkeypatch,
        forward_context=forward_context,
        hash_calls=hash_calls,
    )
    module = _import_patch_module(monkeypatch)
    monkeypatch.setattr(module.torch, "ops", _fake_hash_ops(hash_calls))
    module.apply_afd_hash_ids_alignment_patch()

    selector.select_experts(
        hidden_states=torch.zeros(4, 8),
        router_logits=torch.zeros(4, 8),
        top_k=2,
        use_grouped_topk=True,
        renormalize=False,
        e_score_correction_bias=None,
        topk_group=1,
        num_expert_group=1,
        routed_scaling_factor=1.0,
        tid2eid=torch.zeros(16, 2, dtype=torch.int32),
    )

    assert forward_context.alignment_calls == []
    assert hash_calls[0]["input_ids"].tolist() == [3, 0, 7, 11]


def test_unaligned_ids_keep_the_upstream_realignment(monkeypatch):
    """Ids that do not describe the router rows keep the upstream alignment."""

    ids = torch.tensor([3, 7], dtype=torch.int32)
    forward_context = _RecordingForwardContext(input_ids=ids)
    hash_calls: list[dict[str, Any]] = []
    selector = _install_fake_modules(
        monkeypatch,
        forward_context=forward_context,
        hash_calls=hash_calls,
    )
    module = _import_patch_module(monkeypatch)
    monkeypatch.setattr(module.torch, "ops", _fake_hash_ops(hash_calls))
    module.apply_afd_hash_ids_alignment_patch()

    selector.select_experts(
        hidden_states=torch.zeros(4, 8),
        router_logits=torch.zeros(4, 8),
        top_k=2,
        use_grouped_topk=True,
        renormalize=False,
        e_score_correction_bias=None,
        topk_group=1,
        num_expert_group=1,
        routed_scaling_factor=1.0,
        tid2eid=torch.zeros(16, 2, dtype=torch.int32),
    )

    assert forward_context.alignment_calls == [("pad_and_split", 2)]


def test_patch_raises_without_the_selector(monkeypatch):
    """A renamed selector has to fail the rebind instead of passing silently.

    AFD patches the pinned vLLM-Ascend revision, so the patch reads the attribute
    instead of probing for it: a revision that dropped the selector would
    otherwise leave the upstream id re-alignment in place, which is the behaviour
    this patch exists to replace.
    """

    forward_context = _RecordingForwardContext(input_ids=torch.arange(4))
    hash_calls: list[dict[str, Any]] = []
    _install_fake_modules(
        monkeypatch,
        forward_context=forward_context,
        hash_calls=hash_calls,
        with_selector=False,
    )
    module = _import_patch_module(monkeypatch)

    with pytest.raises(AttributeError, match="_select_experts_with_fusion_ops"):
        module.apply_afd_hash_ids_alignment_patch()


def test_ids_describe_router_rows_counts_one_id_per_row():
    module = importlib.import_module(_PATCH_MODULE)

    assert module.ids_describe_router_rows(torch.zeros(6), torch.zeros(6, 4))
    assert not module.ids_describe_router_rows(torch.zeros(3), torch.zeros(6, 4))
