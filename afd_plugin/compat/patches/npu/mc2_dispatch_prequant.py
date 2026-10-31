# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""Let MC2 dispatch take an activation that is already quantized.

Upstream source: vllm_ascend/ops/fused_moe/token_dispatcher.py (8d4409d),
``TokenDispatcherWithMC2.get_dispatch_mc2_kwargs``.

Patch reason: the Attention-side gate hands the FFN an activation quantized on the
other role, together with a two-dimensional E8M0 scale, and the prepare patch
publishes that scale as the routing's ``pertoken_scale``. MC2's dispatch kwargs do
not read it: ``scales`` is ``None`` and ``quant_mode`` resolves to the MXFP
communication mode, so the operator would quantize a tensor that is already
quantized and the scale from the other role would be dropped.

Patch functionality: when the routing carries a scale, pass it through and set
``quant_mode`` to 0, which tells the operator the activation is already quantized.
Every other configuration keeps upstream's behaviour exactly: no scale means
``scales`` stays ``None`` and the quant mode is resolved as before.

The contract comes from a device smoke that this branch does not carry: the script
(``tests/npu/mc2_prequant_smoke.py``) and its runner landed with the pre-quantized
dispatch work, commit ``1096b53`` on the experiment branch. What it established: the
installed operator accepts FP8 activations with a two-dimensional E8M0 scale at
``quant_mode=0`` and returns a scale of the same shape next to the activation. The
branch that owns this patch has not repeated that run, so the acceptance above is a
recorded result rather than a current measurement.
Where that output goes downstream is upstream's existing path: ``token_dispatch``
unpacks it as ``dynamic_scale`` and the MXFP quant method normalizes its layout
when it consumes it, so nothing here converts it.

Signature: matches upstream; no added parameters.
"""

from __future__ import annotations

from typing import Any

import torch

# Resolved on first use rather than at import time. The patch is installed inside a
# worker process, but keeping the import out of module scope lets a CPU-only host
# import this module and exercise the patch's branching against stand-ins.
TokenDispatcherWithMC2: Any = None
_get_expert_token_nums_type: Any = None



def _ensure_upstream_imports() -> None:
    """Resolve the vLLM-Ascend names once, leaving supplied stand-ins alone."""

    global TokenDispatcherWithMC2, _get_expert_token_nums_type

    if TokenDispatcherWithMC2 is not None:
        return

    from vllm_ascend.ops.fused_moe.token_dispatcher import (
        TokenDispatcherWithMC2 as upstream_dispatcher,
    )
    from vllm_ascend.ops.fused_moe.token_dispatcher import (
        _get_expert_token_nums_type as upstream_nums_type,
    )

    TokenDispatcherWithMC2 = upstream_dispatcher
    _get_expert_token_nums_type = upstream_nums_type


def _for_testing_reset() -> None:
    """Drop resolved upstream objects. Only tests use this."""

    global TokenDispatcherWithMC2, _get_expert_token_nums_type

    TokenDispatcherWithMC2 = None
    _get_expert_token_nums_type = None


# The only pre-quantized combination the device smoke established, so the only one
# this patch accepts. Anything else keeps upstream's behaviour rather than being
# reinterpreted as already-quantized: a model-dtype activation with some other
# scale would otherwise be passed to the operator as if its scale described it.
PREQUANT_ACTIVATION_DTYPES = (torch.float8_e4m3fn,)
PREQUANT_SCALE_DTYPE = torch.float8_e8m0fnu
PREQUANT_QUANT_MODE = 0


def _require_prequant_contract(token_dispatch_input: Any, routing_scale: Any) -> None:
    """Reject a scale that does not describe this activation.

    The check runs before the dispatch kwargs are built so a mis-specified scale
    fails here, naming what was wrong, instead of surfacing as a shape error inside
    the operator or as silently wrong values.
    """

    quant = token_dispatch_input.quant
    if not quant.is_mxfp:
        raise RuntimeError(
            "AFD MC2 pre-quantized dispatch expects an MXFP quant feature; got "
            f"is_mxfp={quant.is_mxfp}",
        )
    if routing_scale.dim() != 2:
        raise RuntimeError(
            "AFD MC2 pre-quantized dispatch expects a two-dimensional scale; got "
            f"{tuple(routing_scale.shape)}",
        )
    if routing_scale.dtype is not PREQUANT_SCALE_DTYPE:
        raise RuntimeError(
            "AFD MC2 pre-quantized dispatch expects an E8M0 scale; got "
            f"{routing_scale.dtype}",
        )
    if token_dispatch_input.hidden_states.dtype not in PREQUANT_ACTIVATION_DTYPES:
        raise RuntimeError(
            "AFD MC2 pre-quantized dispatch expects an E4M3 activation; got "
            f"{token_dispatch_input.hidden_states.dtype}",
        )
    mxfp = quant.mxfp
    act_quant_type = None if mxfp is None else mxfp.act_quant_type
    if act_quant_type is not None and act_quant_type not in PREQUANT_ACTIVATION_DTYPES:
        raise RuntimeError(
            "AFD MC2 pre-quantized dispatch expects an E4M3 activation type; got "
            f"{act_quant_type}",
        )


# Upstream source: vllm_ascend/ops/fused_moe/token_dispatcher.py (8d4409d),
# TokenDispatcherWithMC2.get_dispatch_mc2_kwargs, copied in full.
# Patch reason: ``scales`` was hardcoded to None and the quant mode ignored the
# routing's pre-quantized activation, so a scale produced on the Attention side
# could not reach the operator.
# Patch functionality: forward the routing's pertoken scale and select
# quant_mode 0 when it is present; leave every other path as upstream has it.
# Signature: matches upstream; no added parameters.
def get_dispatch_mc2_kwargs(
    self,
    token_dispatch_input: Any,
):
    _ensure_upstream_imports()

    hidden_states = token_dispatch_input.hidden_states
    topk_weights = token_dispatch_input.topk_weights
    topk_ids = token_dispatch_input.topk_ids
    expert_map = token_dispatch_input.routing.expert_map
    global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
    comm_quant_mode = token_dispatch_input.quant.comm_quant_mode

    # ### PATCH START: the activation may already be quantized
    # Set only when the Attention side quantized the activation and sent its scale;
    # absent for the FFN-side gate, which sends a model-dtype activation.
    routing_scale = token_dispatch_input.routing.pertoken_scale
    # ### PATCH END: the activation may already be quantized

    assert expert_map is not None, "expert_map is required for MC2 token dispatch."
    # NOTE: quant_mode differs by quant feature:
    # - Legacy int communication quantization uses quant_mode=2.
    # - A5 MXFP communication uses quant_mode=4.
    if comm_quant_mode is not None:
        quant_mode = comm_quant_mode
    elif token_dispatch_input.quant.dispatch_with_quant:
        quant_mode = 4 if self.need_shared_expert_args and token_dispatch_input.quant.is_mxfp else 2
    else:
        quant_mode = 0
    # ### PATCH START: a scale from the Attention side means no quantization here
    # quant_mode 0 states the activation is already quantized, so the operator reads
    # the scale instead of producing one. An explicit communication quant mode that
    # disagrees is a configuration conflict, not something to silently overrule: the
    # caller asked for one quantisation scheme and the routing carries another.
    if routing_scale is not None:
        if comm_quant_mode is not None and int(comm_quant_mode) != PREQUANT_QUANT_MODE:
            raise RuntimeError(
                "AFD MC2 pre-quantized dispatch: comm_quant_mode="
                f"{comm_quant_mode} conflicts with a pre-quantized activation, which "
                f"requires quant_mode={PREQUANT_QUANT_MODE}",
            )
        _require_prequant_contract(token_dispatch_input, routing_scale)
        quant_mode = PREQUANT_QUANT_MODE
    # ### PATCH END: a scale from the Attention side means no quantization here
    self.moe_expert_num = len(expert_map) + global_redundant_expert_num
    expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
    kwargs_mc2 = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": self.moe_expert_num,
        "global_bs": self.global_bs,
        "expert_token_nums_type": expert_token_nums_type,
    }
    if self.global_bs == 0:
        kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

    stage1_kwargs = {
        # ### PATCH START: forward the scale that arrived with the activation
        # None keeps upstream's behaviour for an activation this side quantized.
        "scales": routing_scale,
        # ### PATCH END: forward the scale that arrived with the activation
        "quant_mode": quant_mode,
        "group_ep": self.moe_all_to_all_group_name,
        "ep_world_size": self.ep_world_size,
        "ep_rank_id": self.ep_rank_id,
        "comm_alg": self.mc2_comm_alg,
    }
    if self.need_extra_args:
        stage1_kwargs.update(
            {
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            }
        )
    # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
    if (
        self.need_shared_expert_args
        and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
        and token_dispatch_input.quant.dispatch_with_quant
    ):
        y_dtype = torch.float8_e4m3fn
        if (
            token_dispatch_input.quant.mxfp is not None
            and token_dispatch_input.quant.mxfp.act_quant_type is not None
        ):
            y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
        stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
    if self.need_expert_scale:
        stage1_kwargs.update(
            {
                "expert_scales": topk_weights.to(torch.float32),
            }
        )

    kwargs_mc2.update(stage1_kwargs)
    return kwargs_mc2


def apply_afd_mc2_dispatch_prequant_patch() -> bool:
    """Install the patch once. Returns whether it is installed."""

    _ensure_upstream_imports()
    current = TokenDispatcherWithMC2.get_dispatch_mc2_kwargs
    if current is get_dispatch_mc2_kwargs:
        return True
    # Comparing against a captured reference cannot catch a replacement that already
    # happened when the reference was taken. The installed function's qualname is
    # checked instead. This is a name check, not an identity check: a wrapper that
    # keeps upstream's qualified name would still pass, so it catches the ordinary
    # case of another patch having taken the method and nothing more.
    expected_qualname = f"{TokenDispatcherWithMC2.__qualname__}.get_dispatch_mc2_kwargs"
    if getattr(current, "__qualname__", None) != expected_qualname:
        raise RuntimeError(
            "AFD MC2 dispatch patch: get_dispatch_mc2_kwargs is already replaced by "
            f"{getattr(current, '__qualname__', current)!r}, so the upstream body this "
            "patch copies may not match",
        )
    TokenDispatcherWithMC2.get_dispatch_mc2_kwargs = get_dispatch_mc2_kwargs
    return True


__all__ = ["apply_afd_mc2_dispatch_prequant_patch", "get_dispatch_mc2_kwargs"]
