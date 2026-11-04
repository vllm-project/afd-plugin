# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the DSV4.1 NPU role-aware checkpoint filter.

``_checkpoint_weight_roles`` decides which AFD role receives each checkpoint
path. The distinction is not cosmetic: the upstream Ascend loader indexes its
parameter dict by name without a membership check, so a path handed to a role
that never registered it raises ``KeyError`` during weight loading instead of
being skipped.

DSV4.1 adds two ownership questions that DSV4 did not have. Engram n-gram
tables exist only on the Attention role, because only that role executes the
backbone forward. Vision and draft tensors belong to no AFD role at all.

How the functions are loaded
----------------------------
The NPU DSV4.1 module defines these helpers next to its model classes, and
importing it pulls in the whole ``vllm``, ``vllm_ascend`` and ``transformers``
class surface. Stubbing that surface would make this file break on every
upstream addition and would test the stubs as much as the code.

Instead the helpers are read from the module source and executed in an
isolated namespace. They are module-level functions that reference nothing but
each other, three role constants, and two name-prefix tuples, so this
exercises the real code from the real file without importing it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "deepseek_v41.py"
)

_HELPER_NAMES = (
    "_weight_layer_path",
    "_checkpoint_weight_roles",
)


def _load_helpers() -> ModuleType:
    """Execute the role-filter helpers in an isolated namespace.

    Returns:
        A module-like namespace holding ``_checkpoint_weight_roles``.

    Raises:
        AssertionError: If the module no longer defines the helpers, which means
            this test needs to be repointed rather than silently pass.
    """

    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    namespace: dict[str, object] = {
        "frozenset": frozenset,
        "tuple": tuple,
        "int": int,
        "str": str,
        "None": None,
        "_ATTENTION_ROLE": "attention",
        "_FFN_ROLE": "ffn",
        "_ATTENTION_ROLES": frozenset(("attention",)),
        "_FFN_ROLES": frozenset(("ffn",)),
        "_BOTH_ROLES": frozenset(("attention", "ffn")),
        "_NO_ROLES": frozenset(),
        "_MULTIMODAL_NAME_PREFIXES": ("vision.", "aligner.", "image_"),
        "_DEFERRED_NAME_PREFIXES": ("mtp.",),
    }

    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in _HELPER_NAMES:
            continue
        # ``from __future__ import annotations`` makes the annotations lazy, so
        # the definitions can be executed without their referenced types.
        code = compile(ast.Module(body=[node], type_ignores=[]), "<helpers>", "exec")
        exec(code, namespace)  # noqa: S102 - source is this repository's own file
        found.add(node.name)

    assert found == set(_HELPER_NAMES), (
        f"{_MODULE_PATH} no longer defines {sorted(_HELPER_NAMES)}; "
        f"found {sorted(found)}"
    )

    module = ModuleType("dsv41_role_helpers")
    module.__dict__.update(namespace)
    return module


_helpers = _load_helpers()
_checkpoint_weight_roles = _helpers._checkpoint_weight_roles  # type: ignore[attr-defined]


def test_module_and_helpers_are_present() -> None:
    assert _MODULE_PATH.is_file()


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.engram.embed.weight",
        "model.layers.14.engram.embed.weight",
        "model.layers.14.engram.embed.scale",
    ],
)
def test_engram_paths_belong_to_the_attention_role(name: str) -> None:
    """Only Attention allocates n-gram tables, so only it may receive them."""

    assert _checkpoint_weight_roles(
        name,
        attention_owns_engram=True,
    ) == frozenset({"attention"})
    assert (
        _checkpoint_weight_roles(
            name,
            attention_owns_engram=False,
        )
        == frozenset()
    )


@pytest.mark.parametrize(
    "name",
    [
        "vision.blocks.0.attn.qkv.weight",
        "aligner.mlp.0.weight",
        "image_newline",
        "mtp.layers.0.ffn.gate.weight",
    ],
)
def test_non_text_paths_belong_to_no_afd_role(name: str) -> None:
    """The text-only split never executes vision or draft tensors."""

    for attn_owns_gate in (True, False):
        assert (
            _checkpoint_weight_roles(
                name,
                attn_owns_gate=attn_owns_gate,
                attention_owns_engram=True,
            )
            == frozenset()
        )


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.attn.q_a_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ],
)
def test_attention_paths_are_attention_owned(name: str) -> None:
    assert _checkpoint_weight_roles(name) == frozenset({"attention"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.ffn.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.3.down_proj.weight",
    ],
)
def test_expert_paths_are_ffn_owned(name: str) -> None:
    for attn_owns_gate in (True, False):
        assert _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
        ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.ffn.gate.e_score_correction_bias",
    ],
)
def test_gate_paths_follow_the_router_owner(name: str) -> None:
    """A role that built no router raises ``KeyError`` on a foreign gate path."""

    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=True,
    ) == frozenset({"attention", "ffn"})
    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=False,
    ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.attn_norm.weight",
        "model.layers.0.ffn_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.hc_attn_fn",
        "model.layers.0.hc_ffn_base",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ],
)
def test_layer_scaffolding_is_shared_by_both_roles(name: str) -> None:
    assert _checkpoint_weight_roles(name) == frozenset({"attention", "ffn"})


def test_vision_router_bias_stays_on_the_ffn_role() -> None:
    """Only the native MoE registers ``gate.bias_vl``, so Attention has no slot."""

    assert _checkpoint_weight_roles(
        "model.layers.0.ffn.gate.bias_vl",
        attn_owns_gate=True,
    ) == frozenset({"ffn"})


@pytest.mark.parametrize("stage", ["ffn", "mlp"])
def test_async_shared_experts_belong_to_the_payload_owner(stage: str) -> None:
    name = f"model.layers.0.{stage}.shared_experts.gate_proj.weight"

    assert _checkpoint_weight_roles(
        name,
        attention_shared_experts=True,
    ) == frozenset({"attention"})
    assert _checkpoint_weight_roles(name) == frozenset({"ffn"})


def test_unclassified_layer_paths_are_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match="unclassified DeepSeek-V4.1 checkpoint weight",
    ):
        _checkpoint_weight_roles("model.layers.0.unexpected.weight")
