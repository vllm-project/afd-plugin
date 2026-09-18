# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the DSV4 NPU role-aware checkpoint filter.

``_checkpoint_weight_roles`` decides which AFD role receives each checkpoint
path. The distinction is not cosmetic: the upstream Ascend loader indexes its
parameter dict by name without a membership check, so a path handed to a role
that never registered it raises ``KeyError`` during weight loading instead of
being skipped.

The Hash id table is the case that matters. Both roles register their own copy
whenever they build a router, so the table follows the same ownership rule as
the other gate paths rather than being pinned to FFN.

How the functions are loaded
----------------------------
The NPU DSV4 module defines these helpers next to its model classes, and
importing it pulls in the whole ``vllm``, ``vllm_ascend`` and ``transformers``
class surface. Stubbing that surface would make this file break on every
upstream addition and would test the stubs as much as the code.

Instead the two helpers are read from the module source and executed in an
isolated namespace. They are module-level functions that reference nothing but
each other and three role constants, so this exercises the real code from the
real file without importing it.
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
    / "deepseek_v4.py"
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
        "_BOTH_ROLES": frozenset(("attention", "ffn")),
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

    module = ModuleType("dsv4_role_helpers")
    module.__dict__.update(namespace)
    return module


_helpers = _load_helpers()
_checkpoint_weight_roles = _helpers._checkpoint_weight_roles  # type: ignore[attr-defined]


def test_module_and_helpers_are_present() -> None:
    assert _MODULE_PATH.is_file()


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.tid2eid",
        "model.layers.7.ffn.gate.tid2eid",
    ],
)
def test_hash_id_table_follows_the_gate_ownership(name: str) -> None:
    """The Hash table belongs to every role that built a router.

    With the gate on Attention, the Attention gate shell registers ``tid2eid``
    for its Hash layers and routes from it, so the table is shared. With the gate
    on FFN the Attention MoE slot is parameter-free, so the path is FFN-only;
    handing it to Attention raises ``KeyError`` from the upstream loader.
    """

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
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.ffn.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ],
)
def test_gate_paths_skip_attention_when_the_gate_is_on_ffn(name: str) -> None:
    """With the gate on FFN the Attention MoE slot is parameter-free.

    This is the supported CAMP2P configuration, and handing Attention these paths
    raises ``KeyError: 'model.layers.0.mlp.gate.weight'`` because it registered
    no gate at all.
    """

    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=False,
    ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ],
)
def test_gate_paths_stay_shared_when_attention_owns_the_gate(name: str) -> None:
    """Gate-on-Attention builds a router there, so both roles load it."""

    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=True,
    ) == frozenset({"attention", "ffn"})


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
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.ffn.shared_experts.gate_proj.weight",
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
        "model.embed_tokens.weight",
        "model.layers.0.hc_attn_fn",
        "lm_head.weight",
    ],
)
def test_shared_and_non_layer_paths_are_shared(name: str) -> None:
    for attn_owns_gate in (True, False):
        assert _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
        ) == frozenset({"attention", "ffn"})
