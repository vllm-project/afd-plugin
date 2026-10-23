# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe session and public work admission contracts."""

import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest

from afd_plugin.ffn_faas.protocol import MAX_GENERATION, SessionSpec, SourceChunkPlan


@pytest.fixture
def spec() -> SessionSpec:
    return SessionSpec("checkpoint-sha", 64, 32, 4, 2, 4, 2, 2, 5, (1, 4), capacity=8)


@pytest.mark.parametrize(
    "changes",
    [
        {"attention_tp": 3},
        {"ffn_ranks": 3},
        {"hidden_size": 65},
        {"dtype": "float16"},
        {"slots": 0},
        {"capacity": 0},
        {"layer_indices": (4, 1)},
        {"layer_indices": (1, 5)},
        {"model_identity": ""},
        {"abi_version": 99},
    ],
)
def test_incompatible_session_is_rejected(spec: SessionSpec, changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(spec, **changes)


@pytest.mark.parametrize(
    ("sequence", "layer", "tokens"),
    [(0, 1, 1), (MAX_GENERATION + 1, 1, 1), (1, 0, 1), (1, 1, 9), (1, 1, -1)],
)
def test_invalid_work_is_rejected(
    spec: SessionSpec, sequence: int, layer: int, tokens: int
) -> None:
    with pytest.raises(ValueError):
        spec.validate_work(sequence, layer, tokens)


def test_zero_work_and_final_generation_are_admitted(spec: SessionSpec) -> None:
    spec.validate_work(1, 1, 0)
    spec.validate_work(MAX_GENERATION, 4, 8)


def test_tp_chunk_shape_uses_longest_member_and_keeps_zero_members() -> None:
    plan = SourceChunkPlan(3, 5, 1, 1)
    peer = SourceChunkPlan(3, 5, 1, 2)
    assert SourceChunkPlan.from_json(plan.to_json()).agree([peer]) == 2
    assert plan.agree([plan]) == 1


@pytest.mark.parametrize(
    "peer",
    [
        SourceChunkPlan(4, 5, 1, 2),
        SourceChunkPlan(3, 6, 1, 2),
        SourceChunkPlan(3, 5, 4, 2),
        SourceChunkPlan(3, 5, 1, 0),
    ],
)
def test_tp_chunk_shape_rejects_wrong_call_identity(peer: SourceChunkPlan) -> None:
    with pytest.raises(ValueError):
        SourceChunkPlan(3, 5, 1, 1).agree([peer])


def test_tp_chunk_shape_cannot_wrap_generation() -> None:
    with pytest.raises(ValueError, match="generation"):
        SourceChunkPlan(1, MAX_GENERATION, 1, 2).agree([])


@pytest.mark.parametrize(
    "changes",
    [{"slots": 3}, {"layer_indices": (1, 3)}, {"model_identity": "other-sha"}],
)
def test_wire_capability_change_changes_identity(
    spec: SessionSpec, changes: dict
) -> None:
    assert replace(spec, **changes).fingerprint != spec.fingerprint


def test_native_window_covers_all_participants_and_slots(spec: SessionSpec) -> None:
    # The external ABI uses 64-byte lines, 192-byte input headers and
    # 128-byte result headers. The fixed 4A2F fixture needs 10,944 bytes on F.
    assert spec.required_window_bytes == 10944


def test_contract_import_needs_no_device_or_engine_runtime() -> None:
    script = textwrap.dedent("""
        import builtins
        original_import = builtins.__import__
        def isolated_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}:
                raise AssertionError('device or engine import: ' + name)
            return original_import(name, *args, **kwargs)
        builtins.__import__ = isolated_import
        from afd_plugin.compat.npu.ops import bootstrap_afd_custom_op_env
        bootstrap_afd_custom_op_env()
        from afd_plugin.ffn_faas import SessionSpec
        from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription
        from afd_plugin.ffn_faas.v4_runtime import V4FFNModel
        from afd_plugin.ffn_faas.service import FFNService
        from afd_plugin.ffn_faas.runner import FFNModelRunner
        from afd_plugin.ffn_faas.__main__ import main
        assert SessionSpec.__name__ == 'SessionSpec'
        assert V4FFNModel.__name__ == 'V4FFNModel'
    """)
    subprocess.run([sys.executable, "-c", script], check=True)
