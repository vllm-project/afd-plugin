# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone launch configuration preserves the existing CAM contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from afd_plugin.ffn_faas.platforms.npu.adapter import AscendCAMAdapter
from afd_plugin.ffn_faas.platforms.npu.settings import (
    FFNConfig,
    validate_native_expert_map,
    validate_static_experts,
)
from tests.unit.fakes.ffn_faas import cam_cli_args


@pytest.fixture
def settings():
    return FFNConfig(Path("/model"), ffn_ranks=4)


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


@pytest.mark.parametrize("positional", [False, True])
def test_engine_configuration_keeps_user_values(checkpoint_dir, launcher, positional):
    args = cam_cli_args(checkpoint_dir)
    if positional:
        args.model = "/unused-option-model"
        args.model_tag = str(checkpoint_dir)
    launch = AscendCAMAdapter().configure(args)
    assert (launch.rank, launch.local_rank) == (1, 1)
    assert launch.model_path == str(checkpoint_dir)
    assert args.model == str(checkpoint_dir)
    assert args.data_parallel_rank == 1
    assert args.additional_config["enable_cpu_binding"] is False
    assert args.additional_config["afd"]["connector_extra_config"] == {
        "dynamicQuant": 1,
        "attn_ranks_per_dp": 4,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"data_parallel_size": 1},
        {"data_parallel_rank": 0},
        {"distributed_executor_backend": "mp"},
        {"receiver_shutdown_timeout": 0},
    ],
)
def test_launcher_rejects_inconsistent_engine_arguments(
    checkpoint_dir, launcher, change
):
    with pytest.raises(ValueError):
        AscendCAMAdapter().configure(cam_cli_args(checkpoint_dir, **change))


@pytest.mark.parametrize(
    "field,value", [("port", 0), ("role", "attention"), ("num_ffn_ranks", 0)]
)
def test_cli_uses_existing_afd_validation(checkpoint_dir, launcher, field, value):
    args = cam_cli_args(checkpoint_dir)
    args.additional_config["afd"][field] = value
    with pytest.raises(ValueError):
        AscendCAMAdapter().configure(args)


@pytest.mark.parametrize(
    "change",
    [
        {"ffn_ranks": 0},
        {"rank": 4},
        {"local_rank": -1},
        {"shutdown_timeout": 0},
        {"shutdown_timeout": float("nan")},
        {"shutdown_timeout": float("inf")},
    ],
)
def test_invalid_host_launch_rejected(settings, change):
    with pytest.raises(ValueError):
        replace(settings, **change)


@pytest.mark.parametrize(
    "rank,entries,owned",
    [
        (0, [0, 1, -1, -1], {0, 1}),
        (1, [-1, -1, 0, 1], {2, 3}),
    ],
)
def test_native_expert_map_matches_cam(rank, entries, owned):
    assert (
        validate_native_expert_map(
            num_experts=4, ffn_ranks=2, ffn_rank=rank, expert_map=entries
        )
        == owned
    )


@pytest.mark.parametrize(
    "entries",
    [
        [0, 0, -1, -1],
        [-1, -1, 0, 1],
        [1, 0, -1, -1],
        [0, 1],
    ],
)
def test_native_expert_map_mismatch_rejected(entries):
    with pytest.raises(ValueError):
        validate_native_expert_map(
            num_experts=4, ffn_ranks=2, ffn_rank=0, expert_map=entries
        )


@pytest.mark.parametrize("ffn_ranks", [1, 2])
def test_implicit_native_map_only_when_all_experts_are_local(ffn_ranks):
    if ffn_ranks == 1:
        assert validate_native_expert_map(
            num_experts=4, ffn_ranks=1, ffn_rank=0, expert_map=None
        ) == frozenset(range(4))
    else:
        with pytest.raises(ValueError, match="explicit map"):
            validate_native_expert_map(
                num_experts=4, ffn_ranks=2, ffn_rank=0, expert_map=None
            )


@pytest.mark.parametrize(
    "eplb,elastic,redundant,dynamic",
    [
        (True, False, 0, False),
        (False, True, 0, False),
        (False, False, 1, False),
        (False, False, 0, True),
    ],
)
def test_static_cam_adapter_rejects_expert_relocation(
    eplb, elastic, redundant, dynamic
):
    with pytest.raises(ValueError, match="static expert placement"):
        validate_static_experts(
            enable_eplb=eplb,
            enable_elastic_ep=elastic,
            redundant_experts=redundant,
            dynamic_eplb=dynamic,
        )
