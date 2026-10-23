# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4 model metadata/lifecycle boundaries without device or runtime imports.

The geometry is the actual Flash checkpoint's. These cases allocate no tensor
weights and do not establish native sharding, packing or group teardown.
"""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from afd_plugin.ffn_faas.v4_model import V4CheckpointDescription
from afd_plugin.ffn_faas.v4_runtime import V4FFNModel, validate_native_expert_map


@pytest.fixture
def checkpoint(tmp_path: Path) -> V4CheckpointDescription:
    # Raw-index completeness is covered by test_v4_model. This fixture isolates
    # the runtime's metadata seam and never attempts to load its empty index.
    return V4CheckpointDescription(
        root=tmp_path,
        identity="sha256:v4-metadata-fixture",
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=256,
        top_k=6,
        model_layers=43,
        hash_layers=3,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        weight_map=MappingProxyType({}),
        routed_weight_map=MappingProxyType({}),
        metadata_sha256=MappingProxyType({}),
    )


@pytest.fixture
def spec(checkpoint):
    return checkpoint.session_spec(
        attention_ranks=12, attention_tp=4, ffn_ranks=4, capacity=8, slots=2
    )


def test_native_packed_storage_estimate_includes_retained_modelslim_parameters(
    checkpoint, spec
):
    model = V4FFNModel(checkpoint, spec, 0)
    # EP4: 32.25 GiB routed INT4 plus encoded scales/offsets/compensation.
    assert model.parameter_bytes == 34_988_883_968


def test_explicit_first_layer_probe_estimates_only_the_selected_weights(
    checkpoint, spec
):
    first_layer = replace(spec, layer_indices=(0,))
    assert V4FFNModel(checkpoint, first_layer, 0).parameter_bytes == 813_694_976


@pytest.mark.parametrize(
    "changes",
    [
        {"model_identity": "sha256:different-checkpoint"},
        {"weight_quantization": "none", "swiglu_limit": 0.0},
        {"hidden_size": 2048},
        {"swiglu_limit": 0.0},
    ],
)
def test_model_rejects_a_session_from_a_different_checkpoint_contract(
    checkpoint, spec, changes
):
    with pytest.raises(ValueError, match="does not match the W4A8 checkpoint"):
        V4FFNModel(checkpoint, replace(spec, **changes), 0)


@pytest.mark.parametrize("rank", [-1, 4])
def test_model_rejects_rank_outside_the_native_loading_world(checkpoint, spec, rank):
    with pytest.raises(ValueError, match="outside the F-only loading world"):
        V4FFNModel(checkpoint, spec, rank)


def test_unloaded_model_remains_closeable_without_importing_a_runtime(checkpoint, spec):
    model = V4FFNModel(checkpoint, spec, 0)
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="have not been loaded"):
        _ = model.native_view


@pytest.mark.parametrize("world,rank", [(16, 0), (4, 1)])
def test_loading_rejects_a_world_other_than_the_external_f_launcher(
    checkpoint, spec, monkeypatch, world, rank
):
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("RANK", str(rank))
    model = V4FFNModel(checkpoint, spec, 0)
    with pytest.raises(ValueError, match="matching F-only world/rank"):
        model.load(device=0)
    model.close()


def test_loading_requires_external_launcher_metadata(checkpoint, spec, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    with pytest.raises(ValueError, match="external F-only torchrun launcher"):
        V4FFNModel(checkpoint, spec, 0).load(device=0)


@pytest.mark.parametrize("rank", range(4))
def test_native_linear_placement_matches_the_service_expert_ownership(spec, rank):
    first = rank * spec.local_experts
    mapping = [-1] * spec.num_experts
    mapping[first : first + spec.local_experts] = range(spec.local_experts)
    assert validate_native_expert_map(spec, rank, mapping) == frozenset(
        range(first, first + spec.local_experts)
    )


@pytest.mark.parametrize("placement", ["round-robin", "local-permutation"])
def test_a_native_bijection_cannot_silently_change_wire_expert_placement(
    spec, placement
):
    mapping = [-1] * spec.num_experts
    if placement == "round-robin":
        for local in range(spec.local_experts):
            mapping[local * spec.ffn_ranks] = local
    else:
        mapping[: spec.local_experts] = range(spec.local_experts)
        mapping[0], mapping[1] = mapping[1], mapping[0]
    with pytest.raises(ValueError, match="contiguous service routing"):
        validate_native_expert_map(spec, 0, mapping)


@pytest.mark.parametrize("malformation", ["missing-expert", "duplicate-local-id"])
def test_invalid_native_expert_maps_fail_before_source_tensor_loading(
    spec, malformation
):
    mapping = [-1] * spec.num_experts
    mapping[: spec.local_experts] = range(spec.local_experts)
    if malformation == "missing-expert":
        mapping.pop()
    else:
        mapping[1] = mapping[0]
    with pytest.raises(ValueError, match="expert count|local bijection"):
        validate_native_expert_map(spec, 0, mapping)


def test_runtime_metadata_import_has_no_tensor_or_model_runtime_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import afd_plugin.ffn_faas.v4_runtime; "
            "assert {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}"
            ".isdisjoint(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
