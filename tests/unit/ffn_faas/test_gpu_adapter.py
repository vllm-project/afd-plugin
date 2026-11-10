# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU DeepEP M2N launch validation happens on the CPU, before any device."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from afd_plugin.ffn_faas.platforms import get_platform_adapter
from afd_plugin.ffn_faas.service import FFNService
from tests.unit.fakes.afdep import install_fake_afdep
from tests.unit.fakes.ffn_faas import write_qwen3_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_MODULES = ("torch", "vllm", "afdep", "deep_gemm")
GRAPH = ("--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
# Hopper, the oldest architecture the DeepEP M2N connector accepts.
SM90_CAPABILITY = (9, 0)


class DeviceSelectionReachedError(Exception):
    """Stops ``prepare`` at the moment the F rank binds its GPU."""

    def __init__(self, device) -> None:
        super().__init__(device)
        self.device = device


def _afd_config(**overrides) -> str:
    afd = {
        "role": "ffn",
        "connector": "DeepEPM2NAFDConnector",
        "port": 6239,
        "num_attention_ranks": 2,
        "num_ffn_ranks": 2,
        "compute_gate_on_attention": True,
        **overrides,
    }
    return json.dumps({"afd": afd})


@pytest.fixture
def launch_env(monkeypatch):
    """A two-rank torchrun environment for F rank 1."""
    for key, value in {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def configure(tmp_path):
    """Parse a native ``serve`` command line and configure the GPU adapter."""
    pytest.importorskip("vllm")
    from afd_plugin.ffn_faas.__main__ import create_parser

    model = tmp_path / "model"
    model.mkdir()

    def run(*options, afd=None, data_parallel_size="2"):
        adapter = get_platform_adapter("gpu-deepep-m2n")
        args = create_parser(adapter).parse_args(
            [
                "serve",
                str(model),
                "--backend",
                "gpu-deepep-m2n",
                "--data-parallel-size",
                data_parallel_size,
                "--enable-expert-parallel",
                "--additional-config",
                _afd_config(**(afd or {})),
                *options,
            ]
        )
        return adapter.configure(args), args

    return run


def test_backend_selection_imports_no_native_stack():
    script = (
        "import json, sys; "
        "from afd_plugin.ffn_faas.platforms import get_platform_adapter; "
        "get_platform_adapter('gpu-deepep-m2n'); "
        f"print(json.dumps([m for m in {NATIVE_MODULES!r} if m in sys.modules]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_valid_launch_preserves_the_model_path_and_describes_the_rank(
    launch_env, configure, tmp_path
):
    launch, args = configure(*GRAPH, "--enable-dbo")

    assert (launch.rank, launch.local_rank) == (1, 1)
    assert launch.model_path == str((tmp_path / "model").resolve())
    assert launch.backend_name == "DeepEPM2NAFDConnector"
    assert launch.initial_status == {
        "payload_format": "bf16",
        "num_lanes": 2,
        "graph_mode": "FULL_DECODE_ONLY",
        "dbo": True,
        "m2n_paired": False,
    }
    # The engine arguments vLLM builds the F-only config from.
    assert args.model == launch.model_path
    assert args.data_parallel_rank == 1
    assert args.distributed_executor_backend == "external_launcher"
    assert args.skip_tokenizer_init and args.enable_ep_weight_filter


def test_eager_launch_runs_one_lane(launch_env, configure):
    launch, _ = configure("--enforce-eager")

    assert launch.initial_status["graph_mode"] == "eager"
    assert launch.initial_status["num_lanes"] == 1
    assert launch.initial_status["dbo"] is False


@pytest.mark.parametrize(
    "env, options, afd, message",
    [
        ({"WORLD_SIZE": "4"}, ("--enforce-eager",), {}, "must agree"),
        ({}, (), {}, "runs --enforce-eager or"),
        (
            {},
            ("--compilation-config", '{"cudagraph_mode":"FULL"}'),
            {},
            "runs --enforce-eager or",
        ),
        (
            {},
            ("--enforce-eager", "--tensor-parallel-size", "2"),
            {},
            "--tensor-parallel-size 1",
        ),
        (
            {},
            ("--enforce-eager",),
            {"connector": "P2pNcclAFDConnector"},
            "requires DeepEPM2NAFDConnector",
        ),
        (
            {},
            ("--enforce-eager",),
            {"connector_extra_config": {"num_lanes": 2}},
            "num_lanes",
        ),
        (
            {},
            ("--enforce-eager",),
            {"connector_extra_config": {"payload_format": "fp8_block128_fp32"}},
            "payload_format='bf16'",
        ),
    ],
    ids=[
        "world-size",
        "unset-graph-mode",
        "full-graph-mode",
        "tensor-parallel",
        "connector",
        "lanes-vs-dbo",
        "fp8-payload",
    ],
)
def test_invalid_launch_is_rejected(
    launch_env, configure, monkeypatch, env, options, afd, message
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        configure(*options, afd=afd)


def test_cli_reports_a_bad_launch_as_a_usage_error(tmp_path):
    pytest.importorskip("vllm")
    model = tmp_path / "model"
    model.mkdir()
    env = {**os.environ, "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "afd_plugin.ffn_faas",
            "serve",
            str(model),
            "--backend",
            "gpu-deepep-m2n",
            "--data-parallel-size",
            "2",
            "--enable-expert-parallel",
            "--enforce-eager",
            "--additional-config",
            _afd_config(),
            "--status-port",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "must agree" in result.stderr


def test_device_ids_place_the_rank_on_its_assigned_gpu(
    launch_env, configure, monkeypatch, tmp_path
):
    """``--device-ids 2,3`` puts F local rank 1 on GPU 3, as vLLM's Worker does.

    vLLM only records the resolved IDs in ``parallel_config``; its GPU Worker
    registers them with ``set_assigned_physical_gpu_ids`` before mapping the
    local rank to a device. The F rank must apply the same mapping, or it lands
    on ``cuda:<LOCAL_RANK>``, which may be an Attention GPU. The device itself
    is never touched: binding it stops ``prepare``.
    """
    torch = pytest.importorskip("torch")
    platform_interface = pytest.importorskip("vllm.platforms.interface")
    install_fake_afdep()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # The mapping is process-wide in vLLM; start and end without one.
    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: SM90_CAPABILITY
    )

    def stop_at_device_selection(device):
        raise DeviceSelectionReachedError(device)

    monkeypatch.setattr(torch.accelerator, "set_device_index", stop_at_device_selection)
    write_qwen3_checkpoint(tmp_path / "model")
    launch, _ = configure(
        "--enforce-eager",
        "--device-ids",
        "2,3",
        afd={"connector_extra_config": {"expert_backend": "fused_experts"}},
    )

    backend = launch.backend_factory()
    with pytest.raises(DeviceSelectionReachedError) as selected:
        backend.prepare(Mock(spec=FFNService))

    assert selected.value.device == torch.device("cuda", 3)
