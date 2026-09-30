# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import tests.conftest as conftest
from tests.conftest import REPO_ROOT, RUNNER_CLEANUP_TIMEOUT_S, run_runner
from tests.e2e import runner
from tests.e2e.accuracy import gsm8k as helpers_gsm8k
from tests.e2e.models.deepseek_v2_lite import (
    test_deepseek_v2_lite as deepseek_v2_lite_e2e,
)
from tests.e2e.models.deepseek_v4_flash import config as dsv4_config
from tests.e2e.models.qwen3_6 import test_qwen3_6 as qwen3_6_e2e
from tests.e2e.models.qwen3_moe import test_qwen3_moe as qwen3_moe_e2e


def test_baseline_entrypoint_uses_four_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = deepseek_v2_lite_e2e.build_runner_command(
        "baseline-graph",
        tmp_path,
    )

    assert command[command.index("--attention-devices") + 1] == "2,4,6,8"
    assert "--ffn-devices" not in command


@pytest.mark.parametrize(
    "scenario", ["afd-eager-2a1f", "afd-graph-2a1f", "afd-graph-dbo-2a1f"]
)
def test_afd_entrypoint_ignores_the_fourth_device(monkeypatch, tmp_path, scenario):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = deepseek_v2_lite_e2e.build_runner_command(scenario, tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4"
    assert command[command.index("--ffn-devices") + 1] == "6"


@pytest.mark.parametrize(
    "scenario",
    ["afd-eager-2a2f", "afd-graph-2a2f", "afd-graph-dbo-2a2f"],
)
def test_afd_2a2f_entrypoint_uses_all_four_devices(monkeypatch, tmp_path, scenario):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = deepseek_v2_lite_e2e.build_runner_command(scenario, tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4"
    assert command[command.index("--ffn-devices") + 1] == "6,8"


@pytest.mark.parametrize("scenario", deepseek_v2_lite_e2e.SCENARIOS)
def test_deepseek_v2_lite_entrypoint_limits_context(
    monkeypatch,
    tmp_path,
    scenario,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")
    monkeypatch.delenv("AFD_E2E_DEVICES", raising=False)

    command = deepseek_v2_lite_e2e.build_runner_command(scenario, tmp_path)

    assert "--common-vllm-arg=--max-model-len=4096" in command


def test_qwen3_moe_baseline_entrypoint_uses_four_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = qwen3_moe_e2e.build_runner_command("baseline-graph", tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4,6,8"
    assert "--ffn-devices" not in command


@pytest.mark.parametrize("scenario", qwen3_moe_e2e.SCENARIOS)
def test_qwen3_moe_entrypoint_limits_context(monkeypatch, tmp_path, scenario):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")
    monkeypatch.delenv("AFD_E2E_DEVICES", raising=False)

    command = qwen3_moe_e2e.build_runner_command(scenario, tmp_path)

    assert "--common-vllm-arg=--max-model-len=4096" in command


@pytest.mark.parametrize("scenario", qwen3_moe_e2e.SCENARIOS[1:])
def test_qwen3_moe_afd_entrypoint_uses_only_the_standard_2a1f_devices(
    monkeypatch,
    tmp_path,
    scenario,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = qwen3_moe_e2e.build_runner_command(scenario, tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4"
    assert command[command.index("--ffn-devices") + 1] == "6"


def test_qwen3_6_baseline_entrypoint_uses_four_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = qwen3_6_e2e.build_runner_command("baseline-graph", tmp_path)

    assert command[command.index("--attention-devices") + 1] == "2,4,6,8"
    assert "--ffn-devices" not in command
    assert "--common-vllm-arg=--language-model-only" in command


@pytest.mark.parametrize("scenario", qwen3_6_e2e.SCENARIOS[1:])
def test_qwen3_6_afd_entrypoint_uses_standard_2a1f_text_only_commands(
    monkeypatch,
    tmp_path,
    scenario,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_E2E_DEVICES", "2,4,6,8")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    command = qwen3_6_e2e.build_runner_command(scenario, tmp_path)

    assert command[command.index("--model") + 1] == "model"
    assert command[command.index("--attention-devices") + 1] == "2,4"
    assert command[command.index("--ffn-devices") + 1] == "6"
    assert command[command.index("--served-model-name-prefix") + 1] == "qwen3-6-afd"
    assert "--common-vllm-arg=--max-model-len=4096" in command
    assert "--common-vllm-arg=--language-model-only" in command
    assert "--compute-gate-on-attention" not in command


def test_qwen3_6_entrypoint_rejects_non_gpu_backends(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    with pytest.raises(RuntimeError, match="supports only the 'gpu' backend"):
        qwen3_6_e2e.build_runner_command("afd-eager-2a1f", tmp_path)


@pytest.mark.parametrize(
    ("first_suite", "first_model"),
    [
        (deepseek_v2_lite_e2e, "/models/deepseek-v2-lite"),
        (qwen3_moe_e2e, "/models/qwen3-moe"),
    ],
)
def test_generated_gpu_model_is_scoped_to_its_e2e_suite(
    monkeypatch,
    tmp_path,
    first_suite,
    first_model,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.delenv("AFD_GPU_E2E_MODEL", raising=False)
    monkeypatch.setattr(first_suite, "download_dataset", lambda *_args: None)
    monkeypatch.setattr(
        first_suite,
        "download_model",
        lambda _repo_id: Path(first_model),
    )

    first_fixture = first_suite._prepare_e2e_assets.__wrapped__()
    next(first_fixture)
    assert os.environ["AFD_GPU_E2E_MODEL"] == first_model

    with pytest.raises(StopIteration):
        next(first_fixture)
    assert "AFD_GPU_E2E_MODEL" not in os.environ

    second_model = "/models/qwen3-6"
    monkeypatch.setattr(qwen3_6_e2e, "download_dataset", lambda *_args: None)
    monkeypatch.setattr(
        qwen3_6_e2e,
        "download_model",
        lambda _repo_id: Path(second_model),
    )
    second_fixture = qwen3_6_e2e._prepare_e2e_assets.__wrapped__()
    next(second_fixture)
    try:
        command = qwen3_6_e2e.build_runner_command("afd-eager-2a1f", tmp_path)
        assert command[command.index("--model") + 1] == second_model
    finally:
        with pytest.raises(StopIteration):
            next(second_fixture)

    assert "AFD_GPU_E2E_MODEL" not in os.environ


@pytest.mark.parametrize(
    ("suite", "backend", "model_env_name"),
    [
        (deepseek_v2_lite_e2e, "gpu", "AFD_GPU_E2E_MODEL"),
        (deepseek_v2_lite_e2e, "npu", "AFD_NPU_E2E_MODEL"),
        (qwen3_moe_e2e, "gpu", "AFD_GPU_E2E_MODEL"),
        (qwen3_6_e2e, "gpu", "AFD_GPU_E2E_MODEL"),
    ],
)
def test_e2e_fixture_restores_explicit_model_override(
    monkeypatch,
    suite,
    backend,
    model_env_name,
):
    explicit_model = "/models/user-selected"
    monkeypatch.setenv("AFD_E2E_BACKEND", backend)
    monkeypatch.setenv(model_env_name, explicit_model)
    monkeypatch.setattr(suite, "download_dataset", lambda *_args: None)
    monkeypatch.setattr(
        suite,
        "download_model",
        lambda _repo_id: pytest.fail("explicit model must not download"),
    )

    fixture = suite._prepare_e2e_assets.__wrapped__()
    next(fixture)
    assert os.environ[model_env_name] == explicit_model

    with pytest.raises(StopIteration):
        next(fixture)
    assert os.environ[model_env_name] == explicit_model


def test_run_runner_forwards_cancellation_and_reaps(monkeypatch):
    command = [sys.executable, "-m", "tests.e2e.runner"]
    previous_handlers = {
        signal.SIGTERM: object(),
        signal.SIGINT: object(),
    }
    installed_handlers: dict[int, Callable[[int, object], None]] = {}
    signal_calls = []
    kill_calls = []
    popen_calls = []

    class FakeProcess:
        pid = 321
        returncode = None
        wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
                pytest.fail("the first cancellation signal must unwind the call")
            if len(self.wait_calls) == 2:
                installed_handlers[signal.SIGINT](signal.SIGINT, None)
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def fake_popen(actual_command, **kwargs):
        popen_calls.append((actual_command, kwargs))
        return process

    def fake_signal(signum, handler):
        signal_calls.append((signum, handler))
        installed_handlers[signum] = handler

    monkeypatch.setattr(conftest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        conftest.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(conftest.signal, "signal", fake_signal)
    monkeypatch.setattr(
        conftest.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(conftest.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(SystemExit) as error:
        run_runner(command)

    assert error.value.code == 128 + signal.SIGTERM
    assert popen_calls == [
        (
            command,
            {
                "cwd": REPO_ROOT,
                "env": None,
                "start_new_session": True,
            },
        ),
    ]
    assert kill_calls == [(321, signal.SIGTERM), (321, 9)]
    assert process.wait_calls == [None, RUNNER_CLEANUP_TIMEOUT_S, None]
    assert signal_calls[-2:] == list(previous_handlers.items())


def test_run_runner_forwards_signal_received_during_spawn(monkeypatch):
    handlers: dict[int, Callable[[int, object], None]] = {}
    forwarded = []

    class FakeProcess:
        pid = 321

        def poll(self):
            return 0

    def fake_popen(*_args, **_kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return FakeProcess()

    monkeypatch.setattr(conftest.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(conftest.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(
        conftest.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        conftest.os,
        "killpg",
        lambda pid, sig: forwarded.append((pid, sig)),
        raising=False,
    )

    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
        run_runner(["runner"])

    assert forwarded == [(321, signal.SIGTERM)]


def test_run_runner_fails_for_a_nonzero_exit(monkeypatch):
    command = [sys.executable, "-m", "tests.e2e.runner"]

    class FailedProcess:
        pid = 321
        returncode = 17

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        conftest.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(conftest.signal, "signal", lambda *_args: None)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_runner(command)

    assert error.value.returncode == 17


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="deepseek-ai/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="127.0.0.1",
        afd_port=1239,
        served_model_name_prefix="deepseek-v2-lite-afd",
        num_attention_ranks=2,
        num_ffn_ranks=1,
        attention_devices="0,1",
        ffn_devices="2",
        device_backend="gpu",
        tp_size=1,
        attention_tp_size=None,
        ffn_tp_size=None,
        scenario="afd-eager-2a1f",
        baseline=False,
        cuda_graph_full_decode_only=False,
        cudagraph_capture_size=64,
        enable_dbo=False,
        dbo_decode_token_threshold=1,
        dbo_prefill_token_threshold=None,
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_connector_extra_config=[],
        use_decode_bench_connector=False,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        gsm8k_output_path="/tmp/gsm8k-results",
        completion_output_path="/tmp/dsv4-completions.json",
        use_v2_model_runner=False,
    )


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--num-attention-ranks", "4"],
        ["--num-ffn-ranks", "2"],
        ["--tp-size", "2"],
        ["--attention-tp-size", "2"],
        ["--ffn-tp-size", "2"],
        ["--cuda-graph-full-decode-only"],
        ["--cudagraph-capture-size", "16"],
        ["--enable-dbo"],
        ["--dbo-decode-token-threshold", "2"],
        ["--dbo-prefill-token-threshold", "16"],
        ["--prompt", "Hello"],
        ["--max-tokens", "32"],
        ["--temperature", "0.5"],
        ["--num-requests", "2"],
        ["--request-concurrency", "2"],
        ["--expect-text", "answer"],
    ],
)
def test_parse_args_rejects_legacy_fixed_scenario_options(monkeypatch, legacy_args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model",
            "model",
            "--scenario",
            "afd-eager-2a1f",
            "--gsm8k-output-path",
            "results",
            *legacy_args,
        ],
    )

    with pytest.raises(SystemExit) as error:
        runner.parse_args()

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("baseline-graph", (True, True, False, 4, 0, 1, 1, 1, False)),
        ("afd-eager-2a1f", (False, False, False, 2, 1, 1, 1, 1, False)),
        ("afd-graph-2a1f", (False, True, False, 2, 1, 1, 1, 1, False)),
        ("afd-graph-dbo-2a1f", (False, True, True, 2, 1, 1, 1, 1, False)),
        ("afd-eager-2a2f", (False, False, False, 2, 2, 1, 1, 1, False)),
        ("afd-graph-2a2f", (False, True, False, 2, 2, 1, 1, 1, False)),
        ("afd-graph-dbo-2a2f", (False, True, True, 2, 2, 1, 1, 1, False)),
        ("afd-eager-async-cam", (False, False, False, 2, 2, 1, 2, 1, False)),
        ("afd-async-ubatch", (False, False, False, 2, 1, 1, 2, 1, False)),
        (runner.DSV4_ASYNC_CAM_SCENARIO, (False, False, False, 8, 8, 1, 4, 1, False)),
        (
            dsv4_config.DSV4_SYNC_CAMP2P_A5_SCENARIO,
            # The A5 profile runs Attention DP2/TP1 and FFN DP2/TP1 with ACL
            # graph capture like its launch script, but without that script's
            # native DBO, whose split path is the current suspect for the A5 DSA
            # attention operator tiling failure.
            (False, True, False, 2, 2, 1, 1, 1, False),
        ),
        (
            dsv4_config.DSV4_SYNC_CAMP2P_A3_SCENARIO,
            # Sixteen dies: Attention DP2/TP4 and FFN DP8/TP1 with expert
            # parallelism, the shape its recorded deployment uses.
            (False, False, False, 8, 8, 1, 4, 1, True),
        ),
        ("afd-v2-eager-1a1f", (False, False, False, 1, 1, 1, 1, 1, True)),
        ("afd-v2-eager-dp2", (False, False, False, 2, 2, 1, 1, 1, True)),
        ("afd-v2-eager-tp2", (False, False, False, 2, 2, 1, 2, 2, True)),
        ("afd-v2-graph-1a1f", (False, True, False, 1, 1, 1, 1, 1, True)),
        ("afd-v2-graph-dp2", (False, True, False, 2, 2, 1, 1, 1, True)),
        ("afd-v2-graph-tp2", (False, True, False, 2, 2, 1, 2, 2, True)),
    ],
)
def test_configure_scenario_overwrites_fixed_topology_and_features(
    scenario,
    expected,
):
    args = _args()
    args.scenario = scenario
    args.num_attention_ranks = 99
    args.num_ffn_ranks = 99
    args.tp_size = 99
    args.cuda_graph_full_decode_only = False
    args.enable_dbo = False

    runner.configure_scenario(args)

    assert (
        args.baseline,
        args.cuda_graph_full_decode_only,
        args.enable_dbo,
        args.num_attention_ranks,
        args.num_ffn_ranks,
        args.tp_size,
        args.attention_tp_size,
        args.ffn_tp_size,
        args.use_v2_model_runner,
    ) == expected
    # A scenario that carries its own launch profile (the DSV4 synchronous
    # cases) uses that profile's graph setting instead of the default.
    profile = dsv4_config.sync_shape(scenario)
    if args.cuda_graph_full_decode_only:
        expected_capture_size = (
            profile.cudagraph_capture_size
            if profile is not None
            else runner.DEFAULT_CUDAGRAPH_CAPTURE_SIZE
        )
        assert args.cudagraph_capture_size == expected_capture_size
    if args.enable_dbo:
        assert args.dbo_decode_token_threshold == 1
        assert args.dbo_prefill_token_threshold == 8


def test_async_cam_scenario_builds_dp1tp2_attention_and_dp2tp1_ffn():
    args = _args()
    args.scenario = "afd-eager-async-cam"
    args.device_backend = "npu"
    runner.configure_scenario(args)
    runner.validate_topology(args, ["0", "1"], ["2", "3"])

    attention_command = runner.build_vllm_command(args, role="attention")
    ffn_command = runner.build_vllm_command(args, role="ffn")

    assert attention_command[attention_command.index("--data-parallel-size") + 1] == "1"
    assert (
        attention_command[attention_command.index("--tensor-parallel-size") + 1] == "2"
    )
    assert ffn_command[ffn_command.index("--data-parallel-size") + 1] == "2"
    assert ffn_command[ffn_command.index("--tensor-parallel-size") + 1] == "1"
    attention_config = json.loads(
        attention_command[attention_command.index("--additional-config") + 1],
    )["afd"]
    assert attention_config["connector"] == runner.ASYNC_AFD_CONNECTOR
    assert attention_config["async"] is True
    assert attention_config["compute_gate_on_attention"] is True
    assert attention_config["connector_extra_config"]["attn_ranks_per_dp"] == 2
    assert "--enable-expert-parallel" in ffn_command


def test_async_ubatch_scenario_enforces_token_split_moe_ubatching():
    args = _args()
    args.scenario = "afd-async-ubatch"
    args.device_backend = "npu"
    args.afd_connector_extra_config = [
        json.dumps(
            {
                "attn_ranks_per_dp": 1,
                "async_moe_ubatching": False,
                "async_moe_num_ubatches": 99,
                "async_moe_split": "request",
            },
        ),
    ]
    runner.configure_scenario(args)
    runner.validate_topology(args, ["0", "1"], ["2"])

    attention_command = runner.build_vllm_command(args, role="attention")

    assert attention_command[attention_command.index("--data-parallel-size") + 1] == "1"
    assert (
        attention_command[attention_command.index("--tensor-parallel-size") + 1] == "2"
    )
    attention_config = json.loads(
        attention_command[attention_command.index("--additional-config") + 1],
    )["afd"]
    assert attention_config["connector"] == runner.ASYNC_AFD_CONNECTOR
    assert attention_config["async"] is True
    assert attention_config["compute_gate_on_attention"] is True
    extra_config = attention_config["connector_extra_config"]
    assert extra_config["async_moe_ubatching"] is True
    assert extra_config["async_moe_num_ubatches"] == 2
    assert extra_config["async_moe_split"] == "token"
    assert extra_config["attn_ranks_per_dp"] == 2


@pytest.mark.parametrize(
    ("scenario", "expected_dp_size", "expected_tp_size", "uses_graph"),
    [
        ("afd-v2-eager-1a1f", "1", "1", False),
        ("afd-v2-eager-dp2", "2", "1", False),
        ("afd-v2-eager-tp2", "1", "2", False),
        ("afd-v2-graph-1a1f", "1", "1", True),
        ("afd-v2-graph-dp2", "2", "1", True),
        ("afd-v2-graph-tp2", "1", "2", True),
    ],
)
def test_v2_scenarios_build_exact_commands(
    scenario,
    expected_dp_size,
    expected_tp_size,
    uses_graph,
):
    args = _args()
    args.scenario = scenario
    runner.configure_scenario(args)

    for role in ("attention", "ffn"):
        command = runner.build_vllm_command(args, role=role)
        assert command[command.index("--data-parallel-size") + 1] == expected_dp_size
        assert command[command.index("--tensor-parallel-size") + 1] == expected_tp_size
        assert ("--enforce-eager" in command) is not uses_graph
        assert ("--compilation-config" in command) is uses_graph
        assert "--enable-dbo" not in command
        assert "--max-num-batched-tokens" not in command
        assert "--no-enable-prefix-caching" in command
        assert "--no-enable-chunked-prefill" in command
        assert "--no-async-scheduling" in command


@pytest.mark.parametrize(
    ("field", "value", "error_message"),
    [
        ("afd_async", True, "synchronous AFD"),
        ("compute_gate_on_attention", True, "compute_gate_on_attention=false"),
        ("afd_connector", "OtherConnector", "P2pNcclAFDConnector"),
        ("afd_connector_extra_config", ['{"mode":"unsupported"}'], "extra config"),
        ("use_decode_bench_connector", True, "decode-bench connector"),
    ],
)
def test_v2_scenarios_reject_unsupported_options(field, value, error_message):
    args = _args()
    args.scenario = "afd-v2-eager-1a1f"
    setattr(args, field, value)

    with pytest.raises(ValueError, match=error_message):
        runner.configure_scenario(args)


def test_v2_scenarios_reject_npu_backend():
    args = _args()
    args.scenario = "afd-v2-eager-1a1f"
    args.device_backend = "npu"
    runner.configure_scenario(args)

    with pytest.raises(ValueError, match="require GPU"):
        runner.validate_topology(args, ["0"], ["1"])


def test_build_baseline_command_uses_native_dp4_graph_server():
    args = _args()
    args.scenario = "baseline-graph"
    runner.configure_scenario(args)

    command = runner.build_baseline_command(args)

    assert "--additional-config" not in command
    assert command[command.index("--data-parallel-size") + 1] == "4"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert "--enable-expert-parallel" in command
    assert json.loads(command[command.index("--compilation-config") + 1]) == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }


def test_build_baseline_command_configures_graceful_shutdown_timeout():
    args = _args()
    args.scenario = "baseline-graph"
    runner.configure_scenario(args)

    command = runner.build_baseline_command(args)

    shutdown_timeout_index = command.index("--shutdown-timeout")
    assert command[shutdown_timeout_index + 1] == "10"


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_build_vllm_command_configures_graceful_shutdown_timeout(role):
    args = _args()
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role=role)

    shutdown_timeout_index = command.index("--shutdown-timeout")
    assert command[shutdown_timeout_index + 1] == "10"


@pytest.mark.parametrize(
    "passthrough_arg",
    ["--additional-config", '--additional-config={"afd":{}}'],
)
def test_build_baseline_command_rejects_additional_config_passthrough(
    passthrough_arg,
):
    args = _args()
    args.scenario = "baseline-graph"
    args.common_vllm_arg = [passthrough_arg]
    runner.configure_scenario(args)

    with pytest.raises(ValueError, match="--additional-config"):
        runner.build_baseline_command(args)


def test_validate_topology_accepts_four_baseline_ranks_without_ffn_ranks():
    args = _args()
    args.scenario = "baseline-graph"
    runner.configure_scenario(args)

    runner.validate_topology(args, ["0", "1", "2", "3"], [])


@pytest.mark.parametrize(
    "scenario",
    [runner.ASYNC_CAM_SCENARIO, runner.ASYNC_UBATCH_SCENARIO],
)
def test_validate_topology_rejects_async_cam_scenarios_on_gpu(scenario):
    args = _args()
    args.scenario = scenario
    args.device_backend = "gpu"
    runner.configure_scenario(args)

    ffn_devices = ["2", "3"] if scenario == runner.ASYNC_CAM_SCENARIO else ["2"]
    with pytest.raises(ValueError, match="require NPU"):
        runner.validate_topology(args, ["0", "1"], ffn_devices)


@pytest.mark.parametrize(
    ("attention_devices", "ffn_devices", "error_message"),
    [
        (["0"], ["2"], "--attention-devices must contain exactly 2 devices"),
        (["0", "1"], [], "--ffn-devices must contain exactly 1 device"),
    ],
)
def test_validate_topology_rejects_wrong_device_count(
    attention_devices,
    ffn_devices,
    error_message,
):
    args = _args()
    runner.configure_scenario(args)

    with pytest.raises(ValueError, match=error_message):
        runner.validate_topology(args, attention_devices, ffn_devices)


@pytest.mark.parametrize(
    ("attention_devices", "ffn_devices", "error_message"),
    [
        (["0", "0"], ["2"], "Attention devices must be unique"),
        (["0", "1"], ["2", "2"], "FFN devices must be unique"),
        (["0", "1"], ["1"], "Attention and FFN devices must not overlap"),
    ],
)
def test_validate_topology_rejects_reused_devices(
    attention_devices,
    ffn_devices,
    error_message,
):
    args = _args()

    with pytest.raises(ValueError, match=error_message):
        runner.validate_topology(args, attention_devices, ffn_devices)


@pytest.mark.parametrize(
    ("scenario", "backend", "expected_plugins"),
    [
        ("baseline-graph", "gpu", ""),
        ("baseline-graph", "npu", "ascend"),
        ("afd-eager-2a1f", "gpu", "afd"),
        ("afd-eager-2a1f", "npu", "ascend,afd"),
        ("afd-v2-eager-1a1f", "gpu", "afd"),
    ],
)
def test_build_env_uses_the_scenario_plugin_allowlist(
    scenario,
    backend,
    expected_plugins,
):
    args = _args()
    args.scenario = scenario
    args.device_backend = backend
    runner.configure_scenario(args)

    env = runner.build_env("0", args, role="baseline" if args.baseline else "attention")

    assert env["VLLM_PLUGINS"] == expected_plugins


@pytest.mark.parametrize(
    ("backend", "visible_devices_env"),
    [
        ("gpu", "CUDA_VISIBLE_DEVICES"),
        ("npu", "ASCEND_RT_VISIBLE_DEVICES"),
    ],
)
def test_print_command_reports_the_actual_visible_devices_environment(
    backend,
    visible_devices_env,
    capsys,
):
    runner.print_command("ATTN", ["vllm", "serve"], backend, "0,1")

    assert f"({visible_devices_env}=0,1)" in capsys.readouterr().out


def test_extract_gsm8k_sample_count_returns_effective_count():
    assert (
        helpers_gsm8k._extract_gsm8k_sample_count(
            {"n-samples": {"gsm8k": {"effective": 7}}},
        )
        == 7
    )


@pytest.mark.parametrize(
    "results",
    [
        {},
        {"n-samples": {"gsm8k": {"effective": "not-a-number"}}},
        {"n-samples": {"gsm8k": {"effective": 7.5}}},
    ],
)
def test_extract_gsm8k_sample_count_rejects_missing_or_malformed_results(results):
    with pytest.raises((KeyError, ValueError), match="GSM8K sample count"):
        helpers_gsm8k._extract_gsm8k_sample_count(results)


def test_run_lm_eval_uses_builtin_gsm8k_and_inherits_hf_home(
    monkeypatch,
    tmp_path,
):
    popen_calls = []
    hf_home = str(tmp_path / "huggingface")
    monkeypatch.setenv("HF_HOME", hf_home)

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        raise RuntimeError("stop after inspecting lm-eval invocation")

    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="stop after inspecting"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
        )

    command, kwargs = popen_calls[0]
    assert command[command.index("--tasks") + 1] == "gsm8k"
    assert "--include_path" not in command
    assert "--limit" not in command
    assert kwargs["env"]["HF_HOME"] == hf_home


def test_run_lm_eval_passes_max_gen_toks_to_local_completions(
    monkeypatch,
    tmp_path,
):
    popen_calls = []

    def fake_popen(command, **_kwargs):
        popen_calls.append(command)
        raise RuntimeError("stop after inspecting lm-eval invocation")

    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="stop after inspecting"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
            max_gen_toks=321,
        )

    command = popen_calls[0]
    model_args = command[command.index("--model_args") + 1]
    assert "max_gen_toks=321" in model_args.split(",")


def test_run_lm_eval_passes_num_concurrent_to_local_completions(
    monkeypatch,
    tmp_path,
):
    popen_calls = []

    def fake_popen(command, **_kwargs):
        popen_calls.append(command)
        raise RuntimeError("stop after inspecting lm-eval invocation")

    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="stop after inspecting"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
            num_concurrent=12,
        )

    command = popen_calls[0]
    model_args = command[command.index("--model_args") + 1]
    assert "num_concurrent=12" in model_args.split(",")


def test_run_lm_eval_reads_timestamped_results_file(monkeypatch, tmp_path):
    output_path = tmp_path / "results"
    model_output_path = output_path / "deepseek-v2-lite-afd-attention"
    model_output_path.mkdir(parents=True)
    expected_results = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.25}},
        "n-samples": {"gsm8k": {"original": 1319, "effective": 7}},
    }
    (model_output_path / "results_2026-08-07T11-16-36.json").write_text(
        json.dumps(expected_results),
    )

    class CompletedProcess:
        pid = 404
        stdout = io.StringIO(
            "| strict-match | 5 | exact_match | 0.25 | +/- | 0.0 |\n",
        )
        returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        helpers_gsm8k.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletedProcess(),
    )

    results = helpers_gsm8k._run_lm_eval(
        "http://127.0.0.1:8000",
        "model",
        output_path=str(output_path),
    )

    assert results == expected_results


def test_run_lm_eval_custom_timeout_cleans_its_group_reaps_and_joins(
    monkeypatch,
    tmp_path,
):
    popen_calls = []
    group_alive = True
    leader_signal_calls = []
    group_signal_calls = []
    reader_join_calls = []

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")
        returncode = None
        direct_kill_calls = 0
        wait_calls = []

        def kill(self):
            self.direct_kill_calls += 1

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = -9
            return self.returncode

    class FakeReader:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            reader_join_calls.append(timeout)

        def is_alive(self):
            return False

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    def fake_kill(pid, sig):
        leader_signal_calls.append((pid, sig))

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            group_alive = False

    monotonic_values = iter(
        [100.0, 111.0, 8000.0, 8021.0, 9000.0, 9000.0],
    )
    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", FakeReader)
    monkeypatch.setattr(
        helpers_gsm8k.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(helpers_gsm8k, "signal", signal, raising=False)
    monkeypatch.setattr(helpers_gsm8k.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(helpers_gsm8k.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(TimeoutError, match="lm-eval exceeded 10s budget") as error:
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
            timeout_s=10,
        )

    assert popen_calls[0][1]["start_new_session"] is True
    assert process.direct_kill_calls == 0
    assert leader_signal_calls == [(404, signal.SIGTERM)]
    assert group_signal_calls == [
        (404, 0),
        (404, 9),
        (404, 0),
    ]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "forced SIGKILL" in str(error.value.__cause__)
    assert process.wait_calls == [helpers_gsm8k.LM_EVAL_REAP_TIMEOUT_S]
    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_run_lm_eval_rejects_a_reader_that_does_not_join(monkeypatch, tmp_path):
    reader_join_calls = []

    class CompletedProcess:
        pid = 404
        stdout = io.StringIO(
            '{"results":{"gsm8k":{"exact_match":1.0}}}\n',
        )
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    class StuckReader:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            reader_join_calls.append(timeout)

        def is_alive(self):
            return True

    monkeypatch.setattr(
        helpers_gsm8k.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletedProcess(),
    )
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", StuckReader)
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: 100.0)

    with pytest.raises(RuntimeError, match="reader thread"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
        )

    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_run_lm_eval_defers_a_first_signal_until_cleanup_finishes(
    monkeypatch,
    tmp_path,
):
    events = []
    installed_handlers = {}
    group_alive = True

    def previous_term_handler(signum, _frame):
        events.append(("delegate", signum))
        raise SystemExit(128 + signum)

    previous_handlers = {
        signal.SIGTERM: previous_term_handler,
        signal.SIGINT: signal.SIG_IGN,
    }

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            raise RuntimeError("reap failed")

    class FakeReader:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            events.append(("reader-join", timeout))

        def is_alive(self):
            return False

    def fake_signal(signum, handler):
        installed_handlers[signum] = handler
        action = "restore" if handler is previous_handlers[signum] else "install"
        events.append((action, signum))

    def fake_kill(_pid, sig):
        nonlocal group_alive
        events.append(("cleanup-term", sig))
        group_alive = False
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)

    def fake_killpg(_pid, sig):
        events.append(("probe", sig))
        if not group_alive:
            raise ProcessLookupError

    process = FakeProcess()
    monotonic_values = iter([100.0, 111.0, 8000.0, 9000.0, 9000.0])
    monkeypatch.setattr(
        helpers_gsm8k.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(helpers_gsm8k.signal, "signal", fake_signal)
    monkeypatch.setattr(
        helpers_gsm8k.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", FakeReader)
    monkeypatch.setattr(helpers_gsm8k.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(
        helpers_gsm8k.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(SystemExit) as error:
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
            timeout_s=10,
        )

    assert error.value.code == 128 + signal.SIGTERM
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "reap failed" in str(error.value.__cause__)
    assert events == [
        ("install", signal.SIGTERM),
        ("install", signal.SIGINT),
        ("cleanup-term", signal.SIGTERM),
        ("probe", 0),
        ("wait", helpers_gsm8k.LM_EVAL_REAP_TIMEOUT_S),
        ("reader-join", helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S),
        ("restore", signal.SIGTERM),
        ("restore", signal.SIGINT),
        ("delegate", signal.SIGTERM),
    ]


def test_run_lm_eval_handles_signal_received_during_spawn(monkeypatch, tmp_path):
    handlers: dict[int, Callable[[int, object], None]] = {}
    cleaned = []

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")

    class FakeReader:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    def previous_handler(signum, _frame):
        raise SystemExit(128 + signum)

    def fake_popen(*_args, **_kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return FakeProcess()

    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", FakeReader)
    monkeypatch.setattr(
        helpers_gsm8k.signal,
        "getsignal",
        lambda _signum: previous_handler,
    )
    monkeypatch.setattr(
        helpers_gsm8k.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        helpers_gsm8k,
        "_finish_lm_eval_cleanup",
        lambda process, *_args, **_kwargs: cleaned.append(process.pid),
    )

    with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path),
            timeout_s=0,
        )

    assert cleaned == [404]


@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_run_gsm8k_evaluation_uses_the_builtin_task_and_default_limit(
    monkeypatch,
    backend,
):
    args = _args()
    args.device_backend = backend
    calls = []
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)

    def fake_run_lm_eval(base_url, model_name, **kwargs):
        calls.append((base_url, model_name, kwargs))
        return {
            "n-samples": {"gsm8k": {"effective": 7}},
            "results": {"gsm8k": {"exact_match": 0.27}},
        }

    monkeypatch.setattr(runner, "_run_lm_eval", fake_run_lm_eval)

    runner.run_gsm8k_evaluation(args)

    assert calls == [
        (
            "http://127.0.0.1:18100",
            "deepseek-v2-lite-afd-attention",
            {
                "output_path": "/tmp/gsm8k-results",
                "num_fewshot": 8,
                "tokenizer": "deepseek-ai/DeepSeek-V2-Lite",
                "limit": 7,
            },
        ),
    ]


def test_run_gsm8k_evaluation_runs_the_full_dataset_when_requested(
    monkeypatch,
):
    args = _args()
    calls = []
    monkeypatch.setenv("AFD_GSM8K_LIMIT", "all")

    def fake_run_lm_eval(base_url, model_name, **kwargs):
        calls.append((base_url, model_name, kwargs))
        return {
            "n-samples": {"gsm8k": {"effective": 1319}},
            "results": {"gsm8k": {"exact_match": 0.27}},
        }

    monkeypatch.setattr(runner, "_run_lm_eval", fake_run_lm_eval)

    runner.run_gsm8k_evaluation(args)

    assert calls == [
        (
            "http://127.0.0.1:18100",
            "deepseek-v2-lite-afd-attention",
            {
                "output_path": "/tmp/gsm8k-results",
                "num_fewshot": 8,
                "tokenizer": "deepseek-ai/DeepSeek-V2-Lite",
                "limit": None,
                "timeout_s": 28800,
            },
        ),
    ]


def test_run_gsm8k_evaluation_uses_batch_two_for_async_token_split(
    monkeypatch,
):
    args = _args()
    args.scenario = runner.ASYNC_UBATCH_SCENARIO
    args.device_backend = "npu"
    runner.configure_scenario(args)
    calls = []
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)

    def fake_run_lm_eval(base_url, model_name, **kwargs):
        calls.append((base_url, model_name, kwargs))
        return {
            "n-samples": {"gsm8k": {"effective": 7}},
            "results": {"gsm8k": {"exact_match": 0.27}},
        }

    monkeypatch.setattr(runner, "_run_lm_eval", fake_run_lm_eval)

    runner.run_gsm8k_evaluation(args)

    assert calls[0][2]["batch_size"] == 2


def test_run_gsm8k_evaluation_concurrency_and_sample_floor_for_dbo(monkeypatch):
    args = _args()
    args.scenario = "afd-graph-dbo-2a1f"
    args.device_backend = "gpu"
    runner.configure_scenario(args)
    calls = []
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)

    def fake_run_lm_eval(base_url, model_name, **kwargs):
        calls.append((base_url, model_name, kwargs))
        return {
            "n-samples": {"gsm8k": {"effective": runner.DBO_EVAL_MIN_SAMPLES}},
            "results": {"gsm8k": {"exact_match": 0.27}},
        }

    monkeypatch.setattr(runner, "_run_lm_eval", fake_run_lm_eval)

    runner.run_gsm8k_evaluation(args)

    assert calls[0][2]["num_concurrent"] == runner.DBO_EVAL_NUM_CONCURRENT
    assert calls[0][2]["limit"] == runner.DBO_EVAL_MIN_SAMPLES


def test_run_gsm8k_evaluation_rejects_an_incomplete_full_dataset(
    monkeypatch,
):
    args = _args()
    monkeypatch.setenv("AFD_GSM8K_LIMIT", "all")
    monkeypatch.setattr(
        runner,
        "_run_lm_eval",
        lambda *_args, **_kwargs: {
            "n-samples": {"gsm8k": {"effective": 1318}},
            "results": {"gsm8k": {"exact_match": 1.0}},
        },
    )

    with pytest.raises(RuntimeError, match="expected 1319"):
        runner.run_gsm8k_evaluation(args)


def test_run_gsm8k_evaluation_rejects_a_non_seven_sample_result(
    monkeypatch,
):
    args = _args()
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)
    monkeypatch.setattr(
        runner,
        "_run_lm_eval",
        lambda *_args, **_kwargs: {
            "n-samples": {"gsm8k": {"effective": 6}},
            "results": {"gsm8k": {"exact_match": 1.0}},
        },
    )

    with pytest.raises(RuntimeError, match="expected 7"):
        runner.run_gsm8k_evaluation(args)


@pytest.mark.parametrize(
    "accuracy",
    [0.26, pytest.param(float("nan"), id="nan")],
)
def test_run_gsm8k_evaluation_rejects_accuracy_that_does_not_meet_gate(
    monkeypatch,
    accuracy,
):
    args = _args()
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)
    monkeypatch.setattr(
        runner,
        "_run_lm_eval",
        lambda *_args, **_kwargs: {
            "n-samples": {"gsm8k": {"effective": 7}},
            "results": {"gsm8k": {"exact_match": accuracy}},
        },
    )

    with pytest.raises(RuntimeError, match="below the required threshold"):
        runner.run_gsm8k_evaluation(args)


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({"choices": [{"text": "B"}]}, None),
        ({"choices": []}, "no choices"),
        ({"choices": [{"text": ""}]}, "no text"),
        ({"choices": [{"text": 32}]}, "no text"),
    ],
)
def test_run_completion_evaluation_validates_one_32_token_request(
    monkeypatch,
    body,
    error,
):
    args = _args()
    args.scenario = "afd-eager-async-cam"
    runner.configure_scenario(args)
    request = {}

    def fake_urlopen(http_request, timeout):
        request.update(json.loads(http_request.data))
        request["timeout"] = timeout
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    if error:
        with pytest.raises(RuntimeError, match=error):
            runner.run_completion_evaluation(args)
    else:
        runner.run_completion_evaluation(args)
        assert request == {
            "model": "deepseek-v2-lite-afd-attention",
            "prompt": runner.ACCOUNTING_PROMPT,
            "max_tokens": 32,
            "temperature": 0,
            "timeout": 120,
        }


def test_ensure_processes_alive_reports_exited_process_returncode():
    process = argparse.Namespace(poll=lambda: 17)

    with pytest.raises(RuntimeError, match="returncode=17"):
        runner.ensure_processes_alive([process])  # type: ignore[list-item]


def test_terminate_processes_rejects_cleanup_failure(monkeypatch):
    monkeypatch.setattr(
        runner,
        "terminate_process_groups",
        lambda *_args, **_kwargs: ["cleanup failed"],
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.terminate_processes([])


@pytest.mark.parametrize("native_handler", [False, True])
def test_main_checks_processes_and_restores_signal_handlers_when_cleanup_fails(
    monkeypatch,
    capsys,
    native_handler,
):
    args = _args()
    args.attention_devices = "0"
    args.ffn_devices = "1"
    args.afd_connector = None
    process = argparse.Namespace(poll=lambda: None)
    checked_processes = []
    installed_handlers = []
    previous_handlers = {
        signal.SIGTERM: None if native_handler else "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }

    class FakeLogThread:
        joined = False

        def join(self, timeout):
            self.joined = True

    log_thread = FakeLogThread()

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "validate_topology", lambda *_args: None)
    monkeypatch.setattr(runner, "build_vllm_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "build_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "start_process", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "stream_output", lambda *_args: log_thread)
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(runner, "run_gsm8k_evaluation", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "ensure_processes_alive",
        lambda processes: checked_processes.append(list(processes)),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup failed"),
        ),
    )
    monkeypatch.setattr(
        runner.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(
        runner.signal,
        "signal",
        lambda signum, handler: installed_handlers.append((signum, handler)),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.main()

    assert checked_processes == [[process, process]] * 2
    assert log_thread.joined is True
    assert installed_handlers[2:] == [
        (signum, signal.SIG_DFL if handler is None else handler)
        for signum, handler in previous_handlers.items()
    ]
    assert "E2E SCENARIO" not in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_error:
        installed_handlers[0][1](signal.SIGTERM, None)
    assert exit_error.value.code == 128 + signal.SIGTERM


def test_main_defers_a_first_signal_until_cleanup_failure_is_reported(monkeypatch):
    args = _args()
    args.attention_devices = "0"
    args.ffn_devices = "1"
    args.afd_connector = None
    process = argparse.Namespace(poll=lambda: None)
    installed_handlers = {}
    cleanup_events = []
    previous_handlers = {
        signal.SIGTERM: "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }

    class FakeLogThread:
        joined = False

        def join(self, timeout):
            self.joined = True

    log_thread = FakeLogThread()

    def fake_signal(signum, handler):
        installed_handlers[signum] = handler

    def cleanup_with_first_signal(_processes, **_kwargs):
        cleanup_events.append("started")
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        cleanup_events.append("completed")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "validate_topology", lambda *_args: None)
    monkeypatch.setattr(runner, "build_vllm_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "build_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "start_process", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "stream_output", lambda *_args: log_thread)
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(runner, "run_gsm8k_evaluation", lambda *_args: None)
    monkeypatch.setattr(runner, "terminate_processes", cleanup_with_first_signal)
    monkeypatch.setattr(
        runner.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 128 + signal.SIGTERM
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "cleanup failed"
    assert cleanup_events == ["started", "completed"]
    assert log_thread.joined is True
    assert installed_handlers == previous_handlers


def test_runner_drops_flashcomm_for_npu_role_without_tp(monkeypatch):
    args = _args()
    args.device_backend = "npu"
    args.ffn_tp_size = 1
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

    env = runner.build_env("2,3", args, role="ffn")

    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env


def test_build_env_marks_managed_process_trees_by_role():
    args = _args()
    args.device_backend = "npu"

    attention_env = runner.build_env(
        "0,1",
        args,
        role="attention",
        e2e_run_id="run-1",
    )
    ffn_env = runner.build_env(
        "2,3",
        args,
        role="ffn",
        e2e_run_id="run-1",
    )

    assert attention_env[runner.E2E_RUN_ID_ENV] == "run-1"
    assert attention_env[runner.E2E_PROCESS_ROLE_ENV] == "attention"
    assert ffn_env[runner.E2E_RUN_ID_ENV] == "run-1"
    assert ffn_env[runner.E2E_PROCESS_ROLE_ENV] == "ffn"


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [("afd-eager-2a1f", "0"), ("afd-v2-eager-1a1f", "1")],
)
def test_runner_selects_gpu_model_runner_from_scenario(
    monkeypatch,
    scenario,
    expected,
):
    args = _args()
    args.scenario = scenario
    runner.configure_scenario(args)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    env = runner.build_env("0,1", args, role="attention")

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == expected


@pytest.mark.parametrize(
    ("scenario", "attention_devices", "ffn_devices"),
    [
        ("afd-v2-eager-1a1f", "0", "1"),
        ("afd-v2-eager-dp2", "0,1", "2,3"),
        ("afd-v2-eager-tp2", "0,1", "2,3"),
    ],
)
def test_v2_model_entry_builds_runner_command_with_exact_devices(
    monkeypatch,
    tmp_path,
    scenario,
    attention_devices,
    ffn_devices,
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "/models/DeepSeek-V2-Lite")
    monkeypatch.delenv("AFD_E2E_DEVICES", raising=False)

    command = deepseek_v2_lite_e2e.build_runner_command(scenario, tmp_path)

    assert command[command.index("--attention-devices") + 1] == attention_devices
    assert command[command.index("--ffn-devices") + 1] == ffn_devices


def test_configure_scenario_separates_prefill_and_decode_steps_for_dbo():
    args = _args()
    args.scenario = "afd-graph-dbo-2a1f"
    args.common_vllm_arg = ["--unrelated-arg"]
    runner.configure_scenario(args)

    assert "--no-enable-chunked-prefill" in args.common_vllm_arg

    args.scenario = "afd-graph-2a1f"
    args.common_vllm_arg = ["--unrelated-arg"]
    runner.configure_scenario(args)

    assert "--no-enable-chunked-prefill" not in args.common_vllm_arg


def test_build_env_enables_debug_logging_for_dbo_scenarios(monkeypatch):
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)

    args = _args()
    args.scenario = "afd-graph-dbo-2a1f"
    runner.configure_scenario(args)
    dbo_env = runner.build_env("0,1", args, role="attention")

    assert dbo_env["VLLM_LOGGING_LEVEL"] == "DEBUG"

    args.scenario = "afd-graph-2a1f"
    runner.configure_scenario(args)
    plain_env = runner.build_env("0,1", args, role="attention")

    assert "VLLM_LOGGING_LEVEL" not in plain_env


def test_stream_output_records_attention_split_steps(monkeypatch):
    split_steps: list[float] = []
    process: Any = argparse.Namespace(
        stdout=io.StringIO(
            "DEBUG [gpu_model_runner.py] ubatch_slices: [UBatchSlice(...)], ...\n"
            "DEBUG [gpu_model_runner.py] ubatch_slices: None, ...\n"
            "INFO serving\n",
        ),
    )
    timestamps = iter([101.0, 102.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(timestamps))

    thread = runner.stream_output("attention", process, split_steps)
    thread.join(timeout=5)

    assert split_steps == [101.0]


def test_assert_dbo_live_split_coverage_passes_when_steps_in_window(monkeypatch):
    args = _args()
    args.device_backend = "gpu"
    monkeypatch.setattr(runner.time, "time", lambda: 102.0)

    runner.assert_dbo_live_split_coverage([101.0, 101.4], 100.0, args)


def test_assert_dbo_live_split_coverage_fails_on_warmup_only(monkeypatch):
    args = _args()
    args.device_backend = "gpu"
    monkeypatch.setattr(runner.time, "time", lambda: 102.0)

    with pytest.raises(RuntimeError, match="no live request was ever split"):
        runner.assert_dbo_live_split_coverage([99.0], 100.0, args)


def test_assert_dbo_live_split_coverage_fails_without_evidence(monkeypatch):
    args = _args()
    args.device_backend = "gpu"
    monkeypatch.setattr(runner.time, "time", lambda: 102.0)

    with pytest.raises(RuntimeError, match="no live request was ever split"):
        runner.assert_dbo_live_split_coverage([], 100.0, args)
