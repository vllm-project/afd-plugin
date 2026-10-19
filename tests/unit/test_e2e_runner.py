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
from typing import Any, cast

import pytest

import tests.conftest as conftest
from tests.conftest import REPO_ROOT, RUNNER_CLEANUP_TIMEOUT_S, run_runner
from tests.e2e import runner
from tests.e2e.accuracy import gsm8k as helpers_gsm8k
from tests.e2e.models.deepseek_v2_lite import (
    test_deepseek_v2_lite as deepseek_v2_lite_e2e,
)
from tests.e2e.models.qwen3_6 import test_qwen3_6 as qwen3_6_e2e
from tests.e2e.models.qwen3_moe import test_qwen3_moe as qwen3_moe_e2e

GPU_MODEL_SUITES = {
    "deepseek-v2-lite": deepseek_v2_lite_e2e,
    "qwen3-moe": qwen3_moe_e2e,
    "qwen3-6": qwen3_6_e2e,
}


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
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


# --- Model E2E entrypoints ---------------------------------------------------


@pytest.mark.parametrize(
    ("suite_name", "scenario", "devices", "attention_devices", "ffn_devices"),
    [
        ("deepseek-v2-lite", "baseline-graph", "2,4,6,8", "2,4,6,8", None),
        ("deepseek-v2-lite", "afd-eager-2a1f", "2,4,6,8", "2,4", "6"),
        ("deepseek-v2-lite", "afd-graph-dbo-2a2f", "2,4,6,8", "2,4", "6,8"),
        ("deepseek-v2-lite", "afd-v2-eager-1a1f", None, "0", "1"),
        ("deepseek-v2-lite", "afd-v2-eager-dp2", None, "0,1", "2,3"),
        ("deepseek-v2-lite", "afd-v2-graph-tp2", None, "0,1", "2,3"),
        ("qwen3-moe", "baseline-graph", "2,4,6,8", "2,4,6,8", None),
        ("qwen3-moe", "afd-graph-dbo-2a1f", "2,4,6,8", "2,4", "6"),
        ("qwen3-6", "baseline-graph", "2,4,6,8", "2,4,6,8", None),
        ("qwen3-6", "afd-eager-2a1f", "2,4,6,8", "2,4", "6"),
    ],
)
def test_model_entrypoint_selects_role_devices(
    monkeypatch, tmp_path, suite_name, scenario, devices, attention_devices, ffn_devices
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")
    if devices is None:
        monkeypatch.delenv("AFD_E2E_DEVICES", raising=False)
    else:
        monkeypatch.setenv("AFD_E2E_DEVICES", devices)

    command = GPU_MODEL_SUITES[suite_name].build_runner_command(scenario, tmp_path)

    assert _option(command, "--attention-devices") == attention_devices
    if ffn_devices is None:
        assert "--ffn-devices" not in command
    else:
        assert _option(command, "--ffn-devices") == ffn_devices


@pytest.mark.parametrize(
    ("suite", "scenario"),
    [
        pytest.param(suite, scenario, id=f"{name}-{scenario}")
        for name, suite in GPU_MODEL_SUITES.items()
        for scenario in suite.SCENARIOS
    ],
)
def test_every_gpu_model_case_uses_v2_with_scoped_feature_evidence(
    monkeypatch, tmp_path, suite, scenario
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "gpu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")
    monkeypatch.delenv("AFD_E2E_DEVICES", raising=False)
    command = suite.build_runner_command(scenario, tmp_path)
    assert _option(command, "--model") == "model"
    assert "--common-vllm-arg=--max-model-len=4096" in command
    if suite is qwen3_6_e2e:
        assert "--common-vllm-arg=--language-model-only" in command
        assert _option(command, "--served-model-name-prefix") == "qwen3-6-afd"
        assert "--compute-gate-on-attention" not in command

    monkeypatch.setattr(sys, "argv", ["runner", *command[3:]])
    args = runner.parse_args()
    runner.configure_scenario(args)
    runner.validate_topology(
        args,
        runner.parse_csv(args.attention_devices),
        runner.parse_csv(args.ffn_devices),
    )
    for role in ("baseline",) if args.baseline else ("attention", "ffn"):
        service_command = (
            runner.build_baseline_command(args)
            if role == "baseline"
            else runner.build_vllm_command(args, role=role)
        )
        expected_probe = runner.needs_mrv2_execution_evidence(args)
        assert ("--worker-extension-cls" in service_command) is expected_probe
        if expected_probe:
            assert (
                _option(service_command, "--worker-extension-cls")
                == "tests.e2e.mrv2_evidence.Worker"
            )
        env = runner.build_env("0", args, role=role)
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        if expected_probe:
            assert env["VLLM_SERVER_DEV_MODE"] == "1"


def test_qwen3_6_entrypoint_rejects_non_gpu_backends(monkeypatch, tmp_path):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "model")

    with pytest.raises(RuntimeError, match="supports only the 'gpu' backend"):
        qwen3_6_e2e.build_runner_command("afd-eager-2a1f", tmp_path)


@pytest.mark.parametrize(
    ("backend", "scenario", "limit", "override", "expected"),
    [
        ("gpu", "afd-graph-dbo-2a2f", None, None, "0.25"),
        ("gpu", "afd-graph-dbo-2a2f", "24", "0.30", "0.30"),
        ("gpu", "afd-graph-dbo-2a2f", "256", None, None),
        ("npu", "afd-graph-dbo-2a2f", "24", None, "0.25"),
        ("gpu", "afd-v2-graph-dp2", None, None, "0.25"),
        ("gpu", "afd-graph-dbo-2a2f", "all", None, None),
        ("gpu", "afd-v2-eager-dp2", None, None, None),
        ("gpu", "afd-v2-eager-dbo-dp2", None, None, None),
        ("gpu", "afd-v2-graph-dbo-dp2", None, None, None),
    ],
)
def test_deepseek_smoke_threshold_preserves_large_runs_and_overrides(
    monkeypatch, tmp_path, backend, scenario, limit, override, expected
):
    monkeypatch.setenv("AFD_E2E_BACKEND", backend)
    monkeypatch.setenv(f"AFD_{backend.upper()}_E2E_MODEL", "model")
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)
    if limit is not None:
        monkeypatch.setenv("AFD_GSM8K_LIMIT", limit)
    monkeypatch.delenv("AFD_GSM8K_THRESHOLD", raising=False)
    if override is not None:
        monkeypatch.setenv("AFD_GSM8K_THRESHOLD", override)

    def check_runner_env(command, env):
        assert env.get("AFD_GSM8K_THRESHOLD") == expected

    monkeypatch.setattr(deepseek_v2_lite_e2e, "run_runner", check_runner_env)
    deepseek_v2_lite_e2e.test_deepseek_v2_lite(scenario, tmp_path)
    assert os.environ.get("AFD_GSM8K_THRESHOLD") == override


@pytest.mark.parametrize(
    ("suite", "backend", "explicit_model"),
    [
        (deepseek_v2_lite_e2e, "gpu", None),
        (qwen3_moe_e2e, "gpu", None),
        (qwen3_6_e2e, "gpu", None),
        (deepseek_v2_lite_e2e, "gpu", "/models/user-selected"),
        (deepseek_v2_lite_e2e, "npu", "/models/user-selected"),
        (qwen3_moe_e2e, "gpu", "/models/user-selected"),
        (qwen3_6_e2e, "gpu", "/models/user-selected"),
    ],
)
def test_e2e_fixture_scopes_the_model_override_to_its_suite(
    monkeypatch, tmp_path, suite, backend, explicit_model
):
    model_env_name = f"AFD_{backend.upper()}_E2E_MODEL"
    downloaded_model = f"/models/{suite.__name__}"
    monkeypatch.setenv("AFD_E2E_BACKEND", backend)
    if explicit_model is None:
        monkeypatch.delenv(model_env_name, raising=False)
    else:
        monkeypatch.setenv(model_env_name, explicit_model)
    monkeypatch.setattr(suite, "download_dataset", lambda *_args: None)

    def download_model(_repo_id):
        if explicit_model is not None:
            pytest.fail("explicit model must not download")
        return Path(downloaded_model)

    monkeypatch.setattr(suite, "download_model", download_model)

    fixture = suite._prepare_e2e_assets.__wrapped__()
    next(fixture)
    expected_model = explicit_model or downloaded_model
    assert os.environ[model_env_name] == expected_model
    command = suite.build_runner_command("afd-eager-2a1f", tmp_path)
    assert _option(command, "--model") == expected_model

    with pytest.raises(StopIteration):
        next(fixture)
    assert os.environ.get(model_env_name) == explicit_model


# --- run_runner (tests/conftest.py) ------------------------------------------


def test_run_runner_forwards_cancellation_and_reaps(
    monkeypatch,
):  # long-test: signal delivery and process reaping form one lifecycle contract
    command = [sys.executable, "-m", "tests.e2e.runner"]
    previous_handlers = {signal.SIGTERM: object(), signal.SIGINT: object()}
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
        conftest.signal, "getsignal", lambda signum: previous_handlers[signum]
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
        (command, {"cwd": REPO_ROOT, "env": None, "start_new_session": True})
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
    class FailedProcess:
        pid = 321
        returncode = 17

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        conftest.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess()
    )
    monkeypatch.setattr(conftest.signal, "signal", lambda *_args: None)

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_runner([sys.executable, "-m", "tests.e2e.runner"])

    assert error.value.returncode == 17


# --- Runner scenario configuration and commands ------------------------------


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--num-attention-ranks", "4"],
        ["--tp-size", "2"],
        ["--enable-dbo"],
        ["--prompt", "Hello"],
        ["--expect-text", "answer"],
    ],
)
def test_parse_args_rejects_legacy_fixed_scenario_options(monkeypatch, legacy_args):
    argv = "runner --model model --scenario afd-eager-2a1f --gsm8k-output-path out"
    monkeypatch.setattr(sys, "argv", [*argv.split(), *legacy_args])

    with pytest.raises(SystemExit) as error:
        runner.parse_args()

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("baseline-graph", (True, True, False, 4, 0, 1, 1, 1)),
        ("afd-eager-2a1f", (False, False, False, 2, 1, 1, 1, 1)),
        ("afd-graph-2a1f", (False, True, False, 2, 1, 1, 1, 1)),
        ("afd-graph-dbo-2a1f", (False, True, True, 2, 1, 1, 1, 1)),
        ("afd-eager-2a2f", (False, False, False, 2, 2, 1, 1, 1)),
        ("afd-graph-2a2f", (False, True, False, 2, 2, 1, 1, 1)),
        ("afd-graph-dbo-2a2f", (False, True, True, 2, 2, 1, 1, 1)),
        ("afd-eager-async-cam", (False, False, False, 2, 2, 1, 2, 1)),
        ("afd-async-ubatch", (False, False, False, 2, 1, 1, 2, 1)),
    ],
)
def test_configure_scenario_overwrites_fixed_topology_and_features(scenario, expected):
    args = _args()
    args.scenario = scenario
    if scenario in (runner.ASYNC_CAM_SCENARIO, runner.ASYNC_UBATCH_SCENARIO):
        args.device_backend = "npu"
    args.num_attention_ranks = 99
    args.num_ffn_ranks = 99
    args.tp_size = 99

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
    ) == expected
    if args.cuda_graph_full_decode_only:
        assert args.cudagraph_capture_size == runner.DEFAULT_CUDAGRAPH_CAPTURE_SIZE
    if args.enable_dbo:
        # vLLM 0.30 requires DBO thresholds >= the microbatch count (2).
        assert args.dbo_decode_token_threshold == 2
        assert args.dbo_prefill_token_threshold == 8


@pytest.mark.parametrize(
    ("scenario", "ffn_devices", "extra_config", "expected_extra_config"),
    [
        (runner.ASYNC_CAM_SCENARIO, ["2", "3"], [], {"attn_ranks_per_dp": 2}),
        (
            runner.ASYNC_UBATCH_SCENARIO,
            ["2"],
            [
                json.dumps(
                    {
                        "attn_ranks_per_dp": 1,
                        "async_moe_ubatching": False,
                        "async_moe_num_ubatches": 99,
                        "async_moe_split": "request",
                    }
                )
            ],
            {
                "attn_ranks_per_dp": 2,
                "async_moe_ubatching": True,
                "async_moe_num_ubatches": 2,
                "async_moe_split": "token",
            },
        ),
    ],
)
def test_async_npu_scenarios_build_tp2_attention_with_async_connector(
    scenario, ffn_devices, extra_config, expected_extra_config
):
    args = _args()
    args.scenario = scenario
    args.device_backend = "npu"
    args.afd_connector_extra_config = extra_config
    runner.configure_scenario(args)
    runner.validate_topology(args, ["0", "1"], ffn_devices)

    attention_command = runner.build_vllm_command(args, role="attention")
    ffn_command = runner.build_vllm_command(args, role="ffn")

    assert _option(attention_command, "--data-parallel-size") == "1"
    assert _option(attention_command, "--tensor-parallel-size") == "2"
    assert _option(ffn_command, "--data-parallel-size") == str(len(ffn_devices))
    assert _option(ffn_command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in ffn_command
    config = json.loads(_option(attention_command, "--additional-config"))["afd"]
    assert config["connector"] == runner.ASYNC_AFD_CONNECTOR
    assert config["async"] is True
    assert config["compute_gate_on_attention"] is True
    assert config["connector_extra_config"].items() >= expected_extra_config.items()


@pytest.mark.parametrize(
    ("scenario", "expected_dp_size", "expected_tp_size", "uses_graph"),
    [
        ("afd-v2-eager-1a1f", "1", "1", False),
        ("afd-v2-eager-dp2", "2", "1", False),
        ("afd-v2-eager-dbo-dp2", "2", "1", False),
        ("afd-v2-graph-dbo-dp2", "2", "1", True),
        ("afd-v2-eager-tp2", "1", "2", False),
        ("afd-v2-graph-1a1f", "1", "1", True),
        ("afd-v2-graph-dp2", "2", "1", True),
        ("afd-v2-graph-tp2", "1", "2", True),
    ],
)
def test_v2_scenarios_build_exact_commands(
    scenario, expected_dp_size, expected_tp_size, uses_graph
):
    args = _args()
    args.scenario = scenario
    runner.configure_scenario(args)

    for role in ("attention", "ffn"):
        command = runner.build_vllm_command(args, role=role)
        assert _option(command, "--data-parallel-size") == expected_dp_size
        assert _option(command, "--tensor-parallel-size") == expected_tp_size
        assert _option(command, "--shutdown-timeout") == "10"
        assert ("--enforce-eager" in command) is not uses_graph
        assert ("--compilation-config" in command) is uses_graph
        assert ("--enable-dbo" in command) is ("-dbo-" in scenario)
        assert "--max-num-batched-tokens" not in command
        assert "--no-enable-prefix-caching" not in command
        assert "--no-enable-chunked-prefill" not in command
        assert "--no-async-scheduling" not in command
        assert ("--worker-extension-cls" in command) is (
            uses_graph or "-dbo-" in scenario
        )
        if scenario in runner.V2_DBO_COMPARISON_SCENARIOS:
            assert _option(command, "--max-num-seqs") == "8"
        if "-dbo-" in scenario:
            assert _option(command, "--dbo-decode-token-threshold") == "2"
            assert _option(command, "--dbo-prefill-token-threshold") == "8"
            if uses_graph:
                assert json.loads(_option(command, "--compilation-config")) == {
                    "cudagraph_mode": "FULL_DECODE_ONLY"
                }
                assert _option(command, "--cudagraph-capture-sizes") == "8"


def test_gpu_scenarios_leave_connector_contract_validation_to_runtime():
    args = _args()
    args.scenario = "afd-v2-eager-1a1f"
    args.afd_async = True
    args.compute_gate_on_attention = True
    args.afd_connector = "OtherConnector"
    args.afd_connector_extra_config = ['{"mode":"configured"}']
    args.use_decode_bench_connector = True

    runner.configure_scenario(args)
    command = runner.build_vllm_command(args, role="attention")

    config = json.loads(_option(command, "--additional-config"))["afd"]
    assert config["async"] is True
    assert config["compute_gate_on_attention"] is True
    assert config["connector"] == "OtherConnector"
    assert config["connector_extra_config"] == {"mode": "configured"}
    assert "--kv-transfer-config" in command


def test_build_baseline_command_uses_native_dp4_graph_server():
    args = _args()
    args.scenario = "baseline-graph"
    runner.configure_scenario(args)

    command = runner.build_baseline_command(args)

    assert "--additional-config" not in command
    assert _option(command, "--data-parallel-size") == "4"
    assert _option(command, "--tensor-parallel-size") == "1"
    assert _option(command, "--shutdown-timeout") == "10"
    assert "--enable-expert-parallel" in command
    assert "--worker-extension-cls" not in command
    assert json.loads(_option(command, "--compilation-config")) == {
        "cudagraph_mode": "FULL_DECODE_ONLY"
    }
    for passthrough_arg in ("--additional-config", '--additional-config={"afd":{}}'):
        args.common_vllm_arg = [passthrough_arg]
        with pytest.raises(ValueError, match="--additional-config"):
            runner.build_baseline_command(args)


@pytest.mark.parametrize(
    ("scenario", "backend", "attention_devices", "ffn_devices", "error_message"),
    [
        ("baseline-graph", "gpu", ["0", "1", "2", "3"], [], None),
        ("afd-v2-eager-1a1f", "npu", ["0"], ["1"], "require GPU"),
        (runner.ASYNC_CAM_SCENARIO, "gpu", ["0", "1"], ["2", "3"], "require NPU"),
        (runner.ASYNC_UBATCH_SCENARIO, "gpu", ["0", "1"], ["2"], "require NPU"),
        ("afd-eager-2a1f", "gpu", ["0"], ["2"], "attention-devices .* exactly 2"),
        ("afd-eager-2a1f", "gpu", ["0", "1"], [], "ffn-devices .* exactly 1"),
        ("afd-eager-2a1f", "gpu", ["0", "0"], ["2"], "Attention devices must be"),
        ("afd-eager-2a1f", "gpu", ["0", "1"], ["2", "2"], "FFN devices must be"),
        ("afd-eager-2a1f", "gpu", ["0", "1"], ["1"], "must not overlap"),
    ],
)
def test_validate_topology(
    scenario, backend, attention_devices, ffn_devices, error_message
):
    args = _args()
    args.scenario = scenario
    args.device_backend = backend
    runner.configure_scenario(args)

    if error_message is None:
        runner.validate_topology(args, attention_devices, ffn_devices)
    else:
        with pytest.raises(ValueError, match=error_message):
            runner.validate_topology(args, attention_devices, ffn_devices)


@pytest.mark.parametrize(
    ("scenario", "backend", "plugins", "v2_runner", "flashcomm", "logging_level"),
    [
        ("baseline-graph", "gpu", "", "1", "1", None),
        ("baseline-graph", "npu", "ascend", "1", None, None),
        ("afd-graph-2a1f", "gpu", "afd", "1", "1", None),
        ("afd-graph-dbo-2a1f", "gpu", "afd", "1", "1", "DEBUG"),
        ("afd-eager-2a2f", "npu", "ascend,afd", "1", None, None),
        ("afd-graph-dbo-2a2f", "npu", "ascend,afd", "1", None, "DEBUG"),
        ("afd-v2-eager-1a1f", "gpu", "afd", "1", "1", None),
    ],
)
def test_build_env_scopes_plugins_runner_and_backend_options(
    monkeypatch, scenario, backend, plugins, v2_runner, flashcomm, logging_level
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    args = _args()
    args.scenario = scenario
    args.device_backend = backend
    runner.configure_scenario(args)
    role = "baseline" if args.baseline else "ffn"

    env = runner.build_env("0,1", args, role=role, e2e_run_id="run-1")

    assert env["VLLM_PLUGINS"] == plugins
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == v2_runner
    assert env.get("VLLM_ASCEND_ENABLE_FLASHCOMM1") == flashcomm
    assert env.get("VLLM_LOGGING_LEVEL") == logging_level
    assert env[runner.E2E_RUN_ID_ENV] == "run-1"
    assert env[runner.E2E_PROCESS_ROLE_ENV] == role
    if backend == "npu":
        command = (
            runner.build_baseline_command(args)
            if args.baseline
            else runner.build_vllm_command(args, role="attention")
        )
        assert (
            "--worker-extension-cls" in command
        ) is runner.needs_mrv2_execution_evidence(args)


@pytest.mark.parametrize(
    "common_args",
    [
        ["--unrelated-arg"],
        ["--enable-prefix-caching", "--async-scheduling", "--enable-chunked-prefill"],
        [
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--no-enable-chunked-prefill",
        ],
    ],
)
@pytest.mark.parametrize(
    "scenario",
    [
        "afd-eager-2a1f",
        "afd-graph-dbo-2a2f",
        "afd-v2-eager-dbo-dp2",
        "afd-v2-graph-dbo-dp2",
    ],
)
def test_runner_preserves_native_runtime_option_choices(common_args, scenario):
    args = _args()
    args.scenario = scenario
    args.common_vllm_arg = list(common_args)
    runner.configure_scenario(args)

    assert args.common_vllm_arg == common_args
    command = runner.build_vllm_command(args, role="attention")
    for option in (
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--async-scheduling",
        "--no-async-scheduling",
        "--enable-chunked-prefill",
        "--no-enable-chunked-prefill",
    ):
        assert command.count(option) == common_args.count(option)


@pytest.mark.parametrize(
    ("backend", "visible_devices_env"),
    [("gpu", "CUDA_VISIBLE_DEVICES"), ("npu", "ASCEND_RT_VISIBLE_DEVICES")],
)
def test_print_command_reports_the_actual_visible_devices_environment(
    backend, visible_devices_env, capsys
):
    runner.print_command("ATTN", ["vllm", "serve"], backend, "0,1")

    assert f"({visible_devices_env}=0,1)" in capsys.readouterr().out


# --- lm-eval invocation and GSM8K gating -------------------------------------


def _inline_thread(on_join: Callable[[float | None], None], alive: bool = False):
    class InlineThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            on_join(timeout)

        def is_alive(self):
            return alive

    return InlineThread


def test_extract_gsm8k_sample_count():
    results = {"n-samples": {"gsm8k": {"effective": 7}}}
    assert helpers_gsm8k._extract_gsm8k_sample_count(results) == 7
    malformed: tuple[dict[str, Any], ...] = (
        {},
        {"n-samples": {"gsm8k": {"effective": "not-a-number"}}},
        {"n-samples": {"gsm8k": {"effective": 7.5}}},
    )
    for malformed_results in malformed:
        with pytest.raises((KeyError, ValueError), match="GSM8K sample count"):
            helpers_gsm8k._extract_gsm8k_sample_count(malformed_results)


@pytest.mark.parametrize(
    ("options", "expected_model_args"),
    [
        ({}, []),
        (
            {"max_gen_toks": 321, "num_concurrent": 12},
            ["max_gen_toks=321", "num_concurrent=12"],
        ),
    ],
)
def test_run_lm_eval_builds_builtin_gsm8k_command(
    monkeypatch, tmp_path, options, expected_model_args
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
            **options,
        )

    command, kwargs = popen_calls[0]
    assert _option(command, "--tasks") == "gsm8k"
    assert "--include_path" not in command
    assert "--limit" not in command
    assert kwargs["env"]["HF_HOME"] == hf_home
    model_args = _option(command, "--model_args").split(",")
    assert all(arg in model_args for arg in expected_model_args)


def test_run_lm_eval_reads_timestamped_results_file(monkeypatch, tmp_path):
    output_path = tmp_path / "results"
    model_output_path = output_path / "deepseek-v2-lite-afd-attention"
    model_output_path.mkdir(parents=True)
    expected_results = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.25}},
        "n-samples": {"gsm8k": {"original": 1319, "effective": 7}},
    }
    (model_output_path / "results_2026-08-07T11-16-36.json").write_text(
        json.dumps(expected_results)
    )

    class CompletedProcess:
        pid = 404
        stdout = io.StringIO("| strict-match | 5 | exact_match | 0.25 | +/- | 0.0 |\n")
        returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        helpers_gsm8k.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess()
    )

    results = helpers_gsm8k._run_lm_eval(
        "http://127.0.0.1:8000", "model", output_path=str(output_path)
    )

    assert results == expected_results


def test_lm_eval_timeout_reaps_process_and_reader(  # long-test: reap and join
    monkeypatch, tmp_path
):
    popen_calls = []
    group_alive = True
    leader_signal_calls = []
    group_signal_calls = []
    reader_join_calls: list[float | None] = []

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

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            group_alive = False

    monotonic_values = iter([100.0, 111.0, 8000.0, 8021.0, 9000.0, 9000.0])
    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        helpers_gsm8k.threading, "Thread", _inline_thread(reader_join_calls.append)
    )
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(helpers_gsm8k, "signal", signal, raising=False)
    monkeypatch.setattr(
        helpers_gsm8k.os,
        "kill",
        lambda pid, sig: leader_signal_calls.append((pid, sig)),
        raising=False,
    )
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
    assert group_signal_calls == [(404, 0), (404, 9), (404, 0)]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "forced SIGKILL" in str(error.value.__cause__)
    assert process.wait_calls == [helpers_gsm8k.LM_EVAL_REAP_TIMEOUT_S]
    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_run_lm_eval_rejects_a_reader_that_does_not_join(monkeypatch, tmp_path):
    reader_join_calls: list[float | None] = []

    class CompletedProcess:
        pid = 404
        stdout = io.StringIO('{"results":{"gsm8k":{"exact_match":1.0}}}\n')
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        helpers_gsm8k.subprocess, "Popen", lambda *_args, **_kwargs: CompletedProcess()
    )
    monkeypatch.setattr(
        helpers_gsm8k.threading,
        "Thread",
        _inline_thread(reader_join_calls.append, alive=True),
    )
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: 100.0)

    with pytest.raises(RuntimeError, match="reader thread"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000", "model", output_path=str(tmp_path / "results")
        )

    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_lm_eval_signal_waits_for_cleanup(  # long-test: signal and cleanup lifecycle
    monkeypatch, tmp_path
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
        helpers_gsm8k.signal, "getsignal", lambda signum: previous_handlers[signum]
    )
    monkeypatch.setattr(helpers_gsm8k.signal, "signal", fake_signal)
    monkeypatch.setattr(
        helpers_gsm8k.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(
        helpers_gsm8k.threading,
        "Thread",
        _inline_thread(lambda timeout: events.append(("reader-join", timeout))),
    )
    monkeypatch.setattr(helpers_gsm8k.os, "kill", fake_kill, raising=False)
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: next(monotonic_values))

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

    class UnstartedReader:
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
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", UnstartedReader)
    monkeypatch.setattr(
        helpers_gsm8k.signal, "getsignal", lambda _signum: previous_handler
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
            "http://127.0.0.1:8000", "model", output_path=str(tmp_path), timeout_s=0
        )

    assert cleaned == [404]


@pytest.mark.parametrize(
    ("scenario", "backend", "limit_env", "expected_options"),
    [
        ("afd-eager-2a1f", "gpu", None, {"limit": 7}),
        ("afd-eager-2a1f", "npu", None, {"limit": 7}),
        ("afd-eager-2a1f", "gpu", "all", {"limit": None, "timeout_s": 28800}),
        (runner.ASYNC_UBATCH_SCENARIO, "npu", None, {"limit": 7, "batch_size": 2}),
        ("afd-graph-dbo-2a1f", "gpu", None, {"limit": 24, "num_concurrent": 12}),
        # The V2 DBO comparison cases share one evaluation workload.
        ("afd-v2-eager-dp2", "gpu", None, {"limit": 128, "num_concurrent": 12}),
        ("afd-v2-eager-dbo-dp2", "gpu", None, {"limit": 128, "num_concurrent": 12}),
        ("afd-v2-graph-dbo-dp2", "gpu", None, {"limit": 128, "num_concurrent": 12}),
    ],
)
def test_gsm8k_scenario_workload_and_result_validation(  # long-test: evaluation
    monkeypatch, scenario, backend, limit_env, expected_options
):
    args = _args()
    args.scenario = scenario
    args.device_backend = backend
    runner.configure_scenario(args)
    if limit_env is None:
        monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)
    else:
        monkeypatch.setenv("AFD_GSM8K_LIMIT", limit_env)
    calls = []

    expected_count = expected_options["limit"] or runner.GSM8K_FULL_SAMPLE_COUNT

    def fake_run_lm_eval(base_url, model_name, **kwargs):
        calls.append((base_url, model_name, kwargs))
        return {
            "n-samples": {"gsm8k": {"effective": expected_count}},
            "results": {"gsm8k": {"exact_match": 0.5}},
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
                **expected_options,
            },
        )
    ]
    for effective, accuracy, error_message in (
        (expected_count - 1, 1.0, f"expected {expected_count}"),
        (expected_count, 0.26, "below the required threshold"),
        (expected_count, float("nan"), "below the required threshold"),
    ):
        monkeypatch.setattr(
            runner,
            "_run_lm_eval",
            lambda *_args, effective=effective, accuracy=accuracy, **_kwargs: {
                "n-samples": {"gsm8k": {"effective": effective}},
                "results": {"gsm8k": {"exact_match": accuracy}},
            },
        )
        with pytest.raises(RuntimeError, match=error_message):
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
    monkeypatch, body, error
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


# --- Runner process lifecycle -------------------------------------------------


def test_ensure_processes_alive_reports_exited_process_returncode():
    process = argparse.Namespace(poll=lambda: 17)

    with pytest.raises(RuntimeError, match="returncode=17"):
        runner.ensure_processes_alive([process])  # type: ignore[list-item]


def test_terminate_processes_rejects_cleanup_failure(monkeypatch):
    monkeypatch.setattr(
        runner, "terminate_process_groups", lambda *_args, **_kwargs: ["cleanup failed"]
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.terminate_processes([])


class _JoinRecorder:
    joined = False

    def join(self, timeout):
        self.joined = True


def _stub_main_pipeline(monkeypatch, process, log_thread: _JoinRecorder) -> None:
    args = _args()
    args.attention_devices = "0"
    args.ffn_devices = "1"
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "validate_topology", lambda *_args: None)
    monkeypatch.setattr(runner, "build_vllm_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "build_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "start_process", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "stream_output", lambda *_args: log_thread)
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(runner, "run_gsm8k_evaluation", lambda *_args: None)


@pytest.mark.parametrize("native_handler", [False, True])
def test_main_checks_processes_and_restores_signal_handlers_when_cleanup_fails(
    monkeypatch, capsys, native_handler
):
    process = argparse.Namespace(poll=lambda: None)
    log_thread = _JoinRecorder()
    checked_processes = []
    installed_handlers = []
    previous_handlers = {
        signal.SIGTERM: None if native_handler else "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }
    _stub_main_pipeline(monkeypatch, process, log_thread)
    monkeypatch.setattr(
        runner,
        "ensure_processes_alive",
        lambda processes: checked_processes.append(list(processes)),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup failed")
        ),
    )
    monkeypatch.setattr(
        runner.signal, "getsignal", lambda signum: previous_handlers[signum]
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
    log_thread = _JoinRecorder()
    installed_handlers = {}
    cleanup_events = []
    previous_handlers = {
        signal.SIGTERM: "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }

    def fake_signal(signum, handler):
        installed_handlers[signum] = handler

    def cleanup_with_first_signal(_processes, **_kwargs):
        cleanup_events.append("started")
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        cleanup_events.append("completed")
        raise RuntimeError("cleanup failed")

    _stub_main_pipeline(monkeypatch, argparse.Namespace(poll=lambda: None), log_thread)
    monkeypatch.setattr(runner, "terminate_processes", cleanup_with_first_signal)
    monkeypatch.setattr(
        runner.signal, "getsignal", lambda signum: previous_handlers[signum]
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


@pytest.mark.parametrize("ffn_tp_size", [1, 2])
def test_runner_drops_flashcomm_for_npu_role(monkeypatch, ffn_tp_size):
    args = _args()
    args.device_backend = "npu"
    args.ffn_tp_size = ffn_tp_size
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

    env = runner.build_env("2,3", args, role="ffn")

    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"


@pytest.mark.parametrize(
    "scenario",
    ["baseline-graph", "afd-eager-2a2f", "afd-graph-2a1f", "afd-graph-dbo-2a2f"],
)
def test_npu_shared_cases_migrate_to_native_v2(scenario):
    args = _args()
    args.scenario = scenario
    args.device_backend = "npu"
    runner.configure_scenario(args)
    command = (
        runner.build_baseline_command(args)
        if args.baseline
        else runner.build_vllm_command(args, role="attention")
    )
    assert runner.build_env("0", args)["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert ("--worker-extension-cls" in command) is (
        not args.baseline and args.cuda_graph_full_decode_only
    )


@pytest.mark.parametrize(
    "scenario, backend",
    [
        ("afd-graph-2a1f", "gpu"),
        ("afd-v2-eager-dbo-dp2", "gpu"),
        ("afd-v2-graph-dbo-dp2", "gpu"),
        ("afd-v2-graph-tp2", "gpu"),
        ("afd-graph-2a1f", "npu"),
        ("afd-graph-dbo-2a1f", "npu"),
        ("afd-graph-dbo-2a2f", "npu"),
    ],
)
def test_feature_evidence_requires_every_rank_and_joint_execution(scenario, backend):
    args = _args()
    args.device_backend = backend
    args.scenario = scenario
    runner.configure_scenario(args)
    layout = [[stage, [4] * 2] for stage in range(2)]
    layouts = json.dumps({"two_stage": [layout], "consecutive_graph": [layout]})
    evidence = [
        (
            role,
            f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank={dp} tp_rank={tp} "
            "graph_replays=2 two_stage_steps=2 two_stage_graph_replays=2 "
            f"layouts={layouts}",
        )
        for role, ranks in (
            ("attention", args.num_attention_ranks),
            ("ffn", args.num_ffn_ranks),
        )
        for dp in range(ranks // runner.role_tp_size(args, role))
        for tp in range(runner.role_tp_size(args, role))
    ]
    runner.assert_mrv2_execution_evidence(evidence, args)
    with pytest.raises(RuntimeError, match="lacks live"):
        runner.assert_mrv2_execution_evidence(evidence[:-1], args)
    eager_only = [
        (role, line.replace("graph_replays=2", "graph_replays=0"))
        for role, line in evidence
    ]
    if args.cuda_graph_full_decode_only:
        with pytest.raises(RuntimeError, match="lacks live"):
            runner.assert_mrv2_execution_evidence(eager_only, args)
    else:
        runner.assert_mrv2_execution_evidence(eager_only, args)
    unsplit = [
        (
            role,
            line.replace("two_stage_steps=2", "two_stage_steps=0").replace(
                "two_stage_graph_replays=2", "two_stage_graph_replays=0"
            ),
        )
        for role, line in evidence
    ]
    if args.enable_dbo:
        with pytest.raises(RuntimeError, match="lacks live"):
            runner.assert_mrv2_execution_evidence(unsplit, args)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_dbo_requires_matching_layout_on_every_rank(graph, backend):
    args = _args()
    args.device_backend = backend
    args.scenario = "afd-v2-graph-dbo-dp2" if graph else "afd-v2-eager-dbo-dp2"
    runner.configure_scenario(args)
    layout = [[0, [4, 4]], [1, [4, 4]]]
    key = "consecutive_graph" if graph else "two_stage"

    def summary(dp, layout):
        return (
            f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank={dp} tp_rank=0 "
            "graph_replays=2 two_stage_steps=2 two_stage_graph_replays=2 "
            f"layouts={json.dumps({key: [layout]})}"
        )

    evidence = [
        (role, summary(dp, layout)) for role in ("attention", "ffn") for dp in range(2)
    ]
    runner.assert_mrv2_execution_evidence(evidence, args)
    for changed_ranks in ((0,), (0, 1)):
        mismatched = [
            (role, summary(dp, [[0, [3, 3]], [1, [3, 3]]]))
            if role == "ffn" and dp in changed_ranks
            else (role, line)
            for (role, line), dp in zip(evidence, (0, 1, 0, 1), strict=True)
        ]
        with pytest.raises(RuntimeError, match="matching two-stage layouts"):
            runner.assert_mrv2_execution_evidence(mismatched, args)
    if graph:
        total_replays_without_consecutive_pair = [
            (role, line.replace(json.dumps({key: [layout]}), json.dumps({key: []})))
            for role, line in evidence
        ]
        with pytest.raises(RuntimeError, match="consecutive graph replays"):
            runner.assert_mrv2_execution_evidence(
                total_replays_without_consecutive_pair, args
            )


@pytest.mark.parametrize(
    "scenario", [runner.ASYNC_UBATCH_SCENARIO, runner.DSV4_ASYNC_CAM_SCENARIO]
)
def test_async_dbo_qualification_requires_live_attention_stages_without_ffn_layout(
    scenario,
):
    args = _args()
    args.device_backend = "npu"
    args.scenario = scenario
    runner.configure_scenario(args)
    assert not args.enable_dbo
    if scenario == runner.DSV4_ASYNC_CAM_SCENARIO:
        # The original short-request smoke is below native thresholds. A
        # separate DBO qualification deliberately supplies eligible work.
        assert not runner.needs_mrv2_execution_evidence(args)
        args.enable_dbo = True
    assert runner.needs_mrv2_execution_evidence(args)
    assert runner.execution_evidence_roles(args) == ("attention",)
    layout = [
        [0, [4] * (args.num_attention_ranks // args.attention_tp_size)],
        [1, [4] * (args.num_attention_ranks // args.attention_tp_size)],
    ]
    layouts = json.dumps({"two_stage": [layout]})
    evidence = [
        (
            "attention",
            f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank={dp} tp_rank={tp} "
            f"two_stage_steps=1 layouts={layouts}",
        )
        for dp in range(args.num_attention_ranks // args.attention_tp_size)
        for tp in range(args.attention_tp_size)
    ]
    runner.assert_mrv2_execution_evidence(evidence, args)
    if scenario == runner.DSV4_ASYNC_CAM_SCENARIO:
        # Independent DP scheduling legitimately produces different local stage
        # sizes. Every TP peer still completed both nonempty stages.
        independent = []
        for index, (role, line) in enumerate(evidence):
            local_dp = index // args.attention_tp_size
            local_layout = [[0, [3 + local_dp]], [1, [4 + local_dp]]]
            independent.append(
                (
                    role,
                    line.split(" layouts=", 1)[0]
                    + " layouts="
                    + json.dumps({"two_stage": [local_layout]}),
                )
            )
        runner.assert_mrv2_execution_evidence(independent, args)
    with pytest.raises(RuntimeError, match="lacks live"):
        runner.assert_mrv2_execution_evidence(
            [
                (role, line.replace("two_stage_steps=1", "two_stage_steps=0"))
                for role, line in evidence
            ],
            args,
        )


def test_dsv4_short_concurrent_case_keeps_native_conditional_dbo():
    args = _args()
    args.device_backend = "npu"
    args.scenario = runner.DSV4_ASYNC_CAM_SCENARIO
    runner.configure_scenario(args)
    command = runner.build_vllm_command(args, role="attention")
    extra = json.loads(command[command.index("--additional-config") + 1])
    assert extra["afd"]["connector_extra_config"]["async_moe_ubatching"] is True
    assert not args.enable_dbo
    assert "--dbo-decode-token-threshold" not in command
    assert "--dbo-prefill-token-threshold" not in command
    assert "--worker-extension-cls" not in command
    assert not runner.needs_mrv2_execution_evidence(args)


def test_mrv2_evidence_rpc_uses_attention_without_ffn_http(monkeypatch):
    args = _args()
    calls = []

    def urlopen(request, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(b'{"results":[null]}')

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    for method in ("afd_reset_execution_evidence", "afd_log_execution_evidence"):
        runner.mrv2_execution_evidence_rpc(args, method)
    assert [(url, body["method"]) for url, body, _ in calls] == [
        ("http://127.0.0.1:18100/collective_rpc", method)
        for method in ("afd_reset_execution_evidence", "afd_log_execution_evidence")
    ]
    assert all(
        body["timeout"] == timeout == runner.MRV2_EVIDENCE_RPC_TIMEOUT_S
        for _, body, timeout in calls
    )


@pytest.mark.parametrize("scenario", ["afd-v2-graph-1a1f", "afd-v2-eager-dbo-dp2"])
def test_mrv2_evidence_readiness_requires_only_attention_api(monkeypatch, scenario):
    args = _args()
    args.scenario = scenario
    args.startup_timeout = 30
    runner.configure_scenario(args)
    calls = []

    class Response(io.BytesIO):
        status = 200

    def urlopen(url, timeout):
        calls.append(url)
        return Response()

    monkeypatch.setattr(runner.urllib.request, "urlopen", urlopen)
    runner.wait_for_openai_api(args, [])
    assert calls == ["http://127.0.0.1:18100/v1/models"]


def test_gpu_summary_wait_requires_unique_ranks_and_checks_liveness(monkeypatch):
    args = _args()
    args.scenario = "afd-v2-eager-dbo-dp2"
    runner.configure_scenario(args)
    evidence = [
        (role, f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank={dp} tp_rank=0")
        for role, dp in (("attention", 0), ("attention", 0), ("ffn", 0), ("ffn", 1))
    ]

    def sleep(seconds):
        evidence.append(
            ("attention", f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank=1 tp_rank=0")
        )

    monkeypatch.setattr(runner.time, "sleep", sleep)
    runner.wait_for_mrv2_execution_summaries(evidence, args, [])
    assert (
        "attention",
        f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank=1 tp_rank=0",
    ) in evidence


@pytest.mark.parametrize("failure", ["timeout", "process-exit"])
def test_gpu_summary_wait_is_bounded_and_preserves_process_failure(
    monkeypatch, failure
):
    args = _args()
    args.scenario = "afd-v2-graph-1a1f"
    runner.configure_scenario(args)
    ticks = iter((0, runner.MRV2_EVIDENCE_RPC_TIMEOUT_S))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    process = argparse.Namespace(args=["vllm"], poll=lambda: 1)
    with pytest.raises(RuntimeError):
        runner.wait_for_mrv2_execution_summaries(
            [],
            args,
            [cast(subprocess.Popen[str], process)] if failure == "process-exit" else [],
        )


def test_stream_output_collects_bounded_gpu_summaries(capsys):
    line = (
        f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank=0 tp_rank=0 "
        "graph_replays=2 two_stage_steps=2 two_stage_graph_replays=2\n"
    )
    process = argparse.Namespace(stdout=io.StringIO("ordinary log\n" + line))
    evidence: list[tuple[str, str]] = []
    thread = runner.stream_output(
        "attention",
        cast(subprocess.Popen[str], process),
        mrv2_execution_evidence=evidence,
    )
    thread.join()
    assert evidence == [("attention", line.strip())]
    assert "ordinary log" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [None, "reset", "missing-summary"])
def test_mrv2_evidence_window_and_failure_cleanup(monkeypatch, capsys, failure):
    args = _args()
    args.scenario = "afd-v2-graph-1a1f"
    args.attention_devices, args.ffn_devices = "0", "1"
    process = argparse.Namespace(poll=lambda: None)
    events = []
    markers = []
    native_build_env = runner.build_env

    def build_env(*args, **kwargs):
        env = native_build_env(*args, **kwargs)
        marker = Path(env["AFD_E2E_MRV2_EVIDENCE_START"])
        assert not marker.exists()
        markers.append(marker)
        return env

    class LogThread:
        def __init__(self, role, evidence):
            self.role, self.evidence = role, evidence

        def join(self, timeout):
            events.append(f"joined-{self.role}")
            assert markers[0].parent.exists()

    def evidence_rpc(_args, method):
        events.append(method)
        assert markers[0].exists() == (method == "afd_log_execution_evidence")
        if failure == "reset" and method == "afd_reset_execution_evidence":
            raise RuntimeError("RPC reset failed")

    def evaluate(_args):
        assert markers[0] == markers[1] and markers[0].exists()
        events.append("evaluate")

    def terminate(*_args, **_kwargs):
        assert markers[0].parent.exists()
        events.append("cleanup")

    def wait_summaries(evidence, _args, _processes):
        assert markers[0].with_name("evaluation-complete").exists()
        events.append("summaries")
        if failure == "missing-summary":
            raise RuntimeError("GPU ffn lacks live execution summaries")
        evidence.extend(
            (
                role,
                f"{runner.MRV2_EVIDENCE_PREFIX}dp_rank=0 tp_rank=0 "
                "graph_replays=1 two_stage_steps=0 two_stage_graph_replays=0",
            )
            for role in ("attention", "ffn")
        )

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "build_env", build_env)
    monkeypatch.setattr(runner, "start_process", lambda *_args: process)
    monkeypatch.setattr(
        runner,
        "stream_output",
        lambda role, _process, evidence: LogThread(role, evidence),
    )
    monkeypatch.setattr(
        runner, "wait_for_openai_api", lambda *_args: events.append("ready")
    )
    monkeypatch.setattr(runner, "run_gsm8k_evaluation", evaluate)
    monkeypatch.setattr(runner, "mrv2_execution_evidence_rpc", evidence_rpc)
    monkeypatch.setattr(runner, "wait_for_mrv2_execution_summaries", wait_summaries)
    monkeypatch.setattr(runner, "terminate_processes", terminate)
    if failure is None:
        assert runner.main() == 0
        assert "E2E SCENARIO" in capsys.readouterr().out
    else:
        with pytest.raises(RuntimeError, match="RPC reset failed|lacks live"):
            runner.main()
        assert "E2E SCENARIO" not in capsys.readouterr().out
    assert events[0:2] == ["ready", "afd_reset_execution_evidence"]
    assert events[-3:] == ["cleanup", "joined-ffn", "joined-attention"]
    assert not markers[0].parent.exists()
    if failure != "reset":
        assert events[2:5] == ["evaluate", "afd_log_execution_evidence", "summaries"]


@pytest.mark.parametrize("scenario", sorted(runner.V2_DBO_COMPARISON_SCENARIOS))
def test_v2_dbo_comparison_uses_identical_evaluation(monkeypatch, scenario):
    args = _args()
    args.scenario = scenario
    runner.configure_scenario(args)
    monkeypatch.delenv("AFD_GSM8K_LIMIT", raising=False)
    calls = []

    def evaluate(*positional, **kwargs):
        calls.append(kwargs)
        return {
            "n-samples": {"gsm8k": {"effective": 128}},
            "results": {"gsm8k": {"exact_match": 0.5}},
        }

    monkeypatch.setattr(runner, "_run_lm_eval", evaluate)
    runner.run_gsm8k_evaluation(args)
    assert calls[0]["limit"] == 128
    assert calls[0]["num_concurrent"] == 12
    assert calls[0]["num_fewshot"] == 8
