# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import sys

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import completions
from tests.e2e.models.deepseek_v4_flash import test_sync_camp2p_npu as entrypoint
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
    DSV4_SYNC_SHAPES,
)

# (attention DP, attention TP, FFN DP, FFN TP) each profile must produce. A5
# shards both roles by data parallel with expert parallelism; A3 uses the shape
# its recorded deployment runs, Attention DP2/TP4 and FFN DP8/TP1 with expert
# parallelism on sixteen dies.
EXPECTED_PARALLELISM = {
    DSV4_SYNC_CAMP2P_A5_SCENARIO: ("2", "1", "2", "1"),
    DSV4_SYNC_CAMP2P_A3_SCENARIO: ("2", "4", "8", "1"),
}


def _flag_value(command: list[str], flag: str) -> str | None:
    """Return a flag's value, or None when the scenario omits the flag."""
    return command[command.index(flag) + 1] if flag in command else None


def _devices_for(scenario: str) -> str:
    count = DSV4_SYNC_SHAPES[scenario].device_count
    return ",".join(str(index) for index in range(count))


def _arguments(
    monkeypatch,
    tmp_path,
    *,
    scenario: str = DSV4_SYNC_CAMP2P_A5_SCENARIO,
    model: str = "/models/dsv4",
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_E2E_DEVICES", _devices_for(scenario))
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", model)
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_API_PORT", raising=False)
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_AFD_PORT", raising=False)
    command = entrypoint.build_runner_command(scenario, tmp_path / "responses.json")
    monkeypatch.setattr(sys, "argv", ["runner", *command[3:]])
    return runner.parse_args()


@pytest.mark.parametrize("scenario", sorted(EXPECTED_PARALLELISM))
def test_dsv4_sync_fixed_deployment(monkeypatch, tmp_path, scenario):
    args = _arguments(monkeypatch, tmp_path, scenario=scenario)
    profile = DSV4_SYNC_SHAPES[scenario]
    runner.configure_scenario(args)
    runner.validate_topology(
        args,
        runner.parse_csv(args.attention_devices),
        runner.parse_csv(args.ffn_devices),
    )
    # The synchronous transport must not borrow the async CAM teardown, which
    # defers an FFN SIGKILL for a pending CAM receive.
    assert not runner.uses_npu_async_process_cleanup(args)
    assert args.afd_async is False
    assert args.compute_gate_on_attention is False
    assert args.gsm8k_output_path is None
    assert args.cuda_graph_full_decode_only is profile.use_graph
    # The recorded A5 DBO stays off, so no DBO flag reaches vLLM.
    assert args.enable_dbo is False
    expected = EXPECTED_PARALLELISM[scenario]
    for role, dp, tp in (
        ("attention", expected[0], expected[1]),
        ("ffn", expected[2], expected[3]),
    ):
        command = runner.build_vllm_command(args, role=role)
        assert command[command.index("--data-parallel-size") + 1] == dp
        assert command[command.index("--tensor-parallel-size") + 1] == tp
        assert command[command.index("--max-model-len") + 1] == profile.max_model_len
        assert (
            _flag_value(command, "--max-num-batched-tokens")
            == profile.max_num_batched_tokens
        )
        assert (
            _flag_value(command, "--gpu-memory-utilization")
            == profile.memory_utilization
        )
        assert ("--enable-expert-parallel" in command) is (
            profile.enable_expert_parallel
        )
        assert "--enable-dbo" not in command
        assert ("--enforce-eager" in command) is not profile.use_graph
        if profile.compilation_config is not None:
            # A verbatim profile passes its host script's compilation config and
            # nothing else from the case's graph deployment.
            assert (
                json.loads(
                    command[command.index("--compilation-config") + 1],
                )
                == profile.compilation_config
            )
            assert "--cudagraph-capture-sizes" not in command
            assert "--max-cudagraph-capture-size" not in command
        elif profile.use_graph:
            capture_size = str(profile.cudagraph_capture_size)
            assert command[command.index("--cudagraph-capture-sizes") + 1] == (
                capture_size
            )
            assert "--compilation-config" in command
            assert command[command.index("--max-num-seqs") + 1] == (
                profile.max_num_seqs or capture_size
            )
        if profile.verbatim_launch:
            # Only the script's own deployment flags, plus the tokenizer mode and
            # parsers the concurrent chat oracle needs, plus the cache layout the
            # profile records where the script leaves a default it cannot use.
            for absent in (
                "--api-server-count",
                "--seed",
                "--data-parallel-address",
                "--no-disable-hybrid-kv-cache-manager",
                "--enable-chunked-prefill",
            ):
                assert absent not in command, absent
            assert _flag_value(command, "--block-size") == profile.block_size
            assert ("--no-enable-prefix-caching" in command) is (
                profile.disable_prefix_caching
            )
        else:
            assert command[command.index("--api-server-count") + 1] == "1"
            assert command[command.index("--seed") + 1] == "1024"
            assert command[command.index("--block-size") + 1] == "128"
            assert "--no-enable-prefix-caching" in command
            assert "--enable-chunked-prefill" in command
        if profile.quantization_from_checkpoint:
            assert command[command.index("--quantization") + 1] == "ascend"
        else:
            assert "--quantization" not in command
        assert "--kv-transfer-config" not in command
        config = json.loads(command[command.index("--additional-config") + 1])
        # Every DSV4 case pins the model-path switches, because the pinned
        # runtime defaults the multistream DSA overlap to True and that RoPE
        # path fails to tile on A5.
        assert config["enable_dsv4_shared_compressor_workspace"] is False
        assert config["multistream_dsv4_dsa_overlap"] is False
        assert config["enable_dsa_cp"] is False
        assert config["enable_cpu_binding"] is True
        # Exact equality also pins the absent keys: CAMP2P rejects both the
        # asynchronous mode and gate-on-Attention.
        expected_afd = {
            "role": role,
            "connector": "CAMP2pAFDConnector",
            "host": "192.0.2.1",
            "port": 6456,
            "num_attention_ranks": int(expected[0]) * int(expected[1]),
            "num_ffn_ranks": int(expected[2]) * int(expected[3]),
        }
        if profile.connector_extra_config is not None:
            expected_afd["connector_extra_config"] = profile.connector_extra_config
        assert config["afd"] == expected_afd
        env = runner.build_env("0", args, role=role, e2e_run_id="test")
        assert env["VLLM_PLUGINS"] == "ascend,afd"
        # Attention TP>1 has a TP/SP token split, but this connector path must
        # not inherit the async case's FlashComm1 setting.
        assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env


@pytest.mark.parametrize("scenario", sorted(EXPECTED_PARALLELISM))
def test_dsv4_sync_entrypoint_rejects_wrong_devices(monkeypatch, tmp_path, scenario):
    count = DSV4_SYNC_SHAPES[scenario].device_count
    wrong_counts = [
        ",".join(str(index) for index in range(count - 1)),
        ",".join(str(index) for index in range(count + 1)),
    ]
    for devices in [*wrong_counts, ",".join(["0"] * count)]:
        _arguments(monkeypatch, tmp_path, scenario=scenario)
        monkeypatch.setenv("AFD_E2E_DEVICES", devices)
        with pytest.raises(
            RuntimeError,
            match=rf"exactly {count} devices|devices must be unique",
        ):
            entrypoint.build_runner_command(scenario, tmp_path / "responses.json")


def test_dsv4_sync_scenarios_reject_each_others_device_count(monkeypatch, tmp_path):
    """The two shapes are host-specific; the other host's list must fail."""
    a5_count = DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A5_SCENARIO].device_count
    a3_count = DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A3_SCENARIO].device_count
    assert a5_count != a3_count
    _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO)
    monkeypatch.setenv("AFD_E2E_DEVICES", _devices_for(DSV4_SYNC_CAMP2P_A3_SCENARIO))
    with pytest.raises(RuntimeError, match=rf"exactly {a5_count} devices"):
        entrypoint.build_runner_command(
            DSV4_SYNC_CAMP2P_A5_SCENARIO,
            tmp_path / "responses.json",
        )


@pytest.mark.parametrize(
    "field", ["common_vllm_arg", "attention_vllm_arg", "ffn_vllm_arg"]
)
def test_dsv4_sync_rejects_deployment_overrides(monkeypatch, tmp_path, field):
    args = _arguments(monkeypatch, tmp_path)
    setattr(args, field, ["--max-num-batched-tokens=1"])
    with pytest.raises(ValueError, match="extra vLLM arguments"):
        runner.configure_scenario(args)


def test_dsv4_sync_a5_never_passes_quantization(monkeypatch, tmp_path):
    """The A5 launch script passes no `--quantization` at all."""
    args = _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO)
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    assert "--quantization" not in command


def test_dsv4_sync_profiles_run_as_smoke_cases():
    """Both synchronous profiles check the plumbing; the async case the answer."""
    assert DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A3_SCENARIO].check_answer is False
    assert DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A5_SCENARIO].check_answer is False
    # The asynchronous case carries no synchronous profile, so it keeps the
    # oracle's exact-answer default.
    assert runner.sync_shape(runner.DSV4_ASYNC_CAM_SCENARIO) is None


def test_dsv4_sync_served_response_keeps_the_shape_checks():
    """A profile that skips the answer still requires a finished response."""

    def response(content: object, finish_reason: object) -> dict:
        return {
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason},
            ],
        }

    for content, finish_reason in (
        ("28", "stop"),
        ("10 10:56:33 10:56:33", "length"),
        ("I cannot compute that.", "stop"),
    ):
        completions.validate_response(
            response(content, finish_reason),
            check_answer=False,
        )

    with pytest.raises(RuntimeError, match="returned empty content"):
        completions.validate_response(response("", "stop"), check_answer=False)
    with pytest.raises(RuntimeError, match="did not finish normally"):
        completions.validate_response(response("28", None), check_answer=False)
    # A host that does compare the answer keeps the exact terminal state.
    with pytest.raises(RuntimeError, match="did not finish normally"):
        completions.validate_response(response("28", "length"))


def test_dsv4_sync_oracle_uses_the_profile_answer_policy(monkeypatch, tmp_path):
    """The runner hands each profile's own answer policy to the oracle."""
    seen: dict = {}

    def capture(**kwargs: object) -> None:
        seen.clear()
        seen.update(kwargs)

    monkeypatch.setattr(runner, "evaluate_completions", capture)

    for scenario, expected in (
        (DSV4_SYNC_CAMP2P_A5_SCENARIO, False),
        (DSV4_SYNC_CAMP2P_A3_SCENARIO, False),
    ):
        args = _arguments(monkeypatch, tmp_path, scenario=scenario)
        runner.configure_scenario(args)

        runner.run_concurrent_completion_evaluation(args)

        assert seen["check_answer"] is expected


def test_dsv4_sync_a5_environment_follows_the_launch_script(monkeypatch):
    """The A5 profile keeps the script's buffer size, allocator, and start method."""
    monkeypatch.delenv("HCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    monkeypatch.delenv("HCCL_BUFFSIZE", raising=False)
    monkeypatch.delenv("PYTORCH_NPU_ALLOC_CONF", raising=False)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    env = entrypoint.build_environment(DSV4_SYNC_CAMP2P_A5_SCENARIO)
    assert env["HCCL_BUFFSIZE"] == "2048"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in env
    assert "AFD_FORCE_SPAWN_MULTIPROCESSING" not in env
    # The NIC variables stay optional on a single host.
    assert "GLOO_SOCKET_IFNAME" not in env
    assert "TP_SOCKET_IFNAME" not in env
    # A caller who pins the buffer size keeps it.
    monkeypatch.setenv("HCCL_BUFFSIZE", "4096")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth-test")
    env = entrypoint.build_environment(DSV4_SYNC_CAMP2P_A5_SCENARIO)
    assert env["HCCL_BUFFSIZE"] == "4096"
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test"
    assert env["TP_SOCKET_IFNAME"] == "eth-test"


def test_dsv4_sync_a5_afd_host_defaults_to_loopback(monkeypatch, tmp_path):
    """The A5 script announces 127.0.0.1 and needs no NIC variable."""
    _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO)
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    command = entrypoint.build_runner_command(
        DSV4_SYNC_CAMP2P_A5_SCENARIO,
        tmp_path / "responses.json",
    )
    assert command[command.index("--afd-host") + 1] == "127.0.0.1"


def test_dsv4_sync_a3_requires_the_caller_address(monkeypatch, tmp_path):
    """The A3 profile still takes the caller's advertised rendezvous address."""
    _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A3_SCENARIO)
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    with pytest.raises(RuntimeError, match="HCCL_IF_IP"):
        entrypoint.build_runner_command(
            DSV4_SYNC_CAMP2P_A3_SCENARIO,
            tmp_path / "responses.json",
        )
