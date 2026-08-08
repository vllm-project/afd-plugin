from __future__ import annotations

import argparse
import io
import json
import urllib.error

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v2_lite import test_async_cam_npu as async_cam_e2e
from tests.e2e.models.deepseek_v2_lite import test_e2e_npu as npu_e2e
from tests.e2e.models.qwen3_moe import test_e2e_gpu as qwen3_moe_e2e
from tests.e2e.runner import build_vllm_command


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        num_attention_ranks=2,
        num_ffn_ranks=2,
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="127.0.0.1",
        afd_port=6249,
        startup_timeout=900,
        served_model_name_prefix="deepseek-v2-lite-afd",
        prompt="San Francisco is a",
        max_tokens=16,
        temperature=0.0,
        num_requests=None,
        request_concurrency=None,
        cuda_graph_full_decode_only=False,
        cudagraph_capture_size=64,
        enable_dbo=False,
        dbo_decode_token_threshold=1,
        dbo_prefill_token_threshold=None,
        use_decode_bench_connector=False,
        common_vllm_arg=["--trust-remote-code"],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        tp_size=1,
        attention_tp_size=None,
        ffn_tp_size=None,
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_connector_extra_config=[],
        device_backend="gpu",
    )


def _arg_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_runner_uses_native_dp_for_attention_topology():
    command = build_vllm_command(_args(), role="attention")

    assert command.count("serve") == 1
    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert "--worker-cls" not in command
    assert _arg_value(command, "--port") == "18100"
    assert _arg_value(command, "--served-model-name") == (
        "deepseek-v2-lite-afd-attention"
    )

    additional_config = json.loads(_arg_value(command, "--additional-config"))
    assert additional_config["afd"]["num_attention_ranks"] == 2
    assert additional_config["afd"]["num_ffn_ranks"] == 2
    assert "enabled" not in additional_config["afd"]
    assert "connector_extra_config" not in additional_config["afd"]
    assert "afd_role_rank" not in additional_config["afd"]


def test_runner_uses_native_dp_for_ffn_topology():
    command = build_vllm_command(_args(), role="ffn")

    assert _arg_value(command, "--data-parallel-size") == "2"
    assert _arg_value(command, "--tensor-parallel-size") == "1"
    assert "--enable-expert-parallel" in command
    assert "--worker-cls" not in command
    assert _arg_value(command, "--port") == "18101"
    assert _arg_value(command, "--served-model-name") == "deepseek-v2-lite-afd-ffn"


def test_runner_uses_plugin_decode_bench_connector():
    args = _args()
    args.use_decode_bench_connector = True

    command = build_vllm_command(args, role="attention")
    kv_transfer_config = json.loads(_arg_value(command, "--kv-transfer-config"))

    assert kv_transfer_config["kv_connector"] == "AFDDecodeBenchConnector"
    assert kv_transfer_config["kv_connector_module_path"] == (
        "tools.benchmarks.decode_bench"
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_runner_uses_auto_worker_selection_for_npu(role):
    args = _args()
    args.device_backend = "npu"

    command = build_vllm_command(args, role=role)

    assert "--worker-cls" not in command


def test_runner_builds_npu_async_cam_role_specific_topology():
    args = _args()
    args.device_backend = "npu"
    args.num_attention_ranks = 2
    args.num_ffn_ranks = 2
    args.attention_tp_size = 2
    args.ffn_tp_size = 1
    args.afd_connector = runner.ASYNC_AFD_CONNECTOR
    args.afd_async = True
    args.compute_gate_on_attention = True
    args.afd_connector_extra_config = [
        (
            '{"dynamicQuant":0,'
            '"async_moe_ubatching":false,"async_moe_num_ubatches":2,'
            '"async_moe_split":"request","attn_ranks_per_dp":2}'
        ),
    ]

    attention_command = build_vllm_command(args, role="attention")
    ffn_command = build_vllm_command(args, role="ffn")

    assert _arg_value(attention_command, "--data-parallel-size") == "1"
    assert _arg_value(attention_command, "--tensor-parallel-size") == "2"
    assert _arg_value(ffn_command, "--data-parallel-size") == "2"
    assert _arg_value(ffn_command, "--tensor-parallel-size") == "1"

    additional_config = json.loads(_arg_value(attention_command, "--additional-config"))
    afd_config = additional_config["afd"]
    assert afd_config["connector"] == runner.ASYNC_AFD_CONNECTOR
    assert afd_config["async"] is True
    assert afd_config["compute_gate_on_attention"] is True
    assert afd_config["num_attention_ranks"] == 2
    assert afd_config["num_ffn_ranks"] == 2
    assert afd_config["connector_extra_config"]["attn_ranks_per_dp"] == 2
    assert afd_config["connector_extra_config"]["async_moe_split"] == "request"


def test_runner_rejects_role_rank_count_not_divisible_by_tp():
    args = _args()
    args.attention_tp_size = 3

    with pytest.raises(ValueError, match="attention rank count"):
        runner.validate_topology(args, ["0", "1"], ["2", "3"])


def test_runner_drops_flashcomm_for_npu_role_without_tp(monkeypatch):
    args = _args()
    args.device_backend = "npu"
    args.ffn_tp_size = 1
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

    env = runner.build_env("2,3", args, role="ffn")

    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env


def test_runner_forces_gpu_v1_model_runner(monkeypatch):
    args = _args()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    env = runner.build_env("0,1", args, role="attention")

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_async_cam_env_registers_cam_vendor_before_existing_paths(monkeypatch):
    existing_opp = "/opt/existing/opp/vendor"
    existing_lib = "/opt/existing/lib"
    monkeypatch.setenv(
        "ASCEND_CUSTOM_OPP_PATH",
        f"{existing_opp}:{async_cam_e2e.CAM_VENDOR_PATH}",
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        ":".join(
            [
                existing_lib,
                str(async_cam_e2e.CAM_OP_API_LIB_PATH),
                str(async_cam_e2e.CAM_OP_API_PATH),
            ],
        ),
    )
    monkeypatch.setenv("HCCL_BUFFSIZE", "8192")

    env = async_cam_e2e._async_cam_env()

    assert env["ASCEND_CUSTOM_OPP_PATH"].split(":") == [
        str(async_cam_e2e.CAM_VENDOR_PATH),
        existing_opp,
    ]
    assert env["LD_LIBRARY_PATH"].split(":") == [
        str(async_cam_e2e.CAM_OP_API_PATH),
        str(async_cam_e2e.CAM_OP_API_LIB_PATH),
        existing_lib,
    ]
    assert env["HCCL_BUFFSIZE"] == async_cam_e2e.CAM_HCCL_BUFFSIZE


def test_runner_fails_fast_when_server_exits_before_api_is_ready():
    args = _args()
    process = argparse.Namespace(
        args=["vllm", "serve"],
        poll=lambda: 1,
    )

    with pytest.raises(RuntimeError, match="exited before Attention API was ready"):
        runner.wait_for_openai_api(args, [process])


def test_runner_sends_one_concurrent_request_per_attention_dp_rank(monkeypatch):
    args = _args()
    calls = []

    def fake_request_completion(_args):
        calls.append(_args)
        return {"id": len(calls)}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    responses = runner.request_completions(args)

    assert len(calls) == 2
    assert len(responses) == 2


def test_npu_eager_dbo_sends_enough_requests_for_two_ubatches(monkeypatch):
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command

    monkeypatch.setattr(npu_e2e, "_model_path", lambda: "/models/DeepSeek-V2-Lite")
    monkeypatch.setattr(npu_e2e.subprocess, "run", fake_run)

    npu_e2e._run_e2e(
        npus=["0", "1", "2", "3"],
        api_port_base=18400,
        afd_port=6279,
        dbo=True,
    )

    command = captured["command"]
    assert _arg_value(command, "--num-requests") == "4"
    assert _arg_value(command, "--request-concurrency") == "4"


def test_request_completion_includes_http_error_body(monkeypatch):
    args = _args()
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:18100/v1/completions",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"CUDA out of memory"}'),
    )

    def fake_urlopen(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        runner.request_completion(args)


def test_runner_rejects_partial_concurrent_failures(monkeypatch, capsys):
    args = _args()
    args.num_requests = 3
    calls = []

    def fake_request_completion(_args):
        calls.append(_args)
        if len(calls) == 2:
            raise RuntimeError("transient request failure")
        return {"id": len(calls)}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    with pytest.raises(RuntimeError, match="1/3 completion requests failed"):
        runner.request_completions(args)

    assert len(calls) == 3
    assert "transient request failure" in capsys.readouterr().err


def test_qwen3_moe_e2e_uses_model_specific_environment(monkeypatch):
    monkeypatch.setenv("AFD_GPU_E2E_MODEL", "/models/DeepSeek-V2-Lite")
    monkeypatch.setenv("AFD_QWEN3_MOE_E2E_MODEL", "/models/Qwen3-30B-A3B")

    assert qwen3_moe_e2e._model_path() == "/models/Qwen3-30B-A3B"
