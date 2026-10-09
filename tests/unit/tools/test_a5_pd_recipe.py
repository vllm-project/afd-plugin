# SPDX-License-Identifier: Apache-2.0
"""Validate A5 PD configuration and launcher wiring without importing NPU code."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = (
    Path(__file__).resolve().parents[3] / "recipe/npu/P2pHcclAFDConnector/deepseek_v4"
)
GRAPH_SWITCHES = (
    "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP",
    "AFD_HCCL_GRAPH_U2_HYBRID_DAG",
    "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
    "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
    "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
)


@pytest.fixture
def local_env(tmp_path):
    resources = tmp_path / 'local resources "quoted"'
    resources.mkdir()
    for device in range(8):
        (resources / f"ub_endpoint_npu_{device}.json").write_text(
            json.dumps({"device": device})
        )
    env = dict(os.environ)
    for name in GRAPH_SWITCHES:
        env.pop(name, None)
    env.update(
        PYTHON_BIN=sys.executable,
        PREFILL_DP_SIZE="2",
        PREFILL_TP_SIZE="1",
        ATTENTION_RANKS="4",
        ASCEND_LOCAL_COMM_RES_PATH=str(resources),
    )
    return env, resources


def build_config(env, role="kv_consumer", devices="2,3,4,5"):
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; build_a5_pd_kv_config "$2" "$3" "$4" "$5"',
            "a5-pd-test",
            str(RECIPE / "common.sh"),
            role,
            'engine-"quoted"',
            "31000",
            devices,
        ],
        env=env,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "role,devices,dp",
    [
        ("kv_producer", "0,1", 2),
        ("kv_producer", "0,1,2,3,4,5,6,7", 8),
        ("kv_consumer", "2,3,4,5", 2),
    ],
)
def test_local_resource_path_is_escaped_and_role_metadata_is_preserved(
    local_env, role, devices, dp
):
    env, resources = local_env
    env["PREFILL_DP_SIZE"] = str(dp)
    result = build_config(env, role, devices)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    extra = config["kv_connector_extra_config"]
    assert extra["ascend_local_comm_res_path"] == str(resources)
    assert extra["prefill"] == {"dp_size": dp, "tp_size": 1}
    assert extra["decode"] == {"dp_size": 4, "tp_size": 1}
    assert config["kv_role"] == role
    assert config["kv_port"] == 31000
    assert config["engine_id"] == 'engine-"quoted"'


@pytest.mark.parametrize("value", ["", "relative/hixlep", "/nonexistent/a5-hixlep"])
def test_missing_or_relative_resource_path_is_rejected(local_env, value):
    env, _ = local_env
    env["ASCEND_LOCAL_COMM_RES_PATH"] = value
    result = build_config(env)
    assert result.returncode != 0
    assert "existing local absolute directory" in result.stderr


def test_endpoint_checks_use_physical_devices_not_local_rank(local_env):
    env, resources = local_env
    (resources / "ub_endpoint_npu_0.json").unlink()
    result = build_config(env)
    assert result.returncode == 0, result.stderr
    (resources / "ub_endpoint_npu_5.json").unlink()
    result = build_config(env)
    assert result.returncode != 0
    assert "ub_endpoint_npu_5.json" in result.stderr


def test_invalid_endpoint_json_is_rejected(local_env):
    env, resources = local_env
    (resources / "ub_endpoint_npu_2.json").write_text("invalid")
    result = build_config(env)
    assert result.returncode != 0
    assert "unreadable or invalid" in result.stderr


@pytest.mark.parametrize("devices", ["2,3,4", "2,3,4,4", "2,3,4,8"])
def test_device_mapping_must_match_a5_role(local_env, devices):
    env, _ = local_env
    result = build_config(env, devices=devices)
    assert result.returncode != 0
    assert "device list does not match" in result.stderr


@pytest.mark.parametrize("mode", ["defaults", "all-off", "mixed"])
def test_graph_default_and_override_values_survive_configuration(local_env, mode):
    env, _ = local_env
    expected = ["1"] * len(GRAPH_SWITCHES)
    if mode != "defaults":
        expected = ["0"] * len(GRAPH_SWITCHES)
        if mode == "mixed":
            expected[1] = expected[3] = "1"
        env.update(zip(GRAPH_SWITCHES, expected, strict=True))
    script = """
set -euo pipefail
source "$1"
EXECUTION_MODE=full-decode-only U_BATCHES=2
CUDAGRAPH_CAPTURE_SIZES='1 2 4 8'
MAX_CUDAGRAPH_CAPTURE_SIZE=8
DBO_DECODE_TOKEN_THRESHOLD=2 DBO_PREFILL_TOKEN_THRESHOLD=12
configure_execution
shift
for name in "$@"; do printf '%s\\n' "${!name}"; done
test "$AFD_HCCL_EAGER_U2_STREAM_OVERLAP" = 0
test "${SCHEDULING_ARGS[0]}" = --no-async-scheduling
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "graph-test",
            str(RECIPE / "common.sh"),
            *GRAPH_SWITCHES,
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize(
    "role,pd,expected_kv_role",
    [
        ("prefill", "1", "kv_producer"),
        ("attention", "1", "kv_consumer"),
        ("attention", "0", None),
        ("ffn", "0", None),
    ],
)
def test_launchers_wire_only_a5_pd_producer_and_consumer(
    local_env, tmp_path, role, pd, expected_kv_role
):
    env, resources = local_env
    recipe_copy = tmp_path / "recipe"
    shutil.copytree(RECIPE, recipe_copy)
    with (recipe_copy / "common.sh").open("a") as file:
        file.write("""
# Test-only substitutes for model/runtime imports and service startup.
validate_model_config() { :; }
preflight_role() { configure_execution; }
run_role_service() {
  "$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}
""")
    env.update(
        MODEL_PATH=str(tmp_path / "model"),
        NIC_NAME="test0",
        HCCL_IF_IP="192.0.2.1",
        ENABLE_PD=pd,
        ENABLE_DSPARK="0",
        PREFILL_ENGINE_ID="producer",
        PREFILL_KV_PORT="30000",
        DECODE_ENGINE_ID="consumer",
        DECODE_KV_PORT="31000",
        PREFILL_DEVICES="0,1",
        ATTENTION_DEVICES="2,3,4,5",
        FFN_DEVICES="6,7",
        EXECUTION_MODE="eager",
        U_BATCHES="1",
    )
    if expected_kv_role is None:
        env.pop("ASCEND_LOCAL_COMM_RES_PATH")
    result = subprocess.run(
        ["bash", str(recipe_copy / f"afd_{role}.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    if expected_kv_role is None:
        assert "--kv-transfer-config" not in argv
    else:
        config = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        assert config["kv_role"] == expected_kv_role
        assert config["kv_connector_extra_config"]["ascend_local_comm_res_path"] == str(
            resources
        )
