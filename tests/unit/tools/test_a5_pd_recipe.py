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
    for name in ("MAX_CUDAGRAPH_CAPTURE_SIZE", "CUDAGRAPH_CAPTURE_SIZES"):
        env.pop(name, None)
    env.pop("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", None)
    env.update(
        PYTHON_BIN=sys.executable,
        PREFILL_DP_SIZE="2",
        PREFILL_TP_SIZE="1",
        ATTENTION_RANKS="4",
        TENSOR_PARALLEL_SIZE="1",
        MAX_NUM_SEQS="8",
        MAX_NUM_BATCHED_TOKENS="4096",
        ASCEND_LOCAL_COMM_RES_PATH=str(resources),
    )
    return env, resources


@pytest.fixture
def mock_recipe(tmp_path):
    recipe_copy = tmp_path / "recipe"
    shutil.copytree(RECIPE, recipe_copy)
    with (recipe_copy / "common.sh").open("a") as file:
        file.write("""
# Test-only substitutes for model/runtime imports and service startup.
validate_model_config() { :; }
preflight_role() { configure_parallelism; validate_devices; configure_execution; }
run_role_service() {
  "$PYTHON_BIN" -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@"
}
""")
    return recipe_copy


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
    env["AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] = "1"
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
    "value,expected", [(None, "1"), ("", "1"), ("0", "0"), ("1", "1")]
)
def test_eager_u2_overlap_is_exported_without_disabling_dbo(local_env, value, expected):
    env, _ = local_env
    env.update(dict.fromkeys(GRAPH_SWITCHES, "0"))
    if value is not None:
        env["AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] = value
    script = """
source "$1"
EXECUTION_MODE=eager U_BATCHES=2
DBO_DECODE_TOKEN_THRESHOLD=2 DBO_PREFILL_TOKEN_THRESHOLD=12
configure_execution
bash -c 'printf "%s\\n" "$AFD_HCCL_EAGER_U2_STREAM_OVERLAP"'
printf '%s\\n' "${EXECUTION_ARGS[*]}" "${UBATCH_ARGS[*]}" "${SCHEDULING_ARGS[*]}"
"""
    result = subprocess.run(
        ["bash", "-c", script, "eager-u2-test", str(RECIPE / "common.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        expected,
        "--enforce-eager",
        "--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12",
        "--no-async-scheduling",
    ]


@pytest.mark.parametrize("value", ["2", "true"])
def test_eager_u2_overlap_rejects_invalid_values(local_env, value):
    env, _ = local_env
    env["AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] = value
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; EXECUTION_MODE=eager U_BATCHES=2; '
                "DBO_DECODE_TOKEN_THRESHOLD=2 DBO_PREFILL_TOKEN_THRESHOLD=12; "
                "configure_execution"
            ),
            "eager-u2-test",
            str(RECIPE / "common.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "AFD_HCCL_EAGER_U2_STREAM_OVERLAP must be 0 or 1" in result.stderr


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
    local_env, tmp_path, mock_recipe, role, pd, expected_kv_role
):
    env, resources = local_env
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
        ["bash", str(mock_recipe / f"afd_{role}.sh")],
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


def test_attention_dspark_uses_v023_mtp_compat_method(local_env, tmp_path, mock_recipe):
    env, _ = local_env

    model = tmp_path / "dspark-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"dspark_block_size": 5}))
    env.update(
        MODEL_PATH=str(model),
        NIC_NAME="test0",
        HCCL_IF_IP="192.0.2.1",
        ENABLE_PD="0",
        ENABLE_DSPARK="1",
        ATTENTION_DEVICES="2,3,4,5",
        FFN_DEVICES="6,7",
        EXECUTION_MODE="eager",
        U_BATCHES="1",
    )
    env.pop("ASCEND_LOCAL_COMM_RES_PATH")

    result = subprocess.run(
        ["bash", str(mock_recipe / "afd_attention.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    config = json.loads(argv[argv.index("--speculative-config") + 1])
    assert config == {
        "method": "mtp",
        "num_speculative_tokens": 5,
        "draft_sample_method": "greedy",
        "enforce_eager": True,
    }


@pytest.mark.parametrize(
    "seqs,spec,maximum,sizes",
    [
        (8, 0, 8, [1, 2, 4, 8]),
        (24, 0, 24, [1, 2, 4, 8, 16, 24]),
        (24, 5, 144, [6, 12, 24, 48, 96, 144]),
        (48, 0, 48, [1, 2, 4, 8, 16, 32, 48]),
    ],
)
def test_graph_defaults_cover_full_batch_before_u2(
    local_env, seqs, spec, maximum, sizes
):
    env, _ = local_env
    env.update(MAX_NUM_SEQS=str(seqs), DSPARK_BLOCK_SIZE=str(spec))
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; configure_graph_capture; '
                'printf "%s\\n" "$MAX_CUDAGRAPH_CAPTURE_SIZE" '
                '"$CUDAGRAPH_CAPTURE_SIZES"'
            ),
            "graph-shape-test",
            str(RECIPE / "common.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    max_line, sizes_line = result.stdout.splitlines()
    assert int(max_line) == maximum
    assert [int(size) for size in sizes_line.split()] == sizes


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"MAX_CUDAGRAPH_CAPTURE_SIZE": "8"}, "does not cover"),
        ({"CUDAGRAPH_CAPTURE_SIZES": "1 2 4 8"}, "does not cover"),
        (
            {"MAX_CUDAGRAPH_CAPTURE_SIZE": "24", "CUDAGRAPH_CAPTURE_SIZES": "1 8"},
            "largest capture size",
        ),
        ({"MAX_CUDAGRAPH_CAPTURE_SIZE": "0"}, "must be positive"),
        ({"CUDAGRAPH_CAPTURE_SIZES": "0 24"}, "must be positive"),
        ({"MAX_CUDAGRAPH_CAPTURE_SIZE": "bad"}, "must be integers"),
        (
            {"CUDAGRAPH_CAPTURE_SIZES": "1 2 4 8 16 32"},
            "must include full Decode shape 24",
        ),
        (
            {"DSPARK_BLOCK_SIZE": "5", "MAX_CUDAGRAPH_CAPTURE_SIZE": "146"},
            "multiple of decode query length 6",
        ),
        (
            {"DSPARK_BLOCK_SIZE": "5", "MAX_NUM_BATCHED_TOKENS": "100"},
            "exceeds MAX_NUM_BATCHED_TOKENS",
        ),
    ],
)
def test_invalid_graph_overrides_fail_before_start(local_env, overrides, error):
    env, _ = local_env
    env.update(MAX_NUM_SEQS="24", DSPARK_BLOCK_SIZE="0")
    env.update(overrides)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; configure_graph_capture',
            "graph-shape-test",
            str(RECIPE / "common.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert error in result.stderr


@pytest.mark.parametrize("dspark,maximum", [("0", 24), ("1", 144)])
def test_attention_passes_derived_graph_sizes_to_vllm(
    local_env, tmp_path, mock_recipe, dspark, maximum
):
    env, _ = local_env
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"dspark_block_size": 5}))
    env.update(
        MODEL_PATH=str(model),
        HCCL_IF_IP="192.0.2.1",
        EXECUTION_MODE="full-decode-only",
        U_BATCHES="2",
        ENABLE_DSPARK=dspark,
        MAX_NUM_SEQS="24",
    )
    result = subprocess.run(
        ["bash", str(mock_recipe / "afd_attention.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    assert int(argv[argv.index("--max-cudagraph-capture-size") + 1]) == maximum
    sizes_start = argv.index("--cudagraph-capture-sizes") + 1
    sizes_end = argv.index("--compilation-config")
    sizes = [int(size) for size in argv[sizes_start:sizes_end]]
    assert sizes[-1] == maximum
    assert ("--speculative-config" in argv) == (dspark == "1")


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "a,f,tp", [(4, 2, 1), (4, 4, 1), (4, 4, 2), (4, 8, 1), (8, 8, 2)]
)
def test_launchers_derive_dp_from_physical_ranks(
    local_env, mock_recipe, role, a, f, tp
):
    env, _ = local_env
    env.update(
        MODEL_PATH="/unused/model",
        EXECUTION_MODE="eager",
        U_BATCHES="1",
        ENABLE_DSPARK="0",
        ENABLE_PD="0",
        ATTENTION_RANKS=str(a),
        FFN_RANKS=str(f),
        TENSOR_PARALLEL_SIZE=str(tp),
        ATTENTION_HOST_IP="192.0.2.1",
        FFN_HOST_IP="192.0.2.2",
        HCCL_IF_IP="192.0.2.1" if role == "attention" else "192.0.2.2",
        AFD_HOST="192.0.2.2",
        ATTENTION_DEVICES=",".join(map(str, range(a))),
        FFN_DEVICES=",".join(map(str, range(f))),
    )
    result = subprocess.run(
        ["bash", str(mock_recipe / f"afd_{role}.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    ranks = a if role == "attention" else f
    assert int(argv[argv.index("--data-parallel-size") + 1]) == ranks // tp
    assert int(argv[argv.index("--tensor-parallel-size") + 1]) == tp
    config = json.loads(argv[argv.index("--additional-config") + 1])["afd"]
    assert (config["num_attention_ranks"], config["num_ffn_ranks"]) == (a, f)


@pytest.mark.parametrize(
    "a,f,tp,error",
    [
        (0, 2, 1, "1..8"),
        (9, 2, 1, "1..8"),
        (4, 2, 4, "=1 or 2"),
        (3, 3, 2, "divisible"),
        (6, 4, 1, "integer multiple"),
        (4, 2, 2, "requires equal"),
    ],
)
def test_unsupported_parallelism_is_rejected(local_env, a, f, tp, error):
    env, _ = local_env
    env.update(
        ATTENTION_RANKS=str(a),
        FFN_RANKS=str(f),
        TENSOR_PARALLEL_SIZE=str(tp),
        ROLE="attention",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; configure_parallelism',
            "parallel-test",
            str(RECIPE / "common.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert error in result.stderr


@pytest.mark.parametrize("role", ["prefill", "attention"])
def test_tp2_pd_launchers_use_physical_devices_and_matching_kv_topology(
    local_env, mock_recipe, role
):
    env, _ = local_env
    env.update(
        MODEL_PATH="/unused/model",
        NIC_NAME="test0",
        HCCL_IF_IP="192.0.2.1",
        ENABLE_PD="1",
        ENABLE_DSPARK="0",
        EXECUTION_MODE="eager",
        U_BATCHES="1",
        PREFILL_DP_SIZE="2",
        PREFILL_TP_SIZE="2",
        PREFILL_DEVICES="0,1,2,3",
        ATTENTION_RANKS="4",
        FFN_RANKS="4",
        TENSOR_PARALLEL_SIZE="2",
        ATTENTION_DEVICES="0,1,2,3",
        FFN_DEVICES="4,5,6,7",
        PREFILL_ENGINE_ID="producer",
        PREFILL_KV_PORT="30000",
        DECODE_ENGINE_ID="consumer",
        DECODE_KV_PORT="31000",
    )
    result = subprocess.run(
        ["bash", str(mock_recipe / f"afd_{role}.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    assert argv[argv.index("--data-parallel-size") + 1] == "2"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    extra = json.loads(argv[argv.index("--kv-transfer-config") + 1])[
        "kv_connector_extra_config"
    ]
    assert extra["prefill"] == {"dp_size": 2, "tp_size": 2}
    assert extra["decode"] == {"dp_size": 2, "tp_size": 2}


def test_pd_rejects_prefill_tp_smaller_than_decode(local_env):
    env, _ = local_env
    env["TENSOR_PARALLEL_SIZE"] = "2"
    result = build_config(env)
    assert result.returncode != 0
    assert "Prefill TP >= Attention TP" in result.stderr


def test_ffn_graph_defaults_do_not_inherit_attention_speculation(
    local_env, mock_recipe
):
    env, _ = local_env
    env.update(
        MODEL_PATH="/unused/model",
        HCCL_IF_IP="192.0.2.1",
        ENABLE_DSPARK="0",
        DSPARK_BLOCK_SIZE="5",
        EXECUTION_MODE="full-decode-only",
        U_BATCHES="2",
        MAX_NUM_SEQS="48",
    )
    result = subprocess.run(
        ["bash", str(mock_recipe / "afd_ffn.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    assert argv[argv.index("--max-cudagraph-capture-size") + 1] == "48"
    assert "--speculative-config" not in argv


def test_same_host_device_overlap_still_fails(local_env, mock_recipe):
    env, _ = local_env
    env.update(
        MODEL_PATH="/unused/model",
        HCCL_IF_IP="192.0.2.1",
        ATTENTION_RANKS="4",
        FFN_RANKS="4",
        ATTENTION_DEVICES="0,1,2,3",
        FFN_DEVICES="2,3,4,5",
    )
    result = subprocess.run(
        ["bash", str(mock_recipe / "afd_ffn.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "overlap at device 2" in result.stderr


@pytest.mark.parametrize("tp,dp", [("1", 8), ("2", 4)])
def test_pd_consumer_metadata_is_not_fixed_to_a4(local_env, tp, dp):
    env, _ = local_env
    env.update(ATTENTION_RANKS="8", TENSOR_PARALLEL_SIZE=tp, PREFILL_TP_SIZE=tp)
    result = build_config(env, devices="0,1,2,3,4,5,6,7")
    assert result.returncode == 0, result.stderr
    extra = json.loads(result.stdout)["kv_connector_extra_config"]
    assert extra["decode"] == {"dp_size": dp, "tp_size": int(tp)}


def test_explicit_graph_buckets_are_preserved_with_speculative_alignment(local_env):
    env, _ = local_env
    env.update(
        MAX_NUM_SEQS="24",
        DSPARK_BLOCK_SIZE="5",
        CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 16 24 48 96 144",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; configure_graph_capture; '
                'printf "%s\\n" "$MAX_CUDAGRAPH_CAPTURE_SIZE" '
                '"$CUDAGRAPH_CAPTURE_SIZES"'
            ),
            "graph-shape-test",
            str(RECIPE / "common.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["144", "6 12 18 24 48 96 144"]


@pytest.mark.parametrize(
    "spec,maximum,required", [("0", "32", "24"), ("5", "192", "144")]
)
def test_larger_graph_max_keeps_a_bucket_at_the_decode_sequence_limit(
    local_env, spec, maximum, required
):
    env, _ = local_env
    env.update(
        MAX_NUM_SEQS="24", DSPARK_BLOCK_SIZE=spec, MAX_CUDAGRAPH_CAPTURE_SIZE=maximum
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; configure_graph_capture; '
                'printf "%s\\n" "$CUDAGRAPH_CAPTURE_SIZES"'
            ),
            "graph-shape-test",
            str(RECIPE / "common.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert required in result.stdout.split()
