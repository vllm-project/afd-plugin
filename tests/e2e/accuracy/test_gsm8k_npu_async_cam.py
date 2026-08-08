# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GSM8K accuracy evaluation for CAMAsyncAFDConnector on Ascend NPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from afd_plugin.envs import force_balanced_topk_ids_enabled
from tests.e2e.conftest import AFDServer, _launch_afd_server
from tests.e2e.helpers_gsm8k import (
    _extract_gsm8k_accuracy,
    _run_lm_eval,
)

ASYNC_MOE_NUM_STAGES = 2
DEEPSEEK_V3_2_MODEL_TYPE = "deepseek_v32"
DEEPSEEK_V3_2_ARCHITECTURE = "DeepseekV32ForCausalLM"
DEEPSEEK_V3_2_NUM_HIDDEN_LAYERS = 61


def _default_tasks_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.parent / "lm-evaluation-harness" / "lm_eval" / "tasks" / "gsm8k"


def _npu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get(
            "AFD_NPU_ASYNC_CAM_E2E_DEVICES",
            "0,1,2,3,4,5,6,7",
        ).split(",")
        if item.strip()
    ]


def _validate_full_deepseek_v3_2_model(model: str) -> None:
    """Reject reduced or non-V3.2 checkpoints before the expensive evaluation."""
    config_path = Path(model) / "config.json"
    assert config_path.is_file(), (
        f"DeepSeek-V3.2 E2E requires a local checkpoint with config.json; got {model!r}"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    assert (
        config.get("model_type") == DEEPSEEK_V3_2_MODEL_TYPE
        or DEEPSEEK_V3_2_ARCHITECTURE in architectures
    ), f"expected a DeepSeek-V3.2 checkpoint, got config from {config_path}"
    assert config.get("num_hidden_layers") == DEEPSEEK_V3_2_NUM_HIDDEN_LAYERS, (
        "DeepSeek-V3.2 E2E must use all 61 layers; "
        f"got num_hidden_layers={config.get('num_hidden_layers')!r}"
    )


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
@pytest.mark.parametrize(
    "enable_attention_sp",
    (True, False),
    ids=("sp", "no-sp"),
)
@pytest.mark.parametrize(
    ("split", "async_moe_ubatching"),
    (
        ("token", True),
        ("request", True),
        ("request", False),
    ),
    ids=("token-ubatch", "request-ubatch", "no-ubatch"),
)
@pytest.mark.skipif(
    os.environ.get("AFD_NPU_ASYNC_CAM_RUN_DP3TP2") != "1",
    reason=(
        "DP3TP2+EP2 GSM8K test is opt-in; set AFD_NPU_ASYNC_CAM_RUN_DP3TP2=1 to enable"
    ),
)
def test_gsm8k_lm_eval_async_cam_dp3tp2_ep2(
    npu_available: bool,
    npu_e2e_model: str,
    npu_vllm_bin: str,
    tmp_path: Path,
    split: str,
    enable_attention_sp: bool,
    async_moe_ubatching: bool,
) -> None:
    """Check six SP and async-MoE modes on DP3TP2+DP2TP1/EP2."""

    pytest.importorskip("lm_eval", reason="lm-eval not installed")
    npus = _npu_list()
    if len(npus) < 8:
        pytest.skip(
            f"async CAM DP3TP2+EP2 GSM8K test requires 8 NPUs; got {len(npus)}",
        )
    tasks_dir = os.environ.get("AFD_NPU_GSM8K_TASK_DIR") or str(
        _default_tasks_dir(),
    )
    if not Path(tasks_dir).is_dir():
        pytest.skip(
            f"offline gsm8k task dir not found: {tasks_dir}. "
            "Stage gsm8k parquet + gsm8k.yaml, or set "
            "AFD_NPU_GSM8K_TASK_DIR.",
        )

    threshold = float(os.environ.get("AFD_GSM8K_THRESHOLD", "0.20"))
    tolerance = float(os.environ.get("AFD_GSM8K_TOLERANCE", "0.05"))
    configured_limit = os.environ.get("AFD_GSM8K_LIMIT")
    limit = int(configured_limit) if configured_limit else None
    configured_batch_size = os.environ.get("AFD_GSM8K_BATCH_SIZE")
    effective_batch_size = (
        int(configured_batch_size) if configured_batch_size is not None else 8
    )
    connector_settings: dict[str, bool | int | str] = {
        "dynamicQuant": 0,
        "attn_ranks_per_dp": 2,
    }
    if async_moe_ubatching:
        connector_settings.update(
            {
                "async_moe_ubatching": True,
                "async_moe_num_ubatches": ASYNC_MOE_NUM_STAGES,
                "async_moe_split": split,
            },
        )
    connector_extra_config = json.dumps(
        connector_settings,
        separators=(",", ":"),
    )
    common_vllm_args = [
        "--trust-remote-code",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "8000",
        "--no-enable-prefix-caching",
    ]
    max_model_len = os.environ.get("AFD_NPU_ASYNC_CAM_E2E_MAX_MODEL_LEN")
    if max_model_len:
        common_vllm_args.extend(["--max-model-len", max_model_len])

    # FlashComm1 is an Attention-local token layout. Set both roles explicitly
    # so the plain DP+TP test cannot inherit an ambient SP setting and FFN never
    # participates in sequence parallelism.
    attention_env = {
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1" if enable_attention_sp else "0",
    }
    ffn_env = {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}

    afd_server: AFDServer | None = None
    try:
        afd_server = _launch_afd_server(
            model=npu_e2e_model,
            backend="npu",
            vllm_bin=npu_vllm_bin,
            connector="CAMAsyncAFDConnector",
            attention_devices=npus[:6],
            ffn_devices=npus[6:8],
            attention_tp_size=2,
            ffn_tp_size=1,
            afd_async=True,
            compute_gate_on_attention=True,
            afd_connector_extra_config=[connector_extra_config],
            api_port_base=int(
                os.environ.get("AFD_NPU_ASYNC_CAM_E2E_API_PORT", "19080"),
            ),
            afd_port=int(
                os.environ.get("AFD_NPU_ASYNC_CAM_E2E_AFD_PORT", "6453"),
            ),
            startup_timeout=float(
                os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "900"),
            ),
            served_model_name_prefix="deepseek-v2-lite-afd",
            common_vllm_args=common_vllm_args,
            attention_env=attention_env,
            ffn_env=ffn_env,
        )
        run_mode = "-".join(
            (
                split,
                "sp" if enable_attention_sp else "no-sp",
                "ubatch" if async_moe_ubatching else "no-ubatch",
            ),
        )
        results = _run_lm_eval(
            base_url=afd_server.base_url,
            model_name=afd_server.served_model,
            output_path=str(tmp_path / f"lm_eval_output_{run_mode}"),
            tokenizer=npu_e2e_model,
            tasks_dir=tasks_dir,
            limit=limit,
            batch_size=effective_batch_size,
        )
        accuracy = _extract_gsm8k_accuracy(results)
        effective_threshold = threshold - tolerance
        print(
            f"\n[GSM8K NPU async CAM mode={run_mode}] "
            f"accuracy={accuracy:.4f} threshold={threshold} "
            f"tolerance={tolerance} effective_min={effective_threshold:.4f} "
            f"batch_size={effective_batch_size}",
        )
        assert accuracy >= effective_threshold, (
            f"GSM8K async CAM ({run_mode=}) accuracy {accuracy:.4f} < "
            f"effective threshold {effective_threshold:.4f} "
            f"(threshold={threshold}, tolerance={tolerance})"
        )
    finally:
        if afd_server is not None:
            afd_server.shutdown()


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.eval
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("AFD_NPU_ASYNC_CAM_RUN_V3_2_DP2TP8_EP16") != "1",
    reason=(
        "DeepSeek-V3.2 DP2TP8+EP16 GSM8K test is opt-in; set "
        "AFD_NPU_ASYNC_CAM_RUN_V3_2_DP2TP8_EP16=1 to enable"
    ),
)
def test_gsm8k_lm_eval_async_cam_v3_2_dp2tp8_ep16(
    npu_available: bool,
    npu_e2e_model: str,
    tmp_path: Path,
) -> None:
    """Evaluate the externally launched two-node, full-model deployment."""
    pytest.importorskip("lm_eval", reason="lm-eval not installed")
    _validate_full_deepseek_v3_2_model(npu_e2e_model)
    assert not force_balanced_topk_ids_enabled(), (
        "DeepSeek-V3.2 accuracy must run with AFD_FORCE_BALANCED_TOPK_IDS disabled"
    )

    tasks_dir = os.environ.get("AFD_NPU_GSM8K_TASK_DIR") or str(
        _default_tasks_dir(),
    )
    if not Path(tasks_dir).is_dir():
        pytest.skip(
            f"offline gsm8k task dir not found: {tasks_dir}. "
            "Stage gsm8k parquet + gsm8k.yaml, or set "
            "AFD_NPU_GSM8K_TASK_DIR.",
        )

    base_url = os.environ.get(
        "AFD_NPU_ASYNC_CAM_V3_2_BASE_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")
    model_name = os.environ.get(
        "AFD_NPU_ASYNC_CAM_V3_2_SERVED_MODEL",
        "deepseek_v3_2",
    )
    threshold = float(
        os.environ.get("AFD_NPU_ASYNC_CAM_V3_2_GSM8K_THRESHOLD", "0.80"),
    )
    tolerance = float(
        os.environ.get("AFD_NPU_ASYNC_CAM_V3_2_GSM8K_TOLERANCE", "0.05"),
    )
    configured_limit = os.environ.get("AFD_GSM8K_LIMIT")
    limit = int(configured_limit) if configured_limit else None
    batch_size = int(os.environ.get("AFD_GSM8K_BATCH_SIZE", "8"))

    results = _run_lm_eval(
        base_url=base_url,
        model_name=model_name,
        output_path=str(tmp_path / "lm_eval_output_v3_2_dp2tp8_ep16"),
        tokenizer=npu_e2e_model,
        tasks_dir=tasks_dir,
        limit=limit,
        batch_size=batch_size,
    )
    accuracy = _extract_gsm8k_accuracy(results)
    effective_threshold = threshold - tolerance
    print(
        "\n[GSM8K NPU async CAM DeepSeek-V3.2 DP2TP8+EP16] "
        f"accuracy={accuracy:.4f} threshold={threshold} "
        f"tolerance={tolerance} effective_min={effective_threshold:.4f} "
        f"batch_size={batch_size}",
    )
    assert accuracy >= effective_threshold, (
        f"DeepSeek-V3.2 GSM8K accuracy {accuracy:.4f} < effective threshold "
        f"{effective_threshold:.4f} (threshold={threshold}, "
        f"tolerance={tolerance})"
    )
