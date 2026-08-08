# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen3 MoE CUDA AFD E2E execution matrix.

The focused cases cover startup, request pressure, CUDA graphs, DBO,
data-parallel AFD topologies, tensor parallelism, and asymmetric
Attention/FFN ranks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "tests" / "e2e" / "runner.py"
DEFAULT_GRAPH_CAPTURE_SIZE = 8
DEFAULT_PROMPT = "San Francisco is a"


@dataclass(frozen=True)
class Qwen3MoeE2ECase:
    name: str
    attention_ranks: int = 1
    ffn_ranks: int = 1
    tp_size: int = 1
    graph_capture_size: int = 0
    dbo: bool = False
    requests: int = 1
    concurrency: int = 1
    max_tokens: int = 4


QWEN3_MOE_GPU_E2E_CASES = (
    Qwen3MoeE2ECase("1a1f_eager_smoke"),
    Qwen3MoeE2ECase(
        "1a1f_eager_concurrency32",
        requests=64,
        concurrency=32,
    ),
    Qwen3MoeE2ECase(
        "1a1f_graph_capture8",
        graph_capture_size=DEFAULT_GRAPH_CAPTURE_SIZE,
        requests=8,
        concurrency=8,
    ),
    Qwen3MoeE2ECase(
        "1a1f_dbo_batch8",
        dbo=True,
        requests=8,
        concurrency=8,
        max_tokens=8,
    ),
    Qwen3MoeE2ECase("2a2f_eager_smoke", attention_ranks=2, ffn_ranks=2),
    Qwen3MoeE2ECase(
        "2a2f_graph_capture8",
        attention_ranks=2,
        ffn_ranks=2,
        graph_capture_size=DEFAULT_GRAPH_CAPTURE_SIZE,
        requests=8,
        concurrency=8,
    ),
    Qwen3MoeE2ECase(
        "2a2f_dbo_batch8",
        attention_ranks=2,
        ffn_ranks=2,
        dbo=True,
        requests=8,
        concurrency=8,
    ),
    Qwen3MoeE2ECase(
        "tp2_eager",
        attention_ranks=2,
        ffn_ranks=2,
        tp_size=2,
        requests=4,
        concurrency=4,
    ),
    Qwen3MoeE2ECase(
        "tp2_graph_capture8",
        attention_ranks=2,
        ffn_ranks=2,
        tp_size=2,
        graph_capture_size=DEFAULT_GRAPH_CAPTURE_SIZE,
        requests=8,
        concurrency=8,
    ),
    Qwen3MoeE2ECase(
        "2a1f_eager",
        attention_ranks=2,
        requests=4,
        concurrency=4,
    ),
)


def _model_path() -> str:
    model = os.environ.get("AFD_QWEN3_MOE_E2E_MODEL")
    if not model:
        pytest.skip("set AFD_QWEN3_MOE_E2E_MODEL to a Qwen3 MoE model path")
    return model


def _gpu_list() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("AFD_GPU_E2E_GPUS", "0,1,2,3").split(",")
        if item.strip()
    ]


def _run_qwen3_moe_gpu_e2e(
    case: Qwen3MoeE2ECase,
    case_index: int,
) -> None:
    gpus = _gpu_list()
    required_gpus = case.attention_ranks + case.ffn_ranks
    if len(gpus) < required_gpus:
        pytest.skip(f"{case.name} requires {required_gpus} GPUs; got {len(gpus)}")

    api_port = int(os.environ.get("AFD_QWEN3_E2E_API_PORT_BASE", "24000"))
    afd_port = int(os.environ.get("AFD_QWEN3_E2E_AFD_PORT_BASE", "17000"))
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        _model_path(),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--num-attention-ranks",
        str(case.attention_ranks),
        "--num-ffn-ranks",
        str(case.ffn_ranks),
        "--attention-gpus",
        ",".join(gpus[: case.attention_ranks]),
        "--ffn-gpus",
        ",".join(gpus[case.attention_ranks : required_gpus]),
        "--api-port-base",
        str(api_port + case_index * 10),
        "--afd-port",
        str(afd_port + case_index),
        "--tp-size",
        str(case.tp_size),
        "--startup-timeout",
        os.environ.get("AFD_GPU_E2E_STARTUP_TIMEOUT", "900"),
        "--max-tokens",
        str(case.max_tokens),
        "--prompt",
        DEFAULT_PROMPT,
        "--num-requests",
        str(case.requests),
        "--request-concurrency",
        str(case.concurrency),
        "--served-model-name-prefix",
        f"qwen3-moe-{case.name}",
        "--common-vllm-arg=--max-model-len",
        "--common-vllm-arg=1024",
        "--common-vllm-arg=--gpu-memory-utilization",
        "--common-vllm-arg="
        + os.environ.get("AFD_GPU_E2E_GPU_MEMORY_UTILIZATION", "0.85"),
    ]
    if case.graph_capture_size:
        command.extend(
            [
                "--cuda-graph-full-decode-only",
                "--cudagraph-capture-size",
                str(case.graph_capture_size),
            ],
        )
    if case.dbo:
        command.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                "1",
                "--dbo-prefill-token-threshold",
                "8",
            ],
        )

    subprocess.run(command, cwd=REPO_ROOT, check=True)


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(
    ("case_index", "case"),
    tuple(enumerate(QWEN3_MOE_GPU_E2E_CASES)),
    ids=[case.name for case in QWEN3_MOE_GPU_E2E_CASES],
)
def test_qwen3_moe_gpu_matrix(
    case_index: int,
    case: Qwen3MoeE2ECase,
) -> None:
    _run_qwen3_moe_gpu_e2e(
        case,
        case_index,
    )
