# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-local CUDA cases; importing IDs never resolves model assets."""

from __future__ import annotations

from tests.e2e.case import (
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    Case,
    CaseEnvironment,
    Deployment,
    Gsm8kSettings,
    Gsm8kTask,
    Mrv2Check,
    Phase,
    Role,
)

SCENARIOS = (
    "baseline-graph",
    "afd-eager-2a1f",
    "afd-graph-2a1f",
    # Native hybrid GDN does not support DBO.
)
GSM8K_SAMPLES = 7
GSM8K_THRESHOLD = 0.27
CAPTURE_SIZE = 8
MAX_MODEL_LEN = 4096


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend != "gpu":
        raise ValueError("Qwen3.6 MoE CUDA E2E requires the gpu backend")
    profiles = {
        "baseline-graph": (True, True, 4, 0),
        "afd-eager-2a1f": (False, False, 2, 1),
        "afd-graph-2a1f": (False, True, 2, 1),
    }
    baseline, graph, attention_ranks, ffn_ranks = profiles[case_id]
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    return Case(
        id=case_id,
        model="Qwen3.6 MoE",
        served_model_name_prefix="qwen3-6-afd",
        deployment=Deployment(
            backend="gpu",
            attention=Role(attention_ranks, execution=execution),
            ffn=None if baseline else Role(ffn_ranks, execution=execution),
            connector=None if baseline else "P2pNcclAFDConnector",
            cudagraph_capture_size=CAPTURE_SIZE if graph else None,
            max_num_seqs=CAPTURE_SIZE if graph else None,
            max_num_batched_tokens=CAPTURE_SIZE if baseline else None,
        ),
        common_vllm_args=(
            f"--max-model-len={MAX_MODEL_LEN}",
            "--language-model-only",
        ),
        phases=(
            Phase(
                "evaluation",
                Gsm8kTask(
                    Gsm8kSettings(
                        sample_limit=GSM8K_SAMPLES,
                        threshold=GSM8K_THRESHOLD,
                    )
                ),
                (Mrv2Check(("attention", "ffn"), graph=True),)
                if not baseline and graph
                else (),
            ),
        ),
    )


if __name__ == "__main__":
    from tests.e2e.runner import case_main

    raise SystemExit(case_main(build_case, SCENARIOS))
