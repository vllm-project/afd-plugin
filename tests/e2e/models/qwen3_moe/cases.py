# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-local CUDA cases; importing IDs never resolves model assets."""

from __future__ import annotations

import sys
from dataclasses import replace

from tests.e2e.case import (
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    Case,
    CaseEnvironment,
    DboConfig,
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
    "afd-graph-dbo-2a1f",
)
DEEPEP_M2N_SCENARIOS = (
    "afd-deepep-m2n-eager-2a2f",
    "afd-deepep-m2n-graph-2a2f",
    "afd-deepep-m2n-graph-dbo-2a2f",
    "afd-deepep-m2n-eager-tp2-2a2f",
    "afd-deepep-m2n-graph-tp2-2a2f",
)
MULTI_POD_IDS = (
    "afd-deepep-m2n-eager-2a2f",
    "afd-deepep-m2n-graph-2a2f",
    "afd-deepep-m2n-graph-dbo-2a2f",
)
MULTI_POD_GSM8K_SAMPLES = 100

GSM8K_SAMPLES = 7
GSM8K_THRESHOLD = 0.27
DBO_CONCURRENCY = 12
DBO_MINIMUM_SAMPLES = 2 * DBO_CONCURRENCY
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 8
CAPTURE_SIZE = 8
MAX_MODEL_LEN = 4096


def build_multi_pod_case(case_id: str, environment: CaseEnvironment) -> Case:
    if case_id not in MULTI_POD_IDS:
        raise ValueError(f"unsupported GPU multi-pod case {case_id!r}")
    case = build_case(case_id, environment)
    dbo = case.deployment.dbo is not None
    return replace(
        case,
        # Preserve the served name used by the existing distributed entry.
        served_model_name_prefix="deepseek-v2-lite-afd",
        phases=(
            Phase(
                "evaluation",
                Gsm8kTask(
                    Gsm8kSettings(
                        sample_limit=MULTI_POD_GSM8K_SAMPLES,
                        threshold=GSM8K_THRESHOLD,
                        num_concurrent=DBO_CONCURRENCY if dbo else None,
                        minimum_samples=DBO_MINIMUM_SAMPLES if dbo else 1,
                    )
                ),
            ),
        ),
    )


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend != "gpu":
        raise ValueError("Qwen3 MoE CUDA E2E requires the gpu backend")
    profiles = {
        "baseline-graph": (True, True, False, 4, 1, 0, 1),
        "afd-eager-2a1f": (False, False, False, 2, 1, 1, 1),
        "afd-graph-2a1f": (False, True, False, 2, 1, 1, 1),
        "afd-graph-dbo-2a1f": (False, True, True, 2, 1, 1, 1),
        "afd-deepep-m2n-eager-2a2f": (False, False, False, 2, 1, 2, 1),
        "afd-deepep-m2n-graph-2a2f": (False, True, False, 2, 1, 2, 1),
        "afd-deepep-m2n-graph-dbo-2a2f": (False, True, True, 2, 1, 2, 1),
        "afd-deepep-m2n-eager-tp2-2a2f": (False, False, False, 1, 2, 2, 1),
        "afd-deepep-m2n-graph-tp2-2a2f": (False, True, False, 1, 2, 2, 1),
    }
    baseline, graph, dbo, a_dp, a_tp, f_dp, f_tp = profiles[case_id]
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    deepep = case_id in DEEPEP_M2N_SCENARIOS
    return Case(
        id=case_id,
        model="Qwen3 MoE",
        served_model_name_prefix="qwen3-moe-afd",
        deployment=Deployment(
            backend="gpu",
            attention=Role(a_dp, a_tp, execution),
            ffn=None if baseline else Role(f_dp, f_tp, execution),
            connector=None
            if baseline
            else "DeepEPM2NAFDConnector"
            if deepep
            else "P2pNcclAFDConnector",
            compute_gate_on_attention=deepep,
            dbo=DboConfig(DBO_DECODE_THRESHOLD, DBO_PREFILL_THRESHOLD) if dbo else None,
            cudagraph_capture_size=CAPTURE_SIZE if graph else None,
            max_num_seqs=CAPTURE_SIZE if graph else None,
            max_num_batched_tokens=CAPTURE_SIZE if baseline else None,
        ),
        common_vllm_args=(f"--max-model-len={MAX_MODEL_LEN}",),
        phases=(
            Phase(
                "evaluation",
                Gsm8kTask(
                    Gsm8kSettings(
                        sample_limit=DBO_MINIMUM_SAMPLES if dbo else GSM8K_SAMPLES,
                        threshold=GSM8K_THRESHOLD,
                        num_concurrent=DBO_CONCURRENCY if dbo else None,
                        minimum_samples=DBO_MINIMUM_SAMPLES if dbo else 1,
                    )
                ),
                (Mrv2Check(("attention", "ffn"), graph=graph, dbo=dbo),)
                if not baseline and (graph or dbo)
                else (),
            ),
        ),
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["multi-pod"]:
        from tests.e2e.multi_pod.runner import case_main as multi_pod_case_main

        raise SystemExit(
            multi_pod_case_main(build_multi_pod_case, MULTI_POD_IDS, sys.argv[2:])
        )
    from tests.e2e.runner import case_main

    raise SystemExit(case_main(build_case, (*SCENARIOS, *DEEPEP_M2N_SCENARIOS)))
