# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V2-Lite IDs and model-local Case construction.

Collection imports IDs only. build_case executes in the runner subprocess.
GPU and NPU declarations share capabilities, not scenario selection in the runner.
"""

from __future__ import annotations

import sys
from dataclasses import replace

from tests.e2e.case import (
    ATTENTION_ROLE,
    EXECUTION_EAGER,
    EXECUTION_FULL_DECODE_GRAPH,
    FFN_ROLE,
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

V2_SCENARIOS = (
    "afd-v2-eager-1a1f",
    "afd-v2-eager-dp2",
    "afd-v2-eager-tp2",
    "afd-v2-graph-1a1f",
    "afd-v2-graph-dp2",
    "afd-v2-graph-tp2",
    "afd-v2-eager-dbo-dp2",
    "afd-v2-graph-dbo-dp2",
    "afd-v2-graph-dbo-tp2",
)
DEEPEP_M2N_SCENARIOS = (
    "afd-deepep-m2n-eager-2a2f",
    "afd-deepep-m2n-graph-2a2f",
    "afd-deepep-m2n-graph-dbo-2a2f",
    "afd-deepep-m2n-eager-tp2-2a2f",
    "afd-deepep-m2n-graph-tp2-2a2f",
)
GPU_IDS = (
    "baseline-graph",
    *V2_SCENARIOS,
    "afd-eager-2a1f",
    "afd-graph-2a1f",
    "afd-graph-dbo-2a1f",
)
NPU_IDS = (
    "baseline-graph",
    "afd-eager-2a2f",
    "afd-graph-2a2f",
    "afd-graph-dbo-2a2f",
    "afd-eager-2a1f",
    "afd-graph-2a1f",
    "afd-graph-dbo-2a1f",
)
V2_DBO_COMPARISON_SCENARIOS = frozenset(
    ("afd-v2-eager-dp2", "afd-v2-eager-dbo-dp2", "afd-v2-graph-dbo-dp2"),
)
V2_SINGLE_RANK_SCENARIOS = frozenset(("afd-v2-eager-1a1f", "afd-v2-graph-1a1f"))
MULTI_POD_IDS = ("afd-graph-2a2f", "afd-graph-dbo-2a2f")
DEFAULT_GSM8K_SAMPLE_LIMIT = 7
V2_DBO_SAMPLE_LIMIT = 128
SMOKE_THRESHOLD = 0.25
ACCURACY_THRESHOLD = 0.27
SMOKE_MAX_SAMPLES = 24
COMPARISON_CONCURRENCY = 12
CAPTURE_SIZE = 8
MAX_MODEL_LEN = 4096
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 8


def stable_ids(backend: str) -> tuple[str, ...]:
    if backend == "gpu":
        return GPU_IDS
    if backend == "npu":
        return NPU_IDS
    raise ValueError(f"unknown backend {backend!r}")


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    backend = environment.device_backend
    allowed = stable_ids(backend) + (DEEPEP_M2N_SCENARIOS if backend == "gpu" else ())
    if case_id not in allowed:
        raise ValueError(f"unsupported {backend} DeepSeek-V2-Lite case {case_id!r}")
    return _build_case(case_id, environment)


def build_multi_pod_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend != "gpu" or case_id not in MULTI_POD_IDS:
        raise ValueError(f"unsupported GPU multi-pod case {case_id!r}")
    case = _build_case(case_id, environment)
    dbo = case.deployment.dbo is not None
    # These existing distributed cases gate accuracy, liveness and cleanup.
    # Their original 0.27 threshold is independent of the single-host smoke rule.
    return replace(
        case,
        phases=(
            Phase(
                "evaluation",
                Gsm8kTask(
                    Gsm8kSettings(
                        sample_limit=SMOKE_MAX_SAMPLES
                        if dbo
                        else DEFAULT_GSM8K_SAMPLE_LIMIT,
                        threshold=ACCURACY_THRESHOLD,
                        num_concurrent=COMPARISON_CONCURRENCY if dbo else None,
                        minimum_samples=SMOKE_MAX_SAMPLES if dbo else 1,
                    )
                ),
            ),
        ),
    )


def _build_case(case_id: str, environment: CaseEnvironment) -> Case:
    # The table is model-local. No shared module selects a model or case ID.
    # baseline, graph, dbo, Attention DP/TP, FFN DP/TP
    profiles = {
        "baseline-graph": (True, True, False, 4, 1, 0, 1),
        "afd-v2-eager-1a1f": (False, False, False, 1, 1, 1, 1),
        "afd-v2-eager-dp2": (False, False, False, 2, 1, 2, 1),
        "afd-v2-eager-tp2": (False, False, False, 1, 2, 1, 2),
        "afd-v2-graph-1a1f": (False, True, False, 1, 1, 1, 1),
        "afd-v2-graph-dp2": (False, True, False, 2, 1, 2, 1),
        "afd-v2-graph-tp2": (False, True, False, 1, 2, 1, 2),
        "afd-v2-eager-dbo-dp2": (False, False, True, 2, 1, 2, 1),
        "afd-v2-graph-dbo-dp2": (False, True, True, 2, 1, 2, 1),
        "afd-v2-graph-dbo-tp2": (False, True, True, 1, 2, 1, 2),
        "afd-eager-2a1f": (False, False, False, 2, 1, 1, 1),
        "afd-graph-2a1f": (False, True, False, 2, 1, 1, 1),
        "afd-graph-dbo-2a1f": (False, True, True, 2, 1, 1, 1),
        "afd-eager-2a2f": (False, False, False, 2, 1, 2, 1),
        "afd-graph-2a2f": (False, True, False, 2, 1, 2, 1),
        "afd-graph-dbo-2a2f": (False, True, True, 2, 1, 2, 1),
        "afd-deepep-m2n-eager-2a2f": (False, False, False, 2, 1, 2, 1),
        "afd-deepep-m2n-graph-2a2f": (False, True, False, 2, 1, 2, 1),
        "afd-deepep-m2n-graph-dbo-2a2f": (False, True, True, 2, 1, 2, 1),
        "afd-deepep-m2n-eager-tp2-2a2f": (False, False, False, 1, 2, 2, 1),
        "afd-deepep-m2n-graph-tp2-2a2f": (False, True, False, 1, 2, 2, 1),
    }
    backend = environment.device_backend
    baseline, graph, dbo, a_dp, a_tp, f_dp, f_tp = profiles[case_id]
    execution = EXECUTION_FULL_DECODE_GRAPH if graph else EXECUTION_EAGER
    comparison = case_id in V2_DBO_COMPARISON_SCENARIOS
    deepep = case_id in DEEPEP_M2N_SCENARIOS
    minimum_samples = SMOKE_MAX_SAMPLES if comparison or dbo else 1
    sample_limit = (
        V2_DBO_SAMPLE_LIMIT
        if comparison
        else max(DEFAULT_GSM8K_SAMPLE_LIMIT, minimum_samples)
    )
    # The TP2 DBO case has always retained 0.27 even for a 24-sample run.
    smoke = case_id != "afd-v2-graph-dbo-tp2"
    return Case(
        id=case_id,
        model="DeepSeek-V2-Lite",
        served_model_name_prefix="deepseek-v2-lite-afd",
        deployment=Deployment(
            backend=backend,
            connector=(
                None
                if baseline
                else "DeepEPM2NAFDConnector"
                if deepep
                else "P2pNcclAFDConnector"
                if backend == "gpu"
                else "CAMP2pAFDConnector"
            ),
            attention=Role(a_dp, a_tp, execution),
            ffn=None if baseline else Role(f_dp, f_tp, execution),
            compute_gate_on_attention=deepep,
            dbo=DboConfig(DBO_DECODE_THRESHOLD, DBO_PREFILL_THRESHOLD) if dbo else None,
            cudagraph_capture_size=CAPTURE_SIZE if graph else None,
            max_num_seqs=CAPTURE_SIZE if graph or comparison else None,
            max_num_batched_tokens=CAPTURE_SIZE if baseline else None,
        ),
        common_vllm_args=(f"--max-model-len={MAX_MODEL_LEN}",),
        phases=(
            Phase(
                name="evaluation",
                task=Gsm8kTask(
                    Gsm8kSettings(
                        sample_limit=sample_limit,
                        threshold=ACCURACY_THRESHOLD
                        if comparison or not smoke
                        else SMOKE_THRESHOLD,
                        num_concurrent=COMPARISON_CONCURRENCY
                        if comparison or dbo
                        else None,
                        minimum_samples=minimum_samples,
                        smoke_max_samples=SMOKE_MAX_SAMPLES if smoke else None,
                        smoke_threshold=SMOKE_THRESHOLD if smoke else None,
                        larger_sample_threshold=ACCURACY_THRESHOLD if smoke else None,
                    ),
                ),
                checks=(
                    (Mrv2Check((ATTENTION_ROLE, FFN_ROLE), graph=graph, dbo=dbo),)
                    if not baseline and (graph or dbo)
                    else ()
                ),
            ),
        ),
    )


if __name__ == "__main__":
    if sys.argv[1:2] == ["multi-pod"]:
        from tests.e2e.multi_pod.runner import case_main as multi_pod_case_main

        raise SystemExit(
            multi_pod_case_main(build_multi_pod_case, MULTI_POD_IDS, sys.argv[2:])
        )
    # The shared entry receives the builder, and never imports this model.
    from tests.e2e.runner import case_main

    raise SystemExit(
        case_main(
            build_case,
            tuple(dict.fromkeys((*GPU_IDS, *NPU_IDS, *DEEPEP_M2N_SCENARIOS))),
        )
    )
