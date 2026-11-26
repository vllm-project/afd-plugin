# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Model-local CUDA cases; importing IDs never resolves model assets."""

from __future__ import annotations

from dataclasses import replace

from tests.e2e.case import (
    Case,
    CaseEnvironment,
    FfnService,
    Gsm8kSettings,
    Gsm8kTask,
    Mrv2Check,
    Phase,
)
from tests.e2e.presets import DECODE_GRAPH, EAGER

# Native hybrid GDN does not support DBO, so no case enables it.
SCENARIOS = (
    "baseline-graph",
    "afd-eager-2a1f",
    "afd-graph-2a1f",
)
# DeepEP M2N routes on Attention, which also runs the gated shared expert.
DEEPEP_M2N_SCENARIOS = (
    "afd-deepep-m2n-eager-2a2f",
    "afd-deepep-m2n-graph-2a2f",
    "afd-deepep-m2n-graph-tp2-2a2f",
)
# The same DeepEP M2N deployments with the FFN side served by the standalone FFN
# service (torchrun -m afd_plugin.ffn_faas) instead of `vllm serve`; Attention
# launches exactly as in the mirrored case.
DEEPEP_M2N_FFN_SERVICE_SCENARIOS = (
    "afd-deepep-m2n-faas-eager-2a2f",
    "afd-deepep-m2n-faas-graph-2a2f",
    "afd-deepep-m2n-faas-graph-tp2-2a2f",
)
FFN_SERVICE_BACKEND = "gpu-deepep-m2n"
TEXT_ONLY_ARG = "--language-model-only"
GSM8K_SAMPLES = 7
GSM8K_THRESHOLD = 0.27
CAPTURE_SIZE = 8
MAX_MODEL_LEN = 4096


def build_case(case_id: str, environment: CaseEnvironment) -> Case:
    if environment.device_backend != "gpu":
        raise ValueError("Qwen3.6 MoE CUDA E2E requires the gpu backend")
    # baseline, graph, Attention DP/TP, FFN ranks
    profiles = {
        "baseline-graph": (True, True, 4, 1, 0),
        "afd-eager-2a1f": (False, False, 2, 1, 1),
        "afd-graph-2a1f": (False, True, 2, 1, 1),
        "afd-deepep-m2n-eager-2a2f": (False, False, 2, 1, 2),
        "afd-deepep-m2n-graph-2a2f": (False, True, 2, 1, 2),
        "afd-deepep-m2n-graph-tp2-2a2f": (False, True, 1, 2, 2),
        "afd-deepep-m2n-faas-eager-2a2f": (False, False, 2, 1, 2),
        "afd-deepep-m2n-faas-graph-2a2f": (False, True, 2, 1, 2),
        "afd-deepep-m2n-faas-graph-tp2-2a2f": (False, True, 1, 2, 2),
    }
    baseline, graph, attention_dp, attention_tp, ffn_ranks = profiles[case_id]
    execution = DECODE_GRAPH if graph else EAGER
    ffn_service = case_id in DEEPEP_M2N_FFN_SERVICE_SCENARIOS
    deepep = case_id in DEEPEP_M2N_SCENARIOS or ffn_service
    deployment = execution.deployment(
        backend="gpu",
        attention=(attention_dp, attention_tp),
        ffn=None if baseline else (ffn_ranks, 1),
        connector=None
        if baseline
        else "DeepEPM2NAFDConnector"
        if deepep
        else "P2pNcclAFDConnector",
        compute_gate_on_attention=deepep,
        capture_size=CAPTURE_SIZE,
    )
    if ffn_service:
        assert deployment.ffn is not None
        # The service takes no common vLLM arguments; F needs the text-only flag.
        deployment = replace(
            deployment,
            ffn=replace(deployment.ffn, vllm_args=(TEXT_ONLY_ARG,)),
            ffn_service=FfnService(FFN_SERVICE_BACKEND),
        )
    return Case(
        id=case_id,
        model="Qwen3.6 MoE",
        served_model_name_prefix="qwen3-6-afd",
        deployment=deployment,
        common_vllm_args=(
            f"--max-model-len={MAX_MODEL_LEN}",
            TEXT_ONLY_ARG,
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

    raise SystemExit(
        case_main(
            build_case,
            (*SCENARIOS, *DEEPEP_M2N_SCENARIOS, *DEEPEP_M2N_FFN_SERVICE_SCENARIOS),
        )
    )
